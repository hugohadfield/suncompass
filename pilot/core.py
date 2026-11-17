"""Coordinate definitions, COLMAP binary input, and transactional result storage."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import struct
import numpy as np

CAMERAS = ('right-forward', 'front-forward', 'left-forward', 'left-backward', 'right-backward')
PAIRS = tuple(itertools.combinations(CAMERAS, 2))
UPSTREAM = '58e9009dbc81c0af589e2e3e2fe14abda2cfc85f'
WAYVE = '2fc2fa9606b328cfa3dbfa11ae1c6ace99c240eb'
CONVENTIONS = 'front-camera-plane-left-positive-v1'

def dumps(x):
    return json.dumps(x, sort_keys=True, allow_nan=False)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def fingerprint(x):
    return hashlib.sha256(dumps(x).encode()).hexdigest()

def wrap(x):
    return (x + 180.0) % 360.0 - 180.0

def azimuth(f, l):
    if not np.isfinite([f, l]).all() or math.hypot(f, l) < 1e-8:
        return None
    return math.degrees(math.atan2(l, f))

def quaternion_rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError('invalid COLMAP quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

def reference_headings(poses):
    """poses contain COLMAP camera-to-world rotations in OpenCV camera axes."""
    for r in poses.values():
        if not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
            raise ValueError('invalid rotation')
    if 'front-forward' not in poses:
        return {}
    front = poses['front-forward']
    out = {}
    for camera, r in poses.items():
        relative = front.T @ r
        forward = relative[:, 2]
        heading = azimuth(forward[2], -forward[0])
        tilt = math.degrees(math.acos(float(np.clip(relative[1, 1], -1, 1))))
        pitch = math.degrees(math.asin(float(np.clip(-forward[1], -1, 1))))
        out[camera] = dict(heading_deg=heading, vertical_misalignment_deg=tilt, pitch_deg=pitch)
    return out

def pair_result(a, b):
    ref = None if a['heading_deg'] is None or b['heading_deg'] is None else wrap(b['heading_deg'] - a['heading_deg'])
    estimate = None if a['sun_deg'] is None or b['sun_deg'] is None else wrap(a['sun_deg'] - b['sun_deg'])
    error = None if ref is None or estimate is None else wrap(estimate - ref)
    return dict(reference_deg=ref, estimate_deg=estimate, error_deg=error,
                absolute_error_deg=None if error is None else abs(error),
                reason=None if error is not None else 'invalid_prediction_or_reference')

def _read(f, fmt):
    size = struct.calcsize('<' + fmt)
    b = f.read(size)
    if len(b) != size:
        raise ValueError('truncated COLMAP file')
    return struct.unpack('<' + fmt, b)

def colmap(root):
    # Official COLMAP binary layout. Only camera calibration and image poses are needed.
    model_specs = {0:('SIMPLE_PINHOLE',3),1:('PINHOLE',4),2:('SIMPLE_RADIAL',4),
                   3:('RADIAL',5),4:('OPENCV',8),5:('OPENCV_FISHEYE',8),
                   6:('FULL_OPENCV',12),7:('FOV',5),8:('SIMPLE_RADIAL_FISHEYE',4),
                   9:('RADIAL_FISHEYE',5),10:('THIN_PRISM_FISHEYE',12)}
    cameras, images = {}, {}
    with open(Path(root) / 'cameras.bin', 'rb') as f:
        for _ in range(_read(f, 'Q')[0]):
            cid, mid, width, height = _read(f, 'iiQQ')
            if mid not in model_specs:
                raise ValueError(f'unsupported COLMAP camera model {mid}')
            model, n = model_specs[mid]
            params = _read(f, 'd'*n)
            if width <= 0 or height <= 0 or not np.isfinite(params).all() or params[0] <= 0:
                raise ValueError('invalid camera calibration')
            cameras[cid] = dict(model=model, width=width, height=height, params=params)
    with open(Path(root) / 'images.bin', 'rb') as f:
        for _ in range(_read(f, 'Q')[0]):
            iid, *p = _read(f, 'idddddddi')
            q, t, cid = p[:4], p[4:7], p[7]
            name = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError('unterminated COLMAP image name')
                if ch == b'\0':
                    break
                name.extend(ch)
            name = name.decode('utf-8')
            if name in images:
                raise ValueError('duplicate COLMAP image name')
            points = _read(f, 'Q')[0]
            # Seek, then verify remaining file boundary even for the last record.
            end = f.tell() + points * 24
            if end > Path(f.name).stat().st_size:
                raise ValueError('truncated COLMAP observations')
            f.seek(end)
            if not np.isfinite(t).all():
                raise ValueError('invalid translation')
            images[name] = dict(image_id=iid, camera_id=cid, qvec=q, tvec=t,
                                c2w=quaternion_rotation(q).T.tolist(), calibration=cameras[cid])
    return images

SCHEMA = '''
CREATE TABLE IF NOT EXISTS scenes (scene_id TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL, manifest_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, scene_id TEXT REFERENCES scenes, method TEXT NOT NULL, fingerprint TEXT NOT NULL, config_json TEXT NOT NULL, status TEXT NOT NULL, started TEXT NOT NULL, ended TEXT);
CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs, started TEXT NOT NULL, ended TEXT, status TEXT NOT NULL, details_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS timestamps (run_id TEXT REFERENCES runs, timestamp TEXT, relative_seconds REAL, wall_seconds REAL, commit_seconds REAL, committed TEXT, PRIMARY KEY(run_id,timestamp));
CREATE TABLE IF NOT EXISTS predictions (run_id TEXT, timestamp TEXT, camera TEXT, sun_deg REAL, heading_deg REAL, reason TEXT, details_json TEXT NOT NULL, PRIMARY KEY(run_id,timestamp,camera), FOREIGN KEY(run_id,timestamp) REFERENCES timestamps);
CREATE TABLE IF NOT EXISTS pairs (run_id TEXT, timestamp TEXT, camera_a TEXT, camera_b TEXT, reference_deg REAL, estimate_deg REAL, error_deg REAL, absolute_error_deg REAL, reason TEXT, PRIMARY KEY(run_id,timestamp,camera_a,camera_b), FOREIGN KEY(run_id,timestamp) REFERENCES timestamps);
CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs, recorded TEXT, details_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs, recorded TEXT, kind TEXT, details_json TEXT NOT NULL);
'''

def connect(path):
    c = sqlite3.connect(path, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON')
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.executescript(SCHEMA)
    return c

def ensure_run(c, run_id, manifest, config, now):
    mh = fingerprint(manifest)
    existing = c.execute('SELECT manifest_hash FROM scenes WHERE scene_id=?', (manifest['scene_id'],)).fetchone()
    if existing and existing[0] != mh:
        raise ValueError('scene manifest changed; use a separate database')
    fp = fingerprint(config)
    row = c.execute('SELECT fingerprint FROM runs WHERE run_id=?', (run_id,)).fetchone()
    if row and row[0] != fp:
        raise ValueError('resume refused: configuration, input, code, environment, or model changed')
    with c:
        c.execute('INSERT OR IGNORE INTO scenes VALUES (?,?,?)', (manifest['scene_id'], mh, dumps(manifest)))
        c.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,NULL)',
                  (run_id, manifest['scene_id'], config['method'], fp, dumps(config), 'pending', now))

def commit_timestamp(c, run_id, timestamp, relative, predictions, wall, now):
    if set(predictions) != set(CAMERAS):
        raise ValueError('five explicit prediction outcomes required')
    with c:
        c.execute('INSERT INTO timestamps VALUES (?,?,?,?,?,?)', (run_id, timestamp, relative, wall, None, now))
        for camera, p in predictions.items():
            c.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)',
                      (run_id,timestamp,camera,p['sun_deg'],p['heading_deg'],p['reason'],dumps(p)))
        for a,b in PAIRS:
            p = pair_result(predictions[a], predictions[b])
            c.execute('INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?)',
                      (run_id,timestamp,a,b,p['reference_deg'],p['estimate_deg'],p['error_deg'],p['absolute_error_deg'],p['reason']))
