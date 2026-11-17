"""Download just the selected archive; create a content-addressed input manifest."""
import csv
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from .core import CAMERAS, WAYVE, colmap, sha256, dumps, reference_headings

FOLDER = '1XXlhGut7c5PVeBj4qwORlHoGdeSYxGgt'

def prepare(root, scene_id='scene_006'):
    import gdown
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance_file = root / 'source.json'
    filename = f'{scene_id}.zip'
    if not provenance_file.exists():
        entries = gdown.download_folder(id=FOLDER, skip_download=True, quiet=True, use_cookies=False)
        chosen = [e for e in entries if Path(e.path).name == filename]
        if len(chosen) != 1:
            raise ValueError(f'expected exactly one {filename}, found {chosen}')
        provenance_file.write_text(dumps(dict(folder_id=FOLDER, file_id=chosen[0].id, filename=filename, scene_id=scene_id)))
    source = json.loads(provenance_file.read_text())
    source_scene = source.get('scene_id', 'scene_006')
    if source_scene != scene_id:
        raise ValueError(f'data root contains {source_scene}, not requested {scene_id}')
    if source.get('filename') != filename:
        raise ValueError(f'source filename {source.get("filename")} does not match {filename}')
    archive = root / source['filename']
    if not archive.exists():
        if shutil.disk_usage(root).free < 25 * 2**30:
            raise RuntimeError('insufficient initial staging space')
        partial = root / (source['filename'] + '.partial')
        gdown.download(id=source['file_id'], output=str(partial), resume=True, use_cookies=False)
        if not zipfile.is_zipfile(partial):
            raise ValueError('download is not a valid scene ZIP')
        partial.rename(archive)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        for member in entries:
            target = (root / member.filename).resolve()
            if root not in target.parents or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('unsafe archive member')
        size = sum(e.file_size for e in entries)
        if not (root / '.extracted').exists():
            if shutil.disk_usage(root).free < size + 20 * 2**30:
                raise RuntimeError('extraction would leave less than 20 GiB free')
            z.extractall(root)
            (root / '.extracted').write_text(sha256(archive))
    source.update(archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size, extracted_bytes=size,
                  wayve_revision=WAYVE, dataset='WayveScenes101', release='initial-2024-06-17')
    provenance_file.write_text(dumps(source))
    url = f'https://raw.githubusercontent.com/wayveai/wayve_scenes/{WAYVE}/data/scene_metadata.csv'
    metadata_path = root / 'scene_metadata.csv'
    if not metadata_path.exists():
        metadata_path.write_bytes(urllib.request.urlopen(url).read())
    return validate(root, scene_id)

def validate(root, scene_id=None):
    root = Path(root).resolve()
    source = json.loads((root / 'source.json').read_text())
    scene_id = scene_id or source.get('scene_id', 'scene_006')
    source_scene = source.get('scene_id', 'scene_006')
    if source_scene != scene_id:
        raise ValueError(f'data root contains {source_scene}, not requested {scene_id}')
    archive = root / source['filename']
    if source.get('archive_sha256') != sha256(archive):
        raise ValueError('archive changed since staging')
    scene = root / scene_id
    metadata_path = root / 'scene_metadata.csv'
    metadata = next(row for row in csv.DictReader(metadata_path.open()) if row['scene_id'] == scene_id)
    if metadata['Weather'] != 'Sunny' or metadata['Time of Day'] != 'Day':
        raise ValueError('scene weather/time metadata differs from selected baseline')
    poses = colmap(scene / 'colmap_sparse/rig')
    records, failures = {}, []
    for cam in CAMERAS:
        paths = sorted((scene / 'images' / cam).glob('*'))
        for p in paths:
            if not p.is_file():
                continue
            ts = p.stem
            int(ts)  # Preserve the source text, but require numeric ordering.
            key = f'{cam}/{p.name}'
            if ts in records and cam in records[ts]:
                raise ValueError('duplicate timestamp/camera')
            row = dict(path=str(p.relative_to(root)), sha256=sha256(p), bytes=p.stat().st_size)
            try:
                with Image.open(p) as im:
                    im.load()
                    row.update(width=im.width, height=im.height, mode=im.mode)
            except Exception as e:
                row['input_error'] = f'{type(e).__name__}: {e}'
                failures.append([ts, cam, row['input_error']])
            row['pose'] = poses.get(key)
            if row['pose'] is None:
                failures.append([ts, cam, 'missing_pose'])
            records.setdefault(ts, {})[cam] = row
    timestamps = sorted(records, key=int)
    if not timestamps:
        raise ValueError('no scene images found')
    numeric = list(map(int, timestamps))
    if len(set(numeric)) != len(numeric):
        raise ValueError('ambiguous numeric timestamps')
    deltas = np.diff(numeric)
    # Raw time units are deliberately not asserted without source documentation.
    cadence = dict(nominal_hz=10, raw_step_values=sorted(set(map(int, deltas))),
                   raw_unit='unverified', relative_time_basis='ordinal_index / 10 (nominal)')
    for ts, rows in records.items():
        missing = set(CAMERAS) - set(rows)
        failures.extend([ts, cam, 'missing_image'] for cam in sorted(missing))
        rotations = {cam: np.array(v['pose']['c2w']) for cam, v in rows.items() if v['pose']}
        refs = reference_headings(rotations)
        for cam, row in rows.items():
            row['reference'] = refs.get(cam)
    manifest = dict(scene_id=scene_id, source=source, metadata=metadata,
                    metadata_sha256=sha256(metadata_path),
                    calibration_hashes={p.name: sha256(p) for p in (scene / 'colmap_sparse/rig').glob('*.bin')},
                    timestamps=timestamps, cadence=cadence, records=records, failures=failures,
                    counts=dict(timestamps=len(timestamps), images=sum(map(len, records.values())),
                                expected_pairs=len(timestamps)*10))
    (root / 'manifest.json').write_text(dumps(manifest))
    print(dumps(dict(counts=manifest['counts'], cadence=cadence, failures=failures, weather=metadata)))
    return manifest

def verify_manifest(root, manifest):
    root = Path(root)
    if sha256(root / 'scene_metadata.csv') != manifest['metadata_sha256']:
        raise ValueError('metadata fingerprint changed')
    for name, digest in manifest['calibration_hashes'].items():
        if sha256(root / manifest['scene_id'] / 'colmap_sparse/rig' / name) != digest:
            raise ValueError('calibration fingerprint changed')
    for rows in manifest['records'].values():
        for row in rows.values():
            if sha256(root / row['path']) != row['sha256']:
                raise ValueError('input image fingerprint changed')
