"""Independent result acceptance against an immutable OG single-crop database."""
import json
import math
from pathlib import Path
import sqlite3


def readonly(path):
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    return c


def verify_scene(database, baseline, scene, revision):
    with readonly(database) as c, readonly(baseline) as b:
        for db in (c, b):
            assert [tuple(r) for r in db.execute('PRAGMA integrity_check')] == [('ok',)]
            assert not db.execute('PRAGMA foreign_key_check').fetchall()
        runs = c.execute('SELECT * FROM runs').fetchall()
        old = b.execute("SELECT * FROM runs WHERE method='single' AND status='complete'").fetchall()
        assert len(runs) == len(old) == 1
        r, o = runs[0], old[0]
        rid, oid = r['run_id'], o['run_id']
        assert r['scene_id'] == o['scene_id'] == scene
        assert r['method'] == 'single-max' and r['status'] == 'complete'
        assert rid == 'sunny63-single-max-' + scene
        cfg, ocfg = json.loads(r['config_json']), json.loads(o['config_json'])
        assert cfg['subset'] == ocfg['subset'] == 'all'
        assert cfg['code_revision'] == revision
        for k in ('timestamps', 'manifest_hash', 'checkpoint_sha256', 'environment', 'seed', 'dtype', 'conventions', 'metrics_version'):
            assert cfg[k] == ocfg[k], k
        assert cfg['hardware']['gpu'] == ocfg['hardware']['gpu']
        assert cfg['preprocessing']['center_crop'] == 'min(width,height)'
        assert cfg['preprocessing']['internal_crop'] is False
        assert cfg['preprocessing']['model_passes'] == 1
        manifest = json.loads(c.execute('SELECT manifest_json FROM scenes WHERE scene_id=?', (scene,)).fetchone()[0])
        om = json.loads(b.execute('SELECT manifest_json FROM scenes WHERE scene_id=?', (scene,)).fetchone()[0])
        assert manifest == om
        timestamps = cfg['timestamps']; n = len(timestamps); assert n > 0
        assert {x[0] for x in c.execute('SELECT timestamp FROM timestamps WHERE run_id=?', (rid,))} == set(timestamps)
        def rows(db, table, run, keys):
            result = {tuple(r[k] for k in keys):dict(r) for r in db.execute('SELECT * FROM '+table+' WHERE run_id=?', (run,))}
            return result
        pred = rows(c, 'predictions', rid, ['timestamp','camera'])
        orig = rows(b, 'predictions', oid, ['timestamp','camera'])
        assert pred.keys() == orig.keys() and len(pred) == 5*n
        for k, p in pred.items():
            q = orig[k];d, od = json.loads(p['details_json']), json.loads(q['details_json'])
            assert p['heading_deg'] == q['heading_deg']
            for name in ('input_sha256', 'path', 'reference'): assert d[name] == od[name]
            assert d['crop_count'] == 1 and d['transform_seconds'] > 0
            assert p['sun_deg'] is None or math.isfinite(p['sun_deg'])
        pairs = rows(c,'pairs',rid,['timestamp','camera_a','camera_b'])
        opairs = rows(b,'pairs',oid,['timestamp','camera_a','camera_b'])
        assert pairs.keys() == opairs.keys() and len(pairs) == 10*n
        wrap = lambda a:(a+180)%360-180
        for k, p in pairs.items():
            assert p['reference_deg'] == opairs[k]['reference_deg']
            a,bp = pred[k[:1]+(k[1],)], pred[k[:1]+(k[2],)]
            estimate = None if a['sun_deg'] is None or bp['sun_deg'] is None else wrap(a['sun_deg']-bp['sun_deg'])
            error = None if estimate is None or p['reference_deg'] is None else wrap(estimate-p['reference_deg'])
            assert p['estimate_deg'] == estimate and p['error_deg'] == error
            assert p['absolute_error_deg'] == (None if error is None else abs(error))
        valid = sum(p['error_deg'] is not None for p in pairs.values())
        return dict(scene_id=scene,run_id=rid,passed=True,timestamps=n,predictions=len(pred),pairs=len(pairs),valid_pairs=valid,invalid_pairs=len(pairs)-valid,invalid_predictions=sum(p['sun_deg'] is None for p in pred.values()),checkpoint_sha256=cfg['checkpoint_sha256'],code_revision=revision,baseline_revision=ocfg['code_revision'],identical_manifest_checkpoint_environment_gpu_pair_keys_references=True)
