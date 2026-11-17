"""Numeric reports, scientific plots, and application-consistent backup validation."""
import csv
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from .core import dumps, sha256, fingerprint, connect

def readonly(path):
    c=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    return c

def statistics(values):
    x=np.asarray(values,dtype=float)
    if not len(x): return dict(valid=0)
    a=np.abs(x)
    return dict(valid=len(x),mae_deg=float(a.mean()),median_deg=float(np.median(a)),
                rmse_deg=float(np.sqrt((x*x).mean())),p90_deg=float(np.percentile(a,90)),
                p95_deg=float(np.percentile(a,95)),max_deg=float(a.max()),
                circular_bias_deg=math.degrees(math.atan2(np.sin(np.deg2rad(x)).mean(),np.cos(np.deg2rad(x)).mean())),
                within_5=float(np.mean(a<=5)),within_10=float(np.mean(a<=10)),within_20=float(np.mean(a<=20)))

def summarize(database):
    c=readonly(database)
    output={}
    for r in c.execute('SELECT * FROM runs ORDER BY run_id'):
        rid=r['run_id']; cfg=json.loads(r['config_json'])
        pairs=[dict(x) for x in c.execute('SELECT * FROM pairs WHERE run_id=? ORDER BY timestamp,camera_a,camera_b',(rid,))]
        pred=[json.loads(x[0]) for x in c.execute('SELECT details_json FROM predictions WHERE run_id=?',(rid,))]
        times=[x[0] for x in c.execute('SELECT wall_seconds FROM timestamps WHERE run_id=?',(rid,))]
        stats=statistics([p['error_deg'] for p in pairs if p['error_deg'] is not None])
        stats.update(total=len(pairs),invalid=sum(p['error_deg'] is None for p in pairs))
        by_pair={}
        for a,b in sorted({(p['camera_a'],p['camera_b']) for p in pairs}):
            selected=[p for p in pairs if (p['camera_a'],p['camera_b'])==(a,b)]
            by_pair[f'{a} -> {b}']=statistics([p['error_deg'] for p in selected if p['error_deg'] is not None])
            by_pair[f'{a} -> {b}'].update(total=len(selected),invalid=sum(p['error_deg'] is None for p in selected))
        seconds=sum(times)
        output[rid]=dict(method=r['method'],subset=cfg['subset'],status=r['status'],scene_id=r['scene_id'],
                         metrics=stats,by_pair=by_pair,timestamps=len(times),expected_timestamps=len(cfg['timestamps']),
                         image_predictions=len(pred),measured_seconds=seconds,
                         images_per_second=len(pred)/seconds if seconds else None,
                         mean_timestamp_seconds=float(np.mean(times)) if times else None,
                         peak_allocated_bytes=max((p.get('peak_allocated_bytes',0) for p in pred),default=0),
                         peak_reserved_bytes=max((p.get('peak_reserved_bytes',0) for p in pred),default=0),
                         peak_rss_bytes=max((p.get('rss_bytes',0) for p in pred),default=0),
                         minimum_gpu_free_bytes=min((p['gpu_free_bytes'] for p in pred if 'gpu_free_bytes' in p),default=None),
                         total_transform_seconds=sum(p.get('transform_seconds',0) for p in pred),
                         total_model_seconds=sum(p.get('model_seconds',0) for p in pred),
                         total_decode_seconds=sum(p.get('decode_seconds',0) for p in pred))
    c.close()
    return output

def report(database, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    summary=summarize(database)
    (out/'summary.json').write_text(dumps(summary))
    c=readonly(database)
    rows=[dict(r) for r in c.execute('SELECT p.*, t.relative_seconds FROM pairs p JOIN timestamps t USING(run_id,timestamp) ORDER BY run_id,relative_seconds,camera_a,camera_b')]
    if rows:
        with (out/'pair_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    scenes=sorted({s['scene_id'] for s in summary.values()})
    title=', '.join(scenes) if scenes else 'no scenes'
    lines=[f'# Sun Compass / WayveScenes101 {title}','',
           'Reference yaw is derived from COLMAP camera poses in the front-camera plane. ',
           'Results are exploratory: frames and camera pairs are correlated, and camera tilt may contribute to disagreement.','',
           '| Run | Status | Valid / total pairs | MAE (deg) | p95 (deg) | Images/s |',
           '| --- | --- | --- | --- | --- | --- |']
    for rid,s in summary.items():
        m=s['metrics']
        lines.append(f"| {rid} | {s['status']} | {m['valid']} / {m['total']} | {m.get('mae_deg',float('nan')):.3f} | {m.get('p95_deg',float('nan')):.3f} | {(s['images_per_second'] or 0):.3f} |")
    final={rid:s for rid,s in summary.items() if s['subset']=='all'}
    for rid,s in (final or summary).items():
        selected=[r for r in rows if r['run_id']==rid and r['error_deg'] is not None]
        if not selected: continue
        fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
        for pair in sorted({(r['camera_a'],r['camera_b']) for r in selected}):
            rr=[r for r in selected if (r['camera_a'],r['camera_b'])==pair]
            axes[0].plot([r['relative_seconds'] for r in rr],[r['error_deg'] for r in rr],label=' → '.join(pair),linewidth=1)
        axes[0].set(xlabel='Nominal relative time (s)',ylabel='Signed wrapped yaw error (degrees)',title=rid)
        axes[0].legend(fontsize=7,ncol=2)
        axes[0].grid(alpha=.2)
        axes[1].hist([r['absolute_error_deg'] for r in selected],bins=np.linspace(0,180,37))
        axes[1].set(xlabel='Absolute wrapped yaw error (degrees)',ylabel='Pair observations')
        fig.savefig(out/f'{rid}.png',dpi=140);plt.close(fig)
        lines.extend(['',f'![{rid}]({rid}.png)'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    c.close()
    return summary

def row_counts(c):
    tables=['scenes','runs','attempts','timestamps','predictions','pairs','telemetry','events']
    return {t:c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in tables}

def backup(database,destination):
    target=Path(destination);target.mkdir(parents=True,exist_ok=False)
    writer=connect(database)
    with writer:
        writer.execute('INSERT INTO events(run_id,recorded,kind,details_json) VALUES (NULL,?,?,?)',
                       (dt.datetime.now(dt.timezone.utc).isoformat(),'backup_requested',dumps(dict(destination=str(target)))))
    writer.close()
    src=readonly(database)
    dst=sqlite3.connect(target/'results.sqlite')
    src.backup(dst)
    if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or dst.execute('PRAGMA foreign_key_check').fetchall():
        raise ValueError('backup integrity failure')
    counts=row_counts(dst)
    dst.close();src.close()
    summary=summarize(target/'results.sqlite')
    info=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=sha256(target/'results.sqlite'),
              counts=counts,summary_hash=fingerprint(summary))
    (target/'backup.json').write_text(dumps(info))
    (target/'summary.json').write_text(dumps(summary))
    return info

def restore_check(backup_dir,restore_dir):
    source=Path(backup_dir);destination=Path(restore_dir)
    info=json.loads((source/'backup.json').read_text())
    if sha256(source/'results.sqlite')!=info['sha256']:
        raise ValueError('backup checksum mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'results.sqlite',destination/'results.sqlite')
    c=readonly(destination/'results.sqlite')
    if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or c.execute('PRAGMA foreign_key_check').fetchall():
        raise ValueError('restore integrity failure')
    if row_counts(c)!=info['counts']: raise ValueError('restore count mismatch')
    c.close()
    if fingerprint(summarize(destination/'results.sqlite'))!=info['summary_hash']:
        raise ValueError('restored report differs')
    receipt=dict(verified=True,backup_sha256=info['sha256'],counts=info['counts'],summary_hash=info['summary_hash'])
    (destination/'restore-verification.json').write_text(dumps(receipt))
    return receipt
