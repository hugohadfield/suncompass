"""Validate all paired runs, reproduce baseline metrics, and save comparison artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.single_crop_max.verify import verify_scene, readonly
from experiments.single_crop_max.metrics import summarize_comparison
from pilot.core import sha256
from pilot.report import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--historical-analysis',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=root/'comparison';out.mkdir(exist_ok=True)
    ids=sorted(json.loads((root/'selection.json').read_text()))
    expected=sorted(r['scene_id'] for r in csv.DictReader((root/'baseline/scene_metadata.csv').open()) if r['Weather']=='Sunny')
    assert ids==expected and len(ids)==63 and 'scene_055' in ids
    records=[];receipts=[];pair_export=[];pred_export=[];by_pair={'single':{},'single-max':{}}
    for sid in ids:
        base=root/'baseline'/sid/'results.sqlite';db=root/'results'/sid/'backup.sqlite'
        receipt=verify_scene(db,base,sid,a.revision)
        saved=json.loads((root/'results'/sid/'validation.json').read_text())
        assert sha256(db)==saved.pop('backup_sha256') and saved==receipt
        receipts.append(receipt)
        record={'scene_id':sid,'timestamps':receipt['timestamps']}
        for method,path in [('single',base),('single-max',db)]:
            with readonly(path) as c:
                run=c.execute('SELECT * FROM runs WHERE method=? AND status=?',(method,'complete')).fetchone();rid=run['run_id']
                rows=[dict(r) for r in c.execute('SELECT * FROM pairs WHERE run_id=? ORDER BY timestamp,camera_a,camera_b',(rid,))]
                errors=[r['error_deg'] for r in rows]
                record['single_errors' if method=='single' else 'max_errors']=errors
                record['single_seconds' if method=='single' else 'max_seconds']=c.execute('SELECT sum(wall_seconds) FROM timestamps WHERE run_id=?',(rid,)).fetchone()[0]
                for row in rows:
                    pair_export.append(dict(scene_id=sid,method=method,**row))
                    key=row['camera_a']+' -> '+row['camera_b']
                    by_pair[method].setdefault(key,[]).append(row['error_deg'])
                if sid=='scene_006':
                    pred_export.extend(dict(method=method,**dict(r)) for r in c.execute('SELECT * FROM predictions WHERE run_id=? ORDER BY timestamp,camera',(rid,)))
        records.append(record)
    analysis=summarize_comparison(records)
    assert analysis['timestamps_per_method']==12606 and analysis['predictions_per_method']==63030 and analysis['pairs_per_method']==126060
    # Independent metric oracle: maintained pilot reporting, not Qwen's implementation.
    for method,key in [('single','single_errors'),('single-max','max_errors')]:
        values=[v for r in records for v in r[key]]
        wanted=statistics([v for v in values if v is not None])
        wanted.update(total=len(values),invalid=values.count(None))
        for k,v in wanted.items():assert np.isclose(analysis['pooled'][method][k],v,rtol=1e-12,atol=1e-12),(method,k)
        for record,row in zip(records,analysis['by_scene']):
            want=statistics([v for v in record[key] if v is not None])
            for k,v in want.items():assert np.isclose(row[method][k],v,rtol=1e-12,atol=1e-12),(row['scene_id'],method,k)
    historical=json.loads(a.historical_analysis.read_text())
    for k,v in historical['pooled']['single'].items():assert np.isclose(analysis['pooled']['single'][k],v,rtol=1e-12,atol=1e-12),k
    analysis['historical_letterbox_pooled']=historical['pooled']['letterbox']
    analysis['historical_analysis_sha256']=sha256(a.historical_analysis)
    analysis['by_pair']={m:{k:dict(statistics([x for x in v if x is not None]),total=len(v),invalid=v.count(None)) for k,v in pairs.items()} for m,pairs in by_pair.items()}
    analysis['provenance']=receipts
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    for name,rows in [('pair-results.csv',pair_export),('scene006-predictions.csv',pred_export)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=[dict(scene_id=r['scene_id'],single_mae=r['single']['mae_deg'],single_max_mae=r['single-max']['mae_deg'],mae_change=r['mae_change_deg'],single_p95=r['single']['p95_deg'],single_max_p95=r['single-max']['p95_deg'],single_within10=r['single']['within_10'],single_max_within10=r['single-max']['within_10']) for r in analysis['by_scene']]
    with (out/'per-scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for method,key,label,color in [('single','single_errors','OG single crop','#bf614c'),('single-max','max_errors','Single Crop Max','#177f91')]:
        vals=np.sort(np.abs([v for r in records for v in r[key] if v is not None]))
        axes[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color)
    axes[0].set(xlabel='Absolute wrapped yaw error (degrees)',ylabel='Cumulative fraction',xlim=(0,180),ylim=(0,1));axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].scatter([r['single_mae'] for r in table],[r['single_max_mae'] for r in table],s=24,color='#177f91')
    axes[1].plot([0,180],[0,180],linestyle='--',color='#889099',linewidth=1)
    axes[1].set(xlabel='OG single crop: scene MAE (degrees)',ylabel='Single Crop Max: scene MAE (degrees)',xlim=(0,180),ylim=(0,180));axes[1].grid(alpha=.2)
    fig.suptitle('Same 63 Sunny scenes · frozen checkpoint · COLMAP relative yaw')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    verification=dict(passed=True,scene_count=63,identical_scene_set_including_dusk=True,independent_metric_oracle=True,historical_single_metrics_reproduced=True,provenance_checks=receipts,artifacts={f.name:sha256(f) for f in out.iterdir() if f.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('by_scene','by_pair','provenance')},indent=2))


if __name__=='__main__':main()
