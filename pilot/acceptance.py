"""Readiness evidence and a full-run gate; no privileged operations here."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import numpy as np
from .core import CAMERAS, fingerprint, dumps, sha256, connect
from .report import summarize, backup, report, readonly

def check_recovery(database, method):
    c=readonly(database)
    before=f'benchmark-{method}';after=f'recovery-{method}'
    timestamps=[r[0] for r in c.execute('SELECT timestamp FROM timestamps WHERE run_id=? ORDER BY CAST(timestamp AS INTEGER)',(after,))]
    if len(timestamps)!=2: raise ValueError('recovery must contain exactly two timestamps')
    for ts in timestamps:
        for table, cols, order in [('predictions','camera,sun_deg,heading_deg,reason','camera'),
                                   ('pairs','camera_a,camera_b,reference_deg,estimate_deg,error_deg,reason','camera_a,camera_b')]:
            a=[tuple(r) for r in c.execute(f'SELECT {cols} FROM {table} WHERE run_id=? AND timestamp=? ORDER BY {order}',(before,ts))]
            b=[tuple(r) for r in c.execute(f'SELECT {cols} FROM {table} WHERE run_id=? AND timestamp=? ORDER BY {order}',(after,ts))]
            if a!=b: raise ValueError(f'recovery predictions differ: {method} {table} {ts}')
    attempts=[r[0] for r in c.execute('SELECT status FROM attempts WHERE run_id=? ORDER BY id',(after,))]
    if 'interrupted' not in attempts or attempts[-1]!='complete':raise ValueError('missing interruption/resume evidence')
    c.close()
    return dict(passed=True,identical_predictions=True,timestamps=2,attempt_statuses=attempts)

def run_command(args,method,subset,run_id,fault=0):
    cmd=[sys.executable,'-m','pilot','run','--data',args.data,'--database',args.database,
         '--weights',args.weights,'--method',method,'--subset',subset,'--run-id',run_id]
    if fault:cmd+=['--fault-after-predictions',str(fault)]
    return subprocess.run(cmd).returncode

def readiness(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    if manifest['failures']:raise ValueError('resolve manifest failures before readiness')
    if manifest['counts']['timestamps']!=200 or manifest['counts']['images']!=1000:
        raise ValueError('unexpected scene coverage; inspect before proceeding')
    recovery={}
    for method in ('single','scan'):
        if run_command(args,method,'benchmark',f'benchmark-{method}'):
            raise RuntimeError(f'{method} benchmark failed')
        c=readonly(args.database)
        r=c.execute('SELECT status FROM runs WHERE run_id=?',(f'recovery-{method}',)).fetchone()
        c.close()
        if r is None:
            result=run_command(args,method,'recovery',f'recovery-{method}',7)
            if result!=-9:raise RuntimeError(f'expected injected SIGKILL, got {result}')
            c=readonly(args.database)
            count=c.execute('SELECT COUNT(*) FROM timestamps WHERE run_id=?',(f'recovery-{method}',)).fetchone()[0]
            c.close()
            if count!=1:raise ValueError('incomplete timestamp was not discarded')
        if run_command(args,method,'recovery',f'recovery-{method}'):
            raise RuntimeError(f'{method} resume failed')
        recovery[method]=check_recovery(args.database,method)
    summary=report(args.database,output/'readiness-report')
    for method in ('single','scan'):
        s=summary[f'benchmark-{method}']
        if s['timestamps']!=15 or s['image_predictions']!=75 or s['metrics']['total']!=150:
            raise ValueError('benchmark coverage mismatch')
        if s['metrics']['invalid']:raise ValueError('benchmark contains invalid outcomes')
        if s['minimum_gpu_free_bytes'] is None or s['minimum_gpu_free_bytes']<2**30:
            raise ValueError('GPU headroom gate failed')
        s['projected_full_seconds']=s['mean_timestamp_seconds']*len(manifest['timestamps'])
    free=shutil.disk_usage(output).free
    if free<20*2**30:raise ValueError('disk gate failed')
    bundle=output/'readiness-backup'
    if bundle.exists():
        bundle=output/f'readiness-backup-{int(time.time())}'
    info=backup(args.database,bundle)
    shutil.copy2(Path(args.data)/'manifest.json',bundle/'manifest.json')
    shutil.copy2(Path(args.data)/'source.json',bundle/'source.json')
    shutil.copy2(Path(__file__).resolve().parents[1]/'requirements-pilot.lock',bundle/'requirements-pilot.lock')
    evidence=dict(passed=True,stage='local-readiness',manifest_hash=fingerprint(manifest),
                  code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  checkpoint_sha256=sha256(args.weights),recovery=recovery,summary=summary,
                  disk_free_bytes=free,backup_dir=str(bundle),backup_sha256=info['sha256'])
    (output/'readiness.json').write_text(dumps(evidence))
    print('LOCAL READINESS PASSED. Full evaluation requires PC4 backup/restore receipt.',flush=True)

def full(args):
    output=Path(args.output)
    e=json.loads((output/'readiness.json').read_text())
    r=json.loads((output/'pc4-restore-verification.json').read_text())
    if not e['passed'] or not r['verified'] or r['backup_sha256']!=e['backup_sha256']:
        raise ValueError('missing or mismatched backup/restore gate')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    if revision!=e['code_revision'] or fingerprint(manifest)!=e['manifest_hash'] or sha256(args.weights)!=e['checkpoint_sha256']:
        raise ValueError('readiness evidence is for different code, inputs, or weights')
    c=connect(args.database)
    with c:
        c.execute('INSERT INTO events(run_id,recorded,kind,details_json) VALUES (NULL,?,?,?)',
                  (str(time.time()),'pc4_restore_verified',dumps(r)))
    c.close()
    # Benchmark resume fingerprints also enforce the exact dependency environment and hardware.
    for method in ('single','scan'):
        if run_command(args,method,'benchmark',f'benchmark-{method}'):
            raise ValueError('readiness runtime fingerprint no longer matches')
        if run_command(args,method,'all',f'full-{method}'):
            raise RuntimeError(f'full {method} failed; deliberate resume required')
    report(args.database,output/'final-report')
    target=output/f'final-backup-{int(time.time())}'
    info=backup(args.database,target)
    for name in ('manifest.json','source.json'):shutil.copy2(Path(args.data)/name,target/name)
    shutil.copy2(Path(__file__).resolve().parents[1]/'requirements-pilot.lock',target/'requirements-pilot.lock')
    (output/'completion.json').write_text(dumps(dict(complete=True,backup_dir=str(target),backup=info)))
    print('FULL EVALUATION COMPLETE; copy final backup to PC4.',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['readiness','full'])
    for k in ('data','database','weights','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    (readiness if a.stage=='readiness' else full)(a)
if __name__=='__main__':main()
