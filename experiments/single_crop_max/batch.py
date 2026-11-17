"""Run one staged scene at a time under a single owned Slurm allocation."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.single_crop_max.verify import verify_scene
from pilot.core import sha256


def remove_staged(root, scene, receipt):
    """Only remove this experiment's newly staged scene after a verified result."""
    root = Path(root).absolute()
    assert not root.is_symlink() and root.resolve() == root
    selection = json.loads((root/'selection.json').read_text())
    assert scene in selection and receipt['scene_id'] == scene and receipt['passed'] is True
    data = root/'staged'/scene
    assert not (root/'staged').is_symlink() and not data.is_symlink()
    assert data.resolve().parent == root/'staged'
    assert (data/'READY').read_text().strip() == scene
    marker = json.loads((data/'owner.json').read_text())
    assert marker == {'experiment':'single-crop-max-20260927','scene_id':scene}
    assert not any(p.is_symlink() for p in data.rglob('*'))
    saved = json.loads((root/'results'/scene/'validation.json').read_text())
    assert saved == receipt
    assert sha256(root/'results'/scene/'backup.sqlite') == receipt['backup_sha256']
    shutil.rmtree(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--weights',required=True)
    a=p.parse_args();root=Path(a.root).resolve();source=Path(__file__).resolve().parents[2]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    ids=json.loads((root/'selection.json').read_text())
    expected=sorted(r['scene_id'] for r in csv.DictReader((root/'baseline/scene_metadata.csv').open()) if r['Weather']=='Sunny')
    assert sorted(ids)==expected and len(ids)==len(set(ids))==63 and 'scene_055' in ids
    for sid in ids:
        out=root/'results'/sid;out.mkdir(parents=True,exist_ok=True)
        db=out/'results.sqlite';baseline=root/'baseline'/sid/'results.sqlite'
        if (out/'validation.json').exists():
            verify_scene(db,baseline,sid,revision)
            continue
        data=root/'staged'/sid
        deadline=time.monotonic()+1800
        while not (data/'READY').exists():
            if time.monotonic()>deadline:raise TimeoutError('staging '+sid)
            time.sleep(1)
        assert (data/'READY').read_text().strip()==sid
        assert json.loads((data/'owner.json').read_text())=={'experiment':'single-crop-max-20260927','scene_id':sid}
        assert json.loads((data/'manifest.json').read_text())==json.loads((root/'baseline'/sid/'manifest.json').read_text())
        with (out/'inference.log').open('a') as log:
            subprocess.run([sys.executable,'-m','pilot','run','--data',str(data),'--database',str(db),'--weights',a.weights,'--run-id','sunny63-single-max-'+sid,'--method','single-max','--subset','all'],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt=verify_scene(db,baseline,sid,revision)
        from pilot.report import summarize,readonly
        (out/'summary.json').write_text(json.dumps(summarize(db),indent=2))
        import sqlite3
        with readonly(db) as c,sqlite3.connect(out/'backup.sqlite') as b:c.backup(b)
        receipt['backup_sha256']=sha256(out/'backup.sqlite')
        (out/'validation.json').write_text(json.dumps(receipt,indent=2))
        remove_staged(root,sid,receipt)
        print(json.dumps(receipt),flush=True)
    (root/'completion.json').write_text(json.dumps({'passed':True,'scenes':ids,'code_revision':revision},indent=2))


if __name__=='__main__':main()
