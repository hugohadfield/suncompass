"""Disposable safety fixtures for experiment-owned staging cleanup."""
import json
import pytest
from experiments.single_crop_max.batch import remove_staged
from pilot.core import sha256


def fixture(tmp_path):
    root=tmp_path/'run';root.mkdir();scene='scene_006'
    (root/'selection.json').write_text(json.dumps([scene]))
    data=root/'staged'/scene;data.mkdir(parents=True)
    (data/'READY').write_text(scene)
    (data/'owner.json').write_text(json.dumps({'experiment':'single-crop-max-20260927','scene_id':scene}))
    out=root/'results'/scene;out.mkdir(parents=True)
    (out/'backup.sqlite').write_bytes(b'fixture backup')
    receipt={'scene_id':scene,'passed':True,'backup_sha256':sha256(out/'backup.sqlite')}
    (out/'validation.json').write_text(json.dumps(receipt))
    return root,scene,data,out,receipt


def test_cleanup_owned_fixture(tmp_path):
    root,sid,data,_,receipt=fixture(tmp_path)
    remove_staged(root,sid,receipt)
    assert not data.exists() and (root/'results'/sid/'backup.sqlite').exists()


@pytest.mark.parametrize('fault',['foreign_scene','owner','incomplete','backup','symlink'])
def test_cleanup_rejects_foreign_or_unverified(tmp_path,fault):
    root,sid,data,out,receipt=fixture(tmp_path)
    if fault=='foreign_scene':sid='scene_055'
    elif fault=='owner':(data/'owner.json').write_text('{}')
    elif fault=='incomplete':receipt['passed']=False
    elif fault=='backup':(out/'backup.sqlite').write_bytes(b'changed')
    elif fault=='symlink':(data/'foreign').symlink_to(out,target_is_directory=True)
    with pytest.raises((AssertionError,FileNotFoundError)):
        remove_staged(root,sid,receipt)
    assert data.exists()
