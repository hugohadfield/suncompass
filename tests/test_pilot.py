import json
import math
from pathlib import Path
import sqlite3
import struct
import numpy as np
import pytest
from pilot.core import *
from pilot.report import backup, restore_check, statistics

# OpenCV columns: right, down, forward. Positive yaw turns camera left.
def yaw(deg):
    a=math.radians(deg)
    return np.array([[math.cos(a),0,-math.sin(a)],[0,1,0],[math.sin(a),0,math.cos(a)]])

def prediction(sun=10,heading=0):
    return dict(sun_deg=sun,heading_deg=heading,reason=None)

def fixture_db(path):
    c=connect(path)
    manifest={'scene_id':'scene_006'}
    config={'method':'single','subset':'recovery','timestamps':['1','2']}
    ensure_run(c,'test',manifest,config,'now')
    return c,manifest,config

def test_reference_sign_and_common_world_rotation():
    poses={CAMERAS[0]:yaw(-40),'front-forward':yaw(0),'left-forward':yaw(45)}
    r=reference_headings(poses)
    assert r['right-forward']['heading_deg']==pytest.approx(-40)
    assert r['left-forward']['heading_deg']==pytest.approx(45)
    world=quaternion_rotation([math.cos(.3),math.sin(.3),0,0]) @ yaw(70)
    rr=reference_headings({k:world@v for k,v in poses.items()})
    for k in r: assert rr[k]['heading_deg']==pytest.approx(r[k]['heading_deg'])

def test_pair_sign_wrap_reversal():
    a,b=prediction(80,-40),prediction(-5,45)
    p=pair_result(a,b)
    assert p['reference_deg']==pytest.approx(85)
    assert p['estimate_deg']==pytest.approx(85)
    assert p['error_deg']==pytest.approx(0)
    assert pair_result(b,a)['estimate_deg']==pytest.approx(-85)
    assert wrap(180)==-180
    assert wrap(-181)==179
    assert pair_result(prediction(179,0),prediction(-179,2))['error_deg']==-4

def test_invalid_references_and_vectors():
    assert azimuth(0,0) is None
    assert azimuth(float('nan'),1) is None
    assert pair_result(prediction(None,0),prediction())['error_deg'] is None
    assert reference_headings({'left-forward':yaw(20)})=={}
    with pytest.raises(ValueError): reference_headings({'front-forward':np.zeros((3,3))})
    with pytest.raises(ValueError): quaternion_rotation([0,0,0,0])

def test_atomicity_and_duplicates(tmp_path):
    c,_,_=fixture_db(tmp_path/'db')
    predictions={cam:prediction() for cam in CAMERAS}
    predictions[CAMERAS[-1]]['bad']=float('nan')
    with pytest.raises(ValueError): commit_timestamp(c,'test','1',0,predictions,1,'now')
    assert c.execute('SELECT COUNT(*) FROM timestamps').fetchone()[0]==0
    assert c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]==0
    del predictions[CAMERAS[-1]]['bad']
    commit_timestamp(c,'test','1',0,predictions,1,'now')
    with pytest.raises(sqlite3.IntegrityError):commit_timestamp(c,'test','1',0,predictions,1,'now')
    assert c.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==10
    assert c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]==5

def test_resume_fingerprint(tmp_path):
    c,manifest,config=fixture_db(tmp_path/'db')
    ensure_run(c,'test',manifest,config,'later')
    with pytest.raises(ValueError): ensure_run(c,'test',manifest,{**config,'method':'scan'},'later')
    with pytest.raises(ValueError): ensure_run(c,'test',{**manifest,'changed':True},config,'later')

def test_missing_camera_explicit_invalid_pairs(tmp_path):
    c,_,_=fixture_db(tmp_path/'db')
    predictions={cam:prediction() for cam in CAMERAS}
    predictions['left-backward']=dict(sun_deg=None,heading_deg=None,reason='missing_image')
    commit_timestamp(c,'test','1',0,predictions,1,'now')
    assert c.execute('SELECT COUNT(*) FROM pairs WHERE error_deg IS NULL').fetchone()[0]==4
    assert c.execute('SELECT COUNT(*) FROM pairs WHERE error_deg IS NOT NULL').fetchone()[0]==6

def test_backup_restore_and_tampering(tmp_path):
    c,_,_=fixture_db(tmp_path/'db')
    commit_timestamp(c,'test','1',0,{cam:prediction() for cam in CAMERAS},1,'now')
    info=backup(tmp_path/'db',tmp_path/'backup')
    receipt=restore_check(tmp_path/'backup',tmp_path/'restored')
    assert receipt['verified']
    assert receipt['counts']['pairs']==10
    with (tmp_path/'backup/results.sqlite').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError): restore_check(tmp_path/'backup',tmp_path/'badrestore')

def test_metric_definitions():
    s=statistics([-10,0,10])
    assert s['mae_deg']==pytest.approx(20/3)
    assert s['rmse_deg']==pytest.approx(math.sqrt(200/3))
    assert s['within_5']==pytest.approx(1/3)
    assert s['circular_bias_deg']==pytest.approx(0)
    assert statistics([])=={'valid':0}

def test_colmap_binary_and_truncation(tmp_path):
    (tmp_path/'cameras.bin').write_bytes(struct.pack('<QiiQQdddddddd',1,7,5,1920,1080,*range(1,9)))
    b=struct.pack('<Qidddddddi',1,42,1,0,0,0,1,2,3,7)+b'front-forward/1.jpeg\0'+struct.pack('<Q',0)
    (tmp_path/'images.bin').write_bytes(b)
    row=colmap(tmp_path)['front-forward/1.jpeg']
    assert row['image_id']==42
    assert row['calibration']['model']=='OPENCV_FISHEYE'
    assert np.allclose(row['c2w'],np.eye(3))
    (tmp_path/'images.bin').write_bytes(b[:-1])
    with pytest.raises(ValueError):colmap(tmp_path)
