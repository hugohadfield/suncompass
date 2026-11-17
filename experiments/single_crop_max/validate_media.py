"""Check decoded movie format and saved frame semantics against the accepted template."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from PIL import Image
import imageio_ffmpeg


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_gif(path):
    with Image.open(path) as im:
        size=im.size;durations=[]
        for k in range(im.n_frames):
            im.seek(k);im.load();durations.append(im.info['duration'])
        return dict(size=list(size),frames=len(durations),durations=durations,loop=im.info.get('loop'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
    a=p.parse_args();out=a.output
    gif=out/'scene_006-single-max-five-cameras.gif';mp4=gif.with_suffix('.mp4')
    baseline=inspect_gif(a.template);actual=inspect_gif(gif)
    assert actual==baseline
    assert actual['frames']==100 and actual['size']==[1584,1132]
    assert actual['durations']==[200]*100 and actual['loop']==0
    checks=json.loads((out/'validation.json').read_text());assert len(checks)==1
    check=checks[0];assert check['method']=='single-max' and check['input_hashes_verified']==1000
    rows=[r for r in csv.DictReader(a.predictions.open()) if r['method']=='single-max']
    preds={(r['timestamp'],r['camera']):r for r in rows}
    ts=sorted({r['timestamp'] for r in rows});assert len(ts)==200 and len(rows)==len(preds)==1000
    frames=check['frames_and_angles'];assert len(frames)==100
    cameras={'left-forward','front-forward','right-forward','left-backward','right-backward'}
    for i,frame in zip(range(0,200,2),frames):
        assert frame['frame_index']==i and frame['timestamp']==ts[i]
        assert set(frame['aligned_sun_angles'])==cameras
        for cam,angle in frame['aligned_sun_angles'].items():
            row=preds[ts[i],cam]
            expected=(float(row['sun_deg'])+float(row['heading_deg'])+180)%360-180
            assert math.isclose(angle,expected,abs_tol=1e-12)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(mp4),'-f','null','-'],check=True,capture_output=True)
    for i in (0,100,198):
        with Image.open(out/f'scene_006-single-max-frame-{i:03}.png') as im:
            im.load();assert list(im.size)==actual['size']
    receipt={'passed':True,'exact_template_format':actual,'template_sha256':digest(a.template),'predictions_sha256':digest(a.predictions),'all_gif_frames_decoded':True,'mp4_full_decode':True,'named_camera_timestamp_angle_checks':500,'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}}
    (out/'independent-validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('artifacts','exact_template_format')},indent=2))


if __name__=='__main__':main()
