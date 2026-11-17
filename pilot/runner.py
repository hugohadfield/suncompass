"""A single worker, deterministic predictions, timestamp-sized transactions."""
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from PIL import Image
import psutil
from .core import *
from .prepare import verify_manifest

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def command(args):
    return subprocess.check_output(args, text=True).strip()

def hardware():
    return dict(host=command(['hostname']), python=sys.version, platform=command(['uname','-a']),
                gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']))

def sample(process):
    fields = 'uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw'
    return dict(gpu_csv=command(['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits']),
                fields=fields, rss_bytes=process.memory_info().rss,
                process_cpu_percent=process.cpu_percent(), recorded=now())

class Monitor:
    def __init__(self, database, run_id):
        self.database, self.run_id = database, run_id
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
    def work(self):
        c = connect(self.database)
        process = psutil.Process()
        process.cpu_percent()
        while not self.stop.is_set():
            try:
                s = sample(process)
                with c:
                    c.execute('INSERT INTO telemetry(run_id,recorded,details_json) VALUES (?,?,?)',
                              (self.run_id,now(),dumps(s)))
            except Exception as e:
                print(f'telemetry failure: {e}', flush=True)
            self.stop.wait(5)
        c.close()
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=10)

class Predictor:
    def __init__(self, weights):
        import torch
        from suncompass import SunCompass
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is forbidden for this pilot')
        torch.set_num_threads(2)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        # This historical experiment explicitly retains the 224-pixel contract.
        self.sun = SunCompass(weights_path=weights, preprocessing="legacy")
        self.sun.set_eval(dropout=False)
        self.transform_seconds = self.model_seconds = 0
        self.crop_count = 0
        original_transform = self.sun.apply_transforms
        def transform(image):
            start = time.perf_counter()
            result = original_transform(image)
            self.transform_seconds += time.perf_counter() - start
            return result
        self.sun.apply_transforms = transform
        original_letterbox = self.sun.apply_letterbox_transforms
        def letterbox_transform(image):
            start = time.perf_counter()
            result = original_letterbox(image)
            self.transform_seconds += time.perf_counter() - start
            return result
        self.sun.apply_letterbox_transforms = letterbox_transform
        original_single_max = self.sun.apply_single_max_transforms
        def single_max_transform(image):
            start = time.perf_counter()
            result = original_single_max(image)
            self.transform_seconds += time.perf_counter() - start
            return result
        self.sun.apply_single_max_transforms = single_max_transform
        original_forward = self.sun.model.forward
        def forward(x, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_forward(x, *args, **kwargs)
            torch.cuda.synchronize()
            self.model_seconds += time.perf_counter() - start
            self.crop_count += 1
            return result
        self.sun.model.forward = forward
    def predict(self, path, method):
        torch = self.torch
        start = time.perf_counter()
        with Image.open(path) as im:
            image = np.array(im.convert('RGB'))
        decode = time.perf_counter()-start
        self.transform_seconds = self.model_seconds = 0
        self.crop_count = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            if method == 'single':
                outputs = self.sun.predict(image)
            elif method == 'scan':
                outputs = self.sun.scan_predict(image)
            elif method == 'letterbox':
                outputs = self.sun.letterbox_predict(image)
            elif method == 'single-max':
                outputs = self.sun.single_max_predict(image)
            else:
                raise ValueError(f"Unknown method: {method}")
        torch.cuda.synchronize()
        infer = time.perf_counter()-start
        values = list(map(float, outputs))
        finite = len(values) == 2 and np.isfinite(values).all()
        sun = azimuth(*values) if finite else None
        details = dict(raw_outputs=[v if math.isfinite(v) else None for v in values],
                       vector_magnitude=math.hypot(*values) if finite else None,
                       sun_deg=sun, reason=None if sun is not None else 'nonfinite_or_near_zero_vector',
                       decode_seconds=decode, prediction_seconds=infer,
                       transform_seconds=self.transform_seconds, model_seconds=self.model_seconds,
                       crop_count=self.crop_count,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       rss_bytes=psutil.Process().memory_info().rss,
                       gpu_free_bytes=torch.cuda.mem_get_info()[0])
        if details['gpu_free_bytes'] < 2**30:
            raise RuntimeError('less than 1 GiB GPU headroom')
        return details

def choose_timestamps(manifest, subset):
    ts = manifest['timestamps']
    if subset == 'all': return ts
    if subset == 'recovery': return ts[:2]
    if subset == 'benchmark':
        if len(ts) < 15: raise ValueError('scene too short for benchmark')
        return ts[:5] + ts[len(ts)//2-2:len(ts)//2+3] + ts[-5:]
    raise ValueError('unknown subset')

def execute(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    root = Path(args.data).resolve()
    database = Path(args.database).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    # One process owns all GPU experiment work, including across different databases.
    lock = open(Path.home() / '.suncompass-pilot.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root/'manifest.json').read_text())
    verify_manifest(root, manifest)
    timestamps = choose_timestamps(manifest,args.subset)
    weights = Path(args.weights).resolve()
    repo = Path(__file__).resolve().parents[1]
    if command(['git','-C',str(repo),'status','--porcelain','--untracked-files=normal']):
        raise RuntimeError('refuse experiment with dirty code checkout')
    packages = command([sys.executable,'-m','pip','freeze'])
    config = dict(method=args.method, subset=args.subset, timestamps=timestamps,
                  manifest_hash=fingerprint(manifest), checkpoint_sha256=sha256(weights),
                  upstream_revision=UPSTREAM, wayve_revision=WAYVE,
                  code_revision=command(['git','-C',str(repo),'rev-parse','HEAD']),
                  environment=packages, hardware=hardware(), seed=0, dtype='float32',
                  conventions=CONVENTIONS, metrics_version='wrapped-error-degrees-v1',
                  preprocessing=dict(decoder='Pillow RGB',center_crop=300,resize=224,
                     resnet_weights_transform='IMAGENET1K_V1',dropout=False,scan_stride=20,
                     scan_edges='upstream exclusive width/height minus 300',aggregate='componentwise median'),
                  cuda_device=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    if args.method == 'letterbox':
        config['preprocessing'] = dict(
            decoder='Pillow RGB', center_crop=None, resize=224,
            resampling='Pillow bilinear', dimension_rounding='nearest integer clamped 1..224',
            padding_rgb=[124,116,104], padding_alignment='center extra odd pixel right/bottom',
            normalization_mean=[.485,.456,.406], normalization_std=[.229,.224,.225],
            internal_crop=False, dropout=False, model_passes=1)
    if args.method == 'single-max':
        config['preprocessing'] = dict(
            decoder='Pillow RGB', center_crop='min(width,height)', crop_origin='floor((width-side)/2), floor((height-side)/2)',
            odd_excess='right/bottom', resize=224, resampling='Pillow bilinear',
            normalization_mean=[.485,.456,.406], normalization_std=[.229,.224,.225],
            internal_crop=False, dropout=False, model_passes=1)
    c=connect(database)
    ensure_run(c,args.run_id,manifest,config,now())
    if c.execute('SELECT status FROM runs WHERE run_id=?',(args.run_id,)).fetchone()[0] == 'complete':
        print('Already complete:',args.run_id,flush=True)
        return
    with c:
        c.execute("UPDATE attempts SET ended=?,status='interrupted' WHERE run_id=? AND status='running'",(now(),args.run_id))
        attempt = c.execute('INSERT INTO attempts(run_id,started,status,details_json) VALUES (?,?,?,?)',
                     (args.run_id,now(),'running',dumps(dict(argv=sys.argv,pid=os.getpid())))).lastrowid
        c.execute("UPDATE runs SET status='running',ended=NULL WHERE run_id=?",(args.run_id,))
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    done={r[0] for r in c.execute('SELECT timestamp FROM timestamps WHERE run_id=?',(args.run_id,))}
    prediction_count=0
    try:
        with Monitor(database,args.run_id):
            predictor=Predictor(str(weights))
            first=manifest['records'][timestamps[0]][CAMERAS[0]]
            warmup=predictor.predict(root/first['path'],args.method)
            with c:
                c.execute('INSERT INTO events(run_id,recorded,kind,details_json) VALUES (?,?,?,?)',
                          (args.run_id,now(),'warmup',dumps(warmup)))
            for ts in timestamps:
                if ts in done: continue
                if shutil.disk_usage(database.parent).free < 20*2**30:
                    raise RuntimeError('disk free space fell below 20 GiB')
                start=time.perf_counter()
                predictions={}
                rows=manifest['records'][ts]
                for cam in CAMERAS:
                    row=rows.get(cam)
                    ref=(row or {}).get('reference') or {}
                    p=dict(sun_deg=None,heading_deg=ref.get('heading_deg'),reason='missing_image',
                           reference=ref)
                    if row:
                        p.update(input_sha256=row['sha256'], path=row['path'])
                        try:
                            p.update(predictor.predict(root/row['path'],args.method))
                        except (OSError, ValueError) as e:
                            p.update(reason=f'input_or_prediction_error:{type(e).__name__}:{e}')
                        if p['heading_deg'] is None:
                            p['reason']='missing_or_invalid_reference'
                    predictions[cam]=p
                    prediction_count+=1
                    if args.fault_after_predictions and prediction_count == args.fault_after_predictions:
                        print('Injected SIGKILL mid-timestamp',ts,flush=True)
                        os.kill(os.getpid(),signal.SIGKILL)
                wall=time.perf_counter()-start
                commit_start=time.perf_counter()
                index=manifest['timestamps'].index(ts)
                commit_timestamp(c,args.run_id,ts,index/10,predictions,wall,now())
                commit_seconds=time.perf_counter()-commit_start
                with c:
                    c.execute('UPDATE timestamps SET commit_seconds=? WHERE run_id=? AND timestamp=?',
                              (commit_seconds,args.run_id,ts))
                print(dumps(dict(run=args.run_id,timestamp=ts,completed=len(done)+1,total=len(timestamps),seconds=wall)),flush=True)
                done.add(ts)
        with c:
            c.execute("UPDATE runs SET status='complete',ended=? WHERE run_id=?",(now(),args.run_id))
            c.execute("UPDATE attempts SET status='complete',ended=? WHERE id=?",(now(),attempt))
    except BaseException as e:
        with c:
            c.execute("UPDATE runs SET status='interrupted',ended=? WHERE run_id=?",(now(),args.run_id))
            c.execute("UPDATE attempts SET status='interrupted',ended=? WHERE id=?",(now(),attempt))
            c.execute('INSERT INTO events(run_id,recorded,kind,details_json) VALUES (?,?,?,?)',
                      (args.run_id,now(),'failure',dumps(dict(type=type(e).__name__,message=str(e)))))
        raise
    finally:
        c.close()
        lock.close()
