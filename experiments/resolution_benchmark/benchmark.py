#!/usr/bin/env python3
"""Isolated SunCompass CUDA capacity probes; never changes production preprocessing.

Run the controller inside a Slurm GPU allocation. Each case gets a fresh process
so an expected CUDA OOM cannot contaminate the next measurement. No downloads,
training checkpoints, dataset labels, or production result files are created.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 ** 2
PROBE_CONTRACT = "independent-rgb-adam1e-5-amp1024-v1"


def digest(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def child(args):
    import numpy as np
    from PIL import Image
    import torch
    import torchvision
    sys.path.insert(0, str(ROOT))
    from suncompass.pretrained_resnet import ResNetRegression

    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(size=args.size, batch=args.batch, mode=args.mode,
                  precision=args.precision, seconds=args.seconds,
                  started=time.time(), status="error")
    try:
        # Refuse an implicit ImageNet download; full SunCompass weights follow.
        cache = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        if not cache.is_file():
            raise RuntimeError(f"Required existing ImageNet cache missing: {cache}")
        model = ResNetRegression(output_size=2, fine_tune=args.mode != "full")
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        if args.mode == "full":
            model.requires_grad_(True)
        model = model.cuda()
        model.train(args.mode != "inference")
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-5) if args.mode != "inference" else None
        scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=args.precision == "amp")

        # Real, geometry-preserving 1024x512 -> square letterbox on CPU, then
        # resident NCHW FP32 GPU input (including padded pixels' actual cost).
        content_height = max(1, round(args.size / 2))
        tensors = []
        for _ in range(args.batch):
            image = Image.fromarray(np.random.randint(0, 256, (512, 1024, 3), dtype=np.uint8))
            canvas = Image.new("RGB", (args.size, args.size), (124, 116, 104))
            canvas.paste(image.resize((args.size, content_height), Image.Resampling.BILINEAR),
                         (0, (args.size - content_height) // 2))
            array = np.array(canvas).astype(np.float32).transpose(2, 0, 1) / 255.0
            array = (array - np.array([.485, .456, .406], dtype=np.float32)[:, None, None])
            array /= np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
            tensors.append(torch.from_numpy(array))
        inputs = torch.stack(tensors).cuda()
        del tensors, array
        angles = torch.linspace(-2.5, 2.5, args.batch, device="cuda")
        targets = torch.stack((angles.cos(), angles.sin()), dim=1)
        seen = []
        hook = model.resnet.conv1.register_forward_pre_hook(
            lambda module, arguments: seen.append(list(arguments[0].shape)))
        original = torch.cat([p.detach().flatten() for p in model.fc_end.parameters()]).clone()
        overflow_count = 0

        def step():
            nonlocal overflow_count
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(optimizer is not None):
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "amp"):
                    output = model(inputs, preprocessed=True)
                    loss = torch.nn.functional.mse_loss(output, targets)
                if optimizer:
                    scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    overflow_count += int(scaler.get_scale() < scale)
            torch.cuda.synchronize()
            if output.shape != (args.batch, 2) or not torch.isfinite(output).all().item():
                raise RuntimeError("Non-finite or incorrectly shaped two-component output")
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite MSE")
            return loss.item()

        for _ in range(args.warmup):
            step()
        assert seen[0] == [args.batch, 3, args.size, args.size], seen[0]
        hook.remove()
        warmup_overflows = overflow_count
        overflow_count = 0
        original = torch.cat([p.detach().flatten() for p in model.fc_end.parameters()]).clone()
        warmup_peak = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        times = []
        losses = []
        started = time.perf_counter()
        while len(times) < args.steps or time.perf_counter() - started < args.seconds:
            tick = time.perf_counter()
            losses.append(step())
            times.append(time.perf_counter() - tick)
        free, total = torch.cuda.mem_get_info()
        peak_reserved = max(warmup_peak, torch.cuda.max_memory_reserved())
        # Includes CUDA context/non-PyTorch memory; conservative allocator peak.
        external = max(0, total - free - torch.cuda.memory_reserved())
        estimated_headroom = total - peak_reserved - external
        changed = not torch.equal(original, torch.cat([p.detach().flatten() for p in model.fc_end.parameters()]))
        if optimizer and (not changed or overflow_count):
            raise RuntimeError(f"Invalid training probe: changed={changed}, overflows={overflow_count}")
        result.update(status="ok", steps=len(times), elapsed_seconds=sum(times),
                      images_per_second=args.batch * len(times) / sum(times),
                      median_step_ms=statistics.median(times) * 1000,
                      p95_step_ms=float(np.percentile(times, 95)) * 1000,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / MIB,
                      peak_reserved_mib=peak_reserved / MIB,
                      external_mib=external / MIB,
                      estimated_headroom_mib=estimated_headroom / MIB,
                      device_total_mib=total / MIB,
                      actual_conv1_input=seen[0], actual_target_shape=list(targets.shape),
                      first_loss=losses[0], last_loss=losses[-1],
                      optimizer_updated=changed, amp_overflows=overflow_count,
                      warmup_amp_overflows=warmup_overflows,
                      trainable_parameters=sum(p.numel() for p in trainable),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      torch=torch.__version__, torchvision=torchvision.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(), python=platform.python_version())
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except Exception as error:
        # CUDA libraries can fail to allocate handles/workspaces before the
        # PyTorch allocator can raise torch.OutOfMemoryError.
        allocation_failure = any(marker in str(error) for marker in
                                 ("CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED"))
        result.update(status="oom" if allocation_failure else "error", error=repr(error))
    result["finished"] = time.time()
    write_json(args.result, result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("ok", "oom") else 1


def telemetry():
    fields = "temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total,clocks.sm"
    command = ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    values = subprocess.check_output(command, text=True).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("Benchmark requires exactly one visible GPU")
    return dict(zip(("temp_c", "power_w", "gpu_util_pct", "used_mib", "total_mib", "clock_mhz"),
                    map(float, values[0].split(","))), timestamp=time.time())


def controller(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Run inside a Slurm GPU allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, code_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        checkpoint_sha256=digest(args.weights), benchmark_sha256=digest(Path(__file__)),
        hostname=platform.node(), slurm_job_id=os.environ["SLURM_JOB_ID"],
        started=time.time(), input="synthetic RGB 1024x512, square letterbox, ImageNet normalization",
        scope="resident-input compute capacity, excludes decode/I/O/prefetch and accuracy",
        cudnn_benchmark=False, deterministic_cudnn=True, tf32=False,
        warmup=args.warmup, steps=args.steps, stop_temperature_c=args.max_temp,
        probe_contract=PROBE_CONTRACT)
    reusable = {}
    if args.reuse:
        previous = json.loads((args.reuse / "provenance.json").read_text())
        # This predecessor has the same compute contract, before the metadata
        # field was introduced; only OOM classification has changed.
        compatible = previous.get("probe_contract") == PROBE_CONTRACT or previous["code_revision"] == "eb53aa277a72ea7a5a8a6266fa97263b25198bc4"
        if not compatible or any(previous[key] != metadata[key] for key in
                                 ("checkpoint_sha256", "warmup", "steps", "input", "hostname")):
            raise RuntimeError("Reuse requires the same compute contract, checkpoint, host and settings")
        metadata["reused_provenance"] = previous
        metadata["reused_from"] = str(args.reuse)
        for item in json.loads((args.reuse / "results.json").read_text()):
            if item["status"] in ("ok", "oom"):
                reusable[(item["size"], item["batch"], item["mode"], item["precision"], item["seconds"])] = item
    write_json(args.output / "provenance.json", metadata)
    (args.output / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    (args.output / "nvidia-smi-before.txt").write_text(subprocess.check_output(["nvidia-smi", "-q"], text=True))
    results = []

    def probe(size, batch, mode, precision, seconds=0):
        name = f"{len(results):03d}-{size}-b{batch}-{mode}-{precision}" + ("-soak" if seconds else "")
        path = args.output / (name + ".json")
        key = (size, batch, mode, precision, seconds)
        if key in reusable:
            result = dict(reusable[key])
            result.update(case=name,
                          source_revision=result.get("source_revision", previous["code_revision"]),
                          reused_from=str(args.reuse / (reusable[key]["case"] + ".json")))
            write_json(path, result)
            results.append(result)
            write_json(args.output / "results.json", results)
            print("REUSED", name, result["status"], flush=True)
            return result
        command = [sys.executable, str(Path(__file__)), "--child", "--weights", str(args.weights),
                   "--size", str(size), "--batch", str(batch), "--mode", mode, "--precision", precision,
                   "--warmup", str(args.warmup), "--steps", str(args.steps), "--seconds", str(seconds),
                   "--result", str(path)]
        print("START", name, flush=True)
        samples = []
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            start = time.monotonic()
            try:
                while process.poll() is None:
                    sample = telemetry()
                    samples.append(sample)
                    with (args.output / "telemetry.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(case=name, **sample)) + "\n")
                    if sample["temp_c"] >= args.max_temp:
                        raise RuntimeError(f"Temperature stop: {sample['temp_c']} C")
                    if time.monotonic() - start > max(180, seconds + 90):
                        raise RuntimeError(f"Case timeout: {name}")
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if not path.exists():
            raise RuntimeError(f"Child failed, see {name}.log, exit {process.returncode}")
        result = json.loads(path.read_text())
        result.update(case=name, source_revision=metadata["code_revision"],
                      max_temp_c=max(s["temp_c"] for s in samples),
                      max_power_w=max(s["power_w"] for s in samples),
                      max_device_used_mib=max(s["used_mib"] for s in samples),
                      max_gpu_util_pct=max(s["gpu_util_pct"] for s in samples))
        write_json(path, result)
        results.append(result)
        write_json(args.output / "results.json", results)
        print("RESULT", json.dumps(result), flush=True)
        if result["status"] == "error" or process.returncode:
            raise RuntimeError(f"Probe error: {name}: {result.get('error')}")
        return result

    # Baselines reflect actual old 224 input and the requested 400 comparison.
    for size in (224, 400):
        for precision in ("fp32", "amp"):
            probe(size, 32, "partial", precision)
    # Capacity bracketing, with integer refinement at the requested resolutions.
    for size in (512, 768, 1024, 1080):
        for mode in ("partial", "full"):
            for precision in ("fp32", "amp"):
                lower, upper = 0, 33
                for batch in (1, 2, 4, 8, 16, 32):
                    result = probe(size, batch, mode, precision)
                    if result["status"] == "oom":
                        upper = batch
                        break
                    lower = batch
                if size in (1024, 1080):
                    while upper - lower > 1:
                        batch = (upper + lower) // 2
                        result = probe(size, batch, mode, precision)
                        if result["status"] == "oom":
                            upper = batch
                        else:
                            lower = batch
    # Resolution envelope at microbatch 1; stop each profile at its first OOM.
    for mode in ("partial", "full"):
        for precision in ("fp32", "amp"):
            for size in (1280, 1536, 2048, 2560, 3072, 4096):
                if probe(size, 1, mode, precision)["status"] == "oom":
                    break
    for size in (1024, 1080):
        for precision in ("fp32", "amp"):
            for batch in (1, 8, 32):
                if probe(size, batch, "inference", precision)["status"] == "oom":
                    break
    # Sustained real optimizer steps at 1080 with >=1 GiB estimated headroom.
    for mode, precision in (("partial", "fp32"), ("full", "amp")):
        choices = [r for r in results if r["size"] == 1080 and r["mode"] == mode
                   and r["precision"] == precision and r["status"] == "ok"
                   and r["estimated_headroom_mib"] >= 1024]
        if not choices:
            raise RuntimeError(f"No 1080 safe-headroom candidate: {mode}/{precision}")
        batch = max(r["batch"] for r in choices)
        probe(1080, batch, mode, precision, seconds=args.soak_seconds)
    metadata.update(finished=time.time(), status="complete", cases=len(results))
    write_json(args.output / "provenance.json", metadata)
    (args.output / "nvidia-smi-after.txt").write_text(subprocess.check_output(["nvidia-smi", "-q"], text=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse", type=Path, help="Reuse compatible completed probes with their original provenance")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--soak-seconds", type=int, default=180)
    parser.add_argument("--max-temp", type=int, default=85)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--mode", choices=("partial", "full", "inference"))
    parser.add_argument("--precision", choices=("fp32", "amp"))
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1 or args.soak_seconds < 1:
        parser.error("warmup, steps and soak-seconds must be positive")
    if not args.child and args.output is None:
        parser.error("--output is required for the controller")
    return child(args) if args.child else controller(args)


if __name__ == "__main__":
    sys.exit(main())
