# RTX 3070 resolution and training capacity

## Measured outcome — 6 November 2026 Pacific

**The RTX 3070 ran both 1024×1024 and 1080×1080 probes.** With the existing
partial fine-tuning policy, FP16 mixed precision and batch 32 measured:

| Canvas | Training images/s | Peak reserved VRAM | Estimated headroom |
| --- | ---: | ---: | ---: |
| 1024×1024 | 142.9 | 4.13 GiB | 3.26 GiB |
| 1080×1080 | 124.5 | 4.59 GiB | 2.81 GiB |

Start the pilot at **1024×1024, AMP, batch 32** under the existing fine-tuning
policy. A 1024×512 source fits without reducing its width; a 1080 canvas adds
upsampling without adding source detail. 1080 remains a supported option.
If every layer is unfrozen, use the measured conservative AMP batches of 16
at 1024 or 8 at 1080. With the existing policy and FP32, use batch 16.
These are starting configurations for a real-data smoke run.

The 152-case sweep includes 138 successful probes and 14 expected memory-limit
results. All four training profiles passed batch-one probes through 4096×4096;
the absolute batch-one resolution limit was not reached. Two three-minute 1080
runs completed 693 partial/FP32 and 1,399 full/AMP optimizer steps. Their median
GPU loads were 100% and 99%, with zero observed device-memory growth between
first/last steady thirds, and peak temperatures 78°C and 80°C. All 35 repository
tests passed with the CUDA contract tests enabled.

See the [full measurements](report.md), [compact numerical summary](summary.json),
and the [independent audit](verification.json). This is a compute-capacity
experiment using synthetic data; it does not measure end-to-end loader throughput
or train a new production checkpoint.

## Method and reproduction

This benchmark measures the existing two-output SunCompass ResNet18 at its
**actual convolution input resolution**, bypassing the internal 224 resize with
`preprocessed=True`. It does not change the training loader or inference API.

The source input is seeded synthetic RGB at 1024×512, resized proportionally and
centered on a square RGB(124,116,104) canvas, then ImageNet-normalized once.
Padding is dense and consumes the same convolution memory as image content.
Each batch member has independently generated content to avoid degenerate
batch-normalization statistics from exact duplicate inputs. Targets are
synthetic `[cos(angle), sin(angle)]` forward/left components. No
AMOS data or accuracy assessment is involved.

Profiles use the unchanged original checkpoint, MSE and Adam (learning rate
1e-5 for these synthetic stress probes): `partial` trains
layer3, layer4, the original 1000-output classifier and fc_end exactly as the
current trainer does; `full` unfreezes every parameter; `inference` uses eval and
disabled gradients. FP32 and FP16 autocast/GradScaler are measured separately.
TF32 and cuDNN autotuning are disabled; cuDNN deterministic mode is enabled.
Each child starts with fresh weights/Adam state. Warmup includes Adam state
allocation. The first conv input shape is asserted; training probes require
finite losses/outputs, changed output-head parameters (weight or bias) and no
scaler overflows during measured steps. AMP starts at scale 1024; any warmup
scale calibration is recorded separately. The bias check matters because the checkpoint's ReLU can
zero every classifier feature for a synthetic input while the bias still trains.

Every capacity case runs in a fresh subprocess. Expected OOM results are saved,
including explicit cuBLAS/cuDNN allocation failures; unrelated runtime errors
stop the run. CUDA memory is released by process exit. Larger batches stop after their
first OOM, with integer refinement at 1024/1080. Batch sizes above 32 are not
tested. Resolution probes at batch 1 extend to 4096 or first OOM. Two 180-second
1080 training soaks use the largest tested batch with estimated ≥1 GiB headroom.
Temperature telemetry is sampled once per second, with an 85°C stop. No GPU
power, clock, fan, driver or persistent cluster configuration is changed.

Run from a committed checkout in an environment with one visible CUDA GPU:

```sh
python experiments/resolution_benchmark/benchmark.py \
  --weights suncompass/3_224_224_resnet_baseline_5_all.pth \
  --output /path/to/new/results
```

The output directory must be new. Provenance, environment, individual case
JSON/log files, aggregate results, and sampled telemetry persist on disk.
`--reuse /path/to/prior/results` can retain completed probes with the same
compute contract, checkpoint, host and settings. Every reused result retains
its original source revision and artifact path; the parent provenance is
embedded. Failed probes are rerun. This is explicit reuse of recorded evidence,
not a claim that those cases executed again under the new controller revision.
Throughput includes synchronized optimizer steps and finite-value checks but
excludes input decoding, CPU augmentation, transfers, prefetch, validation and
checkpointing. Peak reservation includes warmup. Estimated VRAM headroom uses
PyTorch peak reservation plus end-of-case non-PyTorch device use; sampled device
memory provides a separate cross-check. This is a practical compute-capacity
bound for this environment, not a promise of end-to-end training throughput or
the maximum possible resolution with other memory-saving techniques.

Generate the report and independently audit the saved measurements after the
controller reports `status: complete`:

```sh
python experiments/resolution_benchmark/analyze.py /path/to/results --output /path/to/report
python experiments/resolution_benchmark/verify.py /path/to/results --output /path/to/report/verification.json
```

The audit checks real convolution/target shapes, finite outputs/losses, optimizer
updates, target-resolution coverage, two sustained 1080 runs, high GPU load and
device-memory stability. It does not rerun training.
