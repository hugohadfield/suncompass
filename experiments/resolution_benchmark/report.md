# PC2 SunCompass GPU resolution benchmark

Source: `bc19afc52060ec00860e492f796de63d42064237`; Slurm job `60`.
Checkpoint SHA-256: `5b104506ec52a8f90380f031e8037e35d6d8e6ab09ee95132a01d232e092c99e`.

152 cases: 138 successful, 14 expected OOMs, 0 errors. Maximum sampled GPU temperature 80°C; power 220.2 W.

73 completed probes were retained from Slurm job `59` at source `eb53aa277a72ea7a5a8a6266fa97263b25198bc4`. The compute contract is identical; the continuation adds explicit CUDA-library allocation-failure classification. Individual records retain their original revision and artifact path.


## Requested training resolutions

Safe means a tested batch with estimated ≥1 GiB VRAM headroom. A fit at batch 32 is a lower bound on capacity. Throughput excludes input I/O, decoding, augmentation, transfers and checkpoints.

| Input | Training | Precision | Largest fit | First OOM | Safe tested batch | Images/s | Peak reserved GiB | Headroom GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512² | full | amp | 32 | — | 32 | 282.5 | 2.71 | 4.68 |
| 512² | full | fp32 | 32 | — | 32 | 140.3 | 5.00 | 2.41 |
| 512² | partial | amp | 32 | — | 32 | 536.9 | 1.09 | 6.31 |
| 512² | partial | fp32 | 32 | — | 32 | 280.7 | 2.16 | 5.24 |
| 768² | full | amp | 32 | — | 32 | 128.0 | 6.33 | 1.07 |
| 768² | full | fp32 | 16 | 32 | 16 | 64.8 | 5.93 | 1.48 |
| 768² | partial | amp | 32 | — | 32 | 248.9 | 2.36 | 5.04 |
| 768² | partial | fp32 | 32 | — | 32 | 131.5 | 4.78 | 2.63 |
| 1024² | full | amp | 28 | 29 | 16 | 72.5 | 5.35 | 2.05 |
| 1024² | full | fp32 | 16 | 17 | 8 | 36.5 | 5.30 | 2.10 |
| 1024² | partial | amp | 32 | — | 32 | 142.9 | 4.13 | 3.26 |
| 1024² | partial | fp32 | 32 | — | 16 | 78.1 | 4.25 | 3.15 |
| 1080² | full | amp | 25 | 26 | 8 | 62.9 | 3.59 | 3.81 |
| 1080² | full | fp32 | 14 | 15 | 8 | 31.3 | 5.96 | 1.44 |
| 1080² | partial | amp | 32 | — | 32 | 124.5 | 4.59 | 2.81 |
| 1080² | partial | fp32 | 32 | — | 16 | 62.4 | 4.73 | 2.67 |

## Sustained 1080 training

- partial/fp32, batch 16: 693 steps over 180.2 s, 61.5 images/s, 4.73 GiB peak reserved, 2.67 GiB estimated headroom, 78°C max; 0 AMP overflows.
- full/amp, batch 8: 1399 steps over 180.1 s, 62.2 images/s, 3.55 GiB peak reserved, 3.85 GiB estimated headroom, 80°C max; 0 AMP overflows.

## Batch-one resolution envelope

These are tested bounds, not theoretical limits. No activation checkpointing, gradient accumulation, compilation or alternative memory format was used.

- partial/fp32: largest tested fit 4096²; first tested OOM not reached.
- partial/amp: largest tested fit 4096²; first tested OOM not reached.
- full/fp32: largest tested fit 4096²; first tested OOM not reached.
- full/amp: largest tested fit 4096²; first tested OOM not reached.

## Scope and interpretation

The original loader's RandomCrop(400) is not the network input limit: the model's own default preprocessing crops to 224. Every successful probe asserts its real conv1 input shape and bypasses that preprocessing.

Synthetic 1024×512 RGB fixtures are independently generated per batch member, letterboxed with centered padding, and normalized once. All image pixels and padding are processed densely. Two targets represent forward/left components. This validates compute capacity and finite optimizer updates, not label semantics, training convergence or accuracy. The AMOS importer/exporter and production preprocessing still need separate implementation.

Partial fine-tuning retains the existing trainable layer3/layer4/classifier/head configuration. Full fine-tuning trains every parameter. Both use Adam/MSE. FP32 and FP16 AMP are separate profiles. Timed steps synchronize CUDA and check finite outputs/loss; warmup initializes optimizer state. Each case runs in a fresh process, with expected OOMs preserved. Headroom combines peak PyTorch reservation (including warmup) and end-of-case external device use. Temperature, power, utilization and device memory are sampled every second.

The CPU input pipeline and a full validation/checkpoint cycle can use additional resources; begin a real-data smoke run at a conservative batch before choosing the final training configuration. No production checkpoint was updated.

Analysis source: `b62ea79e41be982657692c9ee76e7088e7a9062e`, script SHA-256 `0d329ae3ef8c33037a030b64b78c38c1bf63fb61b8130cdb69fc6a2b58d1eccc`.
