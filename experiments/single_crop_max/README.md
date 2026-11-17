# Sunny63 Single Crop Max — 27 September 2026

Hypothesis: retaining the largest centered square gives the frozen crop-trained
network more useful context than the original 300×300 crop. This is an inference
preprocessing comparison, with no retraining, tuning, or weight changes.

Method ID: `single-max`; display name: **Single Crop Max**. Crop side is
`min(width,height)`, origin uses floor division with odd excess removed from
right/bottom; Pillow bilinear resize to 224×224, ImageNet normalization once,
and the existing explicit preprocessed model path. One model pass per image.

Comparison: the exact 63 historical Sunny scenes, including scene_055 (Dusk),
12,606 timestamps, 63,030 images and 126,060 unordered camera pairs. Reuse the
OG single run's embedded manifests and compare input hashes, timestamp/camera
keys, checkpoint, environment, GPU and COLMAP references independently. Existing
OG predictions are reused. The scene order starts with scene_006 for validation;
no scene is selected by accuracy. There are no new training/validation splits.

Keep run databases, predictions and manifests in artifact storage. The independent validator is `verify.py`.

Reference yaw is COLMAP-derived camera orientation, not surveyed heading or solar
position. Correlated frames/pairs, fisheye views and camera tilt limit inference.
The checkpoint remains crop-trained. Single Crop Max is a selectable experiment,
not a claim of superior performance. The five-camera movie replays saved outputs.

## Completed result

Pooled MAE: **59.73° Single Crop Max versus 81.81° OG single crop** (27.0% lower).
Scene MAE improved in 54/63 scenes and worsened in nine. Within 10° increased
from 6.81% to 12.44%. All 63,030 predictions and 126,060 pairs are valid.
Scene 006 MAE is 43.36° versus 72.18° OG. Historical letterbox pooled MAE
was 63.52°; this was not rerun. Full metrics and scene list: [results.json](results.json).

Measured processing totaled 876.07 s versus historical OG 694.68 s (1.26×).
This includes decoding/instrumentation and excludes warmup/staging; historical
machine/cache state differs, so this is not a controlled latency benchmark.

All 35 implementation tests passed with CUDA checkpoint coverage and no skips.
All scene manifests, image hashes, checkpoint, package environment, GPU identity,
reference values and pair keys match OG; stored wrapped errors were independently
recomputed. Original pooled metrics reproduced the archived report exactly.

Inference/rendering revision: `37b48263f546594d4a18f44804508be92a57cbe3`.
Analysis/report/validator revision: `c5bc7d7aa1a6362d3691e31b17b20fad0d395dcd`.
Checkpoint: `5b104506ec52a8f90380f031e8037e35d6d8e6ab09ee95132a01d232e092c99e`.
The run used Python 3.12.3, Torch 2.8.0/CUDA 12.8, torchvision 0.23.0,
RTX 3070, driver 595.71.05; exact environment/configuration is embedded in each DB.
Inference was Slurm job 54; application completion.json and 63 independently
checked receipts confirm completion. Slurm accounting storage is disabled, so
no historical sacct exit-code receipt is claimed.

## Reproduction

From this repository, with the preserved private artifact root and original
library data, use the existing inference runtime and requirements-pilot.lock.
`selection.json` freezes the 63 scene IDs; stage the OG manifests and identical
image/calibration bytes under each scene's data root. Each run uses:

```sh
python -m pilot run --data DATA --database NEW_DATABASE --weights CHECKPOINT \
  --run-id sunny63-single-max-SCENE_ID --method single-max --subset all
```

Never reuse an old run ID with changed fingerprints. The checked-in batch wrapper
is specific to the documented private experiment layout and owned staging markers.

```sh
python -m experiments.single_crop_max.analyze --root ARTIFACT_ROOT \
  --revision 37b48263f546594d4a18f44804508be92a57cbe3 \
  --historical-analysis SUNNY63_HISTORICAL_ANALYSIS_JSON
python experiments/single_crop_max/render.py --data SCENE_LIBRARY \
  --predictions ARTIFACT_ROOT/comparison/scene006-predictions.csv \
  --output ARTIFACT_ROOT/media
python experiments/single_crop_max/validate_media.py --output ARTIFACT_ROOT/media \
  --predictions ARTIFACT_ROOT/comparison/scene006-predictions.csv --template ORIGINAL_SINGLE_GIF
python -m experiments.single_crop_max.write_report --root ARTIFACT_ROOT
```

Movie contract: same 1584×1132 five-camera layout, 100 sampled frames, 5 fps,
20-second loop, original Arial fonts and camera/dial semantics. Crop outline,
input preview and method label reflect Single Crop Max. Media contains private
Wayve images and is deliberately excluded from Git.
