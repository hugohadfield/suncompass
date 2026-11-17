# AMOS import, portable export and 1024×1024 training

The AMOS training CLI uses a square RGB letterbox and ImageNet normalization.
Inference can select the same transform for checkpoints trained on that path.
The default `SunCompass()` interface continues to use the original crop-trained
checkpoint and its preprocessing.

## Import and export

Use the package's **`dataset/manifest.csv`**, whose `relative_image_path` values
are rooted at `dataset/`. Its old absolute `image_path` values are provenance;
they are not used to locate media. The importer checks actual image hashes and
dimensions, retains every source metadata field and preserves supplied splits.
Both the older `capture_timestamp_utc` and rebuilt `timestamp_utc` columns are
supported. Canonical records retain the timestamp role (`capture_estimate` or
`archive`); an archive timestamp is not represented as a verified capture clock.

When `original_relative_path` is present, the importer uses the **native JPEGs**
for training export. It also checks every supplied `letterboxed_png_sha256`,
original JPEG hash, native/input dimensions, centering, scale-one/no-crop claim,
and reference PNG pixel: native RGB content must be identical and the declared
padding must be black. The supplied preletterboxed PNGs remain reference media.
Our shared training/raw-inference transform uses RGB(124,116,104) padding, so the
native originals are used consistently for both paths. Do not mix the package's
black-padded helper inputs with an experiment trained using this export contract.

```sh
PYTHON=python

# Optional standalone canonical import; an existing different file is refused.
"$PYTHON" -m suncompass.amos import \
  --manifest /path/to/package/dataset/manifest.csv \
  --image-root /path/to/package/dataset \
  --output /path/to/run/imported.json

# Either a raw CSV or imported JSON may be exported directly.
PYTHON="$PYTHON" ./scripts/export-amos.sh \
  --manifest /path/to/run/imported.json \
  --image-root /path/to/package/dataset \
  --output /path/to/run/dataset

"$PYTHON" -m suncompass.amos verify --dataset /path/to/run/dataset
```

The output contains copied original image bytes under `train/images/` and
`test/images/` (also `val/images/` if supplied), target CSVs, full provenance in
`manifest.json` and per-file hashes/configuration in `export.json`. No image
resizing, compression or crop is performed during export. Training letterboxes
the copied images at load time. All CSV paths are relative to their CSV's
directory, so moving the exported directory does not break the loader.

Repeating export with the same inputs verifies and reuses the output. Corrupt,
unexpected or incompatible output files are rejected, never overwritten. New
exports are built in an owned temporary directory and published after checks.
The split is not regenerated: same-camera UTC source dates and identical image
bytes must not cross train/held-out boundaries. Train plus test or val is required.

## Label contract

Target CSV headers are **exactly `image_path,sun_f,sun_l`**, and each row has
exactly two float32 regression targets after loading. `sun_u` remains in the
provenance manifest; it is not emitted as a third target. Horizontal projections
are **not normalized to unit length**: their magnitude includes solar elevation.

The supported source frame is `yaw_aligned_level_FLU`, with positive left and
solar/heading angles clockwise from north:

```
sun_f = cos(elevation) * cos(solar_azimuth - heading)
sun_l = -cos(elevation) * sin(solar_azimuth - heading)
```

Supplied values are checked for finiteness, valid projection norm and agreement
with angle metadata when present. A supplied `sun_u` must complete a unit 3-D
vector. Unknown frames, missing labels, missing images, duplicate paths, dimension
or checksum mismatches, and split leakage fail explicitly. This preserves and
checks supplied labels; it does not infer camera calibration or recompute solar
positions. The package's `label_status` and all calibration/clock limitations
remain attached to every image.

## Training and inference

The following two-step smoke run uses the verified export and, optionally,
initializes from the repository's original checkpoint. CUDA mixed precision
is the default on a CUDA device. On CPU, pass `--device cpu`.

```sh
python -m suncompass.train \
  --data /path/to/run/dataset --output /path/to/new-run/smoke.pth \
  --weights suncompass/3_224_224_resnet_baseline_5_all.pth \
  --max-steps 2 --batch-size 2 --num-workers 0 --image-size 1024
```

Choose `--image-size 612` for a 612×612 training canvas. Use that same size for
inference. The original Camera 257 comparison used a separately packaged set
of black-padded images and selected checkpoints on its validation split. For
that comparison, the entire 1024×1024 package PNG was resized to 612×612 before
normalization. The export workflow below operates on native RGB sources and
uses mean-color padding.

For a deliberate training experiment, choose epochs/learning rate and remove the
step bound. `--weights` initializes from a SunCompass state dictionary; omitting
it initializes the ResNet from ImageNet. Outputs are a state dictionary plus a
`.pth.json` sidecar with preprocessing, target order, data identity and arguments.
Existing checkpoint outputs are refused. This entry point starts a new run; it
does not restore optimizer/RNG state for training resumption. TensorBoard logs
are placed beside the new checkpoint.

```python
from PIL import Image
import numpy as np
from suncompass import SunCompass

sun = SunCompass(weights_path="/path/to/new-run/model.pth", preprocessing="letterbox", image_size=1024)
sun.set_eval(dropout=False)
rgb = np.array(Image.open("/path/to/image.jpg").convert("RGB"))
forward, left = sun.predict(rgb)  # full image -> 1024 square, normalize once
```

The Wayve pilot continues to use the historical crop and inference experiment
preprocessing. Images are treated in their decoded orientation without EXIF
rotation.

The original AMOS import and preprocessing were checked on pilot packages of
2006 and 2014 Camera 257 captures. The 2014 audit checked the native JPEGs
and reference PNG geometry and passed 71 tests including CUDA contracts.
That audit is independent of the later 50-epoch resolution comparison.
