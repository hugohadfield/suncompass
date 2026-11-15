# Metropolis / 360 panorama evaluation

This experiment measures how consistently the existing SunCompass model predicts
sun azimuth when looking in different directions from the **same 360-degree
capture**.

It works with either:

- any standalone 2:1 equirectangular panorama; or
- a downloaded Mapillary Metropolis dataset containing
  `CAM_EQUIRECTANGULAR` images.

For each panorama, the script renders square rectilinear views at regular yaw
intervals, runs the existing SunCompass checkpoint on each view, and rotates the
predicted local sun azimuth back into panorama coordinates.

A perfectly rotation-equivariant model would return the same panorama-frame
azimuth from every virtual camera heading.

## Setup

Install SunCompass in the normal way from the repository root:

```bash
pip install -e .
```

No Metropolis SDK dependency is required. The experiment reads the small JSON
metadata tables directly.

## Run on any 360 panorama

```bash
python experiments/metropolis_360/evaluate.py \
    --panorama /path/to/panorama.jpg \
    --yaw-step-deg 10 \
    --save-views
```

The panorama should be a standard equirectangular image with roughly a 2:1
width:height ratio.

## Run on Mapillary Metropolis

Download and unpack Metropolis separately from the SunCompass repository, then run:

```bash
python experiments/metropolis_360/evaluate.py \
    --metropolis-root /path/to/metropolis \
    --split val \
    --limit 20 \
    --yaw-step-deg 10
```

The script identifies the `CAM_EQUIRECTANGULAR` sensor through:

- `<split>/sensor.json`
- `<split>/calibrated_sensor.json`
- `<split>/sample_data.json`

and resolves the corresponding panorama JPEGs from the dataset root.

Useful options:

```text
--stride N          use every Nth panorama
--pitch-deg D       virtual camera pitch; positive looks up
--hfov-deg D        virtual horizontal field of view (default 90)
--view-size N       rendered square view size (default 300)
--output-dir PATH   output directory
--save-views        save every rendered perspective view
```

The default `view-size=300` is intentional: the current SunCompass inference
pipeline centre-crops to 300 pixels before its ResNet preprocessing.

## Outputs

The default output directory is `outputs/metropolis_360/`.

`predictions.csv` contains one row per virtual view, including:

- virtual camera yaw;
- predicted forward/left components;
- predicted local azimuth;
- prediction magnitude;
- prediction transformed back to panorama azimuth;
- absolute error from the panorama's circular consensus.

`summary.csv` contains one row per panorama:

- consensus panorama-frame sun azimuth;
- mean / median / p90 consistency error;
- circular resultant length `R`;
- mean prediction magnitude.

`aggregate.json` reports aggregate statistics across all evaluated panoramas.

For the circular resultant length, `R=1` means all corrected predictions agree
perfectly and values closer to zero indicate a broad or multimodal set of
directions.

## What this experiment measures

This first experiment deliberately does **not** require astronomical ground truth.
All virtual views come from exactly the same panorama, so after correcting for
their known yaw they should agree on one sun direction.

That makes this a clean test of SunCompass's rotational consistency without
inter-camera calibration error, timestamp mismatch, or vehicle motion.

A follow-up can use Metropolis timestamp, georeferencing and pose metadata to
compare the panorama consensus against the physically predicted solar direction.
