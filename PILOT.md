# WayveScenes101 Sun Compass pilot

Compare center-crop, scan, and letterbox inference on synchronized camera images in WayveScenes101. The default pilot uses sunny `scene_006`. The reference is relative camera yaw derived from COLMAP poses.

## Source and reproducibility

- Suncompass upstream: `58e9009dbc81c0af589e2e3e2fe14abda2cfc85f`.
- Wayve code/metadata: `2fc2fa9606b328cfa3dbfa11ae1c6ace99c240eb`.
- WayveScenes101 initial release, 17 June 2024; scene archive Drive ID `181smXMRGSVGDlH1DcqODQpmiigVAnfx4`, resolved by filename from the official folder.
- `prepare` stores source metadata, archive SHA-256, original weather row, calibration hashes, per-image hashes, poses, intrinsics and distortion in `manifest.json`. Every run embeds that manifest in SQLite and verifies input hashes before inference.
- Image identifiers match across all five cameras. Their numeric differences are approximately 100,000 raw units, with small jitter; no exact timestamp unit is claimed without documentation. Reported relative seconds are ordinal frame index / 10. Original identifiers are always preserved.
- Camera order: right-forward, front-forward, left-forward, left-backward, right-backward. Each timestamp has five prediction outcomes and all ten unordered pairs, oriented by that order.
- The upstream MIT license remains in `licence.txt`. Dataset access has its own WayveScenes101 non-commercial dataset terms; do not commit or redistribute scene imagery with the public code. No dataset images or run databases belong in Git.

## Environment and model

Use a Python 3.12 virtual environment. The recorded environment used PyTorch 2.8.0, torchvision 0.23.0 and CUDA 12.8. Install the appropriate PyTorch wheels for your platform and the remaining dependencies in `requirements-pilot.lock`. Run `python -m pip check` afterward.

The pilot accepts the repository’s baseline checkpoint explicitly through `--weights`. Keep any new training outputs in external artifact storage.

Compatibility edits use torchvision's explicit `weights=` argument and PyTorch `weights_only=True` state-dictionary loading. Architecture and checkpoint values are unchanged. The upstream ResNet initialization may download its ImageNet weights once, even though the full Sun Compass state dictionary replaces them; cache warming is completed before supervised evaluation.

Both methods decode RGB with Pillow, run float32 under inference mode, disable dropout, use seed zero and deterministic CUDA, and disable TF32. `single` invokes `SunCompass.predict`; `scan` invokes `SunCompass.scan_predict` without batching or replacing its median aggregation. Both preserve the upstream 300-pixel crop, 224 resize, and ResNet preprocessing. Scanning is computationally much more expensive. Timings include instrumentation; decode, outer transforms, model-forward (including its preprocessing), and complete image prediction are recorded separately. Warm-up is excluded from measured timestamp throughput.

## Reference and metrics

Read COLMAP binary camera/image files directly, preserving original quaternion, translation and camera parameters. COLMAP stores world-to-camera rotation in OpenCV right/down/forward axes; invert it before computing reference headings. At each synchronous timestamp, express every camera optical axis in the front-forward camera frame. Reference heading is `atan2(-x, z)` after projection onto its forward-left plane. This definition is invariant to an arbitrary common world rotation; it is not a global geographic heading.

The model estimates forward and left sun components, so `sun = atan2(left, forward)`. For pair A → B, use:

```
estimated_yaw = wrap(sun_A - sun_B)
reference_yaw = wrap(heading_B - heading_A)
signed_error = wrap(estimated_yaw - reference_yaw)
```

All stored angles are degrees; `wrap` maps into [-180,180). Near-zero vector norm (<1e-8), non-finite output, absent inputs, and undefined reference projection produce NULL angles/errors with explicit reasons. Vector magnitude is diagnostic, never calibrated confidence. Camera pitch and vertical-axis misalignment are recorded because the yaw-only comparison assumes approximately aligned vertical axes. Fisheye distortion, occlusion and crop selection are not corrected or tuned in this pilot.

Reports contain MAE, median, RMSE, p90, p95, maximum, circular bias, and fractions within 5/10/20 degrees, overall and per pair. Missing/invalid results stay in coverage denominators but are excluded from numeric accuracy statistics. No frame/pair independence or generalization claim is made.

## Commands

Run from a committed checkout. Set `WAYVE_DATA` to a directory prepared with the archive and metadata layout expected by `pilot.prepare`, and choose a writable `RUN_DIR` outside the checkout.

```sh
python -m pilot prepare --data "$WAYVE_DATA"
python -m pilot validate --data "$WAYVE_DATA"
python -m pytest -q tests/test_pilot.py
python -m pilot benchmark --data "$WAYVE_DATA" --database "$RUN_DIR"/results.sqlite --weights suncompass/3_224_224_resnet_baseline_5_all.pth --method single --run-id benchmark-single
python -m pilot status --database "$RUN_DIR"/results.sqlite
python -m pilot report --database "$RUN_DIR"/results.sqlite --output "$RUN_DIR"/final-report
python -m pilot backup --database "$RUN_DIR"/results.sqlite --output "$RUN_DIR"/manual-backup-UNIQUE
python -m pilot restore-check --backup /path/to/backup --output /path/to/new/restore-directory
```

Use `--method scan` or `--method letterbox` with a distinct run ID. For a different scene, use the same `--scene-id` on prepare, validate, and inference. `run` uses the full scene by default; `--subset benchmark` selects five consecutive beginning, middle and ending timestamps. `--subset recovery` selects the first two. Repeating the identical run command resumes; committed timestamps are skipped. Changed code, environment, model or input fingerprints require a new run ID (a changed scene manifest requires a separate database).

## Data model

`scenes` embeds the original scene manifest; `runs` holds immutable provenance; `attempts` tracks starts/resumes; `timestamps` is the atomic completion boundary; `predictions` contains the five individual outcomes and detailed diagnostics; `pairs` contains ten yaw/error records; `telemetry` samples process/GPU resources every five seconds; `events` records warm-up and failures. Per-image peak allocation/reservation and available GPU memory supplement sampling. Database commit duration is diagnostic and may be NULL if an interruption happens immediately after a successful result commit; the results remain complete.

SQLite is node-local with one experiment writer. Backups use SQLite's online backup API, not a live file copy of the WAL database. Unique keys prevent duplicate counts. Tables use explicit NULLs for invalid measurements. No database service, scheduler, network database, training stack or monitoring daemon is added.

## Letterbox inference baseline

`--method letterbox` uses the same checkpoint and a single forward pass. The full
RGB image is fitted proportionally into 224×224 with Pillow bilinear resampling,
rounded dimensions clamped to 1..224, and centered RGB(124,116,104) padding.
An extra odd padding pixel goes on the right/bottom. ImageNet normalization is
applied once, and the model's internal resize/center crop is bypassed explicitly.
For 1920×1080 the content is 224×126 with 49 rows of padding above and below.
The network architecture, weights and legacy single/scan paths are unchanged.
This tests a crop-trained model on a new input distribution, not a retrained
letterbox model. Use a separate database and the preserved scene manifest.
