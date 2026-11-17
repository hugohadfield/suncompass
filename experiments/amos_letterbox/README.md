# AMOS Camera 257 training and letterbox resolution

We used a fixed-camera AMOS dataset to exercise SunCompass end to end, from
generating camera-relative targets and auditing the splits to training,
checkpoint selection, and evaluation on held-out dates. We then repeated the
50-epoch retrain at a smaller letterbox size on the same RTX A6000.

## Dataset and evaluation

Each source is a native 1024×576 frame from AMOS Camera 257. Each capture
contributes an original and a horizontal mirror. Mirroring reverses the
leftward target component and keeps both versions in the same split.

| Split | Unique captures | Model inputs |
| --- | ---: | ---: |
| Training | 5,000 | 10,000 |
| Validation | 1,000 | 2,000 |
| Test | 750 | 1,500 |

Targets are horizontal sun-direction vectors derived from capture time,
location, and estimated camera heading. Angular error is the absolute wrapped
difference between the predicted and labeled camera-relative azimuth. These
labels measure agreement with the dataset calibration. The experiment covers
one camera, so it provides no estimate of accuracy on an independently
calibrated camera.

The 1024×1024 split audit checked all 13,500 inputs. No source captures, local
capture dates, file hashes, or exact decoded pixels, including pixels matched
under horizontal reflection, appeared in multiple splits. The optimizer used
the training split. Validation selected the checkpoint, and test inference
ran after training. The 612×612 retrain reused the original split lists.

## Learning and resolution results

Both runs used ImageNet-initialized ResNet-18, a fresh regression head, batch
size 32, Adam at 0.0001, and 50 epochs on the same RTX A6000. Random
augmentation and dropout were disabled. Inputs preserved the full field of
view with aspect-ratio-preserving letterboxing. RT v1a consumed the packaged
black-padded PNGs. RT v1c bilinearly resized each full packaged square to
612×612. Both applied ImageNet normalization once. Each run selected
the checkpoint with the lowest validation median angular error.

The 1024×1024 validation median went from 5.69° at epoch 1 to 1.22° at epoch 43.
On the test split, its median was 1.28° and its 95th percentile was 9.34°.
There were 28 test inputs at or above 20° error, from 16 distinct captures.
The worst error was 109.45°.

| Metric | RT v1a · 1024×1024 | RT v1c · 612×612 |
| --- | ---: | ---: |
| Average epoch | 4 min 44.4 s | 1 min 43.3 s |
| Sum of 50 epochs | 3 h 56 min 58 s | 1 h 26 min 6 s |
| Selected checkpoint | Epoch 43 | Epoch 47 |
| Validation median | 1.216° | 1.083° |
| Test median | 1.276° | 1.206° |

Logged epoch times include evaluation and checkpointing. The final test ran
separately. In this pair of runs, 612×612 took 63.7% less time and had similar
median error on the same test inputs. The records cover one run per resolution.
The 612×612 comparison summary supplies a test median and loss. A paired
tail-error comparison would require per-image errors for that checkpoint.

The [evaluation report](rt-v1a-evaluation.pdf) shows the learning curves and
held-out error histograms. The [results file](results.json) records the run
summaries and split fingerprints. The split counts include mirrors as separate inputs, so uncertainty
estimates should account for the pairing.
