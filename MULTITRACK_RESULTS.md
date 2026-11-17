# Eleven circuits: improving the F1-inspired learner

## Open the new artifacts

- **[Training-process animation](multitrack_artifacts/visuals/learning_process.gif)**:
  advances through every actual training round, showing all eleven circuits,
  completed starts, dataset growth, and completed-lap lane accuracy.
- **[Interactive learning and driving replay](multitrack_artifacts/replay.html)**:
  two distinct animation modes, all 121 saved rounds, all thirteen circuits,
  all three fixed test starts, and adjustable playback.
- **[Equal-time driving comparison](multitrack_artifacts/visuals/driving_comparison.gif)**:
  three circuits at rounds 0, 1, 10, and 120.
- **[Track atlas](multitrack_artifacts/visuals/track_atlas.png)**,
  **[improvement plan](IMPROVEMENT_PLAN.md)**, and the new
  **[PowerPoint](multitrack_artifacts/training_deck.pptx)** /
  **[browser deck](multitrack_artifacts/training_deck.html)**.

Double-click `multitrack_artifacts/replay.html` to use the results offline.
On macOS:

```bash
open multitrack_artifacts/replay.html
```

## What changed

The existing car physics, teacher, failure conditions, and original experiment
are preserved. Ten different original circuit layouts were added to Apex Park:
Velocity Ring, Chicane Coast, Hairpin Harbor, Serpentine Valley, Triangle GP,
Square Sprint, Double Apex, Crescent Run, Technical Orchard, and High-Speed Mesa.
Aurora and Delta are two additional **test-only** circuits.

The new learner has **16 inputs and 2,450 parameters**. Its three extra inputs
describe current, upcoming maximum, and lookahead curvature. Every minibatch
balances the eleven training circuits; half of each circuit's examples are
recent learner states and half are historical. Three curriculum stages increase
offset, heading error, and starting speed during data collection.

This was a **fresh training run**, not a misleading relabeling of the old model.
It completed **120 rounds and 7,680 Adam updates**, growing the training dataset
from **47,658 to 285,258 labeled examples**. Another 2,200 action-validation
examples were kept separate.

## Measured improvement

| New experiment | Untrained, round 0 | First complete round, 1 | Final round, 120 |
|---|---:|---:|---:|
| Completed training-circuit starts | 0 / 33 | 33 / 33 | 33 / 33 |
| Mean training-circuit lap progress | 0.23% | 100% | 100% |
| Completed-lap mean lateral error | Not completed | 0.629 m | 0.123 m |
| Average completed lap time | Not completed | 90.1 s | 83.7 s |
| Steering variation | Not comparable | 0.0097 rad/s | 0.0104 rad/s |
| Completed unseen-circuit starts* | 0 / 6 | 6 / 6 | 6 / 6 |

*All unseen checkpoint tests were performed **after the complete training
schedule**, without changing weights or selecting a checkpoint from their
results. The table is post-training analysis, not unseen-track training feedback.

Completion happened quickly, after the first 64 updates. Subsequent training
still matters: mean lateral error over the same 33 completed tests fell about
**80.5%**, and average lap time fell about **7.2%**, between rounds 1 and 120.
Independent action-validation error fell about **74-fold** from the untrained
model. The animation therefore shows lane accuracy as well as completion, so
the improvement after round 1 remains visible.

Steering variation increased slightly; it was not directly optimized.
Round 72, not round 120, had the best training-accuracy ranking. The final model
is genuinely round 120. Regressions and the separate best model are retained.

## Comparison with the previous driver

The old single-track model completed **27 of 33** starts on the expanded training
catalog; the new model completes **33 of 33**. Both models completed **6 of 6**
unseen starts, so unseen completion is not claimed as a new advantage.

The final new model's unseen-circuit lateral error is **0.090 m**, versus
**0.100 m** for the previous driver. With only two unseen tracks and one seed,
this is descriptive evidence, not a statistical generalization guarantee.

Do **not** compare the previous model's overall training-circuit mean error
with the new model's overall mean: the former excludes its six failed laps,
while the latter includes all 33 completed laps. The per-track comparison chart
only shows a model's error when it completes all three starts on that track;
missing bars mean incomplete laps.

## Reproduce

Use the existing isolated environment, or create one with the locked
dependencies as documented in [README.md](README.md). Run inside this project:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v

# A new output directory is required: existing runs cannot be overwritten.
python multitrack.py --output multi_new
python multi_render.py --output multi_new
python multi_verify.py --output multi_new

# Optional browser validation with the declared Playwright dependency:
npm ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:multitrack -- multi_new
python multi_render.py --output multi_new --inventory-only
python multi_verify.py --output multi_new
```

The supplied run took approximately 5.6 minutes for training plus post-training
evaluation on this machine. Seed 2027, all geometry, data sources, aggregation
rounds, and model architecture are saved. Numeric reproducibility across
different hardware/library builds is not guaranteed.

## Audit and package

`multitrack_artifacts/` contains:

| Path | Evidence |
|---|---|
| `data/circuit_catalog.json`, `data/tracks/` | Thirteen named layouts and their fixed train/holdout split |
| `data/*training_data.npz`, `data/validation_data.npz`, `data/schema.json` | Numeric examples, track/source/round IDs, and field definitions |
| `models/round_000.npz` through `round_120.npz` | Every actual training checkpoint |
| `models/final.npz`, `models/best.npz` | Final weights and a separately labeled training-selected best model |
| `trajectories/` | All 4,719 scenario trajectories across training and unseen circuits |
| `metrics.csv`, `history.json`, `scenario_metrics.json` | Raw aggregate and per-track outcomes |
| `training_complete.json`, `manifest.json` | Freeze marker, model hash, configuration, and measured results |
| `transfer_baseline.json` | Honest comparison with the original single-track model |
| `animation_round_sequence.json` | Which actual rounds appear in the learning-process movie |
| `verification.json`, `browser_verification.json`, `artifact_inventory.json` | Re-driving, browser, image, dataset, and hash checks |

Run `python multi_verify.py --bundle` after browser validation to regenerate
`f1-multitrack-lab.zip`. It includes both completed experiments, their data and
animations, source, plans, decks, and dependency lock files; installed
environments, browser executables, and nested ZIP files are excluded.

## Boundaries

These are original **F1-inspired educational circuits**, not official tracks.
The experiment still uses perfect geometric observations and a kinematic
bicycle model, not F1 tires, aerodynamics, weather, vision, opponents, or physical
vehicle deployment. Multiple learning changes were combined without an
ablation study. The next controlled improvements are multiple seeds, more
unseen tracks, noisy sensors and vehicle parameters, and a smoothness-aware
learning objective; see the [plan](IMPROVEMENT_PLAN.md).
