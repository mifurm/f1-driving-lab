# F1 Driving Lab

**New student learning lab:** [eight hands-on modules](student_artifacts/index.html),
[measured experiments](STUDENT_RESULTS.md), [teacher guide and worksheets](student_artifacts/teacher_guide.html),
and [classroom deck](student_artifacts/training_deck.html).
Adds manual data collection, real in-browser training/gradients, matched controller and feature experiments,
three genuinely new test tracks, explicit leakage lessons and real noise/delay/grip experiments.
All earlier training runs and artifacts remain preserved.

**Corner and wall-learning extension:** [close-up simulations for every track](wall_artifacts/index.html),
[measured results](WALL_RESULTS.md), and [updated deck](wall_artifacts/training_deck.html).
Adds oriented-car wall contact, steering-wheel/footprint guides, near-wall recovery,
120 further training rounds, and a safety-selected checkpoint.

**GP-X experiment:** [moving cars, all tracks and full telemetry](race_artifacts/index.html),
[measured results](RACE_RESULTS.md), and
[updated training deck](race_artifacts/training_deck.html).
Includes harder circuits, 120 fresh training rounds, per-attempt learning,
actual training-episode replay, speed, braking, throttle, gears and engine RPM.

**Multi-track extension:** [new results and animations](MULTITRACK_RESULTS.md)
cover eleven training circuits, two unseen circuits, and 120 additional rounds.
The original single-track experiment below is preserved.

**A real, completed 120-round training experiment**, using a small neural network
to drive a fictional car around an original F1-inspired circuit.

## Open the results

- **[Interactive simulation](artifacts/replay.html)**: select any training round
  from 0 through 120, choose one of three test starts, play/pause, and scrub time.
  Double-click the file to open it in a browser; no server or internet is needed.
- **[PowerPoint presentation](artifacts/training_deck.pptx)**:
  15 slides with explanatory speaker notes.
- **[Browser presentation](artifacts/training_deck.html)**:
  self-contained slides; use arrow keys or the navigation buttons.
- **[Animated comparison](artifacts/visuals/training_comparison.gif)**:
  rounds 0, 5, 20, and 120, advancing at the same simulated time.
- **[Visual overview](artifacts/visuals/artifact_preview.png)** and
  **[simple plan](PLAN.md)**.

The portable `f1-driving-lab.zip` bundle includes source, dependencies, data,
trained models, every saved evaluation, animations, charts, and both decks.
It excludes installed environments and downloaded browser executables.

## What actually happened

The 2,306-parameter NumPy neural network executed **120 training rounds and 2,880
Adam weight updates**. Each round added 300 teacher-labeled states collected
while the current learner drove. Training data grew from **13,525 to 49,525**
examples. The independent action-validation dataset has 2,000 examples.

| Fixed three-start benchmark | Untrained, round 0 | First reliable round, 6 | Final round, 120 |
|---|---:|---:|---:|
| Completed laps | 0 / 3 | 3 / 3 | 3 / 3 |
| Average lap progress | 0.30% | 100% | 100% |
| Mean absolute lateral error | 1.238 m* | 0.487 m | 0.182 m |
| Average completed lap time | Not completed | 127.7 s | 110.3 s |
| Steering variation | 0.582 rad/s* | 0.0135 rad/s | 0.0176 rad/s |

*Untrained drives fail almost immediately: their short-trajectory quality
metrics are **not directly comparable** with full-lap metrics.

After learning to finish, the driver became about **63% more accurate laterally**
and **14% quicker** between rounds 6 and 120. Independent action-validation
error fell approximately **92-fold** from the untrained model. Steering
variation increased between rounds 6 and 120: **not every technique metric
improves**, and progress is not guaranteed to be monotonic.

All 121 evaluations, including the baseline, and all **363 scenario trajectories**
are retained. The final model is exactly round 120, not a substituted best
checkpoint. The separately supplied `best.npz` is round 79, ranked by scenario
completion, then progress, then lateral accuracy.

## How it learns

This is **DAgger-style imitation learning**, not reinforcement learning.
A pure-pursuit teacher supplies steering and curvature-aware target-speed
labels. The learner sees lane/heading error, speed, previous steering, and
four forward track previews. A `13 → 48 → 32 → 2` tanh network predicts steering
and target speed. A conventional low-level speed controller applies bounded
acceleration/braking, while the simulator also limits steering rate.

The initial examples include randomized recovery states and three teacher
demonstration laps. After each round, new learner-visited states receive teacher
labels and join the dataset. Random starts are used for data collection; the
three fixed evaluation starts are never intentionally used for collection,
and evaluation trajectories never enter the training data.

## Files

| Path | Contents |
|---|---|
| `driving.py` | Track, vehicle dynamics, teacher, neural network, data collection |
| `train.py` | Data preparation, 120-round training, checkpointing, fixed tests |
| `render.py` | Diagrams, charts, animations, offline replay, PowerPoint/HTML decks |
| `verify.py`, `tests/` | Dynamics/learning tests, artifact checks, browser tests |
| `artifacts/data/` | Track, initial/aggregated/validation NPZ datasets, field schema |
| `artifacts/models/` | Ten numbered checkpoints, final model, separate best model |
| `artifacts/trajectories/` | Three test trajectories for every round and teacher reference |
| `artifacts/metrics.csv` | One aggregate result row per round, including round 0 |
| `artifacts/scenario_metrics.json` | All 363 detailed test outcomes and failure reasons |
| `artifacts/manifest.json` | Configuration, seed, package versions, exact results, limitations |
| `artifacts/visuals/` | PNG/SVG charts, eight GIFs, browser screenshots, slide previews |
| `artifacts/presentation_outline.json` | Slide text, image names, and speaker notes |
| `artifacts/artifact_inventory.json` | Generated artifact sizes and SHA-256 hashes |
| `artifacts/verification.json` | Numerical/data/artifact verification report |
| `artifacts/browser_verification.json` | Browser interaction and rendering verification |

Read an example dataset or model without needing a special ML framework:

```python
import numpy as np
from driving import NeuralDriver

with np.load("artifacts/data/aggregated_training_data.npz") as data:
    print(data["features"].shape)  # (49525, 13)
    print(data["targets"].shape)   # (49525, 2)
    observations = data["features"][:5].copy()

driver = NeuralDriver.load("artifacts/models/final.npz")
print(driver.predict(observations))
```

Feature/target names, source codes, aggregation-round meaning, and trajectory
units are documented in `artifacts/data/schema.json`. NPZ files contain numeric
arrays only and do not require pickle.

## Reproduce training and visuals

Python 3.13 was used for the supplied run. The broad dependency ranges are in
`requirements.txt`; exact installed versions are in `requirements-lock.txt`.
Run these commands inside this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v

# Use a new directory: the supplied completed run is protected from overwrite.
python train.py --output new_run
python render.py --output new_run
python verify.py --output new_run
```

On macOS, `open new_run/replay.html` opens the result. On other platforms,
double-click the HTML file. Training is CPU-only, locally generated, and seeded
with 2026. Exact floating-point reproducibility across different hardware or
numerical-library builds is not guaranteed.

Optional browser verification requires Node.js and the declared Playwright
development dependency:

```bash
npm ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser -- new_run

# Include browser reports/screenshots in the artifact inventory.
python render.py --output new_run --skip-animations
python verify.py --output new_run
```

Browser tests exercise the terminal state of every one of the 363 replayed
scenarios, the playback clock, every slide image, and keyboard navigation.
The Python verifier re-drives the final checkpoint and compares its complete
trajectories with the saved results.

Create or refresh the portable bundle using `python bundle.py`; it validates
the archive CRCs and required files before reporting success.

## Honest boundaries

**This is not a controller for a physical car.** Apex Park is an original
1.597 km circuit, not an official F1 track. The car uses a kinematic bicycle model
with perfect geometry observations; it does not model tire slip, downforce,
suspension, gears, weather, camera perception, opponents, or real F1 speeds.
Off-track driving or exceeding the lateral-acceleration budget ends a test.

The experiment shows improvement on **one known circuit with one training
seed**, not generalization to unseen tracks or statistical significance.
The action-validation set is separate from training, but the repeated fixed
driving tests are a monitoring benchmark, not a blind final holdout. The teacher
is a cautious heuristic, not an optimal racing line. Larger car icons and
accelerated GIF playback are visualization choices, not altered physics.
