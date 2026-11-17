# F1 Driving Lab

**Student ML Driving Lab, version 2:** [twelve hands-on modules](student_v2_artifacts/index.html),
[measured results](STUDENT_V2_RESULTS.md), [teacher guide, rubric and printable worksheets](student_v2_artifacts/teacher_guide.html),
a [class dashboard](student_v2_artifacts/class_dashboard.html) and a [classroom deck](student_v2_artifacts/training_deck.html).
See [Student lab v2](#student-lab-v2) below for what changed, why, and how to run and test it.
To put the labs on the web for a class, see [Deploy to Azure](#deploy-to-azure).

**First student lab (preserved):** [eight hands-on modules](student_artifacts/index.html),
[measured experiments](STUDENT_RESULTS.md), [teacher guide and worksheets](student_artifacts/teacher_guide.html),
and [classroom deck](student_artifacts/training_deck.html).
Adds manual data collection, real in-browser training/gradients, matched controller and feature experiments,
three genuinely new test tracks, explicit leakage lessons and real noise/delay/grip experiments.
All earlier training runs and artifacts remain preserved; `git tag lab-v1` marks the sources that produced them.

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

## Student lab v2

A review of the first student lab found that it taught the mechanics well but measured some things badly. Version 2
keeps every preserved artifact and rebuilds the lab on evidence. Each row is something that was checked, not assumed.

| Found in the first lab | Version 2 |
|---|---|
| Validation held only deep wall-recovery states (median offset 5.08 m) while training was 89.5% near-centreline, so a 7.6x "generalization gap" was distribution shift, not overfitting | Training and validation are *different expert episodes from the same sampler* (93.6% vs 94.7% within 1 m of the centreline); near-identical starts cannot straddle splits; a separately named recovery-stress set is evaluation-only |
| Loss was presented as learning, yet closed-loop success is flat or erratic while loss falls smoothly | Every training run records nominal and recovery driving beside the loss; ten saved reference seeds show the two diverging |
| "Mean ± SD" of three binary outcomes | Seeds or fixed sections are the unit of replication: bootstrap intervals over seeds, paired comparisons with an exact sign test, Wilson intervals only for independent random starts, fewer than ten seeds labelled exploratory. In the browser a diverged seed counts as a failure; the reference pipeline records divergence and `student_verify.py` refuses to bless a run that contains one |
| No student-facing DAgger or covariate shift | Module 7: narrow, broad, staged expert-only control and DAgger with equal budgets, projected state occupancy, and a step-by-step DAgger session. The saved ten-seed run reports that the closed-loop benefit is **not established** at this scale |
| One architecture and one setting | Module 6: width, depth, batch size, schedules, weight decay, gradient clipping, early stopping, label noise, and ridge / nearest-neighbour baselines |
| No account of *why* a drive failed | Module 8: expert labels at the true states, first disagreement, expert-recovery forks (a counterfactual about a state, not proof of cause) and dependency-aware permutation importance |
| Silent failures: NaN reported as "timeout", self-intersecting circuits accepted, unchecked hyper-parameters | `numerical_divergence` is a distinct outcome; circuits are validated strictly (`Apex Park` and `Square Sprint` keep their legacy inner-wall folds, flagged and excluded from headline pools); learning rate, decay and clipping are range-checked |
| Five quiz items, no teacher support | Twelve modules, 16 objectives, 20 quiz items whose wrong answers map to 22 catalogued misconceptions, a rubric, printable worksheets, a results card and a class dashboard; the answer key lives in `teaching_kit.json` |
| Every replay was a 150 m corner section and the lab said "no reward signal": nothing showed a car *learning to drive a whole lap*, and there was no reinforcement learning | Module 12 trains an imitation learner (DAgger) and a PPO reinforcement learner on one circuit and, after every stage of training, drives a whole lap from the start line: a filmstrip, a learning movie, learning curves and a stage table take both from an untrained car that crashes in its first seconds to a car that laps. See [From zero to a lap](#from-zero-to-a-lap-module-12) |
| No version control or CI | Git history, the `lab-v1` tag, GitHub Actions (`.github/workflows/ci.yml`) |

### From zero to a lap (module 12)

Both learners use the same 18-24-16-2 network, the same 18 inputs and the same car; they differ in what they are given. Imitation (DAgger, as in module 7) is given an expert's
steering and speed for the states it visits. Reinforcement is given only a reward (distance along the circuit, minus crashing, minus jerky commands, a small nudge to stay central)
and learns by proximal policy optimisation (`student_rl.js`, a small PPO actor-critic written for the lab, checked against an independent NumPy implementation). After every stage the car as
it stands drives one whole lap from the start line (policy mean, no exploration noise). Ten seeds on each of the nine valid circuits (`tools/journey_reference.cjs`, about 8 minutes on 12 cores):

| | Imitation | Reinforcement |
|---|---|---|
| Driving before the car *stays* on the lap (median, from the first stage after which every later stage laps) | 750 labelled steps (75 s of driving) | 40,960 simulated steps (about 68 min), roughly 55 times as many, with no expert labels |
| Final car completes the lap | 89 of 90 runs | 90 of 90 runs |
| Final lap time relative to the expert's | 1.05 times (mean) | 1.18 times (mean) |

Reward design is part of the lesson: with only distance, crashing and centring in the reward, all ten final cars on the featured circuit flicked their steering every step (4.5° per 0.1 s, the actuator's speed limit);
a small penalty on changing the commands removes it (0.05° per step) at a cost of about 10% in lap time. The module lets students switch that term off and watch.
Both learners are tested on the circuit they trained on, so this compares what each costs here, not which method is better; the hyper-parameters were chosen on a few circuits that the reference also contains.
`student_verify.py` replays all 42 stages of the saved example in the NumPy simulator: every outcome, steering measure and lap time agrees (to 0.1 s).

### Run it

`student_experiments.py` needs the wall lab's saved model and circuits (`wall_artifacts/`, not committed; rebuild them with the
wall-lab commands or unpack a bundle). Everything below is local and needs no server.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python student_experiments.py --output student_v2_artifacts --base wall_artifacts   # ~75 s on 14 cores
node tools/coverage_reference.cjs --artifact student_v2_artifacts                   # ten-seed DAgger reference, ~30 s
node tools/journey_reference.cjs --artifact student_v2_artifacts                    # module 12: 9 circuits x 10 seeds x 2 learners, ~8 min on 12 cores
python student_render.py --output student_v2_artifacts                              # page, teacher guide, dashboard, charts, deck
open student_v2_artifacts/index.html
```

For a classroom: hand students `index.html` (a single self-contained file plus seven scripts), keep `teacher_guide.html`, and
collect the results cards on `class_dashboard.html`. For a test no student could have inspected, generate a pack from a private
seed with `python tools/make_fresh_pack.py --seed <secret> --circuits 3 --output fresh_pack.json` and load it in module 9
after students freeze their models. The three built-in test circuits are sealed by classroom protocol only: their labels ship in the page.

### Statistics policy

- One training seed is one observation. Recipe comparisons use ten seeds and a percentile bootstrap over seeds; fewer is labelled exploratory.
- Trials inside a fixed section share a track and start family, so sections (or seeds) are the unit of replication. Nominal and recovery sections are never pooled.
- Paired comparisons share starts, seeds and noise streams, and report wins/losses/ties with an exact sign test.
- Wilson intervals appear only when every trial is an independent random start.

### Test it

```bash
python -m unittest discover -s tests          # simulator, guards, circuit validation, fixture sync, teaching kit, fresh pack (CI)
npm run test:engine                           # browser engine and statistics vs. the committed NumPy/SciPy fixture (CI)
npm run test:student                          # needs student_v2_artifacts: replays all 2,560 saved NumPy rollouts in JavaScript
npm ci && PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:student   # twelve modules, teacher tools and failure paths in Chromium
python student_verify.py --output student_v2_artifacts [--bundle]        # hashes, exact replays, teaching signals, visuals, browser report
```

The fixture (`tests/fixtures/engine_fixture.json`) stores NumPy/SciPy reference values; `tests/test_fixture_sync.py` recomputes it so a
simulator change that would make the JavaScript engine stale fails CI. The browser engine reproduces 24 fixture trajectories to
1e-6 m and every one of the 2,560 saved rollouts of the 40 retrained networks with identical outcomes.

### Numerical note

Curvature inputs are piecewise constant, so a car whose nearest road point is a centreline vertex could read the neighbouring sample
because of one ulp of rounding noise, which differs between libm builds and between the Python and JavaScript engines.
`driving.sample_index` now assigns positions within 1e-9 samples of a boundary to the sample that starts there. Against the preserved
first lab this changes a continuous metric (never an outcome) in 69 of its 2,304 saved ablation, controller and robustness rollouts; check out `lab-v1` to reproduce those files bit for bit.

### Known limits

The expert is a heuristic (about 2% of typical starts, the fast off-centre ones, trip its own lateral-acceleration limit); a learner cannot exceed its labels.
Grip is a threshold on lateral acceleration and braking, not a tyre model. There is one simulator, one expert and original circuits; nothing here supports a real-car claim.
Module 12's two learners are tested on the circuit they trained on (transfer is module 9's question), its reinforcement learner was tuned on circuits the reference run also contains, and the saved
ten-seed numbers describe that setup rather than reinforcement learning in general.

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

This is **DAgger-style imitation learning**, not reinforcement learning (this is the original single-track experiment; module 12 of the version-2 student lab adds a reinforcement learner and compares the two).
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

## Deploy to Azure

The labs are static files that run in the browser, so the whole site is one [Azure Static Web App](https://learn.microsoft.com/azure/static-web-apps/)
in one resource group: nothing to patch, no server, database or identity. `infra/` defines it in Bicep, `tools/build_site.py` stages the
website and `tools/deploy_azure.py` runs both and uploads the site. It needs the Azure CLI (2.60 or newer, signed in with `az login`) and
Node.js 18 or newer for the upload; the labs must already be generated (see above).

```bash
python tools/deploy_azure.py --what-if               # preview the infrastructure change; creates nothing
python tools/deploy_azure.py                         # infrastructure and site: every lab, Standard plan
python tools/deploy_azure.py --sku Free --labs v2    # the student lab only (about 17 MB), free
python tools/deploy_azure.py --verify-only           # compare the live site with the last build, changing nothing
az deployment sub show --name f1-driving-lab --query properties.outputs.staticWebAppUrl.value -o tsv   # the site's address
az group delete --name rg-f1-driving-lab --yes       # remove everything
```

Run again after regenerating a lab to publish the new build; `--skip-infra` skips the (unchanged) infrastructure step, `--environment`,
`--location` and `--subscription` choose names, region and subscription (`infra/main.bicepparam` holds the defaults, and a test keeps it in step with the script).

- **Plan and cost.** All six labs are about 405 MB. A Static Web App holds 250 MB on the Free plan and 500 MB per environment on Standard
  (about USD 9 a month, which is the default here); the build refuses content that does not fit the plan before anything is created.
- **What is published.** `tools/build_site.py` follows the links from each lab's pages and copies only what they reach (109 files), keeping
  the repository's paths so links between labs and to the `*_RESULTS.md` notes keep working. Models, training data, test screenshots, zip
  bundles, the template pages in the repository root and `classic_ml_samples/` stay on this machine. The build refuses a link that is
  broken, wrongly cased (Azure's file system is case-sensitive), outside the repository or to a private file, and any file that contains
  this machine's repository path (a print-to-PDF of the teacher guide once did; `tests/student-browser.mjs` now strips those links).
- **Public by default.** Anyone with the address can open the site, **including the teacher guide and its answer key**. It sends
  `X-Robots-Tag: noindex` and security headers, which keeps it out of search results but is not access control. The Standard plan can add
  authentication or IP restrictions; see the Static Web Apps documentation. There is no analytics: the labs never upload student work.
- **The upload token** is read from Azure by the script and handed to the upload in its environment; it is never printed, put on a command
  line or written to disk. The upload client is pinned (`@azure/static-web-apps-cli`).
- **Checks.** After uploading, the script downloads every served file and compares its SHA-256 with `site-manifest.json`. The browser
  smoke test opens the labs in Chromium (module 12 end to end), expects no console, page or network errors, and checks the headers,
  file types and 404 page: `PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:deployed -- https://<your-site>.azurestaticapps.net`.

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
