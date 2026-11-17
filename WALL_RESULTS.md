# Taking corners, avoiding wall contact, and staying on the track

**Open the [corner and wall-learning dashboard](wall_artifacts/index.html).**
Choose any of thirteen circuits, then a left turn, right turn, left-wall recovery
or right-wall recovery. Everything works offline.

The cockpit shows a moving race car with steering front wheels, its physical
footprint, red track walls, geometric preview guides, measured wall clearance,
lane offset, heading error, speed, learned target speed, braking and gears.
Switch between close-up following and whole-circuit cameras.

## What the simulations demonstrate

The vehicle has a **4.8 m by 1.8 m oriented rectangular footprint**. Collision
checks include its front and rear corners, not just its center. Both wall
polylines are checked through each motion step, at swept samples separated by
at most 0.5 m and 0.025 radians. Any detected contact ends that drive.

The simulator applies the neural driver's steering and target-speed outputs
unchanged. Ordinary actuator bounds, automatic pedals and automatic shifting
remain, but **there is no wall-avoidance shield or hidden teacher intervention**.
Handling-limit failures, wall contact, timeouts and successful exits are
recorded separately.

Each fixed challenge covers **150 m**. A successful drive must complete that
section without failure and average less than 1 m lateral error over its final
20 states. Near-wall tests begin 6.2 m off center, pointing outward by about
15 degrees at 43.2 km/h. These are deliberately harder recovery starts, not
claims of optimal F1 racing lines.

The **scripted wall-contact illustration** deliberately holds straight steering
while pointing toward a wall. It plays in slow motion, shows a real simulated
footprint contact, and is prominently labeled **NOT ML**. It is never counted
as a learned driver's failure or training result.

## Extended learning and measured outcomes

The completed extension ran **120 additional training rounds and 7,680 new Adam
updates**, starting from the previously trained GP-X weights. It adds projected
footprint margin and a speed-dependent steering-budget observation to the
network: `18 -> 48 -> 32 -> 2`.

Both new input rows start with zero weights, preserving the previous policy's
predictions before training. Neither observation clamps the output. The network
must learn how to respond to them.

| Benchmark | Previous GP-X policy | Selected extension checkpoint 111 |
|---|---:|---:|
| Safe centered training corner/recovery sections | 42/44 | 44/44 |
| Safe centered layout-holdout sections | 7/8 | 8/8 |
| Full-lap regression on the selected model | -- | 39/39 starts |
| Successful paired randomized stress recoveries | 163/208 (78.4%) | 197/208 (94.7%) |
| Wall contacts in the paired stress benchmark | 0 | 0 |
| Independent action-validation weighted MSE | 0.087860 | 0.005979 |

Stress recovery improves by **34 successful drives**, or **16.3 percentage
points**. The selected model's eleven remaining stress failures are excessive
lateral acceleration, not wall contact. Action-validation error falls about
14.7-fold.

**The previous policy already avoided wall contact in these tests.** The
demonstrated gain is completing more physically valid corner/recovery drives,
not reducing an invented wall-crash count. Minimum near-wall clearances become
slightly smaller in this experiment; improved completion does not mean larger
wall buffers. The raw clearance plots expose that tradeoff.

### Why the delivered model is checkpoint 111, not round 120

More training is not always better. The first prototype regressed badly; its
raw results remain in `wall_artifacts_attempt_1/`. The revised training adds
actual expert recovery trajectories and an observable steering-budget feature.
Its later curriculum still exhibits regressions: raw round 120 finishes only
41/44 training sections and 32/33 training full laps.

The delivered model is therefore **checkpoint 111**, selected by:

1. Requiring all 44 training corner/recovery sections to succeed.
2. Requiring all 33 training full-lap regression starts to succeed.
3. Among eligible checkpoints, preferring lower independent action-validation
   MSE.

**No layout-holdout or stress outcomes select the checkpoint.**
The audit is saved in `checkpoint_selection.json`. All checkpoints and all
round-120 failures remain available in the player, metrics and raw data. Final
comparisons and ghosts use checkpoint 111, clearly labeled as selected.

The two layout holdouts never supply training or action-validation examples,
and final weights are frozen for evaluation. However, failed prototypes
previously evaluated those layouts: **these are data-disjoint layout tests,
not pristine unseen experiments**. This limitation is deliberately retained.

## Visuals and measurements for every track

Every cockpit contains 121 rounds times four challenges, all actual training
collections for its training track, an unsafe scripted illustration, the
complete attempt table and two per-track learning charts.

| Circuit / cockpit | Safe sections before/selected | Minimum edge margin before/selected (m) | Animation |
|---|---:|---:|---|
| [Apex Park](wall_artifacts/tracks/track_00.html) | 4/4 -> 4/4 | 1.028 / 0.990 | [GIF](wall_artifacts/visuals/track_00_corners.gif) |
| [Velocity Ring](wall_artifacts/tracks/track_01.html) | 3/4 -> 4/4 | 0.677 / 0.539 | [GIF](wall_artifacts/visuals/track_01_corners.gif) |
| [Chicane Coast](wall_artifacts/tracks/track_02.html) | 4/4 -> 4/4 | 0.936 / 0.897 | [GIF](wall_artifacts/visuals/track_02_corners.gif) |
| [Hairpin Harbor](wall_artifacts/tracks/track_03.html) | 4/4 -> 4/4 | 1.110 / 1.040 | [GIF](wall_artifacts/visuals/track_03_corners.gif) |
| [Serpentine Valley](wall_artifacts/tracks/track_04.html) | 4/4 -> 4/4 | 1.017 / 0.933 | [GIF](wall_artifacts/visuals/track_04_corners.gif) |
| [Triangle GP](wall_artifacts/tracks/track_05.html) | 3/4 -> 4/4 | 0.703 / 0.565 | [GIF](wall_artifacts/visuals/track_05_corners.gif) |
| [Square Sprint](wall_artifacts/tracks/track_06.html) | 4/4 -> 4/4 | 0.972 / 0.954 | [GIF](wall_artifacts/visuals/track_06_corners.gif) |
| [Double Apex](wall_artifacts/tracks/track_07.html) | 4/4 -> 4/4 | 0.646 / 0.462 | [GIF](wall_artifacts/visuals/track_07_corners.gif) |
| [Crescent Run](wall_artifacts/tracks/track_08.html) | 4/4 -> 4/4 | 0.971 / 0.872 | [GIF](wall_artifacts/visuals/track_08_corners.gif) |
| [Technical Orchard](wall_artifacts/tracks/track_09.html) | 4/4 -> 4/4 | 0.767 / 0.617 | [GIF](wall_artifacts/visuals/track_09_corners.gif) |
| [High-Speed Mesa](wall_artifacts/tracks/track_10.html) | 4/4 -> 4/4 | 0.690 / 0.516 | [GIF](wall_artifacts/visuals/track_10_corners.gif) |
| [Aurora layout holdout](wall_artifacts/tracks/track_11.html) | 3/4 -> 4/4 | 0.621 / 0.436 | [GIF](wall_artifacts/visuals/track_11_corners.gif) |
| [Delta layout holdout](wall_artifacts/tracks/track_12.html) | 4/4 -> 4/4 | 1.041 / 0.975 | [GIF](wall_artifacts/visuals/track_12_corners.gif) |

## How the ML driver can stay on track

This remains **DAgger-style imitation learning**, not reinforcement learning.
An expert labels steering and target speed for states visited by the learner.
It teaches slower recovery when the car points outward near an edge, physically
reasonable steering at the current speed, and a return toward the centerline.

The policy sees ideal lane/heading errors, speed, previous steering, four
geometric previews, three curvature observations, projected footprint margin
and a steering-budget feature. It does not receive a track ID, camera image or
hidden wall-avoidance planner. The cockpit's sensor display is an input snapshot,
not a causal explanation of individual neuron reasoning.

Initial data contains **9,900 synthetic recovery observations and 11,880 expert
rollout states**. DAgger adds **290,400** learner-visited observations, totaling
**312,180 recovery examples**. Another **5,632 centerline anchor examples**
retain previously learned lap driving; 2,200 independent observations validate
actions. Every minibatch balances the eleven training circuits and mixes
historical recovery, recent recovery and centerline anchors.

Legacy checkpoints do not contain Adam moments, so the extension continues
their weights with a fresh optimizer state. Randomized starts become harder
over the three training phases. Collection N uses model N-1 before that round's
updates; displayed collection footage includes only the first continuous
episode. Resets are never stitched into a fake drive.

## Artifacts, code and reproduction

- [Interactive dashboard](wall_artifacts/index.html)
- [PowerPoint](wall_artifacts/training_deck.pptx) and
  [browser deck](wall_artifacts/training_deck.html): six updated slides and notes
- `wall_sim.py`: rectangular footprints, wall clearance, sampled motion sweeps
- `wall_train.py`: 120-round extension, expert data, anchors and checkpoint selection
- `wall_select.py`: independently select/revalidate existing trained checkpoints
- `wall_render.py`, `wall_track.html`: close-up driving, per-track movies and input views
- `wall_verify.py`, `tests/wall-browser.mjs`: numerical and browser verification
- `wall_artifacts/data/`: prepared data, base model, anchors, geometry and schema
- `wall_artifacts/models/`: all 121 models and safety-selected `final.npz`
- `wall_artifacts/attempts/`: 6,292 fixed attempts, 39 full laps, 416 paired
  stress simulations and thirteen scripted illustrations
- `wall_artifacts/rollouts/`: 1,320 actual first-episode training recordings
- `wall_artifacts/verification.json`, `browser_verification.json`: validation reports
- `f1-wall-corner-learning-lab.zip`: portable newest-extension source, required
  base inputs, data, models, pages, charts, animations, screenshots and decks

The new ZIP does not duplicate previous experiments or the rejected prototype,
and excludes installed environments and browser executables. All shared helpers
are included. Earlier artifacts and ZIPs remain unchanged.

From the project directory:

```bash
.venv/bin/python -m unittest discover -s tests -v
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:walls
.venv/bin/python wall_render.py --output wall_artifacts --inventory-only
.venv/bin/python wall_verify.py --output wall_artifacts
```

For a fresh reproducible training run from the portable extension's saved
base model and anchor inputs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python wall_train.py --base wall_artifacts --output wall_reproduction
.venv/bin/python wall_render.py --output wall_reproduction
npm ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:walls -- wall_reproduction
.venv/bin/python wall_render.py --output wall_reproduction --inventory-only
.venv/bin/python wall_verify.py --output wall_reproduction
```

Verification checks input derivation, data separation, all checkpoint update
counts, real drivetrain integration, wall flags and clearance measurements,
exact re-simulation of sampled checkpoints, every final full lap and all paired
stress drives. Browser checks exercise every attempt/collection, both cameras,
sensor and ghost controls, actual moving-car canvas changes, sequence advance,
painted wall-contact illustrations and the complete deck.

**Limits:** ideal geometry, one seed, fictional powertrain and kinematic
handling, sampled collision detection, no impact dynamics, no sensor noise,
tires, weather, traffic or racing strategy. A learned policy can still fail;
this is not proof of real-world safety or a controller for a physical car.
