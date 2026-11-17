# GP-X: harder circuits, visible cars and driving telemetry

**Open [the all-track dashboard](race_artifacts/index.html)** in your browser.
Everything plays offline; no server, account or internet connection is needed.
The earlier single-track and multi-track experiments remain unchanged.

## What was completed

Thirteen original F1-inspired layouts now have four added technical left-right
sectors each. Eleven circuits train a fresh neural driver; two are unseen tests.
Every new layout has more meaningful curvature-direction reversals than its
predecessor. These are geometric complexity measurements, not official corner
counts or reproductions of real F1 circuits.

The completed experiment ran **120 actual training rounds and 7,680 Adam weight
updates**. Its training dataset grew from **63,694 to 354,094 labeled examples**,
including 220 newly collected states per training track per round. Another
2,200 observations form an independent action-validation dataset.

| Canonical training benchmark | Untrained | Round 6: first all-track completion | Final round 120 |
|---|---:|---:|---:|
| Tracks completing a lap | 0/11 | 11/11 | 11/11 |
| Mean lap progress | 0.14% | 100% | 100% |
| Completed-lap mean lateral error | Not comparable | 0.249 m | 0.196 m |
| Mean completed lap time | No complete laps | 138.6 s | 133.2 s |
| Action-validation weighted MSE | 0.693839 | 0.043611 | 0.004691 |

From round 6 to round 120, the same eleven canonical full-lap tests become
**21.2% more accurate laterally** and **3.9% quicker** on average.
Validation error falls approximately **148-fold** from the untrained model.
Final recovery validation completes **33/33 training starts and 6/6 unseen
starts**. Results are from one seed and two unseen layouts, not a broad
generalization or safety guarantee.

**Not every attempt is better.** The complete raw histories retain failures,
regressions and speed/accuracy tradeoffs. Short failed drives are never treated
as comparable full laps. The final model is exactly round 120.

## Watch how each driver learns its track

The dashboard embeds a selectable cockpit and animated comparisons for **all
thirteen circuits**. Each standalone cockpit includes:

- **Every attempt:** automatically advance through saved models 0 to 120.
- **Single attempt:** select a model, play/pause or scrub its driving timeline.
- **Comparison:** untrained, selected and final cars at equal simulated time.
- **Actual training rollout:** inspect each collection round's first continuous
  driving episode, available only for training circuits.

Cars have visible bodies, wings, wheels and brake lights. The lights reflect
recorded brake input; HUD values show actual speed, learned target speed,
throttle, brake, integer gear, engine RPM, lap progress and elapsed driving time.
Every track has its own learning charts, final-lap telemetry charts, animation,
comparison image, cockpit screenshot and all-attempt CSV.

Failed cars freeze at their actual recorded endpoint. Their last speed reading
is the last simulated value, not a claim that they continue driving. Icons are
enlarged for visibility; final ghosts are visual comparisons, never simulation
participants. Browser playback is a replay of real recorded training, not a
new training run in JavaScript.

## Measured results and visuals for every circuit

The lane-error comparison below uses **each track's first completed attempt**,
which can differ by track. The common-round benchmark above uses round 6
instead. All listed final telemetry is from the canonical start at round 120.

| Circuit / cockpit | Curvature reversals: old/new | First complete attempt | Lane error: first/final (m) | Final lap (s) | Peak km/h | Braking (s) | Gear changes | Animation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [Apex Park GP-X](race_artifacts/tracks/track_00.html) | 8/24 | 2 | 0.540/0.219 | 156.5 | 93.5 | 34.2 | 22 | [GIF](race_artifacts/visuals/track_00_attempts.gif) |
| [Velocity Ring GP-X](race_artifacts/tracks/track_01.html) | 0/12 | 1 | 0.553/0.189 | 108.1 | 101.5 | 21.3 | 17 | [GIF](race_artifacts/visuals/track_01_attempts.gif) |
| [Chicane Coast GP-X](race_artifacts/tracks/track_02.html) | 8/24 | 1 | 0.586/0.198 | 146.3 | 104.8 | 24.1 | 16 | [GIF](race_artifacts/visuals/track_02_attempts.gif) |
| [Hairpin Harbor GP-X](race_artifacts/tracks/track_03.html) | 6/20 | 1 | 0.738/0.190 | 168.0 | 104.9 | 28.6 | 20 | [GIF](race_artifacts/visuals/track_03_attempts.gif) |
| [Serpentine Valley GP-X](race_artifacts/tracks/track_04.html) | 6/18 | 1 | 0.565/0.174 | 128.7 | 111.0 | 23.9 | 20 | [GIF](race_artifacts/visuals/track_04_attempts.gif) |
| [Triangle GP GP-X](race_artifacts/tracks/track_05.html) | 0/16 | 1 | 0.623/0.210 | 118.1 | 103.1 | 24.1 | 15 | [GIF](race_artifacts/visuals/track_05_attempts.gif) |
| [Square Sprint GP-X](race_artifacts/tracks/track_06.html) | 4/20 | 1 | 1.816/0.249 | 157.1 | 111.3 | 29.1 | 27 | [GIF](race_artifacts/visuals/track_06_attempts.gif) |
| [Double Apex GP-X](race_artifacts/tracks/track_07.html) | 2/16 | 1 | 0.567/0.189 | 124.4 | 108.4 | 18.4 | 18 | [GIF](race_artifacts/visuals/track_07_attempts.gif) |
| [Crescent Run GP-X](race_artifacts/tracks/track_08.html) | 0/16 | 1 | 0.424/0.173 | 102.0 | 111.3 | 17.0 | 20 | [GIF](race_artifacts/visuals/track_08_attempts.gif) |
| [Technical Orchard GP-X](race_artifacts/tracks/track_09.html) | 8/20 | 1 | 0.570/0.183 | 131.3 | 102.5 | 23.4 | 15 | [GIF](race_artifacts/visuals/track_09_attempts.gif) |
| [High-Speed Mesa GP-X](race_artifacts/tracks/track_10.html) | 0/12 | 1 | 0.553/0.182 | 124.7 | 113.2 | 22.5 | 21 | [GIF](race_artifacts/visuals/track_10_attempts.gif) |
| [Unseen Aurora GP-X](race_artifacts/tracks/track_11.html) | 4/18 | 1 | 0.532/0.175 | 122.0 | 98.0 | 22.4 | 14 | [GIF](race_artifacts/visuals/track_11_attempts.gif) |
| [Unseen Delta GP-X](race_artifacts/tracks/track_12.html) | 6/18 | 1 | 0.561/0.192 | 123.8 | 103.4 | 20.0 | 12 | [GIF](race_artifacts/visuals/track_12_attempts.gif) |

Final canonical laps use up to **gear 4 or 5**. Eight gears exist in the
simulation, but these modest speeds do not require all eight.

## What the neural model and drivetrain actually do

This is **DAgger-style imitation learning**, not reinforcement learning.
A pure-pursuit teacher labels steering and curvature-aware target speed.
A `16 -> 48 -> 32 -> 2` tanh network learns those two outputs through NumPy
backpropagation and Adam. Inputs include perfect geometric previews,
lane/heading error, speed, previous steering and three curvature observations.
Training balances every track and mixes recent and historical examples;
randomized recovery starts become harder over three curriculum phases.

The neural model does **not** directly predict pedals or gears. A deterministic
speed controller translates its target speed into throttle or braking.
An eight-speed automatic transmission upshifts above 9,000 RPM, downshifts
below 4,800 RPM, enforces cooldowns and interrupts drive torque for 0.2 seconds
during shifts. Gear ratio, fictional engine torque, braking and drag determine
actual acceleration. Thus brakes and gears affect the simulated drive rather
than being decorative speed-derived labels.

Attempt N is an unassisted fixed-start test of model N. Collection N instead
uses model N-1 at randomized starts, before that round's optimizer updates.
All collection states receive teacher labels, but only the first continuous
episode is shown; reset teleports are never stitched into a false full lap.
The two unseen circuits never supply training or validation observations.
Their checkpoint histories are evaluated after training completes and weights
are frozen.

## Files and reproducibility

| Artifact | Contents |
|---|---|
| [Experiment plan](RACE_PLAN.md) | Scope, attempt definitions, telemetry contract and validation |
| [Circuit atlas](race_artifacts/visuals/complex_track_atlas.png) | Thirteen harder layouts with prior-layout overlays |
| [PowerPoint](race_artifacts/training_deck.pptx) / [browser deck](race_artifacts/training_deck.html) | Eight updated slides with explanatory speaker notes |
| `race_sim.py`, `race_train.py`, `race_render.py` | Harder geometry, drivetrain, training and per-track visuals |
| `race_artifacts/data/` | Circuit geometry, 354,094 training examples, validation and schema |
| `race_artifacts/models/` | All 121 checkpoints and exact final weights |
| `race_artifacts/attempts/` | 1,573 full-resolution fixed attempts, final recovery checks and teacher traces |
| `race_artifacts/rollouts/` | 1,320 actual first-episode training trajectories |
| `race_artifacts/attempt_metrics.json`, `history.json` | Complete per-track and aggregate learning histories |
| `race_verify.py`, `tests/race-browser.mjs` | Independent numerical and browser verification |
| `race_artifacts/verification.json`, `browser_verification.json` | Verified outcomes and interaction results |
| `race_artifacts/artifact_inventory.json` | Artifact sizes and SHA-256 hashes |
| `f1-gpx-telemetry-lab.zip` | Portable newest-experiment source, data, models, pages, visuals and decks |

The portable GP-X ZIP deliberately excludes earlier experiments' large artifact
directories, installed environments and browser binaries. Earlier ZIPs remain
available separately. All required shared Python helpers are included.

From this project directory, using the existing environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python race_verify.py --output race_artifacts
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:race
```

To reproduce a fresh complete experiment without overwriting existing results:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python race_train.py --output race_reproduction
.venv/bin/python race_render.py --output race_reproduction
npm ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:race -- race_reproduction
.venv/bin/python race_render.py --output race_reproduction --inventory-only
.venv/bin/python race_verify.py --output race_reproduction
```

Saved NPZ trajectories retain all **17 telemetry channels at 0.1-second
intervals**. See `data/schema.json` for names, units and control timing.
Interactive pages thin trajectories but retain terminal states, gear and shift
changes, brake transitions and peak speed/brake samples. Pose and speed are
interpolated; gears, RPM, pedals and shift status remain discrete held values.
Numerical verification checks drivetrain integration, real deceleration,
continuous rollouts, exact final re-simulation and file hashes. Browser checks
exercise every attempt and training episode, painted canvases, moving cars,
automatic attempt advance, controls and actual red brake-light pixels.

**Limits:** fictional engine and automatic gearbox, simplified kinematic
handling, perfect sensors, one seed and no tire, weather, traffic, aerodynamics,
fuel, damage or strategic racing model. This educational lab is not authentic
F1 telemetry, a trained manual gearbox or a safe controller for a real car.
