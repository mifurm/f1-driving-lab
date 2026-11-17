# Student ML Driving Lab (version 2)

Open `student_v2_artifacts/index.html` for the eleven interactive modules, `teacher_guide.html` for objectives, rubric, answer key and printable worksheets, and
`class_dashboard.html` to aggregate students' results cards. The first eight-module lab is preserved unchanged in `student_artifacts/`.

## What changed and why

The first lab's validation set held only deep wall-recovery states (median offset 5.08 m) while its training set was 89.5% near-centreline, so a 7.6x "generalization gap" was distribution shift, not overfitting.
Version 2 draws training and validation as **different expert episodes from the same sampler** and adds a separately named **recovery-stress set** (evaluation only).

| Set | Examples | Episodes | Within 1 m of the centreline | Median offset |
|---|---:|---:|---:|---:|
| Training | 5,539 | 176 | 93.6% | 0.13 m |
| Validation | 2,050 | 66 | 94.7% | 0.11 m |
| Stress (evaluation only) | 690 | 44 | 51.9% | 0.86 m |

## What was executed

- **40 retrained models** (4 input variants × 10 seeds × 256 Adam updates), 360 saved checkpoints; diverged seeds: 0.
- **2,560 ablation, 1,920 controller, 3,200 robustness and 1,440 monitor rollouts** (NumPy simulator, 14 workers).
- Final mean losses (full inputs, ten seeds): training 0.0136, validation 0.0155, stress 0.0775.
- Legacy-geometry circuits [0, 6] (Apex Park, Square Sprint: inner-wall loops at bends tighter than half the road width) are excluded from the monitor and headline pools; they stay in the training data for reproducibility.

## Input-removal retraining (ten seeds, valid training tracks)

Mean over seeds with a 95% percentile-bootstrap interval over seeds; per-seed range in the last column. Nominal and recovery sections are never pooled.

| Variant | Nominal sections | Recovery sections | Sealed circuits (12 fixed sections) | Recovery per-seed range |
|---|---|---|---|---|
| All inputs | 93% [88%, 97%] | 50% [36%, 64%] | 6.7/12 (6-9) | 6% to 78% |
| No previews/curvature | 0% [0%, 0%] | 0% [0%, 0%] | 0.1/12 (0-1) | 0% to 0% |
| No direct heading | 90% [84%, 96%] | 46% [33%, 59%] | 6.4/12 (6-9) | 11% to 78% |
| No margin | 92% [87%, 97%] | 43% [32%, 54%] | 6.1/12 (6-7) | 6% to 78% |

Only removing the road previews and curvature is clearly distinguishable from the full model; heading and margin removal sit inside the seed-to-seed spread. On the three sealed circuits almost every network completes the nominal sections and fails the recovery sections, a coverage problem addressed in module 7.

## Loss versus driving

| Update | Mean validation loss | Mean nominal sections (of 8) | Mean recovery sections (of 8) | Recovery per-seed values |
|---:|---:|---:|---:|---|
| 0 | 0.2910 | 0.2 | 0.1 | 0 0 0 0 0 0 1 0 0 0 |
| 32 | 0.0423 | 1.8 | 0.4 | 0 2 0 0 1 0 0 1 0 0 |
| 64 | 0.0238 | 4.2 | 2.2 | 0 6 4 2 4 1 3 1 0 1 |
| 96 | 0.0190 | 4.3 | 2.4 | 1 7 0 6 4 2 2 0 2 0 |
| 128 | 0.0177 | 4.8 | 2.9 | 4 6 0 7 4 1 2 0 4 1 |
| 160 | 0.0170 | 5.4 | 3.4 | 7 6 0 6 5 1 2 0 5 2 |
| 192 | 0.0163 | 5.6 | 3.5 | 6 6 0 6 5 2 2 1 5 2 |
| 224 | 0.0158 | 6.3 | 3.7 | 7 6 0 6 5 2 3 0 5 3 |
| 256 | 0.0155 | 7.1 | 4.2 | 6 6 1 7 5 2 4 1 6 4 |

## Controllers and distribution shift (section-level, ten paired seeds)

| Condition | Nominal sections | Recovery sections | Handling-limit failures |
|---|---|---|---:|
| rules | 72% [50%, 89%] | 0% [0%, 0%] | n/a |
| expert | 100% [100%, 100%] | 100% [100%, 100%] | n/a |
| pretrained | 100% [100%, 100%] | 99% [98%, 100%] | n/a |
| pretrained, Clean | 100% [100%, 100%] | 99% [98%, 100%] | 1 |
| pretrained, Noisy observations | 100% [100%, 100%] | 80% [66%, 91%] | 36 |
| pretrained, 0.5 s observation delay | 94% [83%, 100%] | 58% [37%, 79%] | 1 |
| pretrained, 65% grip envelope | 72% [50%, 89%] | 0% [0%, 0%] | 230 |
| pretrained, Combined shift | 81% [64%, 95%] | 15% [2%, 30%] | 188 |

## Data coverage and DAgger (headless run of the browser engine)

Reference run: 4 recipes × 10 training seeds (601–610), 100 labelled states per track per round,
4 rounds after the first, 120 Adam updates per round. Seed = unit of replication; intervals are 95% percentile bootstrap over seeds.

| Recipe | Nominal sections | Recovery sections | Legacy holdout | Reference occupancy | Visited occupancy | Failed collections | Diverged |
|---|---|---|---|---|---|---|---|
| Narrow expert demonstrations | 87% [70%, 99%] | 58% [40%, 74%] | 70% [54%, 84%] | 81% | 45% | 0 | 0 |
| Broad expert demonstrations | 92% [75%, 100%] | 48% [34%, 61%] | 66% [53%, 78%] | 82% | 59% | 10 | 0 |
| Staged expert control | 92% [78%, 100%] | 54% [40%, 66%] | 71% [56%, 83%] | 81% | 56% | 3 | 0 |
| DAgger | 99% [97%, 100%] | 64% [48%, 81%] | 74% [63%, 85%] | 78% | 71% | 70 | 0 |
| Expert (ceiling) | 18/18 | 18/18 | 8/8 | 100% | n/a | n/a | n/a |

| Paired by seed | Overall difference [95% bootstrap] | Recovery-only difference | Seeds better/worse/equal | Sign test p |
|---|---|---|---|---|
| DAgger − Staged expert control | +8 pts [−5 pts, +26 pts] | +10 pts [−8 pts, +31 pts] | 5/5/0 | 1.000 |
| DAgger − Narrow expert demonstrations | +9 pts [−8 pts, +32 pts] | +7 pts [−19 pts, +35 pts] | 4/6/0 | 0.754 |
| Broad expert demonstrations − Narrow expert demonstrations | −2 pts [−13 pts, +11 pts] | −9 pts [−26 pts, +11 pts] | 3/6/1 | 0.508 |
| Staged expert control − Narrow expert demonstrations | +1 pts [−7 pts, +14 pts] | −3 pts [−17 pts, +16 pts] | 2/7/1 | 0.180 |

Reading of this run (generated from the numbers above):

- DAgger against the staged expert-only control (the one comparison that isolates *who chooses the visited states*): overall +8 pts [−5 pts, +26 pts], recovery-only +10 pts [−8 pts, +31 pts]; seeds better/worse/equal 5/5/0, sign-test p = 1.00. The interval includes zero, so with 10 seeds a closed-loop benefit is **not established**.
- Mechanism, measured separately from closed-loop success: DAgger's data hold 71% of the projected cells its own learners visit, against 45% for narrow demonstrations. Occupancy is a diagnostic, not a guarantee.
- DAgger's learners crashed during 70 collection episodes (narrow demonstrations: 0); those are exactly the states where an expert's correction is needed, which is why failed collections are reported rather than hidden.
- Comparisons with narrow or broad demonstrations change several things at once (schedule, start distribution and who drives); only the comparison with the staged control changes one. Fewer than ten seeds is exploratory, and nominal and recovery sections are never pooled.

## Honest boundaries

- The expert is a heuristic: 5 of 242 typical episodes and 1 of 44 stress drives ended early because the expert itself tripped a limit. A learner cannot exceed its labels.
- The sealed circuits are a classroom protocol, not secrecy: the page embeds them. Blind claims need a teacher-controlled fresh pack (`tools/make_fresh_pack.py`).
- Fixed sections give exact counts; trials inside a section are clustered, so intervals use sections (or seeds) as the unit of replication. Development pools (monitor and the 36 fixed sections) are exploratory.
- Grip is a hard threshold on lateral acceleration and braking, not a tyre-slip model; noise perturbs lane, heading and speed but not the ideal road previews; one simulator, one expert, original circuits, no real-car claim.
- Browser and NumPy use different random streams (so noisy runs differ), but every deterministic quantity agrees: the browser engine replays all 2,560 saved NumPy rollouts of the retrained networks with identical outcomes and metrics within 1e-9 (`tests/student-engine.cjs`), and 24 fixture trajectories match to 1e-6 m (`tests/engine-fixtures.cjs`).
- Curvature inputs are read from the centreline sample that *starts* at a boundary (`driving.SAMPLE_SNAP`), so one ulp of rounding noise cannot flip them between engines or platforms. Against the preserved first lab this changes continuous metrics in 46 of 768 recorded rollouts (never an outcome); `git tag lab-v1` marks the code that produced those files exactly.

## Reproduce

```sh
.venv/bin/python student_experiments.py --output student_v2_artifacts --base wall_artifacts
node tools/coverage_reference.cjs --artifact student_v2_artifacts
.venv/bin/python student_render.py --output student_v2_artifacts
npm run test:student
npm run test:engine && .venv/bin/python -m unittest discover -s tests
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:student
.venv/bin/python student_verify.py --output student_v2_artifacts
```
