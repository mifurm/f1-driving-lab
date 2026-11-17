# Student ML Driving Lab

Open `student_artifacts/index.html` for all eight interactive modules. The existing wall/corner lab is unchanged.
Open `student_artifacts/teacher_guide.html` for printable worksheets and `student_artifacts/training_deck.pptx`
or `training_deck.html` for the new classroom deck.

## What was actually executed

- Four feature masks × three matched seeds × 256 Adam updates: **12 genuinely retrained compact models**.
- A clean/contaminated leakage pair also executes 256 updates each: **3,584 new reference optimizer updates total**.
- **5,632 training / 2,200 independent action-validation examples**.
- **768 ablation, 576 controller and 960 robustness section rollouts**; real raw results are included.
- Three original new test circuits, IDs13–15, were not evaluated by any clean ML policy until every clean model
  and its action-validation selection had been frozen. Legacy holdouts11–12 are explicitly not pristine.
- The frozen original wall-aware driver completes **36/36**
  paired-seed sections on the new circuits. Compact student-sized models are not promised to match the larger
  previously trained model.

## Hands-on functionality

Predict steering/braking before revealing actual outputs; record manual or explicitly marked expert drives;
import/export examples; fit a fresh model in the browser; inspect real gradients, activations and Adam deltas;
replay untrained/intermediate/trained weights; compare actual controller traces at equal time; retrain all
feature variants; freeze/reveal tests; deliberately demonstrate invalid test-label training on a separate clone;
perturb real observations, history and grip; run one-factor paired-seed worksheets and export raw evidence.

Starter rollout frames are grouped by their original 180-step episode. The separate starter validation frames
are independently sampled labeled states, not an invented validation drive. Recorded labels correspond to
pre-step observations. Inputs are masked consistently during fitting and inference.

## Interpretation and limitations

This is imitation learning, not reinforcement learning. The network outputs steering and target speed;
the automatic controller/gearbox determines actual brake, throttle and gears. Imported legacy weights restart
Adam moments and its counter; exported weight files do not falsely claim to preserve optimizer history.

“No heading” also removes the derived margin, but previews can encode orientation. “No margin” leaves lane
and heading from which margin can be inferred. Feature ablation is not proof of unique causal importance.
Action-validation checkpoint selection is not safety checkpoint selection.

Noise perturbs lane, heading and speed but not ideal geometric previews. Delay holds actual prior policy
observations; the ordinary speed controller still senses current speed. Grip scales braking and the handling
limit, not tire slip or weather. No real-F1 physics, real-car safety or impact response is claimed.

Successful centered exits, progress, wall contacts and handling-limit failures are reported separately.
Short failures are never reported as faster completed laps. Sample SD across three paired seeds is not a
confidence interval. Browser and NumPy use different reproducible random streams.

The test seal is a classroom protocol, not secrecy. Anonymous exposure flags persist across page reloads.
The saved example JSON downloads are browser-validation demonstrations, not actual student submissions.
Once test outcomes/labels have been exposed, subsequent
tuning is exploratory. A deliberately contaminated clone is separated from clean models and its scores
cannot support generalization claims. No student data or telemetry is transmitted.

## Reproduce

```sh
.venv/bin/python student_experiments.py --output student_artifacts_reproduction --base wall_artifacts
.venv/bin/python student_render.py --output student_artifacts_reproduction
STUDENT_OUTPUT=student_artifacts_reproduction npm run test:student
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:student -- student_artifacts_reproduction
.venv/bin/python student_verify.py --output student_artifacts_reproduction
```

The portable additive bundle contains the student lab, required source/data, reports and the complete previous
wall/corner lab. Installed environments/browser binaries and older large standalone bundles are excluded.
