# Teaching a fictional car to drive an F1-style circuit

## What must be true

1. **The world must be defined.** Use an original, closed F1-inspired circuit,
   a known track boundary, a fixed time step, and a deterministic vehicle model.
   No proprietary F1 telemetry or official circuit geometry is required.
2. **The car must observe enough to act.** Give it speed, lateral and heading
   errors, its previous steering command, and four forward track previews.
   The inputs are perfect simulated sensors, not camera images.
3. **There must be a learnable target.** A pure-pursuit teacher supplies steering
   and a curvature-aware target speed. A small neural network learns these two
   outputs. A conventional controller converts target speed into acceleration
   or braking.
4. **Examples must include mistakes.** Generate teacher demonstrations and
   recovery examples. After every training round, let the learner drive and
   label the states it actually visits with the teacher. This is dataset
   aggregation (DAgger-style imitation learning), not reinforcement learning.
5. **Improvement must be measured, not assumed.** Freeze three starting scenarios
   and test without teacher intervention after each of 120 training rounds.
   Record progress before a crash, completed laps, lateral error, steering
   variation, speed, and lap time. Keep an untrained baseline.
6. **The run must be reproducible.** Save the seed, configuration, initial and
   aggregated data, model weights, all 121 evaluation rows, representative
   checkpoints, and replay trajectories.

## Pipeline

Original track → teacher demonstrations + recovery examples → neural driver
→ learner rollouts → teacher labels → more training → fixed independent tests
→ charts, animated simulations, interactive replay, presentation.

## Training contract

- Execute **120 actual training rounds**, each with multiple minibatch weight
  updates; evaluate round 0 and every round thereafter.
- Keep the held-out starting scenarios fixed and exclude their exact initial
  states and evaluation trajectories from data collection.
- Generate data locally; no external datasets, services, or paid training.
- Save checkpoints at rounds 0, 1, 5, 10, 20, 40, 60, 80, 100, and 120.
- Treat monotonic improvement as a hypothesis, not a promise. Show regressions
  as well as gains. Require final completion of all fixed test scenarios and
  better average progress than the untrained baseline.
- Use identical road geometry for training and evaluation: this tests driving
  on **one known circuit**, not generalization to unseen tracks.

## Simulator assumptions and limits

The kinematic bicycle model includes wheelbase, steering limits, bounded
acceleration/braking, drag, steering-rate limits, and a lateral-acceleration
budget. Leaving the track or exceeding that budget ends a drive. It does not
model tires, suspension, gears, aerodynamic downforce, overtaking, weather,
vision, or real F1 safety requirements. Model output is a fictional simulated
driver only. An expert demonstration is not an optimal racing line.

## Deliverables

| Artifact | Purpose |
|---|---|
| Source, dependency lock, and tests | Reproduce and inspect the experiment |
| Track data, training data, model/checkpoints | Inspect what the model learned from |
| Metrics CSV and run manifest | Audit all 120 rounds and measured outcomes |
| Track/pipeline diagrams and progress charts | Explain the experiment visually |
| Animated simulation GIF | Compare untrained and trained driving |
| Offline interactive replay | Select saved rounds and watch trajectories |
| PowerPoint and HTML presentation | Explain setup, learning, results, limitations |
| README | Open artifacts and rerun locally |
