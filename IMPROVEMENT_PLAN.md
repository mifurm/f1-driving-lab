# Improving the driving learner

## Implemented in this extension

1. **Learn from eleven distinct circuits.** Keep Apex Park and add ten original
   layouts: Velocity Ring, Chicane Coast, Hairpin Harbor, Serpentine Valley,
   Triangle GP, Square Sprint, Double Apex, Crescent Run, Technical Orchard,
   and High-Speed Mesa. They vary in shape, curvature, straight length, and
   direction changes; they are not rotations of the original track.
2. **Make relevant geometry observable.** Preserve the existing thirteen inputs
   and add current curvature, maximum curvature in the next 66 m, and curvature
   at the speed-dependent lookahead point. The new network has sixteen inputs;
   these are geometric observations, not teacher-action labels.
3. **Balance the learning budget across circuits.** Every round collects the
   same number of learner states on every training circuit. Every minibatch
   gives each circuit an equal share. Half of each circuit's examples come from
   its recent rollouts, half from its entire history. This avoids domination by
   long/easy circuits while retaining past examples.
4. **Increase difficulty progressively.** Start near the centerline with small
   heading errors; then increase offset, heading error, and starting speed.
   Reset failed collection episodes explicitly and count failures. Evaluation
   never uses teacher intervention, resets, or an easier curriculum.
5. **Give each round enough actual learning.** Run 120 rounds with 64 Adam
   updates per round and a decaying learning rate. Save every model, not only
   the best one. Train from scratch to make the learning curve interpretable;
   separately benchmark the existing single-track model for transfer.
6. **Test unseen geometry without using it for training decisions.** Reserve
   Aurora and Delta as two additional test-only circuits. No examples from them
   enter the training/validation pools. Evaluate their baseline and all frozen
   checkpoints only after the 120-round training schedule has finished. Do not
   select a model or tune settings using their outcomes.
7. **Show the actual training process.** Animate every round's track progress,
   completion results, dataset growth, and learning curve. Separately animate
   driving with saved weights. The training-history movie advances *training
   rounds*; the driving comparison advances *simulated time*. Keep those clocks
   explicitly distinct.

## Fixed experiment contract

- Original single-track artifacts stay in `artifacts/` unchanged.
- New run lives in `multitrack_artifacts/`; circuit geometry, split, seeds,
  all training data, checkpoints, fixed-start results, and traces are saved.
- Three fixed starting scenarios on eleven training circuits: 33 monitored
  driving tests per round. The two unseen circuits add six post-training tests
  per frozen checkpoint.
- Independent synthetic action-validation states come only from training
  circuits, with a separate seed; they are never used for weight updates.
- Use the same vehicle dynamics and failure conditions as the original lab.
- Success requires all 33 final training-circuit starts to finish and better
  average progress than the untrained baseline. Report unseen-test outcomes
  honestly even if they fail; unseen success is not a tuning criterion.
- Compare technique over completed laps, not short failed baseline traces.
  Show raw regressions. Do not promise monotonic gains or optimal lap times.

## Further improvements, deliberately not implemented here

Run multiple seeds and confidence intervals; use more blind test circuits;
add sensor noise and vehicle-parameter randomization; experiment with a
smoothness-aware objective; improve the teacher's racing line and speed
planning; then consider reinforcement-learning fine-tuning. Those need separate
controlled experiments. This extension does not claim that each improvement
has been isolated by an ablation study.

These remain educational, original F1-inspired tracks and simplified vehicle
physics, not official F1 data or a controller suitable for a physical car.
