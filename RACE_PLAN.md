# Harder circuits, visible learning attempts, and driving telemetry

## Implemented scope

- Create **GP-X versions of all eleven training circuits and both unseen
  circuits**, preserving the previous layouts and results. Add four technical
  left-right sectors per layout and measure curvature-direction changes.
- Simulate an explicit fictional **eight-speed automatic gearbox**, engine RPM,
  gear-dependent available acceleration, short torque interruptions during
  shifts, throttle, braking, drag, steering, and lateral-acceleration limits.
- The neural driver learns steering and target speed. A deterministic speed
  controller handles throttle/brake; automatic shifting is **not** a learned
  action. Do not imply that all eight gears must be used at these modest speeds.
- Train for 120 rounds with balanced per-track data. Save fixed, unassisted
  full-track driving tests after every update round and actual data-collection
  rollouts separately.
- For **every track**, provide moving-car playback, per-attempt learning charts,
  early/current/final comparisons, speed and target-speed traces, brake/throttle
  bars, gear/RPM displays, and a standalone animation. Show regressions as well
  as gains.
- Reserve the two unseen layouts: evaluate their frozen checkpoints only after
  the schedule is complete. No test-only samples or outcomes tune the driver.

## What an “attempt” means

Attempt 0 is the untrained model's fixed test drive. Attempt N uses the saved
weights after N training rounds. Every fixed attempt on a given track starts
from the same position, heading, and speed, without teacher assistance or
resets. That makes the comparison fair.

Actual DAgger data-collection drives are explicitly labeled **training rollouts**.
They use randomized starts and may reset after a failure. The first continuous
episode of each rollout is saved for playback; it must not be presented as a
full lap or joined to a reset with a fictional moving-car segment.

## Telemetry contract

Each saved row contains pose, speed, applied steering, lane error, progress,
throttle, brake, integer gear, RPM, acceleration, requested target speed and
steering, lateral acceleration, and shift status. Controls on row t are those
applied during the interval ending at t; initial pedals are zero.

Use continuous interpolation only for the car's pose and speed. Pedal commands,
gear, and shift state are held between samples; keep samples at discrete
gear/brake/shift transitions. Brake lights follow recorded brake input, not
decorative animation. Stop failed attempts at their actual terminal state.

## Honest limits and validation

Teacher laps must be feasible on the new geometry with the same track boundary
and lateral-acceleration budget. Require final completion of all eleven
canonical training-track tests; additionally test recovery starts. Verify real
braking/deceleration and gear changes from saved trajectories, not only labels
on a screen. Re-drive final weights and validate every per-track replay.

The model uses perfect simulated geometry and a simplified kinematic vehicle.
The engine and gearbox are fictional; this is not real F1 telemetry, a faithful
F1 powertrain, or a controller for a physical car.
