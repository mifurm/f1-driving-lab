"""Generate data, execute 120 training rounds, and save auditable evaluations."""

import argparse
import csv
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np

from driving import (
    CHECKPOINTS, SCENARIOS, Config, Environment, NeuralDriver, Track,
    collect_initial_data, collect_learner_data, config_dict, simulate, write_json,
)


def train(output):
    started = time.perf_counter()
    config = Config()
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Existing run at {output}; choose a new --output directory")
    for folder in ("data", "models", "trajectories", "visuals"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    track = Track(config)
    track.save(output / "data/track.npz")
    driver = NeuralDriver(config.seed)
    features, targets, sources = collect_initial_data(track, config, rng)
    rounds = np.zeros(len(features), dtype=np.int16)
    initial_count = len(features)
    np.savez_compressed(output / "data/initial_training_data.npz",
                        features=features, targets=targets, source=sources, round=rounds)
    environment = Environment(track, config)
    validation_rng = np.random.default_rng(config.seed + 1)
    validation_features, validation_targets = [], []
    for _ in range(2000):
        state = environment.make_state(
            float(validation_rng.uniform()), float(validation_rng.uniform(-6, 6)),
            float(validation_rng.uniform(-0.45, 0.45)), float(validation_rng.uniform(6, 32)),
        )
        observation, location = environment.observe(state)
        target = environment.teacher(state, location)
        state.steering = float(np.clip(
            target[0] * config.max_steer + validation_rng.normal(0, 0.03),
            -config.max_steer, config.max_steer,
        ))
        validation_features.append(environment.observe(state)[0])
        validation_targets.append(target)
    validation_features = np.asarray(validation_features)
    validation_targets = np.asarray(validation_targets)
    np.savez_compressed(output / "data/validation_data.npz",
                        features=validation_features, targets=validation_targets)
    write_json(output / "data/schema.json", {
        "features": Environment.feature_names,
        "targets": ["steering_divided_by_max_steer", "2_target_speed_divided_by_max_speed_minus_1"],
        "source": {"0": "synthetic_recovery", "1": "teacher_demonstration", "2": "learner_rollout_teacher_label"},
        "round": "0 for initial examples; 1..120 for aggregation round",
        "trajectory_columns": [
            "time_s", "x_m", "y_m", "heading_rad", "speed_m_s",
            "steering_rad", "lateral_error_m", "lap_progress_fraction",
        ],
        "validation": "Independent seeded synthetic states, never used for weight updates",
    })
    expert_metrics = []
    expert_traces = {}
    for index, scenario in enumerate(SCENARIOS):
        metrics, trace = simulate(track, config, None, scenario)
        if not metrics["completed"]:
            raise RuntimeError(f"Teacher failed test: {metrics}")
        expert_metrics.append(metrics)
        expert_traces[f"scenario_{index}"] = trace
    np.savez_compressed(output / "trajectories/teacher.npz", **expert_traces)
    write_json(output / "teacher_metrics.json", expert_metrics)
    history, details = [], []
    best_rank = (-1, -1, -float("inf"))
    best_round = 0
    csv_path = output / "metrics.csv"
    for round_number in range(config.rounds + 1):
        losses = []
        if round_number:
            added_features, added_targets = collect_learner_data(track, config, driver, rng)
            features = np.vstack([features, added_features])
            targets = np.vstack([targets, added_targets])
            sources = np.r_[sources, np.full(len(added_features), 2, dtype=np.uint8)]
            rounds = np.r_[rounds, np.full(len(added_features), round_number, dtype=np.int16)]
            learning_rate = config.learning_rate * (1 - 0.65 * round_number / config.rounds)
            for _ in range(config.updates_per_round):
                indices = rng.integers(0, len(features), size=config.batch_size)
                losses.append(driver.train_batch(features[indices], targets[indices], learning_rate))
        tests, traces = [], {}
        for index, scenario in enumerate(SCENARIOS):
            metrics, trace = simulate(track, config, driver, scenario)
            tests.append(metrics)
            details.append({"round": round_number, **metrics})
            traces[f"scenario_{index}"] = trace
        np.savez_compressed(output / f"trajectories/round_{round_number:03d}.npz", **traces)
        if round_number in CHECKPOINTS:
            driver.save(output / f"models/round_{round_number:03d}.npz")
        completed = sum(test["completed"] for test in tests)
        validation_residual = driver.predict(validation_features) - validation_targets
        completed_times = [test["lap_time_s"] for test in tests if test["completed"]]
        row = {
            "round": round_number, "optimizer_updates": driver.updates,
            "training_samples": len(features),
            "training_weighted_mse": float(np.mean(losses)) if losses else "",
            "validation_weighted_mse": float(np.mean(validation_residual ** 2 * [4, 1])),
            "completed_scenarios": completed,
            "mean_progress_pct": float(np.mean([test["progress"] for test in tests]) * 100),
            "mean_abs_lateral_m": float(np.mean([test["mean_abs_lateral_m"] for test in tests])),
            "mean_speed_kmh": float(np.mean([test["mean_speed_kmh"] for test in tests])),
            "mean_steering_variation_rad_s": float(np.mean([test["steering_variation_rad_s"] for test in tests])),
            "mean_completed_lap_time_s": float(np.mean(completed_times)) if completed_times else "",
        }
        history.append(row)
        with csv_path.open("a" if round_number else "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not round_number:
                writer.writeheader()
            writer.writerow(row)
        rank = (completed, row["mean_progress_pct"], -row["mean_abs_lateral_m"])
        if rank > best_rank:
            best_rank, best_round = rank, round_number
            driver.save(output / "models/best.npz")
        if round_number % 10 == 0 or round_number == 1:
            print(f"Round {round_number:3d}: updates={driver.updates:4d} "
                  f"laps={completed}/3 progress={row['mean_progress_pct']:6.2f}% "
                  f"lateral={row['mean_abs_lateral_m']:.3f}m "
                  f"validation={row['validation_weighted_mse']:.6f}", flush=True)
    driver.save(output / "models/final.npz")
    np.savez_compressed(output / "data/aggregated_training_data.npz",
                        features=features, targets=targets, source=sources, round=rounds)
    write_json(output / "scenario_metrics.json", details)
    first_complete = next((row["round"] for row in history if row["completed_scenarios"] == 3), None)
    final = history[-1]
    success = (final["completed_scenarios"] == 3
               and final["mean_progress_pct"] > history[0]["mean_progress_pct"])
    manifest = {
        "config": config_dict(config), "track_length_m": track.length,
        "track_name": "Original Apex Park (F1-inspired, not an official circuit)",
        "architecture": [13, 48, 32, 2],
        "parameter_count": sum(parameter.size for parameter in driver.parameters),
        "method": "DAgger-style imitation learning; teacher labels, no reward optimization",
        "initial_samples": initial_count, "final_samples": len(features),
        "validation_samples": len(validation_features),
        "actual_training_rounds": config.rounds, "actual_optimizer_updates": driver.updates,
        "evaluation_rounds_including_baseline": len(history),
        "fixed_test_scenarios": SCENARIOS, "checkpoints": CHECKPOINTS,
        "first_all_scenarios_complete_round": first_complete, "best_round_by_completion_then_lateral_error": best_round,
        "baseline": history[0], "final": final, "success_criteria_met": success,
        "elapsed_training_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "matplotlib", "Pillow", "python-pptx")},
        "limitations": [
            "One known circuit; no unseen-track validation",
            "Perfect track previews rather than camera or lidar sensors",
            "Kinematic vehicle, not F1 tire/aerodynamic physics",
            "Imitates a safe heuristic teacher, not an optimal racing driver",
            "Short crash trajectories can misleadingly reduce lateral/smoothness metrics",
            "Fixed repeated tests are a monitoring benchmark, not a blind final holdout",
            "One training seed; no statistical confidence claim",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(f"Saved {output}; success={success}; elapsed={manifest['elapsed_training_seconds']:.1f}s", flush=True)
    if not success:
        raise RuntimeError("Final driver did not meet completion/improvement criteria; inspect metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts", help="New directory for a training run")
    arguments = parser.parse_args()
    train(arguments.output)
