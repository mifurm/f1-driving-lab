"""Balanced curriculum imitation learning on eleven circuits, with blind tests."""

import argparse
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np

from circuits import CIRCUITS, HOLDOUT_CIRCUITS, TRAINING_CIRCUITS
from driving import (
    Config, Environment, NeuralDriver, SCENARIOS, collect_initial_data,
    sample_index, simulate, write_json,
)


MULTI_CONFIG = Config(
    seed=2027, rounds=120, max_steps=2200, initial_recovery_samples=1800,
    rollout_steps=180, updates_per_round=64, batch_size=484,
)
CURVATURE_SCALE = 0.05


class MultiTrackEnvironment(Environment):
    feature_names = Environment.feature_names + [
        "current_signed_curvature_divided_by_0.05",
        "preview_max_abs_curvature_divided_by_0.05",
        "lookahead_signed_curvature_divided_by_0.05",
    ]

    def observe(self, state=None):
        state = self.state if state is None else state
        features, location = super().observe(state)
        distance = location[0]
        preview = distance + np.arange(0, 66, 3)
        indices = sample_index((preview % self.track.length) / self.track.ds, self.track.n)
        lookahead = distance + 8.0 + 0.45 * state.speed
        lookahead_index = int(sample_index((lookahead % self.track.length) / self.track.ds, self.track.n))
        geometry = np.array([
            self.track.curvature[indices[0]],
            np.max(np.abs(self.track.curvature[indices])),
            self.track.curvature[lookahead_index],
        ]) / CURVATURE_SCALE
        return np.r_[features, geometry], location


def curriculum(round_number):
    if round_number <= 40:
        return {"name": "Foundation", "lateral": 1.0, "heading": 0.04, "speed_min": 8, "speed_max": 14}
    if round_number <= 80:
        return {"name": "Recovery", "lateral": 2.5, "heading": 0.10, "speed_min": 8, "speed_max": 18}
    return {"name": "Challenge", "lateral": 4.0, "heading": 0.16, "speed_min": 8, "speed_max": 22}


def collect_curriculum(track, config, driver, rng, round_number):
    environment = MultiTrackEnvironment(track, config)
    phase = curriculum(round_number)
    features, targets = [], []
    failures = 0
    def reset():
        environment.state = environment.make_state(
            float(rng.uniform()), float(rng.uniform(-phase["lateral"], phase["lateral"])),
            float(rng.uniform(-phase["heading"], phase["heading"])),
            float(rng.uniform(phase["speed_min"], phase["speed_max"])),
        )
    reset()
    for _ in range(config.rollout_steps):
        observation, location = environment.observe()
        features.append(observation)
        targets.append(environment.teacher(location=location))
        _, reason, _ = environment.step(driver.predict(observation))
        if reason:
            failures += 1
            reset()
    return np.asarray(features), np.asarray(targets), failures


def balanced_indices(rng, history_by_track, recent_by_track, per_track=44):
    if per_track % 2:
        raise ValueError("Per-track minibatch count must be even")
    if len(history_by_track) != len(recent_by_track):
        raise ValueError("History and recent pools must have matching track counts")
    half = per_track // 2
    return np.concatenate([
        np.r_[rng.choice(history, half), rng.choice(recent, half)]
        for history, recent in zip(history_by_track, recent_by_track)
    ])


def summarize(tests, prefix):
    completed = [test for test in tests if test["completed"]]
    return {
        f"{prefix}_completed": len(completed), f"{prefix}_total": len(tests),
        f"{prefix}_progress_pct": float(np.mean([test["progress"] for test in tests]) * 100),
        f"{prefix}_lateral_completed_m": float(np.mean([test["mean_abs_lateral_m"] for test in completed])) if completed else None,
        f"{prefix}_lap_time_completed_s": float(np.mean([test["lap_time_s"] for test in completed])) if completed else None,
        f"{prefix}_steering_completed_rad_s": float(np.mean([test["steering_variation_rad_s"] for test in completed])) if completed else None,
    }


def evaluate(circuits, tracks, config, driver, round_number, environment_type=MultiTrackEnvironment):
    details, traces = [], {}
    for circuit in circuits:
        for index, scenario in enumerate(SCENARIOS):
            metrics, trace = simulate(
                tracks[circuit.id], config, driver, scenario, environment_type=environment_type,
            )
            details.append({
                "round": round_number, "track_id": circuit.id, "track": circuit.name,
                "split": circuit.split, **metrics,
            })
            traces[f"track_{circuit.id:02d}_scenario_{index}"] = trace
    return details, traces


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(output, original_model):
    start = time.perf_counter()
    output, original_model = Path(output), Path(original_model)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output directory")
    if not original_model.is_file():
        raise FileNotFoundError(f"Original model needed for transfer comparison: {original_model}")
    original_hash = hashlib.sha256(original_model.read_bytes()).hexdigest()
    config = MULTI_CONFIG
    for folder in ("data/tracks", "models", "trajectories", "visuals"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    tracks = {circuit.id: circuit.build(config) for circuit in CIRCUITS}
    catalog = []
    for circuit in CIRCUITS:
        track = tracks[circuit.id]
        track.save(output / f"data/tracks/track_{circuit.id:02d}.npz")
        catalog.append({
            "id": circuit.id, "name": circuit.name, "style": circuit.style,
            "split": circuit.split, "length_m": track.length,
            "knots": circuit.knots.tolist() if circuit.knots is not None else None,
        })
    write_json(output / "data/circuit_catalog.json", catalog)
    rng = np.random.default_rng(config.seed)
    feature_parts, target_parts, source_parts, track_parts = [], [], [], []
    print("Preparing demonstrations and recovery states on all 11 training circuits.", flush=True)
    for circuit in TRAINING_CIRCUITS:
        x, y, source = collect_initial_data(
            tracks[circuit.id], config, rng, environment_type=MultiTrackEnvironment,
        )
        feature_parts.append(x)
        target_parts.append(y)
        source_parts.append(source)
        track_parts.append(np.full(len(x), circuit.id, dtype=np.uint8))
    initial_x, initial_y = np.vstack(feature_parts), np.vstack(target_parts)
    initial_sources, initial_tracks = np.concatenate(source_parts), np.concatenate(track_parts)
    initial_count = len(initial_x)
    capacity = initial_count + config.rounds * config.rollout_steps * len(TRAINING_CIRCUITS)
    features = np.empty((capacity, 16))
    targets = np.empty((capacity, 2))
    sources = np.empty(capacity, dtype=np.uint8)
    track_ids = np.empty(capacity, dtype=np.uint8)
    data_rounds = np.zeros(capacity, dtype=np.int16)
    features[:initial_count], targets[:initial_count] = initial_x, initial_y
    sources[:initial_count], track_ids[:initial_count] = initial_sources, initial_tracks
    np.savez_compressed(output / "data/initial_training_data.npz",
                        features=initial_x, targets=initial_y, source=initial_sources,
                        track_id=initial_tracks, round=data_rounds[:initial_count])
    history_by_track = [np.flatnonzero(initial_tracks == circuit.id) for circuit in TRAINING_CIRCUITS]
    used = initial_count
    validation_rng = np.random.default_rng(config.seed + 1)
    validation_x, validation_y, validation_track_ids = [], [], []
    for circuit in TRAINING_CIRCUITS:
        environment = MultiTrackEnvironment(tracks[circuit.id], config)
        for _ in range(200):
            state = environment.make_state(
                float(validation_rng.uniform()), float(validation_rng.uniform(-6, 6)),
                float(validation_rng.uniform(-0.45, 0.45)), float(validation_rng.uniform(6, 32)),
            )
            _, location = environment.observe(state)
            action = environment.teacher(state, location)
            state.steering = float(np.clip(
                action[0] * config.max_steer + validation_rng.normal(0, 0.03),
                -config.max_steer, config.max_steer,
            ))
            validation_x.append(environment.observe(state)[0])
            validation_y.append(action)
            validation_track_ids.append(circuit.id)
    validation_x, validation_y = np.asarray(validation_x), np.asarray(validation_y)
    np.savez_compressed(output / "data/validation_data.npz",
                        features=validation_x, targets=validation_y,
                        track_id=np.asarray(validation_track_ids, dtype=np.uint8))
    write_json(output / "data/schema.json", {
        "features": MultiTrackEnvironment.feature_names,
        "targets": ["steering_divided_by_max_steer", "2_target_speed_divided_by_max_speed_minus_1"],
        "source": {"0": "synthetic_recovery", "1": "teacher_demonstration", "2": "learner_rollout_teacher_label"},
        "track_id": "IDs from circuit_catalog.json; training and validation contain only IDs 0..10",
        "round": "0 = initial; 1..120 = actual aggregation round",
        "trajectory_columns": [
            "time_s", "x_m", "y_m", "heading_rad", "speed_m_s",
            "steering_rad", "lateral_error_m", "lap_progress_fraction",
        ],
    })
    driver = NeuralDriver(config.seed, input_size=16)
    history, all_details = [], []
    best_rank = (-1, -1, -float("inf"))
    best_round = 0
    for number in range(config.rounds + 1):
        losses, failures = [], 0
        if number:
            recent_by_track = []
            for index, circuit in enumerate(TRAINING_CIRCUITS):
                x, y, failed = collect_curriculum(tracks[circuit.id], config, driver, rng, number)
                indices = np.arange(used, used + len(x))
                features[indices], targets[indices] = x, y
                sources[indices], track_ids[indices], data_rounds[indices] = 2, circuit.id, number
                used += len(x)
                history_by_track[index] = np.r_[history_by_track[index], indices]
                recent_by_track.append(indices)
                failures += failed
            learning_rate = config.learning_rate * (1 - 0.75 * number / config.rounds)
            for _ in range(config.updates_per_round):
                indices = balanced_indices(rng, history_by_track, recent_by_track)
                losses.append(driver.train_batch(features[indices], targets[indices], learning_rate))
        details, traces = evaluate(TRAINING_CIRCUITS, tracks, config, driver, number)
        all_details.extend(details)
        np.savez_compressed(output / f"trajectories/train_round_{number:03d}.npz", **traces)
        driver.save(output / f"models/round_{number:03d}.npz")
        validation_error = driver.predict(validation_x) - validation_y
        row = {
            "round": number, "phase": curriculum(number)["name"] if number else "Untrained",
            "optimizer_updates": driver.updates, "samples": used,
            "collection_failures": failures,
            "training_weighted_mse": float(np.mean(losses)) if losses else None,
            "validation_weighted_mse": float(np.mean(validation_error ** 2 * [4, 1])),
            **summarize(details, "train"),
        }
        history.append(row)
        save_csv(output / "metrics.csv", history)
        rank = (row["train_completed"], row["train_progress_pct"],
                -(row["train_lateral_completed_m"] if row["train_lateral_completed_m"] is not None else 999))
        if rank > best_rank:
            best_rank, best_round = rank, number
            driver.save(output / "models/best.npz")
        if number % 10 == 0 or number == 1:
            print(f"Round {number:3d}: {row['phase']:10s} updates={driver.updates:4d} "
                  f"completed={row['train_completed']:2d}/33 "
                  f"progress={row['train_progress_pct']:6.2f}% "
                  f"validation={row['validation_weighted_mse']:.6f}", flush=True)
    driver.save(output / "models/final.npz")
    np.savez_compressed(output / "data/aggregated_training_data.npz",
                        features=features[:used], targets=targets[:used], source=sources[:used],
                        track_id=track_ids[:used], round=data_rounds[:used])
    frozen_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    training_elapsed = time.perf_counter() - start
    write_json(output / "training_complete.json", {
        "rounds": config.rounds, "updates": driver.updates,
        "elapsed_seconds": training_elapsed, "final_model_sha256": frozen_hash,
        "holdout_evaluation_started": False,
    })
    print("Weights frozen. Evaluating unseen circuits on saved checkpoints; no further training.", flush=True)
    for number in range(config.rounds + 1):
        frozen = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        details, traces = evaluate(HOLDOUT_CIRCUITS, tracks, config, frozen, number)
        all_details.extend(details)
        history[number].update(summarize(details, "holdout"))
        np.savez_compressed(output / f"trajectories/holdout_round_{number:03d}.npz", **traces)
    require_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    if require_hash != frozen_hash:
        raise RuntimeError("Frozen model changed during holdout evaluation")
    transfer_driver = NeuralDriver.load(original_model)
    transfer_details, _ = evaluate(
        CIRCUITS, tracks, config, transfer_driver, -1, environment_type=Environment,
    )
    transfer = {
        "label": "Existing single-track 13-input model; not the new experiment's round 0",
        "original_model_sha256": original_hash,
        **summarize([d for d in transfer_details if d["split"] == "train"], "train"),
        **summarize([d for d in transfer_details if d["split"] == "holdout"], "holdout"),
        "details": transfer_details,
    }
    write_json(output / "transfer_baseline.json", transfer)
    save_csv(output / "metrics.csv", history)
    write_json(output / "history.json", history)
    scenario_order = {scenario["name"]: index for index, scenario in enumerate(SCENARIOS)}
    all_details.sort(key=lambda row: (row["round"], row["track_id"], scenario_order[row["scenario"]]))
    write_json(output / "scenario_metrics.json", all_details)
    first_complete = next((row["round"] for row in history if row["train_completed"] == 33), None)
    final = history[-1]
    success = final["train_completed"] == 33 and final["train_progress_pct"] > history[0]["train_progress_pct"]
    manifest = {
        "config": asdict(config), "architecture": [16, 48, 32, 2],
        "parameter_count": sum(parameter.size for parameter in driver.parameters),
        "method": "Balanced-track curriculum DAgger-style imitation learning",
        "training_track_ids": [c.id for c in TRAINING_CIRCUITS],
        "holdout_track_ids": [c.id for c in HOLDOUT_CIRCUITS],
        "new_training_tracks": 10, "fixed_scenarios": SCENARIOS,
        "actual_training_rounds": config.rounds, "actual_optimizer_updates": driver.updates,
        "initial_samples": initial_count, "final_samples": used,
        "validation_samples": len(validation_x), "per_track_samples_per_round": config.rollout_steps,
        "per_track_minibatch_samples": 44, "recent_fraction": 0.5,
        "first_all_training_starts_complete_round": first_complete,
        "best_round_by_training_completion_then_accuracy": best_round,
        "baseline": history[0], "final": final, "transfer_baseline": {k: v for k, v in transfer.items() if k != "details"},
        "training_seconds": training_elapsed, "total_seconds": time.perf_counter() - start,
        "heldout_evaluation_after_training": True, "final_model_sha256": frozen_hash,
        "success_criteria_met": success,
        "curriculum": [{"round_start": a, "round_end": b, **curriculum(a)}
                       for a, b in ((1, 40), (41, 80), (81, 120))],
        "limitations": [
            "Original F1-inspired circuits, not official F1 layouts",
            "Perfect geometric observations and simplified kinematic physics",
            "One seed and only two unseen tracks; no statistical confidence claim",
            "Multiple changes together; no isolated ablation or guarantee of superiority over old model",
            "Smoothness and lap time are measured, not directly optimized",
            "Heldout checkpoint history is post-training analysis, not feedback for training",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(f"Final training tests {final['train_completed']}/33; "
          f"unseen tests {final['holdout_completed']}/6; saved {output}", flush=True)
    if not success:
        raise RuntimeError("Final training-circuit benchmark did not meet success criteria")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multitrack_artifacts")
    parser.add_argument("--original-model", default="artifacts/models/final.npz")
    arguments = parser.parse_args()
    train(arguments.output, arguments.original_model)
