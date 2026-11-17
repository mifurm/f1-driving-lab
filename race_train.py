"""Train on GP-X circuits and record full telemetry for every track and attempt."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np

from driving import NeuralDriver, SCENARIOS, collect_initial_data, write_json
from multitrack import balanced_indices, curriculum, save_csv
from race_sim import (
    RACE_CONFIG, TRACE_COLUMNS, RaceEnvironment, complex_circuits,
    race_simulate, telemetry_row,
)


def collect_rollout(track, config, driver, rng, number):
    environment = RaceEnvironment(track, config)
    phase = curriculum(number)
    x, y = [], []
    failures = 0
    def reset():
        environment.state = environment.make_state(
            float(rng.uniform()), float(rng.uniform(-phase["lateral"], phase["lateral"])),
            float(rng.uniform(-phase["heading"], phase["heading"])),
            float(rng.uniform(phase["speed_min"], phase["speed_max"])),
        )
    reset()
    _, location = environment.observe()
    previous = location[0]
    progress = 0.0
    first_trace = [telemetry_row(environment, 0, location, progress)]
    recording = True
    first_reason = "collection_horizon"
    for step in range(config.rollout_steps):
        features, location = environment.observe()
        x.append(features)
        y.append(environment.teacher(location=location))
        location, failure, _ = environment.step(driver.predict(features))
        progress += (location[0] - previous + track.length / 2) % track.length - track.length / 2
        previous = location[0]
        if recording:
            first_trace.append(telemetry_row(environment, (step + 1) * config.dt, location, progress))
        if failure:
            failures += 1
            if recording:
                first_reason, recording = failure, False
            reset()
            _, location = environment.observe()
            previous, progress = location[0], 0.0
    return np.asarray(x), np.asarray(y), np.asarray(first_trace), {
        "collection_round": number, "model_round": number - 1,
        "source": "Actual data-collection rollout; first continuous episode only",
        "end_reason": first_reason, "duration_s": float(first_trace[-1][0]),
        "collection_failures": failures, "labeled_states": config.rollout_steps,
    }


def aggregate(rows):
    completed = [row for row in rows if row["completed"]]
    return {
        "completed_tracks": len(completed), "total_tracks": len(rows),
        "mean_progress_pct": float(np.mean([row["progress"] for row in rows]) * 100),
        "mean_completed_lateral_m": float(np.mean([row["mean_abs_lateral_m"] for row in completed])) if completed else None,
        "mean_completed_lap_s": float(np.mean([row["lap_time_s"] for row in completed])) if completed else None,
    }


def train(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Existing run: {output}; choose a new directory")
    start = time.perf_counter()
    config = RACE_CONFIG
    for folder in ("data/tracks", "models", "attempts", "rollouts", "visuals", "tracks"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    catalog, tracks = complex_circuits(config)
    training = [item for item in catalog if item["split"] == "train"]
    holdouts = [item for item in catalog if item["split"] == "holdout"]
    write_json(output / "data/circuits.json", catalog)
    for item in catalog:
        tracks[item["id"]].save(output / f"data/tracks/track_{item['id']:02d}.npz")
    write_json(output / "data/schema.json", {
        "features": RaceEnvironment.feature_names,
        "targets": ["steering_fraction", "2_target_speed_divided_by_max_speed_minus_1"],
        "trajectory_columns": TRACE_COLUMNS,
        "controls_timing": "Controls at row t were applied during the interval ending at t; initial pedals are zero",
        "gearbox": "Eight-speed automatic; shifting is deterministic, not a learned output",
        "source": {"0": "synthetic recovery", "1": "teacher demonstration", "2": "learner rollout with teacher labels"},
        "fixed_attempt": "Attempt N = unassisted canonical test using weights after N training rounds",
        "rollout": "Collection N uses model N-1; only its first continuous episode is replayed",
    })
    rng = np.random.default_rng(config.seed)
    features, targets, sources, track_ids = [], [], [], []
    print("Preparing data for eleven harder circuits and the simulated drivetrain.", flush=True)
    for item in training:
        x, y, source = collect_initial_data(tracks[item["id"]], config, rng, environment_type=RaceEnvironment)
        features.append(x)
        targets.append(y)
        sources.append(source)
        track_ids.append(np.full(len(x), item["id"], dtype=np.uint8))
    initial_x, initial_y = np.vstack(features), np.vstack(targets)
    initial_sources, initial_tracks = np.concatenate(sources), np.concatenate(track_ids)
    used = initial_count = len(initial_x)
    capacity = used + config.rounds * len(training) * config.rollout_steps
    x_data, y_data = np.empty((capacity, 16)), np.empty((capacity, 2))
    source_data, track_data = np.empty(capacity, dtype=np.uint8), np.empty(capacity, dtype=np.uint8)
    round_data = np.zeros(capacity, dtype=np.int16)
    x_data[:used], y_data[:used] = initial_x, initial_y
    source_data[:used], track_data[:used] = initial_sources, initial_tracks
    np.savez_compressed(output / "data/initial_training_data.npz",
                        features=initial_x, targets=initial_y, source=initial_sources,
                        track_id=initial_tracks, round=round_data[:used])
    pools = [np.flatnonzero(initial_tracks == item["id"]) for item in training]
    validation_rng = np.random.default_rng(config.seed + 1)
    validation_x, validation_y, validation_ids = [], [], []
    for item in training:
        environment = RaceEnvironment(tracks[item["id"]], config)
        for _ in range(200):
            state = environment.make_state(
                float(validation_rng.uniform()), float(validation_rng.uniform(-6, 6)),
                float(validation_rng.uniform(-0.45, 0.45)), float(validation_rng.uniform(6, 32)),
            )
            _, location = environment.observe(state)
            target = environment.teacher(state, location)
            state.steering = float(np.clip(
                target[0] * config.max_steer + validation_rng.normal(0, 0.03),
                -config.max_steer, config.max_steer,
            ))
            validation_x.append(environment.observe(state)[0])
            validation_y.append(target)
            validation_ids.append(item["id"])
    validation_x, validation_y = np.asarray(validation_x), np.asarray(validation_y)
    np.savez_compressed(output / "data/validation_data.npz",
                        features=validation_x, targets=validation_y,
                        track_id=np.asarray(validation_ids, dtype=np.uint8))
    driver = NeuralDriver(config.seed, input_size=16)
    history, attempts, rollout_details = [], [], []
    for number in range(config.rounds + 1):
        losses, failures = [], 0
        if number:
            recent, traces = [], {}
            for index, item in enumerate(training):
                x, y, trace, details = collect_rollout(tracks[item["id"]], config, driver, rng, number)
                indices = np.arange(used, used + len(x))
                x_data[indices], y_data[indices] = x, y
                source_data[indices], track_data[indices], round_data[indices] = 2, item["id"], number
                used += len(x)
                pools[index] = np.r_[pools[index], indices]
                recent.append(indices)
                traces[f"track_{item['id']:02d}"] = trace
                rollout_details.append({"track_id": item["id"], **details})
                failures += details["collection_failures"]
            np.savez_compressed(output / f"rollouts/round_{number:03d}.npz", **traces)
            rate = config.learning_rate * (1 - 0.75 * number / config.rounds)
            for _ in range(config.updates_per_round):
                indices = balanced_indices(rng, pools, recent)
                losses.append(driver.train_batch(x_data[indices], y_data[indices], rate))
        driver.save(output / f"models/round_{number:03d}.npz")
        rows, traces = [], {}
        for item in training:
            metric, trace = race_simulate(tracks[item["id"]], config, driver, SCENARIOS[0])
            row = {"round": number, "track_id": item["id"], "split": item["split"], **metric}
            rows.append(row)
            attempts.append(row)
            traces[f"track_{item['id']:02d}"] = trace
        np.savez_compressed(output / f"attempts/train_{number:03d}.npz", **traces)
        error = driver.predict(validation_x) - validation_y
        history.append({
            "round": number, "phase": curriculum(number)["name"] if number else "Untrained",
            "optimizer_updates": driver.updates, "samples": used, "collection_failures": failures,
            "validation_weighted_mse": float(np.mean(error ** 2 * [4, 1])),
            "training_weighted_mse": float(np.mean(losses)) if losses else None,
            **aggregate(rows),
        })
        save_csv(output / "metrics.csv", history)
        if number % 10 == 0 or number == 1:
            print(f"Round {number:3d}: {history[-1]['completed_tracks']:2d}/11 tracks complete "
                  f"progress={history[-1]['mean_progress_pct']:6.2f}% "
                  f"validation={history[-1]['validation_weighted_mse']:.6f}", flush=True)
    driver.save(output / "models/final.npz")
    model_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    write_json(output / "training_complete.json", {
        "rounds": config.rounds, "updates": driver.updates,
        "final_model_sha256": model_hash, "holdouts_not_yet_evaluated": True,
    })
    np.savez_compressed(output / "data/aggregated_training_data.npz",
                        features=x_data[:used], targets=y_data[:used], source=source_data[:used],
                        track_id=track_data[:used], round=round_data[:used])
    print("Weights frozen. Evaluating unseen attempts and final recovery starts.", flush=True)
    for number in range(config.rounds + 1):
        frozen = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        traces = {}
        for item in holdouts:
            metric, trace = race_simulate(tracks[item["id"]], config, frozen, SCENARIOS[0])
            attempts.append({"round": number, "track_id": item["id"], "split": item["split"], **metric})
            traces[f"track_{item['id']:02d}"] = trace
        np.savez_compressed(output / f"attempts/holdout_{number:03d}.npz", **traces)
    final_checks, extra_traces, teacher_traces = [], {}, {}
    for item in catalog:
        for scenario_index, scenario in enumerate(SCENARIOS):
            metric, trace = race_simulate(tracks[item["id"]], config, driver, scenario)
            final_checks.append({"track_id": item["id"], "split": item["split"], **metric})
            extra_traces[f"track_{item['id']:02d}_scenario_{scenario_index}"] = trace
        metric, trace = race_simulate(tracks[item["id"]], config, None, SCENARIOS[0])
        if not metric["completed"]:
            raise RuntimeError(f"Teacher failed harder track {item['name']}")
        teacher_traces[f"track_{item['id']:02d}"] = trace
    np.savez_compressed(output / "attempts/final_checks.npz", **extra_traces)
    np.savez_compressed(output / "attempts/teacher.npz", **teacher_traces)
    attempts.sort(key=lambda row: (row["round"], row["track_id"]))
    write_json(output / "attempt_metrics.json", attempts)
    write_json(output / "rollout_metrics.json", rollout_details)
    write_json(output / "final_checks.json", final_checks)
    write_json(output / "history.json", history)
    train_completed = sum(row["completed"] for row in final_checks if row["split"] == "train")
    heldout_completed = sum(row["completed"] for row in final_checks if row["split"] == "holdout")
    success = train_completed == 33 and history[-1]["mean_progress_pct"] > history[0]["mean_progress_pct"]
    write_json(output / "manifest.json", {
        "config": asdict(config), "architecture": [16, 48, 32, 2],
        "training_track_ids": [item["id"] for item in training],
        "holdout_track_ids": [item["id"] for item in holdouts],
        "initial_samples": initial_count, "final_samples": used,
        "validation_samples": len(validation_x), "actual_training_rounds": config.rounds,
        "actual_optimizer_updates": driver.updates, "per_track_samples_per_round": config.rollout_steps,
        "fixed_attempts_per_track": config.rounds + 1,
        "recorded_training_rollouts": len(rollout_details),
        "final_training_starts_completed": train_completed, "final_holdout_starts_completed": heldout_completed,
        "baseline": history[0], "final": history[-1], "final_model_sha256": model_hash,
        "holdouts_evaluated_after_freeze": True, "success_criteria_met": success,
        "elapsed_seconds": time.perf_counter() - start,
        "gearbox": {
            "gears": 8, "automatic": True, "upshift_rpm": 9000,
            "downshift_rpm": 4800, "torque_interruption_s": 0.2,
            "learned_shifts": False,
        },
        "limitations": [
            "Fictional engine and automatic gearbox, not an actual F1 powertrain",
            "Policy learns steering and target speed; actuators and shifting are deterministic",
            "Perfect geometric sensors and kinematic handling",
            "Single seed, two unseen variants and no real-world safety claim",
            "Actual training rollouts and fixed test attempts have different starting conditions",
        ],
    })
    print(f"Final: {train_completed}/33 training starts, {heldout_completed}/6 unseen starts. Saved {output}", flush=True)
    if not success:
        raise RuntimeError("Harder-circuit driver did not meet fixed-test completion criteria")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="race_artifacts")
    arguments = parser.parse_args()
    train(arguments.output)
