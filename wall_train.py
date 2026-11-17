"""Continue GP-X weights for 120 rounds of boundary-aware corner recovery."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from driving import NeuralDriver, SCENARIOS, Track, write_json
from multitrack import save_csv
from wall_sim import (
    WALL_COLUMNS, WALL_CONFIG, WallEnvironment, add_margin_feature,
    corner_scenarios, wall_row, wall_simulate,
)


def upgraded_driver(path):
    original = NeuralDriver.load(path)
    driver = NeuralDriver(WALL_CONFIG.seed, input_size=18)
    driver.weights[0][:16] = original.weights[0]
    driver.weights[0][16:] = 0
    driver.biases[0][...] = original.biases[0]
    for index in (1, 2):
        driver.weights[index][...] = original.weights[index]
        driver.biases[index][...] = original.biases[index]
    return driver


def random_state(environment, rng, number=120):
    edge = 6.0 if number <= 40 else (6.4 if number <= 80 else 6.6)
    heading = .15 if number <= 40 else (.22 if number <= 80 else .28)
    maximum_speed = 12 if number <= 40 else (15 if number <= 80 else 18)
    for _ in range(100):
        side = rng.choice([-1, 1])
        state = environment.make_state(
            float(rng.uniform()), side * float(rng.uniform(3.5, edge)),
            side * float(rng.uniform(-.10, heading)), float(rng.uniform(6, maximum_speed)),
        )
        if environment.walls.clearance(state) > .6:
            return state
    raise RuntimeError("Cannot generate a valid inside-wall recovery start")


def collect(track, config, driver, rng, number):
    environment = WallEnvironment(track, config)
    def reset():
        environment.state = random_state(environment, rng, number)
        environment.contact = False
        environment.swept_clearance = environment.walls.clearance(environment.state)
        return environment.observe()[1][0]
    previous, progress = reset(), 0.0
    _, location = environment.observe()
    first = [wall_row(environment, 0, location, 0, 150)]
    recording, reason, failures = True, "collection_horizon", 0
    x, y = [], []
    for step in range(config.rollout_steps):
        features, location = environment.observe()
        x.append(features)
        y.append(environment.teacher(location=location))
        location, failure, _ = environment.step(driver.predict(features))
        progress += (location[0] - previous + track.length / 2) % track.length - track.length / 2
        previous = location[0]
        if recording:
            first.append(wall_row(environment, (step + 1) * config.dt, location, progress, 150))
        if failure:
            failures += 1
            if recording:
                recording, reason = False, failure
            previous, progress = reset(), 0.0
    return np.asarray(x), np.asarray(y), np.asarray(first), {
        "collection_round": number, "model_round": number - 1,
        "end_reason": reason, "failures": failures, "labeled_states": config.rollout_steps,
    }


def select_checkpoint(output, config, tracks, training, history):
    candidates = sorted(history, key=lambda row: (-row["safe_passages"], row["validation_mse"], row["round"]))
    audit = []
    for row in candidates:
        if row["safe_passages"] != 44:
            continue
        model = NeuralDriver.load(output / f"models/round_{row['round']:03d}.npz")
        completed = 0
        for item in training:
            for scenario in SCENARIOS:
                result, _ = wall_simulate(tracks[item["id"]], replace(config, max_steps=2600),
                                          model, scenario, goal=tracks[item["id"]].length)
                completed += result["success"]
        audit.append({"round": row["round"], "validation_mse": row["validation_mse"],
                      "safe_sections": row["safe_passages"], "training_full_laps": completed})
        if completed == 33:
            write_json(output / "checkpoint_selection.json", {
                "criterion": "All 44 training sections and 33 training laps; then lowest action-validation MSE",
                "selected_round": row["round"], "uses_holdout_or_stress_results": False, "candidates": audit,
            })
            print(f"Selected round {row['round']}: 44/44 sections, 33/33 training laps.", flush=True)
            return model, row["round"]
    raise RuntimeError("No checkpoint satisfies training section and full-lap safety")


def train(output, base, rounds=120):
    output, base = Path(output), Path(base)
    if output.exists():
        raise FileExistsError(f"Existing run: {output}; choose a fresh output directory")
    config = replace(WALL_CONFIG, rounds=rounds)
    for directory in ("data", "models", "attempts", "rollouts", "visuals", "tracks"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    catalog = json.loads((base / "data/circuits.json").read_text())
    write_json(output / "data/circuits.json", catalog)
    tracks = {item["id"]: Track(config, item["knots"]) for item in catalog}
    training = [item for item in catalog if item["split"] == "train"]
    holdouts = [item for item in catalog if item["split"] == "holdout"]
    source_model = base / ("data/base_model.npz" if (base / "data/base_model.npz").exists() else "models/final.npz")
    shutil.copyfile(source_model, output / "data/base_model.npz")
    rng = np.random.default_rng(config.seed)
    if (base / "data/anchors.npz").exists():
        with np.load(base / "data/anchors.npz") as data:
            anchors = {key: data[key].copy() for key in data.files}
    else:
        with np.load(base / "data/aggregated_training_data.npz") as data:
            indices = np.concatenate([
                rng.choice(np.flatnonzero((data["track_id"] == item["id"]) & (np.abs(data["features"][:, 0]) < 2 / 9)), 512, replace=False)
                for item in training
            ])
            anchors = {"features": add_margin_feature(data["features"][indices], config),
                       "targets": data["targets"][indices], "track_id": data["track_id"][indices]}
    np.savez_compressed(output / "data/anchors.npz", **anchors)
    # Anchor selection must not advance the recovery-data RNG in a bundled reproduction.
    rng = np.random.default_rng(config.seed + 2)
    x_parts, y_parts, id_parts = [], [], []
    validation_x, validation_y, validation_ids = [], [], []
    validation_rng = np.random.default_rng(config.seed + 3)
    for item in training:
        environment = WallEnvironment(tracks[item["id"]], config)
        for generator, count, xs, ys, ids in [
            (rng, 900, x_parts, y_parts, id_parts),
            (validation_rng, 200, validation_x, validation_y, validation_ids),
        ]:
            for _ in range(count):
                state = random_state(environment, generator)
                if generator.uniform() < .5:
                    state.steering = float(generator.uniform(-.15, .15))
                features, location = environment.observe(state)
                xs.append(features)
                ys.append(environment.teacher(state, location))
                ids.append(item["id"])
    for item in training:
        environment = WallEnvironment(tracks[item["id"]], config)
        for fraction in (.11, .41, .79):
            for side in (-1, 1):
                environment.state = environment.make_state(fraction, side * 5.5, side * .18, 12)
                for _ in range(180):
                    features, location = environment.observe()
                    target = environment.teacher(location=location)
                    x_parts.append(features)
                    y_parts.append(target)
                    id_parts.append(item["id"])
                    _, failure, _ = environment.step(target)
                    if failure:
                        raise RuntimeError(f"Recovery demonstration failed on {item['name']}: {failure}")
    x, y = np.asarray(x_parts), np.asarray(y_parts)
    ids = np.asarray(id_parts, dtype=np.uint8)
    initial_count = len(x)
    np.savez_compressed(output / "data/initial_recovery.npz", features=x, targets=y, track_id=ids)
    np.savez_compressed(output / "data/validation.npz", features=validation_x, targets=validation_y,
                        track_id=validation_ids)
    capacity = initial_count + rounds * len(training) * config.rollout_steps
    xx, yy = np.empty((capacity, 18)), np.empty((capacity, 2))
    track_ids, round_ids = np.empty(capacity, dtype=np.uint8), np.zeros(capacity, dtype=np.int16)
    xx[:initial_count], yy[:initial_count], track_ids[:initial_count] = x, y, ids
    used = initial_count
    pools = [np.flatnonzero(ids == item["id"]) for item in training]
    anchor_pools = [np.flatnonzero(anchors["track_id"] == item["id"]) for item in training]
    driver = upgraded_driver(source_model)
    attempts, history, rollout_metrics = [], [], []
    scenarios = {item["id"]: corner_scenarios(tracks[item["id"]]) for item in catalog}
    write_json(output / "data/scenarios.json", scenarios)
    write_json(output / "data/schema.json", {
        "features": WallEnvironment.feature_names, "trace_columns": WALL_COLUMNS,
        "car_length_m": 4.8, "car_width_m": config.car_width,
        "collision": "Oriented rectangular footprint vs wall polylines; swept samples <=0.5 m and <=0.025 rad",
        "controls": "Unmodified learned steering/target-speed outputs; automatic pedals/gears, no safety shield",
        "success": "150 m section without any failure and mean lateral error <1 m during its last 20 states",
        "optimizer": "Continue weights with a fresh Adam state; previous files contain no optimizer moments",
    })
    for number in range(rounds + 1):
        failures = 0
        if number:
            recent, recordings = [], {}
            for index, item in enumerate(training):
                features, targets, trace, details = collect(tracks[item["id"]], config, driver, rng, number)
                selected = np.arange(used, used + len(features))
                xx[selected], yy[selected] = features, targets
                track_ids[selected], round_ids[selected] = item["id"], number
                used += len(features)
                pools[index] = np.r_[pools[index], selected]
                recent.append(selected)
                recordings[f"track_{item['id']:02d}"] = trace
                rollout_metrics.append({"track_id": item["id"], **details})
                failures += details["failures"]
            np.savez_compressed(output / f"rollouts/round_{number:03d}.npz", **recordings)
            for _ in range(config.updates_per_round):
                historical = np.concatenate([rng.choice(pool, 16) for pool in pools])
                latest = np.concatenate([rng.choice(pool, 16) for pool in recent])
                anchor = np.concatenate([rng.choice(pool, 12) for pool in anchor_pools])
                driver.train_batch(
                    np.vstack([xx[historical], xx[latest], anchors["features"][anchor]]),
                    np.vstack([yy[historical], yy[latest], anchors["targets"][anchor]]),
                    .0005 * (1 - .75 * number / rounds),
                )
        driver.save(output / f"models/round_{number:03d}.npz")
        traces, rows = {}, []
        for item in training:
            for index, scenario in enumerate(scenarios[item["id"]]):
                metric, trace = wall_simulate(tracks[item["id"]], config, driver, scenario)
                row = {"round": number, "track_id": item["id"], "split": item["split"], **metric}
                attempts.append(row)
                rows.append(row)
                traces[f"track_{item['id']:02d}_scenario_{index}"] = trace
        np.savez_compressed(output / f"attempts/train_{number:03d}.npz", **traces)
        error = driver.predict(np.asarray(validation_x)) - validation_y
        history.append({
            "round": number, "updates": driver.updates, "new_samples": used,
            "safe_passages": sum(row["success"] for row in rows),
            "wall_contacts": sum(row["wall_contacts"] for row in rows),
            "validation_mse": float(np.mean(error ** 2 * [4, 1])),
            "mean_exit_lateral_m": float(np.mean([row["exit_mean_lateral_m"] for row in rows if row["success"]])),
            "collection_failures": failures,
        })
        if number % 10 == 0 or number == 1:
            print(f"Wall round {number:3d}: {history[-1]['safe_passages']}/44 safe passages, "
                  f"{history[-1]['wall_contacts']} wall contacts, validation={history[-1]['validation_mse']:.6f}", flush=True)
    driver, selected_round = select_checkpoint(output, config, tracks, training, history)
    driver.save(output / "models/final.npz")
    model_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    write_json(output / "training_complete.json", {"updates": driver.updates, "sha256": model_hash,
                                                  "holdouts_not_evaluated": True})
    print("Weights frozen; testing unseen sections, stress recovery and full-lap regression.", flush=True)
    for number in range(rounds + 1):
        model = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        traces = {}
        for item in holdouts:
            for index, scenario in enumerate(scenarios[item["id"]]):
                metric, trace = wall_simulate(tracks[item["id"]], config, model, scenario)
                attempts.append({"round": number, "track_id": item["id"], "split": item["split"], **metric})
                traces[f"track_{item['id']:02d}_scenario_{index}"] = trace
        np.savez_compressed(output / f"attempts/holdout_{number:03d}.npz", **traces)
    regression, recordings = [], {}
    for item in catalog:
        track = tracks[item["id"]]
        for index, scenario in enumerate(SCENARIOS):
            metric, trace = wall_simulate(track, replace(config, max_steps=2600), driver, scenario, goal=track.length)
            regression.append({"track_id": item["id"], "split": item["split"], **metric})
            recordings[f"track_{item['id']:02d}_scenario_{index}"] = trace
    np.savez_compressed(output / "attempts/full_laps.npz", **recordings)
    stress, stress_traces = [], {}
    stress_rng = np.random.default_rng(config.seed + 4)
    baseline = upgraded_driver(source_model)
    for item in catalog:
        environment = WallEnvironment(tracks[item["id"]], config)
        for index in range(16):
            state = random_state(environment, stress_rng)
            location = environment.track.locate(state.x, state.y)
            scenario = {"name": f"Blind stress {index}", "fraction": location[0] / environment.track.length,
                        "lateral": location[1], "heading_error": float((state.heading - location[2] + np.pi) % (2 * np.pi) - np.pi),
                        "speed": state.speed}
            for label, model in (("previous", baseline), ("final", driver)):
                metric, trace = wall_simulate(tracks[item["id"]], config, model, scenario)
                stress.append({"track_id": item["id"], "split": item["split"], "model": label,
                               "initial": scenario, **metric})
                stress_traces[f"track_{item['id']:02d}_{label}_{index:02d}"] = trace
    np.savez_compressed(output / "attempts/stress.npz", **stress_traces)
    demonstrations = {}
    for item in catalog:
        scenario = {**scenarios[item["id"]][2], "name": "Unsafe scripted wall-contact demonstration",
                    "lateral": 7, "heading_error": .35, "speed": 8}
        metric, trace = wall_simulate(tracks[item["id"]], config,
                                    lambda _: np.array([0, 2 * 8 / config.max_speed - 1]), scenario)
        demonstrations[f"track_{item['id']:02d}"] = trace
        if not metric["wall_contacts"]:
            raise RuntimeError(f"Wall-contact illustration failed on {item['name']}")
    np.savez_compressed(output / "attempts/unsafe_demonstrations.npz", **demonstrations)
    np.savez_compressed(output / "data/aggregated_recovery.npz", features=xx[:used], targets=yy[:used],
                        track_id=track_ids[:used], round=round_ids[:used])
    attempts.sort(key=lambda row: (row["round"], row["track_id"], row["scenario"]))
    write_json(output / "attempt_metrics.json", attempts)
    write_json(output / "rollout_metrics.json", rollout_metrics)
    write_json(output / "full_lap_metrics.json", regression)
    write_json(output / "stress_metrics.json", stress)
    write_json(output / "history.json", history)
    save_csv(output / "metrics.csv", history)
    final_train = sum(row["success"] for row in attempts if row["round"] == selected_round and row["split"] == "train")
    final_unseen = sum(row["success"] for row in attempts if row["round"] == selected_round and row["split"] == "holdout")
    full_train = sum(row["success"] for row in regression if row["split"] == "train")
    full_unseen = sum(row["success"] for row in regression if row["split"] == "holdout")
    write_json(output / "manifest.json", {
        "config": asdict(config), "rounds": rounds, "updates": rounds * config.updates_per_round,
        "selected_round": selected_round, "selected_updates": driver.updates, "architecture": [18, 48, 32, 2],
        "base_sha256": hashlib.sha256(source_model.read_bytes()).hexdigest(),
        "final_sha256": model_hash, "anchors": len(anchors["features"]),
        "initial_recovery_samples": initial_count, "final_recovery_samples": used,
        "training_ids": [item["id"] for item in training], "holdout_ids": [item["id"] for item in holdouts],
        "baseline": history[0], "final": history[selected_round], "last_round": history[-1],
        "final_safe_train": final_train, "final_safe_unseen": final_unseen,
        "full_lap_train": full_train, "full_lap_unseen": full_unseen,
        "stress_previous_safe": sum(row["success"] for row in stress if row["model"] == "previous"),
        "stress_final_safe": sum(row["success"] for row in stress if row["model"] == "final"),
        "stress_previous_contacts": sum(row["wall_contacts"] for row in stress if row["model"] == "previous"),
        "stress_final_contacts": sum(row["wall_contacts"] for row in stress if row["model"] == "final"),
        "success": final_train == 44 and full_train == 33,
        "limits": ["Sampled sweep, not continuous impact physics", "No safety shield or real-car claim",
                   "Projected margin is a derived geometric feature, not a vision sensor",
                   "Only one seed and two unseen layouts", "Unavoidable states may still fail"],
    })
    print(f"Final sections {final_train}/44 train, {final_unseen}/8 unseen; "
          f"full laps {full_train}/33 train, {full_unseen}/6 unseen.", flush=True)
    if final_train != 44 or full_train != 33:
        raise RuntimeError("Training corner or full-lap regression criteria were not met")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="wall_artifacts")
    parser.add_argument("--base", default="race_artifacts")
    parser.add_argument("--rounds", type=int, default=120)
    args = parser.parse_args()
    train(args.output, args.base, args.rounds)
