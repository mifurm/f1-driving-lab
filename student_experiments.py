"""Prepare the offline student lab: episode-structured data, matched retraining and frozen tests.

Differences from the first student lab (preserved in student_artifacts/, source in git history):
validation is an independent set of expert episodes drawn by the same sampler as training (the old
validation set held only deep-recovery states), a separately named recovery-stress set is evaluation
only, every training recipe is repeated for ten seeds, and fixed-section outcomes keep their track
and scenario so they can be analysed with sections as the unit of replication.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy

from driving import NeuralDriver, tracks_from_catalog, write_json
from race_sim import RaceEnvironment
from wall_sim import WALL_CONFIG, WallEnvironment, add_margin_feature, corner_scenarios, wall_simulate


MASKS = {"full": [], "no_future_geometry": list(range(5, 16)),
         "no_heading": [1, 2, 16], "no_margin": [16]}
FEATURE_NAMES = RaceEnvironment.feature_names + [
    "projected_footprint_wall_margin_half_width", "speed_dependent_steering_budget_fraction"]

LAB_VERSION = "student-lab-2"
ABLATION_SEEDS = tuple(range(301, 311))
PAIRED_SEEDS = tuple(range(401, 411))
UPDATES, BATCH, CHECKPOINT_EVERY = 256, 132, 32
TRAINING_TRACKS = tuple(range(11))
MONITOR_TRACKS = (2, 3, 5, 9)
STARTS = {
    "narrow": {"lateral": 1.0, "heading": 0.04, "speed": [8, 14]},
    "typical": {"lateral": 3.0, "heading": 0.12, "speed": [8, 16]},
    "broad": {"lateral": 4.0, "heading": 0.16, "speed": [8, 18]},
    "stress": {"outward": True, "lateral": [5.5, 6.6], "heading": [0.18, 0.30], "speed": [8, 16]},
}
EPISODE_STEPS, FRAME_STRIDE, FRAMES_PER_EPISODE = 150, 4, 32
TRAIN_EPISODES, VALIDATION_EPISODES, STRESS_EPISODES = 16, 6, 4
STRESS_STRIDE, STRESS_FRAMES = 3, 16
STARTER_STRESS_ROWS_PER_TRACK = 20
CONDITIONS = [{"name": "Clean", "noise": 0, "delay": 0, "grip": 1},
              {"name": "Noisy observations", "noise": 1, "delay": 0, "grip": 1},
              {"name": "0.5 s observation delay", "noise": 0, "delay": 5, "grip": 1},
              {"name": "65% grip envelope", "noise": 0, "delay": 0, "grip": .65},
              {"name": "Combined shift", "noise": .5, "delay": 3, "grip": .75}]


def export_model(driver, mask="full"):
    return {"weights": [w.tolist() for w in driver.weights], "biases": [b.tolist() for b in driver.biases],
            "updates": driver.updates, "mask": mask}


def masked(x, mask):
    result = np.array(x, copy=True)
    result[..., MASKS[mask]] = 0
    return result


def new_catalog(config, old):
    catalog = list(old)
    for index in range(3):
        angle = np.linspace(0, 2 * np.pi, 81)
        phase = index * .83 + .31
        radius = 1 + .08 * np.cos(3 * angle + phase) + .04 * np.sin(5 * angle - phase)
        points = np.c_[radius * (235 + index * 12) * np.cos(angle),
                       radius * (170 + index * 9) * np.sin(angle)]
        points[-1] = points[0]
        catalog.append({"id": 13 + index, "name": f"Pristine Student Circuit {index + 1}",
                        "split": "pristine_test", "knots": points.tolist()})
    return catalog


def policy(driver, config, seed=1, mask="full", noise=0, delay=0):
    rng, queue = np.random.default_rng(seed), []
    def predict(features):
        observed = np.array(features[:16], copy=True)
        offset, heading, speed = rng.normal(size=3)
        observed[0] += noise * offset / (config.track_width / 2)
        angle = np.arctan2(observed[1], observed[2]) + noise * .05 * heading
        observed[1:3] = np.sin(angle), np.cos(angle)
        observed[3] = np.clip(observed[3] + noise * .6 * speed / config.max_speed, 0, 1)
        queue.append(add_margin_feature(observed, config))
        return driver.predict(masked(queue[max(0, len(queue) - 1 - delay)], mask))
    return predict


def rule(features):
    steer = np.clip(-.08 * features[0] * 9 - .9 * np.arctan2(features[1], features[2]), -.5, .5)
    return np.array([steer / .5, 2 * 14 / 36 - 1])


def export_track(item, track):
    return {**{key: item[key] for key in ("id", "name", "split")},
            **{key: getattr(track, key).tolist() for key in
               ("xy", "left", "right", "normal", "tangent", "heading", "curvature")},
            "ds": track.ds, "length": track.length}


def row(features, targets, identifier, split, track_id, source, episode=None):
    return {"id": identifier, "features": features.tolist(), "targets": targets.tolist(),
            "split": split, "track_id": int(track_id), "source": source,
            "episode": episode or identifier}


def geometry_summary(track):
    geometry = track.geometry
    return {"min_radius_m": geometry["min_radius_m"], "edge_crossings": geometry["edge_crossings"],
            "valid": not geometry["problems"], "problems": geometry["problems"]}


# ---- episode-structured data ------------------------------------------------------------------------------------------
def sample_start(rng, spec):
    speed, fraction = float(rng.uniform(*spec["speed"])), float(rng.uniform())
    if spec.get("outward"):
        side = -1.0 if rng.uniform() < 0.5 else 1.0
        return {"fraction": fraction, "lateral": side * float(rng.uniform(*spec["lateral"])),
                "heading_error": side * float(rng.uniform(*spec["heading"])), "speed": speed}
    return {"fraction": fraction, "lateral": float(rng.uniform(-spec["lateral"], spec["lateral"])),
            "heading_error": float(rng.uniform(-spec["heading"], spec["heading"])), "speed": speed}


def near_duplicate(a, b):
    return (a["track_id"] == b["track_id"] and abs(a["fraction"] - b["fraction"]) < 0.02
            and abs(a["lateral"] - b["lateral"]) < 0.3 and abs(a["heading_error"] - b["heading_error"]) < 0.02
            and abs(a["speed"] - b["speed"]) < 0.5)


def start_state(environment, rng, spec):
    """Sample a start whose footprint is clear of the walls; also count the starts that were rejected."""
    for rejected in range(60):
        start = sample_start(rng, spec)
        environment.state = environment.make_state(start["fraction"], start["lateral"], start["heading_error"], start["speed"])
        environment.swept_clearance = environment.walls.clearance(environment.state)
        environment.contact = environment.swept_clearance <= 1e-8
        if not environment.contact:
            return start, rejected
    raise RuntimeError("Could not sample a start clear of the walls")


def expert_episode(environment, rng, spec, stride, keep):
    start, rejected = start_state(environment, rng, spec)
    frames, ended = [], "horizon"
    for step in range(EPISODE_STEPS):
        features, location = environment.observe()
        label = environment.teacher(location=location)
        if step % stride == 0 and len(frames) < keep:
            frames.append((features, label))
        _, reason, _ = environment.step(label)
        if reason:
            ended = reason
            break
    return {"start": start, "frames": frames, "ended": ended, "rejected": rejected}


def generate(tracks, spec, episodes, split, label, stride, keep, seed_base):
    features, targets, track_ids, episode_ids, frame_ids, meta = [], [], [], [], [], []
    for track_id in TRAINING_TRACKS:
        environment = WallEnvironment(tracks[track_id], WALL_CONFIG)
        rng = np.random.default_rng([seed_base, track_id])
        for index in range(episodes):
            result = expert_episode(environment, rng, spec, stride, keep)
            for frame, (x, y) in enumerate(result["frames"]):
                features.append(x), targets.append(y), track_ids.append(track_id)
                episode_ids.append(len(meta)), frame_ids.append(frame)
            meta.append({"episode": f"{label}-t{track_id}-e{index}", "index": len(meta), "split": split, "track_id": track_id,
                         "frames": len(result["frames"]), "ended": result["ended"],
                         "rejected_starts": result["rejected"], **result["start"]})
    return {"features": np.asarray(features), "targets": np.asarray(targets), "track_id": np.asarray(track_ids, dtype=np.uint8),
            "episode": np.asarray(episode_ids, dtype=np.int32), "frame": np.asarray(frame_ids, dtype=np.int16), "episodes": meta}


def assert_no_split_leak(*groups):
    starts = [(group[0]["split"], meta) for group in groups for meta in group]
    for i, (split_a, a) in enumerate(starts):
        for split_b, b in starts[i + 1:]:
            if split_a != split_b and near_duplicate(a, b):
                raise RuntimeError(f"Episodes {a['episode']} and {b['episode']} start from nearly the same state")


def episode_rows(data, meta_index, split, prefix=""):
    rows, meta = [], data["episodes"][meta_index]
    for i in np.flatnonzero(data["episode"] == meta_index):
        rows.append(row(data["features"][i], data["targets"][i], f"{prefix}{meta['episode']}-s{data['frame'][i]}", split,
                        data["track_id"][i], "expert demonstration" if split != "stress" else "expert label, deep-recovery start",
                        episode=f"{prefix}{meta['episode']}"))
    return rows


# ---- retraining ---------------------------------------------------------------------------------------------------------
def mse(prediction, targets):
    return float(np.mean((prediction - targets) ** 2 * [4, 1]))


def fit(x, y, validation_x, validation_y, stress_x, stress_y, seed, mask, output, key):
    """256 Adam updates; the checkpoint with the lowest i.i.d. validation error is selected. Stress error is reported only."""
    net = NeuralDriver(seed, input_size=18, hidden_sizes=(24, 16))
    rng = np.random.default_rng([seed, 1])
    xx, vx, sx = masked(x, mask), masked(validation_x, mask), masked(stress_x, mask)
    history, best, diverged = [], None, False
    for update in range(UPDATES + 1):
        if update:
            batch = rng.integers(0, len(xx), size=BATCH)
            try:
                net.train_batch(xx[batch], y[batch], .002 * (1 - .7 * update / UPDATES))
            except FloatingPointError:
                diverged = True
                break
        if update % CHECKPOINT_EVERY == 0:
            validation_mse = mse(net.predict(vx), validation_y)
            history.append({"update": update, "train_mse": mse(net.predict(xx), y), "validation_mse": validation_mse,
                            "stress_mse": mse(net.predict(sx), stress_y)})
            net.save(output / f"models/{key}_update_{update:03d}.npz")
            if best is None or validation_mse < best[0]:
                best = (validation_mse, update, [p.copy() for p in net.parameters])
    for parameter, saved in zip(net.parameters, best[2]):
        parameter[...] = saved
    net.updates = best[1]
    net.save(output / f"models/{key}.npz")
    return net, history, diverged


# ---- parallel closed-loop evaluation ---------------------------------------------------------------------------------------
_WORKER = {}


def _initialise(output, catalog):
    tracks = tracks_from_catalog(WALL_CONFIG, catalog)
    _WORKER.update(output=Path(output), tracks=tracks, splits={item["id"]: item["split"] for item in catalog}, models={},
                   scenarios={item["id"]: corner_scenarios(tracks[item["id"]]) for item in catalog})


def _model(name):
    if name not in _WORKER["models"]:
        _WORKER["models"][name] = NeuralDriver.load(_WORKER["output"] / f"models/{name}.npz")
    return _WORKER["models"][name]


def _jittered(canonical, seed, track_id, index):
    rng = np.random.default_rng([seed, track_id, index])
    return {**canonical, "lateral": canonical["lateral"] + float(rng.uniform(-.15, .15)),
            "heading_error": canonical["heading_error"] + float(rng.uniform(-.01, .01)),
            "speed": canonical["speed"] + float(rng.uniform(-.2, .2))}


def _execute(task):
    config, track_id, index = WALL_CONFIG, task["track_id"], task["scenario_index"]
    track, canonical = _WORKER["tracks"][track_id], _WORKER["scenarios"][track_id][index]
    common = {"track_id": track_id, "split": _WORKER["splits"][track_id], "scenario_index": index}
    kind = task["kind"]
    if kind in ("ablation", "monitor"):
        result, _ = wall_simulate(track, config, policy(_model(task["model"]), config, mask=task["mask"]), canonical)
        return {**task["tag"], **common, **result}
    scenario = _jittered(canonical, task["seed"], track_id, index)
    if kind == "controller":
        label = task["label"]
        driver = rule if label == "rules" else None if label == "expert" else policy(_model("pretrained"), config, task["seed"])
        result, _ = wall_simulate(track, config, driver, scenario)
        return {"controller": label, "seed": task["seed"], **common, **result}
    condition = task["condition"]
    shifted = replace(config, max_braking=config.max_braking * condition["grip"],
                      max_lateral_acceleration=config.max_lateral_acceleration * condition["grip"])
    driver = policy(_model("pretrained"), config, task["seed"], noise=condition["noise"], delay=condition["delay"])
    result, _ = wall_simulate(track, shifted, driver, scenario)
    return {"condition": condition["name"], "settings": condition, "seed": task["seed"], **common, **result}


def run_tasks(tasks, output, catalog, workers):
    if workers <= 1:
        _initialise(output, catalog)
        return [_execute(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise, initargs=(str(output), catalog)) as pool:
        return list(pool.map(_execute, tasks, chunksize=24))


def kind_of(scenario_name):
    return "recovery" if "recovery" in scenario_name.lower() else "nominal"


def prepare(output, base, workers=None):
    started = time.perf_counter()
    output, base = Path(output), Path(base)
    if output.exists():
        raise FileExistsError(f"Existing student lab: {output}")
    workers = workers or min(16, os.cpu_count() or 1)
    for directory in ("data", "models", "visuals"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config = WALL_CONFIG
    old = json.loads((base / "data/circuits.json").read_text())
    catalog = new_catalog(config, old)
    tracks = tracks_from_catalog(config, catalog)
    scenarios = {item["id"]: corner_scenarios(tracks[item["id"]]) for item in catalog}
    for item in catalog[13:]:
        for scenario in scenarios[item["id"]]:
            result, _ = wall_simulate(tracks[item["id"]], config, None, scenario)
            if not result["success"]:
                raise RuntimeError(f"Teacher cannot traverse fresh test geometry: {item['name']}")
    valid_training = [i for i in TRAINING_TRACKS if not tracks[i].geometry["problems"]]
    legacy_folded = [i for i in TRAINING_TRACKS if tracks[i].geometry["problems"]]

    print("Generating independent expert episodes (train, validation, recovery stress).", flush=True)
    train = generate(tracks, STARTS["typical"], TRAIN_EPISODES, "train", "train", FRAME_STRIDE, FRAMES_PER_EPISODE, 2100)
    validation = generate(tracks, STARTS["typical"], VALIDATION_EPISODES, "validation", "val", FRAME_STRIDE, FRAMES_PER_EPISODE, 2200)
    stress = generate(tracks, STARTS["stress"], STRESS_EPISODES, "stress", "stress", STRESS_STRIDE, STRESS_FRAMES, 2300)
    assert_no_split_leak(train["episodes"], validation["episodes"])
    for name, data in (("training", train), ("validation", validation), ("stress", stress)):
        np.savez_compressed(output / f"data/{name}.npz", features=data["features"], targets=data["targets"],
                            track_id=data["track_id"], episode=data["episode"], frame=data["frame"])
    write_json(output / "data/episodes.json", {"train": train["episodes"], "validation": validation["episodes"], "stress": stress["episodes"]})
    x, y, vx, vy, sx, sy = train["features"], train["targets"], validation["features"], validation["targets"], stress["features"], stress["targets"]

    shutil.copyfile(base / "models/final.npz", output / "models/pretrained.npz")
    pretrained = NeuralDriver.load(base / "models/final.npz")
    models, histories, diverged = {}, {}, {}
    for mask in MASKS:
        for seed in ABLATION_SEEDS:
            key = f"{mask}_{seed}"
            models[key], histories[key], diverged[key] = fit(x, y, vx, vy, sx, sy, seed, mask, output, key)
            print(f"Trained {key}: selected update {models[key].updates}"
                  f"{' (DIVERGED, counted as a failed seed)' if diverged[key] else ''}.", flush=True)
    clean_leak, leak_history, _ = fit(x, y, vx, vy, sx, sy, 314, "full", output, "leakage_clean")
    write_json(output / "training_complete.json", {
        "fresh_test_ids": [13, 14, 15], "fresh_test_not_yet_evaluated": True,
        "model_sha256": {key: hashlib.sha256((output / f"models/{key}.npz").read_bytes()).hexdigest() for key in models},
        "leakage_clean_sha256": hashlib.sha256((output / "models/leakage_clean.npz").read_bytes()).hexdigest(),
        "training_ids": list(TRAINING_TRACKS), "seeds": list(ABLATION_SEEDS), "diverged": diverged,
        "selection": "i.i.d. episode validation only; the recovery-stress set is reported, never used to select",
    })
    write_json(output / "training_histories.json", histories)
    print("All clean weights frozen. Evaluating fixed sections, controllers and shifts in parallel.", flush=True)

    ablation_tasks = [{"kind": "ablation", "model": key, "mask": key.rsplit("_", 1)[0], "track_id": item["id"], "scenario_index": index,
                       "tag": {"model": key, "mask": key.rsplit("_", 1)[0], "seed": int(key.rsplit("_", 1)[1])}}
                      for key in models for item in catalog for index in range(4)]
    controller_tasks = [{"kind": "controller", "label": label, "seed": seed, "track_id": item["id"], "scenario_index": index}
                        for label in ("rules", "expert", "pretrained") for seed in PAIRED_SEEDS for item in catalog for index in range(4)]
    robustness_tasks = [{"kind": "robustness", "condition": condition, "seed": seed, "track_id": item["id"], "scenario_index": index}
                        for condition in CONDITIONS for seed in PAIRED_SEEDS for item in catalog for index in range(4)]
    checkpoints = list(range(0, UPDATES + 1, CHECKPOINT_EVERY))
    monitor_tasks = [{"kind": "monitor", "model": f"full_{seed}_update_{update:03d}", "mask": "full", "track_id": track_id,
                      "scenario_index": index, "tag": {"seed": seed, "update": update}}
                     for seed in ABLATION_SEEDS for update in checkpoints for track_id in MONITOR_TRACKS for index in range(4)]
    stage = time.perf_counter()
    ablations = run_tasks(ablation_tasks, output, catalog, workers)
    comparisons = run_tasks(controller_tasks, output, catalog, workers)
    robustness = run_tasks(robustness_tasks, output, catalog, workers)
    monitor = run_tasks(monitor_tasks, output, catalog, workers)
    print(f"{len(ablations)} ablation, {len(comparisons)} controller, {len(robustness)} robustness and {len(monitor)} monitor "
          f"rollouts in {time.perf_counter() - stage:.0f} s on {workers} workers.", flush=True)
    for record in ablations + comparisons + robustness:
        record["kind"] = kind_of(record["scenario"])
    closed_loop = {}
    for record in monitor:
        entry = closed_loop.setdefault(f"full_{record['seed']}", {}).setdefault(record["update"], {"update": record["update"], "nominal": [0, 0], "recovery": [0, 0]})
        slot = entry[kind_of(record["scenario"])]
        slot[0] += record["success"]
        slot[1] += 1
    closed_loop_reference = {key: [{"update": u, "nominal": {"k": e["nominal"][0], "n": e["nominal"][1]},
                                    "recovery": {"k": e["recovery"][0], "n": e["recovery"][1]}} for u, e in sorted(entries.items())]
                             for key, entries in closed_loop.items()}

    test_rng, test_x, test_y, test_ids = np.random.default_rng(315), [], [], []
    for item in catalog[13:]:
        environment = WallEnvironment(tracks[item["id"]], config)
        for _ in range(200):
            state = environment.make_state(float(test_rng.uniform()), float(test_rng.uniform(-6, 6)),
                                           float(test_rng.uniform(-.25, .25)), float(test_rng.uniform(6, 18)))
            features, location = environment.observe(state)
            test_x.append(features), test_y.append(environment.teacher(state, location)), test_ids.append(item["id"])
    tx, ty = np.asarray(test_x), np.asarray(test_y)
    clean_test_mse = mse(clean_leak.predict(tx), ty)
    contaminated, contaminated_history, _ = fit(np.vstack([x, tx]), np.vstack([y, ty]), vx, vy, sx, sy, 314, "full", output, "leakage_contaminated")
    leakage = {"clean_test_mse": clean_test_mse, "contaminated_test_mse": mse(contaminated.predict(tx), ty),
               "clean_train_mse_mislabeled_as_test": mse(clean_leak.predict(x), y), "contaminated_test_observations": len(tx),
               "valid_generalization_claim": False, "clean_history": leak_history, "contaminated_history": contaminated_history}
    np.savez_compressed(output / "data/sealed_test_observations.npz", features=tx, targets=ty, track_id=test_ids)
    test_rows = [row(tx[i], ty[i], f"sealed-test-{i}", "test", test_ids[i], "sealed test label: leakage lesson only") for i in range(len(tx))]

    starter = []
    for track_id in TRAINING_TRACKS:
        for data, split in ((train, "train"), (validation, "validation")):
            first = next(m["index"] for m in data["episodes"] if m["track_id"] == track_id)
            starter += episode_rows(data, first, split, "starter-")
    starter_episodes = {}
    for data, split in ((train, "train"), (validation, "validation")):
        for meta in data["episodes"]:
            if any(r["episode"] == f"starter-{meta['episode']}" for r in starter):
                starter_episodes[f"starter-{meta['episode']}"] = {key: meta[key] for key in ("track_id", "fraction", "lateral", "heading_error", "speed")} | {"split": split}
    stress_rows = []
    for track_id in TRAINING_TRACKS:
        indices = np.flatnonzero(stress["track_id"] == track_id)
        for i in indices[np.linspace(0, len(indices) - 1, STARTER_STRESS_ROWS_PER_TRACK).astype(int)]:
            meta = stress["episodes"][stress["episode"][i]]
            stress_rows.append(row(stress["features"][i], stress["targets"][i], f"{meta['episode']}-s{stress['frame'][i]}", "stress",
                                   track_id, "expert label, deep-recovery start", episode=meta["episode"]))

    golden = []
    for track_id, scenario_index in ((0, 0), (0, 2), (1, 3), (13, 0), (14, 2)):
        environment = WallEnvironment(tracks[track_id], config)
        scenario = scenarios[track_id][scenario_index]
        environment.state = environment.make_state(scenario["fraction"], scenario["lateral"], scenario["heading_error"], scenario["speed"])
        observed, _ = environment.observe()
        expert = environment.teacher()
        prediction = pretrained.predict(observed)
        environment.step(prediction)
        after = environment.state
        golden.append({"track_id": track_id, "scenario": scenario, "features": observed.tolist(), "expert": expert.tolist(),
                       "prediction": prediction.tolist(),
                       "after": {"x": after.x, "y": after.y, "heading": after.heading, "speed": after.speed,
                                 "steering": after.steering, "gear": after.gear, "rpm": after.rpm, "brake": after.brake,
                                 "margin": environment.swept_clearance}})
    gradient_model = NeuralDriver(99, input_size=18, hidden_sizes=(24, 16))
    before_model = export_model(gradient_model)
    batch = [row(x[i], y[i], f"golden-{i}", "train", train["track_id"][i], "parity fixture") for i in range(8)]
    loss_before = gradient_model.train_batch(x[:8], y[:8], .001)
    write_json(output / "data/golden.json", {"physics": golden, "training": {
        "before": before_model, "after": export_model(gradient_model), "batch": batch, "loss": loss_before}})

    stress_completed = sum(m["ended"] == "horizon" for m in stress["episodes"])
    generation = {
        "starts": STARTS, "episode_steps": EPISODE_STEPS, "frame_stride": FRAME_STRIDE, "frames_per_episode": FRAMES_PER_EPISODE,
        "episodes_per_track": {"train": TRAIN_EPISODES, "validation": VALIDATION_EPISODES, "stress": STRESS_EPISODES},
        "sampler": "Track chosen per episode from the eleven training circuits; fraction uniform along the lap; lateral offset, heading error "
                   "and speed uniform inside the named ranges; stress starts point outward on a random side",
        "split_rule": "Whole episodes are split by construction; train and validation use the same sampler with independent random starts, "
                      "and near-duplicate starts across splits would raise an error",
        "start_rejection": "A sampled start whose footprint already touches a wall is re-sampled and counted as rejected_starts",
        "stress_set": "Evaluation only: early frames of expert drives from deep-recovery starts; never used to train or select",
        "valid_training_tracks": valid_training, "legacy_geometry_tracks": legacy_folded, "monitor_tracks": list(MONITOR_TRACKS),
        "legacy_holdout_tracks": [11, 12], "sealed_tracks": [13, 14, 15],
        "expert_reliability": {"stress_episodes": len(stress["episodes"]), "stress_completed_150_steps": stress_completed,
                               "stress_rejected_starts": sum(m["rejected_starts"] for m in stress["episodes"]),
                               "typical_rejected_starts": sum(m["rejected_starts"] for m in train["episodes"] + validation["episodes"]),
                               "typical_episodes_ended_early": sum(m["ended"] != "horizon" for m in train["episodes"] + validation["episodes"])},
        "seeds": {"ablation": list(ABLATION_SEEDS), "paired": list(PAIRED_SEEDS)},
        "random_streams": "Initialisation default_rng(seed); mini-batches default_rng([seed, 1]); data default_rng([base, track]); "
                          "evaluation start jitter default_rng([seed, track, scenario])",
    }
    tracks_export = [{**export_track(item, tracks[item["id"]]), "geometry": geometry_summary(tracks[item["id"]])} for item in catalog]
    data = {"version": LAB_VERSION, "config": asdict(config), "features": FEATURE_NAMES, "tracks": tracks_export, "scenarios": scenarios,
            "pretrained": export_model(pretrained), "dataGeneration": generation, "starter": starter, "starterEpisodes": starter_episodes,
            "stressRows": stress_rows, "testRows": test_rows,
            "models": {key: export_model(model, key.rsplit("_", 1)[0]) for key, model in models.items()},
            "histories": histories, "diverged": diverged, "closedLoopReference": closed_loop_reference, "ablations": ablations,
            "comparisons": comparisons, "robustness": robustness, "conditions": CONDITIONS, "leakage": leakage, "golden": golden}
    write_json(output / "data/lab.json", data)
    for name, value in (("ablations", ablations), ("controller_comparisons", comparisons), ("robustness", robustness),
                        ("leakage", leakage), ("closed_loop_reference", closed_loop_reference)):
        write_json(output / f"{name}.json", value)
    write_json(output / "manifest.json", {
        "lab_version": LAB_VERSION, "training_tracks": list(TRAINING_TRACKS), "legacy_holdouts": [11, 12], "pristine_test_tracks": [13, 14, 15],
        "valid_training_tracks": valid_training, "legacy_geometry_tracks": legacy_folded, "monitor_tracks": list(MONITOR_TRACKS),
        "training_samples": len(x), "validation_samples": len(vx), "stress_samples": len(sx),
        "training_episodes": len(train["episodes"]), "validation_episodes": len(validation["episodes"]), "stress_episodes": len(stress["episodes"]),
        "ablation_models": len(models), "ablation_seeds": list(ABLATION_SEEDS), "paired_seeds": list(PAIRED_SEEDS), "updates_per_model": UPDATES,
        "diverged_seeds": [key for key, flag in diverged.items() if flag],
        "ablation_rollouts": len(ablations), "controller_rollouts": len(comparisons), "robustness_rollouts": len(robustness),
        "monitor_rollouts": len(monitor), "source_model_sha256": hashlib.sha256((base / "models/final.npz").read_bytes()).hexdigest(),
        "pristine_test_protocol": "Classroom protocol, not secrecy: all clean weights and selections were frozen and hashed before any "
                                  "evaluation on tracks 13-15, but the page embeds those circuits, so blind claims need a teacher-controlled fresh test",
        "leakage_demo": "Separate intentionally contaminated model; never included in clean benchmark results",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "workers": workers},
        "seconds": time.perf_counter() - started, "data_generation": generation,
    })
    print(f"Prepared {len(models)} models x {len(checkpoints)} checkpoints and {len(ablations) + len(comparisons) + len(robustness) + len(monitor)} "
          f"rollouts in {time.perf_counter() - started:.0f} s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="student_v2_artifacts")
    parser.add_argument("--base", default="wall_artifacts")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    prepare(args.output, args.base, args.workers)
