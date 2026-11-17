"""Prepare an offline student lab with genuine retraining and sealed-track tests."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from driving import NeuralDriver, tracks_from_catalog, write_json
from race_sim import RaceEnvironment
from wall_sim import WALL_CONFIG, WallEnvironment, add_margin_feature, corner_scenarios, wall_simulate


MASKS = {"full": [], "no_future_geometry": list(range(5, 16)),
         "no_heading": [1, 2, 16], "no_margin": [16]}
FEATURE_NAMES = RaceEnvironment.feature_names + [
    "projected_footprint_wall_margin_half_width", "speed_dependent_steering_budget_fraction"]


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


def fit(x, y, validation_x, validation_y, seed, mask, output, key):
    rng = np.random.default_rng(seed)
    net = NeuralDriver(seed, input_size=18, hidden_sizes=(24, 16))
    xx, vx = masked(x, mask), masked(validation_x, mask)
    history = []
    best = None
    for update in range(257):
        if update:
            batch = rng.integers(0, len(xx), size=132)
            net.train_batch(xx[batch], y[batch], .002 * (1 - .7 * update / 256))
        if update % 32 == 0:
            mse = float(np.mean((net.predict(vx) - validation_y) ** 2 * [4, 1]))
            history.append({"update": update, "train_mse": float(np.mean((net.predict(xx) - y) ** 2 * [4, 1])),
                            "validation_mse": mse})
            net.save(output / f"models/{key}_update_{update:03d}.npz")
            if best is None or mse < best[0]:
                best = (mse, update, [p.copy() for p in net.parameters])
    for parameter, saved in zip(net.parameters, best[2]):
        parameter[...] = saved
    net.updates = best[1]
    net.save(output / f"models/{key}.npz")
    return net, history


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


def prepare(output, base):
    output, base = Path(output), Path(base)
    if output.exists():
        raise FileExistsError(f"Existing student lab: {output}")
    for directory in ("data", "models", "visuals"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config = WALL_CONFIG
    old = json.loads((base / "data/circuits.json").read_text())
    catalog = new_catalog(config, old)
    tracks = tracks_from_catalog(config, catalog)
    for item in catalog[13:]:
        for scenario in corner_scenarios(tracks[item["id"]]):
            result, _ = wall_simulate(tracks[item["id"]], config, None, scenario)
            if not result["success"]:
                raise RuntimeError(f"Teacher cannot traverse fresh test geometry: {item['name']}")
    rng = np.random.default_rng(300)
    with np.load(base / "data/aggregated_recovery.npz") as data:
        features, targets, ids = data["features"], data["targets"], data["track_id"]
    indices = np.concatenate([rng.choice(np.flatnonzero(ids == i), 512, replace=False) for i in range(11)])
    x, y, train_ids = features[indices], targets[indices], ids[indices]
    with np.load(base / "data/validation.npz") as data:
        vx, vy, validation_ids = data["features"], data["targets"], data["track_id"]
    np.savez_compressed(output / "data/training.npz", features=x, targets=y, track_id=train_ids)
    np.savez_compressed(output / "data/validation.npz", features=vx, targets=vy, track_id=validation_ids)
    with np.load(base / "data/initial_recovery.npz") as data:
        initial_x, initial_y, initial_ids = data["features"], data["targets"], data["track_id"]
    starter_indices = rng.choice(np.arange(9900, len(initial_x)), 256, replace=False)
    starter = [row(initial_x[i], initial_y[i], f"starter-train-{i}", "train", initial_ids[i], "expert rollout",
                   episode=f"initial-expert-episode-{(i - 9900) // 180}")
               for i in starter_indices]
    for i in rng.choice(len(vx), 64, replace=False):
        starter.append(row(vx[i], vy[i], f"starter-validation-{i}", "validation", validation_ids[i],
                           "independent expert-labeled recovery"))
    pretrained = NeuralDriver.load(base / "models/final.npz")
    shutil.copyfile(base / "models/final.npz", output / "models/pretrained.npz")
    models, histories = {}, {}
    for mask in MASKS:
        for seed in (301, 302, 303):
            key = f"{mask}_{seed}"
            net, history = fit(x, y, vx, vy, seed, mask, output, key)
            models[key], histories[key] = net, history
            print(f"Trained {key}: 256 actual updates; selected validation checkpoint {net.updates}.", flush=True)
    clean_leak, leak_history = fit(x, y, vx, vy, 314, "full", output, "leakage_clean")
    write_json(output / "training_complete.json", {
        "fresh_test_ids": [13, 14, 15], "fresh_test_not_yet_evaluated": True,
        "model_sha256": {key: hashlib.sha256((output / f"models/{key}.npz").read_bytes()).hexdigest() for key in models},
        "leakage_clean_sha256": hashlib.sha256((output / "models/leakage_clean.npz").read_bytes()).hexdigest(),
        "training_ids": list(range(11)), "selection": "Action validation only; all clean weights frozen before test observations",
    })
    write_json(output / "training_histories.json", histories)
    print("All clean weights frozen. Evaluating the three genuinely new tracks.", flush=True)
    scenarios = {item["id"]: corner_scenarios(tracks[item["id"]]) for item in catalog}
    ablations, comparisons, robust = [], [], []
    for key, model in models.items():
        mask, seed = key.rsplit("_", 1)
        for item in catalog:
            for scenario in scenarios[item["id"]]:
                result, _ = wall_simulate(tracks[item["id"]], config, policy(model, config, mask=mask), scenario)
                ablations.append({"model": key, "mask": mask, "seed": int(seed), "track_id": item["id"],
                                  "split": item["split"], **result})
    conditions = [{"name": "Clean", "noise": 0, "delay": 0, "grip": 1},
                  {"name": "Noisy observations", "noise": 1, "delay": 0, "grip": 1},
                  {"name": "0.5 s observation delay", "noise": 0, "delay": 5, "grip": 1},
                  {"name": "65% grip envelope", "noise": 0, "delay": 0, "grip": .65},
                  {"name": "Combined shift", "noise": .5, "delay": 3, "grip": .75}]
    saved_traces = {}
    for item in catalog:
        for scenario_index, canonical in enumerate(scenarios[item["id"]]):
            for seed in (401, 402, 403):
                initial_rng = np.random.default_rng(seed)
                scenario = {**canonical, "lateral": canonical["lateral"] + float(initial_rng.uniform(-.15, .15)),
                            "heading_error": canonical["heading_error"] + float(initial_rng.uniform(-.01, .01)),
                            "speed": canonical["speed"] + float(initial_rng.uniform(-.2, .2))}
                for label, driver in (("rules", rule), ("expert", None), ("pretrained", policy(pretrained, config, seed))):
                    result, trace = wall_simulate(tracks[item["id"]], config, driver, scenario)
                    comparisons.append({"controller": label, "seed": seed, "track_id": item["id"],
                                        "scenario_index": scenario_index, "split": item["split"], **result})
                    if item["id"] in (0, 1, 13) and scenario_index == 3 and seed == 401:
                        saved_traces[f"comparison_{item['id']}_{label}"] = trace
                for condition in conditions:
                    shifted = replace(config, max_braking=config.max_braking * condition["grip"],
                                      max_lateral_acceleration=config.max_lateral_acceleration * condition["grip"])
                    driver = policy(pretrained, config, seed, noise=condition["noise"], delay=condition["delay"])
                    result, trace = wall_simulate(tracks[item["id"]], shifted, driver, scenario)
                    robust.append({"condition": condition["name"], "settings": condition, "seed": seed,
                                   "track_id": item["id"], "scenario_index": scenario_index, "split": item["split"], **result})
                    if item["id"] == 0 and scenario_index == 3 and seed == 401:
                        saved_traces[f"robust_{condition['name']}"] = trace
    np.savez_compressed(output / "data/reference_traces.npz", **saved_traces)
    test_x, test_y, test_ids = [], [], []
    test_rng = np.random.default_rng(315)
    for item in catalog[13:]:
        environment = WallEnvironment(tracks[item["id"]], config)
        for _ in range(200):
            state = environment.make_state(float(test_rng.uniform()), float(test_rng.uniform(-6, 6)),
                                           float(test_rng.uniform(-.25, .25)), float(test_rng.uniform(6, 18)))
            features, location = environment.observe(state)
            test_x.append(features)
            test_y.append(environment.teacher(state, location))
            test_ids.append(item["id"])
    tx, ty = np.asarray(test_x), np.asarray(test_y)
    clean_test_mse = float(np.mean((clean_leak.predict(tx) - ty) ** 2 * [4, 1]))
    contaminated, contaminated_history = fit(np.vstack([x, tx]), np.vstack([y, ty]), vx, vy, 314,
                                             "full", output, "leakage_contaminated")
    contaminated_test_mse = float(np.mean((contaminated.predict(tx) - ty) ** 2 * [4, 1]))
    leakage = {"clean_test_mse": clean_test_mse, "contaminated_test_mse": contaminated_test_mse,
               "clean_train_mse_mislabeled_as_test": float(np.mean((clean_leak.predict(x) - y) ** 2 * [4, 1])),
               "contaminated_test_observations": len(tx), "valid_generalization_claim": False,
               "clean_history": leak_history, "contaminated_history": contaminated_history}
    np.savez_compressed(output / "data/pristine_test_observations.npz", features=tx, targets=ty, track_id=test_ids)
    test_rows = [row(tx[i], ty[i], f"sealed-test-{i}", "test", test_ids[i], "test label: explicit leakage lesson")
                 for i in range(len(tx))]
    golden = []
    for track_id, scenario_index in ((0, 0), (0, 2), (1, 3), (13, 0), (14, 2)):
        environment = WallEnvironment(tracks[track_id], config)
        scenario = scenarios[track_id][scenario_index]
        environment.state = environment.make_state(scenario["fraction"], scenario["lateral"], scenario["heading_error"], scenario["speed"])
        before = environment.state
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
    batch = [row(x[i], y[i], f"golden-{i}", "train", train_ids[i], "parity fixture") for i in range(8)]
    loss_before = gradient_model.train_batch(x[:8], y[:8], .001)
    write_json(output / "data/golden.json", {"physics": golden, "training": {
        "before": before_model, "after": export_model(gradient_model), "batch": batch, "loss": loss_before}})
    write_json(output / "ablations.json", ablations)
    write_json(output / "controller_comparisons.json", comparisons)
    write_json(output / "robustness.json", robust)
    write_json(output / "leakage.json", leakage)
    data = {"config": asdict(config), "features": FEATURE_NAMES,
            "tracks": [export_track(item, tracks[item["id"]]) for item in catalog],
            "scenarios": scenarios, "pretrained": export_model(pretrained), "starter": starter,
            "testRows": test_rows, "models": {key: export_model(model, key.rsplit("_", 1)[0]) for key, model in models.items()},
            "histories": histories, "ablations": ablations, "comparisons": comparisons, "robustness": robust,
            "conditions": conditions, "leakage": leakage, "golden": golden}
    write_json(output / "data/lab.json", data)
    write_json(output / "manifest.json", {
        "modules": 8, "training_tracks": list(range(11)), "legacy_holdouts": [11, 12],
        "pristine_test_tracks": [13, 14, 15], "training_samples": len(x), "validation_samples": len(vx),
        "ablation_models": 12, "updates_per_model": 256, "ablation_seeds": [301, 302, 303],
        "ablation_rollouts": len(ablations), "controller_rollouts": len(comparisons),
        "robustness_rollouts": len(robust), "source_model_sha256": hashlib.sha256((base / "models/final.npz").read_bytes()).hexdigest(),
        "pristine_test_protocol": "All clean model weights and selection frozen before pristine policy evaluation or test labels",
        "leakage_demo": "Separate intentionally contaminated model; never included in clean benchmark results",
    })
    print(f"Prepared eight modules: {len(ablations)} ablation, {len(comparisons)} controller and {len(robust)} robustness rollouts.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="student_artifacts")
    parser.add_argument("--base", default="wall_artifacts")
    args = parser.parse_args()
    prepare(args.output, args.base)
