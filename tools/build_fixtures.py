"""Build tests/fixtures/engine_fixture.json: NumPy/SciPy reference values for the JavaScript engine.

`compute(inputs)` is deterministic; tests/test_fixture_sync.py recomputes it from the committed
inputs, so a change to the Python simulator that would make the fixture stale fails CI.
Regenerating the inputs needs the saved wall artifacts (not committed):

    python tools/build_fixtures.py --pretrained student_artifacts/models/pretrained.npz
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driving import NeuralDriver, learning_rate_at, tracks_from_catalog  # noqa: E402
from race_sim import complex_circuits  # noqa: E402
from student_experiments import export_model, export_track, masked, policy, rule  # noqa: E402
from wall_sim import WALL_CONFIG, WallEnvironment, corner_scenarios, wall_simulate  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/engine_fixture.json"
TRACK_IDS = (1, 3)
WILSON_CASES = [(k, n) for n in (1, 3, 12, 36, 44) for k in range(n + 1)] + [(82, 132), (6, 12), (29, 44)]
SIGN_CASES = [(0, 0), (9, 1), (5, 5), (12, 3), (0, 7), (30, 22)]
SCHEDULE_CASES = [(s, 0.004, u, t) for s in ("constant", "linear", "cosine", "step")
                  for u, t in ((1, 90), (2, 90), (31, 90), (46, 90), (61, 90), (90, 90), (1, 1))]
TRAINING_CASES = [
    {"name": "defaults", "updates": 3, "rate": 0.001, "weight_decay": 0.0, "max_grad_norm": None},
    {"name": "weight decay", "updates": 3, "rate": 0.004, "weight_decay": 0.1, "max_grad_norm": None},
    {"name": "gradient clipping", "updates": 4, "rate": 0.002, "weight_decay": 0.0, "max_grad_norm": 0.05},
    {"name": "decay and clipping", "updates": 4, "rate": 0.003, "weight_decay": 0.05, "max_grad_norm": 0.05},
]
KNN_K = 5
NOISE_SEED = 7


def rounded(values, digits=12):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def weighted_loss(prediction, targets):
    return float(np.mean((prediction - targets) ** 2 * [4.0, 1.0]))


def load_network(exported, hidden_sizes):
    driver = NeuralDriver(0, input_size=18, hidden_sizes=hidden_sizes)
    for i in range(3):
        driver.weights[i][...] = exported["weights"][i]
        driver.biases[i][...] = exported["biases"][i]
    return driver


def build_inputs(pretrained_path):
    """Pick two GP-X circuits, the saved pretrained driver and a fixed pool of expert-labelled states."""
    catalog, tracks = complex_circuits(WALL_CONFIG)
    chosen = [item for item in catalog if item["id"] in TRACK_IDS]
    pretrained = NeuralDriver.load(pretrained_path)
    rng = np.random.default_rng(2031)
    rows = []
    for item in chosen:
        environment = WallEnvironment(tracks[item["id"]], WALL_CONFIG)
        for scenario in corner_scenarios(tracks[item["id"]]):
            environment.state = environment.make_state(scenario["fraction"], scenario["lateral"],
                                                      scenario["heading_error"], scenario["speed"])
            for step in range(60):
                features, location = environment.observe()
                target = environment.teacher(location=location)
                if step % 5 == 0:
                    rows.append({"features": features.tolist(), "targets": target.tolist()})
                if environment.step(target)[1]:
                    break
    keep = rng.choice(len(rows), size=min(60, len(rows)), replace=False)
    return {
        "tracks": [{"id": item["id"], "name": item["name"], "knots": item["knots"]} for item in chosen],
        "pretrained": export_model(pretrained),
        "rows": [rows[i] for i in rng.permutation(keep)],
        "training_start": export_model(NeuralDriver(99, input_size=18, hidden_sizes=(24, 16))),
        "bootstrap": {"values": rng.normal(0.5, 0.2, 10).tolist(),
                      "resamples": rng.integers(0, 10, (200, 10)).tolist()},
        "permutations": [[rng.permutation(30).tolist() for _ in range(2)] for _ in range(3)],
    }


def trajectory_record(label, track_id, scenario_index, metric, trace):
    return {"label": label, "track_id": track_id, "scenario": scenario_index, "end_reason": metric["end_reason"],
            "success": metric["success"], "length": len(trace),
            "x": rounded(trace[:, 1]), "y": rounded(trace[:, 2]), "heading": rounded(trace[:, 3]),
            "speed": rounded(trace[:, 4]), "lateral": rounded(trace[:, 6]), "brake": rounded(trace[:, 9]),
            "gear": trace[:, 10].astype(int).tolist(), "margin": rounded(trace[:, 17])}


def compute(inputs):
    config = WALL_CONFIG
    tracks = tracks_from_catalog(config, inputs["tracks"])
    pretrained = load_network(inputs["pretrained"], (48, 32))
    grip = replace(config, max_braking=config.max_braking * 0.65,
                   max_lateral_acceleration=config.max_lateral_acceleration * 0.65)
    exported, trajectories = [], []
    for item in inputs["tracks"]:
        track = tracks[item["id"]]
        scenarios = corner_scenarios(track)
        exported.append({**export_track({**item, "split": "fixture"}, track), "scenarios": scenarios})
        for index in (0, 2):
            runs = [("expert", config, None), ("rules", config, rule), ("pretrained", config, policy(pretrained, config)),
                    ("delay 3", config, policy(pretrained, config, delay=3)),
                    ("grip 0.65", grip, policy(pretrained, grip))]
            for label, run_config, driver in runs:
                metric, trace = wall_simulate(track, run_config, driver, scenarios[index])
                trajectories.append(trajectory_record(label, item["id"], index, metric, trace))
            metric, trace = wall_simulate(track, config, policy(pretrained, config, seed=NOISE_SEED, noise=1.0), scenarios[index])
            trajectories.append(trajectory_record("noise 1", item["id"], index, metric, trace))
    features = np.array([row["features"] for row in inputs["rows"]])
    targets = np.array([row["targets"] for row in inputs["rows"]])
    training = []
    for case in TRAINING_CASES:
        driver = load_network(inputs["training_start"], (24, 16))
        losses, norms, clipped = [], [], []
        for _ in range(case["updates"]):
            losses.append(driver.train_batch(features[:16], targets[:16], case["rate"],
                                             weight_decay=case["weight_decay"], max_grad_norm=case["max_grad_norm"]))
            norms.append(driver.last_gradient_norm)
            clipped.append(bool(driver.last_clipped))
        training.append({**case, "rows": 16, "losses": rounded(losses), "gradient_norms": rounded(norms),
                         "clipped": clipped, "after": export_model(driver)})
    wilson = []
    for k, n in WILSON_CASES:
        interval = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        wilson.append({"k": k, "n": n, "lo": float(interval.low), "hi": float(interval.high)})
    values = np.array(inputs["bootstrap"]["values"])
    means = np.array([values[idx].mean() for idx in inputs["bootstrap"]["resamples"]])
    bootstrap = {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)),
                 "hi": float(np.percentile(means, 97.5)),
                 "quantiles": {str(q): float(np.percentile(means, 100 * q)) for q in (0.025, 0.1, 0.5, 0.9, 0.975)}}
    sign = [{"wins": w, "losses": lo, "p": 1.0 if w + lo == 0 else float(stats.binomtest(w, w + lo, 0.5).pvalue)}
            for w, lo in SIGN_CASES]
    schedules = [{"schedule": s, "base": b, "update": u, "total": t, "rate": learning_rate_at(s, b, u, t)}
                 for s, b, u, t in SCHEDULE_CASES]
    ridge = []
    for mask, penalty in (("full", 0.01), ("no_heading", 0.1), ("no_future_geometry", 1.0)):
        augmented = np.c_[masked(features, mask), np.ones(len(features))]
        gram = augmented.T @ augmented / len(features) + penalty * np.diag([1.0] * 18 + [0.0])
        weights = np.linalg.solve(gram, augmented.T @ targets / len(features))
        queries = masked(features[:12], mask)
        prediction = np.clip(np.c_[queries, np.ones(len(queries))] @ weights, -1, 1)
        ridge.append({"mask": mask, "lambda": penalty, "weights": rounded(weights), "predictions": rounded(prediction),
                      "loss": weighted_loss(np.clip(augmented @ weights, -1, 1), targets)})
    knn = []
    for mask in ("full", "no_heading"):
        points = masked(features[12:], mask)
        queries = masked(features[:12], mask) + 0.013
        predictions = []
        for query in queries:
            nearest = np.argsort(((points - query) ** 2).sum(axis=1), kind="stable")[:KNN_K]
            predictions.append(targets[12:][nearest].mean(axis=0))
        knn.append({"mask": mask, "k": KNN_K, "predictions": rounded(predictions)})
    used_x, used_y = features[:30], targets[:30]
    baseline = weighted_loss(pretrained.predict(used_x), used_y)
    permutation = []
    for group, orders in zip([[0], [1, 2], list(range(5, 13))], inputs["permutations"]):
        increases = []
        for order in orders:
            shuffled = used_x.copy()
            shuffled[:, group] = used_x[np.array(order)][:, group]
            increases.append(weighted_loss(pretrained.predict(shuffled), used_y) - baseline)
        permutation.append({"indices": group, "increases": rounded(increases)})
    return {
        "config": asdict(config), "tracks": exported, "trajectories": trajectories, "training": training,
        "noise_seed": NOISE_SEED, "noise_draws": np.random.default_rng(NOISE_SEED).normal(size=(400, 3)).tolist(),
        "wilson": wilson, "bootstrap": bootstrap, "sign_test": sign, "schedules": schedules, "ridge": ridge, "knn": knn,
        "permutation": {"baseline": baseline, "groups": permutation, "rows": 30},
        "pretrained_loss": weighted_loss(pretrained.predict(features), targets),
    }


def build(pretrained_path):
    inputs = build_inputs(pretrained_path)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"inputs": inputs, "expected": compute(inputs)}, allow_nan=False,
                                  separators=(",", ":")) + "\n")
    print(f"Wrote {FIXTURE} ({FIXTURE.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", default="student_artifacts/models/pretrained.npz")
    build(parser.parse_args().pretrained)
