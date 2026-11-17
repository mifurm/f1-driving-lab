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
        "ppo": ppo_inputs(),
    }


def trajectory_record(label, track_id, scenario_index, metric, trace):
    return {"label": label, "track_id": track_id, "scenario": scenario_index, "end_reason": metric["end_reason"],
            "success": metric["success"], "length": len(trace),
            "x": rounded(trace[:, 1]), "y": rounded(trace[:, 2]), "heading": rounded(trace[:, 3]),
            "speed": rounded(trace[:, 4]), "lateral": rounded(trace[:, 6]), "brake": rounded(trace[:, 9]),
            "gear": trace[:, 10].astype(int).tolist(), "margin": rounded(trace[:, 17])}


PPO_SHIFTS = [0.0, 0.6, -0.6, 0.02]  # old-policy log-probability offsets: ratios well inside and well outside the clip range
PPO_OPTIONS = {"clip_range": 0.2, "value_coef": 0.5, "entropy_coef": 0.01}


def ppo_inputs():
    """A fixed PPO problem for the JavaScript trainer: random policy and critic networks, a batch of drives and two advantage-estimation trajectories."""
    rng = np.random.default_rng(2077)

    def network(sizes):
        return [{"weights": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)).tolist(), "biases": rng.normal(0, 0.1, n_out).tolist()}
                for n_in, n_out in zip(sizes, sizes[1:])]

    n = 12
    return {"actor": network([18, 24, 16, 2]), "critic": network([18, 24, 16, 1]), "log_std": [-0.7, -1.2],
            "observations": rng.normal(0, 0.5, (n, 18)).tolist(), "noise": rng.normal(size=(n, 2)).tolist(),
            "shifts": [PPO_SHIFTS[i % 4] for i in range(n)], "advantages": rng.normal(size=n).tolist(), "returns": rng.normal(size=n).tolist(),
            "options": PPO_OPTIONS,
            "gae": [{"rewards": [1.0, 0.0, 2.0, -1.0, 0.5], "values": [0.5, 0.4, 0.3, 0.2, 0.1], "done": [0, 0, 1, 0, 0], "last_value": 0.7, "gamma": 0.9, "lambda": 0.8},
                    {"rewards": rng.normal(size=8).tolist(), "values": rng.normal(size=8).tolist(), "done": [0, 1, 0, 0, 0, 1, 0, 0], "last_value": -0.3, "gamma": 0.99, "lambda": 0.95}]}


def mlp_forward(layers, x, squash_last):
    activations = [x]
    for index, layer in enumerate(layers):
        z = activations[-1] @ np.array(layer["weights"]) + np.array(layer["biases"])
        activations.append(np.tanh(z) if squash_last or index < len(layers) - 1 else z)
    return activations


def mlp_backward(layers, activations, d_out, squash_last):
    gradients, delta = [], d_out
    for index in range(len(layers) - 1, -1, -1):
        if squash_last or index < len(layers) - 1:
            delta = delta * (1 - activations[index + 1] ** 2)
        gradients.append({"weights": rounded(activations[index].T @ delta), "biases": rounded(delta.sum(axis=0))})
        delta = delta @ np.array(layers[index]["weights"]).T
    return gradients[::-1]


def gae_reference(case):
    """Textbook generalised advantage estimation as an explicit sum over the steps that follow, stopping at the end of an episode."""
    rewards, values, done = np.array(case["rewards"]), np.array(case["values"]), np.array(case["done"])
    steps = len(rewards)
    deltas = [rewards[t] + case["gamma"] * (case["last_value"] if t == steps - 1 else values[t + 1]) * (1 - done[t]) - values[t] for t in range(steps)]
    advantages = []
    for t in range(steps):
        total, discount = 0.0, 1.0
        for k in range(t, steps):
            total += discount * deltas[k]
            if done[k]:
                break
            discount *= case["gamma"] * case["lambda"]
        advantages.append(total)
    return {"advantages": rounded(advantages), "returns": rounded(np.array(advantages) + values)}


def ppo_expected(case):
    """PPO loss pieces and their analytic gradients for the fixed problem, as one vectorised NumPy pass (the JavaScript trainer loops over samples)."""
    options, x = case["options"], np.array(case["observations"])
    n, log_std = len(x), np.array(case["log_std"])
    sigma = np.exp(log_std)
    actor, critic = mlp_forward(case["actor"], x, True), mlp_forward(case["critic"], x, False)
    mean, value = actor[-1], critic[-1][:, 0]
    noise = np.array(case["noise"])
    actions = mean + sigma * noise
    logp = (-0.5 * noise ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(axis=1)
    old_logp = logp - np.array(case["shifts"])
    advantages, returns = np.array(case["advantages"]), np.array(case["returns"])
    ratio = np.exp(logp - old_logp)
    low, high = 1 - options["clip_range"], 1 + options["clip_range"]
    surrogate = np.minimum(ratio * advantages, np.clip(ratio, low, high) * advantages)
    # A sample's gradient vanishes once the ratio has moved past the clip range in the direction the advantage rewards.
    active = ~(((advantages > 0) & (ratio > high)) | ((advantages < 0) & (ratio < low)))
    d_logp = np.where(active, -advantages * ratio / n, 0.0)
    d_mean = d_logp[:, None] * noise / sigma
    d_log_std = (d_logp[:, None] * (noise ** 2 - 1)).sum(axis=0) - options["entropy_coef"]
    d_value = options["value_coef"] * (value - returns) / n
    return {"mean": rounded(mean), "value": rounded(value), "actions": rounded(actions), "old_logp": rounded(old_logp),
            "policy_loss": float(-surrogate.mean()), "value_loss": float((0.5 * (value - returns) ** 2).mean()),
            "entropy": float(log_std.sum() + np.log(2 * np.pi) + 1), "kl": float((ratio - 1 - np.log(ratio)).mean()), "clip_fraction": float((~active).mean()),
            "actor_gradients": mlp_backward(case["actor"], actor, d_mean, True), "critic_gradients": mlp_backward(case["critic"], critic, d_value[:, None], False),
            "log_std_gradient": rounded(d_log_std), "gae": [gae_reference(c) for c in case["gae"]]}


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
        "ppo": ppo_expected(inputs["ppo"]),
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
