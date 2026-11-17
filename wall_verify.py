"""Independent checks for wall-learning data, controls, collisions and visuals."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import Config, NeuralDriver, SCENARIOS, Track, write_json
from race_verify import check_telemetry, parse_page
from verify import require
from wall_render import load_attempt
from wall_sim import add_margin_feature, wall_simulate
from wall_train import upgraded_driver


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = Config(**manifest["config"])
    catalog = json.loads((output / "data/circuits.json").read_text())
    scenarios = json.loads((output / "data/scenarios.json").read_text())
    metrics = json.loads((output / "attempt_metrics.json").read_text())
    history = json.loads((output / "history.json").read_text())
    checks = []
    def passed(message):
        checks.append(message)
        print("PASS " + message, flush=True)
    require(manifest["rounds"] == 120 and manifest["updates"] == 7680, "Missing extended training")
    require(len(history) == 121 and len(metrics) == 6292, "Missing fixed attempt histories")
    freeze = json.loads((output / "training_complete.json").read_text())
    selection = json.loads((output / "checkpoint_selection.json").read_text())
    require(not selection["uses_holdout_or_stress_results"], "Test results influenced checkpoint selection")
    require(selection["selected_round"] == manifest["selected_round"], "Selected checkpoint differs from the audit")
    require(hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
            == manifest["final_sha256"] == freeze["sha256"], "Frozen weights changed")
    require(hashlib.sha256((output / "data/base_model.npz").read_bytes()).hexdigest()
            == manifest["base_sha256"], "Original weights changed")
    baseline = upgraded_driver(output / "data/base_model.npz")
    checkpoint_zero = NeuralDriver.load(output / "models/round_000.npz")
    for a, b in zip(baseline.parameters, checkpoint_zero.parameters):
        np.testing.assert_array_equal(a, b)
    passed("Pretrained weights preserved; zero-weight margin/budget inputs and 120 rounds / 7,680 fresh Adam updates")
    with np.load(output / "data/aggregated_recovery.npz") as data:
        x, y, ids, rounds = [data[key] for key in ("features", "targets", "track_id", "round")]
    require(x.shape == (manifest["final_recovery_samples"], 18) and y.shape == (len(x), 2), "Incorrect training shapes")
    require(np.isfinite(x).all() and np.isfinite(y).all() and (np.abs(y) <= 1).all(), "Invalid training samples")
    np.testing.assert_allclose(x, add_margin_feature(x[:, :16], config), atol=1e-14, rtol=0)
    require(set(np.unique(ids)) == set(manifest["training_ids"]), "Unseen examples leaked")
    for number in range(1, 121):
        for track_id in manifest["training_ids"]:
            require(np.count_nonzero((rounds == number) & (ids == track_id)) == config.rollout_steps,
                    "Unbalanced learner collection")
    with np.load(output / "data/anchors.npz") as anchors:
        require(len(anchors["features"]) == 5632 and anchors["features"].shape[1] == 18, "Missing retention anchors")
        require(set(np.unique(anchors["track_id"])) == set(manifest["training_ids"]), "Anchor leakage")
        training_keys = {row.tobytes() for row in x}
        training_keys.update(row.tobytes() for row in anchors["features"])
    with np.load(output / "data/validation.npz") as validation:
        require(len(validation["features"]) == 2200, "Validation missing")
        require(not any(row.tobytes() in training_keys for row in validation["features"]), "Action validation leaked")
    passed(f"{len(x):,} recovery examples and 5,632 anchors; balanced aggregation and disjoint validation/layout holdouts")
    tracks = {item["id"]: Track(config, item["knots"]) for item in catalog}
    final = NeuralDriver.load(output / "models/final.npz")
    for number in range(121):
        model = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        require(model.updates == number * 64, "Checkpoint update count wrong")
        for item in catalog:
            for index, scenario in enumerate(scenarios[str(item["id"])]):
                trace = load_attempt(output, item, number, index)
                require(trace.shape[1] == 23 and np.isfinite(trace).all(), "Invalid wall telemetry")
                check_telemetry(trace[:, :17], config)
                metric = next(row for row in metrics if row["round"] == number
                              and row["track_id"] == item["id"] and row["scenario"] == scenario["name"])
                require(int(np.any(trace[:, 22])) == metric["wall_contacts"], "Incorrect contact count")
                require(abs(np.min(trace[:, 17]) - metric["min_clearance_m"]) < 1e-10, "Incorrect clearance")
                if metric["success"]:
                    require((trace[:, 17] > 0).all() and not np.any(trace[:, 22]), "Successful drive crossed a wall")
                if number in (0, 1, 40, 80, 120):
                    reproduced, replay = wall_simulate(tracks[item["id"]], config, model, scenario)
                    np.testing.assert_array_equal(replay, trace)
                    require(reproduced["success"] == metric["success"], "Replay outcome mismatch")
    passed("All 6,292 fixed attempts have real drivetrain telemetry; sampled checkpoints re-drive exactly with wall geometry")
    with np.load(output / "attempts/full_laps.npz") as data:
        rows = json.loads((output / "full_lap_metrics.json").read_text())
        for item in catalog:
            for index, scenario in enumerate(SCENARIOS):
                result, trace = wall_simulate(tracks[item["id"]], replace(config, max_steps=2600),
                                             final, scenario, goal=tracks[item["id"]].length)
                np.testing.assert_array_equal(trace, data[f"track_{item['id']:02d}_scenario_{index}"])
                require(result["wall_contacts"] == 0 if result["success"] else True, "False full-lap success")
        require(sum(row["success"] for row in rows if row["split"] == "train") == 33, "Full-lap regression failed")
    passed("All 39 full-lap regression trajectories reproduce independently")
    stress = json.loads((output / "stress_metrics.json").read_text())
    require(len(stress) == 416, "Blind stress cases missing")
    with np.load(output / "attempts/stress.npz") as data:
        for item in catalog:
            for label, model in (("previous", baseline), ("final", final)):
                for index in range(16):
                    row = next(row for row in stress if row["track_id"] == item["id"]
                               and row["model"] == label and row["scenario"] == f"Blind stress {index}")
                    result, trace = wall_simulate(tracks[item["id"]], config, model, row["initial"])
                    np.testing.assert_array_equal(trace, data[f"track_{item['id']:02d}_{label}_{index:02d}"])
                    require(result["success"] == row["success"], "Stress outcomes incorrect")
    passed("All 208 paired blind stress scenarios reproduce for both previous and extended models")
    for number in range(1, 121):
        with np.load(output / f"rollouts/round_{number:03d}.npz") as data:
            require(len(data.files) == 11, "Missing actual training episodes")
            for trace in (data[key] for key in data.files):
                check_telemetry(trace[:, :17], config)
                require(np.max(np.linalg.norm(np.diff(trace[:, 1:3], axis=0), axis=1))
                        <= config.max_speed * config.dt + 1e-9, "Reset teleports joined into playback")
    passed("All 1,320 actual first-episode training rollouts are continuous")
    for item in catalog:
        page = parse_page(output / f"tracks/track_{item['id']:02d}.html")
        require(len(page["attempts"]) == 121 and all(len(round_attempts) == 4 for round_attempts in page["attempts"]),
                "Per-track UI is incomplete")
        require(len(page["rollouts"]) == (120 if item["split"] == "train" else 0), "Wrong rollout split")
        require(page["unsafe"][-1][22] == 1, "Scripted wall-contact illustration missing")
        for number, cases in enumerate(page["attempts"]):
            for index, case in enumerate(cases):
                trace = load_attempt(output, item, number, index)
                np.testing.assert_allclose(case["trace"][-1], trace[-1], atol=.0000006, rtol=0)
                require(round(float(np.min(trace[:, 17])), 6) == min(row[17] for row in case["trace"]),
                        "Minimum wall clearance lost during thinning")
        with Image.open(output / f"visuals/track_{item['id']:02d}_corners.gif") as image:
            require(image.n_frames >= 100, "Corner animation missing")
            for frame in range(image.n_frames):
                image.seek(frame)
                image.load()
    require(len(Presentation(output / "training_deck.pptx").slides) == 6, "Updated deck missing")
    browser = json.loads((output / "browser_verification.json").read_text())
    require(browser["passed"] and browser["attempts"] == 6292 and browser["rollouts"] == 1320
            and browser["wall_demos"] == 13 and browser["slides"] == 6, "Browser checks incomplete")
    passed("All thirteen corner animations decode; every browser attempt, rollout, contact demo and six-slide deck verified")
    for item in json.loads((output / "artifact_inventory.json").read_text()):
        path = output / item["path"]
        require(path.stat().st_size == item["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Artifact changed: {path}")
    passed("Artifact sizes and SHA-256 hashes match")
    write_json(output / "verification.json", {"passed": True, "checks": checks,
                                             "fixed_attempts": 6292, "stress_simulations": 416})


def bundle(output):
    output = Path(output).resolve()
    root = Path(__file__).resolve().parent
    destination = root / "f1-wall-corner-learning-lab.zip"
    sources = list(root.glob("*.py")) + [root / name for name in (
        "wall_track.html", "race_track.html", "WALL_RESULTS.md",
        "requirements.txt", "requirements-lock.txt", "package.json", "package-lock.json")]
    sources += list((root / "tests").glob("*.py")) + list((root / "tests").glob("*.mjs"))
    files = sources + [path for path in output.rglob("*") if path.is_file()]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None, "Portable archive failed CRC verification")
    print(f"Verified portable wall/corner bundle: {destination} ({destination.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="wall_artifacts")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    verify(args.output)
    if args.bundle:
        bundle(args.output)
