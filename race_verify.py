"""Verify recorded driving attempts, real drivetrain telemetry and per-track media."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import Config, NeuralDriver, SCENARIOS, Track, write_json
from multi_render import inventory
from race_render import load_attempt
from race_sim import (
    FINAL_DRIVE, GEAR_RATIOS, WHEEL_RADIUS, RaceEnvironment, RaceState,
    race_simulate, telemetry_row,
)
from verify import DataParser, require


def check_telemetry(trace, config):
    require(trace.ndim == 2 and trace.shape[1] == 17 and np.isfinite(trace).all(), "Invalid telemetry matrix")
    np.testing.assert_allclose(np.diff(trace[:, 0]), config.dt, atol=1e-11, rtol=0)
    require((trace[:, 8:10] >= 0).all() and (trace[:, 8:10] <= 1).all(), "Invalid pedals")
    require((trace[:, 4] >= 0).all() and (trace[:, 4] <= config.max_speed).all(), "Invalid speed")
    require(np.isin(trace[:, 16], [0, 1]).all(), "Invalid shift status")
    require(not np.any((trace[:, 8] > 1e-8) & (trace[:, 9] > 1e-8)), "Throttle and brakes conflict")
    require((trace[:, 10] == np.round(trace[:, 10])).all()
            and (trace[:, 10] >= 1).all() and (trace[:, 10] <= 8).all(), "Invalid integer gear")
    require((np.abs(np.diff(trace[:, 10])) <= 1).all(), "Skipped automatic gear")
    expected_rpm = np.clip(
        trace[:, 4] * 60 / (2 * np.pi * WHEEL_RADIUS) * FINAL_DRIVE
        * GEAR_RATIOS[trace[:, 10].astype(int) - 1], 3000, 12000,
    )
    np.testing.assert_allclose(trace[:, 11], expected_rpm, rtol=0, atol=1e-8)
    expected_speed = np.clip(trace[:-1, 4] + trace[1:, 12] * config.dt, 0, config.max_speed)
    np.testing.assert_allclose(trace[1:, 4], expected_speed, rtol=0, atol=1e-11)
    require((trace[trace[:, 9] > 0.01, 12] < 0).all(), "Braking does not cause deceleration")
    require(np.max(np.abs(np.diff(trace[:, 5]))) <= config.steering_rate * config.dt + 1e-10,
            "Steering-rate violation")


def parse_page(path):
    parser = DataParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return json.loads("".join(parser.parts))


def reproduce_rollout(track, config, driver, trace):
    first = trace[0]
    environment = RaceEnvironment(track, config)
    environment.state = RaceState(
        first[1], first[2], first[3], first[4], first[5],
        gear=int(first[10]), rpm=first[11], target_speed=first[4],
    )
    _, location = environment.observe()
    previous, progress = location[0], 0.0
    rebuilt = [telemetry_row(environment, 0, location, progress)]
    for index in range(1, len(trace)):
        features, _ = environment.observe()
        location, _, _ = environment.step(driver.predict(features))
        progress += (location[0] - previous + track.length / 2) % track.length - track.length / 2
        previous = location[0]
        rebuilt.append(telemetry_row(environment, index * config.dt, location, progress))
    np.testing.assert_allclose(rebuilt, trace, atol=1e-10, rtol=0)


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = Config(**manifest["config"])
    catalog = json.loads((output / "data/circuits.json").read_text())
    metrics = json.loads((output / "attempt_metrics.json").read_text())
    records = {(row["round"], row["track_id"]): row for row in metrics}
    checks = []
    def record(message):
        checks.append(message)
        print(f"PASS {message}")
    require(len(catalog) == 13 and len(manifest["training_track_ids"]) == 11, "Missing harder circuits")
    for item in catalog:
        require(item["reversals"] > item["previous_reversals"], "Circuit did not become more technical")
    require(len(metrics) == 13 * 121, "Missing per-track attempt history")
    history = json.loads((output / "history.json").read_text())
    require(len(history) == 121 and manifest["actual_training_rounds"] == 120, "Training incomplete")
    for number, row in enumerate(history):
        require(row["optimizer_updates"] == number * config.updates_per_round, "Missing weight updates")
        require(row["samples"] == manifest["initial_samples"] + number * 11 * config.rollout_steps,
                "Missing or unbalanced aggregation")
    record("All 13 layouts are more technical; 120 rounds and 121 attempts per track recorded")
    with np.load(output / "data/aggregated_training_data.npz") as data:
        x, y, ids, rounds = [data[key] for key in ("features", "targets", "track_id", "round")]
    require(x.shape == (manifest["final_samples"], 16) and y.shape == (len(x), 2), "Incorrect data shapes")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite training data")
    require(set(np.unique(ids)) == set(manifest["training_track_ids"]), "Unseen data leaked into training")
    for track_id in manifest["training_track_ids"]:
        for number in range(1, 121):
            require(np.count_nonzero((ids == track_id) & (rounds == number)) == config.rollout_steps,
                    "Unbalanced per-track training data")
    with np.load(output / "data/validation_data.npz") as validation:
        require(set(np.unique(validation["track_id"])) == set(manifest["training_track_ids"]), "Holdout validation leakage")
        keys = {row.tobytes() for row in x}
        require(not any(row.tobytes() in keys for row in validation["features"]), "Validation leaked into training")
    record(f"{len(x):,} examples; equal per-track aggregation; action validation and unseen tracks disjoint")
    model_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    freeze = json.loads((output / "training_complete.json").read_text())
    require(model_hash == freeze["final_model_sha256"] == manifest["final_model_sha256"], "Frozen weights changed")
    require(freeze["holdouts_not_yet_evaluated"] and manifest["holdouts_evaluated_after_freeze"], "Unseen test protocol violated")
    tracks = {item["id"]: Track(config, item["knots"]) for item in catalog}
    for number in range(121):
        model = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        require(model.updates == number * config.updates_per_round, "Incorrect saved model update count")
        for item in catalog:
            trace = load_attempt(output, item, number)
            check_telemetry(trace, config)
            expected = records[(number, item["id"])]
            require(abs(float(np.clip(trace[-1, 7], 0, 1)) - expected["progress"]) < 1e-10,
                    "Incorrect progress in recorded attempt")
            require(int(np.count_nonzero(np.diff(trace[:, 10]))) == expected["gear_changes"], "Fake shift count")
    record("All 1,573 attempts have valid speed integration, real brake deceleration, RPM and integer shifts")
    final_model = NeuralDriver.load(output / "models/final.npz")
    final_checks = json.loads((output / "final_checks.json").read_text())
    completed = {"train": 0, "holdout": 0}
    with np.load(output / "attempts/final_checks.npz") as saved:
        for item in catalog:
            for index, scenario in enumerate(SCENARIOS):
                result, trace = race_simulate(tracks[item["id"]], config, final_model, scenario)
                expected = next(row for row in final_checks
                                if row["track_id"] == item["id"] and row["scenario"] == scenario["name"])
                completed[item["split"]] += result["completed"]
                np.testing.assert_array_equal(trace, saved[f"track_{item['id']:02d}_scenario_{index}"])
                for key, value in result.items():
                    if isinstance(value, float):
                        require(abs(value - expected[key]) < 1e-10, f"Non-reproducible final metric {key}")
                    else:
                        require(value == expected[key], f"Non-reproducible final outcome {key}")
                require(result["braking_time_s"] > 0 and result["gear_changes"] > 0,
                        "Final driver lacks real braking/gear activity")
    require(completed["train"] == 33 and completed["holdout"] == manifest["final_holdout_starts_completed"],
            "Final completion was misreported")
    record(f"Final weights independently re-driven: {completed['train']}/33 train and {completed['holdout']}/6 unseen starts")
    rollout_metrics = json.loads((output / "rollout_metrics.json").read_text())
    require(len(rollout_metrics) == 1320, "Missing actual training episodes")
    for number in range(1, 121):
        with np.load(output / f"rollouts/round_{number:03d}.npz") as data:
            require(len(data.files) == 11, "Missing per-track data-collection recordings")
            for track_id in manifest["training_track_ids"]:
                trace = data[f"track_{track_id:02d}"]
                check_telemetry(trace, config)
                distances = np.linalg.norm(np.diff(trace[:, 1:3], axis=0), axis=1)
                require(np.max(distances) <= config.max_speed * config.dt + 1e-8, "Reset teleport joined into a drive")
                if number in (1, 40, 80, 120):
                    reproduce_rollout(tracks[track_id], config,
                                      NeuralDriver.load(output / f"models/round_{number-1:03d}.npz"), trace)
    record("All 1,320 training rollouts are continuous; sampled episodes reproduce prior-round policy controls")
    for item in catalog:
        page = parse_page(output / f"tracks/track_{item['id']:02d}.html")
        require(len(page["attempts"]) == 121, "Per-track replay is missing attempts")
        require(len(page["rollouts"]) == (120 if item["split"] == "train" else 0), "Training footage split is wrong")
        for number, attempt in enumerate(page["attempts"]):
            trace = load_attempt(output, item, number)
            np.testing.assert_allclose(attempt["trace"][-1], trace[-1], atol=0.000006, rtol=0)
            require(attempt["metric"] == records[(number, item["id"])], "Wrong replay measurements")
            times = {row[0] for row in attempt["trace"]}
            changes = np.flatnonzero(
                (np.diff(trace[:, 10]) != 0) | (np.diff(trace[:, 16]) != 0)
                | (np.diff(trace[:, 9] > 0.02) != 0),
            )
            for change in np.r_[changes, changes + 1, np.argmax(trace[:, 4]), np.argmax(trace[:, 9])]:
                require(round(float(trace[change, 0]), 5) in times, "A transition or telemetry peak was lost")
        with Image.open(output / f"visuals/track_{item['id']:02d}_attempts.gif") as image:
            require(image.n_frames > 100, "Missing moving-car animation")
            for frame in range(image.n_frames):
                image.seek(frame)
                image.load()
        for suffix in ("learning", "telemetry", "comparison"):
            with Image.open(output / f"visuals/track_{item['id']:02d}_{suffix}.png") as image:
                image.verify()
    require(len(Presentation(output / "training_deck.pptx").slides) == 8, "PowerPoint is incomplete")
    record("Every circuit has all-round playback, retained pedal/shift transitions and peaks, charts and a decoded animation; eight-slide PowerPoint opens")
    browser_file = output / "browser_verification.json"
    require(browser_file.exists(), "Run the race browser checks before final artifact verification")
    browser = json.loads(browser_file.read_text())
    require(browser["passed"] and browser["attempts_tested"] == 1573
            and browser["training_rollouts_tested"] == 1320
            and browser["brake_lights_tested"] == 13
            and browser["presentation_slides_tested"] == 8, "Browser verification incomplete")
    record("Browser verified every attempt/rollout, all controls, real brake-light pixels and per-track telemetry")
    for entry in json.loads((output / "artifact_inventory.json").read_text()):
        path = output / entry["path"]
        require(path.stat().st_size == entry["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Artifact changed: {path}")
    record("All generated artifact sizes and SHA-256 hashes match")
    write_json(output / "verification.json", {"passed": True, "checks": checks,
                                            "final_starts_completed": completed, "fixed_attempts": len(metrics)})


def bundle(output):
    output = Path(output).resolve()
    root = Path(__file__).resolve().parent
    require(json.loads((output / "verification.json").read_text())["passed"], "Run not verified")
    require(json.loads((output / "browser_verification.json").read_text())["passed"], "Browser not verified")
    destination = root / "f1-gpx-telemetry-lab.zip"
    sources = list(root.glob("*.py")) + [
        root / name for name in (
            "race_track.html", "RACE_PLAN.md", "RACE_RESULTS.md",
            "requirements.txt", "requirements-lock.txt", "package.json", "package-lock.json",
        )
    ] + list((root / "tests").glob("*.py")) + list((root / "tests").glob("*.mjs"))
    paths = sources + [path for path in output.rglob("*") if path.is_file()]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None, "GP-X archive failed CRC verification")
        for item in json.loads((output / "data/circuits.json").read_text()):
            archive.getinfo(str(Path(root.name) / output.relative_to(root) / f"tracks/track_{item['id']:02d}.html"))
    print(f"Verified GP-X bundle: {destination} ({destination.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="race_artifacts")
    parser.add_argument("--bundle", action="store_true")
    arguments = parser.parse_args()
    verify(arguments.output)
    if arguments.bundle:
        bundle(arguments.output)
