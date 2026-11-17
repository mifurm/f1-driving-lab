"""Verify the saved experiment, model replay, data integrity, and visual artifacts."""

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import Config, NeuralDriver, Track, simulate, write_json


class DataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reading = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "experiment-data":
            self.reading = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.reading = False

    def handle_data(self, data):
        if self.reading:
            self.parts.append(data)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = Config(**manifest["config"])
    checks = []
    def record(name):
        checks.append(name)
        print(f"PASS {name}")
    require(config.rounds >= 100, "Fewer than 100 training rounds")
    with (output / "metrics.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    require(len(history) == config.rounds + 1, "Missing evaluation rounds")
    require([int(row["round"]) for row in history] == list(range(config.rounds + 1)),
            "Training rounds are not consecutive")
    for number, row in enumerate(history):
        require(int(row["optimizer_updates"]) == number * config.updates_per_round,
                f"No evidence of expected optimizer updates in round {number}")
        require(int(row["training_samples"]) == manifest["initial_samples"] + number * config.rollout_steps,
                f"Wrong aggregation count in round {number}")
    record("120 actual training rounds / 2,880 optimizer updates / 121 evaluations")
    with np.load(output / "data/aggregated_training_data.npz") as data:
        features, targets, sources, rounds = [data[key] for key in ("features", "targets", "source", "round")]
    require(features.shape == (manifest["final_samples"], 13), "Incorrect feature dimensions")
    require(targets.shape == (manifest["final_samples"], 2), "Incorrect target dimensions")
    require(np.isfinite(features).all() and np.isfinite(targets).all(), "Non-finite training data")
    require(np.max(np.abs(targets)) <= 1, "Invalid action labels")
    require(np.count_nonzero(sources == 0) == config.initial_recovery_samples, "Recovery data missing")
    require(np.count_nonzero(sources == 1) == manifest["initial_samples"] - config.initial_recovery_samples,
            "Teacher demonstrations missing")
    require(np.count_nonzero(sources == 2) == config.rounds * config.rollout_steps, "DAgger samples missing")
    for number in range(1, config.rounds + 1):
        require(np.count_nonzero(rounds == number) == config.rollout_steps, f"Missing data for round {number}")
    with np.load(output / "data/initial_training_data.npz") as initial:
        np.testing.assert_array_equal(initial["features"], features[:manifest["initial_samples"]])
    with np.load(output / "data/validation_data.npz") as validation:
        training_keys = {row.tobytes() for row in features}
        require(not any(row.tobytes() in training_keys for row in validation["features"]),
                "Exact validation observations leaked into training")
    record("49,525 labeled examples; every aggregation round present; validation disjoint")
    track = Track(config)
    driver = NeuralDriver.load(output / "models/final.npz")
    require(driver.updates == manifest["actual_optimizer_updates"], "Saved final model has wrong update count")
    details = json.loads((output / "scenario_metrics.json").read_text())
    require(len(details) == 3 * (config.rounds + 1), "Missing scenario-level measurements")
    for index, scenario in enumerate(manifest["fixed_test_scenarios"]):
        result, trace = simulate(track, config, driver, scenario)
        expected = next(row for row in details if row["round"] == config.rounds and row["scenario"] == scenario["name"])
        require(result["completed"] == 1, f"Final checkpoint cannot complete {scenario['name']}")
        for key, value in result.items():
            if isinstance(value, float):
                require(abs(value - expected[key]) < 1e-10, f"Non-reproducible {key}")
            else:
                require(value == expected[key], f"Non-reproducible {key}")
        with np.load(output / f"trajectories/round_{config.rounds:03d}.npz") as stored:
            np.testing.assert_array_equal(trace, stored[f"scenario_{index}"])
    require(manifest["final"]["mean_progress_pct"] > manifest["baseline"]["mean_progress_pct"],
            "No progress improvement")
    record("Final model independently replayed: all 3 laps reproduce saved traces exactly")
    for number in manifest["checkpoints"]:
        model = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        require(model.updates == number * config.updates_per_round, f"Bad checkpoint {number}")
    for number in range(config.rounds + 1):
        with np.load(output / f"trajectories/round_{number:03d}.npz") as stored:
            for index in range(3):
                trace = stored[f"scenario_{index}"]
                require(trace.shape[1] == 8 and len(trace) > 1, "Invalid trajectory")
                require(np.isfinite(trace).all(), "Non-finite trajectory")
                require((np.diff(trace[:, 0]) > 0).all(), "Non-increasing trajectory time")
    record("All checkpoints and all 363 saved scenario trajectories are valid")
    parser = DataParser()
    parser.feed((output / "replay.html").read_text())
    replay = json.loads("".join(parser.parts))
    require(len(replay["traces"]) == config.rounds + 1, "Interactive replay missing rounds")
    for number, scenarios in enumerate(replay["traces"]):
        require(len(scenarios) == 3, "Interactive replay missing scenario")
        for index, trace in enumerate(scenarios):
            with np.load(output / f"trajectories/round_{number:03d}.npz") as stored:
                np.testing.assert_allclose(trace[-1], stored[f"scenario_{index}"][-1], atol=0.000051, rtol=0)
    record("Offline replay contains all 121 rounds, all test starts, and correct terminal states")
    gif_count = 0
    for path in (output / "visuals").iterdir():
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
        elif path.suffix == ".gif":
            with Image.open(path) as image:
                require(image.n_frames > 1, f"Not an animation: {path}")
                for frame in range(image.n_frames):
                    image.seek(frame)
                    image.load()
            gif_count += 1
    require(gif_count == 8, f"Expected 8 animations, found {gif_count}")
    record("All chart PNGs and all frames of 8 driving GIFs decode successfully")
    with zipfile.ZipFile(output / "training_deck.pptx") as archive:
        require(archive.testzip() is None, "Invalid PowerPoint zip package")
    deck = Presentation(output / "training_deck.pptx")
    require(len(deck.slides) == 15, "Incorrect presentation slide count")
    for slide in deck.slides:
        require(slide.notes_slide.notes_text_frame.text.strip(), "Missing speaker notes")
        for shape in slide.shapes:
            require(shape.left >= 0 and shape.top >= 0
                    and shape.left + shape.width <= deck.slide_width
                    and shape.top + shape.height <= deck.slide_height,
                    "Shape extends beyond slide")
    html_deck = (output / "training_deck.html").read_text()
    require(html_deck.count('<section class="slide') == 15, "HTML deck missing slides")
    record("15-slide PowerPoint and HTML decks; speaker notes and slide bounds verified")
    inventory = json.loads((output / "artifact_inventory.json").read_text())
    for item in inventory:
        path = output / item["path"]
        require(path.stat().st_size == item["bytes"], f"Size mismatch: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Hash mismatch: {path}")
    record("Artifact inventory SHA-256 hashes match all generated deliverables")
    write_json(output / "verification.json", {
        "passed": True, "checks": checks,
        "actual_training_rounds": config.rounds, "actual_optimizer_updates": driver.updates,
        "final_replayed_scenarios_completed": 3,
    })
    print(f"All {len(checks)} verification groups passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts")
    arguments = parser.parse_args()
    verify(arguments.output)
