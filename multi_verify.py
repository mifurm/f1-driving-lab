"""Audit the multi-track run and optionally package both completed experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import Config, NeuralDriver, simulate, tracks_from_catalog, write_json
from multi_render import index_details, load_round, trace_key
from multitrack import MultiTrackEnvironment
from verify import DataParser, require


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = Config(**manifest["config"])
    history = json.loads((output / "history.json").read_text())
    catalog = json.loads((output / "data/circuit_catalog.json").read_text())
    details = json.loads((output / "scenario_metrics.json").read_text())
    indexed = index_details(details)
    checks = []
    def record(message):
        checks.append(message)
        print(f"PASS {message}")
    require(len(manifest["training_track_ids"]) == 11 and manifest["new_training_tracks"] == 10,
            "Ten additional training circuits are not present")
    require(len(manifest["holdout_track_ids"]) == 2, "Missing unseen test circuits")
    require(not set(manifest["training_track_ids"]) & set(manifest["holdout_track_ids"]), "Track split leakage")
    require(len(catalog) == 13 and len({item["name"] for item in catalog}) == 13, "Duplicate/missing layouts")
    require(len(history) == config.rounds + 1 and config.rounds >= 100, "Incomplete training schedule")
    for number, row in enumerate(history):
        require(row["round"] == number, "Non-consecutive training history")
        require(row["optimizer_updates"] == number * config.updates_per_round, "Missing actual optimizer updates")
        require(row["samples"] == manifest["initial_samples"] + number * 11 * config.rollout_steps,
                "Unequal or missing per-track aggregation")
    record("11 training circuits + 2 disjoint unseen circuits; 120 rounds / 7,680 weight updates")
    with np.load(output / "data/aggregated_training_data.npz") as data:
        x, y, track_ids, rounds, sources = [data[key] for key in ("features", "targets", "track_id", "round", "source")]
    require(x.shape == (manifest["final_samples"], 16) and y.shape == (len(x), 2), "Incorrect dataset dimensions")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite dataset")
    require(np.max(np.abs(y)) <= 1, "Out-of-range action labels")
    require(set(np.unique(track_ids)) == set(manifest["training_track_ids"]), "Holdout examples leaked into training")
    for track_id in manifest["training_track_ids"]:
        require(np.count_nonzero((track_ids == track_id) & (sources == 0)) == config.initial_recovery_samples,
                f"Missing recovery examples on track {track_id}")
        for number in range(1, config.rounds + 1):
            require(np.count_nonzero((track_ids == track_id) & (rounds == number)) == config.rollout_steps,
                    f"Unbalanced data on track {track_id}, round {number}")
    with np.load(output / "data/initial_training_data.npz") as initial:
        np.testing.assert_array_equal(initial["features"], x[:manifest["initial_samples"]])
    with np.load(output / "data/validation_data.npz") as validation:
        require(set(np.unique(validation["track_id"])) == set(manifest["training_track_ids"]),
                "Unseen data leaked into action validation")
        training_keys = {row.tobytes() for row in x}
        require(not any(row.tobytes() in training_keys for row in validation["features"]),
                "Action-validation observations leaked into training")
    record(f"{len(x):,} labeled examples; every track/round balanced; validation and unseen data disjoint")
    freeze = json.loads((output / "training_complete.json").read_text())
    final_hash = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    require(final_hash == freeze["final_model_sha256"] == manifest["final_model_sha256"], "Frozen weights changed")
    require(manifest["heldout_evaluation_after_training"], "Unseen tests were not post-training")
    require(freeze["holdout_evaluation_started"] is False, "Training freeze was not recorded before unseen tests")
    require(manifest["best_round_by_training_completion_then_accuracy"] in range(config.rounds + 1),
            "Invalid training-selected checkpoint")
    record("Final model freeze recorded before unseen testing; frozen model hash unchanged")
    tracks = tracks_from_catalog(config, catalog)
    driver = NeuralDriver.load(output / "models/final.npz")
    require(driver.weights[0].shape[0] == 16 and driver.updates == config.rounds * config.updates_per_round,
            "Incorrect final architecture or optimizer count")
    completion = {"train": 0, "holdout": 0}
    for item in catalog:
        saved = load_round(output, config.rounds, item["split"])
        for scenario, start in enumerate(manifest["fixed_scenarios"]):
            result, trace = simulate(tracks[item["id"]], config, driver, start,
                                     environment_type=MultiTrackEnvironment)
            expected = indexed[(config.rounds, item["id"], scenario)]
            completion[item["split"]] += result["completed"]
            for key, value in result.items():
                if isinstance(value, float):
                    require(abs(value - expected[key]) < 1e-10, f"Final metric mismatch: {key}")
                else:
                    require(value == expected[key], f"Final outcome mismatch: {key}")
            np.testing.assert_array_equal(trace, saved[trace_key(item["id"], scenario)])
    require(completion["train"] == 33, "Final driver did not complete all training starts")
    require(completion["holdout"] == manifest["final"]["holdout_completed"], "Unseen result was misreported")
    record(f"Final checkpoint independently re-driven: {completion['train']}/33 train, {completion['holdout']}/6 unseen starts")
    require(len(details) == (config.rounds + 1) * 39, "Missing scenario metrics")
    for number in range(config.rounds + 1):
        checkpoint = NeuralDriver.load(output / f"models/round_{number:03d}.npz")
        require(checkpoint.updates == number * config.updates_per_round, "Incorrect numbered checkpoint")
        for split, track_count in (("train", 11), ("holdout", 2)):
            traces = load_round(output, number, split)
            require(len(traces) == track_count * 3, "Missing trajectories")
            for trace in traces.values():
                require(trace.shape[1] == 8 and np.isfinite(trace).all(), "Invalid trajectory values")
                require((np.diff(trace[:, 0]) > 0).all(), "Non-increasing driving time")
    record("All 121 checkpoints and 4,719 scenario trajectories present and finite")
    parser = DataParser()
    parser.feed((output / "replay.html").read_text())
    payload = json.loads("".join(parser.parts))
    require(len(payload["frames"]) == 121, "Replay missing rounds")
    for number, frame in enumerate(payload["frames"]):
        require(len(frame) == 13, "Replay missing circuits")
        saved = {**load_round(output, number, "train"), **load_round(output, number, "holdout")}
        for track_id, scenarios in enumerate(frame):
            require(len(scenarios) == 3, "Replay missing starts")
            for scenario, item in enumerate(scenarios):
                np.testing.assert_allclose(item["trace"][-1], saved[trace_key(track_id, scenario)][-1],
                                           atol=0.000051, rtol=0)
                require(item["metric"] == indexed[(number, track_id, scenario)], "Incorrect replay outcome")
    record("Interactive replay includes all rounds, all 13 circuits, all starts, and correct terminal poses")
    sequence = json.loads((output / "animation_round_sequence.json").read_text())["rounds"]
    require(set(sequence) == set(range(121)), "Learning animation skips training rounds")
    require(sequence == sorted(sequence), "Learning animation misorders rounds")
    for name in ("learning_process.gif", "driving_comparison.gif"):
        with Image.open(output / "visuals" / name) as image:
            require(image.n_frames > 100, f"Incomplete animation: {name}")
            for frame in range(image.n_frames):
                image.seek(frame)
                image.load()
    for path in (output / "visuals").glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    record("Every round represented in the learning animation; both GIFs and all PNGs decode")
    with zipfile.ZipFile(output / "training_deck.pptx") as archive:
        require(archive.testzip() is None, "Invalid new presentation package")
    deck = Presentation(output / "training_deck.pptx")
    require(len(deck.slides) == 8, "Missing new presentation slides")
    for slide in deck.slides:
        require(slide.notes_slide.notes_text_frame.text.strip(), "Missing speaker notes")
        for shape in slide.shapes:
            require(shape.left >= 0 and shape.top >= 0
                    and shape.left + shape.width <= deck.slide_width
                    and shape.top + shape.height <= deck.slide_height, "Slide shape out of bounds")
    record("Eight-slide extension deck, speaker notes, and PowerPoint package verified")
    browser_path = output / "browser_verification.json"
    if browser_path.exists():
        browser = json.loads(browser_path.read_text())
        require(browser["passed"] and browser["drive_states_tested"] == 4719
                and browser["learning_rounds_tested"] == 121
                and browser["canvas_rendering_tested"], "Browser checks incomplete")
        record("Browser validated every learning round, every driving outcome, canvas pixels and playback")
    for item in json.loads((output / "artifact_inventory.json").read_text()):
        path = output / item["path"]
        require(path.stat().st_size == item["bytes"], f"Artifact size mismatch: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Artifact hash mismatch: {path}")
    record("All inventoried artifact sizes and SHA-256 hashes match")
    write_json(output / "verification.json", {
        "passed": True, "checks": checks, "training_rounds": config.rounds,
        "optimizer_updates": driver.updates, "completed_final_starts": completion,
        "scenario_trajectories_verified": len(details),
    })
    print(f"All {len(checks)} verification groups passed.")


def bundle(output):
    root = Path(__file__).resolve().parent
    output = Path(output)
    require(json.loads((output / "verification.json").read_text())["passed"], "Experiment not verified")
    require(json.loads((output / "browser_verification.json").read_text())["passed"], "Browser not verified")
    destination = root / "f1-multitrack-lab.zip"
    excluded = {".venv", ".browsers", "node_modules", "__pycache__", ".git"}
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".zip" and path.name != ".DS_Store"
        and not any(part in excluded for part in path.relative_to(root).parts)
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None, "Portable archive failed CRC checks")
        for name in ("replay.html", "models/final.npz", "verification.json", "visuals/learning_process.gif"):
            archive.getinfo(str(Path(root.name) / output.relative_to(root) / name)
                            if output.is_absolute() else str(Path(root.name) / output / name))
    print(f"Verified complete bundle: {destination} ({destination.stat().st_size / 1e6:.1f} MB, {len(paths)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multitrack_artifacts")
    parser.add_argument("--bundle", action="store_true")
    arguments = parser.parse_args()
    verify(arguments.output)
    if arguments.bundle:
        bundle(arguments.output)
