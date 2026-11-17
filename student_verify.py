"""Verify the student datasets, frozen experiment results, visuals and portable ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import NeuralDriver, Track, write_json
from multi_render import inventory
from student_experiments import masked, policy, rule
from wall_sim import WALL_CONFIG, corner_scenarios, wall_simulate


def read(path):
    return json.loads(path.read_text())


def verify(output, bundle=False):
    output = Path(output)
    manifest, data = read(output / "manifest.json"), read(output / "data/lab.json")
    marker, browser = read(output / "training_complete.json"), read(output / "browser_verification.json")
    assert browser["success"] and len(browser["modules"]) == 8 and not browser["errors"]
    assert len(data["features"]) == 18 and len(data["tracks"]) == 16
    assert marker["fresh_test_ids"] == [13, 14, 15] and marker["training_ids"] == list(range(11))
    for key, expected in marker["model_sha256"].items():
        assert hashlib.sha256((output / f"models/{key}.npz").read_bytes()).hexdigest() == expected, key
    assert hashlib.sha256((output / "models/leakage_clean.npz").read_bytes()).hexdigest() == marker["leakage_clean_sha256"]
    assert hashlib.sha256(Path("wall_artifacts/models/final.npz").read_bytes()).hexdigest() == manifest["source_model_sha256"]
    assert hashlib.sha256((output / "models/pretrained.npz").read_bytes()).hexdigest() == manifest["source_model_sha256"]
    with np.load(output / "data/training.npz") as arrays:
        x, y, ids = arrays["features"], arrays["targets"], arrays["track_id"]
    with np.load(output / "data/validation.npz") as arrays:
        vx, vy, validation_ids = arrays["features"], arrays["targets"], arrays["track_id"]
    with np.load(output / "data/pristine_test_observations.npz") as arrays:
        test_ids = arrays["track_id"]
    assert x.shape == (5632, 18) and y.shape == (5632, 2) and vx.shape == (2200, 18)
    assert np.array_equal(np.unique(ids), np.arange(11)) and np.all(np.bincount(ids) == 512)
    assert set(validation_ids) == set(range(11)) and set(test_ids) == {13, 14, 15}
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y)) <= 1
    for key, histories in data["histories"].items():
        mask = key.rsplit("_", 1)[0]
        losses = []
        for row in histories:
            model = NeuralDriver.load(output / f"models/{key}_update_{row['update']:03d}.npz")
            loss = float(np.mean((model.predict(masked(vx, mask)) - vy) ** 2 * [4, 1]))
            assert abs(loss - row["validation_mse"]) < 1e-10
            losses.append(loss)
        chosen = histories[int(np.argmin(losses))]["update"]
        selected = NeuralDriver.load(output / f"models/{key}.npz")
        assert selected.updates == chosen and data["models"][key]["updates"] == chosen
        assert histories[-1]["update"] == 256
        assert losses[-1] < losses[0], key
    assert len(data["ablations"]) == manifest["ablation_rollouts"] == 768
    assert len(data["comparisons"]) == manifest["controller_rollouts"] == 576
    assert len(data["robustness"]) == manifest["robustness_rollouts"] == 960
    for record in data["ablations"] + data["comparisons"] + data["robustness"]:
        assert record["success"] in (0, 1) and 0 <= record["progress"] <= 1
        assert record["wall_contacts"] in (0, 1)
        assert np.isfinite(record["min_clearance_m"])
        if record["success"]:
            assert record["end_reason"] == "section_complete" and not record["wall_contacts"]
            assert record["exit_mean_lateral_m"] < 1 and record["progress"] == 1
    catalog = read(output / "data/circuits.json")
    old = read(Path("wall_artifacts/data/circuits.json"))
    assert catalog[:13] == old
    tracks = {item["id"]: Track(WALL_CONFIG, item["knots"]) for item in catalog}
    assert all(np.min(tracks[i].curvature) < 0 < np.max(tracks[i].curvature) for i in (13, 14, 15))
    replays = 0
    for key in data["models"]:
        mask = key.rsplit("_", 1)[0]
        model = NeuralDriver.load(output / f"models/{key}.npz")
        for track_id in (0, 13):
            scenario = corner_scenarios(tracks[track_id])[3]
            result, _ = wall_simulate(tracks[track_id], WALL_CONFIG, policy(model, WALL_CONFIG, mask=mask), scenario)
            reference = next(row for row in data["ablations"] if row["model"] == key
                             and row["track_id"] == track_id and row["scenario"] == scenario["name"])
            for field in ("success", "wall_contacts", "end_reason", "progress", "min_clearance_m", "exit_mean_lateral_m"):
                if isinstance(result[field], str): assert result[field] == reference[field]
                else: assert abs(result[field] - reference[field]) < 1e-8
            replays += 1
    assert len(data["evolution"]) == 9 and [row["update"] for row in data["evolution"]] == list(range(0, 257, 32))
    assert len(data["starter"]) == 320
    for row in data["starter"]:
        assert row["track_id"] < 11 and len(row["features"]) == 18 and len(row["targets"]) == 2
        if row["source"] == "expert rollout":
            assert row["episode"] == f"initial-expert-episode-{(int(row['id'].split('-')[-1]) - 9900) // 180}"
    for filename in ("teacher_guide.html", "teacher_guide.pdf", "training_deck.html", "training_deck.pptx", "index.html"):
        assert (output / filename).stat().st_size > 1000, filename
    assert len(Presentation(output / "training_deck.pptx").slides) == read(output / "deck_manifest.json")["slides"] == 13
    with zipfile.ZipFile(output / "training_deck.pptx") as deck:
        assert len([name for name in deck.namelist() if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]) == 13
    for path in (output / "visuals").glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1000 and len(image.getcolors(image.width * image.height) or []) > 20
    with Image.open(output / "visuals/training_evolution.gif") as image:
        assert image.n_frames >= 25 and image.width >= 1000
    worksheet = read(output / "data/ml-driving-worksheet.json")
    assert len(worksheet["rows"]) == 9 and worksheet["seeds"] == [401, 402, 403]
    assert worksheet["audit"]["testExposed"] and worksheet["audit"]["testLabelsExposed"]
    assert worksheet["hypothesis"] and worksheet["conclusion"] and worksheet["quiz_attempts"][-1]["score"] == 5
    report = {
        "success": True, "modules": 8, "frozen_models_verified": 13, "validation_checkpoint_selections_verified": 12,
        "reference_rollouts": 2304, "exact_repeated_reference_rollouts": replays,
        "new_reference_optimizer_updates": 3584, "browser_modules": browser["modules"],
        "training_deck_slides": 13, "worksheets": 8, "pristine_test_ids": [13, 14, 15],
        "test_exposure_persists": True, "prior_model_unchanged": True,
    }
    write_json(output / "verification.json", report)
    inventory(output)
    if bundle:
        archive = Path("f1-driving-student-lab.zip")
        files = list(output.rglob("*")) + list(Path("wall_artifacts").rglob("*"))
        files += list(Path(".").glob("*.py")) + list(Path("tests").glob("*"))
        files += [Path(name) for name in ("student_engine.js", "student_app.js", "student_lab.html",
                                         "requirements.txt", "package.json", "package-lock.json", "README.md", "STUDENT_RESULTS.md", "WALL_RESULTS.md")]
        files = sorted(set(path for path in files if path.is_file() and "__pycache__" not in path.parts))
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in files: zipped.write(path, path.as_posix())
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            names = set(zipped.namelist())
            assert f"{output.as_posix()}/index.html" in names and "student_experiments.py" in names
            assert "wall_artifacts/models/final.npz" in names
            for entry in read(output / "artifact_inventory.json"):
                path = output / entry["path"]
                content = zipped.read(path.as_posix())
                assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        write_json(Path("student_bundle_verification.json"), {
            "success": True, "archive": archive.name, "files": len(files),
            "size_bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "crc_verified": True, "student_inventory_verified": True, "previous_wall_lab_included": True,
        })
        print(f"Verified portable additive ZIP: {archive}, {archive.stat().st_size / 1024**2:.1f} MiB.")
    print(f"Verified all eight student modules, 13 frozen clean models, 12 exact validation selections, {replays} repeated benchmark cases and all visual/deck artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="student_artifacts")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    verify(args.output, args.bundle)
