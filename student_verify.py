"""Verify the version-2 student lab: data validity, frozen models, exact replays, teaching kit, documents, visuals and browser run.

Each check states what it protects. The "teaching signals" at the end are not integrity checks: they confirm that the
lessons the lab claims to teach are actually visible in the saved results (for example, that validation loss tracks
training loss while the recovery-stress loss does not), so a regenerated lab cannot silently lose its point.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from pptx import Presentation

from driving import NeuralDriver, tracks_from_catalog, write_json
from multi_render import inventory
from student_experiments import CONDITIONS, JOURNEY_PRIVATE, MONITOR_TRACKS, _jittered, lap, masked, mse, near_duplicate, policy, rule
from wall_sim import WALL_CONFIG, corner_scenarios, wall_simulate

SCRIPTS = (("student_engine.js", "engine.js"), ("student_tools.js", "tools.js"), ("student_rl.js", "rl.js"), ("student_journey.js", "journey.js"),
           ("student_app.js", "app.js"), ("student_labs.js", "labs.js"), ("student_journey_ui.js", "journey_ui.js"))
BUILD_FILES = ("index.html", "app.js", "engine.js", "tools.js", "rl.js", "journey.js", "labs.js", "journey_ui.js", "class_dashboard.html", "teacher_guide.html")  # keep in step with tests/student-browser.mjs
RESULT_FIELDS = ("success", "section_complete", "centered_exit", "wall_contacts", "end_reason", "progress", "duration_s", "min_clearance_m",
                 "exit_mean_lateral_m", "mean_lateral_m", "peak_speed_kmh", "braking_s")
END_REASONS = {"section_complete", "wall_contact", "lateral_acceleration_limit", "timeout", "numerical_divergence"}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(result, reference, label):
    for field in RESULT_FIELDS:
        if isinstance(reference[field], (str, bool)):
            require(result[field] == reference[field], f"{label}: {field} {result[field]!r} != {reference[field]!r}")
        else:
            require(abs(float(result[field]) - reference[field]) < 1e-8, f"{label}: {field} {result[field]} != {reference[field]}")


def lookup(rows, **keys):
    matches = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    require(len(matches) == 1, f"expected one record for {keys}, found {len(matches)}")
    return matches[0]


# ---- provenance and data validity ------------------------------------------------------------------------------------
def verify_provenance(output, manifest, data, marker):
    require(manifest["lab_version"] == data["version"] == "student-lab-2", "lab version")
    require(marker["fresh_test_ids"] == [13, 14, 15] and marker["training_ids"] == list(range(11)), "track split")
    for key, expected in marker["model_sha256"].items():
        require(sha256(output / f"models/{key}.npz") == expected, f"model hash {key}")
    require(sha256(output / "models/leakage_clean.npz") == marker["leakage_clean_sha256"], "clean leakage model hash")
    require(sha256(output / "models/pretrained.npz") == manifest["source_model_sha256"], "pretrained model hash")
    source = Path("wall_artifacts/models/final.npz")
    if source.exists():
        require(sha256(source) == manifest["source_model_sha256"], "pretrained weights differ from the wall lab's final model")


def load_arrays(output):
    arrays = {}
    for name in ("training", "validation", "stress"):
        with np.load(output / f"data/{name}.npz") as stored:
            arrays[name] = {key: stored[key] for key in stored.files}
    return arrays


def verify_data(output, manifest, data, arrays):
    half_width = data["config"]["track_width"] / 2
    sizes = {"training": manifest["training_samples"], "validation": manifest["validation_samples"], "stress": manifest["stress_samples"]}
    near_centre = {}
    for name, array in arrays.items():
        require(len(array["features"]) == sizes[name] and array["features"].shape[1] == 18 and array["targets"].shape[1] == 2, f"{name} shape")
        require(np.isfinite(array["features"]).all() and np.isfinite(array["targets"]).all() and np.max(np.abs(array["targets"])) <= 1, f"{name} values")
        require(set(array["track_id"].tolist()) == set(range(11)), f"{name} must cover the eleven training circuits")
        near_centre[name] = float(np.mean(np.abs(array["features"][:, 0]) * half_width < 1))
    episodes = read(output / "data/episodes.json")
    require({split: len(items) for split, items in episodes.items()} ==
            {"train": manifest["training_episodes"], "validation": manifest["validation_episodes"], "stress": manifest["stress_episodes"]}, "episode counts")
    for split, array in (("train", arrays["training"]), ("validation", arrays["validation"]), ("stress", arrays["stress"])):
        require(set(array["episode"].tolist()) == set(range(len(episodes[split]))), f"{split} rows must map onto their own episodes")
        for track_id in range(11):
            require(any(e["track_id"] == track_id for e in episodes[split]), f"{split} has no episode on circuit {track_id}")
    # Whole episodes are split, so no start state may be shared (or nearly shared) across train, validation and stress.
    starts = [(split, e) for split, items in episodes.items() for e in items]
    for i, (split_a, a) in enumerate(starts):
        for split_b, b in starts[i + 1:]:
            require(split_a == split_b or not near_duplicate(a, b), f"near-duplicate start across {split_a}/{split_b}: {a['episode']} / {b['episode']}")
    training_rows = {tuple(np.round(row, 9)) for row in arrays["training"]["features"]}
    require(not any(tuple(np.round(row, 9)) in training_rows for row in arrays["validation"]["features"]), "a validation row duplicates a training row")
    # The classroom copies of the data (JavaScript lab) obey the same rules.
    for rows, splits in ((data["starter"], {"train", "validation"}), (data["stressRows"], {"stress"}), (data["testRows"], {"test"})):
        require({r["split"] for r in rows} == splits, f"classroom rows must be split {splits}")
        for r in rows:
            require(len(r["features"]) == 18 and len(r["targets"]) == 2 and np.isfinite(r["features"]).all() and max(map(abs, r["targets"])) <= 1, r["id"])
    for episode, meta in data["starterEpisodes"].items():
        require(episode.startswith("starter-" + ("train" if meta["split"] == "train" else "val")), episode)
    starter = [(m["split"], {**m, "episode": e}) for e, m in data["starterEpisodes"].items()]
    for i, (split_a, a) in enumerate(starter):
        for split_b, b in starter[i + 1:]:
            require(split_a == split_b or not near_duplicate(a, b), f"near-duplicate starter episodes {a['episode']} / {b['episode']}")
    require({r["track_id"] for r in data["testRows"]} == {13, 14, 15} and all(r["track_id"] < 11 for r in data["starter"] + data["stressRows"]), "sealed circuits must stay out of training data")
    return near_centre


# ---- training histories ---------------------------------------------------------------------------------------------------
def verify_training(output, manifest, data, arrays):
    train, validation, stress = arrays["training"], arrays["validation"], arrays["stress"]
    final = {}
    for key, history in data["histories"].items():
        mask = key.rsplit("_", 1)[0]
        require([h["update"] for h in history] == list(range(0, manifest["updates_per_model"] + 1, 32)) or data["diverged"][key], f"{key} checkpoints")
        losses = []
        for row in history:
            model = NeuralDriver.load(output / f"models/{key}_update_{row['update']:03d}.npz")
            computed = {"train_mse": mse(model.predict(masked(train["features"], mask)), train["targets"]),
                        "validation_mse": mse(model.predict(masked(validation["features"], mask)), validation["targets"]),
                        "stress_mse": mse(model.predict(masked(stress["features"], mask)), stress["targets"])}
            for field, value in computed.items():
                require(abs(value - row[field]) < 1e-10, f"{key} update {row['update']} {field}")
            losses.append(row["validation_mse"])
        chosen = history[int(np.argmin(losses))]["update"]
        selected = NeuralDriver.load(output / f"models/{key}.npz")
        require(selected.updates == chosen == data["models"][key]["updates"], f"{key}: selection must be the lowest i.i.d. validation loss")
        require(losses[-1] < losses[0], f"{key}: training must lower validation loss")
        final[key] = history[-1]
    return final


# ---- rollouts ----------------------------------------------------------------------------------------------------------------
def verify_records(manifest, data):
    require(len(data["ablations"]) == manifest["ablation_rollouts"] == manifest["ablation_models"] * 16 * 4, "ablation record count")
    require(len(data["comparisons"]) == manifest["controller_rollouts"] == 3 * len(manifest["paired_seeds"]) * 16 * 4, "controller record count")
    require(len(data["robustness"]) == manifest["robustness_rollouts"] == len(CONDITIONS) * len(manifest["paired_seeds"]) * 16 * 4, "robustness record count")
    for record in data["ablations"] + data["comparisons"] + data["robustness"]:
        require(record["success"] in (0, 1) and 0 <= record["progress"] <= 1 and record["wall_contacts"] in (0, 1), "record ranges")
        require(record["end_reason"] in END_REASONS and np.isfinite(record["min_clearance_m"]), "record outcome")
        require(record["kind"] in ("nominal", "recovery") and record["scenario_index"] in range(4), "record section")
        if record["success"]:
            require(record["end_reason"] == "section_complete" and not record["wall_contacts"] and record["exit_mean_lateral_m"] < 1 and record["progress"] == 1, "a success must be a complete, centred, wall-free exit")
    require(not manifest["diverged_seeds"], "the reference run must have no diverged seeds (they would have to be reported as failures)")
    sections = {(r["track_id"], r["scenario_index"], r["scenario"]) for r in data["ablations"]}
    require(len(sections) == 64, "every circuit must contribute its four canonical sections")


def verify_consistency(data, manifest):
    """Two independent evaluations of the same weights (full-run tables and per-checkpoint monitor) must agree."""
    checked = 0
    for seed in manifest["ablation_seeds"]:
        key = f"full_{seed}"
        selected = data["models"][key]["updates"]
        entry = next(e for e in data["closedLoopReference"][key] if e["update"] == selected)
        for kind in ("nominal", "recovery"):
            rows = [r for r in data["ablations"] if r["model"] == key and r["track_id"] in MONITOR_TRACKS and r["kind"] == kind]
            require(entry[kind] == {"k": sum(r["success"] for r in rows), "n": len(rows)}, f"{key} {kind}: monitor and full-run tables disagree")
            checked += 1
    return checked


def verify_replays(output, data, tracks):
    replays = 0
    for key in data["models"]:
        mask = key.rsplit("_", 1)[0]
        model = NeuralDriver.load(output / f"models/{key}.npz")
        for track_id in (3, 13):
            scenario = corner_scenarios(tracks[track_id])[3]
            result, _ = wall_simulate(tracks[track_id], WALL_CONFIG, policy(model, WALL_CONFIG, mask=mask), scenario)
            same(result, lookup(data["ablations"], model=key, track_id=track_id, scenario_index=3), f"replay {key} track {track_id}")
            replays += 1
    pretrained = NeuralDriver.load(output / "models/pretrained.npz")
    for seed in (401, 410):
        for track_id, index in ((2, 0), (3, 2), (13, 3)):
            scenario = _jittered(corner_scenarios(tracks[track_id])[index], seed, track_id, index)
            for label in ("rules", "expert", "pretrained"):
                driver = rule if label == "rules" else None if label == "expert" else policy(pretrained, WALL_CONFIG, seed)
                result, _ = wall_simulate(tracks[track_id], WALL_CONFIG, driver, scenario)
                same(result, lookup(data["comparisons"], controller=label, seed=seed, track_id=track_id, scenario_index=index), f"replay {label} {seed}")
                replays += 1
            for condition in CONDITIONS:
                shifted = replace(WALL_CONFIG, max_braking=WALL_CONFIG.max_braking * condition["grip"],
                                  max_lateral_acceleration=WALL_CONFIG.max_lateral_acceleration * condition["grip"])
                driver = policy(pretrained, WALL_CONFIG, seed, noise=condition["noise"], delay=condition["delay"])
                result, _ = wall_simulate(tracks[track_id], shifted, driver, scenario)
                same(result, lookup(data["robustness"], condition=condition["name"], seed=seed, track_id=track_id, scenario_index=index), f"replay {condition['name']} {seed}")
                replays += 1
    return replays


# ---- module 12: from zero to a lap ----------------------------------------------------------------------------------------------
def lap_percent(result):
    return 100 if result["end_reason"] == "section_complete" else min(99, int(np.floor(result["progress"] * 100)))


def first_lap(rows):
    return next((row[0] for row in rows if row[2]), None)


def stable_lap(rows):
    start = len(rows)
    while start > 0 and rows[start - 1][2]:
        start -= 1
    return rows[start][0] if start < len(rows) else None


def verify_journey(output, data, tracks):
    """Module 12's saved reference: the page embeds it unchanged, every saved stage of the featured example is replayed in the independent NumPy simulator,
    the experts' laps are re-driven, and the summary table is re-derived from the raw runs."""
    path = output / "data/journey_reference.json"
    require(path.exists(), "run `node tools/journey_reference.cjs --artifact <output>`, then student_render.py: module 12 needs its saved reference")
    reference = read(path)
    require(data.get("journeyReference") == {key: value for key, value in reference.items() if key not in JOURNEY_PRIVATE}, "the page embeds a stale journey reference; re-run student_render.py")
    settings, featured, limit, minimum = reference["settings"], reference["featured"], reference["settings"]["lap"]["flickerLimit"], reference["settings"]["lap"]["flickerMinSeconds"]
    track, replayed, smooth_exact = tracks[featured["trackId"]], 0, 0
    for method in ("imitation", "reinforcement"):
        stages = featured[method]["stages"]
        require(len(stages) >= 10 and stages[0]["steps"] == 0, f"the {method} example must start from an untrained car and have many stages")
        for stage in stages:
            result, trace = lap(track, stage["model"])
            steps, percent, completed, seconds, flicker = stage["row"]
            label = f"{method} stage {stage['index']}"
            # A car that flicks its steering for a long drive is chaotic (a 1e-14 difference between two correct simulators grows to metres in seconds), so only its outcome is compared.
            exact = flicker <= limit or seconds < minimum
            require((result["end_reason"] == "section_complete") == bool(completed), f"{label}: NumPy completes the lap differently ({result['end_reason']})")
            require(abs(lap_percent(result) - percent) <= (1 if exact else 100), f"{label}: NumPy drives {lap_percent(result)}% of the lap, the page saved {percent}%")
            python_flicker = float(np.abs(np.diff(trace[:, 5])).mean() * 180 / np.pi)
            require((python_flicker > limit) == (flicker > limit), f"{label}: NumPy and the page disagree on whether the steering flicks ({python_flicker:.2f} against {flicker})")
            if exact:
                require(abs(result["duration_s"] - seconds) < .11 and abs(python_flicker - flicker) < .02, f"{label}: lap {result['duration_s']:.2f} s / {python_flicker:.3f}° against {seconds} s / {flicker}°")
                smooth_exact += 1
            replayed += 1
    expert_laps = 0
    for circuit_id, circuit in reference["circuits"].items():
        result, _ = lap(tracks[int(circuit_id)])
        require(result["end_reason"] == "section_complete" and abs(result["duration_s"] - circuit["expertSeconds"]) < .11, f"expert lap on circuit {circuit_id}")
        expert_laps += 1
        for method in ("imitation", "reinforcement"):
            runs = [run["stages"] for run in reference["runs"][method][circuit_id]]
            summary = circuit[method]
            require(len(runs) == settings["seeds"] == summary["seeds"], f"{method} circuit {circuit_id}: seed count")
            require(float(np.median([first_lap(rows) for rows in runs if first_lap(rows) is not None])) == summary["firstLapSteps"]["median"], f"{method} circuit {circuit_id}: first-lap median")
            require(float(np.median([stable_lap(rows) for rows in runs if stable_lap(rows) is not None])) == summary["stableLapSteps"]["median"], f"{method} circuit {circuit_id}: stays-lapping median")
            require(sum(rows[-1][2] for rows in runs) == summary["finalLaps"], f"{method} circuit {circuit_id}: final laps")
            ratios = [rows[-1][3] / circuit["expertSeconds"] for rows in runs if rows[-1][2]]
            require(abs(float(np.mean(ratios)) - summary["ratio"]["mean"]) < 1e-9, f"{method} circuit {circuit_id}: lap-time ratio")
            require(sum(rows[-1][3] >= minimum and rows[-1][4] > limit for rows in runs) == summary["flickering"], f"{method} circuit {circuit_id}: flicking cars")
            lost = sum(first_lap(rows) is not None and any(not row[2] for row in rows[[row[0] for row in rows].index(first_lap(rows)):]) for rows in runs)
            require(lost == summary["lostLap"], f"{method} circuit {circuit_id}: seeds that lost a lap")
            require(all(len(rows) == len(summary["curve"]) for rows in runs), f"{method} circuit {circuit_id}: every seed shares one stage schedule")
    overall, ablation = reference["overall"], reference["ablation"]
    imitation, reinforcement = overall["imitation"], overall["reinforcement"]
    signals = {
        "journey_untrained_cars_fail_at_once": all(not featured[m]["stages"][0]["row"][2] and featured[m]["stages"][0]["row"][1] <= 5 for m in ("imitation", "reinforcement")),
        "journey_every_learner_can_end_with_a_lapping_car": all(featured[m]["stages"][-1]["row"][2] for m in ("imitation", "reinforcement")),
        "journey_both_learners_lap_in_nearly_every_run": imitation["finalLaps"] >= .95 * imitation["runs"] and reinforcement["finalLaps"] >= .95 * reinforcement["runs"],
        "journey_reward_alone_needs_far_more_driving_than_labels": reinforcement["stableLapSteps"]["median"] >= 10 * imitation["stableLapSteps"]["median"],
        "journey_final_cars_are_near_expert_speed": 0.9 < imitation["ratio"] < 1.3 and 0.9 < reinforcement["ratio"] < 1.4,
        "journey_without_a_smoothness_reward_the_steering_flicks": ablation["without"]["flickering"] >= .8 * ablation["without"]["seeds"] and ablation["with"]["flickering"] <= .1 * ablation["with"]["seeds"],
        "journey_imitation_cars_never_flick": imitation["flickering"] == 0,
    }
    failed = [name for name, ok in signals.items() if not ok]
    require(not failed, f"journey teaching signals missing from the saved results: {failed}")
    return {"stages_replayed_in_numpy": replayed, "stages_matching_to_0.1s": smooth_exact, "expert_laps": expert_laps, "runs": imitation["runs"] + reinforcement["runs"] + ablation["without"]["seeds"], "signals": signals}


# ---- teaching kit, documents, visuals, browser --------------------------------------------------------------------------------
def verify_documents(output, data, kit):
    require(data["teaching"] == kit, "the page embeds a stale teaching kit; re-run student_render.py")
    for source, target in SCRIPTS:
        require(sha256(source) == sha256(output / target), f"{target} is not a copy of {source}; re-run student_render.py")
    page = (output / "index.html").read_text()
    require("__DATA__" not in page and all(f'src="{target}"' in page for _, target in SCRIPTS), "index.html is not assembled")
    guide = (output / "teacher_guide.html").read_text()
    for objective in kit["objectives"]:
        require(objective["id"] in guide, f"teacher guide misses {objective['id']}")
    for misconception in kit["misconceptions"]:
        require(misconception["id"] in guide, f"teacher guide misses {misconception['id']}")
    for worksheet in kit["worksheets"]:
        require(worksheet["title"] in guide, f"teacher guide misses worksheet {worksheet['title']}")
    require("Correct" in guide or "correct" in guide, "teacher guide needs the answer key")
    dashboard = (output / "class_dashboard.html").read_text()
    require("__KIT__" not in dashboard and "f1-ml-lab-results-card" in dashboard, "class dashboard is not assembled")
    for name in ("teacher_guide.html", "class_dashboard.html", "training_deck.html", "training_deck.pptx", "index.html"):
        require((output / name).stat().st_size > 1000, name)
    slides = read(output / "deck_manifest.json")["slides"]
    require(len(Presentation(output / "training_deck.pptx").slides) == slides, "deck slide count")
    with zipfile.ZipFile(output / "training_deck.pptx") as deck:
        notes = [n for n in deck.namelist() if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")]
        require(len(notes) == slides, "every slide needs speaker notes")
    return slides


def verify_visuals(output):
    for path in (output / "visuals").glob("*.png"):
        with Image.open(path) as image:
            require(image.width >= 1000 and len(image.getcolors(image.width * image.height) or []) > 20, f"{path.name} looks empty")
    names = {p.name for p in (output / "visuals").glob("*.png")}
    require({"feature_learning.png", "loss_vs_driving.png", "robustness.png", "journey_filmstrip.png", "journey_curves.png"} <= names, "missing charts")
    with Image.open(output / "visuals/training_evolution.gif") as image:
        require(image.n_frames >= 25 and image.width >= 1000, "training evolution animation")
    with Image.open(output / "visuals/journey_learning.gif") as image:
        require(image.n_frames >= 100 and image.width >= 900, "journey learning movie")


def verify_browser(output, kit):
    report = read(output / "browser_verification.json")
    require(report["success"] and not report["errors"], f"browser run failed: {report['errors']}")
    require(report["modules"] == [m["id"] for m in kit["modules"]], "the browser run must visit every module in order")
    failed = [name for name, ok in report["checks"].items() if ok is not True]
    require(not failed, f"browser checks failed: {failed}")
    # The report is evidence only for the build it exercised: a re-rendered page needs a fresh browser run.
    tested = report.get("build_sha256", {})
    stale = [name for name in BUILD_FILES if tested.get(name) != sha256(output / name)]
    require(not stale, f"browser_verification.json was produced for a different build of {stale}; re-run the browser test")
    require((output / "teacher_guide.pdf").stat().st_size > 1000, "teacher_guide.pdf was not produced by the browser run")
    return report


# ---- teaching signals ------------------------------------------------------------------------------------------------------
def rate(rows, **keys):
    rows = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    return sum(r["success"] for r in rows) / len(rows)


def teaching_signals(manifest, data, final, near_centre):
    valid, seeds = set(manifest["valid_training_tracks"]), manifest["ablation_seeds"]
    train_rows = [r for r in data["ablations"] if r["split"] == "train" and r["track_id"] in valid]
    full = [final[f"full_{s}"] for s in seeds]
    nominal_rate = lambda mask: float(np.mean([rate(train_rows, model=f"{mask}_{s}", kind="nominal") for s in seeds]))  # noqa: E731
    recovery_rate = lambda mask: float(np.mean([rate(train_rows, model=f"{mask}_{s}", kind="recovery") for s in seeds]))  # noqa: E731
    signals = {
        "validation_matches_training_distribution": abs(near_centre["training"] - near_centre["validation"]) < 0.05,
        "stress_is_a_different_distribution": near_centre["stress"] < 0.7 and near_centre["training"] > 0.85,
        "no_fake_generalization_gap": np.mean([f["validation_mse"] for f in full]) < 1.5 * np.mean([f["train_mse"] for f in full]),
        "stress_loss_exposes_coverage_not_overfitting": np.mean([f["stress_mse"] for f in full]) > 2 * np.mean([f["validation_mse"] for f in full]),
        "removing_road_previews_breaks_driving": nominal_rate("no_future_geometry") < 0.2 and nominal_rate("full") > 0.7,
        "recovery_is_harder_than_nominal": recovery_rate("full") < nominal_rate("full") - 0.2,
    }
    controllers = [r for r in data["comparisons"] if r["split"] == "train" and r["track_id"] in valid]
    signals["rules_cannot_recover"] = rate(controllers, controller="rules", kind="recovery") < 0.1
    signals["expert_is_the_ceiling"] = rate(controllers, controller="expert") > 0.98 and rate(controllers, controller="pretrained") > 0.95
    clean = [r for r in data["robustness"] if r["condition"] == "Clean" and r["split"] == "train" and r["track_id"] in valid]
    shifted = [r for r in data["robustness"] if r["condition"] == "Combined shift" and r["split"] == "train" and r["track_id"] in valid]
    signals["distribution_shift_costs_performance"] = rate(shifted, split="train") < rate(clean, split="train") - 0.2
    mean_validation = np.mean([[h["validation_mse"] for h in data["histories"][f"full_{s}"]] for s in seeds], axis=0)
    signals["validation_loss_is_monotone_on_average"] = bool(np.all(np.diff(mean_validation) < 0))
    recovery = [[h["recovery"]["k"] for h in data["closedLoopReference"][f"full_{s}"]] for s in seeds]
    signals["closed_loop_success_is_erratic_while_loss_is_smooth"] = sum(any(b < a for a, b in zip(r, r[1:])) for r in recovery) >= len(seeds) // 2
    failed = [name for name, ok in signals.items() if not ok]
    require(not failed, f"teaching signals missing from the saved results: {failed}")
    return {name: bool(ok) for name, ok in signals.items()}


# ---- bundle ----------------------------------------------------------------------------------------------------------------------
def build_bundle(output):
    archive = Path("f1-driving-student-lab-v2.zip")
    files = list(output.rglob("*")) + list(Path("wall_artifacts").rglob("*")) + list(Path("tests").rglob("*")) + list(Path("tools").rglob("*"))
    files += list(Path(".").glob("*.py")) + list(Path(".github").rglob("*"))
    files += [Path(name) for name in ("student_engine.js", "student_tools.js", "student_rl.js", "student_journey.js", "student_app.js", "student_labs.js", "student_journey_ui.js", "student_lab.html", "student_dashboard.html",
                                      "teaching_kit.json", "requirements.txt", "package.json", "package-lock.json", "README.md", "STUDENT_V2_RESULTS.md", "WALL_RESULTS.md")]
    files = sorted({path for path in files if path.is_file() and "__pycache__" not in path.parts})
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in files:
            zipped.write(path, path.as_posix())
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "corrupt bundle")
        names = set(zipped.namelist())
        require(f"{output.as_posix()}/index.html" in names and "student_experiments.py" in names and "teaching_kit.json" in names, "bundle contents")
        require("wall_artifacts/models/final.npz" in names, "the wall lab's final model must ship with the bundle")
        for entry in read(output / "artifact_inventory.json"):
            require(hashlib.sha256(zipped.read((output / entry["path"]).as_posix())).hexdigest() == entry["sha256"], f"bundle copy of {entry['path']} differs")
    write_json(Path("student_v2_bundle_verification.json"), {
        "success": True, "archive": archive.name, "files": len(files), "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "crc_verified": True, "student_inventory_verified": True})
    return archive


def verify(output, bundle=False):
    output = Path(output)
    manifest, data, marker = read(output / "manifest.json"), read(output / "data/lab.json"), read(output / "training_complete.json")
    kit = read("teaching_kit.json")
    verify_provenance(output, manifest, data, marker)
    arrays = load_arrays(output)
    near_centre = verify_data(output, manifest, data, arrays)
    final = verify_training(output, manifest, data, arrays)
    verify_records(manifest, data)
    consistency = verify_consistency(data, manifest)
    tracks = tracks_from_catalog(WALL_CONFIG, read(output / "data/circuits.json"))
    replays = verify_replays(output, data, tracks)
    journey = verify_journey(output, data, tracks)
    slides = verify_documents(output, data, kit)
    verify_visuals(output)
    browser = verify_browser(output, kit)
    signals = teaching_signals(manifest, data, final, near_centre)
    report = {
        "success": True, "lab_version": manifest["lab_version"], "modules": len(kit["modules"]), "frozen_models_verified": len(marker["model_sha256"]) + 2,
        "validation_checkpoint_selections_verified": len(data["histories"]), "checkpoint_losses_recomputed": sum(len(h) for h in data["histories"].values()),
        "reference_rollouts": len(data["ablations"]) + len(data["comparisons"]) + len(data["robustness"]) + manifest["monitor_rollouts"],
        "exact_repeated_reference_rollouts": replays, "monitor_vs_full_run_agreements": consistency,
        "near_centreline_share": near_centre, "teaching_signals": {**signals, **journey["signals"]}, "journey": {k: v for k, v in journey.items() if k != "signals"}, "browser_modules": browser["modules"], "browser_checks": sorted(browser["checks"]),
        "training_deck_slides": slides, "worksheets": len(kit["worksheets"]), "quiz_items": len(kit["quiz"]), "sealed_test_ids": [13, 14, 15],
    }
    write_json(output / "verification.json", report)
    inventory(output)
    if bundle:
        archive = build_bundle(output)
        print(f"Verified portable ZIP: {archive}, {archive.stat().st_size / 1024**2:.1f} MiB.")
    print(f"Verified {len(kit['modules'])} modules, {report['frozen_models_verified']} frozen models, {len(data['histories'])} validation selections, "
          f"{replays} exact replays, {journey['stages_replayed_in_numpy']} journey stages replayed in NumPy, {len(signals) + len(journey['signals'])} teaching signals and the browser run in {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="student_v2_artifacts")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    verify(args.output, args.bundle)
