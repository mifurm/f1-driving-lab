"""Select a safety-valid checkpoint without rerunning or hiding training rounds."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from driving import Config, SCENARIOS, Track, write_json
from wall_sim import wall_simulate
from wall_train import select_checkpoint


def select(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = Config(**manifest["config"])
    catalog = json.loads((output / "data/circuits.json").read_text())
    tracks = {item["id"]: Track(config, item["knots"]) for item in catalog}
    training = [item for item in catalog if item["split"] == "train"]
    history = json.loads((output / "history.json").read_text())
    model, number = select_checkpoint(output, config, tracks, training, history)
    model.save(output / "models/final.npz")
    sha = hashlib.sha256((output / "models/final.npz").read_bytes()).hexdigest()
    write_json(output / "training_complete.json", {
        "sha256": sha, "training_updates": manifest["updates"], "selected_model_updates": model.updates,
        "selection_training_only": True, "histories_previously_evaluated": True,
    })
    records = json.loads((output / "attempt_metrics.json").read_text())
    regression, traces = [], {}
    for item in catalog:
        for index, scenario in enumerate(SCENARIOS):
            metric, trace = wall_simulate(tracks[item["id"]], replace(config, max_steps=2600),
                                         model, scenario, goal=tracks[item["id"]].length)
            regression.append({"track_id": item["id"], "split": item["split"], **metric})
            traces[f"track_{item['id']:02d}_scenario_{index}"] = trace
    np.savez_compressed(output / "attempts/full_laps.npz", **traces)
    write_json(output / "full_lap_metrics.json", regression)
    stress = json.loads((output / "stress_metrics.json").read_text())
    with np.load(output / "attempts/stress.npz") as data:
        stress_traces = {key: data[key].copy() for key in data.files}
    for index, row in enumerate(stress):
        if row["model"] != "final":
            continue
        metric, trace = wall_simulate(tracks[row["track_id"]], config, model, row["initial"])
        stress[index] = {**row, **metric}
        trial = int(row["scenario"].split()[-1])
        stress_traces[f"track_{row['track_id']:02d}_final_{trial:02d}"] = trace
    np.savez_compressed(output / "attempts/stress.npz", **stress_traces)
    write_json(output / "stress_metrics.json", stress)
    manifest.update({
        "selected_round": number, "selected_updates": model.updates, "final_sha256": sha,
        "final": history[number], "last_round": history[-1],
        "final_safe_train": sum(row["success"] for row in records if row["round"] == number and row["split"] == "train"),
        "final_safe_unseen": sum(row["success"] for row in records if row["round"] == number and row["split"] == "holdout"),
        "full_lap_train": sum(row["success"] for row in regression if row["split"] == "train"),
        "full_lap_unseen": sum(row["success"] for row in regression if row["split"] == "holdout"),
        "stress_final_safe": sum(row["success"] for row in stress if row["model"] == "final"),
        "stress_final_contacts": sum(row["wall_contacts"] for row in stress if row["model"] == "final"),
        "success": True,
    })
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "selected_round", "final_safe_train", "final_safe_unseen", "full_lap_train", "full_lap_unseen",
        "stress_previous_safe", "stress_final_safe", "stress_previous_contacts", "stress_final_contacts")}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="wall_artifacts")
    select(parser.parse_args().output)
