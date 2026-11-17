"""Visualize the multi-circuit training history, saved drives, and blind tests."""

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

from circuits import CIRCUITS
from driving import Config, SCENARIOS, write_json
from render import (
    BACKGROUND, BLUE, GOLD, MUTED, PANEL, RED, TEAL, WHITE,
    draw_track, presentation, save_figure,
)


def load_round(output, number, split):
    with np.load(output / f"trajectories/{split}_round_{number:03d}.npz") as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def trace_key(track_id, scenario=0):
    return f"track_{track_id:02d}_scenario_{scenario}"


def index_details(details):
    scenario_ids = {scenario["name"]: index for index, scenario in enumerate(SCENARIOS)}
    return {
        (row["round"], row["track_id"], scenario_ids[row["scenario"]]): row
        for row in details
    }


def make_charts(output, tracks, history, details, manifest):
    directory = output / "visuals"
    figure, axes = plt.subplots(4, 4, figsize=(14, 12))
    for axis, circuit in zip(axes.flat, CIRCUITS):
        track = tracks[circuit.id]
        draw_track(axis, track)
        axis.plot(*track.xy.T, color=BLUE if circuit.split == "holdout" else TEAL, lw=1)
        axis.set_title(f"{circuit.id:02d}  {circuit.name}\n"
                       f"{track.length / 1000:.2f} km  /  {circuit.split.upper()}", fontsize=10)
    for axis in list(axes.flat)[len(CIRCUITS):]:
        axis.axis("off")
    axes[3, 1].text(0, 0.7, "11 TRAINING CIRCUITS\nOriginal + 10 new layouts",
                   color=TEAL, fontsize=13, weight="bold", transform=axes[3, 1].transAxes, linespacing=2)
    axes[3, 2].text(0, 0.7, "2 UNSEEN CIRCUITS\nNever used for training",
                   color=BLUE, fontsize=13, weight="bold", transform=axes[3, 2].transAxes, linespacing=2)
    figure.suptitle("MORE TRACKS, MORE DRIVING SITUATIONS", fontsize=21)
    figure.tight_layout()
    save_figure(figure, directory, "track_atlas", svg=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.axis("off")
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    boxes = [
        (0.2, 3.5, "DIVERSE GEOMETRY", "11 original training layouts\n2 unseen test-only layouts"),
        (4.25, 3.5, "BETTER OBSERVATIONS", "13 original inputs + 3 curvature inputs\nSame physical car and failure limits"),
        (8.3, 3.5, "BALANCED DATA", "180 new states / track / round\nEqual minibatch share for every track"),
        (8.3, 0.65, "CURRICULUM", "Foundation → recovery → challenge\nHalf recent / half historical examples"),
        (4.25, 0.65, "120 REAL ROUNDS", "64 Adam updates each round\nAll weights and trajectories saved"),
        (0.2, 0.65, "FREEZE, THEN TEST", "33 monitored starts / round\n6 unseen starts tested afterward"),
    ]
    for x, y, title, subtitle in boxes:
        axis.add_patch(FancyBboxPatch((x, y), 3.35, 1.7, boxstyle="round,pad=0.12",
                                     facecolor=PANEL, edgecolor=TEAL, lw=1.4))
        axis.text(x + 0.12, y + 1.19, title, color=TEAL, weight="bold", fontsize=11)
        axis.text(x + 0.12, y + 0.52, subtitle, fontsize=9, linespacing=1.8)
    for start, end in [
        ((3.7, 4.35), (4.05, 4.35)), ((7.75, 4.35), (8.1, 4.35)),
        ((9.95, 3.3), (9.95, 2.55)), ((8.1, 1.5), (7.75, 1.5)),
        ((4.05, 1.5), (3.7, 1.5)),
    ]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=GOLD, lw=2))
    axis.set_title("IMPROVING THE LEARNING PROCESS", fontsize=20, pad=14)
    save_figure(figure, directory, "improvement_pipeline", svg=True)

    rounds = np.array([row["round"] for row in history])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for prefix, color, label in [("train", TEAL, "Training-circuit tests"), ("holdout", BLUE, "Unseen tests (post-training)")]:
        axes[0].plot(rounds, [row[f"{prefix}_progress_pct"] for row in history], color=color, lw=1.8, label=label)
        axes[1].plot(rounds, [row[f"{prefix}_completed"] / row[f"{prefix}_total"] * 100 for row in history],
                     color=color, lw=1.8, label=label)
    for axis in axes:
        axis.axvline(40, color=GOLD, ls=":", alpha=0.6)
        axis.axvline(80, color=GOLD, ls=":", alpha=0.6)
        axis.set(xlabel="Actual training round", ylim=(-2, 105))
        axis.grid(alpha=0.4)
    axes[0].set(ylabel="Average lap progress (%)", title="The raw learning curve")
    axes[1].set(ylabel="Completed fixed starts (%)", title="Finishing, not just predicting")
    axes[0].legend(fontsize=9, loc="lower right")
    figure.suptitle("120 ROUNDS  /  FOUNDATION → RECOVERY → CHALLENGE", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, "progress")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].semilogy(rounds, [row["validation_weighted_mse"] for row in history], color=TEAL)
    axes[0].set(ylabel="Independent validation weighted MSE", title="Imitation accuracy")
    axes[1].plot(rounds, [row["samples"] for row in history], color=GOLD)
    axes[1].set(ylabel="Labeled training states", title="Growing balanced dataset")
    axes[2].plot(rounds, [row["collection_failures"] for row in history], color=RED)
    axes[2].set(ylabel="Failures during data collection", title="Increasingly difficult starts")
    for axis in axes:
        axis.set_xlabel("Training round")
        axis.grid(alpha=0.4)
        for boundary in (40, 80):
            axis.axvline(boundary, color=MUTED, ls=":", alpha=0.5)
    figure.tight_layout()
    save_figure(figure, directory, "learning")

    complete = [row for row in history if row["train_completed"] == row["train_total"]]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, key, label, color in [
        (axes[0], "train_lateral_completed_m", "Mean lateral error (m)", TEAL),
        (axes[1], "train_lap_time_completed_s", "Average completed lap time (s)", BLUE),
        (axes[2], "train_steering_completed_rad_s", "Steering variation (rad/s)", GOLD),
    ]:
        axis.plot([row["round"] for row in complete], [row[key] for row in complete], color=color, lw=1.5)
        axis.set(xlabel="Training round", ylabel=label)
        axis.grid(alpha=0.4)
    figure.suptitle("TECHNIQUE  /  ONLY ROUNDS FINISHING ALL 33 TRAINING STARTS", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, "quality")

    indexed = index_details(details)
    figure, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, circuit in zip(axes, [c for c in CIRCUITS if c.split == "holdout"]):
        draw_track(axis, tracks[circuit.id])
        trace = load_round(output, 120, "holdout")[trace_key(circuit.id)]
        metric = indexed[(120, circuit.id, 0)]
        axis.plot(trace[:, 1], trace[:, 2], color=TEAL if metric["completed"] else RED, lw=2)
        count = sum(indexed[(120, circuit.id, scenario)]["completed"] for scenario in range(3))
        axis.set_title(f"{circuit.name}\n{count}/3 fixed starts completed", fontsize=13)
    figure.suptitle("UNSEEN GEOMETRY  /  TESTED ONLY AFTER WEIGHTS WERE FROZEN", fontsize=16)
    figure.tight_layout()
    save_figure(figure, directory, "unseen")

    figure, axis = plt.subplots(figsize=(12, 6))
    names = [c.name for c in CIRCUITS]
    old = manifest["transfer_baseline"]
    transfer_details = json.loads((output / "transfer_baseline.json").read_text())["details"]
    prior_errors, new_errors = [], []
    for circuit in CIRCUITS:
        prior = [d for d in transfer_details if d["track_id"] == circuit.id]
        final = [indexed[(120, circuit.id, scenario)] for scenario in range(3)]
        prior_errors.append(np.mean([d["mean_abs_lateral_m"] for d in prior]) if all(d["completed"] for d in prior) else np.nan)
        new_errors.append(np.mean([d["mean_abs_lateral_m"] for d in final]) if all(d["completed"] for d in final) else np.nan)
    positions = np.arange(len(CIRCUITS))
    axis.bar(positions - 0.18, prior_errors, 0.36, color=MUTED, label="Previous single-track model")
    axis.bar(positions + 0.18, new_errors, 0.36, color=TEAL, label="New multi-track model")
    axis.set_xticks(positions, names, rotation=35, ha="right")
    axis.set(ylabel="Mean lateral error over three completed laps (m)",
             title="TRANSFER COMPARISON  /  COMPLETED-LAP ACCURACY")
    axis.legend(fontsize=10)
    axis.grid(axis="y", alpha=0.35)
    figure.text(0.08, 0.02,
                f"Previous model: {old['train_completed']}/33 training starts, {old['holdout_completed']}/6 unseen starts."
                " Missing bars mean an incomplete set of laps.", color=MUTED, fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(figure, directory, "transfer_comparison")


def training_animation(output, tracks, history, details):
    indexed = index_details(details)
    path_cache = [
        {key: trace[:, 1:3].copy() for key, trace in load_round(output, number, "train").items()
         if key.endswith("_scenario_0")}
        for number in range(121)
    ]
    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.8], hspace=0.35, wspace=0.18)
    artists = []
    training_circuits = [c for c in CIRCUITS if c.split == "train"]
    for index, circuit in enumerate(training_circuits):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        draw_track(axis, tracks[circuit.id])
        line, = axis.plot([], [], lw=1.7)
        marker, = axis.plot([], [], marker="o", color=GOLD, ms=3)
        title = axis.set_title("", fontsize=9)
        artists.append((circuit.id, line, marker, title))
    information = figure.add_subplot(grid[2, 3])
    information.axis("off")
    counter = information.text(0.04, 0.9, "", transform=information.transAxes,
                               color=TEAL, fontsize=12, va="top", linespacing=1.9)
    progress_axis = figure.add_subplot(grid[0, 4])
    progress_axis.set(xlim=(0, 120), ylim=(0, 105), xlabel="Training round",
                      title="Mean lap progress (%)")
    progress_axis.grid(alpha=0.35)
    progress_line, = progress_axis.plot([], [], color=TEAL, lw=2, label="11 training circuits")
    unseen_line, = progress_axis.plot([], [], color=BLUE, lw=1.6, label="2 unseen (post-test)")
    cursor = progress_axis.axvline(0, color=GOLD, lw=1)
    progress_axis.legend(loc="lower right", fontsize=8)
    accuracy_axis = figure.add_subplot(grid[1, 4])
    maximum_error = max(row["train_lateral_completed_m"] for row in history
                        if row["train_lateral_completed_m"] is not None)
    accuracy_axis.set(xlim=(0, 120), ylim=(0, maximum_error * 1.12),
                      xlabel="Training round", title="Completed-lap lane error (m)")
    progress_axis.title.set_fontsize(11)
    accuracy_axis.title.set_fontsize(11)
    accuracy_axis.grid(alpha=0.35)
    accuracy_line, = accuracy_axis.plot([], [], color=TEAL, lw=2)
    accuracy_cursor = accuracy_axis.axvline(0, color=GOLD, lw=1)
    notes = figure.add_subplot(grid[2, 4])
    notes.axis("off")
    score = notes.text(0.02, 0.9, "", transform=notes.transAxes, va="top", fontsize=12, linespacing=2)
    notes.text(0.02, 0.12, "Clock = training rounds,\nnot simulated driving time.\nUnseen results are post-training.",
               transform=notes.transAxes, color=MUTED, fontsize=9, linespacing=1.5)
    heading = figure.suptitle("", fontsize=21, y=0.98)
    sequence = [0] * 3 + [n for n in range(1, 16) for _ in range(2)] + list(range(16, 121)) + [120] * 5
    def frame(frame_index):
        number = sequence[frame_index]
        row = history[number]
        heading.set_text(f"LEARNING TO DRIVE 11 CIRCUITS  /  ROUND {number:03d} OF 120")
        for track_id, line, marker, title in artists:
            path = path_cache[number][trace_key(track_id)]
            metric = indexed[(number, track_id, 0)]
            finished = sum(indexed[(number, track_id, s)]["completed"] for s in range(3))
            line.set_data(path[:, 0], path[:, 1])
            line.set_color(TEAL if metric["completed"] else RED)
            marker.set_data([path[-1, 0]], [path[-1, 1]])
            title.set_text(f"{CIRCUITS[track_id].name}\n{finished}/3 completed starts")
        numbers = [h["round"] for h in history[:number + 1]]
        progress_line.set_data(numbers, [h["train_progress_pct"] for h in history[:number + 1]])
        unseen_line.set_data(numbers, [h["holdout_progress_pct"] for h in history[:number + 1]])
        cursor.set_xdata([number, number])
        completed = [h for h in history[:number + 1] if h["train_completed"] == 33]
        accuracy_line.set_data([h["round"] for h in completed],
                               [h["train_lateral_completed_m"] for h in completed])
        accuracy_cursor.set_xdata([number, number])
        counter.set_text(f"{row['phase'].upper()}\n{row['samples']:,} labels\n"
                         f"{row['optimizer_updates']:,} updates\n"
                         f"Val. MSE {row['validation_weighted_mse']:.4f}")
        score.set_text(f"TRAIN: {row['train_completed']}/33 starts\n"
                       f"Progress {row['train_progress_pct']:.1f}%\n"
                       f"UNSEEN: {row['holdout_completed']}/6 starts")
    animation = FuncAnimation(figure, frame, frames=len(sequence), interval=200)
    animation.save(output / "visuals/learning_process.gif", writer=PillowWriter(fps=5), dpi=85)
    frame(len(sequence) - 1)
    figure.savefig(output / "visuals/learning_process_final.png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    write_json(output / "animation_round_sequence.json", {
        "clock": "Actual training round; repeated frames pause at early/final checkpoints",
        "rounds": sequence, "fps": 5,
        "source": "Saved learner-only evaluations, not interpolated model weights",
    })


def driving_animation(output, tracks, details):
    rounds, track_ids = [0, 1, 10, 120], [2, 3, 9]
    indexed = index_details(details)
    traces = {number: load_round(output, number, "train") for number in rounds}
    figure, axes = plt.subplots(3, 4, figsize=(13, 10))
    artists = []
    for row, track_id in enumerate(track_ids):
        for column, number in enumerate(rounds):
            axis = axes[row, column]
            draw_track(axis, tracks[track_id])
            metric = indexed[(number, track_id, 0)]
            axis.set_title(f"{CIRCUITS[track_id].name} / round {number}", fontsize=9)
            line, = axis.plot([], [], color=TEAL if metric["completed"] else RED, lw=1.5)
            marker, = axis.plot([], [], marker="o", color=GOLD, ms=4)
            status = axis.text(0.02, 0.02, "", transform=axis.transAxes, fontsize=8,
                               bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.9})
            artists.append((traces[number][trace_key(track_id)], line, marker, status))
    figure.suptitle("THE CAR DRIVING WITH DIFFERENT WEIGHTS  /  EQUAL SIMULATED TIME", fontsize=15)
    figure.tight_layout()
    times = np.linspace(0, max(item[0][-1, 0] for item in artists), 160)
    def frame(index):
        for trace, line, marker, status in artists:
            position = min(int(np.searchsorted(trace[:, 0], times[index])), len(trace) - 1)
            state = trace[position]
            line.set_data(trace[:position + 1, 1], trace[:position + 1, 2])
            marker.set_data([state[1]], [state[2]])
            status.set_text(f"{state[0]:.1f}s / {state[4] * 3.6:.0f} km/h / {min(state[7], 1) * 100:.1f}%")
    animation = FuncAnimation(figure, frame, frames=len(times), interval=100)
    animation.save(output / "visuals/driving_comparison.gif", writer=PillowWriter(fps=10), dpi=80)
    plt.close(figure)


def make_replay(output, tracks, history, details, manifest):
    indexed = index_details(details)
    frames = []
    for number in range(121):
        arrays = {**load_round(output, number, "train"), **load_round(output, number, "holdout")}
        frame = []
        for circuit in CIRCUITS:
            scenarios = []
            for scenario in range(3):
                trace = arrays[trace_key(circuit.id, scenario)]
                indices = np.unique(np.r_[np.arange(0, len(trace), 10), len(trace) - 1])
                scenarios.append({
                    "trace": np.round(trace[indices], 4).tolist(),
                    "metric": indexed[(number, circuit.id, scenario)],
                })
            frame.append(scenarios)
        frames.append(frame)
    payload = {
        "catalog": json.loads((output / "data/circuit_catalog.json").read_text()),
        "tracks": [
            {name: np.round(getattr(tracks[c.id], name)[::3], 3).tolist()
             for name in ("xy", "left", "right")} for c in CIRCUITS
        ],
        "history": history, "frames": frames, "manifest": manifest,
    }
    template = Path(__file__).with_name("multitrack_replay.html").read_text(encoding="utf-8")
    (output / "replay.html").write_text(
        template.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False)),
        encoding="utf-8",
    )


def make_deck(output, history, manifest):
    final, baseline = manifest["final"], manifest["baseline"]
    slides = [
        ("A DRIVER FOR ELEVEN CIRCUITS", [
            "Original Apex Park plus ten new F1-inspired track layouts.",
            "One shared neural driver; two additional unseen test-only circuits.",
            f"120 real training rounds and {manifest['actual_optimizer_updates']:,} weight updates.",
        ], "track_atlas", "All layouts are original educational circuits, not official F1 geometry. The original single-track experiment remains intact."),
        ("IMPROVE THE LEARNING PROCESS", [
            "Add curvature observations without changing the vehicle physics.",
            "Balance every minibatch across all eleven training circuits.",
            "Use harder recovery starts and retain historical examples.",
            "Freeze weights before testing unseen circuits.",
        ], "improvement_pipeline", "The full plan is IMPROVEMENT_PLAN.md. These changes are a combined experiment, not a controlled ablation of individual interventions."),
        ("MORE DATA, MORE REAL UPDATES", [
            f"{manifest['initial_samples']:,} initial examples → {manifest['final_samples']:,} final examples.",
            "Each round collects 180 teacher-labeled learner states per circuit.",
            "64 Adam updates per round; 44 minibatch examples per circuit.",
            "Independent validation examples never update weights.",
        ], "learning", "Half of a track's batch comes from its latest rollout, half from its complete history. Foundation, Recovery and Challenge phases change collection starts, never the fixed evaluation scenarios."),
        ("MEASURED MULTI-TRACK LEARNING", [
            f"Average training-track progress: {baseline['train_progress_pct']:.1f}% → {final['train_progress_pct']:.1f}%.",
            f"Final monitored starts: {final['train_completed']}/33 completed.",
            f"First round completing all 33: {manifest['first_all_training_starts_complete_round']}.",
            "All raw rounds, including regressions, are retained.",
        ], "progress", "Unseen curves were generated by re-evaluating frozen checkpoints after the whole training schedule; they were not training feedback."),
        ("DOES TECHNIQUE IMPROVE?", [
            f"Lane error: {history[1]['train_lateral_completed_m']:.3f} m in round 1 → {final['train_lateral_completed_m']:.3f} m.",
            f"Final average completed lap time: {final['train_lap_time_completed_s']:.1f} s.",
            "Quality comparisons exclude incomplete sets of laps.",
            "Lap time and smoothness were not direct learning objectives.",
        ], "quality", "Compare only rounds with 33 of 33 completed training-start laps. Short failed baseline drives cannot establish better smoothness."),
        ("UNSEEN TESTS AFTER FREEZING", [
            "Aurora and Delta never contribute training or validation examples.",
            f"Final unseen starts: {final['holdout_completed']}/6 completed.",
            "No checkpoint selection or hyperparameter tuning uses these results.",
            "Two unseen tracks and one seed are limited evidence, not proof of safety.",
        ], "unseen", "Both circuits were designed and assigned to the holdout split before the run. Teacher traversability tests validate geometry, not learned-policy performance."),
        ("WATCH THE SIMULATIONS", [
            "Learning-history GIF advances actual training rounds.",
            "Driving-comparison GIF advances equal simulated time.",
            "Interactive replay covers every round, track, and fixed test start.",
            "The previous single-track model is a separately labeled transfer baseline.",
        ], "track_atlas", "Use replay.html for training-grid playback and individual drives. Transfer comparison is not presented as the new model's untrained baseline."),
        ("WHAT THIS DOES NOT PROVE", [
            "Perfect geometry and kinematic physics, not real F1 tire or aero dynamics.",
            "No claim that every metric improves, or that this is the optimal driver.",
            "No ablation, multiple-seed confidence, or real-vehicle validation.",
            "Source, data, models, measurements and verification reports are included.",
        ], "transfer_comparison", "The old model may already transfer well. The transfer chart reports actual completed-lap outcomes and omits incomplete sets rather than comparing short crashes with full laps."),
    ]
    presentation(output, manifest, history, slides=slides)


def inventory(output):
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("artifact_inventory.json", "verification.json"):
            files.append({
                "path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    write_json(output / "artifact_inventory.json", files)


def render(output, skip_animations=False, inventory_only=False):
    output = Path(output)
    if inventory_only:
        inventory(output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest["success_criteria_met"]:
        raise RuntimeError("Training success criteria were not met")
    history = json.loads((output / "history.json").read_text())
    details = json.loads((output / "scenario_metrics.json").read_text())
    config = Config(**manifest["config"])
    tracks = {c.id: c.build(config) for c in CIRCUITS}
    make_charts(output, tracks, history, details, manifest)
    make_replay(output, tracks, history, details, manifest)
    make_deck(output, history, manifest)
    print("Rendered track atlas, improvement plan, measured charts, all-round replay and new deck.", flush=True)
    if not skip_animations:
        training_animation(output, tracks, history, details)
        driving_animation(output, tracks, details)
        print("Rendered every-round learning movie and equal-time driving comparison.", flush=True)
    inventory(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multitrack_artifacts")
    parser.add_argument("--skip-animations", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    arguments = parser.parse_args()
    render(arguments.output, arguments.skip_animations, arguments.inventory_only)
