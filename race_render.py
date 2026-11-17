"""Per-track moving-car comparisons, cockpit telemetry, and learning charts."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt

from circuits import CIRCUITS
from driving import Config, track_from_catalog, write_json
from multi_render import inventory
from multitrack import save_csv
from render import BLUE, GOLD, MUTED, PANEL, RED, TEAL, WHITE, draw_track, presentation, save_figure


def load_attempt(output, item, number):
    with np.load(output / f"attempts/{item['split']}_{number:03d}.npz") as data:
        return data[f"track_{item['id']:02d}"].copy()


def thin_trace(trace):
    changes = np.flatnonzero(
        (np.diff(trace[:, 10]) != 0)
        | (np.diff(trace[:, 16]) != 0)
        | (np.diff(trace[:, 9] > 0.02) != 0)
        | (np.abs(np.diff(trace[:, 9])) > 0.15)
        | (np.abs(np.diff(trace[:, 8])) > 0.25)
    )
    indices = np.unique(np.r_[
        np.arange(0, len(trace), 5), changes, changes + 1,
        np.argmax(trace[:, 9]), np.argmax(trace[:, 4]), len(trace) - 1,
    ])
    return np.round(trace[indices], 5).tolist()


def car_parts(axis, color):
    parts = []
    body = Polygon([[4.4, 0], [2.4, 1.1], [-2.8, 1.2], [-4, 0],
                    [-2.8, -1.2], [2.4, -1.1]], color=color, zorder=8)
    axis.add_patch(body)
    parts.append(body)
    for x, y, w, h in [
        (3.8, -2.8, 0.8, 5.6), (-4.1, -2.6, 0.8, 5.2),
        (1.7, -2.9, 2.0, 1.1), (1.7, 1.8, 2.0, 1.1),
        (-3.2, -2.9, 2.0, 1.1), (-3.2, 1.8, 2.0, 1.1),
    ]:
        part = Rectangle((x, y), w, h, facecolor="#05090f", edgecolor=MUTED, lw=0.3, zorder=9)
        axis.add_patch(part)
        parts.append(part)
    cockpit = Polygon([[-1.5, -0.65], [0.7, -0.65], [0.7, 0.65], [-1.5, 0.65]],
                      color=WHITE, zorder=10)
    axis.add_patch(cockpit)
    parts.append(cockpit)
    lamps = []
    for y in (-1.6, 1):
        lamp = Rectangle((-3.6, y), 0.6, 0.6, color="#54252a", zorder=11)
        axis.add_patch(lamp)
        parts.append(lamp)
        lamps.append(lamp)
    return parts, lamps


def pose_car(axis, parts, lamps, state):
    transform = Affine2D().scale(2.5).rotate(state[3]).translate(state[1], state[2]) + axis.transData
    for part in parts:
        part.set_transform(transform)
    for lamp in lamps:
        lamp.set_facecolor(RED if state[9] > 0.02 else "#54252a")


def charts(output, item, track, records, final_trace):
    name = f"track_{item['id']:02d}"
    directory = output / "visuals"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    numbers = [row["round"] for row in records]
    axes[0].plot(numbers, [row["progress"] * 100 for row in records], color=TEAL)
    axes[0].set(ylabel="Lap progress (%)", xlabel="Saved model attempt", ylim=(0, 105))
    completed = [row for row in records if row["completed"]]
    axes[1].plot([row["round"] for row in completed], [row["mean_abs_lateral_m"] for row in completed],
                 color=GOLD, marker=".", ms=3)
    axes[1].set(ylabel="Completed-lap lane error (m)", xlabel="Saved model attempt")
    for axis in axes:
        axis.grid(alpha=0.4)
    figure.suptitle(f"{item['name']} / ALL 121 ATTEMPTS", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, f"{name}_learning")
    figure, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(final_trace[:, 0], final_trace[:, 4] * 3.6, color=TEAL, label="Actual speed")
    axes[0].plot(final_trace[:, 0], final_trace[:, 13] * 3.6, color=MUTED, ls="--", label="Learned target speed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(fontsize=9)
    axes[1].plot(final_trace[:, 0], final_trace[:, 8] * 100, color=TEAL, label="Throttle")
    axes[1].plot(final_trace[:, 0], final_trace[:, 9] * 100, color=RED, label="Brake")
    axes[1].set(ylabel="Pedal (%)", ylim=(-3, 103))
    axes[1].legend(fontsize=9)
    axes[2].step(final_trace[:, 0], final_trace[:, 10], where="post", color=BLUE)
    axes[2].set(ylabel="Gear", yticks=list(range(1, 9)), ylim=(0.8, 8.2))
    axes[3].plot(final_trace[:, 0], final_trace[:, 11], color=GOLD)
    axes[3].set(ylabel="Engine RPM", xlabel="Actual simulated driving time (s)")
    for axis in axes:
        axis.grid(alpha=0.4)
    figure.suptitle(f"{item['name']} / FINAL DRIVER TELEMETRY", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, f"{name}_telemetry")
    save_csv(output / f"data/{name}_attempts.csv", records)


def attempt_movie(output, item, track, records):
    early = next((row["round"] for row in records if row["completed"] and row["round"] > 0), 1)
    selected = [0, early, 120]
    traces = [load_attempt(output, item, number) for number in selected]
    figure = plt.figure(figsize=(12, 7.5))
    grid = figure.add_gridspec(2, 6, height_ratios=[2.0, 1.0], hspace=0.35, wspace=0.6)
    artists = []
    for index, (number, trace, color, label) in enumerate(zip(
        selected, traces, [RED, GOLD, TEAL], ["UNTRAINED", "EARLY LEARNER", "FINAL DRIVER"],
    )):
        axis = figure.add_subplot(grid[0, index * 2:index * 2 + 2])
        draw_track(axis, track)
        axis.set_title(f"{label} / attempt {number}", fontsize=10)
        trail, = axis.plot([], [], color=color, lw=1.3)
        parts, lamps = car_parts(axis, color)
        text = axis.text(0.02, 0.04, "", transform=axis.transAxes, fontsize=8,
                         bbox={"facecolor": PANEL, "edgecolor": "none", "alpha": 0.95})
        artists.append((axis, trace, trail, parts, lamps, text, records[number]))
    progress = figure.add_subplot(grid[1, :3])
    progress.plot([row["round"] for row in records], [row["progress"] * 100 for row in records], color=TEAL, lw=1.6)
    progress.set(xlabel="Training attempt / model checkpoint", ylabel="Lap progress (%)", ylim=(0, 105))
    accuracy = figure.add_subplot(grid[1, 3:])
    completed = [row for row in records if row["completed"]]
    accuracy.plot([row["round"] for row in completed], [row["mean_abs_lateral_m"] for row in completed], color=GOLD, lw=1.5)
    accuracy.set(xlabel="Training attempt / model checkpoint", ylabel="Completed-lap error (m)")
    for axis in (progress, accuracy):
        axis.grid(alpha=0.4)
        for number in selected:
            axis.axvline(number, color=MUTED, ls=":", lw=0.7)
    figure.suptitle(f"{item['name'].upper()} / WATCH THE CAR LEARN", fontsize=17)
    figure.text(0.12, 0.01, "Equal simulated time. Failed cars freeze at their recorded endpoint. Car icons enlarged.",
                color=MUTED, fontsize=9)
    times = np.linspace(0, max(trace[-1, 0] for trace in traces), 200)
    def frame(index):
        for axis, trace, trail, parts, lamps, text, metric in artists:
            cursor = min(int(np.searchsorted(trace[:, 0], times[index])), len(trace) - 1)
            state = trace[cursor]
            trail.set_data(trace[:cursor + 1, 1], trace[:cursor + 1, 2])
            pose_car(axis, parts, lamps, state)
            ending = metric["end_reason"].replace("_", " ") if cursor == len(trace) - 1 else "driving"
            text.set_text(
                f"{state[0]:.1f}s / {state[4] * 3.6:.0f} km/h / G{int(state[10])}\n"
                f"Brake {state[9] * 100:.0f}% / Throttle {state[8] * 100:.0f}% / {state[11]:.0f} RPM\n"
                f"{ending} / {min(max(state[7], 0), 1) * 100:.1f}%"
            )
    animation = FuncAnimation(figure, frame, frames=len(times), interval=80)
    animation.save(output / f"visuals/track_{item['id']:02d}_attempts.gif",
                   writer=PillowWriter(fps=12), dpi=80)
    frame(len(times) - 1)
    figure.savefig(output / f"visuals/track_{item['id']:02d}_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return selected


def make_page(output, item, track, records, rollout_metrics, manifest):
    attempts = [
        {"round": number, "metric": records[number],
         "trace": thin_trace(load_attempt(output, item, number))}
        for number in range(121)
    ]
    rollouts = []
    if item["split"] == "train":
        for number in range(1, 121):
            with np.load(output / f"rollouts/round_{number:03d}.npz") as arrays:
                trace = arrays[f"track_{item['id']:02d}"].copy()
            rollouts.append({
                **rollout_metrics[(number, item["id"])], "trace": thin_trace(trace),
            })
    data = {
        "circuit": item, "config": manifest["config"], "attempts": attempts, "rollouts": rollouts,
        "track": {key: np.round(getattr(track, key)[::2], 4).tolist()
                  for key in ("xy", "left", "right")},
        "early_round": next((row["round"] for row in records if row["round"] > 0 and row["completed"]), 1),
    }
    template = Path(__file__).with_name("race_track.html").read_text(encoding="utf-8")
    (output / f"tracks/track_{item['id']:02d}.html").write_text(
        template.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)),
        encoding="utf-8",
    )


def make_index(output, catalog, all_records, manifest):
    cards = []
    for item in catalog:
        rows = all_records[item["id"]]
        first = next((row for row in rows if row["completed"]), None)
        final = rows[-1]
        error = (f"{first['mean_abs_lateral_m']:.3f} → {final['mean_abs_lateral_m']:.3f} m"
                 if first and final["completed"] else "See measured attempt history")
        cards.append(
            f'<article><h2>{html.escape(item["name"])}</h2>'
            f'<p>{item["split"].upper()} · curvature reversals {item["previous_reversals"]} → {item["reversals"]}'
            f' · lane error {error}</p>'
            f'<img loading="lazy" src="visuals/track_{item["id"]:02d}_attempts.gif" '
            f'alt="Moving cars on {html.escape(item["name"])} with speed braking gears and learning curves">'
            f'<p><a href="tracks/track_{item["id"]:02d}.html">Every attempt and live telemetry</a>'
            f' · <a href="visuals/track_{item["id"]:02d}_attempts.gif">Animation</a></p></article>'
        )
    options = "".join(f'<option value="{item["id"]}">{html.escape(item["name"])} · {item["split"]}</option>'
                      for item in catalog)
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GP-X Driver Lab</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101722;color:#ecf2fa;font:16px system-ui}
header,main{max-width:1500px;margin:auto;padding:24px}h1{font-size:32px;margin:8px 0}p{color:#9bacbf;line-height:1.6}
select{font:inherit;color:#ecf2fa;background:#182333;border:1px solid #4de2c2;padding:12px;border-radius:8px}
iframe{width:100%;height:1370px;border:1px solid #334358;border-radius:12px;background:#182333;margin-top:20px}
.gallery{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:#182333;border:1px solid #334358;border-radius:12px;padding:18px}
article img{width:100%;border-radius:8px}a{color:#4de2c2}h2{font-size:21px}.badge{color:#4de2c2;font-weight:bold}
@media(max-width:800px){.gallery{grid-template-columns:1fr}iframe{height:1850px}}
</style></head><body><header><div class="badge">GP-X / DRIVER LEARNING LAB</div>
<h1>Harder tracks. Visible cars. Every driving attempt.</h1>"""
    document += (
        f"<p>11 harder training circuits + 2 unseen circuits · 120 actual training rounds · "
        f"{manifest['final_training_starts_completed']}/33 final training starts and "
        f"{manifest['final_holdout_starts_completed']}/6 unseen starts completed.</p>"
        "<p>Choose any circuit below. Watch every saved attempt, compare early and final drivers, "
        "or inspect actual training rollouts. Speed, brakes, throttle, gear and RPM are recorded simulation values.</p>"
        f'<label for="trackSelect">Circuit </label><select id="trackSelect">{options}</select>'
        '<p><a href="../RACE_PLAN.md">Experiment plan</a> · <a href="../RACE_RESULTS.md">Results and reproduction</a>'
        ' · <a href="training_deck.html">Training deck</a>'
        ' · <a href="data/circuits.json">Complex circuit geometry</a></p></header><main>'
        '<iframe id="cockpit" src="tracks/track_00.html" title="Per-track moving-car telemetry and attempt player"></iframe>'
        '<h1>Training visuals for every circuit</h1><p>Each animation below has a moving car, pedal/gear telemetry, '
        'and that track’s learning curves. Full histories—not just selected checkpoints—are available in each cockpit page.</p>'
        '<div class="gallery">' + "".join(cards) + "</div>"
        "<p>Eight-speed automatic gearbox and engine physics are fictional. The neural driver learns steering and target speed; "
        "actuator control and shifting are deterministic. Not real F1 telemetry or a physical-car controller.</p>"
        '</main><script>document.getElementById("trackSelect").addEventListener("change",e=>{'
        'document.getElementById("cockpit").src=`tracks/track_${String(e.target.value).padStart(2,"0")}.html`});'
        "</script></body></html>"
    )
    (output / "index.html").write_text(document, encoding="utf-8")


def race_deck(output):
    manifest = json.loads((output / "manifest.json").read_text())
    history = json.loads((output / "history.json").read_text())
    first = next(row for row in history if row["completed_tracks"] == 11)
    slides = [
        ("HARDER TRACKS, A LEARNING DRIVER", [
            "11 training circuits and 2 unseen layouts, with four added technical sectors each.",
            "120 real training rounds. Visible cars and telemetry for every track.",
            "Fictional F1-inspired physics, not a real vehicle controller.",
        ], "complex_track_atlas", "Dotted lines show the prior scaled layout. The new solid route adds technical chicanes. All previous experiment artifacts remain unchanged."),
        ("HOW THE DRIVER LEARNS", [
            "A pure-pursuit teacher labels steering and curvature-aware target speed.",
            "16 inputs → 48 → 32 → 2 tanh neural network; NumPy backpropagation and Adam.",
            "Imitation learning, not reinforcement learning.",
        ], "track_00_learning", "Inputs include lane and heading error, speed, prior steering, geometric previews and three curvature observations. Pedals and shifts are deterministic actuator outputs, not independent learned actions."),
        ("THE ACTUAL TRAINING PROCESS", [
            f"{manifest['initial_samples']:,} teacher/recovery examples grow to {manifest['final_samples']:,}.",
            "220 new labeled states per track per round; balanced recent/historical batches.",
            "7,680 optimizer updates. Randomized starts become progressively harder.",
        ], "track_06_learning", "DAgger-style aggregation labels states visited by the current learner. Collection N uses model N-1. All 1,320 first continuous training episodes are recorded without joining reset teleports."),
        ("WATCH THE SIMULATIONS", [
            "Each track has 121 identical-start, unassisted model test attempts.",
            "Moving cars have wings, wheels, brake lights and real recorded telemetry.",
            "Compare attempts or autoplay 0–120; actual training episodes are separate.",
        ], "track_00_comparison", "Open the dashboard and select any circuit. Comparisons use equal simulated time; failed cars freeze at the true endpoint. Icons are enlarged. The offline browser replays recorded training, not new online optimization."),
        ("BRAKES, SPEED AND GEARS", [
            "Learned target speed drives actual throttle and braking inputs.",
            "Eight-speed automatic gearbox: RPM thresholds, gear-dependent torque and shift interruption.",
            "Every cockpit shows speed, pedals, gear, RPM and a driving timeline.",
        ], "track_04_telemetry", "Braking subtracts acceleration in the simulator. Gear changes affect available acceleration, with 0.2-second torque interruptions. The observed final canonical drives use up to gears 4–5, not all eight. No claim of authentic F1 powertrain behavior."),
        ("MEASURED IMPROVEMENT", [
            "Untrained: 0/11 complete, 0.14% mean progress. Final: 11/11 and 100%.",
            f"First all-track completion: round {first['round']}; lane error {first['mean_completed_lateral_m']:.3f} → {manifest['final']['mean_completed_lateral_m']:.3f} m.",
            "Action-validation error falls about 148-fold; raw regressions are retained.",
        ], "track_06_learning", "The lane-error comparison is at a common checkpoint for all eleven canonical tests, not short failed drives. Accuracy and lap time do not improve monotonically. Full attempt tables and CSV data allow independent inspection."),
        ("UNSEEN TRACKS AND REPRODUCIBILITY", [
            "Final model finishes all 33 training starts and all 6 unseen starts.",
            "Unseen checkpoints are evaluated only after training and weight freezing.",
            "Final trajectories re-drive exactly; datasets, weights and source are included.",
        ], "track_11_comparison", "Only two held-out layouts and one random seed were tested. There is no claim of universal generalization, sensor-noise robustness, racing strategy or real-world safety. Test outcomes did not tune the training schedule."),
        ("EXPLORE EVERY CIRCUIT", [
            "13 standalone animations and 13 interactive cockpit pages.",
            "1,573 fixed attempts; 1,320 actual training episodes; all telemetry retained.",
            "Inspect the dashboard, results, numerical verification and browser screenshots.",
        ], "track_12_telemetry", "Open index.html offline. Choose sequence, single, comparison or training-rollout mode. Inspect brakes and gears using the timeline. RACE_RESULTS.md contains reproduction commands and per-track measured results."),
    ]
    presentation(output, manifest, history, slides=slides, replay_href="index.html")


def render(output, skip_movies=False, inventory_only=False, deck_only=False):
    output = Path(output)
    if inventory_only:
        inventory(output)
        return
    if deck_only:
        race_deck(output)
        catalog = json.loads((output / "data/circuits.json").read_text())
        metrics = json.loads((output / "attempt_metrics.json").read_text())
        rows = {item["id"]: [row for row in metrics if row["track_id"] == item["id"]] for item in catalog}
        make_index(output, catalog, rows, json.loads((output / "manifest.json").read_text()))
        inventory(output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest["success_criteria_met"]:
        raise RuntimeError("Training did not meet its completion criteria")
    catalog = json.loads((output / "data/circuits.json").read_text())
    metrics = json.loads((output / "attempt_metrics.json").read_text())
    all_records = {item["id"]: [row for row in metrics if row["track_id"] == item["id"]] for item in catalog}
    rollout_metrics = {(row["collection_round"], row["track_id"]): row
                       for row in json.loads((output / "rollout_metrics.json").read_text())}
    config = Config(**manifest["config"])
    figure, axes = plt.subplots(4, 4, figsize=(14, 12))
    for axis, item in zip(axes.flat, catalog):
        track = track_from_catalog(config, item)
        draw_track(axis, track)
        base = CIRCUITS[item["id"]].build(config)
        axis.plot(*(base.xy * 1.15).T, color=MUTED, ls=":", lw=0.8)
        axis.plot(*track.xy.T, color=TEAL if item["split"] == "train" else BLUE, lw=1)
        axis.set_title(f"{item['name']}\nturn reversals {item['previous_reversals']} → {item['reversals']}", fontsize=9)
    for axis in list(axes.flat)[len(catalog):]:
        axis.axis("off")
    figure.suptitle("HARDER GP-X CIRCUITS / FOUR NEW TECHNICAL SECTORS PER TRACK", fontsize=17)
    figure.tight_layout()
    save_figure(figure, output / "visuals", "complex_track_atlas", svg=True)
    movie_rounds = {}
    for item in catalog:
        track = track_from_catalog(config, item)
        records = all_records[item["id"]]
        charts(output, item, track, records, load_attempt(output, item, 120))
        make_page(output, item, track, records, rollout_metrics, manifest)
        if not skip_movies:
            movie_rounds[str(item["id"])] = attempt_movie(output, item, track, records)
        print(f"Rendered {item['name']}: per-attempt learning, speed/brake/gear telemetry and moving cars.", flush=True)
    make_index(output, catalog, all_records, manifest)
    if not skip_movies:
        write_json(output / "animation_manifest.json", {
            "per_track_comparison_rounds": movie_rounds,
            "interactive_attempts_per_track": 121,
            "clock": "Equal simulated driving time within each comparison; all-round sequence in interactive player",
            "car": "Visible body, wings, wheels and brake lights tied to recorded braking",
        })
        race_deck(output)
    inventory(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="race_artifacts")
    parser.add_argument("--skip-movies", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--deck-only", action="store_true")
    arguments = parser.parse_args()
    render(arguments.output, arguments.skip_movies, arguments.inventory_only, arguments.deck_only)
