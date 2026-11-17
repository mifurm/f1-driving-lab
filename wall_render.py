"""Per-track close-up corner, wall-contact and learning visualizations."""

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import numpy as np

from driving import Config, Track, write_json
from multi_render import inventory
from multitrack import save_csv
from race_render import car_parts
from render import BLUE, GOLD, MUTED, RED, TEAL, draw_track, presentation, save_figure
from wall_sim import footprint


def load_attempt(output, item, number, scenario):
    with np.load(output / f"attempts/{item['split']}_{number:03d}.npz") as data:
        return data[f"track_{item['id']:02d}_scenario_{scenario}"]


def thin(trace):
    changes = np.flatnonzero((np.diff(trace[:, 10]) != 0) | (np.diff(trace[:, 16]) != 0)
                             | (np.diff(trace[:, 22]) != 0) | (np.diff(trace[:, 9] > .02) != 0))
    return np.round(trace[np.unique(np.r_[np.arange(0, len(trace), 3), changes, changes + 1,
                                         np.argmin(trace[:, 17]), len(trace) - 1])], 6).tolist()


def metrics_for(records, track_id, number):
    return [row for row in records if row["track_id"] == track_id and row["round"] == number]


def render_track(output, item, track, records, scenarios, manifest, movies):
    rows = [row for row in records if row["track_id"] == item["id"]]
    names = [scenario["name"] for scenario in scenarios]
    attempts = [
        [{"round": number, "metric": next(row for row in rows if row["round"] == number and row["scenario"] == names[index]),
          "trace": thin(load_attempt(output, item, number, index))}
         for index in range(4)] for number in range(manifest["rounds"] + 1)
    ]
    rollouts = []
    if item["split"] == "train":
        rollout_records = json.loads((output / "rollout_metrics.json").read_text())
        for number in range(1, manifest["rounds"] + 1):
            with np.load(output / f"rollouts/round_{number:03d}.npz") as data:
                trace = data[f"track_{item['id']:02d}"]
            metric = next(row for row in rollout_records if row["track_id"] == item["id"] and row["collection_round"] == number)
            rollouts.append({**metric, "trace": thin(trace)})
    with np.load(output / "attempts/unsafe_demonstrations.npz") as data:
        demonstration = thin(data[f"track_{item['id']:02d}"])
    default = min((2, 3), key=lambda index: (attempts[0][index]["metric"]["success"],
                                             attempts[0][index]["metric"]["min_clearance_m"]))
    payload = {"circuit": item, "scenarios": scenarios, "attempts": attempts, "rollouts": rollouts,
               "unsafe": demonstration, "default_scenario": default, "rounds": manifest["rounds"],
               "selected_round": manifest["selected_round"],
               "track": {"xy": track.xy.tolist(), "left": track.left.tolist(), "right": track.right.tolist(),
                         "length": track.length, "ds": track.ds, "tangent": track.tangent.tolist()}}
    template = Path(__file__).with_name("wall_track.html").read_text()
    (output / f"tracks/track_{item['id']:02d}.html").write_text(
        template.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False)))
    save_csv(output / f"data/track_{item['id']:02d}_metrics.csv", rows)
    numbers = list(range(manifest["rounds"] + 1))
    safe = [sum(row["success"] for row in metrics_for(rows, item["id"], number)) for number in numbers]
    margin = [min(attempts[number][index]["metric"]["min_clearance_m"] for index in (2, 3)) for number in numbers]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(numbers, safe, color=TEAL)
    axes[0].set(xlabel="Additional training round", ylabel="Safe centered corner passages / 4", ylim=(-.1, 4.2))
    axes[1].plot(numbers, margin, color=GOLD)
    axes[1].set(xlabel="Additional training round", ylabel="Worst edge-recovery wall clearance (m)")
    for axis in axes:
        axis.grid(alpha=.4)
    figure.suptitle(item["name"] + " / LEARNING TO STAY ON TRACK")
    figure.tight_layout()
    save_figure(figure, output / "visuals", f"track_{item['id']:02d}_learning")
    if not movies:
        return
    early = next((number for number in numbers[1:] if attempts[number][default]["metric"]["success"]), manifest["rounds"])
    selected = [0, early, manifest["selected_round"]]
    traces = [load_attempt(output, item, number, default) for number in selected]
    figure = plt.figure(figsize=(12, 7))
    grid = figure.add_gridspec(2, 6, height_ratios=[2, 1], wspace=.6, hspace=.4)
    artists = []
    for index, (number, trace, color) in enumerate(zip(selected, traces, [GOLD, BLUE, TEAL])):
        axis = figure.add_subplot(grid[0, index * 2:index * 2 + 2])
        draw_track(axis, track)
        axis.plot(*track.left.T, color=RED, lw=1)
        axis.plot(*track.right.T, color=RED, lw=1)
        axis.set_title(["PREVIOUS GP-X", f"EXTENSION {number}", f"SELECTED {number}"][index], fontsize=10)
        trail, = axis.plot([], [], color=color, lw=1)
        parts, lamps = car_parts(axis, color)
        box = Polygon(np.zeros((4, 2)), fill=False, edgecolor=TEAL, lw=.8, zorder=12)
        axis.add_patch(box)
        text = axis.text(.02, .03, "", transform=axis.transAxes, fontsize=8,
                         bbox={"facecolor": "#182333", "edgecolor": "none", "alpha": .95})
        artists.append((axis, trace, trail, parts, lamps, box, text, attempts[number][default]["metric"]))
    a, b = figure.add_subplot(grid[1, :3]), figure.add_subplot(grid[1, 3:])
    a.plot(numbers, safe, color=TEAL)
    a.set(xlabel="Additional training round", ylabel="Safe passages / 4", ylim=(-.1, 4.2))
    b.plot(numbers, margin, color=GOLD)
    b.set(xlabel="Additional training round", ylabel="Worst edge margin (m)")
    for axis in (a, b):
        axis.grid(alpha=.4)
    figure.suptitle(item["name"] + " / CORNERS WITHOUT WALL CONTACT", fontsize=16)
    figure.text(.12, .01, names[default] + " | Follow cameras, equal simulated clock. No steering shield.", fontsize=9, color=MUTED)
    times = np.linspace(0, max(trace[-1, 0] for trace in traces), 150)
    def frame(index):
        for axis, trace, trail, parts, lamps, box, text, metric in artists:
            cursor = min(int(np.searchsorted(trace[:, 0], times[index])), len(trace) - 1)
            state = trace[cursor]
            axis.set_xlim(state[1] - 36, state[1] + 36)
            axis.set_ylim(state[2] - 36, state[2] + 36)
            trail.set_data(trace[:cursor + 1, 1], trace[:cursor + 1, 2])
            transform = Affine2D().scale(.5, .30).rotate(state[3]).translate(state[1], state[2]) + axis.transData
            for part in parts:
                part.set_transform(transform)
            for lamp in lamps:
                lamp.set_facecolor(RED if state[9] > .02 else "#54252a")
            c, s = np.cos(state[3]), np.sin(state[3])
            rectangle = np.array([[2.4, .9], [2.4, -.9], [-2.4, -.9], [-2.4, .9]])
            box.set_xy(rectangle @ np.array([[c, s], [-s, c]]) + state[1:3])
            box.set_edgecolor(RED if state[22] else TEAL)
            ending = metric["end_reason"].replace("_", " ") if cursor == len(trace) - 1 else "driving"
            text.set_text(f"{state[0]:.1f}s | {state[4]*3.6:.0f} km/h | G{int(state[10])}\n"
                          f"Wall margin {state[17]:.2f} m | Brake {state[9]*100:.0f}%\n{ending}")
    animation = FuncAnimation(figure, frame, frames=len(times), interval=83)
    animation.save(output / f"visuals/track_{item['id']:02d}_corners.gif", writer=PillowWriter(fps=12), dpi=80)
    frame(len(times) - 1)
    figure.savefig(output / f"visuals/track_{item['id']:02d}_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(figure)


def deck(output, manifest):
    slides = [
        ("LEARNING TO STAY ON TRACK", [
            "Continue the GP-X model for 120 additional training rounds.",
            "Two corner sections and two outward near-wall recoveries per track.",
            "13 tracks, real recorded controls, close-up moving cars and learning curves.",
        ], "corner_atlas", "Eleven layouts train the policy; two remain unseen. This extension starts from pretrained GP-X weights, not a new untrained model."),
        ("HOW WALL CONTACT IS SIMULATED", [
            "A 4.8 m x 1.8 m oriented rectangle includes front and rear corners.",
            "Test the footprint against both wall polylines throughout each step.",
            "No safety shield alters the learned controls.",
        ], "track_01_comparison", "Sweep samples are at most 0.5 m apart and 0.025 rad apart. Wall contact terminates the drive; there is no impact, bounce or damage model. The unsafe scripted demo is not an ML result."),
        ("WHAT THE ML DRIVER CAN OBSERVE", [
            "Lane offset, heading error, speed and forward track previews.",
            "Add projected footprint margin and speed-dependent steering-budget inputs.",
            "Both start with zero weights, preserving the previous policy before training.",
        ], "track_05_learning", "The two derived features are observations, not output clamps. Actual curved-wall clearance is independently measured. The final network predicts ordinary steering and target speed; no controller enforces the budget."),
        ("THE EXTENDED TRAINING", [
            "Balanced corner/edge rollouts receive expert labels for steering and target speed.",
            "Centerline anchor examples protect previously learned lap driving.",
            f"7,680 new Adam updates; safety-selected checkpoint {manifest['selected_round']}.",
        ], "track_01_learning", "Legacy checkpoints lack Adam moments, so weights continue with a fresh optimizer state. Selection requires all training sections and full training laps, then minimizes action-validation error. Test/stress outcomes never select the model. All 121 checkpoints, including regressions at round 120, remain available."),
        ("MEASURED SAFETY, NOT CLAIMED SAFETY", [
            f"Canonical sections: {manifest['baseline']['safe_passages']}/44 before, {manifest['final_safe_train']}/44 after.",
            f"Final unseen sections: {manifest['final_safe_unseen']}/8. Full laps: {manifest['full_lap_train']}/33 + {manifest['full_lap_unseen']}/6.",
            f"Blind stress: {manifest['stress_previous_safe']}/208 before, {manifest['stress_final_safe']}/208 after.",
        ], "track_07_learning", f"Stress wall contacts: {manifest['stress_previous_contacts']} before, {manifest['stress_final_contacts']} after. Failure categories distinguish wall contact, excessive lateral acceleration, timeout and uncentered exits. A failed prototype previously evaluated the layout holdouts; they remain data-disjoint but are not pristine unseen tests."),
        ("WATCH THE SIMULATIONS", [
            "Select a circuit and corner/recovery challenge; play every additional round.",
            "Follow the car, inspect steering, footprint, wall margin and ideal sensor guides.",
            "Use the clearly labeled scripted demo to see an actual wall contact.",
        ], "track_12_comparison", "All results are recorded simulations. The model learns steering and target speed; pedals and gears remain automatic. Raw failures and regressions are retained. No real-world vehicle safety claim."),
    ]
    presentation(output, manifest, [], slides=slides, replay_href="index.html")


def render(output, skip_movies=False, inventory_only=False):
    output = Path(output)
    if inventory_only:
        inventory(output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    catalog = json.loads((output / "data/circuits.json").read_text())
    scenarios = json.loads((output / "data/scenarios.json").read_text())
    records = json.loads((output / "attempt_metrics.json").read_text())
    config = Config(**manifest["config"])
    figure, axes = plt.subplots(4, 4, figsize=(14, 12))
    for axis, item in zip(axes.flat, catalog):
        track = Track(config, item["knots"])
        draw_track(axis, track)
        axis.plot(*track.left.T, color=RED, lw=.7)
        axis.plot(*track.right.T, color=RED, lw=.7)
        for index in (np.argmax(track.curvature), np.argmin(track.curvature)):
            axis.plot(*track.xy[index], "o", color=GOLD, ms=5)
        axis.set_title(item["name"], fontsize=9)
    for axis in list(axes.flat)[len(catalog):]:
        axis.axis("off")
    figure.suptitle("CORNERS AND WALLS / 13 TRACKS, FOUR FIXED CHALLENGES EACH")
    figure.tight_layout()
    save_figure(figure, output / "visuals", "corner_atlas")
    for item in catalog:
        render_track(output, item, Track(config, item["knots"]), records,
                     scenarios[str(item["id"])], manifest, not skip_movies)
        print(f"Rendered wall/corner learning for {item['name']}", flush=True)
    options = "".join(f'<option value="{item["id"]}">{html.escape(item["name"])} / {item["split"]}</option>' for item in catalog)
    cards = "".join(
        f'<article><h2>{html.escape(item["name"])}</h2>'
        f'<a href="tracks/track_{item["id"]:02d}.html">All corners, recovery attempts and sensor views</a>'
        f'<img loading="lazy" src="visuals/track_{item["id"]:02d}_corners.gif" alt="Corner and wall-recovery learning">'
        f'<a href="visuals/track_{item["id"]:02d}_corners.gif">Open animation</a></article>' for item in catalog)
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learning corners and wall avoidance</title><style>
:root{{color-scheme:dark}}body{{background:#101722;color:#ecf2fa;font:16px system-ui;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}
a{{color:#4de2c2}}p{{color:#9bacbf;line-height:1.6}}select{{padding:12px;background:#182333;color:#ecf2fa;font:inherit}}
iframe{{width:100%;height:1500px;border:1px solid #334358;margin:20px 0}}.gallery{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
article{{padding:18px;background:#182333;border-radius:12px}}img{{width:100%;margin:12px 0}}@media(max-width:800px){{.gallery{{grid-template-columns:1fr}}iframe{{height:2200px}}}}
</style><main><h1>Taking corners. Recovering near walls. Staying on track.</h1>
<p>120 additional training rounds on 11 circuits, plus 2 unseen tests. Canonical safe passages:
{manifest['baseline']['safe_passages']}/44 before → {manifest['final_safe_train']}/44 after; unseen {manifest['final_safe_unseen']}/8.
Full-lap regression: {manifest['full_lap_train']}/33 training and {manifest['full_lap_unseen']}/6 unseen starts.
Safety-selected checkpoint {manifest['selected_round']}; all rounds 0–120, including late regressions, are retained.</p>
<p>Watch the oriented car footprint and front-wheel steering. Green/red wall margins are recorded physical measurements.
The model sees ideal geometry, projected footprint margin and a steering-budget feature; neither clamps its controls.</p>
<p><a href="training_deck.html">Updated training deck</a> · <a href="../WALL_RESULTS.md">Measured results and reproduction</a></p>
<label for="trackSelect">Track </label><select id="trackSelect">{options}</select>
<iframe id="cockpit" src="tracks/track_00.html" title="Corner and wall-learning cockpit"></iframe>
<h1>Corner and wall-learning visuals for every track</h1><div class="gallery">{cards}</div>
<p>Wall-contact demos are explicitly scripted illustrations, not ML results. Prior policy already avoids walls on many tests;
we report real failures, not fabricated crashes. Perfect sensors, sampled collisions and fictional vehicle dynamics.</p></main>
<script>document.getElementById("trackSelect").onchange=e=>document.getElementById("cockpit").src=
`tracks/track_${{String(e.target.value).padStart(2,"0")}}.html`;</script></html>"""
    (output / "index.html").write_text(document)
    if not skip_movies:
        deck(output, manifest)
    inventory(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="wall_artifacts")
    parser.add_argument("--skip-movies", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    render(args.output, args.skip_movies, args.inventory_only)
