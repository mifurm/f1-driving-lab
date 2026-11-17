"""Build the offline student pages, measured charts, animation, classroom decks and teacher materials."""

import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt

from driving import NeuralDriver, track_from_catalog, tracks_from_catalog, write_json
from student_experiments import ABLATION_SEEDS, JOURNEY_PRIVATE, MASKS, export_model, lap, new_catalog
from wall_sim import WALL_CONFIG, corner_scenarios, footprint, wall_simulate

KIT_PATH = Path("teaching_kit.json")
SCRIPTS = (("student_engine.js", "engine.js"), ("student_tools.js", "tools.js"), ("student_rl.js", "rl.js"), ("student_journey.js", "journey.js"),
           ("student_app.js", "app.js"), ("student_labs.js", "labs.js"), ("student_journey_ui.js", "journey_ui.js"))
MASK_NAMES = {"full": "All inputs", "no_future_geometry": "No previews/curvature", "no_heading": "No direct heading", "no_margin": "No margin"}
COLORS = {"full": "#4de2c2", "no_future_geometry": "#ff8990", "no_heading": "#76a9ff", "no_margin": "#ffc56b"}


def script_safe(value):
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def bootstrap(values, seed=17, iterations=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.min()), float(values.max())
    rng = np.random.default_rng(seed)
    means = np.sort(values[rng.integers(0, len(values), (iterations, len(values)))].mean(axis=1))
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def section_rates(rows):
    """Mean success over seeds for every (track, section): the section is the unit of replication."""
    groups = {}
    for row in rows:
        groups.setdefault((row["track_id"], row["scenario_index"]), []).append(row["success"])
    return [float(np.mean(values)) for values in groups.values()]


def seed_rates(rows, mask, seeds, kind=None):
    rates = []
    for seed in seeds:
        own = [r for r in rows if r["mask"] == mask and r["seed"] == seed and (kind is None or r["kind"] == kind)]
        rates.append(sum(r["success"] for r in own) / len(own))
    return rates


def pct(value):
    return f"{100 * value:.0f}%"


def interval_text(values, seed=17):
    mean, low, high = bootstrap(values, seed)
    return f"{pct(mean)} [{pct(low)}, {pct(high)}]"


def charts(data, output):
    plt.style.use("dark_background")
    valid = set(data["dataGeneration"]["valid_training_tracks"])
    train_rows = [r for r in data["ablations"] if r["split"] == "train" and r["track_id"] in valid]
    sealed_rows = [r for r in data["ablations"] if r["split"] == "pristine_test"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (title, rows, kind) in zip(axes, (("Nominal sections, valid training tracks", train_rows, "nominal"), ("Recovery sections, valid training tracks", train_rows, "recovery"),
                                              ("Sealed circuits: 12 fixed sections", sealed_rows, None))):
        for i, mask in enumerate(MASKS):
            rates = seed_rates(rows, mask, ABLATION_SEEDS, kind)
            mean, low, high = bootstrap(rates)
            ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]], color="white", capsize=6, lw=2, zorder=3)
            ax.scatter(np.full(len(rates), i) + np.linspace(-.18, .18, len(rates)), rates, color=COLORS[mask], s=28, zorder=4)
        ax.set(xticks=range(4), xticklabels=[MASK_NAMES[m] for m in MASKS], ylim=(-.03, 1.05), title=title, ylabel="Safe, centered exits per seed")
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Ten training seeds per input variant: dots = seeds, bar = mean with 95% percentile-bootstrap interval over seeds")
    fig.savefig(output / "visuals/feature_learning.png", dpi=150)
    plt.close(fig)
    seeds = ABLATION_SEEDS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    for seed in seeds:
        history = data["histories"][f"full_{seed}"]
        axes[0].plot([h["update"] for h in history], [h["validation_mse"] for h in history], color="#ffc56b", alpha=.45)
        reference = data["closedLoopReference"][f"full_{seed}"]
        axes[1].plot([h["update"] for h in reference], [h["recovery"]["k"] for h in reference], color="#be9cff", alpha=.45)
    mean_val = np.mean([[h["validation_mse"] for h in data["histories"][f"full_{s}"]] for s in seeds], axis=0)
    mean_rec = np.mean([[h["recovery"]["k"] for h in data["closedLoopReference"][f"full_{s}"]] for s in seeds], axis=0)
    updates = [h["update"] for h in data["histories"][f"full_{seeds[0]}"]]
    axes[0].plot(updates, mean_val, color="white", lw=3)
    axes[1].plot(updates, mean_rec, color="white", lw=3)
    axes[0].set(yscale="log", title="Validation loss (ten seeds): smooth and monotonic", xlabel="Adam update", ylabel="weighted action MSE")
    axes[1].set(title="Recovery sections completed of 8 (same seeds): erratic", xlabel="Adam update", ylabel="sections completed safely", ylim=(-.2, 8.2))
    fig.savefig(output / "visuals/loss_vs_driving.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    conditions = [c["name"] for c in data["conditions"]]
    for offset, kind, color in ((-.2, "nominal", "#76a9ff"), (.2, "recovery", "#be9cff")):
        for i, name in enumerate(conditions):
            rows = [r for r in data["robustness"] if r["condition"] == name and r["split"] == "train" and r["track_id"] in valid and r["kind"] == kind]
            mean, low, high = bootstrap(section_rates(rows))
            ax.bar(i + offset, mean, width=.38, color=color, label=kind if i == 0 else None)
            ax.errorbar(i + offset, mean, yerr=[[mean - low], [high - mean]], color="white", capsize=4)
    ax.set(xticks=range(len(conditions)), xticklabels=["Clean", "Noise", "0.5 s delay", "65% grip", "Combined"], ylim=(0, 1.08), ylabel="Safe, centered exits (section-level)",
           title="Same pretrained driver, ten paired seeds: bars = mean over sections, whiskers = bootstrap interval over sections")
    ax.legend()
    fig.savefig(output / "visuals/robustness.png", dpi=150)
    plt.close(fig)


def animation(data, output, catalog):
    track = track_from_catalog(WALL_CONFIG, catalog[0])
    scenario = corner_scenarios(track)[3]
    steps = [0, 128, data["models"]["full_301"]["updates"]]
    traces, metrics = [], []
    for step in steps:
        model = NeuralDriver.load(output / f"models/full_301_update_{step:03d}.npz")
        metric, trace = wall_simulate(track, WALL_CONFIG, model, scenario)
        traces.append(trace)
        metrics.append(metric)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), layout="constrained")
    lines, cars, footprints, hud = [], [], [], []
    for ax, step, metric in zip(axes, steps, metrics):
        ax.plot(track.left[:, 0], track.left[:, 1], "#ff5b65", lw=1)
        ax.plot(track.right[:, 0], track.right[:, 1], "#ff5b65", lw=1)
        ax.plot(track.xy[:, 0], track.xy[:, 1], "--", color="#78899e", lw=.7)
        line, = ax.plot([], [], "#4de2c2", lw=1.5)
        lines.append(line)
        car = Polygon(np.zeros((4, 2)), facecolor="#ffc56b", edgecolor="#ecf2fa", lw=.7)
        ax.add_patch(car)
        cars.append(car)
        boundary = Polygon(np.zeros((4, 2)), fill=False, edgecolor="#4de2c2", lw=1)
        ax.add_patch(boundary)
        footprints.append(boundary)
        hud.append(ax.text(.02, .02, "", transform=ax.transAxes, fontsize=9))
        ax.set_aspect("equal")
        ax.set_title(f"Actual saved update {step}\n{metric['end_reason'].replace('_', ' ')}", fontsize=11)
        ax.tick_params(labelsize=7)
    fig.suptitle("Actual training checkpoints (seed 301) / equal simulated time / failures freeze, no scripted improvements", fontsize=12)
    duration = max(trace[-1, 0] for trace in traces)
    frames = np.arange(0, duration + .12, .12)

    def update(frame):
        for ax, trace, line, car, boundary, text in zip(axes, traces, lines, cars, footprints, hud):
            index = min(len(trace) - 1, int(np.searchsorted(trace[:, 0], frames[frame], side="right") - 1))
            state = trace[index]
            pose = type("Pose", (), {"x": state[1], "y": state[2], "heading": state[3]})()
            boundary.set_xy(footprint(pose))
            c, s = np.cos(state[3]), np.sin(state[3])
            local = np.array([[2.2, 0], [.7, .5], [-1.8, .4], [-1.8, -.4], [.7, -.5]])
            car.set_xy(local @ np.array([[c, s], [-s, c]]) + state[1:3])
            line.set_data(trace[:index + 1, 1], trace[:index + 1, 2])
            ax.set_xlim(state[1] - 28, state[1] + 28)
            ax.set_ylim(state[2] - 28, state[2] + 28)
            text.set_text(f"{state[0]:.1f}s | {state[4]*3.6:.0f} km/h | G{int(state[10])}\nbrake {state[9]*100:.0f}% | margin {state[17]:.2f}m")
        return lines + cars + footprints + hud

    update(min(len(frames) - 1, 20))
    fig.savefig(output / "visuals/checkpoint_comparison.png", dpi=150)
    movie = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    movie.save(output / "visuals/training_evolution.gif", writer=PillowWriter(fps=16))
    plt.close(fig)


def document(title, body):
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:17px system-ui;max-width:1080px;margin:40px auto;padding:20px;background:#101722;color:#ecf2fa}}
h1,h2,h3{{color:#4de2c2}}p,li{{line-height:1.6}}a{{color:#ffc56b}}section{{padding:24px;margin:20px 0;border:1px solid #536780;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #536780;padding:8px;text-align:left;vertical-align:top}}
img{{max-width:100%}}.response{{height:110px;border-bottom:1px solid #536780}}.small{{font-size:14px;color:#acbbce}}
@media print{{body{{background:white;color:black}}section{{break-inside:avoid}}h1,h2,h3{{color:black}}.small{{color:black}}}}</style>
<h1>{html.escape(title)}</h1><p><a href="index.html">Open the interactive laboratory</a> · <a href="class_dashboard.html">Class dashboard</a></p>{body}</html>"""


def esc(text):
    return html.escape(str(text))


def teacher_guide(data, kit, manifest):
    objectives = {o["id"]: o["text"] for o in kit["objectives"]}
    misconceptions = {m["id"]: m for m in kit["misconceptions"]}
    worksheets = {w["module"]: [] for w in kit["worksheets"]}
    for w in kit["worksheets"]:
        worksheets[w["module"]].append(w)
    generation = data["dataGeneration"]
    parts = [f"""<section><h2>Setup and protocol</h2><ul>{''.join(f'<li>{esc(line)}</li>' for line in kit['teacher']['setup'])}</ul>
<p>The lab trains {manifest['ablation_models']} compact 18→24→16→2 networks (four input variants × {len(manifest['ablation_seeds'])} seeds) for {manifest['updates_per_model']} genuine Adam updates each.
Training uses {manifest['training_samples']:,} expert-labelled states from {manifest['training_episodes']} episodes; validation uses {manifest['validation_samples']:,} states from {manifest['validation_episodes']} different episodes drawn by the same sampler;
the recovery-stress set has {manifest['stress_samples']:,} states from {manifest['stress_episodes']} deep-recovery drives and is evaluation only.</p></section>"""]
    parts.append("<section><h2>Learning objectives</h2><table><tr><th>ID</th><th>By the end, students can…</th><th>Modules</th><th>Quiz items</th><th>Worksheets</th></tr>" + "".join(
        f"<tr><td>{o['id']}</td><td>{esc(o['text'])}</td><td>{esc(', '.join(m['title'] for m in kit['modules'] if o['id'] in m['objectives']))}</td>"
        f"<td>{esc(', '.join(q['id'] for q in kit['quiz'] if q['objective'] == o['id']))}</td><td>{esc(', '.join(w['title'] for w in kit['worksheets'] if o['id'] in w['objectives']))}</td></tr>"
        for o in kit["objectives"]) + "</table></section>")
    total = sum(m["minutes"] for m in kit["modules"])
    for index, module in enumerate(kit["modules"], 1):
        parts.append(f"<section><h2>{index}. {esc(module['title'])} <span class='small'>({module['minutes']} min; prerequisites: {esc(module['prerequisites'])})</span></h2>"
                     f"<p><strong>Objectives:</strong> {esc('; '.join(f'{i}: {objectives[i]}' for i in module['objectives']))}</p>"
                     f"<p><strong>Discuss:</strong></p><ul>{''.join(f'<li>{esc(q)}</li>' for q in module['discussion'])}</ul>"
                     + ("<p><strong>Misconceptions to watch for:</strong></p><ul>" + "".join(
                         f"<li><em>{m}</em>: {esc(misconceptions[m]['text'])} <span class='small'>Remedy: {esc(misconceptions[m]['remedy'])}</span></li>" for m in module["misconceptions"]) + "</ul>" if module["misconceptions"] else "")
                     + "".join(f"<p><strong>Worksheet — {esc(w['title'])}:</strong> {esc(w['prompt'])}</p><p>Hypothesis / settings / evidence / conclusion:</p><div class='response'></div>" for w in worksheets.get(module["id"], []))
                     + "</section>")
    parts.append(f"<section><h2>Suggested sessions</h2><p>The modules total about {total} minutes. Two 90-minute sessions work well: (1) modules 1-4 and 6; "
                 "(2) modules 5, 7, 8, 9 and 10, finishing with module 11 and the post-lesson concept check. Module 12 (about 35 minutes; it needs modules 2 and 7) makes a good third session "
                 "or a closing demonstration: training both learners takes about a minute, so start the reinforcement run, discuss the filmstrip while it trains, then play the learning movie. "
                 "Use the pre/post quiz as a conversation starter; the class dashboard aggregates the cards students download.</p></section>")
    parts.append("<section><h2>Rubric for guided experiments</h2><table><tr><th>Criterion</th>" + "".join(f"<th>{esc(l)}</th>" for l in kit["rubric"]["levels"]) + "</tr>" + "".join(
        f"<tr><td><strong>{esc(c['name'])}</strong></td>" + "".join(f"<td>{esc(d)}</td>" for d in c["descriptors"]) + "</tr>" for c in kit["rubric"]["criteria"]) + "</table></section>")
    quiz = []
    for number, q in enumerate(kit["quiz"], 1):
        quiz.append(f"<h3>{number}. {esc(q['prompt'])} <span class='small'>({q['objective']})</span></h3><ul>" + "".join(
            f"<li>{'<strong>✔ ' if c.get('correct') else ''}{esc(c['text'])}{'</strong>' if c.get('correct') else ''} — <span class='small'>{esc(c['why'])}"
            f"{' Misconception ' + c['misconception'] if c.get('misconception') else ''}</span></li>" for c in q["choices"]) + "</ul>")
    parts.append("<section><h2>Concept check: answer key with explanations</h2><p class='small'>The lab shuffles the choices for every attempt, so the key lists the correct choice, not its position.</p>" + "".join(quiz) + "</section>")
    parts.append("<section><h2>Statistics policy used throughout the lab</h2><ul>"
                 "<li><strong>Training seed</strong> is the replication unit when comparing recipes: each dot is one independently trained network; bars are means with percentile-bootstrap intervals over seeds, and fewer than ten seeds is labelled exploratory.</li>"
                 "<li><strong>Fixed sections</strong> give exact counts. Trials inside one section share a track and a start family, so the <strong>section</strong> is the unit: section-level success rates, cluster bootstrap over sections, comparisons paired by section with an exact sign test.</li>"
                 "<li><strong>Wilson intervals</strong> appear only for independent random starts with one trial each.</li>"
                 "<li>The development pools (monitor sections, the 36 fixed sections) are exploratory: repeated looking while tuning makes later numbers optimistic. The sealed circuits are a classroom protocol, not secrecy; blind claims need a teacher-controlled fresh pack "
                 "(<code>tools/make_fresh_pack.py</code>) revealed after students freeze their recipes.</li></ul></section>")
    expert = generation["expert_reliability"]
    parts.append("<section><h2>Known limitations</h2><ul>" + "".join(f"<li>{esc(line)}</li>" for line in kit["teacher"]["limits"]) + f"<li>Measured expert reliability: {expert['stress_completed_150_steps']} of {expert['stress_episodes']} deep-recovery drives and "
                 f"{expert['typical_episodes_ended_early']} typical episodes ended early because the expert itself tripped a limit. A learner cannot exceed its labels.</li>"
                 f"<li>Legacy-geometry circuits {', '.join(str(i) for i in generation['legacy_geometry_tracks'])} are excluded from the monitor and headline pools.</li></ul></section>")
    return document("Teacher Guide: objectives, rubric, answer key and printable worksheets", "".join(parts))


def coverage_reading(summary):
    """Plain-language reading of the saved coverage run, generated from its numbers so the text cannot contradict them."""
    arms, seeds = {a["arm"]: a for a in summary["arms"]}, summary["settings"]["seeds"]
    percent = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    points = lambda v: f"{100 * v:+.0f} pts".replace("-", "−")  # noqa: E731
    lines = []
    pair = next((c for c in summary["comparisons"] if (c["a"], c["b"]) == ("dagger", "bc_staged")), None)
    if pair:
        overall, recovery = pair["overall"], pair["recovery"]
        clear = overall["lo"] > 0 or overall["hi"] < 0
        lines.append(f"- DAgger against the staged expert-only control (the one comparison that isolates *who chooses the visited states*): overall {points(overall['mean'])} "
                     f"[{points(overall['lo'])}, {points(overall['hi'])}], recovery-only {points(recovery['mean'])} [{points(recovery['lo'])}, {points(recovery['hi'])}]; "
                     f"seeds better/worse/equal {overall['wins']}/{overall['losses']}/{overall['ties']}, sign-test p = {overall['p']:.2f}. "
                     + ("The interval excludes zero." if clear else f"The interval includes zero, so with {seeds} seeds a closed-loop benefit is **not established**."))
    if {"dagger", "bc_narrow"} <= set(arms):
        dagger, narrow = arms["dagger"], arms["bc_narrow"]
        lines.append(f"- Mechanism, measured separately from closed-loop success: DAgger's data hold {percent(dagger['supportVisited'])} of the projected cells its own learners visit, "
                     f"against {percent(narrow['supportVisited'])} for narrow demonstrations. Occupancy is a diagnostic, not a guarantee.")
        lines.append(f"- DAgger's learners crashed during {dagger['failed']} collection episodes (narrow demonstrations: {narrow['failed']}); "
                     "those are exactly the states where an expert's correction is needed, which is why failed collections are reported rather than hidden.")
    lines.append("- Comparisons with narrow or broad demonstrations change several things at once (schedule, start distribution and who drives); only the comparison with the staged control changes one. "
                 "Fewer than ten seeds is exploratory, and nominal and recovery sections are never pooled.")
    return lines


def results_markdown(data, manifest, coverage, journey, output):
    valid = set(manifest["valid_training_tracks"])
    train_rows = [r for r in data["ablations"] if r["split"] == "train" and r["track_id"] in valid]
    sealed = [r for r in data["ablations"] if r["split"] == "pristine_test"]
    seeds = manifest["ablation_seeds"]
    lines = [f"""# Student ML Driving Lab (version 2)

Open `student_v2_artifacts/index.html` for the twelve interactive modules (module 12 takes an imitation learner and a reinforcement learner from an untrained car to a whole lap), `teacher_guide.html` for objectives, rubric, answer key and printable worksheets, and
`class_dashboard.html` to aggregate students' results cards. The first eight-module lab is preserved unchanged in `student_artifacts/`.

## What changed and why

The first lab's validation set held only deep wall-recovery states (median offset 5.08 m) while its training set was 89.5% near-centreline, so a 7.6x "generalization gap" was distribution shift, not overfitting.
Version 2 draws training and validation as **different expert episodes from the same sampler** and adds a separately named **recovery-stress set** (evaluation only).

| Set | Examples | Episodes | Within 1 m of the centreline | Median offset |
|---|---:|---:|---:|---:|"""]
    for name, key, episodes in (("Training", "training", manifest["training_episodes"]), ("Validation", "validation", manifest["validation_episodes"]), ("Stress (evaluation only)", "stress", manifest["stress_episodes"])):
        with np.load(Path(output) / "data" / f"{key}.npz") as arrays:
            offset = np.abs(arrays["features"][:, 0]) * data["config"]["track_width"] / 2
        lines.append(f"| {name} | {len(offset):,} | {episodes} | {100 * np.mean(offset < 1):.1f}% | {np.median(offset):.2f} m |")
    final = {mask: [data["histories"][f"{mask}_{s}"][-1] for s in seeds] for mask in MASKS}
    lines += ["", "## What was executed", "",
              f"- **{manifest['ablation_models']} retrained models** ({len(MASKS)} input variants × {len(seeds)} seeds × {manifest['updates_per_model']} Adam updates), {sum(len(v) for v in data['histories'].values())} saved checkpoints; diverged seeds: {len(manifest['diverged_seeds'])}.",
              f"- **{manifest['ablation_rollouts']:,} ablation, {manifest['controller_rollouts']:,} controller, {manifest['robustness_rollouts']:,} robustness and {manifest['monitor_rollouts']:,} monitor rollouts** (NumPy simulator, {manifest['environment']['workers']} workers).",
              f"- Final mean losses (full inputs, ten seeds): training {np.mean([r['train_mse'] for r in final['full']]):.4f}, validation {np.mean([r['validation_mse'] for r in final['full']]):.4f}, stress {np.mean([r['stress_mse'] for r in final['full']]):.4f}.",
              f"- Legacy-geometry circuits {manifest['legacy_geometry_tracks']} (Apex Park, Square Sprint: inner-wall loops at bends tighter than half the road width) are excluded from the monitor and headline pools; they stay in the training data for reproducibility.", "",
              "## Input-removal retraining (ten seeds, valid training tracks)", "",
              "Mean over seeds with a 95% percentile-bootstrap interval over seeds; per-seed range in the last column. Nominal and recovery sections are never pooled.", "",
              "| Variant | Nominal sections | Recovery sections | Sealed circuits (12 fixed sections) | Recovery per-seed range |", "|---|---|---|---|---|"]
    for mask in MASKS:
        nominal, recovery = seed_rates(train_rows, mask, seeds, "nominal"), seed_rates(train_rows, mask, seeds, "recovery")
        sealed_counts = [sum(r["success"] for r in sealed if r["mask"] == mask and r["seed"] == s) for s in seeds]
        lines.append(f"| {MASK_NAMES[mask]} | {interval_text(nominal)} | {interval_text(recovery)} | {np.mean(sealed_counts):.1f}/12 ({min(sealed_counts)}-{max(sealed_counts)}) | {pct(min(recovery))} to {pct(max(recovery))} |")
    lines += ["", "Only removing the road previews and curvature is clearly distinguishable from the full model; heading and margin removal sit inside the seed-to-seed spread. On the three sealed circuits almost every network completes the nominal sections and fails the recovery sections, a coverage problem addressed in module 7.", "",
              "## Loss versus driving", "",
              "| Update | Mean validation loss | Mean nominal sections (of 8) | Mean recovery sections (of 8) | Recovery per-seed values |", "|---:|---:|---:|---:|---|"]
    for index in range(len(data["histories"][f"full_{seeds[0]}"])):
        update = data["histories"][f"full_{seeds[0]}"][index]["update"]
        val = np.mean([data["histories"][f"full_{s}"][index]["validation_mse"] for s in seeds])
        nominal = [data["closedLoopReference"][f"full_{s}"][index]["nominal"]["k"] for s in seeds]
        recovery = [data["closedLoopReference"][f"full_{s}"][index]["recovery"]["k"] for s in seeds]
        lines.append(f"| {update} | {val:.4f} | {np.mean(nominal):.1f} | {np.mean(recovery):.1f} | {' '.join(str(v) for v in recovery)} |")
    lines += ["", "## Controllers and distribution shift (section-level, ten paired seeds)", "", "| Condition | Nominal sections | Recovery sections | Handling-limit failures |", "|---|---|---|---:|"]
    for label in ("rules", "expert", "pretrained"):
        rows = [r for r in data["comparisons"] if r["controller"] == label and r["split"] == "train" and r["track_id"] in valid]
        lines.append(f"| {label} | {interval_text(section_rates([r for r in rows if r['kind'] == 'nominal']))} | {interval_text(section_rates([r for r in rows if r['kind'] == 'recovery']))} | n/a |")
    for condition in data["conditions"]:
        rows = [r for r in data["robustness"] if r["condition"] == condition["name"] and r["split"] == "train" and r["track_id"] in valid]
        lines.append(f"| pretrained, {condition['name']} | {interval_text(section_rates([r for r in rows if r['kind'] == 'nominal']))} | {interval_text(section_rates([r for r in rows if r['kind'] == 'recovery']))} | {sum(r['end_reason'] == 'lateral_acceleration_limit' for r in rows)} |")
    if coverage:
        lines += ["", "## Data coverage and DAgger (headless run of the browser engine)", "", coverage["text"], "", "Reading of this run (generated from the numbers above):", "",
                  *coverage_reading(coverage["data"]["summary"])]
    if journey:
        lines += ["", "## From zero to a lap: imitation versus reinforcement learning (headless run of the browser engine)", "",
                  "Module 12 trains both learners on one circuit and, after every stage of training, drives one whole lap from the start line with no help. "
                  "Reinforcement learning is a PPO actor-critic with the same 18-24-16-2 policy network, trained from a reward only (no expert labels).", "", journey["text"], "",
                  "Reading of this run (generated from the numbers above):", "", *journey_reading(journey["data"]), "",
                  "![A lap attempt at each stage of training](student_v2_artifacts/visuals/journey_filmstrip.png)", "",
                  "![The learning movie: each stage drives its whole lap, imitation left and reinforcement right](student_v2_artifacts/visuals/journey_learning.gif)", "",
                  "![Learning curves, steering smoothness and driving needed on every circuit](student_v2_artifacts/visuals/journey_curves.png)"]
    expert = manifest["data_generation"]["expert_reliability"]
    lines += ["", "## Honest boundaries", "",
              f"- The expert is a heuristic: {expert['typical_episodes_ended_early']} of {manifest['training_episodes'] + manifest['validation_episodes']} typical episodes and {expert['stress_episodes'] - expert['stress_completed_150_steps']} of {expert['stress_episodes']} stress drives ended early because the expert itself tripped a limit. A learner cannot exceed its labels.",
              "- The sealed circuits are a classroom protocol, not secrecy: the page embeds them. Blind claims need a teacher-controlled fresh pack (`tools/make_fresh_pack.py`).",
              "- Fixed sections give exact counts; trials inside a section are clustered, so intervals use sections (or seeds) as the unit of replication. Development pools (monitor and the 36 fixed sections) are exploratory.",
              "- Grip is a hard threshold on lateral acceleration and braking, not a tyre-slip model; noise perturbs lane, heading and speed but not the ideal road previews; one simulator, one expert, original circuits, no real-car claim.",
              "- Browser and NumPy use different random streams (so noisy runs differ), but every deterministic quantity agrees: the browser engine replays all 2,560 saved NumPy rollouts of the retrained networks with identical outcomes and metrics within 1e-9 (`tests/student-engine.cjs`), and 24 fixture trajectories match to 1e-6 m (`tests/engine-fixtures.cjs`).",
              "- Curvature inputs are read from the centreline sample that *starts* at a boundary (`driving.SAMPLE_SNAP`), so one ulp of rounding noise cannot flip them between engines or platforms. Against the preserved first lab this changes a continuous metric in 69 of its 2,304 saved rollouts (never an outcome); `git tag lab-v1` marks the code that produced those files exactly.",
              "- Module 12's reinforcement learner is a small PPO actor-critic written for this lab (`student_rl.js`), not a library. Its learning rate (3e-4) and steering-smoothness weight (0.3) were chosen on a handful of circuits and seeds that the reference run also contains, "
              "so the reference describes this setup and is not a blind test of reinforcement learning. Both learners are tested on the circuit they trained on; module 9 is where transfer to unseen circuits is measured.",
              "- Module 12's saved example is replayed stage by stage in the NumPy simulator by `student_verify.py`. A car that steers smoothly reproduces to the 0.1 s; a car that flicks its steering is chaotic (a 1e-14 difference between two correct simulators grows to metres within 15 s), "
              "so for those only the outcome is expected to agree. That is one more reason the smoothness term is in the default reward.", "",
              "## Reproduce", "", "```sh",
              ".venv/bin/python student_experiments.py --output student_v2_artifacts --base wall_artifacts",
              "node tools/coverage_reference.cjs --artifact student_v2_artifacts",
              "node tools/journey_reference.cjs --artifact student_v2_artifacts    # about 8 minutes on 12 cores",
              ".venv/bin/python student_render.py --output student_v2_artifacts",
              "npm run test:student",
              "npm run test:engine && .venv/bin/python -m unittest discover -s tests",
              "PLAYWRIGHT_BROWSERS_PATH=\"$PWD/.browsers\" npm run test:browser:student",
              ".venv/bin/python student_verify.py --output student_v2_artifacts", "```", ""]
    return "\n".join(lines)


def deck(data, kit, output, manifest, journey=None):
    slides = [("From watching a driver to testing an ML idea", ["Twelve hands-on lessons on top of the preserved driving simulator", "Real supervised updates and simulated motion, not animation-only improvement",
                                                                 "The car remains fictional; no real-driving safety claims"], None)]
    for index, module in enumerate(kit["modules"], 1):
        slides.append((f"{index}. {module['title']}", [kit["objectives"][int(o[1:]) - 1]["text"] for o in module["objectives"][:2]] + [f"Discuss: {module['discussion'][0]}"], None))
    slides += [("Measured: input removal over ten seeds", ["Dots are seeds; bars are means with bootstrap intervals over seeds", "Only removing previews is clearly different; heading and margin sit inside the seed spread"], "visuals/feature_learning.png"),
               ("Measured: loss versus driving", ["Validation loss is smooth and monotonic", "Recovery success is erratic across seeds: low loss is not driving success"], "visuals/loss_vs_driving.png"),
               ("Measured: distribution shift", ["Section-level success with bootstrap intervals over sections", "Grip is a hard threshold in this toy model"], "visuals/robustness.png")]
    if journey:
        overall, ablation = journey["data"]["overall"], journey["data"]["ablation"]
        slides.append(("Measured: from an untrained car to a lap", [
            f"Each picture is a whole lap by the car as it stood at that stage; imitation stays on the lap from a median {overall['imitation']['stableLapSteps']['median']:,.0f} labelled steps, "
            f"reinforcement from {overall['reinforcement']['stableLapSteps']['median']:,.0f} simulated steps with no labels",
            f"Without a steering-smoothness reward, {ablation['without']['flickering']} of {ablation['without']['seeds']} reinforcement cars flicked the steering every step; with it, {ablation['with']['flickering']}"], "visuals/journey_filmstrip.png"))
    slides.append(("What to take away", ["Name the unit of replication before choosing an interval", "Cover the states you will actually visit; check with a matched control", "Freeze before you test; say what an interval does not cover"], None))
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    sections = []
    for index, (title, bullets, image) in enumerate(slides, 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(.6), Inches(.35), Inches(12), Inches(.9))
        box.text_frame.text = title
        box.text_frame.paragraphs[0].font.size = Pt(30)
        if image:
            slide.shapes.add_picture(str(output / image), Inches(.6), Inches(2.5), width=Inches(12), height=Inches(4.5))
        text = slide.shapes.add_textbox(Inches(.65), Inches(1.35), Inches(12), Inches(1.4 if image else 4.5))
        for row, bullet in enumerate(bullets):
            paragraph = text.text_frame.paragraphs[0] if row == 0 else text.text_frame.add_paragraph()
            paragraph.text = bullet
            paragraph.font.size = Pt(16 if image else 24)
            paragraph.space_after = Pt(12)
        slide.notes_slide.notes_text_frame.text = "Use the corresponding interactive module. " + " ".join(bullets) + " Distinguish learned steering/target speed from automatic pedals/gears; do not claim real-car safety."
        sections.append(f"<section><h2>{index}. {esc(title)}</h2><ul>" + "".join(f"<li>{esc(b)}</li>" for b in bullets) + "</ul>" + (f'<img src="{image}">' if image else "") + "</section>")
    presentation.save(output / "training_deck.pptx")
    (output / "training_deck.html").write_text(document("Classroom Deck: Twelve Ways to Understand ML Driving", "".join(sections)))
    write_json(output / "deck_manifest.json", {"slides": len(slides), "worksheets": len(kit["worksheets"]), "speaker_notes": True})


def coverage_summary(output):
    path = output / "data/coverage_reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    return {"data": reference, "text": reference["markdown"]}


def journey_summary(output):
    path = output / "data/journey_reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    return {"data": reference, "page": {key: value for key, value in reference.items() if key not in JOURNEY_PRIVATE}, "text": reference["markdown"]}


def key_stages(stages, count=10):
    """The pictures of the page's filmstrip: spaced more closely early on, always keeping the first lap and the last stage (mirrors keyStages in student_journey_ui.js)."""
    last = len(stages) - 1
    if last < count:
        return list(range(len(stages)))
    picks = {int(np.floor(last * (i / (count - 1)) ** 1.6 + .5)) for i in range(count)}
    first = next((i for i, stage in enumerate(stages) if stage["row"][2]), -1)
    if first > 0 and first not in picks:
        picks.discard(min(sorted(i for i in picks if 0 < i < last), key=lambda i: abs(i - first)))
        picks.add(first)
    return sorted(picks)


def stage_label(method, stage):
    if stage["index"] == 0:
        return "Untrained"
    if method == "imitation":
        return "Expert demos" if stage["round"] == 0 else f"DAgger round {stage['round']}"
    return f"Iteration {stage['iteration']}"


def journey_reading(reference):
    """Plain-language reading of the saved journey run, generated from its numbers so the text cannot contradict them."""
    overall, ablation, dt = reference["overall"], reference["ablation"], .1
    imitation, reinforcement = overall["imitation"], overall["reinforcement"]
    minutes = lambda steps: steps * dt / 60  # noqa: E731
    lines = [f"- The first lap needs far less experience with an expert's labels than with a reward alone: imitation's median first lap came after {imitation['firstLapSteps']['median']:,.0f} labelled steps "
             f"({minutes(imitation['firstLapSteps']['median']):.1f} min of driving), reinforcement's after {reinforcement['firstLapSteps']['median']:,.0f} simulated steps ({minutes(reinforcement['firstLapSteps']['median']):.0f} min).",
             f"- A lap that *stays* a lap is a stricter test (the first stage from which every later stage also completes it): imitation {imitation['stableLapSteps']['median']:,.0f} steps, reinforcement "
             f"{reinforcement['stableLapSteps']['median']:,.0f} steps, {reinforcement['stableLapSteps']['median'] / imitation['stableLapSteps']['median']:.0f}× as many.",
             f"- The final cars: imitation completes the lap in {imitation['finalLaps']} of {imitation['runs']} runs at {imitation['ratio']:.2f}× the expert's lap time on average, reinforcement in {reinforcement['finalLaps']} of "
             f"{reinforcement['runs']} runs at {reinforcement['ratio']:.2f}×. Neither learner improves smoothly: after first completing the lap, imitation failed it again at some later stage in "
             f"{imitation['lostLap']} of {imitation['runs']} runs and reinforcement in {reinforcement['lostLap']} of {reinforcement['runs']} (in the final car: {imitation['regressed']} and {reinforcement['regressed']}).",
             f"- Reward design matters: on circuit {ablation['trackId']} the final cars trained with the steering-smoothness term flick the steering in {ablation['with']['flickering']} of {ablation['with']['seeds']} runs "
             f"({ablation['with']['flicker']:.2f}° per step on average); trained without it, {ablation['without']['flickering']} of {ablation['without']['seeds']} do ({ablation['without']['flicker']:.2f}° per step). "
             f"Lap times were {ablation['with']['ratio']['mean']:.2f}× and {ablation['without']['ratio']['mean']:.2f}× the expert's: the penalty costs a little speed.",
             "- Imitation also needed an expert that can label any state; reinforcement needed only a score and a simulator it can crash a hundred thousand times. "
             "Both learn one circuit and are tested on that circuit: this compares what each costs, not which method is better in general."]
    return lines


def journey_replays(reference, track):
    """The key stages of the saved example and the whole lap each one drives, replayed in the NumPy simulator."""
    featured = reference["featured"]
    picks = {method: key_stages(featured[method]["stages"]) for method in ("imitation", "reinforcement")}
    return picks, {(method, index): lap(track, featured[method]["stages"][index]["model"]) for method in picks for index in picks[method]}


def journey_animation(reference, output, catalog, track, picks, replays):
    """The learning movie as a GIF: the car as it stood at each key stage drives its whole lap, imitation on the left and reinforcement on the right."""
    featured, methods = reference["featured"], ("imitation", "reinforcement")
    colors = {"imitation": "#ffc56b", "reinforcement": "#76a9ff"}
    short = {"wall_contact": "hit the wall", "off_track": "left the road", "lateral_acceleration_limit": "exceeded the grip limit"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), layout="constrained")
    outline = np.vstack([track.left, track.right[::-1]])
    lines, dots, notes = {}, {}, {}
    for ax, method in zip(axes, methods):
        ax.fill(outline[:, 0], outline[:, 1], color="#405064")
        ax.set_aspect("equal")
        ax.axis("off")
        lines[method], = ax.plot([], [], color=colors[method], lw=2)
        dots[method], = ax.plot([], [], "o", color=colors[method], mec="white", ms=8)
        notes[method] = ax.text(.5, .5, "", transform=ax.transAxes, ha="center", va="center", fontsize=13, weight="bold")
    title = fig.suptitle("", fontsize=13)
    pictures, seconds_per_frame = max(len(p) for p in picks.values()), 3.2  # 32x at ten frames a second
    chosen = lambda method, picture: picks[method][min(picture, len(picks[method]) - 1)]  # noqa: E731
    frames = []
    for picture in range(pictures):
        duration = max(replays[method, chosen(method, picture)][1][-1, 0] for method in methods)
        frames += [(picture, i * seconds_per_frame) for i in range(max(10, int(np.ceil(duration / seconds_per_frame))))] + [(picture, duration + 1)] * 6

    def draw(frame):
        picture, now = frame
        for ax, method in zip(axes, methods):
            index = chosen(method, picture)
            result, trace = replays[method, index]
            at = int(np.clip(np.searchsorted(trace[:, 0], now, side="right") - 1, 0, len(trace) - 1))
            lines[method].set_data(trace[:at + 1, 1], trace[:at + 1, 2])
            dots[method].set_data([trace[at, 1]], [trace[at, 2]])
            done = result["end_reason"] == "section_complete"
            if now < trace[-1, 0]:
                notes[method].set_text("")
            else:
                notes[method].set_text(f"LAP {result['duration_s']:.0f} s" if done else f"{min(99, int(np.floor(result['progress'] * 100)))}% of the lap\n{short.get(result['end_reason'], result['end_reason'])}")
            notes[method].set_color("#4de2c2" if done else "#ff8990")
            ax.set_title(f"{'Imitation (DAgger)' if method == 'imitation' else 'Reinforcement (PPO)'}\n{stage_label(method, featured[method]['stages'][index])} · {featured[method]['stages'][index]['steps']:,} steps",
                         fontsize=11, color=colors[method])
        title.set_text(f"{track_name(catalog, featured['trackId'])}: the car as it stood at stage {picture + 1} of {pictures}, driving a whole lap from the start line")
        return []

    draw(frames[0])
    FuncAnimation(fig, draw, frames=frames, interval=100, blit=False).save(output / "visuals/journey_learning.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


def journey_charts(reference, output, catalog):
    plt.style.use("dark_background")
    featured, tracks = reference["featured"], tracks_from_catalog(WALL_CONFIG, catalog)
    track, circuit = tracks[featured["trackId"]], str(featured["trackId"])
    colors, outline = {"imitation": "#ffc56b", "reinforcement": "#76a9ff"}, np.vstack([track.left, track.right[::-1]])
    short = {"wall_contact": "wall", "off_track": "off road", "lateral_acceleration_limit": "grip limit"}
    columns = 10
    picks, replays = journey_replays(reference, track)
    fig, axes = plt.subplots(2, columns, figsize=(20, 6.4), layout="constrained")
    for row, method in enumerate(("imitation", "reinforcement")):
        stages = featured[method]["stages"]
        for column in range(columns):
            ax = axes[row, column]
            ax.set_aspect("equal")
            ax.axis("off")
            if column >= len(picks[method]):
                continue
            stage = stages[picks[method][column]]
            result, trace = replays[method, picks[method][column]]
            done = result["end_reason"] == "section_complete"
            ax.fill(outline[:, 0], outline[:, 1], color="#405064")
            ax.plot(trace[:, 1], trace[:, 2], color=colors[method], lw=1.6)
            ax.plot(*trace[0, 1:3], "o", color="white", ms=3)
            if done:
                ax.plot(*trace[0, 1:3], "o", mfc="none", mec="#4de2c2", ms=11, mew=2)
            else:
                ax.plot(*trace[-1, 1:3], "x", color="#ff5b65", ms=9, mew=2.5)
            ax.set_title(stage_label(method, stage), fontsize=10)
            outcome = f"LAP {result['duration_s']:.0f} s" if done else f"{min(99, int(np.floor(result['progress'] * 100)))}% · {short.get(result['end_reason'], result['end_reason'])}"
            ax.text(.5, -.02, f"{outcome}\n{stage['steps']:,} steps", transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#4de2c2" if done else "#ff8990")
        axes[row, 0].text(-.05, .5, {"imitation": "Imitation\n(DAgger)", "reinforcement": "Reinforcement\n(PPO)"}[method], transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=12, color=colors[method])
    fig.suptitle(f"{track_name(catalog, featured['trackId'])}: one lap attempt by the car as it stood at each stage of training (NumPy replay of the saved example, seed {featured['seed']})")
    fig.savefig(output / "visuals/journey_filmstrip.png", dpi=140)
    plt.close(fig)
    journey_animation(reference, output, catalog, track, picks, replays)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), layout="constrained")
    data = reference["circuits"][circuit]
    for method in ("imitation", "reinforcement"):
        curve = data[method]["curve"]
        x = np.maximum([p["steps"] for p in curve], 30)
        axes[0, 0].fill_between(x, [p["lo"] for p in curve], [p["hi"] for p in curve], color=colors[method], alpha=.2)
        axes[0, 0].plot(x, [p["median"] for p in curve], color=colors[method], lw=2, label=f"{method}: median and range of {data[method]['seeds']} seeds")
        stages = featured[method]["stages"]
        axes[0, 0].plot(np.maximum([s["row"][0] for s in stages], 30), [s["row"][1] for s in stages], "o", color=colors[method], ms=4)
        seconds = [(max(p["steps"], 30), p["seconds"]) for p in curve if p["seconds"] is not None and p["laps"] * 2 >= data[method]["seeds"]]
        axes[0, 1].plot(*zip(*seconds), "o-", color=colors[method], lw=2, ms=4, label=method)
    axes[0, 0].set(xscale="log", ylim=(-2, 102), xlabel="simulated driving steps so far (log scale, 10 steps = 1 s)", ylabel="% of the lap completed", title="Share of the lap completed before the car failed (dots: seed 1)")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    axes[0, 1].axhline(data["expertSeconds"], ls="--", color="#4de2c2", label="expert")
    axes[0, 1].set(xscale="log", xlabel="simulated driving steps so far (log scale)", ylabel="lap time (s): median of the seeds that complete it", title="Lap time, at stages where at least half the seeds complete the lap")
    axes[0, 1].legend()
    limit = reference["settings"]["lap"]["flickerLimit"]
    groups = [("Imitation", [r["stages"][-1][4] for r in reference["runs"]["imitation"][circuit]], colors["imitation"]),
              ("RL with smoothness term", [r["stages"][-1][4] for r in reference["runs"]["reinforcement"][circuit]], colors["reinforcement"]),
              ("RL without it", [r["stages"][-1][4] for r in reference["runs"]["reinforcement_no_smoothness"][circuit]], "#ff8990")]
    for i, (name, values, color) in enumerate(groups):
        axes[1, 0].scatter(np.full(len(values), i) + np.linspace(-.15, .15, len(values)), values, color=color, s=40)
    axes[1, 0].axhline(limit, ls="--", color="#acbbce")
    axes[1, 0].set(xticks=range(3), xticklabels=[g[0] for g in groups], ylabel="degrees the steering moves per 0.1 s step", title=f"Final car's steering, one dot per seed (dashed: {limit}° = flicking)")
    ids = sorted(reference["circuits"], key=int)
    stable_steps = {method: [reference["circuits"][i][method]["stableLapSteps"] for i in ids] for method in colors}
    for method, shift in (("imitation", -.18), ("reinforcement", .18)):
        stable = stable_steps[method]
        position = np.arange(len(ids)) + shift
        axes[1, 1].errorbar([s["median"] for s in stable], position, xerr=[[s["median"] - s["q1"] for s in stable], [s["q3"] - s["median"] for s in stable]], fmt="o", color=colors[method], capsize=3, label=method)
    typical = np.median([r["median"] / i["median"] for r, i in zip(stable_steps["reinforcement"], stable_steps["imitation"])])
    axes[1, 1].set(xscale="log", yticks=range(len(ids)), yticklabels=[f"{i} {reference['circuits'][i]['name'].replace(' GP-X', '')}" for i in ids], xlabel="driving steps until the car stays lapping (median and quartiles over seeds)",
                   title=f"Every circuit: reinforcement needs about {typical:.0f}× the driving")
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(loc="center")
    fig.suptitle(f"From zero to a lap: imitation against reinforcement learning ({reference['settings']['seeds']} seeds per circuit)")
    fig.savefig(output / "visuals/journey_curves.png", dpi=140)
    plt.close(fig)


def track_name(catalog, track_id):
    return next(item["name"] for item in catalog if item["id"] == track_id)


def render(output, visuals=True):
    output = Path(output)
    data = json.loads((output / "data/lab.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    kit = json.loads(KIT_PATH.read_text())
    data["evolution"] = [{"update": update, "model": export_model(NeuralDriver.load(output / f"models/full_301_update_{update:03d}.npz"))} for update in range(0, 257, 32)]
    data["teaching"] = kit
    coverage = coverage_summary(output)
    if coverage:
        data["coverageReference"] = coverage["data"]["summary"]
    journey = journey_summary(output)
    if journey:
        data["journeyReference"] = journey["page"]
    write_json(output / "data/lab.json", data)
    for source, target in SCRIPTS:
        shutil.copyfile(source, output / target)
    template = Path("student_lab.html").read_text()
    (output / "index.html").write_text(template.replace("__DATA__", script_safe(data)))
    (output / "class_dashboard.html").write_text(Path("student_dashboard.html").read_text().replace("__KIT__", script_safe(kit)))
    catalog = new_catalog(WALL_CONFIG, json.loads(Path("wall_artifacts/data/circuits.json").read_text()))
    write_json(output / "data/circuits.json", catalog)
    (output / "teacher_guide.html").write_text(teacher_guide(data, kit, manifest))
    Path("STUDENT_V2_RESULTS.md").write_text(results_markdown(data, manifest, coverage, journey, output))
    if visuals:
        charts(data, output)
        animation(data, output, catalog)
        if journey:
            journey_charts(journey["data"], output, catalog)
        deck(data, kit, output, manifest, journey)
    print(f"Rendered {len(kit['modules'])} modules, teacher guide, class dashboard{', charts, checkpoint GIF and deck' if visuals else ''} in {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="student_v2_artifacts")
    parser.add_argument("--skip-visuals", action="store_true", help="Only the pages and documents (fast; for development)")
    arguments = parser.parse_args()
    render(arguments.output, not arguments.skip_visuals)
