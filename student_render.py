"""Build the offline student pages, measured charts, animation and classroom decks."""

import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt

from driving import NeuralDriver, Track, write_json
from student_experiments import export_model, new_catalog
from wall_sim import WALL_CONFIG, corner_scenarios, footprint, wall_simulate


LESSONS = [
    ("Predict before moving", "Change speed, lane offset or heading. Make a steering/braking prediction, reveal the real output, then replay.",
     "Does changing speed alter steering even when the lane offset stays fixed? Distinguish the policy's target speed from automatic braking."),
    ("Build your own dataset", "Drive with A/D and W/S, or explicitly choose expert collection. Capture inputs before commands and use separate episode splits.",
     "What will contradictory steering labels teach? Validation should use different drives and starts, not shuffled neighboring frames."),
    ("One real minibatch update", "Inspect activations, weighted errors, backpropagation, Adam deltas and an independent finite-difference gradient check.",
     "Explain why the Adam delta need not equal minus learning rate times gradient. Why can a loss increase on an individual step?"),
    ("Rules versus learned control", "Run a fixed rule, perfect-geometry teacher and neural policy with identical jittered starts and three paired seeds.",
     "A short failed run is not a fast lap. Compare safe centered exits, progress, handling failures and actual wall contacts separately."),
    ("Remove information and retrain", "Use four input masks and three seeds with identical examples, architectures, batch streams and optimization budgets.",
     "The no-heading variant still has orientation information in previews. The no-margin variant can infer margin from lane and heading."),
    ("Generalization and leakage", "Freeze the model, reveal three newly created test circuits, then intentionally train a separate clone on exposed test labels.",
     "Once outcomes are exposed, subsequent tuning is exploratory. A contaminated score is not an unseen-test result even if it improves."),
    ("Robustness and distribution shift", "Change real sensor noise, observation delay or the braking/handling envelope while keeping model and paired starts fixed.",
     "Previews remain ideal and the speed controller sees current speed. Grip does not model slip or rain. State these limitations."),
    ("Scientific worksheets", "Write a hypothesis first, vary exactly one factor, repeat the same three seeds, and export settings and raw outcomes.",
     "Sample standard deviation over three runs is not a confidence interval or evidence of real-car safety. Describe contradictory evidence."),
]


def fraction(rows):
    return 100 * sum(row["success"] for row in rows) / len(rows)


def curves(data, output):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for mask, color in zip(("full", "no_future_geometry", "no_heading", "no_margin"),
                           ("#4de2c2", "#ff8990", "#76a9ff", "#ffc56b")):
        histories = [data["histories"][f"{mask}_{seed}"] for seed in (301, 302, 303)]
        losses = np.array([[row["validation_mse"] for row in history] for history in histories])
        updates = [row["update"] for row in histories[0]]
        mean, sd = losses.mean(axis=0), losses.std(axis=0, ddof=1)
        axes[0].plot(updates, mean, color=color, label=mask)
        axes[0].fill_between(updates, np.maximum(mean - sd, 1e-6), mean + sd, alpha=.15, color=color)
    axes[0].set(xlabel="Actual Adam update", ylabel="Separate action-validation MSE (steering weight 4)", yscale="log",
                title="Genuine retraining: three matched seeds per mask")
    axes[0].legend(fontsize=8)
    names = list(("full", "no_future_geometry", "no_heading", "no_margin"))
    for offset, split, color in [(-.18, "train", "#4de2c2"), (.18, "pristine_test", "#ffc56b")]:
        rates = np.array([[fraction([row for row in data["ablations"] if row["mask"] == mask
                                   and row["seed"] == seed and row["split"] == split])
                           for seed in (301, 302, 303)] for mask in names])
        axes[1].bar(np.arange(4) + offset, rates.mean(axis=1), width=.35, color=color,
                    yerr=rates.std(axis=1, ddof=1), label=split, capsize=4)
    axes[1].set(xticks=np.arange(4), xticklabels=["All", "No previews/\ncurvature", "No direct\nheading", "No margin"],
                ylabel="Safe, centered section exits (%)", ylim=(0, 110), title="Selected only by training-track action validation")
    axes[1].legend(fontsize=8)
    fig.suptitle("Feature ablations: information removal is not guaranteed to hurt every measured score")
    fig.savefig(output / "visuals/feature_learning.png", dpi=150)
    plt.close(fig)
    conditions = [condition["name"] for condition in data["conditions"]]
    rows = [[row for row in data["robustness"] if row["condition"] == condition] for condition in conditions]
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    ax.bar(np.arange(len(rows)), [fraction(group) for group in rows], color=["#4de2c2", "#76a9ff", "#ffc56b", "#ff8990", "#be9cff"])
    ax.set(xticks=np.arange(len(rows)), xticklabels=["Clean", "Noise", "0.5 s delay", "65% grip", "Combined"],
           ylabel="Safe, centered section exits (%)", ylim=(0, 110),
           title="Real distribution shift / same pretrained weights and three paired initial-condition seeds")
    for index, group in enumerate(rows):
        ax.text(index, fraction(group) + 2, f"{sum(r['success'] for r in group)}/{len(group)}", ha="center")
    fig.savefig(output / "visuals/robustness.png", dpi=150); plt.close(fig)


def animation(data, output, catalog):
    track = Track(WALL_CONFIG, catalog[0]["knots"])
    scenario = corner_scenarios(track)[3]
    steps = [0, 128, data["models"]["full_301"]["updates"]]
    traces, metrics = [], []
    for step in steps:
        model = NeuralDriver.load(output / f"models/full_301_update_{step:03d}.npz")
        metric, trace = wall_simulate(track, WALL_CONFIG, model, scenario)
        traces.append(trace); metrics.append(metric)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), layout="constrained")
    lines, cars, footprints, hud = [], [], [], []
    for ax, step, metric in zip(axes, steps, metrics):
        ax.plot(track.left[:, 0], track.left[:, 1], "#ff5b65", lw=1)
        ax.plot(track.right[:, 0], track.right[:, 1], "#ff5b65", lw=1)
        ax.plot(track.xy[:, 0], track.xy[:, 1], "--", color="#78899e", lw=.7)
        line, = ax.plot([], [], "#4de2c2", lw=1.5); lines.append(line)
        car = Polygon(np.zeros((4, 2)), facecolor="#ffc56b", edgecolor="#ecf2fa", lw=.7); ax.add_patch(car); cars.append(car)
        boundary = Polygon(np.zeros((4, 2)), fill=False, edgecolor="#4de2c2", lw=1); ax.add_patch(boundary); footprints.append(boundary)
        text = ax.text(.02, .02, "", transform=ax.transAxes, fontsize=9); hud.append(text)
        ax.set_aspect("equal"); ax.set_title(f"Actual saved update {step}\n{metric['end_reason'].replace('_', ' ')}", fontsize=11)
        ax.tick_params(labelsize=7)
    fig.suptitle("Actual training checkpoints / equal simulated time / failures freeze, no scripted improvements", fontsize=12)
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
            ax.set_xlim(state[1] - 28, state[1] + 28); ax.set_ylim(state[2] - 28, state[2] + 28)
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
h1,h2{{color:#4de2c2}}p,li{{line-height:1.6}}a{{color:#ffc56b}}section{{padding:24px;margin:20px 0;border:1px solid #536780;border-radius:12px}}
img{{max-width:100%}}.response{{height:110px;border-bottom:1px solid #536780}}@media print{{body{{background:white;color:black}}section{{break-inside:avoid}}h1,h2{{color:black}}}}</style>
<h1>{html.escape(title)}</h1><p><a href="index.html">Open the interactive laboratory</a></p>{body}</html>"""


def docs(data, output):
    manifest = json.loads((output / "manifest.json").read_text())
    lesson_sections = []
    for index, (title, action, question) in enumerate(LESSONS, 1):
        lesson_sections.append(f"<section><h2>{index}. {html.escape(title)}</h2><p><strong>Action:</strong> {html.escape(action)}</p>"
                               f"<p><strong>Evidence and reflection:</strong> {html.escape(question)}</p>"
                               "<p>Hypothesis / settings / evidence / conclusion:</p><div class='response'></div></section>")
    intro = f"""<section><h2>Teacher setup and protocol</h2><p>Double-click index.html. No server, account, Azure subscription or network is needed.
    Existing 120-round wall/corner work is preserved. New reference experiments trained 12 compact 18→24→16→2
    feature-ablation models for 256 genuine Adam updates each, plus a matched clean/contaminated leakage pair.</p>
    <p>There are {manifest['training_samples']:,} fixed training examples and {manifest['validation_samples']:,} independently
    generated action-validation examples from tracks 0–10. Starter data contains 256 real expert rollout frames
    grouped by their original 180-step driving episodes and 64 independent validation observations.</p>
    <p>All clean weights and action-validation selections were frozen and hashed in training_complete.json
    before ML evaluation or label generation on tracks 13–15. Teacher-only geometry feasibility checks happened beforehand.
    Old holdouts 11–12 are exposed legacy holdouts, not pristine new tests.</p>
    <p>Keep teacher-only reference test charts closed until students freeze their own policies. The seal is procedural;
    offline source data and this teacher deck can be inspected. Once exposed, tuning results are exploratory.</p>
    <p>Anonymous test-exposure flags persist across page reloads. The five saved example JSON downloads in data/
    are browser-validation demonstrations, not submissions from actual students.</p>
    <p>The neural model learns normalized steering and target speed by imitation, not rewards or reinforcement learning.
    Pedals, gears and RPM follow ordinary control rules. Wall contact uses a 4.8 m × 1.8 m oriented rectangular footprint
    with sampled swept checks, not a continuous impact solver. There is no learned safety guarantee, slip model or weather model.</p></section>"""
    closing = """<section><h2>Pre/post concept checks and answer key</h2><p>Before using the lab, ask what changes during training,
    what labels represent, why neighboring frames can leak and whether low loss guarantees safety.
    Repeat afterward; the in-app five-question quiz gives real answer feedback.</p>
    <p>Answers: model weights change; steering and target-speed commands are labels; separate driving episodes avoid
    adjacent-frame leakage; freeze before exposing test outcomes; low action loss alone proves neither completion nor safety.</p>
    <p>Suggested class: 15 min prediction/data, 20 min gradient/control, 20 min ablation/generalization,
    20 min robustness/scientific report. Discuss failures rather than hiding them.</p></section>"""
    (output / "teacher_guide.html").write_text(document("Teacher Guide and Eight Printable ML Worksheets", intro + "".join(lesson_sections) + closing))
    pretrained_rows = [row for row in data["comparisons"] if row["controller"] == "pretrained"]
    fresh_rows = [row for row in pretrained_rows if row["split"] == "pristine_test"]
    results = f"""# Student ML Driving Lab

Open `student_artifacts/index.html` for all eight interactive modules. The existing wall/corner lab is unchanged.
Open `student_artifacts/teacher_guide.html` for printable worksheets and `student_artifacts/training_deck.pptx`
or `training_deck.html` for the new classroom deck.

## What was actually executed

- Four feature masks × three matched seeds × 256 Adam updates: **12 genuinely retrained compact models**.
- A clean/contaminated leakage pair also executes 256 updates each: **3,584 new reference optimizer updates total**.
- **{manifest['training_samples']:,} training / {manifest['validation_samples']:,} independent action-validation examples**.
- **768 ablation, 576 controller and 960 robustness section rollouts**; real raw results are included.
- Three original new test circuits, IDs13–15, were not evaluated by any clean ML policy until every clean model
  and its action-validation selection had been frozen. Legacy holdouts11–12 are explicitly not pristine.
- The frozen original wall-aware driver completes **{sum(r['success'] for r in fresh_rows)}/{len(fresh_rows)}**
  paired-seed sections on the new circuits. Compact student-sized models are not promised to match the larger
  previously trained model.

## Hands-on functionality

Predict steering/braking before revealing actual outputs; record manual or explicitly marked expert drives;
import/export examples; fit a fresh model in the browser; inspect real gradients, activations and Adam deltas;
replay untrained/intermediate/trained weights; compare actual controller traces at equal time; retrain all
feature variants; freeze/reveal tests; deliberately demonstrate invalid test-label training on a separate clone;
perturb real observations, history and grip; run one-factor paired-seed worksheets and export raw evidence.

Starter rollout frames are grouped by their original 180-step episode. The separate starter validation frames
are independently sampled labeled states, not an invented validation drive. Recorded labels correspond to
pre-step observations. Inputs are masked consistently during fitting and inference.

## Interpretation and limitations

This is imitation learning, not reinforcement learning. The network outputs steering and target speed;
the automatic controller/gearbox determines actual brake, throttle and gears. Imported legacy weights restart
Adam moments and its counter; exported weight files do not falsely claim to preserve optimizer history.

“No heading” also removes the derived margin, but previews can encode orientation. “No margin” leaves lane
and heading from which margin can be inferred. Feature ablation is not proof of unique causal importance.
Action-validation checkpoint selection is not safety checkpoint selection.

Noise perturbs lane, heading and speed but not ideal geometric previews. Delay holds actual prior policy
observations; the ordinary speed controller still senses current speed. Grip scales braking and the handling
limit, not tire slip or weather. No real-F1 physics, real-car safety or impact response is claimed.

Successful centered exits, progress, wall contacts and handling-limit failures are reported separately.
Short failures are never reported as faster completed laps. Sample SD across three paired seeds is not a
confidence interval. Browser and NumPy use different reproducible random streams.

The test seal is a classroom protocol, not secrecy. Anonymous exposure flags persist across page reloads.
The saved example JSON downloads are browser-validation demonstrations, not actual student submissions.
Once test outcomes/labels have been exposed, subsequent
tuning is exploratory. A deliberately contaminated clone is separated from clean models and its scores
cannot support generalization claims. No student data or telemetry is transmitted.

## Reproduce

```sh
.venv/bin/python student_experiments.py --output student_artifacts_reproduction --base wall_artifacts
.venv/bin/python student_render.py --output student_artifacts_reproduction
STUDENT_OUTPUT=student_artifacts_reproduction npm run test:student
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser:student -- student_artifacts_reproduction
.venv/bin/python student_verify.py --output student_artifacts_reproduction
```

The portable additive bundle contains the student lab, required source/data, reports and the complete previous
wall/corner lab. Installed environments/browser binaries and older large standalone bundles are excluded.
"""
    Path("STUDENT_RESULTS.md").write_text(results)
    slides = [
        ("From watching a driver to testing an ML idea", ["Eight hands-on lessons on top of the preserved driving simulator",
          "Real supervised updates and simulated motion, not animation-only improvement", "The car remains fictional; no real-driving safety claims"]),
    ]
    slides.extend((f"{index}. {title}", [action, question]) for index, (title, action, question) in enumerate(LESSONS, 1))
    slides.extend([
        ("Measured feature-learning experiments", ["4 masks × 3 identical seed schedules × 256 actual Adam updates",
          "5,632 teacher-labeled training states; 2,200 independent action-validation states",
          "256 updates were executed even when an earlier action-validation checkpoint was selected"], "visuals/feature_learning.png"),
        ("Genuine unseen-layout evaluation", [f"Frozen original pretrained driver: {sum(r['success'] for r in fresh_rows)}/{len(fresh_rows)} new-track paired sections",
          "Three newly generated test circuits; all clean weights frozen before test evaluation",
          "Legacy holdouts are exposed; post-reveal tuning is exploratory"], "visuals/checkpoint_comparison.png"),
        ("Distribution shift changes actual outcomes", ["Sensor noise, historical observations and reduced braking/handling envelope",
          "Same pretrained driver and paired seed schedule in all five reference conditions",
          "Report failures honestly; no tire-slip, weather or real-car validation"], "visuals/robustness.png"),
        ("The scientific habit to take away", ["Hypothesis → controlled intervention → repeated evidence → qualified conclusion",
          "Lower training loss is not completion; completion is not a larger wall buffer",
          "Keep raw trajectories, failure cases, seeds, model fingerprints and contamination flags"]),
    ])
    presentation = Presentation(); presentation.slide_width = Inches(13.333); presentation.slide_height = Inches(7.5)
    sections = []
    for index, specification in enumerate(slides, 1):
        title, bullets = specification[:2]; image = specification[2] if len(specification) > 2 else None
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        title_box = slide.shapes.add_textbox(Inches(.6), Inches(.35), Inches(12), Inches(.9))
        title_box.text_frame.text = title; title_box.text_frame.paragraphs[0].font.size = Pt(30)
        if image:
            slide.shapes.add_picture(str(output / image), Inches(.6), Inches(2.5), width=Inches(12), height=Inches(4.5))
        text_box = slide.shapes.add_textbox(Inches(.65), Inches(1.35), Inches(12), Inches(1.4 if image else 4.5))
        for row_index, bullet in enumerate(bullets):
            paragraph = text_box.text_frame.paragraphs[0] if row_index == 0 else text_box.text_frame.add_paragraph()
            paragraph.text = bullet; paragraph.font.size = Pt(16 if image else 25); paragraph.space_after = Pt(12)
        slide.notes_slide.notes_text_frame.text = "Use the corresponding interactive module. " + " ".join(bullets) + \
            " Distinguish learned steering/target speed from automatic pedals/gears; do not claim real-car safety."
        sections.append(f"<section><h2>{index}. {html.escape(title)}</h2><ul>" +
                        "".join(f"<li>{html.escape(bullet)}</li>" for bullet in bullets) +
                        "</ul>" + (f'<img src="{image}">' if image else "") + "</section>")
    presentation.save(output / "training_deck.pptx")
    (output / "training_deck.html").write_text(document("Classroom Deck: Eight Ways to Understand ML Driving", "".join(sections)))
    write_json(output / "deck_manifest.json", {"slides": len(slides), "worksheets": len(LESSONS), "speaker_notes": True})


def render(output):
    output = Path(output)
    data = json.loads((output / "data/lab.json").read_text())
    for row in data["starter"]:
        if row["source"] == "expert rollout":
            row["episode"] = f"initial-expert-episode-{(int(row['id'].split('-')[-1]) - 9900) // 180}"
    data["evolution"] = [{"update": update, "model": export_model(NeuralDriver.load(output / f"models/full_301_update_{update:03d}.npz"))}
                         for update in range(0, 257, 32)]
    write_json(output / "data/lab.json", data)
    shutil.copyfile("student_engine.js", output / "engine.js"); shutil.copyfile("student_app.js", output / "app.js")
    template = Path("student_lab.html").read_text()
    (output / "index.html").write_text(template.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    catalog = new_catalog(WALL_CONFIG, json.loads(Path("wall_artifacts/data/circuits.json").read_text()))
    write_json(output / "data/circuits.json", catalog)
    curves(data, output); animation(data, output, catalog); docs(data, output)
    print(f"Rendered all 8 modules, real checkpoint GIF, measured charts, worksheets and both classroom decks in {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="student_artifacts")
    render(parser.parse_args().output)
