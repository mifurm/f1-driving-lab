"""Render measured training results, animations, offline replay, and slide decks."""

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from driving import CHECKPOINTS, Config, Track, write_json


BACKGROUND = "#101722"
PANEL = "#182333"
WHITE = "#ecf2fa"
MUTED = "#9bacbf"
RED = "#ff5b65"
TEAL = "#4de2c2"
GOLD = "#ffc56b"
BLUE = "#76a9ff"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND, "axes.facecolor": PANEL,
    "axes.edgecolor": MUTED, "axes.labelcolor": WHITE, "text.color": WHITE,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#334358",
    "font.family": "DejaVu Sans", "font.size": 11,
    "savefig.facecolor": BACKGROUND, "axes.titleweight": "bold",
})


def save_figure(figure, directory, name, svg=False):
    figure.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
    if svg:
        figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def draw_track(axis, track):
    axis.fill(*np.vstack([track.left, track.right[::-1]]).T, color="#405064")
    for boundary in (track.left, track.right):
        closed = np.vstack([boundary, boundary[0]])
        axis.plot(*closed.T, color="#c5cfdb", lw=0.8)
    closed = np.vstack([track.xy, track.xy[0]])
    axis.plot(*closed.T, color="#69798c", ls="--", lw=0.6)
    start = np.vstack([track.left[0], track.right[0]])
    axis.plot(*start.T, color=WHITE, lw=3)
    axis.set_aspect("equal")
    axis.set_xlim(track.xy[:, 0].min() - 30, track.xy[:, 0].max() + 30)
    axis.set_ylim(track.xy[:, 1].min() - 30, track.xy[:, 1].max() + 30)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def load_trace(output, round_number, scenario=0):
    with np.load(output / f"trajectories/round_{round_number:03d}.npz") as data:
        return data[f"scenario_{scenario}"].copy()


def charts(output, track, manifest, history, details):
    directory = output / "visuals"
    rounds = np.array([int(row["round"]) for row in history])
    value = lambda name: np.array([float(row[name]) if row[name] != "" else np.nan for row in history])
    figure, axis = plt.subplots(figsize=(11, 7))
    draw_track(axis, track)
    axis.set_title("APEX PARK  /  ORIGINAL F1-INSPIRED CIRCUIT", pad=18, fontsize=17)
    speed = np.sqrt(3.5 / np.maximum(np.abs(track.curvature), 0.003))
    scatter = axis.scatter(*track.xy[::4].T, c=np.minimum(speed[::4], 32) * 3.6,
                           cmap="RdYlGn", s=9, zorder=4)
    figure.colorbar(scatter, ax=axis, shrink=0.6, label="Local teacher curvature speed cap (km/h)")
    for number, fraction in enumerate([0.0, 0.28, 0.63]):
        point = track.point(fraction * track.length)
        axis.annotate(f"TEST START {number + 1}", point, xytext=(12, 12),
                      textcoords="offset points", color=WHITE, fontsize=9,
                      bbox={"facecolor": BACKGROUND, "edgecolor": MUTED, "pad": 4})
    figure.text(0.1, 0.035,
                f"{track.length / 1000:.2f} km  |  {manifest['config']['track_width']:.0f} m wide"
                "  |  straights, sweeping turns, and tight direction changes",
                color=MUTED, fontsize=11)
    save_figure(figure, directory, "track", svg=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (0.35, 3.5, "1  BUILD THE WORLD", "Original circuit\nKinematic bicycle + limits"),
        (4.4, 3.5, "2  PREPARE EXAMPLES", "Teacher laps + recovery states\nIndependent validation data"),
        (8.45, 3.5, "3  TRAIN THE DRIVER", "13 → 48 → 32 → 2 MLP\nSteering + target speed"),
        (8.45, 0.7, "4  LET IT DRIVE", "Learner-only rollout\nTeacher labels visited states"),
        (4.4, 0.7, "5  REPEAT 120 TIMES", "24 Adam updates per round\nAggregate data; no RL reward"),
        (0.35, 0.7, "6  TEST & EXPLAIN", "3 fixed starts every round\nReplays, charts, and deck"),
    ]
    for x, y, title, subtitle in boxes:
        axis.add_patch(FancyBboxPatch((x, y), 3.2, 1.6, boxstyle="round,pad=0.12",
                                     facecolor=PANEL, edgecolor=TEAL, lw=1.4))
        axis.text(x + 0.15, y + 1.12, title, fontsize=11, weight="bold", color=TEAL)
        axis.text(x + 0.15, y + 0.55, subtitle, fontsize=10, linespacing=1.7)
    for start, end in [
        ((3.7, 4.3), (4.2, 4.3)), ((7.75, 4.3), (8.25, 4.3)),
        ((10.05, 3.3), (10.05, 2.55)), ((8.25, 1.5), (7.75, 1.5)),
        ((4.2, 1.5), (3.7, 1.5)), ((6.0, 2.5), (9.0, 3.3)),
    ]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                      mutation_scale=18, color=GOLD, lw=2))
    axis.set_title("A SIMPLE, AUDITABLE LEARNING LOOP", fontsize=19, pad=14)
    save_figure(figure, directory, "pipeline", svg=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(rounds, value("mean_progress_pct"), color=TEAL, lw=2)
    axes[0].fill_between(rounds, value("mean_progress_pct"), color=TEAL, alpha=0.1)
    axes[0].set(xlabel="Training round (0 = untrained)", ylabel="Average lap progress (%)",
                ylim=(0, 105), title="Can it finish the circuit?")
    axes[1].step(rounds, value("completed_scenarios"), color=GOLD, where="mid", lw=2)
    axes[1].set(xlabel="Training round", ylabel="Completed test scenarios",
                yticks=[0, 1, 2, 3], ylim=(-0.15, 3.15), title="Three fixed, learner-only tests")
    for axis in axes:
        axis.grid(alpha=0.4)
    figure.suptitle("120 REAL TRAINING ROUNDS  /  ALL RESULTS, INCLUDING REGRESSIONS", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, "progress")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(rounds, value("validation_weighted_mse"), color=TEAL, label="Independent validation")
    axes[0].semilogy(rounds, value("training_weighted_mse"), color=BLUE, alpha=0.8, label="Sampled training minibatches")
    axes[0].set(xlabel="Training round", ylabel="Weighted MSE (log scale)", title="Learning to imitate the teacher")
    axes[0].legend(fontsize=9)
    axes[1].plot(rounds, value("training_samples"), color=GOLD, lw=2)
    axes[1].set(xlabel="Training round", ylabel="Labeled training examples", title="The dataset grows after every drive")
    for axis in axes:
        axis.grid(alpha=0.4)
    figure.tight_layout()
    save_figure(figure, directory, "learning")

    selected = [0, 1, 5, 20, 60, 120]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, number in zip(axes.flat, selected):
        draw_track(axis, track)
        trace = load_trace(output, number)
        metric = next(row for row in details if row["round"] == number and row["scenario"] == "Start straight")
        axis.plot(trace[:, 1], trace[:, 2], color=TEAL if metric["completed"] else RED, lw=1.8)
        axis.scatter(*trace[-1, 1:3], color=GOLD, s=25, zorder=5)
        axis.set_title(f"Round {number}  |  {metric['progress'] * 100:.1f}%\n"
                       f"{metric['end_reason'].replace('_', ' ')}", fontsize=10)
    figure.suptitle("THE SAME START, THE SAME CIRCUIT, DIFFERENT WEIGHTS", fontsize=16)
    figure.tight_layout()
    save_figure(figure, directory, "trajectories")

    final_trace = load_trace(output, 120)
    with np.load(output / "trajectories/teacher.npz") as data:
        teacher_trace = data["scenario_0"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    channels = [(4, 3.6, "Speed (km/h)"), (5, 180 / np.pi, "Steering (degrees)"), (6, 1, "Lateral error (m)")]
    for axis, (column, scale, label) in zip(axes, channels):
        axis.plot(teacher_trace[:, 7] * 100, teacher_trace[:, column] * scale,
                  color=MUTED, ls="--", lw=1.2, label="Teacher")
        axis.plot(final_trace[:, 7] * 100, final_trace[:, column] * scale,
                  color=TEAL, lw=1.5, label="Round 120 learner")
        axis.set_ylabel(label)
        axis.grid(alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Distance around lap (%)")
    axes[-1].axhline(8.1, color=RED, ls=":", lw=1)
    axes[-1].axhline(-8.1, color=RED, ls=":", lw=1)
    figure.suptitle("DRIVING TECHNIQUE  /  FINAL LEARNER VS HEURISTIC TEACHER", fontsize=15)
    figure.tight_layout()
    save_figure(figure, directory, "technique")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    complete = value("completed_scenarios") == 3
    for axis, field, label, color in [
        (axes[0], "mean_abs_lateral_m", "Mean lateral error (m)", TEAL),
        (axes[1], "mean_steering_variation_rad_s", "Steering variation (rad/s)", GOLD),
        (axes[2], "mean_completed_lap_time_s", "Average lap time (s)", BLUE),
    ]:
        series = value(field)
        axis.plot(rounds[complete], series[complete], color=color, lw=1.3, marker=".", ms=3)
        axis.set(xlabel="Training round", ylabel=label)
        axis.grid(alpha=0.4)
    figure.suptitle("QUALITY AFTER RELIABLE COMPLETION  /  ONLY ROUNDS WITH 3 OF 3 LAPS", fontsize=14)
    figure.text(0.08, 0.01, "Lower error and steering variation are desirable; lap time is not the training objective.",
                color=MUTED, fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(figure, directory, "quality")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.axis("off")
    labels = ["Round", "Adam updates", "Examples", "Finished", "Progress", "Lateral error", "Val. MSE"]
    cells = []
    for number in CHECKPOINTS:
        row = history[number]
        cells.append([
            str(number), row["optimizer_updates"], f"{int(row['training_samples']):,}",
            f"{row['completed_scenarios']}/3", f"{float(row['mean_progress_pct']):.1f}%",
            f"{float(row['mean_abs_lateral_m']):.3f} m", f"{float(row['validation_weighted_mse']):.5f}",
        ])
    table = axis.table(cellText=cells, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#334358")
        cell.set_facecolor("#28403f" if row == len(cells) else PANEL if row else "#344459")
        cell.get_text().set_color(WHITE)
    axis.set_title("SAVED CHECKPOINTS  /  EXACT NUMBERS FROM THE RUN", pad=18, fontsize=16)
    save_figure(figure, directory, "checkpoint_table")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.axis("off")
    positions = [(0.08, "13 INPUTS", "Lane / heading / speed\nPrevious steering\n4 forward track previews"),
                 (0.38, "48 → 32 HIDDEN", "Two tanh layers\nNumPy backpropagation\nAdam optimizer"),
                 (0.73, "2 OUTPUTS", "Steering command\nTarget speed\nBounded low-level actuators")]
    for x, title, subtitle in positions:
        axis.add_patch(FancyBboxPatch((x, 0.2), 0.24, 0.57, transform=axis.transAxes,
                                     boxstyle="round,pad=0.025", edgecolor=TEAL, facecolor=PANEL))
        axis.text(x + 0.12, 0.65, title, ha="center", transform=axis.transAxes, color=TEAL, weight="bold")
        axis.text(x + 0.12, 0.4, subtitle, ha="center", transform=axis.transAxes, fontsize=10, linespacing=2)
    axis.text(0.35, 0.48, "→", transform=axis.transAxes, fontsize=28, ha="center", color=GOLD)
    axis.text(0.68, 0.48, "→", transform=axis.transAxes, fontsize=28, ha="center", color=GOLD)
    axis.set_title(f"A SMALL NEURAL DRIVER  /  {manifest['parameter_count']:,} TRAINABLE PARAMETERS", fontsize=16)
    save_figure(figure, directory, "architecture", svg=True)


def animations(output, track, details):
    comparisons = [0, 5, 20, 120]
    traces = [load_trace(output, number) for number in comparisons]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    artists = []
    for axis, number, trace in zip(axes.flat, comparisons, traces):
        draw_track(axis, track)
        metric = next(row for row in details if row["round"] == number and row["scenario"] == "Start straight")
        axis.set_title(f"Round {number}  /  {metric['end_reason'].replace('_', ' ')}", fontsize=11)
        line, = axis.plot([], [], color=TEAL if metric["completed"] else RED, lw=1.6)
        car = Polygon(np.zeros((4, 2)), closed=True, color=GOLD, zorder=6)
        axis.add_patch(car)
        text = axis.text(0.03, 0.03, "", transform=axis.transAxes, fontsize=9,
                         bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.9})
        artists.append((line, car, text))
    figure.suptitle("FROM UNTRAINED TO A FULL LAP  /  EQUAL SIMULATED TIME", fontsize=15)
    figure.tight_layout()
    maximum_time = max(trace[-1, 0] for trace in traces)
    times = np.linspace(0, maximum_time, 180)
    def frame(index):
        for trace, (line, car, text) in zip(traces, artists):
            cursor = min(int(np.searchsorted(trace[:, 0], times[index])), len(trace) - 1)
            state = trace[cursor]
            line.set_data(trace[:cursor + 1, 1], trace[:cursor + 1, 2])
            cosine, sine = np.cos(state[3]), np.sin(state[3])
            body = np.array([[3.8, 1.9], [3.8, -1.9], [-3.8, -1.9], [-3.8, 1.9]])
            rotation = np.array([[cosine, sine], [-sine, cosine]])
            car.set_xy(body @ rotation + state[1:3])
            text.set_text(f"{state[0]:.1f}s  |  {state[4] * 3.6:.0f} km/h  |  "
                          f"{min(state[7], 1) * 100:.1f}%")
        return [artist for group in artists for artist in group]
    animation = FuncAnimation(figure, frame, frames=len(times), interval=80, blit=False)
    animation.save(output / "visuals/training_comparison.gif", writer=PillowWriter(fps=12), dpi=85)
    plt.close(figure)
    for number in (0, 1, 5, 20, 60, 100, 120):
        trace = load_trace(output, number)
        metric = next(row for row in details if row["round"] == number and row["scenario"] == "Start straight")
        figure, axis = plt.subplots(figsize=(7, 6))
        draw_track(axis, track)
        line, = axis.plot([], [], color=TEAL if metric["completed"] else RED, lw=2)
        car, = axis.plot([], [], marker="o", color=GOLD, ms=7)
        status = axis.text(0.03, 0.03, "", transform=axis.transAxes)
        axis.set_title(f"APEX PARK  /  NEURAL DRIVER ROUND {number}", fontsize=13, pad=12)
        cursors = np.unique(np.linspace(0, len(trace) - 1, min(150, len(trace))).astype(int))
        def single_frame(frame_index):
            cursor = cursors[frame_index]
            state = trace[cursor]
            line.set_data(trace[:cursor + 1, 1], trace[:cursor + 1, 2])
            car.set_data([state[1]], [state[2]])
            status.set_text(f"t={state[0]:.1f}s  |  {state[4] * 3.6:.0f} km/h  |  "
                            f"lap={min(state[7], 1) * 100:.1f}%")
            return line, car, status
        animation = FuncAnimation(figure, single_frame, frames=len(cursors), interval=80)
        animation.save(output / f"visuals/round_{number:03d}.gif", writer=PillowWriter(fps=12), dpi=85)
        plt.close(figure)


REPLAY_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 Driving Lab — 120-round replay</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101722;color:#ecf2fa;font:16px system-ui,sans-serif}
header{padding:24px 32px;border-bottom:1px solid #334358}h1{margin:0 0 8px;font-size:28px}
p{color:#9bacbf;line-height:1.6}main{max-width:1320px;margin:auto;padding:24px;display:grid;grid-template-columns:2fr 1fr;gap:24px}
canvas{width:100%;background:#182333;border-radius:16px;border:1px solid #334358}
.panel{background:#182333;border:1px solid #334358;border-radius:16px;padding:20px}
label{display:block;margin:16px 0 8px;color:#9bacbf}button,select,input{font:inherit}select,button{background:#28403f;color:#ecf2fa;border:1px solid #4de2c2;padding:10px;border-radius:8px}
input[type=range]{width:100%;accent-color:#4de2c2}button{cursor:pointer;margin:12px 8px 8px 0}.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.stat{padding:12px;background:#101722;border-radius:8px}.value{font-size:23px;color:#4de2c2}.caption{font-size:12px;color:#9bacbf}
a{color:#4de2c2}img{max-width:100%;border-radius:12px}.charts{max-width:1320px;margin:auto;padding:0 24px 30px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:800px){main,.charts{grid-template-columns:1fr}header{padding:20px}}
</style></head><body>
<header><h1>APEX PARK / Teaching a fictional car to drive</h1>
<p>120 real training rounds · 121 saved evaluations · Select any round and test start. Everything runs offline.</p></header>
<main><div><canvas id="view" width="1000" height="760" aria-label="Animated overhead driving simulation"></canvas>
<p>Car drawn larger for visibility. Position, heading, speed, steering, and lane error come from saved simulations.
No teacher interventions in these replays. Collisions and lateral-acceleration violations stop the car.</p></div>
<section class="panel"><label for="round">Training round: <strong id="roundLabel">120</strong></label>
<input id="round" type="range" min="0" max="120" value="120" step="1">
<label for="scenario">Fixed test start</label><select id="scenario">
<option value="0">1 · Start straight</option><option value="1">2 · Corner entry</option><option value="2">3 · Recovery start</option></select>
<label for="speed">Playback speed</label><select id="speed"><option value="1">1× real simulated time</option><option value="4">4×</option><option value="8" selected>8×</option><option value="16">16×</option></select>
<div><button id="play">Pause</button><button id="restart">Restart</button></div>
<label for="scrub">Simulation timeline</label><input id="scrub" type="range" min="0" max="1000" value="0">
<div class="stats"><div class="stat"><div class="caption">Speed</div><div class="value" id="velocity"></div></div>
<div class="stat"><div class="caption">Lap progress</div><div class="value" id="progress"></div></div>
<div class="stat"><div class="caption">Lateral error</div><div class="value" id="lateral"></div></div>
<div class="stat"><div class="caption">Simulated time</div><div class="value" id="time"></div></div></div>
<p id="result"></p><p id="roundStats"></p>
<p><a href="training_deck.html">Open presentation</a> · <a href="training_deck.pptx">PowerPoint</a> · <a href="metrics.csv">All-round metrics</a></p>
<p>Imitation learning, not reinforcement learning. One known original circuit, perfect simulated sensors, and simplified physics.
Completion is the primary benchmark; shorter failed drives are not evidence of smoother technique.</p>
</section></main><div class="charts">
<img src="visuals/progress.png" alt="Progress and completed scenarios across all 120 rounds">
<img src="visuals/technique.png" alt="Final driver speed steering and lateral error compared with teacher"></div>
<script id="experiment-data" type="application/json">__DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("experiment-data").textContent);
const $=id=>document.getElementById(id), canvas=$("view"),ctx=canvas.getContext("2d");
let activeRound=120, scenario=0, time=0, playing=true, last=null, cursor=0;
const xs=data.track.xy.map(p=>p[0]),ys=data.track.xy.map(p=>p[1]);
const minX=Math.min(...xs)-30,maxX=Math.max(...xs)+30,minY=Math.min(...ys)-30,maxY=Math.max(...ys)+30;
const scale=Math.min(canvas.width/(maxX-minX),canvas.height/(maxY-minY));
const offsetX=(canvas.width-scale*(maxX-minX))/2,offsetY=(canvas.height-scale*(maxY-minY))/2;
const project=p=>[offsetX+(p[0]-minX)*scale,canvas.height-offsetY-(p[1]-minY)*scale];
function path(points,close=false){ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(close)ctx.closePath()}
function trace(){return data.traces[activeRound][scenario]}
function update(){
 const tr=trace();cursor=0;while(cursor+1<tr.length&&tr[cursor+1][0]<=time)cursor++;
 const s=tr[cursor],metric=data.details[activeRound*3+scenario];
 ctx.clearRect(0,0,canvas.width,canvas.height);
 path(data.track.left.concat([...data.track.right].reverse()),true);ctx.fillStyle="#405064";ctx.fill();
 ctx.strokeStyle="#cad4df";ctx.lineWidth=1.5;path(data.track.left,true);ctx.stroke();path(data.track.right,true);ctx.stroke();
 ctx.strokeStyle="#6c7d91";ctx.setLineDash([8,8]);path(data.track.xy,true);ctx.stroke();ctx.setLineDash([]);
 ctx.strokeStyle="#ecf2fa";ctx.lineWidth=5;path([data.track.left[0],data.track.right[0]]);ctx.stroke();
 path(tr.slice(0,cursor+1).map(row=>[row[1],row[2]]));ctx.strokeStyle=metric.completed?"#4de2c2":"#ff5b65";ctx.lineWidth=3;ctx.stroke();
 const p=project([s[1],s[2]]);ctx.save();ctx.translate(...p);ctx.rotate(-s[3]);
 ctx.fillStyle="#ffc56b";ctx.fillRect(-9,-5,18,10);ctx.fillStyle="#101722";ctx.fillRect(0,-3,5,6);
 ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.strokeRect(-9,-5,18,10);ctx.restore();
 ctx.fillStyle="#ecf2fa";ctx.font="bold 24px system-ui";ctx.fillText(`ROUND ${activeRound} / TEST ${scenario+1}`,24,36);
 $("velocity").textContent=(s[4]*3.6).toFixed(0)+" km/h";$("progress").textContent=(Math.min(s[7],1)*100).toFixed(1)+"%";
 $("lateral").textContent=s[6].toFixed(2)+" m";$("time").textContent=s[0].toFixed(1)+" s";
 $("scrub").value=Math.round(time/tr[tr.length-1][0]*1000);
 const finished=time>=tr[tr.length-1][0];
 $("result").textContent=(finished?"End: ":"Saved outcome: ")+metric.end_reason.replaceAll("_"," ")+
 ` · ${metric.completed?"Full lap in "+metric.duration_s.toFixed(1)+" s":(metric.progress*100).toFixed(1)+"% of a lap"}`;
 const h=data.history[activeRound];
 $("roundStats").textContent=`Weights updated ${h.optimizer_updates.toLocaleString()} times · ${h.training_samples.toLocaleString()} labeled examples · ${h.completed_scenarios}/3 fixed tests completed.`;
}
function reset(){activeRound=+$("round").value;scenario=+$("scenario").value;time=0;last=null;$("roundLabel").textContent=activeRound;update()}
$("round").addEventListener("input",reset);$("scenario").addEventListener("change",reset);
$("restart").addEventListener("click",()=>{reset();playing=true;$("play").textContent="Pause"});
$("play").addEventListener("click",()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";last=null});
$("scrub").addEventListener("input",()=>{const tr=trace();time=+$("scrub").value/1000*tr[tr.length-1][0];update()});
function tick(now){if(last!==null&&playing){const tr=trace();time=Math.min(time+Math.min((now-last)/1000,0.1)*+$("speed").value,tr[tr.length-1][0]);update()}last=now;requestAnimationFrame(tick)}
reset();requestAnimationFrame(tick);
</script></body></html>"""


def replay(output, track, history, details):
    traces = []
    for number in range(121):
        scenarios = []
        for scenario in range(3):
            trace = load_trace(output, number, scenario)
            indices = np.unique(np.r_[np.arange(0, len(trace), 3), len(trace) - 1])
            scenarios.append(np.round(trace[indices], 4).tolist())
        traces.append(scenarios)
    numeric_history = []
    for row in history:
        numeric_history.append({key: float(value) if value != "" else None for key, value in row.items()})
    payload = {
        "track": {name: np.round(getattr(track, name)[::3], 3).tolist() for name in ("xy", "left", "right")},
        "traces": traces, "details": details, "history": numeric_history,
    }
    document = REPLAY_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    (output / "replay.html").write_text(document, encoding="utf-8")


def slide_content(manifest, history):
    baseline, final = manifest["baseline"], manifest["final"]
    first = manifest["first_all_scenarios_complete_round"]
    val_reduction = baseline["validation_weighted_mse"] / final["validation_weighted_mse"]
    complete_rounds = [int(row["round"]) for row in history if int(row["completed_scenarios"]) == 3]
    regressions = sum(float(history[index]["mean_progress_pct"]) < float(history[index - 1]["mean_progress_pct"]) - 0.01
                      for index in range(1, len(history)))
    return [
        ("TEACHING A CAR TO DRIVE", [
            "A fictional car. An original F1-inspired circuit. A small neural network.",
            f"{manifest['actual_training_rounds']} training rounds • {manifest['actual_optimizer_updates']:,} real weight updates",
            f"Untrained progress {baseline['mean_progress_pct']:.1f}% → final progress {final['mean_progress_pct']:.1f}%",
        ], "track", "This is an educational simulation, not real F1 telemetry or a deployable vehicle controller."),
        ("WHAT MUST BE TRUE?", [
            "A closed track, road boundaries, and explicit vehicle limits.",
            "Useful observations and a reliable teacher.",
            "Recovery examples, repeated learning, and reproducible checkpoints.",
            "Fixed learner-only tests that can expose crashes and regressions.",
        ], "pipeline", "The complete assumptions and training contract are in PLAN.md. Success means three of three final test laps and better progress than the untrained baseline."),
        ("THE WORLD: APEX PARK", [
            f"{manifest['track_length_m'] / 1000:.2f} km original circuit • 18 m track width.",
            "Kinematic bicycle: steering, acceleration, braking, drag, and steering-rate limits.",
            "0.1 s simulation steps; off-track and excessive lateral acceleration end a drive.",
            "Not a reproduction of an official F1 circuit.",
        ], "track", "The car is 1.8 m wide with a 2.8 m wheelbase. The lateral-acceleration budget is 9 m/s²; speed is bounded at 36 m/s. These are toy-car settings, not F1 performance specifications."),
        ("THE DRIVER IS SMALL", [
            "13 inputs: lane error, heading, speed, previous steering, and four local track previews.",
            "48 and 32 hidden units; tanh activations.",
            f"2 outputs; {manifest['parameter_count']:,} trainable parameters.",
            "Output steering and target speed; low-level actuators enforce limits.",
        ], "architecture", "The model does not receive the teacher's action as an input. Perfect simulated geometry is available, so this is not an end-to-end vision task."),
        ("PREPARE THE DATA", [
            "10,000 randomized recovery states plus three teacher demonstration laps.",
            f"{manifest['initial_samples']:,} initial labeled examples.",
            "300 learner-visited states are teacher-labeled each round.",
            f"{manifest['final_samples']:,} final training examples; 2,000 separate validation examples.",
        ], "learning", "Data is synthesized locally. Source and aggregation round are recorded per example. The validation dataset is never used for weight updates."),
        ("HOW THE TEACHER DRIVES", [
            "Pure pursuit looks ahead and steers toward the centerline.",
            "Curvature previews reduce target speed before bends.",
            "Recovery steering also reduces target speed.",
            "A safe heuristic to imitate—not an optimal racing line.",
        ], "technique", "The teacher adapts lookahead distance to speed. A 66 m curvature preview and conservative lateral-acceleration target produce labels. Final traces are compared against the teacher, without claiming optimality."),
        ("120 REAL TRAINING ROUNDS", [
            "Drive with current weights; collect teacher labels for visited states.",
            "Append examples to the aggregation dataset.",
            "24 minibatch Adam updates per round; batch size 256.",
            "Decaying learning rate; steering errors weighted four times speed errors.",
        ], "pipeline", "This is DAgger-style imitation learning, not reinforcement learning. Exactly 120 outer training rounds execute 2,880 optimizer updates. Round 0 is evaluated before any weight update."),
        ("TEST WITHOUT INTERVENTION", [
            "Three fixed starting scenarios: straight, corner entry, and recovery.",
            "Evaluate all scenarios at round 0 and after every training round.",
            "Save 363 scenario outcomes and full trajectories.",
            "Primary measure: lap progress and completed scenarios.",
        ], "trajectories", "No teacher mixing or reset-based continuation is allowed inside an evaluation lap. Exact evaluation trajectories are not added to the training dataset. Repeated testing on one known track is monitoring, not a blind generalization test."),
        ("MEASURED IMPROVEMENT", [
            f"Average progress: {baseline['mean_progress_pct']:.1f}% → {final['mean_progress_pct']:.1f}%.",
            f"Completed starts: {baseline['completed_scenarios']}/3 → {final['completed_scenarios']}/3.",
            f"First 3/3 round: {first}; {len(complete_rounds)} rounds achieved 3/3.",
            f"{regressions} round-to-round progress regressions are retained in the chart.",
        ], "progress", "The plotted values are raw measurements, not smoothed claims of monotonic progress. A round can get worse even when validation loss decreases."),
        ("WHAT THE NETWORK LEARNED", [
            f"Independent action-validation error fell {val_reduction:.1f}×.",
            "Learned steering corrects lane and heading errors.",
            "Target speed follows the teacher's cautious cornering profile.",
            "The final lap shows the actual learned control signals.",
        ], "technique", "Validation measures imitation of teacher actions, while fixed closed-loop scenarios measure whether those action predictions suffice to drive. Neither metric proves real-world safety."),
        ("FINISHING IS NOT EVERYTHING", [
            f"Final mean lateral error: {final['mean_abs_lateral_m']:.3f} m.",
            f"Steering variation: {final['mean_steering_variation_rad_s']:.3f} rad/s; higher than round 6.",
            f"Final average completed lap: {final['mean_completed_lap_time_s']:.1f} s.",
            "Quality charts include only rounds that finish all three starts.",
        ], "quality", "Do not compare a short failed baseline trajectory with a full lap and infer smoother driving. Speed and lap time are descriptive, not the objective. Data points between incomplete rounds are connected only for readability."),
        ("CHECKPOINTS YOU CAN AUDIT", [
            "Untrained and nine learned model checkpoints saved.",
            "All 121 rounds have replay trajectories and test metrics.",
            "Final and best-by-completion/lateral-error models saved separately.",
            f"Best monitored round by that ranking: {manifest['best_round_by_completion_then_lateral_error']}.",
        ], "checkpoint_table", "The final model is round 120, not a substituted best model. best.npz is supplied separately and explicitly identified. Training and model outputs are compressed NPZ files readable with NumPy."),
        ("WATCH THE SIMULATIONS", [
            "Open replay.html; choose any of 121 rounds and any test start.",
            "Scrub time, pause, restart, or change playback speed.",
            "Comparison GIF uses equal simulated time across four rounds.",
            "Seven additional animated checkpoint drives are included.",
        ], "trajectories", "Use the offline interactive replay for all 121 rounds. GIFs are accelerated presentations of saved trajectories; they do not constitute new training runs. The car icon is enlarged for visibility."),
        ("LIMITATIONS & HONEST CLAIMS", [
            "One circuit, one seed, perfect geometry—not unseen-track generalization.",
            "No tires, aero, weather, vision, opponents, or real F1 telemetry.",
            "Repeated test monitoring is not a final blind holdout.",
            "A trained toy driver, not a controller for a physical car.",
        ], "track", "There is no statistical significance claim. Real-world autonomous driving would require much richer sensing, dynamics, verification, and safety controls. This project intentionally does not attempt deployment."),
        ("REPRODUCE & EXPLORE", [
            "Source: driving.py, train.py, render.py, verify.py and tests.",
            "Data: initial, aggregated, validation, track, and schema.",
            "Run: python train.py --output new_run; python render.py --output new_run.",
            "Open replay.html or this deck; inspect metrics.csv and manifest.json.",
        ], "progress", "Create the virtual environment and install requirements-lock.txt first; full commands are in README.md. verify.py checks artifacts and replays the final checkpoint to reproduce saved test metrics."),
    ]


def image_data(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def presentation(output, manifest, history, slides=None, replay_href="replay.html"):
    slides = slide_content(manifest, history) if slides is None else slides
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    def text(slide, x, y, w, h, value, size, color=WHITE, bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(0.02)
        paragraph = frame.paragraphs[0]
        paragraph.text = value
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
        return box
    html_slides = []
    for index, (title, paragraphs, image, notes) in enumerate(slides):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(BACKGROUND[1:])
        text(slide, 0.55, 0.28, 11.8, 0.3, "F1 DRIVING LAB  /  EXPERIMENT 2026", 11, TEAL, True)
        text(slide, 0.55, 0.85, 12.2, 0.8, title, 29, WHITE, True)
        for line, paragraph in enumerate(paragraphs):
            text(slide, 0.55, 1.95 + line * 1.04, 4.0, 0.98, paragraph, 18)
        path = output / f"visuals/{image}.png"
        with Image.open(path) as picture:
            width, height = picture.size
        factor = min(7.8 / width, 4.95 / height)
        picture_width, picture_height = width * factor, height * factor
        slide.shapes.add_picture(str(path), Inches(4.95 + (7.8 - picture_width) / 2),
                                 Inches(1.85 + (4.95 - picture_height) / 2),
                                 width=Inches(picture_width), height=Inches(picture_height))
        text(slide, 0.55, 7.05, 10.5, 0.25,
             "Original circuit • Simulated sensors • Imitation learning • Not for real vehicle deployment", 9, MUTED)
        text(slide, 12.1, 7.05, 0.65, 0.25, f"{index + 1:02d} / {len(slides)}", 9, MUTED)
        slide.notes_slide.notes_text_frame.text = notes
        if title == "WATCH THE SIMULATIONS":
            link = text(slide, 0.6, 6.35, 3.8, 0.4, "OPEN INTERACTIVE REPLAY", 15, TEAL, True)
            link.click_action.hyperlink.address = replay_href
        html_slides.append(
            f'<section class="slide{" active" if index == 0 else ""}">'
            '<div class="eyebrow">F1 DRIVING LAB / EXPERIMENT 2026</div>'
            f'<h1>{html.escape(title)}</h1><div class="content"><div>'
            + "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
            + f'</div><img src="{image_data(path)}" alt="{html.escape(title)}"></div>'
            f'<details><summary>Speaker notes</summary><p>{html.escape(notes)}</p></details>'
            f'<footer>Original circuit · Educational simulation · {index + 1} / {len(slides)}</footer></section>'
        )
    presentation.save(output / "training_deck.pptx")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Teaching a Car to Drive — Training Deck</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101722;color:#ecf2fa;font-family:system-ui,sans-serif}
.slide{display:none;max-width:1500px;margin:auto;padding:4vh 5vw;min-height:92vh}.slide.active{display:block}
.eyebrow{font-size:13px;color:#4de2c2;font-weight:700;letter-spacing:2px}h1{font-size:clamp(24px,3vw,48px);margin:24px 0}
.content{display:grid;grid-template-columns:1fr 2fr;gap:4vw;align-items:center}.content p{font-size:clamp(16px,1.6vw,25px);line-height:1.55;margin:24px 0}
img{width:100%;border-radius:10px}footer,details{color:#9bacbf;font-size:13px;margin-top:24px}
nav{position:sticky;bottom:0;padding:12px 5vw;background:#182333;display:flex;gap:12px;align-items:center}
button,a{color:#4de2c2;background:#101722;padding:9px 15px;border:1px solid #334358;border-radius:6px;font:inherit;cursor:pointer}
@media(max-width:800px){.content{grid-template-columns:1fr}.slide{padding:25px}.content p{margin:14px 0}}
@media print{.slide{display:block;break-after:page;height:100vh}.content{grid-template-columns:1fr 2fr}nav,details{display:none}}
</style></head><body>""" + "".join(html_slides) + """
<nav><button id="previous">← Previous</button><span id="count"></span><button id="next">Next →</button>
<a href="replay.html">Simulation replay</a><a href="training_deck.pptx">PowerPoint</a></nav>
<script>"use strict";const slides=[...document.querySelectorAll(".slide")];let current=0;
function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle("active",i===current));
document.getElementById("count").textContent=`${current+1} / ${slides.length}`;window.scrollTo(0,0)}
document.getElementById("previous").onclick=()=>show(current-1);document.getElementById("next").onclick=()=>show(current+1);
document.addEventListener("keydown",e=>{if(["ArrowRight"," ","PageDown"].includes(e.key)){e.preventDefault();show(current+1)}
if(["ArrowLeft","PageUp"].includes(e.key)){e.preventDefault();show(current-1)}});show(0);
</script></body></html>"""
    document = document.replace('href="replay.html"', f'href="{html.escape(replay_href, quote=True)}"')
    (output / "training_deck.html").write_text(document, encoding="utf-8")
    write_json(output / "presentation_outline.json", [
        {"slide": i + 1, "title": title, "body": body, "image": image, "notes": notes}
        for i, (title, body, image, notes) in enumerate(slides)
    ])


def preview_sheet(output):
    names = ["track", "pipeline", "architecture", "progress", "learning", "trajectories", "technique", "quality"]
    figure, axes = plt.subplots(4, 2, figsize=(14, 19))
    for axis, name in zip(axes.flat, names):
        with Image.open(output / f"visuals/{name}.png") as image:
            axis.imshow(image)
        axis.axis("off")
    figure.suptitle("F1 DRIVING LAB  /  ARTIFACT PREVIEW", fontsize=22)
    figure.tight_layout()
    save_figure(figure, output / "visuals", "artifact_preview")


def render(output, skip_animations=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest["success_criteria_met"]:
        raise RuntimeError("Training run did not meet its success criteria")
    with (output / "metrics.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    details = json.loads((output / "scenario_metrics.json").read_text())
    track = Track(Config(**manifest["config"]))
    charts(output, track, manifest, history, details)
    print("Rendered track, pipeline, architecture, metrics, and technique charts.", flush=True)
    replay(output, track, history, details)
    presentation(output, manifest, history)
    preview_sheet(output)
    print("Built offline replay (all 121 rounds) and 15-slide PowerPoint/HTML decks.", flush=True)
    if not skip_animations:
        animations(output, track, details)
        print("Rendered comparison animation and seven individual driving GIFs.", flush=True)
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("artifact_inventory.json", "verification.json"):
            inventory.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(output / "artifact_inventory.json", inventory)
    with zipfile.ZipFile(output / "training_deck.pptx") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("PowerPoint package failed CRC verification")
    print(f"Saved {len(inventory)} inventoried artifacts.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--skip-animations", action="store_true", help="Rebuild charts/decks without regenerating GIFs")
    arguments = parser.parse_args()
    render(arguments.output, arguments.skip_animations)
