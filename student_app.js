"use strict";
const E = window.StudentEngine, data = JSON.parse(document.getElementById("experiment-data").textContent);
const $ = id => document.getElementById(id), cfg = data.config;
const tracks = data.tracks.map(track => new E.Track(track));
const reference = E.Network.fromJSON(data.pretrained);
const lab = {
  rows: [], student: null, losses: [], trainingSnapshots: [], busy: false, module: "predict",
  episode: 0, manual: null, recording: false, packet: null, stage: 0,
  runs: [], replayTime: 0, replayPlaying: false, last: null,
  testExposed: false, testLabelsExposed: false, frozen: null, exposureStorage: true, quizAttempts: [],
  comparisons: null, robustness: null, worksheet: null, localAblations: null,
};
window.studentLab = lab;
const colors = ["#ffc56b", "#76a9ff", "#4de2c2"];
const tickYield = () => new Promise(resolve => requestAnimationFrame(resolve));
function number(id, minimum, maximum, integer = false) {
  const value = Number($(id).value);
  if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value)))
    throw new Error(`${id}: choose ${integer ? "an integer" : "a number"} between ${minimum} and ${maximum}`);
  return value;
}
function message(text) { $("status").textContent = text; }
async function handle(action) {
  if (lab.busy) { $("error").textContent = "Wait for the current experiment to finish."; return; }
  $("error").textContent = "";
  try { await action(); }
  catch (error) {
    lab.recording = false;
    $("error").textContent = error.message;
    console.error(error);
  } finally { lab.busy = false; }
}
function selectedModel() {
  if ($("model").value === "student") {
    if (!lab.student) throw new Error("Train or initialize a student model first.");
    return lab.student;
  }
  return reference;
}
function currentTrack() { return tracks.find(track => track.id === Number($("track").value)); }
function canonical() { return data.scenarios[currentTrack().id][Number($("scenario").value)]; }
function start() {
  return { ...canonical(), lateral: Number($("startLateral").value),
    heading_error: Number($("startHeading").value) * Math.PI / 180,
    speed: Number($("startSpeed").value) / 3.6 };
}
function fillTracks() {
  const selected = $("track").value;
  $("track").replaceChildren();
  for (const track of tracks.filter(t => lab.testExposed || t.id < 13)) {
    const option = document.createElement("option");
    option.value = track.id; option.textContent = `${track.name}${track.id >= 13 ? " / EXPOSED TEST" : track.id >= 11 ? " / legacy holdout" : ""}`;
    $("track").append(option);
  }
  if ([...$("track").options].some(option => option.value === selected)) $("track").value = selected;
}
function fillScenarios() {
  $("scenario").replaceChildren();
  data.scenarios[currentTrack().id].forEach((scenario, i) => {
    const option = document.createElement("option"); option.value = i; option.textContent = scenario.name; $("scenario").append(option);
  });
  $("scenario").value = "2";
  resetStart();
}
function resetStart() {
  lab.recording = false;
  const scenario = canonical();
  $("startSpeed").value = Math.round(scenario.speed * 3.6);
  $("startLateral").value = scenario.lateral;
  $("startHeading").value = Math.round(scenario.heading_error * 180 / Math.PI);
  preview();
}
function table(id, headings, rows) {
  const element = $(id); element.replaceChildren();
  const head = document.createElement("thead"), tr = document.createElement("tr");
  for (const heading of headings) { const th = document.createElement("th"); th.textContent = heading; tr.append(th); }
  head.append(tr); element.append(head);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const value of row) { const td = document.createElement("td"); td.textContent = String(value); line.append(td); }
    body.append(line);
  }
  element.append(body);
}
function preview() {
  $("startSpeedLabel").textContent = `${$("startSpeed").value} km/h`;
  $("startLateralLabel").textContent = `${$("startLateral").value} m`;
  $("startHeadingLabel").textContent = `${$("startHeading").value}°`;
  const env = new E.Environment(currentTrack(), cfg); env.reset(start());
  const features = env.observe().features;
  table("inputTable", ["Input", "Value"], features.map((v, i) => [data.features[i], v.toFixed(5)]));
  $("predictionResult").hidden = true; $("guessFeedback").textContent = "";
  showRuns([{ name: "Starting state / prediction hidden", track: currentTrack(), trace: [env.record()] }], false);
}
function showRuns(runs, play = true) {
  lab.runs = runs; lab.replayTime = 0; lab.replayPlaying = play; lab.last = null; lab.recording = false;
  $("replayPlay").textContent = play ? "Pause replay" : "Play replay";
  draw();
}
function sample(trace, time) {
  let index = 0; while (index + 1 < trace.length && trace[index + 1].time <= time) index++;
  const a = trace[index], b = trace[Math.min(index + 1, trace.length - 1)];
  const f = b.time === a.time ? 0 : E.clip((time - a.time) / (b.time - a.time), 0, 1), state = { ...a };
  for (const key of ["x", "y", "speed", "lateral", "headingError", "time", "progress"])
    state[key] = a[key] + (b[key] - a[key]) * f;
  state.heading = a.heading + E.wrap(b.heading - a.heading) * f;
  return { state, index };
}
function paintCar(c, state, point, scale, color) {
  c.save(); c.translate(...point); c.rotate(-state.heading); c.scale(scale, -scale);
  c.fillStyle = "#05090f"; c.strokeStyle = "#acbbce"; c.lineWidth = .06;
  for (const [x, y] of [[1.2, .55], [1.2, -.55], [-1.3, .55], [-1.3, -.55]]) {
    c.save(); c.translate(x, y); if (x > 0) c.rotate(state.steering);
    c.fillRect(-.425, -.15, .85, .3); c.strokeRect(-.425, -.15, .85, .3); c.restore();
  }
  c.fillRect(1.9, -.9, .35, 1.8); c.fillRect(-2.1, -.8, .35, 1.6);
  c.beginPath(); c.moveTo(2.3, 0); c.lineTo(1, .35); c.lineTo(-1.5, .43); c.lineTo(-2, 0);
  c.lineTo(-1.5, -.43); c.lineTo(1, -.35); c.closePath(); c.fillStyle = color; c.fill();
  c.fillStyle = "#ecf2fa"; c.fillRect(-.65, -.14, .7, .28);
  c.fillStyle = state.brake > .02 ? "#ff5b65" : "#54252a"; c.fillRect(-1.9, -.55, .15, .17); c.fillRect(-1.9, .38, .15, .17);
  c.strokeStyle = state.contact ? "#ff5b65" : "#4de2c2"; c.lineWidth = .07; c.strokeRect(-2.4, -.9, 4.8, 1.8);
  c.restore();
}
function drawScene(c, run, x, width, state, index, color) {
  const track = run.track, height = 620, xs = track.xy.map(p => p[0]), ys = track.xy.map(p => p[1]);
  const bounds = $("camera").value === "follow" ? [state.x - 45, state.x + 45, state.y - 40, state.y + 40] :
    [Math.min(...xs) - 20, Math.max(...xs) + 20, Math.min(...ys) - 20, Math.max(...ys) + 20];
  const [minX, maxX, minY, maxY] = bounds, scale = Math.min((width - 30) / (maxX - minX), (height - 80) / (maxY - minY));
  const ox = x + (width - scale * (maxX - minX)) / 2, oy = 42 + (height - 65 - scale * (maxY - minY)) / 2;
  const project = p => [ox + (p[0] - minX) * scale, oy + (maxY - p[1]) * scale];
  const path = (points, close = false) => {
    c.beginPath(); points.forEach((p, i) => i ? c.lineTo(...project(p)) : c.moveTo(...project(p))); if (close) c.closePath();
  };
  c.save(); c.beginPath(); c.rect(x, 0, width, height); c.clip(); c.fillStyle = "#182333"; c.fillRect(x, 0, width, height);
  path(track.left.concat([...track.right].reverse()), true); c.fillStyle = "#405064"; c.fill();
  c.strokeStyle = "#ff5b65"; c.lineWidth = 1.5; path(track.left, true); c.stroke(); path(track.right, true); c.stroke();
  c.strokeStyle = "#78899e"; c.lineWidth = 1; c.setLineDash([6, 6]); path(track.xy, true); c.stroke(); c.setLineDash([]);
  path([...run.trace.slice(0, index + 1).map(s => [s.x, s.y]), [state.x, state.y]]); c.strokeStyle = color; c.lineWidth = 2; c.stroke();
  paintCar(c, state, project([state.x, state.y]), scale, color);
  c.fillStyle = "#ecf2fa"; c.font = `bold ${width > 500 ? 19 : 13}px system-ui`; c.fillText(run.name, x + 12, 25);
  c.font = "12px system-ui"; c.fillText(`${state.time.toFixed(1)}s | ${(state.speed * 3.6).toFixed(0)} km/h | G${state.gear} | margin ${state.margin.toFixed(2)}m`, x + 12, height - 18);
  if (state.contact) { c.fillStyle = "#ff5b65"; c.font = "bold 22px system-ui"; c.fillText("WALL CONTACT", x + 12, 60); }
  c.restore();
}
function draw() {
  const cv = $("view"), c = cv.getContext("2d"); c.clearRect(0, 0, cv.width, cv.height);
  const visible = lab.runs.slice(0, 3);
  for (const [i, run] of visible.entries()) {
    const pose = sample(run.trace, Math.min(lab.replayTime, run.trace.at(-1).time));
    drawScene(c, run, i * cv.width / visible.length, cv.width / visible.length - 2, pose.state, pose.index, colors[i]);
    if (i === 0) {
      const s = pose.state;
      $("telemetry").textContent = `${s.time.toFixed(1)} s | ${(s.speed * 3.6).toFixed(1)} km/h | target ${(s.targetSpeed * 3.6).toFixed(1)} km/h
steer ${(s.steering * 180 / Math.PI).toFixed(1)}° | brake ${(s.brake * 100).toFixed(0)}% | throttle ${(s.throttle * 100).toFixed(0)}% | G${s.gear} / ${s.rpm.toFixed(0)} RPM
wall margin ${s.margin.toFixed(3)} m | offset ${s.lateral.toFixed(3)} m | ${run.metric?.end_reason ?? "live starting state"}`;
    }
  }
  const duration = Math.max(0, ...visible.map(run => run.trace.at(-1).time));
  $("replayTime").value = duration ? Math.round(lab.replayTime / duration * 1000) : 0;
}
function datasets() {
  E.validateRows(lab.rows);
  const train = lab.rows.filter(row => row.split === "train"), validation = lab.rows.filter(row => row.split === "validation");
  if (!train.length || !validation.length) throw new Error("Capture separate training and validation drives, or load the expert starter dataset.");
  return { train, validation };
}
function updateDataset() {
  $("datasetCounts").textContent = `${lab.rows.filter(r => r.split === "train").length} training / ${lab.rows.filter(r => r.split === "validation").length} validation / ${lab.rows.length} total examples`;
  table("datasetTable", ["ID / episode", "Split / source", "Offset / speed inputs", "Steering / target labels"],
    lab.rows.slice(-30).map(row => [row.id, `${row.split} / ${row.source}`, `${row.features[0].toFixed(3)} / ${row.features[3].toFixed(3)}`,
      row.targets.map(v => v.toFixed(3)).join(" / ")]));
}
function download(name, value, type = "application/json") {
  const url = URL.createObjectURL(new Blob([type === "application/json" ? JSON.stringify(value, null, 2) : value], { type }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function audit(model = selectedModel()) { return { engine: E.version, testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed, model: E.fingerprint(model) }; }
function setExposure() {
  if (lab.exposureStorage) {
    try {
      localStorage.setItem("f1-student-lab-exposure-v1", JSON.stringify({ testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed }));
    } catch (error) {
      if (!["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
      lab.exposureStorage = false;
      $("error").textContent = "Persistent exposure storage is unavailable. Do not claim a pristine test session after reopening this page.";
    }
  }
  $("testExposure").textContent = lab.testExposed ?
    `Test circuits now exposed. Further retraining/testing is exploratory${lab.testLabelsExposed ? "; test labels were exposed in the leakage lesson" : ""}.` :
    "Three new test circuits are sealed. Legacy holdouts are not pristine.";
}
function restoreExposure() {
  try {
    const raw = localStorage.getItem("f1-student-lab-exposure-v1");
    const stored = raw === null ? { testExposed: false, testLabelsExposed: false } : JSON.parse(raw);
    if (typeof stored.testExposed !== "boolean" || typeof stored.testLabelsExposed !== "boolean")
      throw new SyntaxError("Invalid exposure record");
    lab.testExposed = stored.testExposed || stored.testLabelsExposed; lab.testLabelsExposed = stored.testLabelsExposed;
    localStorage.setItem("f1-student-lab-exposure-v1", JSON.stringify(stored));
  } catch (error) {
    if (!["SecurityError", "QuotaExceededError", "SyntaxError"].includes(error.name)) throw error;
    lab.exposureStorage = false; lab.testExposed = true; lab.testLabelsExposed = true;
    $("error").textContent = "Prior exposure cannot be verified: this session is conservatively treated as already exposed, not pristine.";
  }
  setExposure();
}
function drawLoss() {
  const cv = $("lossCanvas"), c = cv.getContext("2d"); c.clearRect(0, 0, cv.width, cv.height);
  c.fillStyle = "#ecf2fa"; c.font = "bold 16px system-ui"; c.fillText("ACTUAL LOSS / cyan training, gold separate validation", 55, 26);
  if (!lab.losses.length) return;
  const max = Math.max(1e-5, ...lab.losses.flatMap(p => [p.train, p.validation])) * 1.1, last = Math.max(1, lab.losses.at(-1).update);
  for (const [field, color] of [["train", "#4de2c2"], ["validation", "#ffc56b"]]) {
    c.beginPath(); lab.losses.forEach((p, i) => { const x = 55 + p.update / last * 990, y = 205 - p[field] / max * 160; i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.strokeStyle = color; c.lineWidth = 2; c.stroke();
  }
  c.fillStyle = "#acbbce"; c.font = "13px system-ui"; c.fillText(`0 to ${last} actual optimizer updates; max loss ${max.toFixed(4)}`, 55, 232);
}
function manualStep() {
  if (!lab.manual) throw new Error("Start a new driving episode first.");
  if (lab.manual.reason) throw new Error("This episode ended. Start a new drive; resets are not joined.");
  if (lab.rows.length >= 20000) throw new Error("Dataset limit reached; download and clear your examples.");
  const env = lab.manual.env, features = env.observe().features;
  const action = $("expertRecording").checked ? env.teacher() :
    [Number($("manualSteer").value) * Math.PI / 180 / cfg.max_steer, 2 * Number($("manualTarget").value) / 3.6 / cfg.max_speed - 1];
  const row = { id: `drive-${lab.episode}-${lab.manual.trace.length}`, episode: `drive-${lab.episode}`,
    features, targets: action, split: lab.manual.split, track_id: env.track.id,
    source: $("expertRecording").checked ? "expert driving (not student)" : "student manual commands" };
  lab.rows.push(row);
  const result = env.step(action); lab.manual.trace.push(env.record()); lab.manual.reason = result.reason;
  lab.runs = [{ name: row.source, track: env.track, trace: lab.manual.trace }]; lab.replayTime = env.time;
  if (result.reason) lab.recording = false;
  $("episodeStatus").textContent = `Episode ${lab.episode}, ${lab.manual.split}: ${lab.manual.trace.length - 1} recorded actions. ${result.reason || "driving"}`;
  updateDataset(); draw();
}
async function fitLocal(seed, mask, train, validation, updates, learningRate, onProgress) {
  const net = new E.Network(seed, [18, 24, 16, 2], mask), rng = new E.RNG(seed + 100);
  let best = net.clone(), bestLoss = net.loss(validation);
  for (let update = 1; update <= updates; update++) {
    const batch = Array.from({ length: 64 }, () => train[rng.index(train.length)]);
    net.trainBatch(batch, learningRate);
    if (update % 4 === 0 || update === updates) {
      const loss = net.loss(validation);
      if (loss < bestLoss) { bestLoss = loss; best = net.clone(); }
      if (onProgress) onProgress(net, update);
      await tickYield();
    }
  }
  return { last: net, best, bestLoss };
}
function pairedStart(base, seed) {
  const rng = new E.RNG(seed);
  return { ...base, lateral: base.lateral + (rng.uniform() - .5) * .3,
    heading_error: base.heading_error + (rng.uniform() - .5) * .02, speed: base.speed + (rng.uniform() - .5) * .4 };
}
function summarizeTrials(rows, labelKey) {
  const labels = [...new Set(rows.map(row => row[labelKey]))];
  return labels.map(label => {
    const trials = rows.filter(row => row[labelKey] === label), stats = E.meanSD(trials.map(row => row.metric.success * 100));
    return [label, `${trials.filter(row => row.metric.success).length}/${trials.length}`,
      `${stats.mean.toFixed(1)} ± ${stats.sd === null ? "n/a" : stats.sd.toFixed(1)}%`,
      trials.reduce((n, row) => n + row.metric.wall_contacts, 0),
      trials.filter(row => row.metric.end_reason === "lateral_acceleration_limit").length,
      (trials.reduce((n, row) => n + row.metric.progress, 0) / trials.length * 100).toFixed(1) + "%"];
  });
}
const trialHeadings = ["Condition / controller", "Safe centered exits", "Mean ± sample SD across seeds", "Wall contacts", "Handling failures", "Mean progress"];
async function runPaired(conditions, model) {
  const track = currentTrack(), initial = start(), rows = [], representative = [];
  for (const condition of conditions) {
    for (const seed of [401, 402, 403]) {
      const result = E.simulate(track, cfg, model, pairedStart(initial, seed), { ...condition, seed });
      rows.push({ condition: condition.name, seed, track_id: track.id, initial: pairedStart(initial, seed), settings: condition, metric: result.metric });
      if (seed === 401) representative.push({ ...result, name: condition.name, track });
      message(`Simulated ${condition.name}, seed ${seed}.`); await tickYield();
    }
  }
  showRuns(representative);
  return { audit: audit(model), initial, track_id: track.id, conditions, seeds: [401, 402, 403], rows };
}
function renderAblations(rows = data.ablations, local = false) {
  const summaries = [];
  for (const mask of Object.keys(E.featureMasks)) {
    const selected = rows.filter(row => row.mask === mask);
    for (const split of local ? ["local"] : lab.testExposed ? ["train", "pristine_test"] : ["train"]) {
      const group = selected.filter(row => local || row.split === split), seeds = [...new Set(group.map(row => row.seed))];
      const rates = seeds.map(seed => {
        const trials = group.filter(row => row.seed === seed);
        return 100 * trials.reduce((n, row) => n + row.success, 0) / trials.length;
      });
      const stats = E.meanSD(rates);
      summaries.push([mask, split, `${stats.mean.toFixed(1)} ± ${stats.sd.toFixed(1)}%`, seeds.join(","),
        group.reduce((n, row) => n + row.wall_contacts, 0)]);
    }
  }
  table("ablationTable", ["Retrained input variant", "Evaluation", "Success mean ± sample SD", "Training seeds", "Wall contacts"], summaries);
  const c = $("ablationCanvas").getContext("2d"); c.clearRect(0, 0, 1100, 280);
  c.fillStyle = "#ecf2fa"; c.font = "bold 16px system-ui"; c.fillText(local ? "LIVE RETRAINED FEATURE EXPERIMENT / same selected start" :
    `REFERENCE RETRAINING / ${lab.testExposed ? "exposed test" : "training-track"} success across 3 seeds`, 40, 25);
  const bars = summaries.filter(row => row[1] === (local ? "local" : lab.testExposed ? "pristine_test" : "train"));
  bars.forEach((row, i) => {
    const value = parseFloat(row[2]), x = 65 + i * 260;
    c.fillStyle = colors[i % 3]; c.fillRect(x, 225 - value * 1.65, 100, value * 1.65);
    c.fillStyle = "#acbbce"; c.font = "12px system-ui"; c.fillText(row[0], x, 250); c.fillText(row[2], x, 50);
  });
}
function explainStage() {
  if (!lab.packet) return;
  const labels = [
    `1 / BATCH: ${lab.packet.batchSize} real labeled observations are presented to the current network.`,
    "2 / PREDICTION: these colored neurons are the actual forward activations before the update.",
    "3 / ERROR: steering error has weight 4, target-speed error weight 1; average over both outputs and all examples.",
    "4 / GRADIENT: backpropagation computes how each parameter changes this measured loss.",
    "5 / UPDATE: Adam applies the recorded parameter deltas using corrected gradient moments. These are the actual changes.",
  ];
  $("stepStage").textContent = labels[lab.stage];
}
function paintActivations(activations) {
  const c = $("activationCanvas").getContext("2d"); c.clearRect(0, 0, 1100, 270);
  activations.forEach((layer, l) => {
    c.fillStyle = "#ecf2fa"; c.font = "14px system-ui"; c.fillText(["18 INPUTS", "24 HIDDEN", "16 HIDDEN", "2 OUTPUTS"][l], 30 + l * 270, 25);
    layer.forEach((value, i) => {
      const intensity = Math.min(1, Math.abs(value)), x = 30 + l * 270 + (i % 6) * 34, y = 50 + Math.floor(i / 6) * 40;
      c.fillStyle = value >= 0 ? `rgba(77,226,194,${.2 + .8 * intensity})` : `rgba(255,91,101,${.2 + .8 * intensity})`;
      c.fillRect(x, y, 26, 28);
    });
  });
}
function install(id, action) { $(id).addEventListener("click", () => handle(action)); }
for (const button of document.querySelectorAll("nav button")) button.onclick = () => {
  lab.recording = false; lab.module = button.dataset.module;
  for (const panel of document.querySelectorAll("section.panel")) panel.hidden = panel.id !== lab.module;
  for (const nav of document.querySelectorAll("nav button")) nav.classList.toggle("active", nav === button);
};
$("track").onchange = fillScenarios; $("scenario").onchange = resetStart;
for (const id of ["startSpeed", "startLateral", "startHeading"]) $(id).oninput = preview;
$("camera").onchange = draw;
$("replayPlay").onclick = () => { lab.recording = false; lab.replayPlaying = !lab.replayPlaying; $("replayPlay").textContent = lab.replayPlaying ? "Pause replay" : "Play replay"; lab.last = null; };
$("replayTime").oninput = () => {
  lab.recording = false; lab.replayPlaying = false;
  const duration = Math.max(0, ...lab.runs.map(run => run.trace.at(-1).time));
  lab.replayTime = Number($("replayTime").value) / 1000 * duration; draw();
};
install("revealPrediction", () => {
  if (!$("guessDirection").value || !$("guessBrake").value) throw new Error("Make both predictions before revealing.");
  const net = selectedModel(), env = new E.Environment(currentTrack(), cfg); env.reset(start());
  const command = net.predict(env.observe().features), direction = Math.abs(command[0]) < .06 ? "straight" : command[0] > 0 ? "left" : "right";
  env.step(command);
  const brakes = env.state.brake > .01;
  $("predictionResult").hidden = false;
  $("predictionResult").textContent = `Actual model output: ${(command[0] * cfg.max_steer * 180 / Math.PI).toFixed(2)}° steering,
target ${((command[1] + 1) * cfg.max_speed / 2 * 3.6).toFixed(1)} km/h. First actuator interval: brake ${(env.state.brake * 100).toFixed(1)}%, throttle ${(env.state.throttle * 100).toFixed(1)}%.`;
  const score = Number($("guessDirection").value === direction) + Number(($("guessBrake").value === "yes") === brakes);
  $("guessFeedback").textContent = `${score}/2 matched. Predicted direction: ${direction}; controller brakes: ${brakes ? "yes" : "no"}.`;
  const result = E.simulate(currentTrack(), cfg, net, start());
  showRuns([{ ...result, name: $("model").selectedOptions[0].textContent, track: currentTrack() }]);
  message(`Simulated the actual policy: ${result.metric.end_reason}.`);
});
install("newDrive", () => {
  if (currentTrack().id >= 11) throw new Error("Record training/validation drives on the eleven training circuits only.");
  const env = new E.Environment(currentTrack(), cfg); env.reset(start());
  if (env.contact) throw new Error("Starting footprint contacts the wall. Move the car inward before recording.");
  do { lab.episode++; } while (lab.rows.some(row => row.episode === `drive-${lab.episode}`));
  lab.manual = { env, split: $("recordSplit").value, reason: "", trace: [env.record()] };
  lab.recording = false; showRuns([{ name: `Episode ${lab.episode} / ${lab.manual.split}`, track: currentTrack(), trace: lab.manual.trace }], false);
  $("episodeStatus").textContent = `Episode ${lab.episode} assigned to ${lab.manual.split}. Its split cannot change mid-drive.`;
});
install("recordStep", manualStep);
install("recordPlay", () => {
  if (!lab.manual || lab.manual.reason) throw new Error("Start a new valid episode first.");
  lab.replayPlaying = false; lab.recording = !lab.recording; lab.last = null;
  $("recordPlay").textContent = lab.recording ? "Pause recording" : "Record live drive";
});
for (const [id, delta] of [["steerLeft", 3], ["steerRight", -3]]) install(id, () => {
  $("manualSteer").value = E.clip(Number($("manualSteer").value) + delta, -28, 28);
});
install("steerCenter", () => { $("manualSteer").value = 0; });
document.addEventListener("keydown", event => {
  if (lab.module !== "dataset" || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (["a", "d", "w", "s"].includes(key)) event.preventDefault();
  if (key === "a" || key === "d") $("manualSteer").value = E.clip(Number($("manualSteer").value) + (key === "a" ? 3 : -3), -28, 28);
  if (key === "w" || key === "s") $("manualTarget").value = E.clip(Number($("manualTarget").value) + (key === "w" ? 3 : -3), 0, 100);
});
install("loadStarter", () => {
  const existing = new Set(lab.rows.map(row => row.id));
  const combined = [...lab.rows, ...data.starter.filter(row => !existing.has(row.id)).map(row => JSON.parse(JSON.stringify(row)))];
  E.validateRows(combined); lab.rows = combined;
  updateDataset(); message("Loaded real expert rollout examples and independent validation labels—not student driving.");
});
install("clearDataset", () => { lab.rows = []; lab.manual = null; lab.recording = false; updateDataset(); setExposure(); });
install("downloadDataset", () => { E.validateRows(lab.rows); download("student-driving-dataset.json", { version: E.version, config: cfg, rows: lab.rows, audit: { testExposed: lab.testExposed } }); });
install("downloadCSV", () => {
  E.validateRows(lab.rows);
  const cells = value => {
    const text = typeof value === "string" && /^[=+@-]/.test(value) ? "'" + value : String(value);
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = [["id", "episode", "split", "source", "track_id", ...data.features, "label_steering_fraction", "label_target_speed_normalized"],
    ...lab.rows.map(row => [row.id, row.episode, row.split, row.source, row.track_id, ...row.features, ...row.targets])];
  download("student-driving-dataset.csv", rows.map(row => row.map(cells).join(",")).join("\r\n"), "text/csv");
});
$("importDataset").onchange = () => handle(async () => {
  const file = $("importDataset").files[0]; if (!file) return;
  const imported = JSON.parse(await file.text());
  if (imported.version !== E.version || ["max_speed", "max_steer", "track_width", "car_width"].some(key => imported.config?.[key] !== cfg[key]))
    throw new Error("Dataset version or observation normalization does not match this lab.");
  E.validateRows(imported.rows);
  lab.rows = imported.rows; lab.manual = null; lab.recording = false; updateDataset(); message("Imported local examples; no data was uploaded.");
});
install("trainStudent", async () => {
  const { train, validation } = datasets(), seed = number("studentSeed", -2147483648, 2147483647, true);
  const rate = number("learningRate", .00001, .05), updates = number("updates", 1, 200, true);
  lab.busy = true; lab.recording = false; lab.frozen = null;
  lab.studentMetadata = { seed, mask: $("studentMask").value, learning_rate: rate, planned_fit_updates: updates };
  lab.student = new E.Network(seed, [18, 24, 16, 2], $("studentMask").value);
  lab.losses = [{ update: 0, train: lab.student.loss(train), validation: lab.student.loss(validation) }];
  lab.trainingSnapshots = [{ update: 0, model: lab.student.serialize() }];
  const result = await fitLocal(seed, $("studentMask").value, train, validation, updates, rate, (net, update) => {
    lab.student = net; lab.losses.push({ update, train: net.loss(train), validation: net.loss(validation) }); drawLoss();
    lab.trainingSnapshots.push({ update, model: net.serialize() });
    $("studentTraining").textContent = `${update}/${updates} real Adam updates; validation loss ${net.loss(validation).toFixed(6)}.`;
  });
  lab.student = result.last; $("model").value = "student";
  $("learningUpdate").max = lab.trainingSnapshots.length - 1;
  $("learningUpdate").value = Math.floor((lab.trainingSnapshots.length - 1) / 2);
  const run = E.simulate(currentTrack(), cfg, lab.student, start());
  showRuns([{ ...run, name: `Student network / ${updates} actual updates`, track: currentTrack() }]);
  message("Trained from scratch. Loss is not a safety guarantee; inspect the actual drive."); setExposure();
});
function replayLearning() {
  if (lab.trainingSnapshots.length < 2) throw new Error("Train a fresh student model first to save actual intermediate weights.");
  const snapshots = [lab.trainingSnapshots[0], lab.trainingSnapshots[Number($("learningUpdate").value)], lab.trainingSnapshots.at(-1)];
  const runs = snapshots.map(snapshot => ({ ...E.simulate(currentTrack(), cfg, E.Network.fromJSON(snapshot.model), start()),
    name: `Actual student weights / update ${snapshot.update}`, track: currentTrack() }));
  showRuns(runs);
  $("learningReplayStatus").textContent = "Same start and equal simulated time. Every trajectory comes from that recorded weight snapshot; late regression remains visible.";
}
install("animateStudentLearning", replayLearning);
$("learningUpdate").oninput = () => handle(replayLearning);
install("animateReferenceLearning", () => {
  const snapshots = [data.evolution[0], data.evolution.find(snapshot => snapshot.update === 128),
    { update: data.models.full_301.updates, model: data.models.full_301 }];
  showRuns(snapshots.map(snapshot => ({ ...E.simulate(currentTrack(), cfg, E.Network.fromJSON(snapshot.model), start()),
    name: `Reference trained weights / update ${snapshot.update}`, track: currentTrack() })));
  message("Actual saved NumPy checkpoints, simulated on your selected track—not invented improvement frames.");
});
install("exportModel", () => {
  if (!lab.student) throw new Error("Train a student model first.");
  download("student-driving-model.json", { ...lab.student.serialize(), config: cfg, training: lab.studentMetadata,
    optimizer_note: "Exported weights only; restarting optimizer requires fresh moments", audit: audit(lab.student) });
});
install("inspectStep", () => {
  const { train, validation } = datasets();
  if (!lab.student) {
    const seed = number("studentSeed", -2147483648, 2147483647, true);
    lab.student = new E.Network(seed, [18, 24, 16, 2], $("studentMask").value);
    lab.studentMetadata = { seed, mask: $("studentMask").value, planned_fit_updates: 0 };
  }
  const before = lab.student.clone(), batch = train.slice(0, 8), rate = number("learningRate", .00001, .05);
  const packet = lab.student.trainBatch(batch, rate); packet.batchSize = batch.length; lab.packet = packet; lab.stage = 0;
  lab.frozen = null;
  const selected = packet.changes[0], epsilon = 1e-5;
  const parameter = selected.i === undefined ? before.biases[selected.layer] : before.weights[selected.layer][selected.i];
  const original = parameter[selected.j];
  parameter[selected.j] = original + epsilon; const plus = before.loss(batch);
  parameter[selected.j] = original - epsilon; const minus = before.loss(batch);
  parameter[selected.j] = original;
  packet.numericGradient = (plus - minus) / (2 * epsilon);
  packet.beforeModel = before.serialize(); packet.afterModel = lab.student.serialize();
  $("gradientCheck").textContent = `${selected.parameter}: backprop ${selected.gradient.toExponential(6)}, finite-difference ${packet.numericGradient.toExponential(6)}, absolute difference ${Math.abs(selected.gradient - packet.numericGradient).toExponential(3)}.`;
  $("stepLoss").textContent = `Minibatch loss ${packet.loss.toFixed(6)} → ${packet.lossAfter.toFixed(6)}; actual update counter ${packet.updates}.`;
  table("batchTable", ["Example / source", "Offset / speed inputs", "Labels", "Predictions", "Errors", "Weighted mean squared error"],
    batch.map((row, i) => {
      const prediction = packet.predictions[i], error = prediction.map((v, j) => v - row.targets[j]);
      return [`${row.id} / ${row.source}`, `${row.features[0].toFixed(4)} / ${row.features[3].toFixed(4)}`,
        row.targets.map(v => v.toFixed(4)).join(" / "), prediction.map(v => v.toFixed(4)).join(" / "),
        error.map(v => v.toFixed(4)).join(" / "), ((4 * error[0] ** 2 + error[1] ** 2) / 2).toFixed(7)];
    }));
  table("weightTable", ["Parameter", "Before", "Gradient", "Actual Adam delta", "After"], packet.changes.map(p =>
    [p.parameter, p.before.toFixed(7), p.gradient.toExponential(4), p.delta.toExponential(4), p.after.toFixed(7)]));
  paintActivations(packet.activations); explainStage();
  lab.losses.push({ update: lab.student.updates, train: lab.student.loss(train), validation: lab.student.loss(validation) }); drawLoss();
  message("A real update was applied. The finite-difference check independently verifies its gradient.");
});
install("stepNext", () => { if (!lab.packet) throw new Error("Inspect a minibatch first."); lab.stage = Math.min(4, lab.stage + 1); explainStage(); });
install("driveAfterStep", () => {
  if (!lab.packet) throw new Error("Inspect a minibatch first.");
  showRuns([["Before real update", lab.packet.beforeModel], ["After real update", lab.packet.afterModel]].map(([name, model]) =>
    ({ ...E.simulate(currentTrack(), cfg, E.Network.fromJSON(model), start()), name, track: currentTrack() })));
});
install("compareControllers", async () => {
  lab.busy = true; const track = currentTrack(), initial = start(), model = selectedModel().clone(), rows = [], runs = [];
  for (const [name, policy] of [["Rules", "rules"], ["Expert", "expert"], ["Neural", model]]) {
    for (const seed of [401, 402, 403]) {
      const result = E.simulate(track, cfg, policy, pairedStart(initial, seed), { seed });
      rows.push({ controller: name, seed, initial: pairedStart(initial, seed), metric: result.metric });
      if (seed === 401) runs.push({ ...result, name, track });
      await tickYield();
    }
  }
  lab.comparisons = { audit: audit(model), track_id: track.id, initial, seeds: [401, 402, 403], rows };
  table("controllerTable", trialHeadings, summarizeTrials(rows, "controller")); showRuns(runs);
  message("Matched starts and equal simulation time; failed cars freeze at their real endpoint.");
});
install("downloadComparison", () => { if (!lab.comparisons) throw new Error("Run a comparison first."); download("controller-comparison.json", lab.comparisons); });
install("localAblation", async () => {
  const { train, validation } = datasets(), updates = number("updates", 1, 200, true), rate = number("learningRate", .00001, .05);
  lab.busy = true; const rows = [], modelAudit = [], track = currentTrack(), initial = start();
  for (const mask of Object.keys(E.featureMasks)) {
    for (const seed of [301, 302, 303]) {
      const result = await fitLocal(seed, mask, train, validation, updates, rate);
      const run = E.simulate(track, cfg, result.best, pairedStart(initial, seed), { seed });
      rows.push({ mask, seed, split: "local", ...run.metric });
      modelAudit.push({ mask, seed, actual_updates: updates, selected_update: result.best.updates, weights: result.best.serialize() });
      $("ablationStatus").textContent = `Really retrained ${mask}, seed ${seed}: ${updates} updates, validation-selected ${result.best.updates}.`;
      await tickYield();
    }
  }
  lab.localAblations = { rows, models: modelAudit, track_id: track.id, initial,
    audit: { engine: E.version, testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed,
      model_protocol: "12 fresh independent models, selected only by action validation" } };
  renderAblations(rows, true); message("All twelve models were independently retrained, never inference-only masked.");
});
install("freezeReveal", async () => {
  const chosen = selectedModel();
  if ($("model").value === "student" && chosen.updates === 0) throw new Error("Train your student model before freezing.");
  lab.busy = true; const alreadyExposed = lab.testExposed;
  lab.frozen = chosen.clone(); const fingerprint = E.fingerprint(lab.frozen), rows = [];
  for (const track of tracks.filter(t => t.id >= 13)) {
    for (const scenario of data.scenarios[track.id]) {
      const result = E.simulate(track, cfg, lab.frozen, scenario);
      rows.push([track.name, scenario.name, result.metric.success ? "SAFE CENTERED EXIT" : result.metric.end_reason,
        `${(result.metric.progress * 100).toFixed(1)}%`, result.metric.min_clearance_m.toFixed(3)]);
      await tickYield();
    }
  }
  if (E.fingerprint(lab.frozen) !== fingerprint) throw new Error("Frozen weights changed during testing");
  lab.testExposed = true; fillTracks(); setExposure(); referenceTables(); renderAblations();
  $("freezeStatus").textContent = `Frozen weights fingerprint ${fingerprint}. Twelve test sections evaluated without weight updates.
${alreadyExposed ? "The test set was already exposed: these are exploratory results." : "This is the first UI reveal in this classroom session."}`;
  table("testTable", ["New test circuit", "Section", "Outcome", "Progress", "Minimum wall margin (m)"], rows);
  $("demonstrateLeak").disabled = false; message("Test circuits exposed only after the model was frozen.");
});
install("demonstrateLeak", async () => {
  if (!lab.frozen || !lab.testExposed) throw new Error("Freeze and reveal tests first.");
  const { train } = datasets(); E.validateRows(data.testRows, true);
  lab.busy = true; const contaminated = lab.frozen.clone(), before = contaminated.loss(data.testRows), rng = new E.RNG(514);
  for (let update = 0; update < 40; update++) {
    const batch = [...Array.from({ length: 32 }, () => train[rng.index(train.length)]),
      ...Array.from({ length: 32 }, () => data.testRows[rng.index(data.testRows.length)])];
    contaminated.trainBatch(batch, .0005);
    if (update % 4 === 0) await tickYield();
  }
  const after = contaminated.loss(data.testRows); lab.testLabelsExposed = true; setExposure();
  table("leakTable", ["Model / evaluation", "Measured error", "Valid new-test generalization?"], [
    ["Frozen clean model", before.toFixed(6), "Only before test exposure"],
    ["Clone trained on these same test labels", after.toFixed(6), "NO — contaminated"],
    ["Python reference clean test", data.leakage.clean_test_mse.toFixed(6), "Clean, frozen before test"],
    ["Python reference contaminated test", data.leakage.contaminated_test_mse.toFixed(6), "NO — contaminated"],
  ]);
  lab.leakage = { before, after, trained_test_labels: true, valid_generalization: false, model: contaminated.serialize() };
  $("leakStatus").textContent = "INVALID BENCHMARK: the separate clone saw the very labels used for its test score. Your clean model was not overwritten.";
  message("Actual test-label training completed. A lower score, if observed, is not honest unseen performance.");
});
install("runRobustness", async () => {
  const noise = number("noise", 0, 3), delay = number("delay", 0, 20, true), grip = number("grip", .2, 1.2);
  lab.busy = true; lab.robustness = await runPaired([{ name: "Clean", noise: 0, delay: 0, grip: 1 },
    { name: "Selected distribution shift", noise, delay, grip }], selectedModel().clone());
  table("robustnessTable", trialHeadings, summarizeTrials(lab.robustness.rows, "condition"));
  message("Observations and braking/handling envelope genuinely changed; no safety controller rescued the policy.");
});
install("downloadRobustness", () => { if (!lab.robustness) throw new Error("Run a robustness test first."); download("robustness-experiment.json", lab.robustness); });
const worksheets = [
  ["Predict steering", "Change speed only. Predict direction and braking before revealing. Explain whether a correct guess proves understanding."],
  ["Label quality", "Collect two separate drives. Compare your commands with expert examples. What would wrong steering labels teach?"],
  ["Gradient mechanics", "Inspect one minibatch. Explain the sign of a gradient, the Adam delta and the finite-difference check."],
  ["Rules and learning", "Keep starts fixed. Compare rule, expert and neural policies. Separate failures from successful full sections."],
  ["Feature information", "Retrain matched input variants. Discuss correlated information that remains after removing a channel."],
  ["Generalization integrity", "Freeze before revealing new tracks. Deliberately leak labels into a separate clone and explain why the score is invalid."],
  ["Robustness", "Change one noise, delay or grip factor. Pair the same three seeds across values and interpret variability."],
  ["Safety tradeoffs", "Measure successful exits, wall contacts and minimum margins separately. Can completion improve while wall clearance shrinks?"],
];
for (const [i, worksheet] of worksheets.entries()) { const option = document.createElement("option"); option.value = i; option.textContent = `${i + 1}. ${worksheet[0]}`; $("worksheet").append(option); }
$("worksheet").onchange = () => { $("worksheetPrompt").textContent = worksheets[Number($("worksheet").value)][1]; };
$("worksheet").onchange();
$("factor").onchange = () => { $("factorValues").value = $("factor").value === "noise" ? "0,0.5,1" : $("factor").value === "delay" ? "0,3,6" : "1,0.8,0.65"; };
install("runWorksheet", async () => {
  if (!$("hypothesis").value.trim()) throw new Error("Write your hypothesis before running the experiment.");
  const hypothesis = $("hypothesis").value, worksheet = worksheets[Number($("worksheet").value)][0];
  const factor = $("factor").value, values = $("factorValues").value.split(",").map(v => Number(v.trim()));
  if (values.length !== 3 || values.some(v => !Number.isFinite(v)) || new Set(values).size !== 3) throw new Error("Provide three distinct numeric values.");
  const maximum = factor === "noise" ? 3 : factor === "delay" ? 20 : 1.2, minimum = factor === "grip" ? .2 : 0;
  if (values.some(v => v < minimum || v > maximum || (factor === "delay" && !Number.isInteger(v)))) throw new Error("Factor values are outside the simulator bounds.");
  lab.busy = true; const conditions = values.map(value => ({ name: `${factor}=${value}`, noise: 0, delay: 0, grip: 1, [factor]: value }));
  lab.worksheet = await runPaired(conditions, selectedModel().clone());
  lab.worksheet.factor = factor; lab.worksheet.values = values; lab.worksheet.hypothesis = hypothesis; lab.worksheet.worksheet = worksheet;
  table("worksheetResults", trialHeadings, summarizeTrials(lab.worksheet.rows, "condition"));
  message("One factor changed; starts, policy and paired seed schedule stayed fixed. Sample SD is not a confidence interval.");
});
install("exportWorksheet", () => {
  if (!lab.worksheet) throw new Error("Run the worksheet experiment first.");
  download("ml-driving-worksheet.json", { ...lab.worksheet, quiz_attempts: lab.quizAttempts,
    evidence: $("evidence").value, conclusion: $("conclusion").value });
});
install("printWorksheet", () => window.print());
const questions = [
  { text: "What changes during supervised training?", choices: ["The road geometry", "The model weights", "Only the animation"], answer: 1 },
  { text: "Which gives an honest new-test evaluation?", choices: ["Tune until the test score looks best", "Freeze weights before revealing test outcomes", "Train on the test labels"], answer: 1 },
  { text: "What does the neural driver output?", choices: ["Steering and target speed", "All pedals and gears directly", "A wall collision reward"], answer: 0 },
  { text: "What does lower training loss guarantee?", choices: ["No crashes", "Real-world safety", "Neither; test actual driving"], answer: 2 },
  { text: "Why split whole driving episodes?", choices: ["Neighboring frames can leak near-identical states", "Validation must be empty", "To make the plots smoother"], answer: 0 },
];
for (const [i, question] of questions.entries()) {
  const p = document.createElement("p"); p.textContent = question.text; $("quiz").append(p);
  question.choices.forEach((choice, j) => {
    const label = document.createElement("label"), input = document.createElement("input"); input.type = "radio"; input.name = `quiz${i}`; input.value = j;
    label.append(input, document.createTextNode(" " + choice)); $("quiz").append(label);
  });
}
install("gradeQuiz", () => {
  const score = questions.reduce((n, q, i) => n + Number(document.querySelector(`input[name="quiz${i}"]:checked`)?.value === String(q.answer)), 0);
  lab.quizAttempts.push({ phase: $("quizPhase").value, score, total: questions.length });
  $("quizFeedback").textContent = `${score}/${questions.length}. Key ideas: weights learn from labels; freeze before tests; output steering/target speed; low loss is not a safety guarantee; avoid adjacent-frame leakage.`;
});
function referenceTables() {
  table("referenceControllers", ["Controller", "All section successes", "Wall contacts", "Pristine new-track successes"],
    ["rules", "expert", "pretrained"].map(controller => {
      const rows = data.comparisons.filter(row => row.controller === controller && (lab.testExposed || row.track_id < 13));
      const tests = rows.filter(row => row.split === "pristine_test");
      return [controller, `${rows.reduce((n, r) => n + r.success, 0)}/${rows.length}`, rows.reduce((n, r) => n + r.wall_contacts, 0),
        lab.testExposed ? `${tests.reduce((n, r) => n + r.success, 0)}/${tests.length}` : "Sealed"];
    }));
  table("referenceRobustness", ["Condition", "Safe section successes", "Wall contacts", "Handling failures"],
    data.conditions.map(condition => {
      const rows = data.robustness.filter(row => row.condition === condition.name && (lab.testExposed || row.track_id < 13));
      return [condition.name, `${rows.reduce((n, r) => n + r.success, 0)}/${rows.length}`, rows.reduce((n, r) => n + r.wall_contacts, 0),
        rows.filter(row => row.end_reason === "lateral_acceleration_limit").length];
    }));
  table("ablationHistories", ["Mask / seed", "Actual training updates", "Selected validation update", "Initial / selected validation loss"],
    Object.entries(data.histories).map(([key, rows]) => [key, 256, data.models[key].updates,
      `${rows[0].validation_mse.toFixed(5)} / ${Math.min(...rows.map(row => row.validation_mse)).toFixed(5)}`]));
}
function animate(now) {
  if (lab.last !== null) {
    const delta = Math.min((now - lab.last) / 1000, .25);
    if (lab.recording && !lab.busy) {
      lab.manual.accumulator = (lab.manual.accumulator || 0) + delta;
      while (lab.manual.accumulator >= cfg.dt && lab.recording) {
        lab.manual.accumulator -= cfg.dt;
        try { manualStep(); } catch (error) { lab.recording = false; $("error").textContent = error.message; console.error(error); }
      }
    } else if (lab.replayPlaying) {
      const duration = Math.max(0, ...lab.runs.map(run => run.trace.at(-1).time));
      lab.replayTime = Math.min(duration, lab.replayTime + delta * 4); draw();
      if (lab.replayTime >= duration) { lab.replayPlaying = false; $("replayPlay").textContent = "Play replay"; }
    }
  }
  lab.last = now; requestAnimationFrame(animate);
}
restoreExposure(); fillTracks(); fillScenarios(); updateDataset(); renderAblations(); referenceTables(); requestAnimationFrame(animate);
