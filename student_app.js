"use strict";
const E = window.StudentEngine, T = window.StudentTools, data = JSON.parse(document.getElementById("experiment-data").textContent);
const $ = id => document.getElementById(id), cfg = data.config, gen = data.dataGeneration, kit = data.teaching;
const tracks = data.tracks.map(track => new E.Track(track));
const trackMap = new Map(tracks.map(t => [t.id, t])), ctx = { cfg, tracks: trackMap };
const legacyFolded = new Set(gen.legacy_geometry_tracks), validIds = gen.valid_training_tracks;
const trainingIds = Array.from({ length: 11 }, (_, i) => i), holdoutIds = gen.legacy_holdout_tracks;
const monitorSections = T.sectionsFor(data, gen.monitor_tracks);
const reference = E.Network.fromJSON(data.pretrained);
const lab = {
  rows: [], episodeStarts: {}, student: null, studentMetadata: null, losses: [], probe: [], trainingSnapshots: [], busy: false, cancel: false, module: "predict",
  episode: 0, manual: null, recording: false, packet: null, stage: 0, runs: [], replayTime: 0, replayPlaying: false, last: null,
  testExposed: false, testLabelsExposed: false, frozen: null, exposureStorage: true, quizAttempts: [], comparisons: null, robustness: null,
  worksheet: null, localAblations: null, baselines: { ridge: null, knn: null }, playground: null, coverage: null, dagger: null, diagnosis: null,
  importance: null, fresh: null, includeLegacy: false, log: { modules: new Set(), counts: {} },
};
window.studentLab = lab;
const colors = ["#ffc56b", "#76a9ff", "#4de2c2"];
const palette = { train: "#4de2c2", validation: "#ffc56b", stress: "#ff8990", nominal: "#76a9ff", recovery: "#be9cff", expert: "#ecf2fa" };
const tickYield = () => document.hidden ? new Promise(resolve => setTimeout(resolve, 0)) : new Promise(resolve => requestAnimationFrame(resolve));
const pct = v => `${(100 * v).toFixed(0)}%`, signed = v => `${v >= 0 ? "+" : "−"}${Math.abs(100 * v).toFixed(0)} pts`;
const fmtP = p => p < 0.001 ? p.toExponential(1) : p.toFixed(3);
const count = key => { lab.log.counts[key] = (lab.log.counts[key] ?? 0) + 1; };
function number(id, minimum, maximum, integer = false) {
  const value = Number($(id).value);
  if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) {
    const label = document.querySelector(`label[for="${id}"]`)?.textContent.trim() ?? id;
    throw new Error(`${label}: choose ${integer ? "an integer" : "a number"} between ${minimum} and ${maximum}`);
  }
  return value;
}
function message(text) { $("status").textContent = text; }
async function handle(action, long = false) {
  if (lab.busy) { $("error").textContent = "Wait for the current experiment to finish."; return; }
  $("error").textContent = ""; lab.cancel = false;
  if (long) { lab.busy = true; $("cancelRun").hidden = false; }
  try { await action(); }
  catch (error) {
    lab.recording = false;
    $("error").textContent = error.message;
    console.error(error);
  } finally { lab.busy = false; $("cancelRun").hidden = true; }
}
function install(id, action, long = false) { $(id).addEventListener("click", () => handle(action, long)); }
$("cancelRun").onclick = () => { lab.cancel = true; message("Cancelling after the current step..."); };
function selectedModel() {
  const choice = $("model").value;
  if (choice === "student") { if (!lab.student) throw new Error("Train or initialize a student model first."); return lab.student; }
  if (choice === "ridge" || choice === "knn") {
    if (!lab.baselines[choice]) throw new Error("Train that baseline in module 6 first.");
    return lab.baselines[choice];
  }
  return reference;
}
function currentTrack() { return trackMap.get(Number($("track").value)); }
function canonical() { return data.scenarios[currentTrack().id][Number($("scenario").value)]; }
function start() {
  return { ...canonical(), lateral: Number($("startLateral").value),
    heading_error: Number($("startHeading").value) * Math.PI / 180, speed: Number($("startSpeed").value) / 3.6 };
}
function fillTracks() {
  const selected = $("track").value;
  $("track").replaceChildren();
  for (const track of tracks.filter(t => lab.testExposed || t.id < 13)) {
    const option = document.createElement("option");
    option.value = track.id;
    option.textContent = `${track.name}${track.id >= 13 ? " / SEALED TEST (exposed)" : track.id >= 11 ? " / legacy holdout" : legacyFolded.has(track.id) ? " / legacy geometry" : ""}`;
    $("track").append(option);
  }
  if ([...$("track").options].some(option => option.value === selected)) $("track").value = selected;
}
function legacyNote() {
  const track = currentTrack(), info = data.tracks.find(t => t.id === track.id);
  $("legacyNote").textContent = legacyFolded.has(track.id) ? `${track.name} is a legacy circuit: its tightest bend (radius ${info.geometry.min_radius_m.toFixed(1)} m) is narrower than half the road, so the inner wall folds into a small loop.
It stays for reproducibility but is excluded from the monitor and headline pools.` : "";
}
function fillScenarios() {
  $("scenario").replaceChildren();
  data.scenarios[currentTrack().id].forEach((scenario, i) => {
    const option = document.createElement("option"); option.value = i; option.textContent = scenario.name; $("scenario").append(option);
  });
  $("scenario").value = "2";
  legacyNote(); resetStart();
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
  if (run.marker && run.trace[run.marker.index]) {
    const mark = run.trace[run.marker.index], p = project([mark.x, mark.y]);
    c.strokeStyle = "#ffc56b"; c.lineWidth = 3; c.beginPath(); c.arc(p[0], p[1], 11, 0, 2 * Math.PI); c.stroke();
    c.fillStyle = "#ffc56b"; c.font = "bold 13px system-ui"; c.fillText(run.marker.label, p[0] + 15, p[1] - 9);
  }
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
  diagnoseCursor?.();
}
const diagnoseCursor = () => window.drawDiagnosisCursor?.();

// ---- generic line plot ------------------------------------------------------------------------------------------------------
function niceTicks(low, high, count = 4) { return Array.from({ length: count + 1 }, (_, i) => low + (high - low) * i / count); }
function plot(id, spec) {
  const cv = $(id), c = cv.getContext("2d"), W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  const m = { l: 66, r: 18, t: 26, b: 36 }, pw = W - m.l - m.r, ph = H - m.t - m.b, log = !!spec.log;
  const tr = v => log ? Math.log10(Math.max(v, 1e-12)) : v, [x0, x1] = spec.x, [y0, y1] = spec.y;
  const mx = v => m.l + (v - x0) / ((x1 - x0) || 1) * pw, my = v => m.t + (1 - (tr(v) - tr(y0)) / ((tr(y1) - tr(y0)) || 1)) * ph;
  c.font = "12px system-ui"; c.lineWidth = 1;
  const yTicks = spec.yTicks ?? (log ? Array.from({ length: Math.max(2, Math.ceil(Math.log10(y1)) - Math.floor(Math.log10(y0)) + 1) }, (_, i) => 10 ** (Math.floor(Math.log10(y0)) + i)).filter(v => v >= y0 * .999 && v <= y1 * 1.001) : niceTicks(y0, y1));
  for (const v of yTicks) {
    c.strokeStyle = "#2a3a50"; c.beginPath(); c.moveTo(m.l, my(v)); c.lineTo(W - m.r, my(v)); c.stroke();
    c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(spec.format ? spec.format(v) : v < .01 && v > 0 ? v.toExponential(0) : String(+v.toFixed(3)), m.l - 6, my(v) + 4);
  }
  c.textAlign = "center";
  for (const v of niceTicks(x0, x1, 5)) { c.fillStyle = "#acbbce"; c.fillText(String(Math.round(v * 10) / 10), mx(v), H - 14); }
  c.textAlign = "left"; c.fillStyle = "#ecf2fa"; c.font = "bold 13px system-ui"; c.fillText(spec.title ?? "", m.l, 16);
  c.font = "12px system-ui"; c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(spec.xLabel ?? "", W - m.r, H - 2); c.textAlign = "left";
  for (const band of spec.bands ?? []) {
    c.globalAlpha = .18; c.fillStyle = band.color; c.beginPath();
    band.points.forEach(([x, lo], i) => i ? c.lineTo(mx(x), my(lo)) : c.moveTo(mx(x), my(lo)));
    [...band.points].reverse().forEach(([x, , hi]) => c.lineTo(mx(x), my(hi))); c.closePath(); c.fill(); c.globalAlpha = 1;
  }
  for (const s of spec.series ?? []) {
    c.strokeStyle = s.color; c.lineWidth = s.width ?? 2; c.setLineDash(s.dash ?? []); c.beginPath();
    s.points.forEach(([x, y], i) => i ? c.lineTo(mx(x), my(y)) : c.moveTo(mx(x), my(y))); c.stroke(); c.setLineDash([]);
    if (s.markers) { c.fillStyle = s.color; for (const [x, y] of s.points) { c.beginPath(); c.arc(mx(x), my(y), 3.5, 0, 2 * Math.PI); c.fill(); } }
  }
  return { mx, my, m, pw, ph };
}
function drawLoss() {
  const points = lab.losses;
  if (!points.length) { plot("lossCanvas", { x: [0, 1], y: [1e-3, 1], log: true, title: "Loss (log scale): train / validation / stress", xLabel: "optimizer updates" }); return; }
  const last = Math.max(1, points.at(-1).update), values = points.flatMap(p => [p.train, p.validation, p.stress]).filter(v => v !== null && v > 0);
  const low = 10 ** Math.floor(Math.log10(Math.min(...values))), high = 10 ** Math.ceil(Math.log10(Math.max(...values)));
  plot("lossCanvas", { x: [0, last], y: [low, high], log: true, title: "ACTUAL LOSS (log scale): weighted action error per state", xLabel: "optimizer updates",
    series: ["train", "validation", "stress"].map(key => ({ color: palette[key], points: points.filter(p => p[key] !== null).map(p => [p.update, p[key]]) })) });
}
function drawProbe() {
  const points = lab.probe, last = Math.max(1, ...lab.losses.map(p => p.update), ...points.map(p => p.update));
  const n = points[0] ? Math.max(points[0].nominal.n, points[0].recovery.n) : 8;
  plot("probeCanvas", { x: [0, last], y: [0, n], yTicks: [0, n / 2, n], title: "CLOSED-LOOP: fixed development sections completed safely (exact counts, no interval)", xLabel: "optimizer updates",
    format: v => String(v), series: ["nominal", "recovery"].map(key => ({ color: palette[key], markers: true, points: points.map(p => [p.update, p[key].k]) })) });
}

// ---- datasets --------------------------------------------------------------------------------------------------------------
function datasets() {
  E.validateRows(lab.rows);
  T.validateEpisodeStarts(lab.episodeStarts);
  const train = lab.rows.filter(row => row.split === "train"), validation = lab.rows.filter(row => row.split === "validation");
  if (!train.length || !validation.length) throw new Error("Capture separate training and validation drives, or load the starter expert episodes.");
  return { train, validation };
}
function updateDataset() {
  const episodes = new Set(lab.rows.map(r => r.episode)).size;
  $("datasetCounts").textContent = `${lab.rows.filter(r => r.split === "train").length} training / ${lab.rows.filter(r => r.split === "validation").length} validation / ${lab.rows.length} total examples in ${episodes} episodes
${data.stressRows.length} stress examples (evaluation only, always available)`;
  table("datasetTable", ["ID / episode", "Split / source", "Offset / speed inputs", "Steering / target labels"],
    lab.rows.slice(-30).map(row => [row.id, `${row.split} / ${row.source}`, `${row.features[0].toFixed(3)} / ${row.features[3].toFixed(3)}`,
      row.targets.map(v => v.toFixed(3)).join(" / ")]));
}
function download(name, value, type = "application/json") {
  const url = URL.createObjectURL(new Blob([type === "application/json" ? JSON.stringify(value, null, 2) : value], { type }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function audit(model = selectedModel()) { return { engine: E.version, testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed, model: E.fingerprint(model) }; }
const EXPOSURE_KEY = "f1-student-lab-exposure-v1";
// Exposure of the sealed circuits can only grow. Another tab may have revealed (or exposed the labels of) the test set since this tab loaded, so what is
// stored is merged into this tab before it is read or written: a stale tab can neither un-expose the circuits nor mislabel a later freeze as the first reveal.
function mergeStoredExposure() {
  if (!lab.exposureStorage) return false;
  let stored;
  try { const raw = localStorage.getItem(EXPOSURE_KEY); stored = raw === null ? null : JSON.parse(raw); }
  catch (error) { if (!["SecurityError", "SyntaxError"].includes(error.name)) throw error; return false; }
  if (!stored || typeof stored.testExposed !== "boolean" || typeof stored.testLabelsExposed !== "boolean") return false;
  const before = [lab.testExposed, lab.testLabelsExposed];
  lab.testExposed = lab.testExposed || stored.testExposed || stored.testLabelsExposed; lab.testLabelsExposed = lab.testLabelsExposed || stored.testLabelsExposed;
  return lab.testExposed !== before[0] || lab.testLabelsExposed !== before[1];
}
function setExposure() {
  mergeStoredExposure();
  if (lab.exposureStorage) {
    try {
      localStorage.setItem(EXPOSURE_KEY, JSON.stringify({ testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed }));
    } catch (error) {
      if (!["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
      lab.exposureStorage = false;
      $("error").textContent = "Persistent exposure storage is unavailable. Do not claim a pristine test session after reopening this page.";
    }
  }
  $("testExposure").textContent = lab.testExposed ?
    `Test circuits now exposed. Further retraining/testing is exploratory${lab.testLabelsExposed ? "; test labels were exposed in the leakage lesson" : ""}.` :
    "Three new test circuits are sealed by classroom protocol (not secrecy). Legacy holdouts are not pristine.";
}
function restoreExposure() {
  try {
    const raw = localStorage.getItem(EXPOSURE_KEY);
    const stored = raw === null ? { testExposed: false, testLabelsExposed: false } : JSON.parse(raw);
    if (typeof stored.testExposed !== "boolean" || typeof stored.testLabelsExposed !== "boolean")
      throw new SyntaxError("Invalid exposure record");
    lab.testExposed = stored.testExposed || stored.testLabelsExposed; lab.testLabelsExposed = stored.testLabelsExposed;
    localStorage.setItem(EXPOSURE_KEY, JSON.stringify(stored));
  } catch (error) {
    if (!["SecurityError", "QuotaExceededError", "SyntaxError"].includes(error.name)) throw error;
    lab.exposureStorage = false; lab.testExposed = true; lab.testLabelsExposed = true;
    $("error").textContent = "Prior exposure cannot be verified: this session is conservatively treated as already exposed, not pristine.";
  }
  setExposure();
}
// Another tab changed the stored record: follow it (the track list and reference tables depend on it). Clearing storage elsewhere un-exposes nothing here.
window.addEventListener("storage", event => {
  if (event.key !== null && event.key !== EXPOSURE_KEY) return;
  const changed = mergeStoredExposure();
  setExposure();
  if (changed) { fillTracks(); referenceTables(); }
});

// ---- manual recording ---------------------------------------------------------------------------------------------------------
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

// ---- statistics tables ------------------------------------------------------------------------------------------------------------
function poolSections(key) {
  if (key === "monitor") return { sections: monitorSections, random: false, label: "16 fixed monitor sections (tracks " + gen.monitor_tracks.join(", ") + ")" };
  if (key === "all") return { sections: T.sectionsFor(data, validIds), random: false, label: `${validIds.length * 4} fixed sections on the valid training tracks` };
  if (key === "selected") {
    const scenario = start();
    return { sections: [{ track_id: currentTrack().id, index: Number($("scenario").value), kind: /recovery/i.test(canonical().name) ? "recovery" : "nominal", scenario }],
      random: false, label: "the selected start only (perturbations of one section)" };
  }
  const spec = key === "stress" ? gen.starts.stress : gen.starts.typical;
  return { sections: T.randomSections(new E.RNG(key === "stress" ? 4321 : 1234), validIds, 24, spec, key === "stress" ? "Random stress start" : "Random typical start"),
    random: true, label: `24 independent random ${key} starts` };
}
async function runExperiment({ conditions, model, poolKey, seeds, label = "experiment" }) {
  const pool = poolSections(poolKey), seedList = Array.from({ length: seeds }, (_, i) => 401 + i);
  const rows = await T.runConditions({ conditions, sections: pool.sections, seeds: seedList, model, ctx, yieldFn: tickYield, shouldCancel: () => lab.cancel,
    onProgress: (done, total) => message(`Simulated section ${done}/${total} for every condition and seed...`) });
  if (lab.cancel) throw new Error("Cancelled: no result was recorded.");
  const independent = pool.random && seeds === 1;
  return { audit: audit(model), label, pool: pool.label, poolKey, seeds: seedList, conditions: conditions.map(T.describeCondition), rows, baseline: conditions[0].name, independent,
    sections: pool.sections.map(s => ({ track_id: s.track_id, name: s.scenario.name, kind: s.kind })),
    summary: T.summarizeConditions(rows, new E.RNG(97), { independentStarts: independent }),
    exploratory: pool.sections.length < 8 || seeds < 10 };
}
function conditionRows(result) {
  return result.summary.map(s => [s.condition, `${s.k}/${s.trials}`, s.sections,
    s.sections > 1 ? `${pct(s.rate)} [${pct(s.lo)}, ${pct(s.hi)}]` : `${pct(s.rate)} (one section: perturbations only, no interval)`,
    s.wilson ? `${pct(s.wilson.lo)} to ${pct(s.wilson.hi)}` : "not valid: trials are clustered by section",
    s.contacts, s.handling, s.invalid, pct(s.progress), s.versus
      ? `${signed(s.versus.mean)} [${signed(s.versus.lo)}, ${signed(s.versus.hi)}]; sections better/worse/equal ${s.versus.wins}/${s.versus.losses}/${s.versus.ties}; sign test p=${fmtP(s.versus.p)}`
      : "baseline"]);
}
const conditionHeadings = result => ["Condition / controller", "Safe / trials", "Sections", "Section-level success [95% CI over sections]", "Wilson interval", "Wall contacts",
  "Handling failures", "Invalid starts", "Mean progress", `Paired by section vs ${result.baseline}`];
function renderConditions(id, sectionsId, result) {
  table(id, conditionHeadings(result), conditionRows(result));
  if (!sectionsId) return;
  const names = result.conditions.map(c => c.name), bySection = new Map();
  for (const row of result.rows) {
    const key = row.section, entry = bySection.get(key) ?? { track: row.track_id, name: result.sections[key].name, kind: row.kind, counts: Object.fromEntries(names.map(n => [n, 0])), n: 0 };
    if (row.condition === names[0]) entry.n++;
    entry.counts[row.condition] += row.metric.success; bySection.set(key, entry);
  }
  table(sectionsId, ["Track", "Section", "Kind", ...names.map(n => `${n}: safe of ${result.seeds.length}`)],
    [...bySection.values()].map(e => [tracks.find(t => t.id === e.track).name, e.name, e.kind, ...names.map(n => `${e.counts[n]}/${e.n}`)]));
}
function representativeRuns(result, model) {
  const section = result.sections.length ? poolSections(result.poolKey).sections[0] : null;
  return result.conditions.slice(0, 3).map(condition => {
    const policy = condition.policy === "model" ? model : condition.policy ?? model;
    const run = E.simulate(trackMap.get(section.track_id), cfg, policy, T.pairedStart(section.scenario, result.seeds[0] * 1009),
      { noise: condition.noise ?? 0, delay: condition.delay ?? 0, grip: condition.grip ?? 1, seed: result.seeds[0] });
    return { ...run, name: condition.name, track: trackMap.get(section.track_id) };
  });
}
const groupRates = rows => {
  const groups = new Map();
  for (const r of rows) groups.set(`${r.track_id}|${r.scenario_index}`, [...(groups.get(`${r.track_id}|${r.scenario_index}`) ?? []), r.success]);
  return [...groups.values()].map(v => v.reduce((a, b) => a + b, 0) / v.length);
};
const interval = (values, rng) => { const b = T.bootstrapMean(values, rng); return `${pct(b.mean)} [${pct(b.lo)}, ${pct(b.hi)}]`; };
const referenceTracks = () => id => lab.includeLegacy || !legacyFolded.has(id);
function sealedBreakdown(rows) {
  return [13, 14, 15].map(id => {
    const own = rows.filter(r => r.track_id === id), part = kind => own.filter(r => r.kind === kind);
    const k = list => `${list.reduce((n, r) => n + r.success, 0)}/${list.length}`;
    return `track ${id}: nominal ${k(part("nominal"))}, recovery ${k(part("recovery"))}`;
  }).join("; ");
}
function referenceTables() {
  const include = referenceTracks(), rng = new E.RNG(5);
  table("referenceControllers", ["Controller", "Nominal sections: success [95% CI over sections]", "Recovery sections", "Wall contacts (all trials)", "Sealed circuits (trials per kind)"],
    [["rules", "Rules"], ["expert", "Expert"], ["pretrained", "Pretrained wall-aware"]].map(([key, name]) => {
      const rows = data.comparisons.filter(r => r.controller === key && r.split === "train" && include(r.track_id));
      return [name, `${interval(groupRates(rows.filter(r => r.kind === "nominal")), rng)} (${groupRates(rows.filter(r => r.kind === "nominal")).length} sections)`,
        `${interval(groupRates(rows.filter(r => r.kind === "recovery")), rng)} (${groupRates(rows.filter(r => r.kind === "recovery")).length} sections)`,
        rows.reduce((n, r) => n + r.wall_contacts, 0), lab.testExposed ? sealedBreakdown(data.comparisons.filter(r => r.controller === key && r.split === "pristine_test")) : "Sealed"];
    }));
  table("referenceRobustness", ["Condition", "Nominal sections: success [95% CI over sections]", "Recovery sections", "Handling failures", "Wall contacts"],
    data.conditions.map(condition => {
      const rows = data.robustness.filter(r => r.condition === condition.name && r.split === "train" && include(r.track_id));
      return [condition.name, interval(groupRates(rows.filter(r => r.kind === "nominal")), rng), interval(groupRates(rows.filter(r => r.kind === "recovery")), rng),
        rows.filter(r => r.end_reason === "lateral_acceleration_limit").length, rows.reduce((n, r) => n + r.wall_contacts, 0)];
    }));
  renderReferenceAblations();
  const masks = Object.keys(E.featureMasks), finals = mask => data.histories ? Object.entries(data.histories).filter(([k]) => k.startsWith(`${mask}_`)) : [];
  table("ablationHistories", ["Variant", "Training loss (mean over seeds)", "Validation loss", "Stress loss", "Selected update (min to max)", "Diverged seeds"],
    masks.map(mask => {
      const entries = finals(mask), last = entries.map(([, h]) => h.at(-1)), mean = key => (last.reduce((n, r) => n + r[key], 0) / last.length).toFixed(4);
      const selected = entries.map(([k]) => data.models[k].updates);
      return [mask, mean("train_mse"), mean("validation_mse"), mean("stress_mse"), `${Math.min(...selected)} to ${Math.max(...selected)}`, entries.filter(([k]) => data.diverged[k]).length];
    }));
  drawReferenceCurves();
}
function perSeedRates(rows, seeds, masks) {
  return Object.fromEntries(masks.map(mask => [mask, seeds.map(seed => {
    const own = rows.filter(r => r.mask === mask && r.seed === seed);
    return own.reduce((n, r) => n + r.success, 0) / own.length;
  })]));
}
function summarizeAblation(ratesByMask, rng, diverged = {}) {
  const base = ratesByMask.full, rows = [];
  for (const [mask, rates] of Object.entries(ratesByMask)) {
    const s = T.summarizeSeeds(rates, rng);
    const paired = mask === "full" ? null : T.pairedDifference(rates, base, rng);
    rows.push({ mask, summary: s, paired, diverged: diverged[mask] ?? 0 });
  }
  return rows;
}
function ablationTable(id, summary, seeds) {
  table(id, ["Retrained input variant", `Success mean [95% bootstrap over ${seeds} seeds]`, "Per-seed range", "Paired vs all inputs (same seeds)", "Diverged seeds"],
    summary.map(r => [r.mask, `${pct(r.summary.mean)} [${pct(r.summary.lo)}, ${pct(r.summary.hi)}]`, `${pct(r.summary.min)} to ${pct(r.summary.max)}`,
      r.paired ? `${signed(r.paired.mean)} [${signed(r.paired.lo)}, ${signed(r.paired.hi)}]; seeds better/worse/equal ${r.paired.wins}/${r.paired.losses}/${r.paired.ties}; sign test p=${fmtP(r.paired.p)}` : "baseline",
      r.diverged]));
}
function drawDotStrip(id, summary, title, order = null) {
  const cv = $(id), c = cv.getContext("2d"), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  const rows = order ? order.map(name => summary.find(r => r.mask === name)).filter(Boolean) : summary, m = { l: 66, r: 18, t: 28, b: 54 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const my = v => m.t + (1 - v) * ph; c.font = "12px system-ui";
  for (const v of [0, .25, .5, .75, 1]) { c.strokeStyle = "#2a3a50"; c.beginPath(); c.moveTo(m.l, my(v)); c.lineTo(W - m.r, my(v)); c.stroke(); c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(pct(v), m.l - 6, my(v) + 4); }
  c.textAlign = "left"; c.fillStyle = "#ecf2fa"; c.font = "bold 13px system-ui"; c.fillText(title, m.l, 16); c.font = "12px system-ui";
  rows.forEach((r, i) => {
    const x = m.l + (i + .5) * pw / rows.length, color = colors[i % 3];
    c.fillStyle = "#acbbce"; c.textAlign = "center"; c.fillText(r.mask, x, H - 30); c.fillText(`${pct(r.summary.mean)} [${pct(r.summary.lo)}, ${pct(r.summary.hi)}]`, x, H - 14);
    c.strokeStyle = color; c.lineWidth = 3; c.beginPath(); c.moveTo(x - 38, my(r.summary.mean)); c.lineTo(x + 38, my(r.summary.mean)); c.stroke();
    c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, my(r.summary.lo)); c.lineTo(x, my(r.summary.hi)); c.moveTo(x - 10, my(r.summary.lo)); c.lineTo(x + 10, my(r.summary.lo)); c.moveTo(x - 10, my(r.summary.hi)); c.lineTo(x + 10, my(r.summary.hi)); c.stroke();
    c.fillStyle = color; r.summary.values.forEach((v, k) => { c.beginPath(); c.arc(x - 30 + (k % 10) * 6.5, my(v), 3.5, 0, 2 * Math.PI); c.fill(); });
  });
  c.textAlign = "left";
}
function renderReferenceAblations() {
  const include = referenceTracks(), masks = Object.keys(E.featureMasks), seeds = gen.seeds.ablation, rng = new E.RNG(11);
  const rows = data.ablations.filter(r => r.split === "train" && include(r.track_id));
  const summary = summarizeAblation(perSeedRates(rows, seeds, masks), rng, Object.fromEntries(masks.map(m => [m, seeds.filter(s => data.diverged[`${m}_${s}`]).length])));
  const note = `Reference: ${lab.includeLegacy ? validIds.length + legacyFolded.size : validIds.length} training tracks × 4 fixed sections (all sections pooled per seed), selected on i.i.d. validation, ${seeds.length} seeds. ${lab.testExposed ? "Sealed circuits: " + sealedSummary(masks, seeds) : "Sealed circuits stay hidden until you freeze a model."}`;
  $("referenceAblationNote").textContent = note;
  ablationTable("referenceAblationTable", summary, seeds.length);
  drawDotStrip("referenceAblationCanvas", summary, "REFERENCE RETRAINING: success per training seed (dots), mean and 95% bootstrap interval over seeds", masks);
}
function sealedSummary(masks, seeds) {
  const rows = data.ablations.filter(r => r.split === "pristine_test");
  return masks.map(mask => `${mask} ${(rows.filter(r => r.mask === mask).reduce((n, r) => n + r.success, 0) / seeds.length).toFixed(1)}/12 sections on average`).join("; ");
}
function drawReferenceCurves() {
  const seeds = gen.seeds.ablation;
  const val = seeds.map(s => data.histories[`full_${s}`].map(h => [h.update, h.validation_mse]));
  const success = seeds.map(s => data.closedLoopReference[`full_${s}`].map(h => [h.update, h.recovery.k]));
  const cv = $("referenceCurves"), c = cv.getContext("2d"); c.clearRect(0, 0, cv.width, cv.height);
  const half = cv.width / 2, panel = (x0, w, series, spec) => {
    const m = { l: 56, r: 14, t: 28, b: 34 }, pw = w - m.l - m.r, ph = cv.height - m.t - m.b, [ya, yb] = spec.y, tr = v => spec.log ? Math.log10(v) : v;
    const mx = v => x0 + m.l + v / 256 * pw, my = v => m.t + (1 - (tr(v) - tr(ya)) / (tr(yb) - tr(ya))) * ph;
    c.font = "12px system-ui"; c.fillStyle = "#ecf2fa"; c.textAlign = "left"; c.fillText(spec.title, x0 + m.l, 16);
    for (const v of spec.ticks) { c.strokeStyle = "#2a3a50"; c.beginPath(); c.moveTo(x0 + m.l, my(v)); c.lineTo(x0 + w - m.r, my(v)); c.stroke(); c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(String(v), x0 + m.l - 6, my(v) + 4); }
    c.textAlign = "center"; for (const u of [0, 64, 128, 192, 256]) { c.fillStyle = "#acbbce"; c.fillText(String(u), mx(u), cv.height - 14); }
    c.lineWidth = 1.2; c.globalAlpha = .55; c.strokeStyle = spec.color;
    for (const s of series) { c.beginPath(); s.forEach(([x, y], i) => i ? c.lineTo(mx(x), my(y)) : c.moveTo(mx(x), my(y))); c.stroke(); }
    c.globalAlpha = 1; c.lineWidth = 3; c.strokeStyle = "#ecf2fa"; c.beginPath();
    series[0].forEach(([x], i) => { const mean = series.reduce((n, s) => n + s[i][1], 0) / series.length; i ? c.lineTo(mx(x), my(mean)) : c.moveTo(mx(x), my(mean)); }); c.stroke();
  };
  panel(0, half, val, { title: "Validation loss, ten seeds (log scale): smooth and monotonic", y: [0.005, 1], log: true, ticks: [0.01, 0.1, 1], color: palette.validation });
  panel(half, half, success, { title: "Recovery sections completed (of 8), same ten seeds: erratic", y: [0, 8], ticks: [0, 2, 4, 6, 8], color: palette.recovery });
}

// ---- module wiring -----------------------------------------------------------------------------------------------------------------
for (const button of document.querySelectorAll("nav button")) button.onclick = () => {
  lab.recording = false; lab.module = button.dataset.module; lab.log.modules.add(lab.module);
  for (const panel of document.querySelectorAll("section.panel")) panel.hidden = panel.id !== lab.module;
  for (const nav of document.querySelectorAll("nav button")) nav.classList.toggle("active", nav === button);
};
$("track").onchange = fillScenarios; $("scenario").onchange = resetStart;
for (const id of ["startSpeed", "startLateral", "startHeading"]) $(id).oninput = preview;
$("camera").onchange = draw;
$("includeLegacy").onchange = () => { lab.includeLegacy = $("includeLegacy").checked; referenceTables(); };
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
  message(`Simulated the actual policy: ${result.metric.end_reason}.`); count("predictions");
});
install("newDrive", () => {
  if (currentTrack().id >= 11) throw new Error("Record training/validation drives on the eleven training circuits only.");
  const env = new E.Environment(currentTrack(), cfg), begin = start(); env.reset(begin);
  if (env.contact) throw new Error("Starting footprint contacts the wall. Move the car inward before recording.");
  const candidate = { split: $("recordSplit").value, track_id: currentTrack().id, fraction: begin.fraction, lateral: begin.lateral, heading_error: begin.heading_error, speed: begin.speed };
  const leak = T.findSplitLeak(lab.episodeStarts, candidate);
  if (leak) throw new Error(`This start is nearly identical to ${leak}, which is in the other split. Identical starts give near-identical frames, so they would leak between training and validation. Change the lane offset, heading, speed or track.`);
  do { lab.episode++; } while (lab.rows.some(row => row.episode === `drive-${lab.episode}`) || lab.episodeStarts[`drive-${lab.episode}`]);
  lab.episodeStarts[`drive-${lab.episode}`] = candidate;
  lab.manual = { env, split: $("recordSplit").value, reason: "", trace: [env.record()] };
  lab.recording = false; showRuns([{ name: `Episode ${lab.episode} / ${lab.manual.split}`, track: currentTrack(), trace: lab.manual.trace }], false);
  $("episodeStatus").textContent = `Episode ${lab.episode} assigned to ${lab.manual.split}. Its split cannot change mid-drive, and its start is recorded so a near-identical start cannot enter the other split.`;
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
  E.validateRows(combined);
  const episodes = { ...lab.episodeStarts, ...data.starterEpisodes }; T.validateEpisodeStarts(episodes);
  lab.rows = combined; lab.episodeStarts = episodes;
  updateDataset(); message("Loaded real expert episodes (train and validation drawn independently from the same sampler), not student driving. The stress set is evaluation only.");
});
install("clearDataset", () => { lab.rows = []; lab.episodeStarts = {}; lab.manual = null; lab.recording = false; updateDataset(); setExposure(); });
install("downloadDataset", () => { E.validateRows(lab.rows); download("student-driving-dataset.json", { version: E.version, config: cfg, rows: lab.rows, episodes: lab.episodeStarts, audit: { testExposed: lab.testExposed } }); });
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
// A file input fires "change" only when the chosen file differs, so retrying the same file (after Clear, or after fixing a rejected file in place) would silently do
// nothing. Reading the file and then emptying the input makes every selection count; the File object stays readable.
function onFile(id, action) {
  const input = $(id);
  input.onchange = () => handle(async () => {
    const file = input.files[0]; input.value = "";
    if (file) await action(file);
  });
}
onFile("importDataset", async file => {
  const imported = JSON.parse(await file.text());
  if (!E.compatibleVersions.includes(imported.version) || ["max_speed", "max_steer", "track_width", "car_width"].some(key => imported.config?.[key] !== cfg[key]))
    throw new Error("Dataset version or observation normalization does not match this lab.");
  E.validateRows(imported.rows); T.validateEpisodeStarts(imported.episodes ?? {});
  lab.rows = imported.rows; lab.episodeStarts = imported.episodes ?? {}; lab.manual = null; lab.recording = false; updateDataset(); message("Imported local examples; no data was uploaded.");
});
install("trainStudent", async () => {
  const { train, validation } = datasets(), seed = number("studentSeed", -2147483648, 2147483647, true);
  const rate = number("learningRate", .00001, .05), updates = number("updates", 1, 400, true), mask = $("studentMask").value, every = Number($("monitorEvery").value);
  lab.recording = false; lab.frozen = null;
  lab.studentMetadata = { seed, mask, learning_rate: rate, planned_fit_updates: updates };
  const initial = new E.Network(seed, [18, 24, 16, 2], mask);
  lab.student = initial; lab.losses = []; lab.probe = []; lab.trainingSnapshots = [{ update: 0, model: initial.serialize() }];
  const result = await T.fit({ seed, mask, train, validation, stress: data.stressRows, updates, learningRate: rate, initial, evalEvery: 4,
    probe: every ? net => T.evaluateSections(net, monitorSections, ctx) : null, probeEvery: every || 20, yieldFn: tickYield, shouldCancel: () => lab.cancel,
    onProgress: (net, point, history) => {
      lab.student = net; lab.losses = history.slice(); lab.probe = history.filter(p => p.probe).map(p => ({ update: p.update, nominal: p.probe.nominal, recovery: p.probe.recovery }));
      lab.trainingSnapshots.push({ update: point.update, model: net.serialize() }); drawLoss(); drawProbe();
      $("studentTraining").textContent = `${point.update}/${updates} real Adam updates; validation ${point.validation.toFixed(5)}, stress ${point.stress.toFixed(5)}.`;
    } });
  lab.student = result.last; lab.losses = result.history; lab.probe = result.history.filter(p => p.probe).map(p => ({ update: p.update, nominal: p.probe.nominal, recovery: p.probe.recovery }));
  lab.lastFit = { stopped: result.stopped, bestUpdate: result.bestUpdate, divergence: result.divergence }; drawLoss(); drawProbe();
  $("model").value = "student"; count("trained_models");
  $("learningUpdate").max = lab.trainingSnapshots.length - 1; $("learningUpdate").value = Math.floor((lab.trainingSnapshots.length - 1) / 2);
  if (result.stopped === "diverged") throw new Error(`Training diverged at update ${result.divergence.update}: ${result.divergence.message}`);
  const run = E.simulate(currentTrack(), cfg, lab.student, start());
  showRuns([{ ...run, name: `Student network / ${lab.student.updates} actual updates`, track: currentTrack() }]);
  message(`Trained from scratch (${result.stopped}). Loss is not a safety guarantee; inspect the actual drive and the closed-loop curve.`); setExposure();
}, true);
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
  const snapshots = [data.evolution[0], data.evolution.find(snapshot => snapshot.update === 128), { update: data.models.full_301.updates, model: data.models.full_301 }];
  showRuns(snapshots.map(snapshot => ({ ...E.simulate(currentTrack(), cfg, E.Network.fromJSON(snapshot.model), start()),
    name: `Reference trained weights / update ${snapshot.update}`, track: currentTrack() })));
  message("Actual saved NumPy checkpoints (seed 301), simulated on your selected track, not invented improvement frames.");
});
install("exportModel", () => {
  if (!lab.student) throw new Error("Train a student model first.");
  download("student-driving-model.json", { ...lab.student.serialize(), config: cfg, training: lab.studentMetadata,
    optimizer_note: "Exported weights only; restarting optimizer requires fresh moments", audit: audit(lab.student) });
});
install("inspectStep", () => {
  const { train } = datasets();
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
  $("stepLoss").textContent = `Minibatch loss ${packet.loss.toFixed(6)} → ${packet.lossAfter.toFixed(6)}; global gradient norm ${packet.gradNorm.toExponential(3)}; actual update counter ${packet.updates}.`;
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
  message("A real update was applied. The finite-difference check independently verifies its gradient."); count("gradient_steps");
});
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
install("stepNext", () => { if (!lab.packet) throw new Error("Inspect a minibatch first."); lab.stage = Math.min(4, lab.stage + 1); explainStage(); });
install("driveAfterStep", () => {
  if (!lab.packet) throw new Error("Inspect a minibatch first.");
  showRuns([["Before real update", lab.packet.beforeModel], ["After real update", lab.packet.afterModel]].map(([name, model]) =>
    ({ ...E.simulate(currentTrack(), cfg, E.Network.fromJSON(model), start()), name, track: currentTrack() })));
});
install("compareControllers", async () => {
  const model = selectedModel(), clone = model.clone ? model.clone() : model;
  const result = await runExperiment({ conditions: [{ name: "Rules", policy: "rules" }, { name: "Expert", policy: "expert" }, { name: "Neural", policy: clone }],
    model: clone, poolKey: $("controllerPool").value, seeds: Number($("controllerSeeds").value), label: "controller comparison" });
  lab.comparisons = result; renderConditions("controllerTable", "controllerSections", result); showRuns(representativeRuns(result, clone)); count("controller_comparisons");
  message(`Matched starts and equal simulation time over ${result.pool}; failed cars freeze at their real endpoint.${result.exploratory ? " Exploratory: fewer than 8 sections or 10 seeds." : ""}`);
}, true);
install("downloadComparison", () => { if (!lab.comparisons) throw new Error("Run a comparison first."); download("controller-comparison.json", lab.comparisons); });
install("localAblation", async () => {
  const { train, validation } = datasets(), updates = number("updates", 1, 400, true), rate = number("learningRate", .00001, .05);
  const seedCount = Number($("ablationSeeds").value), pool = poolSections($("ablationPool").value), masks = Object.keys(E.featureMasks), diverged = Object.fromEntries(masks.map(m => [m, 0]));
  const rates = Object.fromEntries(masks.map(m => [m, []])), models = [];
  for (const mask of masks) for (let i = 0; i < seedCount; i++) {
    const seed = 301 + i, fitted = await T.fit({ seed, mask, train, validation, updates, learningRate: rate, evalEvery: 4, yieldFn: tickYield, shouldCancel: () => lab.cancel });
    if (lab.cancel) throw new Error("Cancelled: no result was recorded.");
    if (fitted.stopped === "diverged") diverged[mask]++;
    const net = fitted.stopped === "diverged" ? null : fitted.best, evaluation = net ? T.evaluateSections(net, pool.sections, ctx) : { k: 0, n: pool.sections.length };
    rates[mask].push(evaluation.k / evaluation.n);
    models.push({ mask, seed, actual_updates: fitted.net.updates, selected_update: fitted.bestUpdate, stopped: fitted.stopped, success: evaluation.k, sections: evaluation.n, weights: net?.serialize() });
    $("ablationStatus").textContent = `Really retrained ${mask}, seed ${seed}: ${fitted.net.updates} updates, validation-selected ${fitted.bestUpdate}. Diverged seeds count as failures.`; await tickYield();
  }
  const summary = summarizeAblation(rates, new E.RNG(13), diverged);
  lab.localAblations = { rows: summary.map(r => ({ mask: r.mask, ...r.summary })), models, pool: pool.label, seeds: seedCount, summary,
    audit: { engine: E.version, testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed, model_protocol: `${masks.length * seedCount} fresh independent models, selected only by i.i.d. validation` } };
  ablationTable("ablationTable", summary, seedCount);
  drawDotStrip("ablationCanvas", summary, `YOUR RETRAINING: success per seed on ${pool.label}`, masks);
  $("ablationStatus").textContent = `${masks.length * seedCount} models retrained on your dataset; evaluation: ${pool.label}.${seedCount < 10 ? " EXPLORATORY: fewer than 10 seeds." : ""} Diverged seeds count as failures.`;
  message(`All ${masks.length * seedCount} models were independently retrained, never inference-only masked.`); count("feature_retrains");
}, true);

// ---- generalization ------------------------------------------------------------------------------------------------------------------
install("freezeReveal", async () => {
  const chosen = selectedModel();
  if ($("model").value === "student" && chosen.updates === 0) throw new Error("Train your student model before freezing.");
  mergeStoredExposure();
  const alreadyExposed = lab.testExposed;
  lab.frozen = chosen.clone ? chosen.clone() : chosen; const fingerprint = E.fingerprint(lab.frozen), rows = [];
  for (const track of tracks.filter(t => t.id >= 13)) {
    for (const scenario of data.scenarios[track.id]) {
      const result = E.simulate(track, cfg, lab.frozen, scenario);
      rows.push([track.name, scenario.name, result.metric.success ? "SAFE CENTERED EXIT" : result.metric.end_reason,
        `${(result.metric.progress * 100).toFixed(1)}%`, result.metric.min_clearance_m.toFixed(3)]);
      await tickYield();
    }
  }
  if (E.fingerprint(lab.frozen) !== fingerprint) throw new Error("Frozen weights changed during testing");
  lab.testExposed = true; fillTracks(); setExposure(); referenceTables();
  lab.frozenAudit = { fingerprint, alreadyExposed, sections: rows.map(r => ({ circuit: r[0], section: r[1], outcome: r[2] })) };
  $("freezeStatus").textContent = `Frozen weights fingerprint ${fingerprint}. Twelve fixed test sections evaluated without weight updates.
${alreadyExposed ? "The test set was already exposed: these are exploratory results." : "This is the first UI reveal in this classroom session."}`;
  table("testTable", ["New test circuit", "Section", "Outcome", "Progress", "Minimum wall margin (m)"], rows);
  $("demonstrateLeak").disabled = false; $("evaluateFresh").disabled = !lab.fresh; message("Test circuits exposed only after the model was frozen."); count("freeze_reveals");
}, true);
install("demonstrateLeak", async () => {
  if (!lab.frozen || !lab.testExposed) throw new Error("Freeze and reveal tests first.");
  const { train } = datasets(); E.validateRows(data.testRows, true);
  const contaminated = lab.frozen.clone(), before = contaminated.loss(data.testRows), rng = new E.RNG(514);
  for (let update = 0; update < 40; update++) {
    const batch = [...Array.from({ length: 32 }, () => train[rng.index(train.length)]), ...Array.from({ length: 32 }, () => data.testRows[rng.index(data.testRows.length)])];
    contaminated.trainBatch(batch, .0005, { trackChanges: false, skipLossAfter: true });
    if (update % 4 === 0) await tickYield();
  }
  const after = contaminated.loss(data.testRows); lab.testLabelsExposed = true; setExposure();
  table("leakTable", ["Model / evaluation", "Measured error", "Valid new-test generalization?"], [
    ["Frozen clean model", before.toFixed(6), "Only before test exposure"],
    ["Clone trained on these same test labels", after.toFixed(6), "NO: contaminated"],
    ["Python reference clean test", data.leakage.clean_test_mse.toFixed(6), "Clean, frozen before test"],
    ["Python reference contaminated test", data.leakage.contaminated_test_mse.toFixed(6), "NO: contaminated"],
  ]);
  lab.leakage = { before, after, trained_test_labels: true, valid_generalization: false, model: contaminated.serialize() };
  $("leakStatus").textContent = "INVALID BENCHMARK: the separate clone saw the very labels used for its test score. Your clean model was not overwritten.";
  message("Actual test-label training completed. A lower score, if observed, is not honest unseen performance."); count("leakage_demos");
}, true);
onFile("freshPack", async file => {
  const pack = JSON.parse(await file.text());
  if (pack.schema !== "f1-ml-lab-fresh-pack-v1" || !Array.isArray(pack.tracks) || !pack.tracks.length || !pack.scenarios) throw new Error("This is not a fresh test pack for this lab.");
  const finite = (...values) => values.every(v => typeof v === "number" && Number.isFinite(v));
  if (new Set(pack.tracks.map(item => item.id)).size !== pack.tracks.length) throw new Error("Pack circuits need distinct ids.");
  const packTracks = pack.tracks.map(item => {
    for (const key of ["xy", "left", "right", "tangent", "normal", "heading", "curvature"]) if (!Array.isArray(item[key]) || !item[key].length) throw new Error(`Pack circuit ${item.name} is missing ${key}.`);
    if (!finite(item.id, item.ds, item.length) || item.ds <= 0 || tracks.some(t => t.id === item.id)) throw new Error(`Pack circuit ${item.name} needs an id that no built-in circuit uses and a positive sample spacing.`);
    const scenarios = pack.scenarios[item.id];
    if (!Array.isArray(scenarios) || !scenarios.length || !scenarios.every(s => typeof s.name === "string" && finite(s.fraction, s.lateral, s.heading_error, s.speed)))
      throw new Error(`Pack circuit ${item.name} has no valid sections.`);
    return new E.Track(item);
  });
  lab.fresh = { pack, tracks: packTracks, fingerprint: pack.fingerprint ?? "unspecified" };
  $("freshStatus").textContent = `Loaded ${packTracks.length} fresh circuits (pack fingerprint ${lab.fresh.fingerprint}). Evaluate once, after freezing.`;
  $("evaluateFresh").disabled = !lab.frozen;
});
install("evaluateFresh", async () => {
  if (!lab.frozen) throw new Error("Freeze a model in the step above first; a fresh pack is for one evaluation of a frozen model.");
  if (!lab.fresh) throw new Error("Load a fresh test pack first.");
  const fingerprint = E.fingerprint(lab.frozen), rows = [];
  for (const track of lab.fresh.tracks) for (const [i, scenario] of lab.fresh.pack.scenarios[track.id].entries()) {
    const result = E.simulate(track, cfg, lab.frozen, scenario);
    rows.push([track.name, scenario.name, result.metric.success ? "SAFE CENTERED EXIT" : result.metric.end_reason, `${(result.metric.progress * 100).toFixed(1)}%`, result.metric.min_clearance_m.toFixed(3)]);
    await tickYield();
  }
  if (E.fingerprint(lab.frozen) !== fingerprint) throw new Error("Frozen weights changed during testing");
  lab.freshResult = { fingerprint, pack: lab.fresh.fingerprint, rows };
  table("freshTable", ["Fresh circuit", "Section", "Outcome", "Progress", "Minimum wall margin (m)"], rows);
  $("freshStatus").textContent = `Frozen model ${fingerprint} on pack ${lab.fresh.fingerprint}: ${rows.filter(r => r[2] === "SAFE CENTERED EXIT").length}/${rows.length} sections. A fixed set of sections, not a rate over circuits in general.`;
  count("fresh_pack_evaluations");
}, true);

// ---- robustness ----------------------------------------------------------------------------------------------------------------------------
install("runRobustness", async () => {
  const noise = number("noise", 0, 3), delay = number("delay", 0, 20, true), grip = number("grip", .2, 1.2), model = selectedModel(), clone = model.clone ? model.clone() : model;
  const result = await runExperiment({ conditions: [{ name: "Clean", noise: 0, delay: 0, grip: 1 }, { name: "Selected distribution shift", noise, delay, grip }],
    model: clone, poolKey: $("robustPool").value, seeds: Number($("robustSeeds").value), label: "robustness" });
  lab.robustness = result; renderConditions("robustnessTable", null, result); showRuns(representativeRuns(result, clone)); count("robustness_runs");
  message(`Observations and braking/handling envelope genuinely changed; no safety controller rescued the policy.${result.exploratory ? " Exploratory: fewer than 8 sections or 10 seeds." : ""}`);
}, true);
install("downloadRobustness", () => { if (!lab.robustness) throw new Error("Run a robustness test first."); download("robustness-experiment.json", lab.robustness); });

// ---- animation loop and start-up ------------------------------------------------------------------------------------------------------------------
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
restoreExposure(); fillTracks(); $("track").value = String(gen.monitor_tracks[1]); fillScenarios(); updateDataset(); referenceTables(); drawLoss(); drawProbe(); lab.log.modules.add("predict"); requestAnimationFrame(animate);
