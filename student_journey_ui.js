"use strict";
// Module 12, "From zero to a lap". Trains the imitation and the reinforcement learner on one circuit, shows a whole-lap attempt for every stage of
// training as a filmstrip, and replays any pair of stages next to the expert. The training and the lap tests are in student_journey.js, which the
// headless reference run (tools/journey_reference.cjs) also uses.
(() => {
  const JR = window.StudentJourney, RL = window.StudentRL, methods = ["imitation", "reinforcement"];
  const methodColor = { imitation: colors[0], reinforcement: colors[1], expert: colors[2] };
  const methodName = { imitation: "Imitation (DAgger)", reinforcement: "Reinforcement (PPO)" };
  const reasonText = { wall_contact: "hit the wall", off_track: "left the road", lateral_acceleration_limit: "exceeded the grip limit", numerical_divergence: "diverged", timeout: "ran out of time" };
  const reasonShort = { wall_contact: "wall", off_track: "off road", lateral_acceleration_limit: "grip limit", numerical_divergence: "diverged", timeout: "timeout" };
  const cap = text => text[0].toUpperCase() + text.slice(1);
  const grouped = n => Math.round(n).toLocaleString("en-US");
  const compactSteps = n => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(Math.round(n));
  const stageTitle = stage => stage.index === 0 ? "Untrained" : stage.method === "imitation" ? (stage.round === 0 ? "Expert demonstrations" : `DAgger round ${stage.round}`) : `Iteration ${stage.iteration}`;
  const stageShort = stage => stage.index === 0 ? "Untrained" : stage.method === "imitation" ? (stage.round === 0 ? "Demos" : `Round ${stage.round}`) : `Iter ${stage.iteration}`;
  const lapOutcome = lap => lap.completed ? `LAP ${lap.seconds.toFixed(0)} s` : `${JR.lapPercent(lap)}% · ${reasonShort[lap.reason] ?? lap.reason}`;
  const lapSentence = lap => lap.completed ? `completed the lap in ${lap.seconds.toFixed(1)} s`
    : `failed at ${JR.lapPercent(lap)}% of the lap (${reasonText[lap.reason] ?? lap.reason}) after ${lap.seconds.toFixed(1)} s`;
  const stagesOf = (j, method) => j?.[method]?.stages ?? [];
  const log10x = steps => Math.log10(Math.max(steps, 30));
  const reference = data.journeyReference ?? null;
  const say = text => { $("journeyStatus").textContent = text; message(text); };

  function freshJourney(trackId) {
    return { trackId, expert: JR.expertLap(trackMap.get(trackId), cfg), imitation: null, reinforcement: null, selected: { imitation: 0, reinforcement: 0 } };
  }
  // The pictures in the filmstrip: spaced more closely early on, always including the first lap and the last stage.
  function keyStages(stages, count = 10) {
    const last = stages.length - 1;
    if (last < count) return stages.map((_, i) => i);
    const picks = new Set(Array.from({ length: count }, (_, i) => Math.round(last * (i / (count - 1)) ** 1.6)));
    const first = stages.findIndex(s => s.lap.completed);
    if (first > 0 && !picks.has(first)) {
      const middle = [...picks].filter(i => i > 0 && i < last), nearest = middle.reduce((best, i) => Math.abs(i - first) < Math.abs(best - first) ? i : best, middle[0]);
      picks.delete(nearest); picks.add(first);
    }
    return [...picks].sort((a, b) => a - b);
  }
  // The saved ten-seed bands only apply to the settings they were produced with.
  function referenceFor(j) {
    if (!reference || !j) return null;
    const settings = reference.settings, circuit = reference.circuits[j.trackId], rl = j.reinforcement?.settings;
    const matches = method => j[method]?.source === "saved" || (method === "imitation" ? j.imitation?.settings.rounds === settings.imitation.rounds
      : rl?.iterations === settings.reinforcement.iterations && rl?.reward.smoothness === settings.reinforcement.reward.smoothness);
    return circuit ? { circuit, imitation: matches("imitation") ? circuit.imitation : null, reinforcement: matches("reinforcement") ? circuit.reinforcement : null } : null;
  }
  const steeringText = lap => lap.seconds < JR.FLICKER_MIN_SECONDS ? "too short a drive to judge the steering"
    : JR.isFlicking(lap.seconds, lap.flicker) ? `FLICKS the steering: ${lap.flicker.toFixed(1)}° per step` : `steering moves ${lap.flicker.toFixed(2)}° per step`;

  // ---- filmstrip ------------------------------------------------------------------------------------------------------------------------
  const boundsCache = new Map();
  function trackBounds(track) {
    if (!boundsCache.has(track.id)) {
      const points = track.left.concat(track.right), xs = points.map(p => p[0]), ys = points.map(p => p[1]);
      boundsCache.set(track.id, { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys), road: track.left.concat([...track.right].reverse()) });
    }
    return boundsCache.get(track.id);
  }
  function drawThumbnail(c, track, stage, x, y, size, color, selected) {
    const b = trackBounds(track), pad = 6, scale = Math.min((size - 2 * pad) / (b.maxX - b.minX), (size - 2 * pad) / (b.maxY - b.minY));
    const ox = x + (size - scale * (b.maxX - b.minX)) / 2, oy = y + (size - scale * (b.maxY - b.minY)) / 2;
    const project = p => [ox + (p[0] - b.minX) * scale, oy + (b.maxY - p[1]) * scale];
    c.fillStyle = "#101722"; c.fillRect(x, y, size, size);
    c.beginPath(); b.road.forEach((p, i) => i ? c.lineTo(...project(p)) : c.moveTo(...project(p))); c.closePath(); c.fillStyle = "#405064"; c.fill();
    const { path, completed } = stage.lap;
    c.beginPath(); path.forEach((p, i) => i ? c.lineTo(...project(p)) : c.moveTo(...project(p))); c.strokeStyle = color; c.lineWidth = 2.2; c.stroke();
    const [sx, sy] = project(path[0]), [ex, ey] = project(path.at(-1));
    c.fillStyle = "#ecf2fa"; c.beginPath(); c.arc(sx, sy, 2.6, 0, 2 * Math.PI); c.fill();
    c.lineWidth = 2.4;
    if (completed) { c.strokeStyle = "#4de2c2"; c.beginPath(); c.arc(sx, sy, 6, 0, 2 * Math.PI); c.stroke(); }
    else { c.strokeStyle = "#ff5b65"; c.beginPath(); c.moveTo(ex - 4.5, ey - 4.5); c.lineTo(ex + 4.5, ey + 4.5); c.moveTo(ex + 4.5, ey - 4.5); c.lineTo(ex - 4.5, ey + 4.5); c.stroke(); }
    c.strokeStyle = selected ? "#ffc56b" : completed ? "#4de2c2" : "#334358"; c.lineWidth = selected ? 3 : 1.5; c.strokeRect(x + .5, y + .5, size - 1, size - 1);
  }
  function drawGallery() {
    const cv = $("journeyGallery"), c = cv.getContext("2d"), W = cv.width, j = lab.journey;
    c.clearRect(0, 0, W, cv.height); lab.journeyCells = [];
    if (!j || !(j.imitation || j.reinforcement)) {
      c.fillStyle = "#acbbce"; c.font = "15px system-ui"; c.textAlign = "left";
      c.fillText("Train both learners, or load the saved example, to fill the filmstrip.", 20, 40); return;
    }
    const track = trackMap.get(j.trackId), columns = 10, cell = (W - 20) / columns, size = cell - 8;
    c.fillStyle = "#acbbce"; c.font = "12px system-ui"; c.textAlign = "right";
    c.fillText(`${track.name} · ${track.length.toFixed(0)} m · expert lap ${j.expert.seconds.toFixed(1)} s`, W - 10, 18);
    methods.forEach((method, row) => {
      const top = 8 + row * 196, entry = j[method];
      c.textAlign = "left"; c.fillStyle = methodColor[method]; c.font = "bold 15px system-ui";
      c.fillText(`${methodName[method]}${entry ? ` · seed ${entry.seed}${entry.source === "saved" ? " · saved example" : ""}` : ""}`, 10, top + 14);
      if (!entry) { c.fillStyle = "#acbbce"; c.font = "13px system-ui"; c.fillText("Not trained yet.", 10, top + 44); return; }
      keyStages(entry.stages).forEach((index, k) => {
        const stage = entry.stages[index], x = 10 + k * cell + 4, y = top + 26, middle = x + size / 2;
        drawThumbnail(c, track, stage, x, y, size, methodColor[method], j.selected[method] === index);
        lab.journeyCells.push({ method, index, x, y, size, height: size + 62 });
        c.textAlign = "center"; c.font = "bold 12px system-ui"; c.fillStyle = "#ecf2fa"; c.fillText(stageShort(stage), middle, y + size + 15);
        c.fillStyle = stage.lap.completed ? "#4de2c2" : "#ff8990"; c.fillText(lapOutcome(stage.lap), middle, y + size + 30);
        c.font = "11px system-ui"; c.fillStyle = "#acbbce";
        c.fillText(`${compactSteps(stage.steps)} steps`, middle, y + size + 44); c.fillText(JR.describeMinutes(stage.minutes), middle, y + size + 57);
      });
    });
  }
  $("journeyGallery").onclick = event => {
    const cv = event.currentTarget, box = cv.getBoundingClientRect(), px = (event.clientX - box.left) * cv.width / box.width, py = (event.clientY - box.top) * cv.height / box.height;
    const hit = lab.journeyCells?.find(cell => px >= cell.x && px <= cell.x + cell.size && py >= cell.y && py <= cell.y + cell.height);
    if (!hit || lab.busy) return;
    lab.journey.selected[hit.method] = hit.index; refresh(); replay();
  };

  // ---- curves, table and summaries ----------------------------------------------------------------------------------------------------
  function drawCurves() {
    const j = lab.journey, ref = referenceFor(j), all = methods.flatMap(m => stagesOf(j, m).map(s => s.steps));
    const x1 = Math.max(4, Math.ceil(Math.log10(Math.max(1000, ...all)))), x0 = Math.log10(30);
    const axis = { x: [x0, x1], xTicks: [x0, ...Array.from({ length: x1 - 1 }, (_, i) => i + 2)], xFormat: v => v < 1.6 ? "0" : compactSteps(10 ** v), xLabel: "simulated driving steps so far (log scale; 10 steps = 1 s)" };
    const line = (method, point) => ({ color: methodColor[method], markers: true, points: point });
    plot("journeyProgress", { ...axis, y: [0, 100], format: v => `${v}%`, title: "Share of the lap completed before the car failed",
      series: methods.filter(m => stagesOf(j, m).length).map(m => line(m, stagesOf(j, m).map(s => [log10x(s.steps), JR.lapPercent(s.lap)]))),
      bands: methods.filter(m => ref?.[m]).map(m => ({ color: methodColor[m], points: ref[m].curve.map(p => [log10x(p.steps), p.lo, p.hi]) })) });
    const times = methods.flatMap(m => stagesOf(j, m).filter(s => s.lap.completed).map(s => s.lap.seconds)), expert = j?.expert.seconds ?? 120;
    const high = Math.ceil(Math.max(expert * 1.25, ...times) / 10) * 10, low = Math.floor(Math.min(expert * .85, ...times) / 10) * 10;
    plot("journeyLapTime", { ...axis, y: [low, high], format: v => `${v.toFixed(0)} s`, title: "Lap time of the laps that were completed (dashed: expert)",
      series: [...methods.filter(m => stagesOf(j, m).some(s => s.lap.completed)).map(m => line(m, stagesOf(j, m).filter(s => s.lap.completed).map(s => [log10x(s.steps), s.lap.seconds]))),
        { color: methodColor.expert, dash: [6, 6], points: [[x0, expert], [x1, expert]] }] });
    $("journeyBandNote").textContent = ref && (ref.imitation || ref.reinforcement)
      ? `Shaded bands: the range over the ${reference.settings.seeds} seeds of the saved reference run on this circuit (the lines are your run). A band that is wide is a learner whose result depends on the seed.`
      : reference ? `The saved ${reference.settings.seeds}-seed bands appear when the budgets are the default ${reference.settings.imitation.rounds} DAgger rounds and ${reference.settings.reinforcement.iterations} PPO iterations.` : "";
  }
  function stageDetail(method) {
    const j = lab.journey, entry = j?.[method];
    if (!entry?.stages.length) return "Not trained yet.";
    const stage = entry.stages[j.selected[method]];
    const cost = method === "imitation" ? `${grouped(stage.steps)} expert-labelled steps` : `${grouped(stage.steps)} simulated driving steps, no labels`;
    return `${stageTitle(stage)} (stage ${stage.index + 1} of ${entry.stages.length})\n${cost}: ${JR.describeMinutes(stage.minutes)} of driving, ${grouped(stage.updates)} weight updates\nLap: ${lapSentence(stage.lap)}; ${steeringText(stage.lap)}.`;
  }
  function renderTable() {
    const j = lab.journey, rows = [];
    for (const method of methods) for (const s of stagesOf(j, method)) {
      const expert = j.expert.seconds;
      rows.push([methodName[method], stageTitle(s), grouped(s.steps), JR.describeMinutes(s.minutes), grouped(s.updates), s.lap.completed ? "completed" : `${JR.lapPercent(s.lap)}%: ${reasonText[s.lap.reason] ?? s.lap.reason}`,
        s.lap.completed ? `${s.lap.seconds.toFixed(1)} s` : "—", s.lap.completed ? `${(s.lap.seconds / expert).toFixed(2)}×` : "—",
        s.lap.seconds < JR.FLICKER_MIN_SECONDS ? "—" : `${s.lap.flicker.toFixed(2)}${JR.isFlicking(s.lap.seconds, s.lap.flicker) ? " (flicking)" : ""}`]);
    }
    table("journeyTable", ["Learner", "Stage", "Driving steps (0.1 s each)", "Driving time", "Weight updates", "Lap result", "Lap time", "Lap time ÷ expert", "Steering movement (° per step)"], rows);
  }
  function renderSummary() {
    const j = lab.journey, lines = [];
    const first = method => stagesOf(j, method).find(s => s.lap.completed), last = method => stagesOf(j, method).at(-1);
    for (const method of methods) {
      const stages = stagesOf(j, method);
      if (!stages.length) continue;
      const a = first(method), z = last(method);
      lines.push(`${cap(method).padEnd(14)} ${a ? `first lap at ${stageTitle(a)}: ${grouped(a.steps)} steps (${JR.describeMinutes(a.minutes)} of driving)` : "has not completed a lap yet"}; final car ${z.lap.completed ? `laps in ${z.lap.seconds.toFixed(1)} s (${(z.lap.seconds / j.expert.seconds).toFixed(2)}× the expert)` : `fails at ${JR.lapPercent(z.lap)}% of the lap`}; ${steeringText(z.lap)}.`);
    }
    const a = first("imitation"), b = first("reinforcement");
    if (a && b && a.steps > 0) lines.push(`Reinforcement needed ${(b.steps / a.steps).toFixed(0)}× more driving steps than imitation for its first lap. Imitation also needed the expert to label every one of its steps.`);
    $("journeyCompare").textContent = lines.join("\n");
  }
  function drawTraining() {
    const history = lab.journey?.reinforcement?.history ?? [], last = Math.max(10, history.at(-1)?.iteration ?? 0);
    plot("journeyTraining", { x: [0, last], y: [0, 100], format: v => `${v}%`, title: "Training drives that ended in a crash (red) and exploration noise, σ × 100 (grey)", xLabel: "PPO iteration",
      series: [{ color: "#ff8990", points: history.filter(h => h.failureRate !== null).map(h => [h.iteration, 100 * h.failureRate]) }, { color: "#acbbce", points: history.map(h => [h.iteration, 100 * h.std]) }] });
  }
  function updateControls() {
    const j = lab.journey;
    for (const method of methods) {
      const entry = j?.[method], slider = $(`journey${cap(method)}Stage`);
      slider.max = entry ? entry.stages.length - 1 : 0; slider.value = entry ? j.selected[method] : 0; slider.disabled = !entry;
      $(`journey${cap(method)}Label`).textContent = entry ? `${methodName[method]} stage: ${stageTitle(entry.stages[j.selected[method]])}` : `${methodName[method]} stage`;
      $(`journey${cap(method)}Detail`).textContent = stageDetail(method);
      const option = $("model").querySelector(`option[value="journey${cap(method)}"]`), ready = stagesOf(j, method).length > 1;
      option.disabled = !ready; option.textContent = ready ? `Final ${method} car from module 12 (trained on circuit ${j.trackId})` : `Final ${method} car (train in module 12)`;
    }
    $("journeyReplay").disabled = $("journeyMovie").disabled = !(j && (j.imitation || j.reinforcement));
  }
  function refresh() { updateControls(); drawGallery(); drawCurves(); drawTraining(); renderTable(); renderSummary(); }
  for (const method of methods) $(`journey${cap(method)}Stage`).oninput = event => { lab.journey.selected[method] = Number(event.target.value); refresh(); };

  // ---- replay ---------------------------------------------------------------------------------------------------------------------------
  // A whole lap lasts minutes and shows best on the whole circuit, so the first journey replay of a visit raises the speed from the lessons' 4x and switches the camera to the whole circuit.
  // Both settings are the student's from then on. The view is remembered when the module is entered and given back, as it was, when it is left.
  // The replay normally sits at the top of the page, off-screen from the filmstrip; in this module it is docked next to the filmstrip (with its camera menu), so a click on a picture and its lap share a screen.
  const homeMarker = document.createComment("replay view"), cameraMarker = document.createComment("camera menu"), cameraBox = $("camera").parentNode;
  $("view").before(homeMarker); cameraBox.before(cameraMarker);
  const docked = () => $("view").parentNode === $("journeyViewHost");
  function journeyView() {
    const view = lab.viewBeforeJourney ??= { speed: $("replaySpeed").value, camera: $("camera").value, taken: false };
    if (view.taken) return;
    view.taken = true;
    if (Number($("replaySpeed").value) < 8) $("replaySpeed").value = "16";
    $("camera").value = "overview";
  }
  for (const nav of document.querySelectorAll("nav button")) nav.addEventListener("click", () => {
    if (lab.module === "journey") {
      lab.viewBeforeJourney ??= { speed: $("replaySpeed").value, camera: $("camera").value, taken: false };
      if (!docked()) { $("journeyViewHost").append($("view"), $("replayControls")); $("journeyViewHost").prepend(cameraBox); }
      return;
    }
    if (docked()) { homeMarker.after($("view"), $("replayControls")); cameraMarker.after(cameraBox); }
    const view = lab.viewBeforeJourney;
    if (!view) return;
    lab.viewBeforeJourney = null; $("replaySpeed").value = view.speed;
    if ($("camera").value !== view.camera) { $("camera").value = view.camera; draw(); }
  });
  function replay() {
    const j = lab.journey;
    if (!j || !(j.imitation || j.reinforcement)) throw new Error("Train a learner or load the saved example first.");
    const track = trackMap.get(j.trackId), runs = [];
    for (const method of methods) {
      const entry = j[method];
      if (!entry) continue;
      const stage = entry.stages[j.selected[method]], run = JR.lapRun(track, cfg, E.Network.fromJSON(stage.model));
      runs.push({ name: `${cap(method)} · ${stageTitle(stage)}`, track, trace: run.trace, metric: run.metric, color: methodColor[method],
        marker: stage.lap.completed ? null : { index: run.trace.length - 1, label: `${JR.lapPercent(stage.lap)}%` } });
    }
    const expert = JR.lapRun(track, cfg, "expert");
    runs.push({ name: "Expert", track, trace: expert.trace, metric: expert.metric, color: methodColor.expert, marker: null });
    journeyView(); showRuns(runs, true);
    $("journeyCaption").textContent = runs.slice(0, -1).map(r => `${r.name}: ${r.metric.end_reason === "section_complete" ? `lap in ${r.metric.duration_s.toFixed(1)} s` : `${r.marker.label} of the lap, then ${r.metric.end_reason.replaceAll("_", " ")}`}`).join("   |   ");
    count("journey_replays");
  }
  $("journeyReplay").onclick = () => {
    if (lab.busy) { $("error").textContent = "Wait for the current experiment to finish."; return; }
    try { $("error").textContent = ""; replay(); } catch (error) { $("error").textContent = error.message; console.error(error); }
  };

  // Plays one picture of the filmstrip after another, each until both learners' laps have ended. Leaving the module or cancelling stops it.
  async function movie() {
    const j = lab.journey;
    if (!j || !(j.imitation || j.reinforcement)) throw new Error("Train a learner or load the saved example first.");
    const keys = Object.fromEntries(methods.map(m => [m, keyStages(stagesOf(j, m))])), frames = Math.max(...methods.map(m => keys[m].length)), here = () => lab.module === "journey" && !lab.cancel;
    for (let f = 0; f < frames && here(); f++) {
      for (const m of methods) if (keys[m].length) j.selected[m] = keys[m][Math.min(f, keys[m].length - 1)];
      refresh(); replay();
      say(`Learning movie: picture ${f + 1} of ${frames}`);
      const shown = lab.runs, end = Math.min(Math.max(...shown.slice(0, -1).map(r => r.trace.at(-1).time)) + 1, Math.max(...shown.map(r => r.trace.at(-1).time)));
      // Another action may replace the replay; then this picture is over.
      while (lab.runs === shown && lab.replayTime < end && here()) await tickYield();
    }
    say(here() ? "Learning movie finished: the last picture is each learner's final car." : "Learning movie stopped.");
    count("journey_movies");
  }
  install("journeyMovie", movie, true);

  // ---- training ---------------------------------------------------------------------------------------------------------------------------
  function target() {
    const trackId = Number($("journeyCircuit").value), seed = number("journeySeed", 1, 999, true);
    if (!lab.journey || lab.journey.trackId !== trackId) lab.journey = freshJourney(trackId);
    return { j: lab.journey, track: trackMap.get(trackId), seed };
  }
  async function trainImitation(chosen = target()) {
    const { j, track, seed } = chosen, rounds = Number($("journeyRounds").value), entry = { seed, settings: { ...JR.defaults.imitation, rounds }, stages: [], source: "trained" };
    j.imitation = entry; j.selected.imitation = 0;
    const run = await JR.imitationJourney({ track, ctx, seed, rounds, yieldFn: tickYield, shouldCancel: () => lab.cancel,
      onStage: (stage, stages) => { entry.stages = stages; j.selected.imitation = stages.length - 1; refresh(); say(`Imitation: ${stageTitle(stage)} · ${grouped(stage.steps)} labelled steps · lap: ${lapOutcome(stage.lap)}`); } });
    if (run.diverged) throw new Error("Imitation training diverged (a weight became non-finite); the stages so far are kept.");
    count("journey_imitation_runs");
  }
  async function trainReinforcement(chosen = target()) {
    const { j, track, seed } = chosen, iterations = Number($("journeyIterations").value), reward = { ...RL.defaultReward, smoothness: Number($("journeyReward").value) };
    const entry = { seed, settings: { iterations, reward }, stages: [], history: [], source: "trained" };
    j.reinforcement = entry; j.selected.reinforcement = 0;
    const run = await JR.reinforcementJourney({ track, ctx, seed, iterations, reward, yieldFn: tickYield, shouldCancel: () => lab.cancel,
      onStage: (stage, stages) => { entry.stages = stages; j.selected.reinforcement = stages.length - 1; refresh(); },
      onIteration: info => {
        entry.history = info.history;
        say(`Reinforcement: iteration ${info.iteration} of ${iterations} · ${grouped(info.steps)} simulated steps (${JR.describeMinutes(info.steps * cfg.dt / 60)} of driving) · `
          + `${info.stats.failureRate === null ? "" : `${(100 * info.stats.failureRate).toFixed(0)}% of training drives crashed (noise on) · `}latest snapshot: ${lapOutcome(entry.stages.at(-1).lap)}`);
        if (info.iteration % 10 === 0) drawTraining();
      } });
    entry.history = run.history; entry.stopped = run.stopped; refresh();
    count("journey_reinforcement_runs");
  }
  function finish(text) { refresh(); say(text); }
  install("journeyImitation", async () => { await trainImitation(); finish(`Imitation finished.${lab.cancel ? " Cancelled: the stages so far are kept." : ""}`); }, true);
  install("journeyReinforcement", async () => { await trainReinforcement(); finish(`Reinforcement finished.${lab.cancel ? " Cancelled: the stages so far are kept." : ""}`); }, true);
  install("journeyBoth", async () => {
    // One circuit and seed for both learners, even if the menus are touched while they train.
    const chosen = target();
    await trainImitation(chosen);
    if (!lab.cancel) await trainReinforcement(chosen);
    finish(lab.cancel ? "Cancelled: the stages so far are kept." : "Both learners trained. Click a picture, or use the sliders and the replay button, to watch any stage drive its lap.");
  }, true);

  install("journeySaved", async () => {
    if (!reference) throw new Error("This page was built without the saved example.");
    const f = reference.featured, track = trackMap.get(f.trackId), j = freshJourney(f.trackId);
    for (const method of methods) {
      const stages = [];
      for (const saved of f[method].stages) {
        if (lab.cancel) throw new Error("Cancelled: nothing was loaded.");
        stages.push(JR.restoreStage(method, track, cfg, saved));
        if (stages.length % 3 === 0) { say(`Rebuilding the saved ${method} stages by driving each one (${stages.length} of ${f[method].stages.length})...`); await tickYield(); }
      }
      j[method] = { seed: f.seed, settings: f[method].settings, stages, history: f[method].history ?? [], source: "saved" };
      j.selected[method] = stages.length - 1;
    }
    lab.journey = j; $("journeyCircuit").value = String(f.trackId); $("journeySeed").value = String(f.seed); refresh();
    const stale = methods.flatMap(m => j[m].stages.filter(s => !s.reproduced).map(s => `${m} stage ${s.index}`));
    say(stale.length ? `Loaded, but ${stale.length} lap(s) differ from the numbers saved with the example (${stale.slice(0, 3).join(", ")}${stale.length > 3 ? "..." : ""}); this browser's arithmetic differs.`
      : `Loaded the saved seed-${f.seed} example. Every one of its ${methods.reduce((n, m) => n + j[m].stages.length, 0)} laps was driven again here and matches the saved numbers.`);
    count("journey_saved_loads");
  }, true);

  // ---- saved reference across circuits -------------------------------------------------------------------------------------------
  function renderReference() {
    if (!reference) return;
    const { settings, circuits, ablation } = reference, ids = settings.circuits, pctText = (k, n) => `${k}/${n}`;
    const first = s => s.firstLapSteps ? `${grouped(s.firstLapSteps.median)} / ${s.stableLapSteps ? grouped(s.stableLapSteps.median) : "never"}` : "never";
    const ratio = s => s.ratio ? `${s.ratio.mean.toFixed(2)}× [${s.ratio.best.toFixed(2)}–${s.ratio.worst.toFixed(2)}]` : "—";
    const finals = s => `${pctText(s.finalLaps, s.seeds)} · ${ratio(s)}`;
    $("journeyReferenceBox").hidden = false;
    $("journeyReferenceNote").textContent = `Saved headless run of the same code as the buttons above: ${ids.length} circuits × ${settings.seeds} seeds (1–${settings.seeds}) × 2 learners. Imitation: ${settings.imitation.rounds} DAgger rounds, `
      + `${settings.imitation.budget} labelled steps per round. Reinforcement: ${settings.reinforcement.iterations} PPO iterations of ${settings.reinforcement.envs * settings.reinforcement.rolloutSteps} steps `
      + `(${grouped(settings.reinforcement.iterations * settings.reinforcement.envs * settings.reinforcement.rolloutSteps)} steps in all), learning rate ${settings.reinforcement.learningRate}. `
      + "“First lap / stays lapping” are medians over seeds, in driving steps: the first stage whose car completes the lap, and the first stage from which it and every later stage complete it. "
      + "“Lost a lap” counts seeds whose car completed the lap at some stage but failed it at a later one: learning is not monotonic for either learner. "
      + "“Lap time ÷ expert” is for seeds whose final car completes the lap (mean, best–worst). A car flicks its steering when the angle moves more than "
      + `${settings.lap.flickerLimit}° per step on average over a drive of at least ${JR.FLICKER_MIN_SECONDS} s.`;
    table("journeyReferenceTable", ["Circuit", "Expert lap", "Imitation: first lap / stays lapping (steps)", "Imitation: seeds that lost a lap", "Imitation: final cars that lap · lap time ÷ expert",
      "Reinforcement: first lap / stays lapping (steps)", "Reinforcement: seeds that lost a lap", "Reinforcement: final cars that lap · lap time ÷ expert", "Reinforcement: final cars that flick the steering"],
    ids.map(id => { const c = circuits[id], i = c.imitation, r = c.reinforcement;
      return [`${id} ${c.name}`, `${c.expertSeconds.toFixed(0)} s`, first(i), pctText(i.lostLap, i.seeds), finals(i), first(r), pctText(r.lostLap, r.seeds), finals(r), pctText(r.flickering, r.seeds)]; }));
    const a = ablation, w = a.with, wo = a.without;
    $("journeyAblation").textContent = `Reward ablation on circuit ${a.trackId}, ${w.seeds} seeds each. With the steering-smoothness term (weight ${a.smoothness}), ${w.flickering}/${w.seeds} final cars flick the steering `
      + `(${w.flicker.toFixed(2)}° per step on average) and they lap in ${w.ratio ? w.ratio.mean.toFixed(2) : "—"}× the expert's time. Without it, ${wo.flickering}/${wo.seeds} flick it `
      + `(${wo.flicker.toFixed(2)}° per step) and they lap in ${wo.ratio ? wo.ratio.mean.toFixed(2) : "—"}× the expert's time. The penalty buys smooth steering with a little lap time.`;
  }

  // ---- start-up -------------------------------------------------------------------------------------------------------------------------------
  for (const id of validIds) { const option = document.createElement("option"); option.value = String(id); option.textContent = `${id} ${trackMap.get(id).name}`; $("journeyCircuit").append(option); }
  $("journeyCircuit").value = String(reference?.featured.trackId ?? validIds[0]);
  $("journeySaved").disabled = !reference;
  $("gotoJourney").onclick = () => $("tabJourney").click();
  renderReference(); refresh();
})();
