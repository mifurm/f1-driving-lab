"use strict";
// Modules 6-8, the guided experiments, the quiz and the results card. Loaded after app.js, whose top-level bindings it shares.
const average = values => values.reduce((a, b) => a + b, 0) / values.length;
const spread = values => values.length > 1 ? Math.sqrt(values.reduce((v, x) => v + (x - average(values)) ** 2, 0) / (values.length - 1)) : 0;
const fixed = (v, digits = 4) => v === null || v === undefined ? "n/a" : v.toFixed(digits);
const subsetOf = (rows, size) => size === "all" || rows.length <= size ? rows : new E.RNG(17).shuffle(rows.slice()).slice(0, size);
const rateInterval = summary => `${pct(summary.mean)} [${pct(summary.lo)}, ${pct(summary.hi)}]`;
const exploratoryNote = seeds => seeds < 10 ? " EXPLORATORY: fewer than 10 seeds." : "";

// ---- 6. training playground --------------------------------------------------------------------------------------------------
function playgroundSettings() {
  return { model: $("playModel").value, hidden1: Number($("hidden1").value), hidden2: Number($("hidden2").value), batch: Number($("playBatch").value),
    rate: number("playRate", .00001, .05), schedule: $("playSchedule").value, updates: number("playUpdates", 10, 800, true),
    decay: number("playDecay", 0, 1), clip: number("playClip", 0, 1000) || null, patience: number("playPatience", 0, 50, true), noise: number("playNoise", 0, 1),
    trainSize: $("trainSize").value === "all" ? "all" : Number($("trainSize").value), seeds: Number($("playSeeds").value), mask: $("studentMask").value };
}
function describeSettings(s) {
  const tail = `${s.trainSize} examples${s.noise ? `, label noise σ ${s.noise}` : ""}`;
  if (s.model === "ridge") return `Linear ridge, ${tail}`;
  if (s.model === "knn") return `Nearest neighbour (k = 5), ${tail}`;
  return `MLP ${s.hidden1}×${s.hidden2}, ${tail}, lr ${s.rate} (${s.schedule}), decay ${s.decay}${s.clip ? `, clip ${s.clip}` : ""}${s.patience ? `, patience ${s.patience}` : ""}, ${s.updates} updates`;
}
// Only the training labels are corrupted; validation and stress labels stay clean.
function corruptLabels(rows, sigma, seed) {
  if (!sigma) return rows;
  const noise = new E.RNG(900 + seed);
  return rows.map(r => ({ ...r, targets: [E.clip(r.targets[0] + sigma * noise.normal(), -1, 1), r.targets[1]] }));
}
async function trainConfig(settings, seed, train, validation) {
  const rows = corruptLabels(subsetOf(train, settings.trainSize), settings.noise, seed);
  let model, history = [], stopped = "completed", bestUpdate = null;
  if (settings.model === "ridge") model = T.RidgeModel.fit(rows, { lambda: .01, mask: settings.mask });
  else if (settings.model === "knn") model = T.KNNModel.fit(rows, { k: Math.min(5, rows.length), mask: settings.mask });
  else {
    const fitted = await T.fit({ seed, sizes: [18, settings.hidden1, settings.hidden2, 2], mask: settings.mask, train: rows, validation, stress: data.stressRows,
      updates: settings.updates, batchSize: settings.batch, learningRate: settings.rate, schedule: settings.schedule, weightDecay: settings.decay,
      maxGradNorm: settings.clip, patience: settings.patience, evalEvery: 5, lossSample: 512, yieldFn: tickYield, shouldCancel: () => lab.cancel });
    model = settings.patience > 0 ? fitted.best : fitted.last; history = fitted.history; stopped = fitted.stopped; bestUpdate = fitted.bestUpdate;
  }
  const failed = stopped === "diverged";
  const monitor = failed ? { k: 0, n: monitorSections.length, nominal: { k: 0, n: 8 }, recovery: { k: 0, n: 8 } } : T.evaluateSections(model, monitorSections, ctx);
  return { seed, model, history, stopped, bestUpdate, monitor, train: T.modelLoss(model, rows), validation: T.modelLoss(model, validation), stress: T.modelLoss(model, data.stressRows) };
}
async function runConfigs(list) {
  const { train, validation } = datasets(), results = [];
  for (const item of list) {
    const runs = [];
    for (let i = 0; i < item.settings.seeds; i++) {
      runs.push(await trainConfig(item.settings, 501 + i, train, validation));
      message(`${item.label}: seed ${i + 1}/${item.settings.seeds} trained and driven through the monitor sections.`); await tickYield();
      if (lab.cancel) throw new Error("Cancelled: no result was recorded.");
    }
    results.push({ label: item.label, settings: item.settings, runs });
  }
  return results;
}
function summarizePlayground(results) {
  const rng = new E.RNG(31);
  return results.map(r => {
    const col = key => r.runs.map(run => run[key]), rates = r.runs.map(run => run.monitor.k / run.monitor.n);
    return { label: r.label, seeds: r.runs.length, train: average(col("train")), trainSd: spread(col("train")), validation: average(col("validation")), validationSd: spread(col("validation")),
      stress: average(col("stress")), stressSd: spread(col("stress")), success: T.summarizeSeeds(rates, rng),
      nominal: average(r.runs.map(run => run.monitor.nominal.k)), recovery: average(r.runs.map(run => run.monitor.recovery.k)),
      bestUpdate: r.runs.some(run => run.bestUpdate !== null) ? average(r.runs.map(run => run.bestUpdate ?? 0)) : null,
      early: r.runs.filter(run => run.stopped === "early_stop").length, diverged: r.runs.filter(run => run.stopped === "diverged").length };
  });
}
function meanCurve(runs, key) {
  const length = Math.min(...runs.map(run => run.history.length)), points = [], band = [];
  for (let i = 0; i < length; i++) {
    const values = runs.map(run => run.history[i][key]), m = average(values), s = spread(values);
    points.push([runs[0].history[i].update, m]); band.push([runs[0].history[i].update, Math.max(m - s, m * .05), m + s]);
  }
  return { points, band };
}
function drawPlayground(results) {
  const mlp = results.filter(r => r.settings.model === "mlp" && r.runs[0].history.length);
  if (!mlp.length) { plot("playCanvas", { x: [0, 1], y: [1e-3, 1], log: true, title: "Baselines have no training curve: compare the table", xLabel: "" }); return; }
  const curves = mlp.map(r => ({ r, train: meanCurve(r.runs, "train"), validation: meanCurve(r.runs, "validation"), stress: meanCurve(r.runs, "stress") }));
  const all = curves.flatMap(c => [c.train, c.validation, c.stress].flatMap(k => k.points.map(p => p[1]))).filter(v => v > 0);
  const low = 10 ** Math.floor(Math.log10(Math.min(...all))), high = 10 ** Math.ceil(Math.log10(Math.max(...all))), last = Math.max(...curves.flatMap(c => c.validation.points.map(p => p[0])));
  const palette6 = ["#ffc56b", "#76a9ff", "#be9cff", "#ff8990", "#4de2c2", "#ecf2fa"];
  let series, bands;
  if (curves.length === 1) {
    const c = curves[0]; series = [["train", c.train], ["validation", c.validation], ["stress", c.stress]].map(([k, v]) => ({ color: palette[k], points: v.points }));
    bands = [["train", c.train], ["validation", c.validation], ["stress", c.stress]].map(([k, v]) => ({ color: palette[k], points: v.band }));
  } else {
    series = curves.flatMap((c, i) => [{ color: palette6[i % 6], points: c.validation.points }, { color: palette6[i % 6], points: c.train.points, dash: [4, 4], width: 1.5 }]); bands = [];
  }
  const geometry = plot("playCanvas", { x: [0, last], y: [low, high], log: true, series, bands, xLabel: "optimizer updates",
    title: curves.length === 1 ? "Mean over seeds: weighted action loss (log scale)" : "Validation loss (solid) and training loss (dashed) per configuration" });
  if (curves.length > 1) {
    const c = $("playCanvas").getContext("2d"); c.font = "12px system-ui";
    curves.forEach((entry, i) => { c.fillStyle = palette6[i % 6]; c.fillText(entry.r.label, geometry.m.l + 12 + i * 250, geometry.m.t + 14); });
  }
}
function renderPlayground(results, note = "") {
  const summary = summarizePlayground(results);
  table("playTable", ["Configuration", "Seeds", "Training loss", "Validation loss", "Stress loss", "Gap (validation − training)", "Safe monitor sections: mean [95% CI over seeds]", "Nominal / recovery (of 8)", "Best update", "Early stops", "Diverged"],
    summary.map(s => [s.label, s.seeds, `${fixed(s.train)} ± ${fixed(s.trainSd)}`, `${fixed(s.validation)} ± ${fixed(s.validationSd)}`, `${fixed(s.stress)} ± ${fixed(s.stressSd)}`,
      fixed(s.validation - s.train), s.seeds > 1 ? rateInterval(s.success) : pct(s.success.mean), `${s.nominal.toFixed(1)} / ${s.recovery.toFixed(1)}`,
      s.bestUpdate === null ? "n/a" : s.bestUpdate.toFixed(0), s.early, s.diverged]));
  drawPlayground(results); lab.playground = { results: results.map(r => ({ label: r.label, settings: r.settings, runs: r.runs.map(run => ({ seed: run.seed, train: run.train, validation: run.validation,
    stress: run.stress, monitor: { k: run.monitor.k, n: run.monitor.n, nominal: run.monitor.nominal, recovery: run.monitor.recovery }, stopped: run.stopped, bestUpdate: run.bestUpdate })) })), summary };
  $("playStatus").textContent = `${results.length} configuration(s) × ${results[0].settings.seeds} seeds.${exploratoryNote(results[0].settings.seeds)} ${note}`;
  count("playground_runs");
}
function enableBaselines(results) {
  for (const r of results) if (r.settings.model !== "mlp") {
    lab.baselines[r.settings.model] = r.runs[0].model; $("model").querySelector(`option[value="${r.settings.model}"]`).disabled = false;
  }
}
install("runPlayground", async () => {
  const settings = playgroundSettings(), results = await runConfigs([{ label: describeSettings(settings), settings }]);
  enableBaselines(results);
  let note = "";
  if (settings.model === "mlp") { lab.student = results[0].runs[0].model; lab.studentMetadata = { seed: 501, mask: settings.mask, learning_rate: settings.rate, planned_fit_updates: settings.updates };
    $("model").value = "student"; note = "The first seed's network is now selectable as your student model."; }
  else { $("model").value = settings.model; note = "The baseline is now selectable in the policy menu."; }
  renderPlayground(results, note);
}, true);
function setPlayground(values) { for (const [id, value] of Object.entries(values)) $(id).value = value; }
$("presetOverfit").onclick = () => { setPlayground({ playModel: "mlp", hidden1: "64", hidden2: "64", trainSize: "24", playNoise: .3, playRate: .004, playSchedule: "constant", playUpdates: 600, playDecay: 0, playClip: 0, playPatience: 0, playBatch: "16" });
  message("Overfitting preset: 24 examples with noisy steering labels, a large network and 600 updates. Press 'Train this configuration': training loss keeps falling while validation loss (clean labels) turns upward."); };
$("presetRegularize").onclick = () => { setPlayground({ playModel: "mlp", hidden1: "64", hidden2: "64", trainSize: "24", playNoise: .3, playRate: .004, playSchedule: "cosine", playUpdates: 600, playDecay: .3, playClip: 0, playPatience: 6, playBatch: "16" });
  message("Regularisation preset: the same noisy data with weight decay, a decaying rate and early stopping. Compare validation loss with the overfitting preset; then try more data in the sweep."); };
install("presetBaselines", async () => {
  const base = playgroundSettings(), list = [{ label: "Neural network", settings: { ...base, model: "mlp" } }, { label: "Linear ridge", settings: { ...base, model: "ridge" } },
    { label: "Nearest neighbour", settings: { ...base, model: "knn" } }];
  const results = await runConfigs(list); enableBaselines(results); renderPlayground(results, "Baselines share the same inputs and masks; the linear and nearest-neighbour models are now selectable in the policy menu.");
}, true);
install("runSweep", async () => {
  const base = playgroundSettings(), kind = $("sweepParameter").value;
  const options = {
    hidden: [[4, 4], [24, 16], [64, 64]].map(([a, b]) => [`${a}×${b} network`, { hidden1: a, hidden2: b }]),
    decay: [0, .05, .3].map(v => [`weight decay ${v}`, { decay: v }]), rate: [.0005, .002, .01].map(v => [`learning rate ${v}`, { rate: v }]),
    size: ["24", "64", "all"].map(v => [`${v} examples`, { trainSize: v === "all" ? "all" : Number(v) }]), updates: [50, 200, 600].map(v => [`${v} updates`, { updates: v }]),
    noise: [0, .15, .4].map(v => [`label noise σ ${v}`, { noise: v }]),
  }[kind];
  const results = await runConfigs(options.map(([label, override]) => ({ label, settings: { ...base, model: "mlp", ...override } })));
  renderPlayground(results, `Sweep of ${kind}; every other setting is held fixed.`);
}, true);

// ---- 7. data coverage and DAgger ---------------------------------------------------------------------------------------------------
const armIds = { bc_narrow: "armBcNarrow", bc_broad: "armBcBroad", bc_staged: "armBcStaged", dagger: "armDagger" };
const armNames = { bc_narrow: "Narrow expert demos", bc_broad: "Broad expert demos", bc_staged: "Staged expert control", dagger: "DAgger" };
const armShort = { bc_narrow: "Narrow", bc_broad: "Broad", bc_staged: "Staged", dagger: "DAgger" };
const armColors = { bc_narrow: "#ffc56b", bc_broad: "#76a9ff", bc_staged: "#be9cff", dagger: "#4de2c2" };
function drawDotPanels(id, panels) {
  const cv = $(id), c = cv.getContext("2d"), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  const half = W / panels.length;
  panels.forEach((panel, p) => {
    const m = { l: 60, r: 14, t: 28, b: 56 }, x0 = p * half, pw = half - m.l - m.r, ph = H - m.t - m.b, my = v => m.t + (1 - v) * ph; c.font = "12px system-ui";
    for (const v of [0, .25, .5, .75, 1]) { c.strokeStyle = "#2a3a50"; c.beginPath(); c.moveTo(x0 + m.l, my(v)); c.lineTo(x0 + half - m.r, my(v)); c.stroke(); c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(pct(v), x0 + m.l - 6, my(v) + 4); }
    c.textAlign = "left"; c.fillStyle = "#ecf2fa"; c.font = "bold 13px system-ui"; c.fillText(panel.title, x0 + m.l, 16); c.font = "12px system-ui";
    panel.rows.forEach((row, i) => {
      const x = x0 + m.l + (i + .5) * pw / panel.rows.length, s = row.summary;
      c.textAlign = "center"; c.fillStyle = "#acbbce"; c.fillText(row.label, x, H - 34); c.fillText(rateInterval(s), x, H - 18);
      c.strokeStyle = row.color; c.lineWidth = 3; c.beginPath(); c.moveTo(x - 34, my(s.mean)); c.lineTo(x + 34, my(s.mean)); c.stroke();
      c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, my(s.lo)); c.lineTo(x, my(s.hi)); c.moveTo(x - 8, my(s.lo)); c.lineTo(x + 8, my(s.lo)); c.moveTo(x - 8, my(s.hi)); c.lineTo(x + 8, my(s.hi)); c.stroke();
      c.fillStyle = row.color; s.values.forEach((v, k) => { c.beginPath(); c.arc(x - 27 + (k % 10) * 6, my(v), 3.4, 0, 2 * Math.PI); c.fill(); });
    });
    c.textAlign = "left";
  });
}
function drawCoverageMaps(entries, expertVisited) {
  const cv = $("coverageMaps"), c = cv.getContext("2d"), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  const panels = entries.length, width = W / panels, m = { l: 36, r: 8, t: 30, b: 22 }, xBins = 36, yBins = 24;
  entries.forEach((entry, p) => {
    const x0 = p * width, pw = width - m.l - m.r, ph = H - m.t - m.b;
    const grid = T.coverageGrid(entry.rows.map(r => T.featurePoint(cfg, r)), { xRange: [-9, 9], yRange: [-.6, .6], xBins, yBins });
    for (let j = 0; j < yBins; j++) for (let i = 0; i < xBins; i++) {
      const v = grid.counts[j * xBins + i]; if (!v) continue;
      c.fillStyle = `rgba(77,226,194,${.12 + .88 * Math.sqrt(v / grid.max)})`; c.fillRect(x0 + m.l + i * pw / xBins, m.t + (yBins - 1 - j) * ph / yBins, pw / xBins + .5, ph / yBins + .5);
    }
    const px = value => x0 + m.l + (value + 9) / 18 * pw, py = value => m.t + (1 - (value + .6) / 1.2) * ph;
    c.fillStyle = "rgba(236,242,250,.35)"; for (const s of expertVisited) { c.fillRect(px(s.lateral) - 1, py(s.heading) - 1, 2, 2); }
    for (const s of entry.visited) { c.fillStyle = s.supported ? "#76a9ff" : "#ff5b65"; c.fillRect(px(s.lateral) - 1.5, py(s.heading) - 1.5, 3, 3); }
    c.strokeStyle = "#334358"; c.strokeRect(x0 + m.l, m.t, pw, ph);
    c.fillStyle = "#ecf2fa"; c.font = "bold 12px system-ui"; c.textAlign = "left"; c.fillText(entry.name, x0 + m.l, 18);
    c.font = "11px system-ui"; c.fillStyle = "#acbbce"; c.textAlign = "center";
    for (const [value, label] of [[-9, "−9"], [0, "0"], [9, "+9"]]) c.fillText(label, px(value), H - 8);
    c.textAlign = "right"; for (const [value, label] of [[-.6, "−0.6"], [0, "0"], [.6, "+0.6"]]) c.fillText(label, x0 + m.l - 4, py(value) + 4);
    c.textAlign = "left";
  });
  c.fillStyle = "#acbbce"; c.font = "11px system-ui"; c.fillText("x: lane offset (m), y: heading error (rad)", 8, H - 8);
}
async function runCoverageExperiment() {
  const arms = Object.keys(armIds).filter(arm => $(armIds[arm]).checked);
  if (!arms.length) throw new Error("Choose at least one recipe.");
  const budget = Number($("covBudget").value), rounds = Number($("covRounds").value), updatesPerRound = Number($("covUpdates").value), seedCount = Number($("covSeeds").value);
  const run = await T.coverageExperiment({ data, ctx, arms, seedCount, budget, rounds, updatesPerRound, trainIds: trainingIds, validIds, holdoutIds, yieldFn: tickYield, shouldCancel: () => lab.cancel,
    onRun: ({ arm, seed, ev, hold }) => { $("covStatus").textContent = `${armNames[arm]}, seed ${seed}: ${ev.k}/${ev.n} training-track sections, ${hold.k}/${hold.n} legacy-holdout sections.`; } });
  const { results, summaries, comparisons, trainSections, holdSections, keep } = run, expertTrain = run.expert.train, expertHold = run.expert.hold;
  const rows = summaries.map(s => [armNames[s.arm], seedCount, rateInterval(s.nominal), rateInterval(s.recovery), rateInterval(s.hold), pct(s.supportReference), pct(s.supportVisited), s.failed, s.rejected, s.diverged]);
  rows.push(["Expert (reference ceiling)", "n/a", `${expertTrain.nominal.k}/${expertTrain.nominal.n}`, `${expertTrain.recovery.k}/${expertTrain.recovery.n}`, `${expertHold.k}/${expertHold.n}`, "100% by definition", "n/a", "n/a", "n/a", "n/a"]);
  table("coverageTable", ["Recipe", "Seeds", `Nominal sections (${trainSections.length / 2}): mean [95% bootstrap over seeds]`, `Recovery sections (${trainSections.length / 2})`,
    `Legacy holdout sections (${holdSections.length})`, "Occupancy of expert reference states", "Occupancy of the learner's visited states", "Failed collection episodes", "Rejected starts", "Diverged seeds"], rows);
  const pairTable = comparisons.map(({ a, b, overall: all, recovery: rec }) =>
    [`${armNames[a]} − ${armNames[b]}`, `${signed(all.mean)} [${signed(all.lo)}, ${signed(all.hi)}]`, `${signed(rec.mean)} [${signed(rec.lo)}, ${signed(rec.hi)}]`, `${all.wins}/${all.losses}/${all.ties}`, fmtP(all.p)]);
  table("coveragePairs", ["Paired by seed", "Overall success difference [95% bootstrap]", "Recovery-only difference", "Seeds better / worse / equal", "Sign test p"], pairTable);
  drawDotPanels("coverageDots", [{ title: "Nominal sections: success per seed", rows: arms.map(arm => ({ label: armShort[arm], color: armColors[arm], summary: summaries.find(s => s.arm === arm).nominal })) },
    { title: "Recovery sections: success per seed", rows: arms.map(arm => ({ label: armShort[arm], color: armColors[arm], summary: summaries.find(s => s.arm === arm).recovery })) }]);
  drawCoverageMaps(arms.map(arm => ({ name: armNames[arm], rows: keep[arm].rows, visited: keep[arm].visited })), expertTrain.visited.filter((_, k) => k % 6 === 0));
  lab.coverage = { settings: run.settings, results, summaries, comparisons, expert: { train: expertTrain.rows, holdout: expertHold.rows } };
  $("covStatus").textContent = `Done: ${arms.length} recipes × ${seedCount} seeds with equal budgets (${budget}×${trainingIds.length} labelled states per round, ${updatesPerRound} updates per round).${exploratoryNote(seedCount)}
Blue/red dots show where the first seed's learner drove: red = its data had no support there. White dots are the expert's own states.`;
  count("coverage_runs");
}
install("runCoverage", runCoverageExperiment, true);

function renderCoverageReference() {
  const reference = data.coverageReference;
  if (!reference) return;
  const { settings, arms, comparisons, expert } = reference;
  $("coverageReferenceBox").hidden = false;
  $("coverageReferenceNote").textContent = `Saved headless run of the same code as the button above: ${arms.length} recipes × ${settings.seeds} training seeds (${settings.seedBase}–${settings.seedBase + settings.seeds - 1}), `
    + `${settings.budget} labelled states per track per round, ${settings.rounds} rounds after the first, ${settings.updatesPerRound} updates per round. `
    + `Nominal and recovery sections are ${reference.sections.training / 2} each; seeds are the unit of replication.${exploratoryNote(settings.seeds)}`;
  const rows = arms.map(a => [a.name, rateInterval(a.nominal), rateInterval(a.recovery), rateInterval(a.hold), pct(a.supportReference), pct(a.supportVisited), a.failed, a.rejected, a.diverged]);
  rows.push(["Expert (reference ceiling)", `${expert.nominal.k}/${expert.nominal.n}`, `${expert.recovery.k}/${expert.recovery.n}`, `${expert.hold.k}/${expert.hold.n}`, "100% by definition", "n/a", "n/a", "n/a", "n/a"]);
  table("coverageReferenceTable", ["Recipe", `Nominal sections: mean [95% bootstrap over ${settings.seeds} seeds]`, "Recovery sections", `Legacy holdout sections (${reference.sections.holdout})`,
    "Occupancy of expert reference states", "Occupancy of the learner's visited states", "Failed collection episodes", "Rejected starts", "Diverged seeds"], rows);
  table("coverageReferencePairs", ["Paired by seed", "Overall success difference [95% bootstrap]", "Recovery-only difference", "Seeds better / worse / equal", "Sign test p"],
    comparisons.map(c => [`${armNames[c.a]} − ${armNames[c.b]}`, `${signed(c.overall.mean)} [${signed(c.overall.lo)}, ${signed(c.overall.hi)}]`, `${signed(c.recovery.mean)} [${signed(c.recovery.lo)}, ${signed(c.recovery.hi)}]`,
      `${c.overall.wins}/${c.overall.losses}/${c.overall.ties}`, fmtP(c.overall.p)]));
  drawDotPanels("coverageReferenceDots", [{ title: "Nominal sections: success per seed", rows: arms.map(a => ({ label: armShort[a.arm], color: armColors[a.arm], summary: a.nominal })) },
    { title: "Recovery sections: success per seed", rows: arms.map(a => ({ label: armShort[a.arm], color: armColors[a.arm], summary: a.recovery })) }]);
}
renderCoverageReference();

// step-by-step DAgger
const dagger = () => { if (!lab.dagger) throw new Error("Start a DAgger session first."); return lab.dagger; };
function daggerMetrics(session, extra = {}) {
  const monitor = T.evaluateSections(session.net, monitorSections, ctx);
  return { round: session.round, size: session.rows.length, validation: T.modelLoss(session.net, session.validation), stress: T.modelLoss(session.net, data.stressRows),
    nominal: monitor.nominal, recovery: monitor.recovery, ...extra };
}
function renderDagger(session) {
  table("daggerTable", ["Round", "Labelled states", "New states: failed episodes / rejected starts", "Loss on the learner's own new states", "Loss on fixed expert validation", "Stress loss", "Nominal (of 8)", "Recovery (of 8)"],
    session.log.map(r => [r.round, r.size, r.round === 0 ? "n/a" : `${r.failed} / ${r.rejected}`, r.newStateLoss === null ? "n/a" : fixed(r.newStateLoss), fixed(r.validation), fixed(r.stress), `${r.nominal.k}`, `${r.recovery.k}`]));
}
// Trains a clone on `rows` and returns it. Cancelling or diverging throws before anything in the session changes, so a DAgger session is never left half-updated.
async function daggerTrain(session, rows, updates) {
  const fitted = await T.fit({ seed: session.seed, mask: "full", train: rows, validation: [], updates, learningRate: .001, initial: session.net.clone(), evalEvery: updates, lossSample: 256,
    rngSeed: session.seed + 100 + session.net.updates, yieldFn: tickYield, shouldCancel: () => lab.cancel });
  if (lab.cancel || fitted.stopped === "cancelled") throw new Error("Cancelled: the DAgger session was not changed.");
  if (fitted.stopped === "diverged") throw new Error(`Training diverged at update ${fitted.divergence.update}: ${fitted.divergence.message}. The DAgger session was not changed.`);
  return fitted.net;
}
install("startDagger", async () => {
  const budget = Number($("covBudget").value), updates = Number($("covUpdates").value), seed = 7, rng = new E.RNG(2024);
  const validation = data.starter.filter(r => r.split === "validation");
  const session = { seed, budget, updates, rng, net: new E.Network(seed, [18, 24, 16, 2], "full"), rows: [], round: 0, episode: 0, validation, log: [], pending: null };
  let failed = 0, rejected = 0;
  for (const id of trainingIds) {
    const part = T.collect({ track: trackMap.get(id), cfg, driver: "expert", rng, steps: budget, spec: gen.starts.narrow, prefix: `d0`, firstEpisode: session.episode, source: "expert demonstration (round 0)" });
    session.episode += part.episodes; session.rows.push(...part.rows); failed += part.failures; rejected += part.rejected; await tickYield();
    if (lab.cancel) throw new Error("Cancelled: no DAgger session was started.");
  }
  session.net = await daggerTrain(session, session.rows, updates);
  session.log.push(daggerMetrics(session, { failed, rejected, newStateLoss: null })); lab.dagger = session; renderDagger(session);
  $("daggerRollout").disabled = false; $("daggerTrain").disabled = true; $("daggerReplay").disabled = false;
  $("daggerStatus").textContent = `Round 0: ${session.rows.length} expert-labelled states, ${updates} updates. Next: let the learner drive from typical starts while the expert labels the states it visits.`;
  count("dagger_sessions");
}, true);
install("daggerRollout", async () => {
  const session = dagger();
  // A rollout consumes the session's random stream and episode numbers. A cancelled or failed one gives them back, so a session replays identically whatever was pressed.
  const saved = { rng: session.rng.state, episode: session.episode };
  let failed = 0, rejected = 0; const added = [];
  try {
    for (const id of trainingIds) {
      const part = T.collect({ track: trackMap.get(id), cfg, driver: session.net, rng: session.rng, steps: session.budget, spec: gen.starts.typical, prefix: `d${session.round + 1}`,
        firstEpisode: session.episode, source: `learner rollout, expert label (round ${session.round + 1})` });
      session.episode += part.episodes; added.push(...part.rows); failed += part.failures; rejected += part.rejected; await tickYield();
      if (lab.cancel) throw new Error("Cancelled: nothing was added to the DAgger session.");
    }
  } catch (error) { session.rng.state = saved.rng; session.episode = saved.episode; throw error; }
  const newStateLoss = T.modelLoss(session.net, added), validationLoss = T.modelLoss(session.net, session.validation);
  session.pending = { rows: added, failed, rejected, newStateLoss };
  const run = E.simulate(currentTrack(), cfg, session.net, start(), { record: true }), divergence = T.firstDivergence(T.actionErrors(run.trace));
  showRuns([{ ...run, name: `Learner after round ${session.round}: ${run.metric.end_reason}`, track: currentTrack(),
    marker: divergence ? { index: divergence.index, label: `first disagreement ${divergence.time.toFixed(1)} s` } : null }], false);
  $("daggerStatus").textContent = `The learner visited ${added.length} states (${failed} failed episodes, ${rejected} rejected starts). Its loss on those states is ${fixed(newStateLoss)}, against ${fixed(validationLoss)} on the fixed expert validation episodes:
${newStateLoss > validationLoss * 1.5 ? "the states it visits are harder than the expert's own, which is covariate shift." : "close to the expert distribution on these starts."}`;
  $("daggerRollout").disabled = true; $("daggerTrain").disabled = false;
}, true);
install("daggerTrain", async () => {
  const session = dagger();
  if (!session.pending) throw new Error("Let the learner drive first.");
  const rows = [...session.rows, ...session.pending.rows], net = await daggerTrain(session, rows, session.updates);
  session.rows = rows; session.net = net; session.round++;
  const { failed, rejected, newStateLoss } = session.pending; session.pending = null;
  session.log.push(daggerMetrics(session, { failed, rejected, newStateLoss })); renderDagger(session);
  $("daggerRollout").disabled = session.round >= 10; $("daggerTrain").disabled = true;
  $("daggerStatus").textContent = `Round ${session.round}: ${session.rows.length} labelled states in the aggregate. Validation uses fixed expert episodes, so its curve is comparable between rounds.`;
}, true);
install("daggerReplay", () => {
  const session = dagger(), run = E.simulate(currentTrack(), cfg, session.net, start(), { record: true });
  showRuns([{ ...run, name: `DAgger learner after round ${session.round}: ${run.metric.end_reason}`, track: currentTrack() }]);
});

// ---- 8. diagnose a failure -----------------------------------------------------------------------------------------------------------
const overlayGeometry = { l: 66, r: 18 };
function drawOverlay() {
  const d = lab.diagnosis, cv = $("overlayCanvas"), c = cv.getContext("2d"), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  if (!d) { c.fillStyle = "#acbbce"; c.font = "14px system-ui"; c.fillText("Run and record a drive to compare the policy with the expert.", 70, 60); return; }
  const rows = d.trace.filter(r => r.action), maxTime = Math.max(d.trace.at(-1).time, .1), pw = W - overlayGeometry.l - overlayGeometry.r;
  const panels = [{ name: "Steering (degrees)", y: [-28.7, 28.7], key: 0, scale: v => v * cfg.max_steer * 180 / Math.PI, top: 22, height: 120 },
    { name: "Target speed (km/h)", y: [0, 130], key: 1, scale: v => (v + 1) * cfg.max_speed / 2 * 3.6, top: 176, height: 100 }];
  const mx = t => overlayGeometry.l + t / maxTime * pw;
  c.font = "12px system-ui";
  for (const panel of panels) {
    const my = v => panel.top + (1 - (v - panel.y[0]) / (panel.y[1] - panel.y[0])) * panel.height;
    c.strokeStyle = "#334358"; c.strokeRect(overlayGeometry.l, panel.top, pw, panel.height);
    c.fillStyle = "#ecf2fa"; c.textAlign = "left"; c.fillText(panel.name, overlayGeometry.l, panel.top - 6);
    for (const v of [panel.y[0], (panel.y[0] + panel.y[1]) / 2, panel.y[1]]) { c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(v.toFixed(0), overlayGeometry.l - 6, my(v) + 4); }
    if (d.divergence !== null) {
      const flagged = d.errors.filter(e => e.error > d.threshold); c.fillStyle = "rgba(255,197,107,.18)";
      for (const e of flagged) c.fillRect(mx(e.time), panel.top, Math.max(2, pw / Math.max(d.trace.length, 1)), panel.height);
    }
    for (const [name, key, color, dash] of [["expert", "expert", "#ecf2fa", [5, 4]], ["policy", "action", "#4de2c2", []]]) {
      c.strokeStyle = color; c.lineWidth = 2; c.setLineDash(dash); c.beginPath();
      rows.forEach((r, i) => { const x = mx(r.time), y = my(panel.scale(r[key][panel.key])); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); c.setLineDash([]);
    }
  }
  c.textAlign = "left"; c.fillStyle = "#4de2c2"; c.fillText("policy (solid)", W - 330, 14); c.fillStyle = "#ecf2fa"; c.fillText("expert at the true state (dashed)", W - 230, 14);
  if (d.divergence) { c.strokeStyle = "#ffc56b"; c.lineWidth = 2; c.beginPath(); c.moveTo(mx(d.divergence.time), 20); c.lineTo(mx(d.divergence.time), 276); c.stroke(); c.fillStyle = "#ffc56b"; c.fillText("first disagreement", mx(d.divergence.time) + 4, 292); }
  c.strokeStyle = "#ff5b65"; c.lineWidth = 2; c.beginPath(); c.moveTo(mx(maxTime), 20); c.lineTo(mx(maxTime), 276); c.stroke();
  if (d.recovery) for (const f of d.recovery.forks) { c.fillStyle = f.success ? "#4de2c2" : "#ff5b65"; c.beginPath(); c.arc(mx(f.time), 305, 5, 0, 2 * Math.PI); c.fill(); }
  c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(d.recovery ? "expert recovery: green = expert finishes from here, red = it cannot" : "time (s)", W - overlayGeometry.r, H - 4);
  c.strokeStyle = "#76a9ff"; c.lineWidth = 1.5; c.beginPath(); c.moveTo(mx(Math.min(lab.replayTime, maxTime)), 20); c.lineTo(mx(Math.min(lab.replayTime, maxTime)), 276); c.stroke();
}
window.drawDiagnosisCursor = () => { if (lab.module === "diagnose" && lab.diagnosis) drawOverlay(); };
function applyDiagnosis() {
  const d = lab.diagnosis; if (!d) return;
  d.threshold = Number($("divergeThreshold").value) / 1000; $("divergeLabel").textContent = `Threshold ${d.threshold.toFixed(3)}: steering error counts four times speed error; must persist for 3 steps.`;
  d.divergence = T.firstDivergence(d.errors, { threshold: d.threshold });
  const marker = d.divergence ? { index: d.divergence.index, label: `first disagreement ${d.divergence.time.toFixed(1)} s` } : null;
  const peak = d.errors.reduce((best, e) => e.error > best.error ? e : best, { error: 0, time: 0 }), end = d.result.metric;
  showRuns([{ ...d.result, name: `${d.modelName}: ${end.end_reason}`, track: d.track, marker }], false);
  $("diagnoseSummary").textContent = `Outcome: ${end.end_reason} after ${d.trace.at(-1).time.toFixed(1)} s (${end.success ? "safe centered exit" : "not a safe exit"}).
First sustained disagreement with the expert: ${d.divergence ? `${d.divergence.time.toFixed(1)} s (weighted error ${d.divergence.error.toFixed(3)})` : "none above the threshold"}. Peak disagreement ${peak.error.toFixed(3)} at ${peak.time.toFixed(1)} s.
${d.recovery ? recoverySummary(d.recovery) : "Run the expert-recovery test to see whether the expert could still have finished from the states along the way."}
A disagreement is not automatically a cause: small differences are everywhere, and what matters is where they compound.`;
  drawOverlay();
}
function recoverySummary(recovery) {
  const last = recovery.lastRecoverable, first = recovery.firstFailure;
  return `Expert recovery: ${first ? `the expert could still finish up to ${last ? last.time.toFixed(1) : "no sampled time"} s and failed from ${first.time.toFixed(1)} s` : "the expert could finish from every sampled state"}; outcomes ${recovery.monotonic ? "switch only once" : "switch more than once"}. This is a counterfactual about states, not proof of cause.`;
}
function recordDiagnosis() {
  const model = selectedModel(), track = currentTrack(), scenario = start();
  const result = E.simulate(track, cfg, model, scenario, { record: true });
  lab.diagnosis = { result, trace: result.trace, errors: T.actionErrors(result.trace), track, scenario, modelName: $("model").selectedOptions[0].textContent, recovery: null };
  table("recoveryTable", ["Time (s)", "Expert finishes from this state?", "End reason", "Minimum clearance (m)"], []);
  applyDiagnosis(); count("diagnoses");
}
install("diagnoseRun", recordDiagnosis);
$("divergeThreshold").oninput = () => { if (lab.diagnosis) applyDiagnosis(); };
$("overlayCanvas").onclick = event => {
  if (!lab.diagnosis) return;
  const cv = $("overlayCanvas"), rect = cv.getBoundingClientRect(), x = (event.clientX - rect.left) / rect.width * cv.width, maxTime = lab.diagnosis.trace.at(-1).time;
  lab.replayTime = E.clip((x - overlayGeometry.l) / (cv.width - overlayGeometry.l - overlayGeometry.r) * maxTime, 0, maxTime); lab.replayPlaying = false; draw();
};
install("recoveryRun", async () => {
  const d = lab.diagnosis; if (!d) throw new Error("Run and record a drive first.");
  d.recovery = await T.expertRecovery({ track: d.track, cfg, trace: d.trace, yieldFn: tickYield });
  table("recoveryTable", ["Time (s)", "Expert finishes from this state?", "End reason", "Minimum clearance (m)"],
    d.recovery.forks.map(f => [f.time.toFixed(1), f.success ? "yes" : "no", f.end_reason, f.min_clearance_m.toFixed(2)]));
  applyDiagnosis(); count("recovery_tests");
}, true);
function drawImportance(results) {
  const cv = $("importanceCanvas"), c = cv.getContext("2d"), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  const groups = results.filter(r => r.kind === "group" || r.derived), top = Math.max(...groups.map(g => g.mean + (g.sd ?? 0)), 1e-9), m = { l: 210, r: 24, t: 28, b: 12 }, rowH = (H - m.t - m.b) / groups.length;
  c.font = "bold 13px system-ui"; c.fillStyle = "#ecf2fa"; c.fillText("Increase in weighted action loss when the group is scrambled (mean ± SD over 5 shuffles)", 12, 16); c.font = "12px system-ui";
  groups.forEach((g, i) => {
    const y = m.t + i * rowH, w = (W - m.l - m.r) * Math.max(g.mean, 0) / top;
    c.fillStyle = "#acbbce"; c.textAlign = "right"; c.fillText(g.name, m.l - 8, y + rowH / 2 + 4); c.fillStyle = g.mean > 0 ? "#4de2c2" : "#ff8990"; c.fillRect(m.l, y + 5, w, rowH - 10);
    if (g.sd !== null) { c.strokeStyle = "#ecf2fa"; c.beginPath(); const x1 = m.l + (W - m.l - m.r) * Math.max(g.mean - g.sd, 0) / top, x2 = m.l + (W - m.l - m.r) * (g.mean + g.sd) / top; c.moveTo(x1, y + rowH / 2); c.lineTo(x2, y + rowH / 2); c.stroke(); }
    c.textAlign = "left"; c.fillStyle = "#ecf2fa"; c.fillText(g.mean.toExponential(2), m.l + w + 6, y + rowH / 2 + 4);
  });
  c.textAlign = "left";
}
install("importanceRun", async () => {
  const model = selectedModel(), set = $("importanceSet").value;
  const rows = set === "stress" ? data.stressRows : lab.rows.filter(r => r.split === "validation").length ? lab.rows.filter(r => r.split === "validation") : data.starter.filter(r => r.split === "validation");
  const results = T.permutationImportance(model, rows, new E.RNG(3), { repeats: 5, names: data.features, cfg, maxRows: 600 });
  await tickYield(); lab.importance = { set, rows: rows.length, results: results.map(({ name, kind, mean, sd, indices }) => ({ name, kind, mean, sd, indices })) };
  drawImportance(results);
  table("importanceTable", ["Input group / input", "Kind", "Increase in loss (mean ± SD over shuffles)", "Inputs"],
    results.map(r => [r.name, r.kind === "group" ? "group" : r.kind, `${r.mean.toExponential(2)} ± ${r.sd === null ? "n/a" : r.sd.toExponential(1)}`, r.indices.map(i => data.features[i]).join(", ")]));
  message(`Permutation importance on ${rows.length} ${set} rows. Baseline weighted loss ${results[0].baseline.toFixed(5)}. Correlated inputs share credit; this is not causal.`); count("importance_runs");
});

// ---- 11. worksheets, quiz and results card --------------------------------------------------------------------------------------------------
const worksheets = kit.worksheets;
const factorDefaults = { noise: "0,0.5,1", delay: "0,3,6", grip: "1,0.8,0.65" };
const numericList = text => text.split(",").map(v => Number(v.trim()));
function controlHTML(label, id, html) { const wrap = document.createElement("div"); const l = document.createElement("label"); l.htmlFor = id; l.textContent = label; wrap.append(l); wrap.insertAdjacentHTML("beforeend", html); return wrap; }
function buildWorksheetControls() {
  const ws = worksheets[Number($("worksheet").value)], host = $("worksheetControls"); host.replaceChildren();
  const seeds = (id, def = 5) => controlHTML("Seeds / replicates", id, `<select id="${id}"><option value="3"${def === 3 ? " selected" : ""}>3 (exploratory)</option><option value="5"${def === 5 ? " selected" : ""}>5 (exploratory)</option><option value="10"${def === 10 ? " selected" : ""}>10</option></select>`);
  const pool = id => controlHTML("Section pool", id, `<select id="${id}"><option value="monitor">16 fixed monitor sections</option><option value="all">36 fixed sections</option><option value="selected">Selected start only</option><option value="typical">24 independent typical starts</option><option value="stress">24 independent stress starts</option></select>`);
  const factor = ws.factors.length ? controlHTML("Change exactly one factor", "wsFactor", `<select id="wsFactor">${ws.factors.map(f => `<option value="${f}">${f}</option>`).join("")}</select>`) : null;
  const values = defaults => controlHTML("Three values, comma-separated", "wsValues", `<input id="wsValues" value="${defaults}">`);
  switch (ws.runner) {
    case "inputSweep": host.append(factor, values("20,40,60")); break;
    case "labelNoise": host.append(values("0,0.15,0.4"), seeds("wsSeeds", 5)); break;
    case "controllers": host.append(pool("wsPool"), seeds("wsSeeds", 10)); break;
    case "masks": case "capacity": host.append(seeds("wsSeeds", 5)); break;
    case "shift": case "safety": host.append(factor, values(factorDefaults.noise), pool("wsPool"), seeds("wsSeeds", 10)); break;
    case "gradientCheck": host.append(controlHTML("Parameters to check", "wsParams", `<input id="wsParams" type="number" value="6" min="2" max="20">`)); break;
    default: break;
  }
  if ($("wsFactor") && ["shift", "safety"].includes(ws.runner)) $("wsFactor").onchange = () => { $("wsValues").value = factorDefaults[$("wsFactor").value]; };
  if ($("wsFactor") && ws.runner === "inputSweep") $("wsFactor").onchange = () => { $("wsValues").value = { speed: "20,40,60", offset: "-3,0,3", heading: "-10,0,10" }[$("wsFactor").value]; };
  $("worksheetPrompt").textContent = ws.prompt;
  $("worksheetObjectives").textContent = `Objectives: ${ws.objectives.map(id => `${id} ${kit.objectives.find(o => o.id === id).text}`).join(" · ")}`;
}
function distinctValues(raw, factor) {
  const values = numericList(raw);
  if (values.length !== 3 || values.some(v => !Number.isFinite(v)) || new Set(values).size !== 3) throw new Error("Provide three distinct numeric values.");
  const bounds = { noise: [0, 3], delay: [0, 20], grip: [.2, 1.2], label_noise: [0, 1], speed: [10, 100], offset: [-7, 7], heading: [-30, 30] }[factor];
  if (bounds && values.some(v => v < bounds[0] || v > bounds[1] || (factor === "delay" && !Number.isInteger(v)))) throw new Error("Factor values are outside the simulator bounds.");
  return values;
}
const wsRunners = {
  async inputSweep(ws) {
    const factor = $("wsFactor").value, values = distinctValues($("wsValues").value, factor), model = selectedModel(), rows = [];
    for (const v of values) {
      const s = start(); if (factor === "speed") s.speed = v / 3.6; if (factor === "offset") s.lateral = v; if (factor === "heading") s.heading_error = v * Math.PI / 180;
      const env = new E.Environment(currentTrack(), cfg); env.reset(s);
      const command = model.predict(env.observe().features); env.step(command);
      rows.push([`${factor} = ${v}`, `${(command[0] * cfg.max_steer * 180 / Math.PI).toFixed(2)}°`, `${((command[1] + 1) * cfg.max_speed / 2 * 3.6).toFixed(1)} km/h`, `${(env.state.brake * 100).toFixed(0)}%`, `${(env.state.throttle * 100).toFixed(0)}%`]);
    }
    return { headings: ["Input", "Steering output", "Target speed output", "Brake in first interval", "Throttle in first interval"], rows, settings: { factor, values }, raw: rows };
  },
  async labelNoise() {
    const values = distinctValues($("wsValues").value, "label_noise"), seeds = Number($("wsSeeds").value), { train, validation } = datasets(), raw = [], rows = [], rng = new E.RNG(61);
    for (const sigma of values) {
      const rates = [], vals = [];
      for (let i = 0; i < seeds; i++) {
        const corrupted = corruptLabels(train, sigma, i);
        const fitted = await T.fit({ seed: 701 + i, train: corrupted, validation, updates: 120, learningRate: .002, evalEvery: 12, yieldFn: tickYield, shouldCancel: () => lab.cancel });
        if (lab.cancel) throw new Error("Cancelled: no result was recorded.");
        const ev = T.evaluateSections(fitted.last, monitorSections, ctx); rates.push(ev.k / ev.n); vals.push(T.modelLoss(fitted.last, validation));
        raw.push({ sigma, seed: 701 + i, validation_loss_clean_labels: vals.at(-1), safe_sections: ev.k, sections: ev.n });
      }
      rows.push([`label noise σ = ${sigma}`, `${average(vals).toFixed(4)} ± ${spread(vals).toFixed(4)}`, rateInterval(T.summarizeSeeds(rates, rng))]);
    }
    return { headings: ["Corrupted steering labels", "Validation loss on clean labels", "Safe monitor sections: mean [95% CI over seeds]"], rows, settings: { values, seeds }, raw };
  },
  async gradientCheck() {
    const n = Number($("wsParams").value), { train } = datasets(), batch = train.slice(0, 8), rng = new E.RNG(5), net = new E.Network(21, [18, 24, 16, 2], "full"), rows = [], raw = [];
    const gradients = net.gradients(batch), after = net.clone(); after.trainBatch(batch, .001);
    for (let k = 0; k < n; k++) {
      const layer = rng.index(3), i = rng.index(net.weights[layer].length), j = rng.index(net.weights[layer][0].length), eps = 1e-5, original = net.weights[layer][i][j];
      net.weights[layer][i][j] = original + eps; const plus = net.loss(batch); net.weights[layer][i][j] = original - eps; const minus = net.loss(batch); net.weights[layer][i][j] = original;
      const numeric = (plus - minus) / (2 * eps), g = gradients.gw[layer][i][j], adam = after.weights[layer][i][j] - original;
      rows.push([`W${layer}[${i},${j}]`, g.toExponential(4), numeric.toExponential(4), (Math.abs(g - numeric) / Math.max(Math.abs(g), 1e-12)).toExponential(2), (-.001 * g).toExponential(3), adam.toExponential(3)]);
      raw.push({ parameter: rows.at(-1)[0], backprop: g, finite_difference: numeric, sgd_step: -.001 * g, adam_first_step: adam });
    }
    return { headings: ["Parameter", "Backprop gradient", "Finite difference", "Relative difference", "Plain gradient step (−lr·g)", "Adam first step"], rows, settings: { parameters: n, learning_rate: .001 }, raw };
  },
  async controllers() {
    const model = selectedModel(), clone = model.clone ? model.clone() : model;
    const result = await runExperiment({ conditions: [{ name: "Rules", policy: "rules" }, { name: "Expert", policy: "expert" }, { name: "Neural", policy: clone }], model: clone, poolKey: $("wsPool").value, seeds: Number($("wsSeeds").value), label: "worksheet controllers" });
    return { headings: conditionHeadings(result), rows: conditionRows(result), settings: { pool: result.pool, seeds: result.seeds }, raw: result.rows };
  },
  async masks() {
    const { train, validation } = datasets(), seeds = Number($("wsSeeds").value), masks = ["full", "no_heading", "no_margin"], rates = Object.fromEntries(masks.map(m => [m, []])), rng = new E.RNG(71);
    for (const mask of masks) for (let i = 0; i < seeds; i++) {
      const fitted = await T.fit({ seed: 301 + i, mask, train, validation, updates: 120, learningRate: .002, evalEvery: 8, yieldFn: tickYield, shouldCancel: () => lab.cancel });
      if (lab.cancel) throw new Error("Cancelled: no result was recorded.");
      const ev = T.evaluateSections(fitted.best, monitorSections, ctx); rates[mask].push(ev.k / ev.n);
    }
    const summary = summarizeAblation(rates, rng);
    return { headings: ["Retrained variant", "Success mean [95% over seeds]", "Per-seed range", "Paired vs all inputs"], rows: summary.map(r => [r.mask, rateInterval(r.summary), `${pct(r.summary.min)} to ${pct(r.summary.max)}`,
      r.paired ? `${signed(r.paired.mean)} [${signed(r.paired.lo)}, ${signed(r.paired.hi)}]; better/worse/equal ${r.paired.wins}/${r.paired.losses}/${r.paired.ties}` : "baseline"]), settings: { masks, seeds }, raw: rates };
  },
  async capacity() {
    const seeds = Number($("wsSeeds").value), base = { ...playgroundSettings(), model: "mlp", trainSize: 24, noise: .3, updates: 600, rate: .004, seeds, decay: 0, patience: 0, clip: null, schedule: "constant", batch: 16 };
    const results = await runConfigs([[4, 4], [24, 16], [64, 64]].map(([a, b]) => ({ label: `${a}×${b} network, 24 noisy examples`, settings: { ...base, hidden1: a, hidden2: b } })));
    renderPlayground(results, "Capacity worksheet.");
    const summary = summarizePlayground(results);
    return { headings: ["Network", "Training loss", "Validation loss (clean labels)", "Gap", "Safe monitor sections [95% over seeds]"], rows: summary.map(s => [s.label, fixed(s.train), fixed(s.validation), fixed(s.validation - s.train), rateInterval(s.success)]), settings: { seeds, examples: 24, label_noise: .3, updates: 600 }, raw: summary };
  },
  async coverage() {
    await runCoverageExperiment();
    return { headings: ["Recipe", "Nominal sections", "Recovery sections", "Occupancy of expert reference states"], rows: lab.coverage.summaries.map(s => [armNames[s.arm], rateInterval(s.nominal), rateInterval(s.recovery), pct(s.supportReference)]), settings: lab.coverage.settings, raw: lab.coverage.results };
  },
  async diagnose() {
    recordDiagnosis();
    if (!lab.diagnosis) throw new Error("Could not record a drive.");
    lab.diagnosis.recovery = await T.expertRecovery({ track: lab.diagnosis.track, cfg, trace: lab.diagnosis.trace, yieldFn: tickYield }); applyDiagnosis();
    const d = lab.diagnosis;
    return { headings: ["Measure", "Value"], rows: [["Outcome", d.result.metric.end_reason], ["First sustained disagreement (s)", d.divergence ? d.divergence.time.toFixed(1) : "none"],
      ["Last time the expert could still finish (s)", d.recovery.lastRecoverable ? d.recovery.lastRecoverable.time.toFixed(1) : "none"], ["Outcomes switch only once", d.recovery.monotonic ? "yes" : "no"]],
      settings: { track: d.track.name, scenario: d.scenario.name }, raw: d.recovery.forks };
  },
  async integrity() {
    if (!lab.frozenAudit) throw new Error("Freeze and reveal in module 9 first.");
    const a = lab.frozenAudit;
    return { headings: ["Record", "Value"], rows: [["Frozen model fingerprint", a.fingerprint], ["Test already exposed before this reveal", a.alreadyExposed ? "yes (results are exploratory)" : "no"], ["Test labels exposed (leakage lesson)", lab.testLabelsExposed ? "yes" : "no"],
      ["Sections evaluated", a.sections.length]], settings: { fingerprint: a.fingerprint }, raw: a.sections };
  },
  async shift(ws) { return shiftRunner(ws, false); },
  async safety(ws) { return shiftRunner(ws, true); },
};
async function shiftRunner(ws, safety) {
  const factor = $("wsFactor").value, values = distinctValues($("wsValues").value, factor), model = selectedModel(), clone = model.clone ? model.clone() : model;
  const conditions = values.map(value => ({ name: `${factor}=${value}`, noise: 0, delay: 0, grip: 1, [factor]: value }));
  const result = await runExperiment({ conditions, model: clone, poolKey: $("wsPool").value, seeds: Number($("wsSeeds").value), label: `worksheet ${factor}` });
  showRuns(representativeRuns(result, clone));
  if (!safety) return { headings: conditionHeadings(result), rows: conditionRows(result), settings: { factor, values, pool: result.pool, seeds: result.seeds }, raw: result.rows };
  return { headings: ["Condition", "Safe / trials", "Wall contacts", "Handling-limit failures", "Mean minimum wall clearance (m)", "Mean progress"],
    rows: result.summary.map(s => [s.condition, `${s.k}/${s.trials}`, s.contacts, s.handling, s.clearance.toFixed(2), pct(s.progress)]), settings: { factor, values, pool: result.pool, seeds: result.seeds }, raw: result.rows };
}
install("runWorksheet", async () => {
  if (!$("hypothesis").value.trim()) throw new Error("Write your hypothesis before running the experiment.");
  const ws = worksheets[Number($("worksheet").value)], result = await wsRunners[ws.runner](ws);
  table("worksheetResults", result.headings, result.rows);
  const fingerprint = (() => { try { return E.fingerprint(selectedModel()); } catch { return null; } })();
  lab.worksheet = { worksheet: ws.title, worksheet_id: ws.id, objectives: ws.objectives, hypothesis: $("hypothesis").value, runner: ws.runner, settings: result.settings, results: { headings: result.headings, rows: result.rows },
    raw: result.raw, audit: { engine: E.version, testExposed: lab.testExposed, testLabelsExposed: lab.testLabelsExposed, model: fingerprint } };
  (lab.worksheetsDone ??= new Set()).add(ws.id); count("worksheets");
  message("One factor changed; starts, policy and paired seed schedule stayed fixed. Intervals describe the stated unit of replication only.");
}, true);
$("worksheet").onchange = buildWorksheetControls;
install("exportWorksheet", () => {
  if (!lab.worksheet) throw new Error("Run the worksheet experiment first.");
  download("ml-driving-worksheet.json", { ...lab.worksheet, quiz_attempts: lab.quizAttempts, evidence: $("evidence").value, conclusion: $("conclusion").value, rubric: kit.rubric });
});
install("printWorksheet", () => window.print());
for (const [i, ws] of worksheets.entries()) { const option = document.createElement("option"); option.value = i; option.textContent = `${i + 1}. ${ws.title}`; $("worksheet").append(option); }

function hashText(text) { let h = 2166136261; for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619); return h >>> 0; }
const quizDone = phase => lab.quizAttempts.some(attempt => attempt.phase === phase);
// The feedback reveals the answers, so a second graded attempt in the same phase would measure memory, not learning: each phase is graded once per session.
function lockQuiz(note = "") {
  for (const input of document.querySelectorAll("#quiz input")) input.disabled = true;
  $("gradeQuiz").disabled = true;
  if (note) $("quizFeedback").textContent = note;
}
function renderQuiz() {
  const attempt = lab.quizAttempts.length, phase = $("quizPhase").value;
  lab.quizOrder = kit.quiz.map(q => new E.RNG(hashText(`${q.id}|${attempt}`)).shuffle(q.choices.map((_, k) => k)));
  $("quiz").replaceChildren(); $("quizReview").replaceChildren(); $("quizFeedback").textContent = ""; $("gradeQuiz").disabled = false;
  kit.quiz.forEach((q, i) => {
    const wrap = document.createElement("div"); wrap.className = "quizitem"; wrap.id = `quizitem${i}`;
    const p = document.createElement("p"); p.textContent = `${i + 1}. ${q.prompt}`; wrap.append(p);
    for (const choiceIndex of lab.quizOrder[i]) {
      const label = document.createElement("label"), input = document.createElement("input");
      input.type = "radio"; input.name = `quiz${i}`; input.value = choiceIndex; label.append(input, document.createTextNode(" " + q.choices[choiceIndex].text)); wrap.append(label);
    }
    $("quiz").append(wrap);
  });
  if (quizDone(phase)) lockQuiz(`The ${phase === "pre" ? "before" : "after"}-lesson check is already recorded in this session. Choose the other phase, or reload the page to start over.`);
}
$("quizPhase").onchange = renderQuiz;
install("gradeQuiz", () => {
  const phase = $("quizPhase").value;
  if (quizDone(phase)) throw new Error(`The ${phase === "pre" ? "before" : "after"}-lesson check is already recorded; choose the other phase.`);
  const answers = kit.quiz.map((q, i) => { const chosen = document.querySelector(`input[name="quiz${i}"]:checked`); return chosen ? Number(chosen.value) : null; });
  if (answers.some(a => a === null)) throw new Error("Answer every question before checking.");
  const byObjective = {}, misconceptions = new Set(); let score = 0;
  $("quizReview").replaceChildren();
  kit.quiz.forEach((q, i) => {
    const choice = q.choices[answers[i]], right = Boolean(choice.correct), correct = q.choices.find(c => c.correct);
    score += Number(right); byObjective[q.objective] = [(byObjective[q.objective]?.[0] ?? 0) + Number(right), (byObjective[q.objective]?.[1] ?? 0) + 1];
    if (!right && choice.misconception) misconceptions.add(choice.misconception);
    const item = $(`quizitem${i}`); item.className = `quizitem ${right ? "right" : "wrong"}`;
    const note = document.createElement("p"); note.className = "note";
    const miss = !right && choice.misconception ? kit.misconceptions.find(m => m.id === choice.misconception) : null;
    note.textContent = right ? `Correct. ${choice.why}` : `Not quite. ${choice.why} The better answer: “${correct.text}”. ${correct.why}${miss ? ` Common misconception (${miss.id}): ${miss.text} ${miss.remedy}` : ""}`;
    item.querySelector(".note")?.remove(); item.append(note);
  });
  const total = kit.quiz.length;
  lab.quizAttempts.push({ phase, score, total, by_objective: byObjective, misconceptions: [...misconceptions] });
  lockQuiz();
  $("quizFeedback").textContent = `${score}/${total}. Review the colour-coded questions: each gives the reasoning, and wrong answers point to the module that addresses the misconception. This ${phase === "pre" ? "before" : "after"}-lesson check is now recorded; choose the other phase for the next one.`;
  const rows = Object.entries(byObjective).map(([id, [k, n]]) => [id, kit.objectives.find(o => o.id === id).text, `${k}/${n}`]);
  const review = document.createElement("div"); review.innerHTML = "<h4>Score by learning objective</h4><div class='table'><table id='quizObjectives'></table></div>"; $("quizReview").append(review);
  table("quizObjectives", ["Objective", "Statement", "Correct"], rows); count("quiz_attempts");
});
function buildCard() {
  const nickname = $("nickname").value.trim().slice(0, 20).replace(/[^\p{L}\p{N} _.-]/gu, "");
  const last = lab.losses.at(-1), probe = lab.probe.at(-1);
  return { schema: "f1-ml-lab-results-card-v1", lab_version: E.version, nickname, modules_visited: [...lab.log.modules], experiments: { ...lab.log.counts },
    worksheets_completed: [...(lab.worksheetsDone ?? [])], quiz: lab.quizAttempts,
    exposure: { test_exposed: lab.testExposed, test_labels_exposed: lab.testLabelsExposed },
    measurements: { latest_training: last ? { updates: last.update, validation_loss: last.validation, stress_loss: last.stress, nominal_sections_safe: probe?.nominal.k ?? null, recovery_sections_safe: probe?.recovery.k ?? null } : null,
      coverage: lab.coverage ? Object.fromEntries(lab.coverage.summaries.map(s => [s.arm, { nominal: s.nominal.mean, recovery: s.recovery.mean, seeds: s.nominal.n }])) : null } };
}
install("downloadCard", () => download("ml-lab-results-card.json", buildCard()));
buildWorksheetControls(); renderQuiz(); drawOverlay(); drawPlayground([{ settings: { model: "none" }, runs: [{ history: [] }] }]);
