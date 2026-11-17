"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const E = require("../student_engine.js");
const T = require("../student_tools.js");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/engine_fixture.json"), "utf8"));
const { inputs, expected } = fixture, cfg = expected.config;
const tracks = new Map(expected.tracks.map(t => [t.id, new E.Track(t)]));
const ctx = { cfg, tracks };
const rows = inputs.rows.map((row, i) => ({ id: `fixture-${i}`, episode: `fixture-episode-${Math.floor(i / 10)}`, split: "train", track_id: 1, source: "fixture", ...row }));
const pretrained = E.Network.fromJSON(inputs.pretrained);
const close = (actual, wanted, tolerance, label) => assert.ok(Math.abs(actual - wanted) <= tolerance, `${label}: ${actual} vs ${wanted}`);
const closeAll = (actual, wanted, tolerance, label) => {
  assert.equal(actual.length, wanted.length, `${label} length`);
  actual.forEach((v, i) => close(v, wanted[i], tolerance, `${label}[${i}]`));
};
const scenario = (id, index) => expected.tracks.find(t => t.id === id).scenarios[index];
const maskedCopy = (features, mask) => E.maskFeatures(features, mask);

test("Wilson intervals agree with SciPy for every tabulated count", () => {
  for (const { k, n, lo, hi } of expected.wilson) {
    const w = T.wilson(k, n);
    close(w.lo, lo, 1e-12, `lo ${k}/${n}`); close(w.hi, hi, 1e-12, `hi ${k}/${n}`);
  }
  assert.throws(() => T.wilson(4, 3)); assert.throws(() => T.wilson(0, 0)); assert.throws(() => T.wilson(1.5, 3));
});

test("Bootstrap percentile interval matches NumPy for the same explicit resamples", () => {
  const { values, resamples } = inputs.bootstrap, result = T.bootstrapMean(values, null, { resamples });
  close(result.mean, expected.bootstrap.mean, 1e-12, "mean");
  close(result.lo, expected.bootstrap.lo, 1e-12, "lo"); close(result.hi, expected.bootstrap.hi, 1e-12, "hi");
  const sorted = Array.from({ length: 200 }, (_, b) => resamples[b].reduce((s, i) => s + values[i], 0) / 10).sort((a, b) => a - b);
  for (const [q, v] of Object.entries(expected.bootstrap.quantiles)) close(T.quantile(sorted, Number(q)), v, 1e-12, `quantile ${q}`);
  assert.deepEqual(T.bootstrapMean([3], new E.RNG(1)), { n: 1, mean: 3, lo: 3, hi: 3 });
  assert.throws(() => T.bootstrapMean([], new E.RNG(1)));
});

test("Exact sign test matches SciPy and paired differences count wins, losses and ties", () => {
  for (const { wins, losses, p } of expected.sign_test) close(T.signTest(wins, losses).p, p, 1e-12, `sign ${wins}/${losses}`);
  const paired = T.pairedDifference([1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1], new E.RNG(9));
  assert.deepEqual([paired.wins, paired.losses, paired.ties, paired.n], [3, 1, 2, 6]);
  assert.ok(paired.lo <= paired.mean && paired.mean <= paired.hi);
  assert.throws(() => T.pairedDifference([1, 2], [1], new E.RNG(1)));
});

test("Learning-rate schedules equal the Python reference", () => {
  for (const c of expected.schedules) close(T.learningRateAt(c.schedule, c.base, c.update, c.total), c.rate, 1e-15, `${c.schedule} ${c.update}/${c.total}`);
  assert.throws(() => T.learningRateAt("warmup", 1, 1, 10));
});

test("Adam updates with weight decay and gradient clipping equal NumPy", () => {
  for (const c of expected.training) {
    const model = E.Network.fromJSON(inputs.training_start), batch = rows.slice(0, c.rows);
    c.losses.forEach((loss, step) => {
      const packet = model.trainBatch(batch, c.rate, { weightDecay: c.weight_decay, maxGradNorm: c.max_grad_norm });
      close(packet.loss, loss, 1e-11, `${c.name} loss ${step}`);
      close(packet.gradNorm, c.gradient_norms[step], 1e-10, `${c.name} gradient norm ${step}`);
      assert.equal(packet.clipped, c.clipped[step], `${c.name} clipped ${step}`);
    });
    for (const field of ["weights", "biases"]) closeAll(model[field].flat(Infinity), c.after[field].flat(Infinity), 1e-10, `${c.name} ${field}`);
    assert.equal(model.updates, c.updates);
  }
});

test("Training guards reject bad hyper-parameters and divergence leaves every parameter untouched", () => {
  const model = E.Network.fromJSON(inputs.training_start), batch = rows.slice(0, 8), before = JSON.stringify(model.serialize());
  for (const rate of [0, -1, NaN, Infinity, 0.06]) assert.throws(() => model.trainBatch(batch, rate), /Learning rate/);
  for (const options of [{ weightDecay: -0.1 }, { weightDecay: 2 }, { weightDecay: NaN }, { maxGradNorm: 0 }, { maxGradNorm: -1 }, { maxGradNorm: Infinity }])
    assert.throws(() => model.trainBatch(batch, 0.001, options), /Weight decay|Gradient-norm/);
  assert.equal(JSON.stringify(model.serialize()), before);
  model.weights[2][0][0] = Infinity;
  const poisoned = JSON.stringify(model.serialize()), updatesBefore = model.updates;
  assert.throws(() => model.trainBatch(batch, 0.001), /numerical_divergence/);
  assert.equal(JSON.stringify(model.serialize()), poisoned); assert.equal(model.updates, updatesBefore);
});

test("Weight decay shrinks weights, clipping shrinks the applied gradient, neither touches biases or defaults", () => {
  const plain = E.Network.fromJSON(inputs.training_start), decayed = plain.clone(), batch = rows.slice(0, 16);
  const start = JSON.parse(JSON.stringify(plain.weights));
  plain.trainBatch(batch, 0.004); const packet = decayed.trainBatch(batch, 0.004, { weightDecay: 0.25 });
  for (let l = 0; l < 3; l++) {
    assert.deepEqual(plain.biases[l], decayed.biases[l]);
    for (let i = 0; i < start[l].length; i++) for (let j = 0; j < start[l][i].length; j++)
      close(decayed.weights[l][i][j], plain.weights[l][i][j] - 0.004 * 0.25 * start[l][i][j], 1e-15, `W${l}[${i}][${j}]`);
  }
  assert.ok(packet.changes.every(change => change.decay === 0 || change.parameter.startsWith("W")));
  const clipped = E.Network.fromJSON(inputs.training_start).trainBatch(batch, 0.001, { maxGradNorm: 0.01 });
  assert.equal(clipped.clipped, true);
  assert.ok(clipped.changes.every(c => Math.abs(c.applied) <= Math.abs(c.gradient) + 1e-18));
});

test("Ridge regression equals the NumPy normal-equation solution for every mask", () => {
  for (const c of expected.ridge) {
    const model = T.RidgeModel.fit(rows, { lambda: c.lambda, mask: c.mask });
    closeAll(model.weights.flat(), c.weights.flat(), 1e-9, `${c.mask} weights`);
    c.predictions.forEach((p, i) => closeAll(model.predict(rows[i].features), p, 1e-9, `${c.mask} prediction ${i}`));
    close(T.modelLoss(model, rows), c.loss, 1e-9, `${c.mask} loss`);
  }
  assert.throws(() => T.RidgeModel.fit(rows, { lambda: 0 }));
  assert.throws(() => T.RidgeModel.fit([], {}));
});

test("k-nearest-neighbour predictions equal a NumPy brute-force search", () => {
  for (const c of expected.knn) {
    const model = T.KNNModel.fit(rows.slice(12), { k: c.k, mask: c.mask });
    c.predictions.forEach((p, i) => closeAll(model.predict(rows[i].features.map(v => v + 0.013)), p, 1e-12, `${c.mask} query ${i}`));
  }
  assert.throws(() => T.KNNModel.fit(rows, { k: 0 })); assert.throws(() => T.KNNModel.fit(rows.slice(0, 3), { k: 5 }));
  assert.equal(T.KNNModel.fit(rows, { k: 1 }).serialize().examples, rows.length);
});

test("Permutation importance equals NumPy for explicit shuffles and is reproducible for seeded ones", () => {
  const groups = [[0], [1, 2], [5, 6, 7, 8, 9, 10, 11, 12]].map(indices => ({ name: `g${indices[0]}`, indices }));
  const result = T.permutationImportance(pretrained, rows.slice(0, 30), new E.RNG(1), { repeats: 2, groups, maxRows: 30,
    permutation: (m, group, repeat) => inputs.permutations[groups.indexOf(group)][repeat] });
  close(result[0].baseline, expected.permutation.baseline, 1e-12, "baseline");
  result.forEach((entry, g) => closeAll(entry.increases, expected.permutation.groups[g].increases, 1e-10, `group ${g}`));
  const a = T.permutationImportance(pretrained, rows, new E.RNG(4), { repeats: 3 }), b = T.permutationImportance(pretrained, rows, new E.RNG(4), { repeats: 3 });
  assert.deepEqual(a.map(x => x.mean), b.map(x => x.mean));
  assert.equal(a.filter(x => x.kind === "group").length, 8); assert.equal(a.filter(x => x.kind === "feature").length, 16);
  assert.equal(a.filter(x => x.kind === "derived").length, 2);
  close(T.modelLoss(pretrained, rows), expected.pretrained_loss, 1e-12, "weighted loss");
});

test("Full closed-loop trajectories match the NumPy simulator for expert, rule, learned, delayed, low-grip and noisy drivers", () => {
  assert.ok(expected.trajectories.length >= 24);
  const flat = expected.noise_draws.flat();
  for (const t of expected.trajectories) {
    const model = t.label === "expert" ? "expert" : t.label === "rules" ? "rules" : pretrained;
    let used = 0;
    const options = t.label === "delay 3" ? { delay: 3 } : t.label === "grip 0.65" ? { grip: 0.65 }
      : t.label === "noise 1" ? { noise: 1, rng: { normal: () => flat[used++] } } : {};
    const { trace, metric } = E.simulate(tracks.get(t.track_id), cfg, model, scenario(t.track_id, t.scenario), options);
    const label = `${t.label} track ${t.track_id} scenario ${t.scenario}`;
    assert.equal(trace.length, t.length, `${label} length`); assert.equal(metric.end_reason, t.end_reason, `${label} end`);
    assert.equal(metric.success, t.success, `${label} success`);
    for (const [key, field, tolerance] of [["x", "x", 1e-6], ["y", "y", 1e-6], ["heading", "heading", 1e-8], ["speed", "speed", 1e-8],
      ["lateral", "lateral", 1e-6], ["brake", "brake", 1e-9], ["margin", "margin", 1e-6]])
      closeAll(trace.map(row => row[field]), t[key], tolerance, `${label} ${key}`);
    assert.deepEqual(trace.map(row => row.gear), t.gear, `${label} gears`);
  }
});

test("Recording and forking never change the simulation: snapshots resume exactly where they were taken", () => {
  const track = tracks.get(3), start = scenario(3, 2), plain = E.simulate(track, cfg, pretrained, start);
  const recorded = E.simulate(track, cfg, pretrained, start, { record: true });
  assert.deepEqual(recorded.metric, plain.metric);
  recorded.trace.forEach((row, i) => i < recorded.trace.length - 1 && assert.ok(row.action && row.expert && row.features.length === 18, `row ${i}`));
  assert.equal(recorded.trace.at(-1).action, undefined);
  const env = new E.Environment(track, cfg); env.reset(start);
  for (let i = 0; i < 25; i++) env.step(pretrained.predict(env.observe().features));
  const snapshot = env.snapshot(), reference = [], restored = [];
  for (let i = 0; i < 20; i++) { env.step(pretrained.predict(env.observe().features)); reference.push(env.record()); }
  env.restore(snapshot);
  for (let i = 0; i < 20; i++) { env.step(pretrained.predict(env.observe().features)); restored.push(env.record()); }
  assert.deepEqual(restored, reference);
  const fork = E.simulate(track, cfg, pretrained, null, { snapshot, maxSteps: 20 });
  assert.equal(fork.trace.length, 21); assert.deepEqual(fork.trace[1].x, reference[0].x);
});

test("A model that outputs non-finite commands ends the drive as numerical_divergence with finite metrics", () => {
  const result = E.simulate(tracks.get(1), cfg, { predict: () => [NaN, 0] }, scenario(1, 0));
  assert.equal(result.metric.end_reason, "numerical_divergence"); assert.equal(result.metric.success, 0);
  assert.ok(Number.isFinite(result.metric.progress) && result.trace.every(row => Number.isFinite(row.x)));
  assert.throws(() => new E.Environment(tracks.get(1), cfg).step([NaN, 0]), /finite/);
});

test("Divergence markers and expert recovery separate action mismatch from the state the car was left in", async () => {
  const track = tracks.get(1), crash = E.simulate(track, cfg, "rules", scenario(1, 2), { record: true });
  assert.equal(crash.metric.end_reason, "lateral_acceleration_limit");
  const errors = T.actionErrors(crash.trace), divergence = T.firstDivergence(errors);
  assert.ok(divergence && divergence.index <= crash.trace.length - 2);
  assert.equal(T.firstDivergence([]), null);
  assert.equal(T.firstDivergence([{ error: 0.5 }, { error: 0.001 }, { error: 0.9 }, { error: 0.9 }, { error: 0.9 }], { sustain: 3 }).error, 0.9);
  const recovery = await T.expertRecovery({ track, cfg, trace: crash.trace });
  assert.ok(recovery.forks.length >= 1 && recovery.forks[0].success, "the expert completes this start from the beginning");
  const expert = E.simulate(track, cfg, "expert", scenario(1, 0), { record: true });
  const clean = await T.expertRecovery({ track, cfg, trace: expert.trace, strideSteps: 15 });
  assert.ok(clean.forks.every(f => f.success) && clean.firstFailure === null && clean.monotonic);
  const doomed = E.simulate(track, cfg, { predict: () => [1, 1] }, scenario(1, 0), { record: true });
  const forced = await T.expertRecovery({ track, cfg, trace: doomed.trace });
  assert.equal(forced.forks[0].success, true);
  const drifting = E.simulate(track, cfg, { predict: () => [0.12, -0.2] }, scenario(1, 0), { record: true, maxSteps: 120 });
  const late = await T.expertRecovery({ track, cfg, trace: drifting.trace, strideSteps: 10 });
  assert.ok(late.forks.length >= 8 && late.forks.every(f => typeof f.success === "boolean" && Number.isFinite(f.min_clearance_m)));
  assert.equal(typeof late.monotonic, "boolean");
});

test("A fork keeps the exit-centering history of the drive it was taken from", () => {
  const track = tracks.get(1), env = new E.Environment(track, cfg); env.reset(scenario(1, 0));
  for (let i = 0; i < 60; i++) env.step(env.teacher());
  const snapshot = env.snapshot();
  const alone = E.simulate(track, cfg, "expert", null, { snapshot, maxSteps: 3, goal: 1e9 });
  const history = Array.from({ length: 19 }, () => ({ lateral: 9 }));
  const kept = E.simulate(track, cfg, "expert", null, { snapshot, maxSteps: 3, goal: 1e9, history });
  assert.ok(kept.metric.exit_mean_lateral_m > alone.metric.exit_mean_lateral_m + 5);
});

test("Collection labels every visited state with the expert, gives unique episodes and reports rejected starts", () => {
  const rng = new E.RNG(5), track = tracks.get(1);
  const { rows: data, failures, episodes, rejected } = T.collect({ track, cfg, driver: "expert", rng, steps: 400, spec: T.defaultStarts.typical, episodeSteps: 100 });
  assert.equal(data.length, 400); assert.equal(failures, 0); assert.equal(episodes, 4); assert.ok(Number.isInteger(rejected) && rejected >= 0);
  assert.equal(new Set(data.map(r => r.id)).size, 400); assert.equal(new Set(data.map(r => r.episode)).size, 4);
  E.validateRows(data);
  const crashing = T.collect({ track, cfg, driver: { predict: () => [1, 1] }, rng: new E.RNG(2), steps: 60, spec: T.defaultStarts.typical });
  assert.ok(crashing.failures > 3 && crashing.rows.length === 60);
  const stress = T.sampleStart(new E.RNG(3), T.defaultStarts.stress);
  assert.ok(Math.abs(stress.lateral) >= 5.5 && Math.abs(stress.lateral) <= 6.6 && Math.sign(stress.lateral) === Math.sign(stress.heading_error));
  const walled = T.collect({ track, cfg, driver: "expert", rng: new E.RNG(8), steps: 20, spec: { lateral: 30, heading: .1, speed: [8, 12] }, episodeSteps: 5 });
  assert.ok(walled.rejected > 0, "starts that touch a wall are counted, not silently dropped");
});

test("Every data strategy gets the same labelled-state budget and update budget, and runs are reproducible", async () => {
  const trackIds = [1, 3], setup = { seed: 3, ctx, trackIds, budget: 40, rounds: 2, updatesPerRound: 12 };
  const results = {};
  assert.deepEqual(Object.keys(T.arms), ["bc_narrow", "bc_broad", "bc_staged", "dagger"]);
  for (const arm of Object.keys(T.arms)) results[arm] = await T.runArm({ arm, ...setup });
  for (const run of Object.values(results)) {
    assert.equal(run.rows.length, 40 * 3 * trackIds.length); assert.equal(run.net.updates, 12 * 3);
    E.validateRows(run.rows);
  }
  assert.ok(results.dagger.rows.some(r => /learner rollout/.test(r.source)) && !results.bc_narrow.rows.some(r => /learner/.test(r.source)));
  assert.ok(!results.bc_staged.rows.some(r => /learner/.test(r.source)) && results.bc_staged.rows.some(r => /typical starts/.test(r.source)),
    "the staged control matches DAgger's schedule and start distribution, with the expert driving");
  assert.deepEqual(results.dagger.rounds.map(r => r.size), results.bc_staged.rounds.map(r => r.size));
  assert.equal(results.dagger.rounds.length, 3); assert.ok(results.dagger.rounds.every(r => "failed" in r && "rejected" in r));
  const again = await T.runArm({ arm: "dagger", ...setup });
  assert.equal(E.fingerprint(again.net), E.fingerprint(results.dagger.net));
  assert.notEqual(E.fingerprint(results.bc_narrow.net), E.fingerprint(results.bc_broad.net));
  assert.notEqual(E.fingerprint(results.bc_staged.net), E.fingerprint(results.dagger.net));
  await assert.rejects(() => T.runArm({ arm: "magic", ...setup }), /Unknown data strategy/);
});

test("Curvature inputs and the expert do not flip on one ulp of distance at a sample boundary", () => {
  const track = tracks.get(1), env = new E.Environment(track, cfg);
  env.reset({ fraction: .3, lateral: 7, heading_error: .2, speed: 12 });
  const { location } = env.observe(), vertex = Math.round(location[0] / track.ds) * track.ds;
  const neighbour = (value, step) => { const view = new BigInt64Array(new Float64Array([value]).buffer); view[0] += BigInt(step); return new Float64Array(view.buffer)[0]; };
  const results = [neighbour(vertex, -1), vertex, neighbour(vertex, 1)].map(distance => {
    const original = track.locate;
    track.locate = () => [distance, location[1], location[2]];
    try { return { curvature: env.observe().features.slice(13, 16), expert: env.teacher() }; } finally { track.locate = original; }
  });
  assert.notEqual(neighbour(vertex, -1), vertex);
  assert.deepEqual(results[0], results[1]); assert.deepEqual(results[2], results[1]);
  assert.deepEqual([E.sampleIndex(354 - 1e-13, 4000), E.sampleIndex(354, 4000), E.sampleIndex(1000 - 1e-12, 1000), E.sampleIndex(1000 - 1e-6, 1000)], [354, 354, 0, 999]);
});

test("The coverage experiment pairs seeds across recipes, keeps nominal and recovery apart, is reproducible and cancellable", async () => {
  const data = { scenarios: Object.fromEntries([1, 3].map(id => [id, expected.tracks.find(t => t.id === id).scenarios])) };
  const setup = { data, ctx, arms: ["bc_narrow", "dagger"], seedCount: 2, budget: 30, rounds: 1, updatesPerRound: 8, trainIds: [1, 3], validIds: [1, 3], holdoutIds: [1] };
  const run = await T.coverageExperiment(setup);
  assert.deepEqual(Object.keys(run.results), ["bc_narrow", "dagger"]);
  assert.equal(run.trainSections.length, 8); assert.equal(run.holdSections.length, 4);
  for (const results of Object.values(run.results)) {
    assert.deepEqual(results.map(r => r.seed), [601, 602]);
    for (const r of results) assert.ok([r.nominal, r.recovery, r.hold, r.overall].every(v => v >= 0 && v <= 1) && r.diverged === false && r.labelled === 30 * 2 * 2);
  }
  assert.equal(run.comparisons.length, 1, "only comparisons between recipes that were run are reported");
  const [pair] = run.comparisons;
  assert.deepEqual([pair.a, pair.b], ["dagger", "bc_narrow"]); assert.equal(pair.overall.wins + pair.overall.losses + pair.overall.ties, 2);
  assert.ok(run.keep.dagger.rows.length > 0 && run.keep.dagger.visited.length > 0 && run.expert.train.nominal.n === 4);
  const again = await T.coverageExperiment(setup);
  assert.deepEqual(again.results, run.results);
  await assert.rejects(() => T.coverageExperiment({ ...setup, shouldCancel: () => true }), /Cancelled/);
});

test("Projected occupancy needs several rows from distinct episodes; coverage grids and tallies describe what was driven", () => {
  const strict = T.supportIndex(rows, cfg), lenient = T.supportIndex(rows, cfg, { minRows: 1, minEpisodes: 1 });
  assert.ok(lenient.size > 0 && lenient.size <= rows.length && strict.size <= lenient.size);
  const singleEpisode = rows.slice(0, 12).map((r, i) => ({ ...r, episode: "same", features: rows[0].features }));
  assert.equal(T.supportIndex(singleEpisode, cfg, { minRows: 3, minEpisodes: 2 }).size, 0, "many rows from one episode are not support");
  assert.equal(T.supportIndex(singleEpisode, cfg, { minRows: 3, minEpisodes: 1 }).size, 1);
  const first = rows[0], visited = [{ lateral: first.features[0] * 9, heading: Math.atan2(first.features[1], first.features[2]), speed: first.features[3] * cfg.max_speed,
    curvature: first.features[13], kind: "nominal" }, { lateral: 8.5, heading: 0.45, speed: 1, curvature: 3, kind: "recovery" }];
  assert.equal(T.supportFraction(visited, lenient), 0.5); assert.equal(T.supportFraction(visited, lenient, "recovery"), 0);
  assert.equal(T.supportFraction([], lenient), null);
  const grid = T.coverageGrid(rows.map(r => T.featurePoint(cfg, r)), { xRange: [-12, 12], yRange: [-Math.PI, Math.PI], xBins: 48, yBins: 25 });
  assert.equal(grid.counts.reduce((a, b) => a + b, 0), rows.length); assert.ok(grid.max >= 1);
  const clipped = T.coverageGrid([[0, 0], [99, 0], [0, 9]], { xRange: [-12, 12], yRange: [-1, 1], xBins: 4, yBins: 4 });
  assert.equal(clipped.counts.reduce((a, b) => a + b, 0), 1, "points outside the plotted ranges are not counted");
  const sections = [{ track_id: 1, scenario: scenario(1, 0), kind: "nominal" }, { track_id: 1, scenario: scenario(1, 2), kind: "recovery" }];
  const evaluation = T.evaluateSections(pretrained, sections, ctx, { visited: true });
  assert.equal(evaluation.n, 2); assert.equal(evaluation.nominal.n, 1); assert.equal(evaluation.recovery.n, 1); assert.equal(evaluation.invalid, 0);
  assert.ok(evaluation.visited.length > 100 && evaluation.visited.every(s => Number.isFinite(s.curvature)));
  const walled = T.evaluateSections(pretrained, [{ track_id: 1, scenario: { ...scenario(1, 0), lateral: 8.8 }, kind: "recovery" }], ctx);
  assert.equal(walled.invalid, 1, "a start inside a wall is reported as invalid, not dropped");
  const reference = T.evaluateSections("expert", sections, ctx, { visited: true });
  assert.equal(reference.k, 2);
});

test("Training loop: schedules, decay, early stopping with best restore, cancellation, probes and divergence", async () => {
  const train = rows.slice(0, 40), validation = rows.slice(40);
  const constant = await T.fit({ seed: 2, train, validation, updates: 40, learningRate: 0.003 });
  const cosine = await T.fit({ seed: 2, train, validation, updates: 40, learningRate: 0.003, schedule: "cosine" });
  assert.equal(constant.history.at(-1).learningRate, 0.003); assert.ok(cosine.history.at(-1).learningRate < 0.0004);
  assert.equal(constant.history[0].update, 0); assert.equal(constant.history.at(-1).update, 40); assert.equal(constant.net.updates, 40);
  const decayed = await T.fit({ seed: 2, train, validation, updates: 40, learningRate: 0.003, weightDecay: 0.5 });
  const norm = net => Math.sqrt(net.weights.flat(2).reduce((s, v) => s + v * v, 0));
  assert.ok(norm(decayed.net) < norm(constant.net));
  const stopper = await T.fit({ seed: 2, train: train.slice(0, 8), validation, updates: 400, learningRate: 0.004, patience: 5, sizes: [18, 48, 32, 2] });
  assert.equal(stopper.stopped, "early_stop"); assert.ok(stopper.net.updates < 400 && stopper.best.updates === stopper.bestUpdate && stopper.bestUpdate < stopper.net.updates);
  assert.ok(T.modelLoss(stopper.best, validation) <= T.modelLoss(stopper.net, validation));
  let calls = 0; const cancelled = await T.fit({ seed: 2, train, validation, updates: 100, shouldCancel: () => ++calls > 10 });
  assert.equal(cancelled.stopped, "cancelled"); assert.equal(cancelled.net.updates, 10);
  const probed = await T.fit({ seed: 2, train, validation, updates: 20, probeEvery: 10, probe: net => ({ updates: net.updates }) });
  assert.deepEqual(probed.history.filter(h => h.probe).map(h => h.probe.updates), [0, 10, 20]);
  const flaky = E.Network.fromJSON(inputs.training_start), original = flaky.trainBatch.bind(flaky); let trainCalls = 0;
  flaky.trainBatch = (...args) => { if (++trainCalls === 3) throw new Error("numerical_divergence: injected for the test"); return original(...args); };
  const diverged = await T.fit({ train, validation, updates: 10, initial: flaky });
  assert.equal(diverged.stopped, "diverged"); assert.match(diverged.divergence.message, /numerical_divergence/);
  assert.equal(diverged.net.updates, 2, "the failed update changed nothing"); assert.equal(diverged.divergence.update, 3);
  const broken = E.Network.fromJSON(inputs.training_start);
  broken.trainBatch = () => { throw new Error("some other bug"); };
  await assert.rejects(() => T.fit({ train, validation, updates: 3, initial: broken }), /some other bug/);
  await assert.rejects(() => T.fit({ train: [], validation }), /at least one example/);
});

test("Paired conditions share starts and noise streams, and sections - not trials - are the unit of replication", async () => {
  const sections = [{ track_id: 1, index: 0, kind: "nominal", scenario: scenario(1, 0) }, { track_id: 3, index: 2, kind: "recovery", scenario: scenario(3, 2) },
    { track_id: 1, index: 2, kind: "recovery", scenario: scenario(1, 2) }, { track_id: 3, index: 0, kind: "nominal", scenario: scenario(3, 0) }];
  const conditions = [{ name: "Clean" }, { name: "Noise 1", noise: 1 }, { name: "Grip .5", grip: 0.5 }];
  const trials = await T.runConditions({ conditions, sections, seeds: [1, 2, 3, 4], model: pretrained, ctx });
  assert.equal(trials.length, 3 * 4 * 4);
  for (const section of [0, 1, 2, 3]) for (const seed of [1, 2, 3, 4]) {
    const group = trials.filter(t => t.section === section && t.seed === seed);
    assert.equal(group.length, 3); assert.ok(group.every(t => JSON.stringify(t.initial) === JSON.stringify(group[0].initial)));
  }
  const summary = T.summarizeConditions(trials, new E.RNG(8));
  assert.equal(summary[0].trials, 16); assert.equal(summary[0].sections, 4); assert.equal(summary[0].versus, undefined); assert.equal(summary[0].wilson, null);
  assert.ok(summary[0].lo <= summary[0].rate && summary[0].rate <= summary[0].hi);
  for (const entry of summary.slice(1)) {
    assert.equal(entry.versus.baseline, "Clean"); assert.equal(entry.versus.sections, 4);
    assert.equal(entry.versus.wins + entry.versus.losses + entry.versus.ties, 4, "pairs are counted per section, not per trial");
  }
  const independent = T.summarizeConditions(trials.filter(t => t.seed === 1), new E.RNG(8), { independentStarts: true });
  assert.equal(independent[0].wilson.n, 4); assert.ok(independent[0].wilson.lo <= independent[0].wilson.p);
  const rules = await T.runConditions({ conditions: [{ name: "Neural" }, { name: "Rules", policy: "rules" }], sections, seeds: [1, 2], model: pretrained, ctx });
  assert.notDeepEqual(rules.filter(r => r.condition === "Neural").map(r => r.metric.end_reason), rules.filter(r => r.condition === "Rules").map(r => r.metric.end_reason));
  const live = [{ name: "Rules", policy: "rules" }, { name: "Neural", policy: pretrained.clone() }, { name: "Plain" }];
  const exported = await T.runConditions({ conditions: live, sections, seeds: [1], model: pretrained, ctx });
  assert.deepEqual([...new Set(exported.map(r => JSON.stringify(r.settings)))].sort(), ['{"name":"Neural","policy":"model"}', '{"name":"Plain"}', '{"name":"Rules","policy":"rules"}']);
  assert.ok(JSON.stringify(exported).length < 100000 && !/weights|"mW"/.test(JSON.stringify(exported)), "an export must never embed the policy's weights");
  assert.deepEqual(live.map(T.describeCondition).map(c => c.policy), ["rules", "model", undefined]);
  assert.ok(typeof live[1].policy === "object", "describing a condition must not strip the live policy from the original");
  const pool = T.randomSections(new E.RNG(6), [1, 3], 6, T.defaultStarts.stress);
  assert.equal(pool.length, 6); assert.ok(pool.every(s => s.kind === "recovery") && new Set(pool.map(s => s.track_id)).size === 2);
});

test("Permutation importance recomputes the derived inputs so shuffled examples stay physically consistent", () => {
  const consistent = rows.map(r => ({ ...r, features: E.derived(r.features, cfg) }));
  const original = consistent.map(r => r.features.slice());
  const lane = T.permutationImportance(pretrained, consistent, new E.RNG(2), { repeats: 3, cfg, groups: [{ name: "lane offset", indices: [0] }] })[0];
  const raw = T.permutationImportance(pretrained, consistent, new E.RNG(2), { repeats: 3, groups: [{ name: "lane offset", indices: [0] }] })[0];
  assert.ok(lane.mean !== raw.mean, "recomputing margin and budget changes the answer");
  consistent.forEach((r, i) => assert.deepEqual(r.features, original[i]));
  const derivedOnly = T.permutationImportance(pretrained, consistent, new E.RNG(2), { repeats: 2, cfg, groups: [{ name: "margin", indices: [16] }] })[0];
  const stillRaw = T.permutationImportance(pretrained, consistent, new E.RNG(2), { repeats: 2, groups: [{ name: "margin", indices: [16] }] })[0];
  assert.deepEqual(derivedOnly.increases, stillRaw.increases, "derived inputs are shuffled on their own");
  const groups = T.defaultFeatureGroups();
  assert.deepEqual(groups.filter(g => g.derived).map(g => g.indices[0]), [16, 17]);
});

test("Episode starts that are near-duplicates cannot straddle splits, whatever their IDs say", () => {
  const start = { track_id: 2, fraction: .31, lateral: 1.5, heading_error: .03, speed: 11 };
  const episodes = { "drive-1": { split: "train", ...start } };
  assert.equal(T.findSplitLeak(episodes, { split: "validation", ...start, lateral: 1.6 }), "drive-1");
  assert.equal(T.findSplitLeak(episodes, { split: "train", ...start }), null, "the same split may repeat a start");
  assert.equal(T.findSplitLeak(episodes, { split: "validation", ...start, track_id: 3 }), null);
  assert.equal(T.findSplitLeak(episodes, { split: "validation", ...start, fraction: .5 }), null);
  assert.throws(() => T.validateEpisodeStarts({ ...episodes, "drive-2": { split: "validation", ...start, speed: 11.2 } }), /leak/);
  assert.equal(T.validateEpisodeStarts({ ...episodes, "drive-2": { split: "validation", ...start, speed: 14 } }), true);
});

test("The stress split is accepted for evaluation rows but never lets one episode span two splits", () => {
  const data = rows.slice(0, 6).map((row, i) => ({ ...row, split: i < 3 ? "train" : "stress", episode: i < 3 ? "a" : "b" }));
  E.validateRows(data);
  assert.throws(() => E.validateRows([{ ...rows[0], split: "stress", episode: "x" }, { ...rows[1], split: "train", episode: "x" }]), /episode/);
  assert.throws(() => E.validateRows([{ ...rows[0], split: "dev" }]), /split/);
  assert.ok(E.compatibleVersions.includes("student-lab-1") && E.compatibleVersions.includes(E.version));
});

test("PPO losses, gradients and advantage estimates agree with an independent vectorised NumPy implementation", () => {
  const R = require("../student_rl.js"), p = inputs.ppo, e = expected.ppo, agent = new R.ActorCritic(1), n = p.observations.length;
  const load = (layers, target) => layers.forEach((layer, l) => {
    const t = target[l];
    for (let i = 0; i < t.nIn; i++) for (let j = 0; j < t.nOut; j++) t.W[j * t.nIn + i] = layer.weights[i][j];
    layer.biases.forEach((v, j) => { t.b[j] = v; });
  });
  load(p.actor, agent.actor); load(p.critic, agent.critic); agent.logStd.set(p.log_std);
  p.observations.forEach((x, i) => {
    closeAll(Array.from(agent.forwardActor(x)), e.mean[i], 1e-12, `policy mean ${i}`);
    close(agent.forwardCritic(x), e.value[i], 1e-12, `value ${i}`);
  });
  const batch = { obs: Float64Array.from(p.observations.flat()), actions: Float64Array.from(e.actions.flat()), logp: Float64Array.from(e.old_logp), advantages: Float64Array.from(p.advantages), returns: Float64Array.from(p.returns) };
  const stats = agent.ppoGradient(batch, Array.from({ length: n }, (_, i) => i), { clipRange: p.options.clip_range, valueCoef: p.options.value_coef, entropyCoef: p.options.entropy_coef });
  for (const [mine, theirs] of [["policyLoss", "policy_loss"], ["valueLoss", "value_loss"], ["entropy", "entropy"], ["kl", "kl"], ["clipFraction", "clip_fraction"]]) close(stats[mine], e[theirs], 1e-11, mine);
  assert.ok(stats.clipFraction > 0 && stats.clipFraction < 1, "the problem exercises both the clipped and the unclipped branch");
  // Gradients in the trainer are stored output-major; NumPy's are input-major.
  const compare = (layers, gradients, label) => layers.forEach((layer, l) => {
    for (let i = 0; i < layer.nIn; i++) for (let j = 0; j < layer.nOut; j++) close(layer.gW[j * layer.nIn + i], gradients[l].weights[i][j], 1e-10, `${label} layer ${l} weight ${i},${j}`);
    closeAll(Array.from(layer.gb), gradients[l].biases, 1e-10, `${label} layer ${l} bias`);
  });
  compare(agent.actor, e.actor_gradients, "actor"); compare(agent.critic, e.critic_gradients, "critic");
  closeAll(Array.from(agent.gLogStd), e.log_std_gradient, 1e-10, "log-std gradient");
  p.gae.forEach((c, k) => {
    const result = R.gae({ rewards: c.rewards, values: c.values, done: c.done, lastValue: c.last_value, gamma: c.gamma, lambda: c.lambda });
    closeAll(Array.from(result.advantages), e.gae[k].advantages, 1e-12, `advantages ${k}`); closeAll(Array.from(result.returns), e.gae[k].returns, 1e-12, `returns ${k}`);
  });
});
