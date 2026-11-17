"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const E = require("../student_engine.js");
const output = path.resolve(process.env.STUDENT_OUTPUT || "student_v2_artifacts");
const data = JSON.parse(fs.readFileSync(path.join(output, "data/lab.json"), "utf8"));
const golden = JSON.parse(fs.readFileSync(path.join(output, "data/golden.json"), "utf8"));
const tracks = data.tracks.map(track => new E.Track(track));
const net = E.Network.fromJSON(data.pretrained);
function close(actual, expected, tolerance = 1e-9, label = "value") {
  assert.ok(Math.abs(actual - expected) < tolerance, `${label}: ${actual} != ${expected}`);
}
test("Python/JavaScript observations, steering, target speed and complete drivetrain step agree", () => {
  for (const fixture of golden.physics) {
    const environment = new E.Environment(tracks.find(t => t.id === fixture.track_id), data.config);
    environment.reset(fixture.scenario);
    environment.observe().features.forEach((v, i) => close(v, fixture.features[i], 1e-8, `input ${i}`));
    environment.teacher().forEach((v, i) => close(v, fixture.expert[i], 1e-8, `expert ${i}`));
    net.predict(fixture.features).forEach((v, i) => close(v, fixture.prediction[i], 1e-10, `prediction ${i}`));
    environment.step(fixture.prediction);
    for (const [key, expected] of Object.entries(fixture.after))
      close(key === "margin" ? environment.margin : environment.state[key], expected, 1e-8, key);
  }
});
test("One full Adam minibatch update agrees with NumPy, not an illustrative animation", () => {
  const fixture = golden.training, model = E.Network.fromJSON(fixture.before);
  const packet = model.trainBatch(fixture.batch, .001);
  close(packet.loss, fixture.loss, 1e-12);
  for (const field of ["weights", "biases"]) {
    const actual = model[field].flat(Infinity), expected = fixture.after[field].flat(Infinity);
    actual.forEach((v, i) => close(v, expected[i], 1e-11, `${field} ${i}`));
  }
});
test("Independent finite differences verify gradients for every layer and a bias", () => {
  const model = E.Network.fromJSON(golden.training.before), rows = golden.training.batch, gradient = model.gradients(rows), epsilon = 1e-5;
  for (const [layer, i, j] of [[0, 0, 0], [0, 16, 2], [1, 3, 5], [2, 5, 1]]) {
    const old = model.weights[layer][i][j];
    model.weights[layer][i][j] = old + epsilon; const plus = model.loss(rows);
    model.weights[layer][i][j] = old - epsilon; const minus = model.loss(rows);
    model.weights[layer][i][j] = old;
    close((plus - minus) / (2 * epsilon), gradient.gw[layer][i][j], 1e-7);
  }
  const old = model.biases[2][0];
  model.biases[2][0] = old + epsilon; const plus = model.loss(rows);
  model.biases[2][0] = old - epsilon; const minus = model.loss(rows);
  model.biases[2][0] = old;
  close((plus - minus) / (2 * epsilon), gradient.gb[2][0], 1e-7);
});
test("Real training lowers error and cloned optimizer updates remain identical", () => {
  const model = new E.Network(7), train = data.starter.filter(row => row.split === "train").slice(0, 32);
  const loss = model.loss(train), before = E.fingerprint(model);
  for (let i = 0; i < 50; i++) model.trainBatch(train, .001);
  assert.ok(model.loss(train) < loss * .6); assert.notEqual(E.fingerprint(model), before);
  const copy = model.clone(); model.trainBatch(train, .001); copy.trainBatch(train, .001);
  assert.deepEqual(model.serialize(), copy.serialize());
  const imported = E.Network.fromJSON(model.serialize()); assert.equal(imported.updates, 0);
  assert.ok(imported.mW.flat(Infinity).every(v => v === 0));
});
test("Ablation masks are active in training and inference without changing initialization", () => {
  for (const mask of Object.keys(E.featureMasks)) {
    const model = new E.Network(10, [18, 24, 16, 2], mask), baseline = new E.Network(10);
    assert.deepEqual(model.weights, baseline.weights);
    const features = data.starter[0].features.slice(), changed = features.slice();
    for (const i of E.featureMasks[mask]) changed[i] += 100;
    assert.deepEqual(model.predict(features), model.predict(changed));
    const gradient = model.gradients(data.starter.slice(0, 8));
    for (const i of E.featureMasks[mask]) assert.ok(gradient.gw[0][i].every(v => v === 0));
  }
});
test("Bad labels, malformed models, cross-episode splits and heldout leakage are rejected", () => {
  const row = data.starter[0];
  assert.throws(() => E.validateRows([{ ...row, targets: [2, 0] }]), /Labels/);
  assert.throws(() => E.validateRows([{ ...row, features: [NaN, ...row.features.slice(1)] }]), /finite/);
  assert.throws(() => E.validateRows([{ ...row, episode: undefined }]), /episode/);
  assert.throws(() => E.validateRows([row, row]), /unique/);
  assert.throws(() => E.validateRows([row, { ...row, id: "another", split: "validation" }]), /episode/);
  assert.throws(() => E.validateRows([{ ...row, track_id: 13 }]), /eleven/);
  assert.throws(() => E.validateRows(data.testRows), /eleven|Test examples/);
  E.validateRows(data.testRows, true);
  assert.throws(() => E.Network.fromJSON({ weights: [0], biases: [0] }), /Invalid model/);
  assert.throws(() => new E.Network().trainBatch([{ features: row.features, targets: [NaN, 0] }]), /finite/);
});
test("Footprint wall contact and actual lateral-limit failures are distinct", () => {
  const environment = new E.Environment(tracks[0], data.config);
  const base = data.scenarios[0][0];
  environment.reset({ ...base, lateral: 8.5, heading_error: 0 });
  assert.ok(environment.contact);
  const hit = E.simulate(tracks[0], data.config, net, { ...base, lateral: 8.5, heading_error: 0 });
  assert.equal(hit.metric.end_reason, "wall_contact"); assert.equal(hit.trace.length, 1);
  environment.reset({ ...base, speed: 30, lateral: 0, heading_error: 0 });
  assert.equal(environment.step([1, 1]).reason, "lateral_acceleration_limit");
});
test("Noise, historical observations, braking grip and seeds change the real simulation", () => {
  const base = data.scenarios[0][2], observation = golden.physics[0].features;
  const a = E.noisyFeatures(observation, data.config, new E.RNG(1), 1);
  const b = E.noisyFeatures(observation, data.config, new E.RNG(1), 1);
  assert.deepEqual(a, b); assert.notDeepEqual(a, observation);
  for (let i = 5; i <= 15; i++) close(a[i], observation[i]);
  const model = { predict: x => [-.2 * x[0], -.5] }, clean = E.simulate(tracks[0], data.config, model, base);
  assert.notDeepEqual(clean.trace, E.simulate(tracks[0], data.config, model, base, { delay: 5 }).trace);
  assert.notDeepEqual(clean.trace, E.simulate(tracks[0], data.config, model, base, { noise: 1 }).trace);
  assert.deepEqual(E.simulate(tracks[0], data.config, model, base, { noise: 1, seed: 3 }),
    E.simulate(tracks[0], data.config, model, base, { noise: 1, seed: 3 }));
  const high = new E.Environment(tracks[0], data.config, 1), low = new E.Environment(tracks[0], data.config, .65);
  high.reset({ ...base, lateral: 0 }); low.reset({ ...base, lateral: 0 });
  high.step([0, -1]); low.step([0, -1]); assert.ok(low.state.speed > high.state.speed);
  assert.throws(() => E.simulate(tracks[0], data.config, net, base, { delay: -1 }), /Delay/);
});
test("Section traces report measured pose, time, pedals, gears and true-state margins", () => {
  const result = E.simulate(tracks[0], data.config, net, data.scenarios[0][2]);
  assert.ok(result.trace.length > 50);
  close(result.metric.duration_s, result.trace.at(-1).time);
  close(result.metric.min_clearance_m, Math.min(...result.trace.map(row => row.margin)));
  assert.ok(result.trace.some(row => row.brake > .2));
  assert.ok(result.trace.every(row => row.gear >= 1 && row.gear <= 8 && Number.isFinite(row.x)));
});
test("Sample variability is reported correctly and empty statistics fail visibly", () => {
  assert.deepEqual(E.meanSD([1, 2, 3]), { mean: 2, sd: 1 });
  assert.deepEqual(E.meanSD([5]), { mean: 5, sd: null });
  assert.throws(() => E.meanSD([]));
});
// ---- version-2 artifacts ----------------------------------------------------------------------------------------------------------
const T = require("../student_tools.js");
const onlyV2 = data.version === "student-lab-2" ? false : "needs the version-2 artifacts (student_v2_artifacts)";
test("The saved lab and the engine speak a compatible version", { skip: onlyV2 }, () => {
  assert.ok(E.compatibleVersions.includes(data.version), `${data.version} is not one of ${E.compatibleVersions}`);
});
test("The browser engine replays every saved NumPy rollout of all 40 retrained networks with identical outcomes", { skip: onlyV2 }, () => {
  assert.equal(data.dataGeneration.seeds.ablation.length, 10); assert.equal(Object.keys(data.models).length, 40);
  const byId = new Map(tracks.map(t => [t.id, t])), nets = new Map(Object.entries(data.models).map(([key, model]) => [key, E.Network.fromJSON(model)]));
  const numeric = ["progress", "duration_s", "min_clearance_m", "exit_mean_lateral_m", "peak_speed_kmh", "braking_s"];
  let worst = 0;
  for (const record of data.ablations) {
    const { metric } = E.simulate(byId.get(record.track_id), data.config, nets.get(record.model), data.scenarios[record.track_id][record.scenario_index]);
    const label = `${record.model} track ${record.track_id} section ${record.scenario_index}`;
    for (const field of ["success", "end_reason", "wall_contacts"]) assert.equal(metric[field], record[field], `${label}: ${field}`);
    for (const field of numeric) { const difference = Math.abs(metric[field] - record[field]); worst = Math.max(worst, difference); assert.ok(difference < 1e-9, `${label}: ${field} ${metric[field]} vs ${record[field]}`); }
  }
  assert.equal(data.ablations.length, 2560); assert.ok(worst < 1e-9);
});
test("The classroom data obey the same rules the page enforces: whole episodes per split, no near-duplicate starts, sealed rows kept apart", { skip: onlyV2 }, () => {
  E.validateRows(data.starter); E.validateRows(data.stressRows); E.validateRows(data.testRows, true);
  T.validateEpisodeStarts(data.starterEpisodes);
  assert.ok(data.starter.every(r => r.track_id < 11) && data.testRows.every(r => r.track_id >= 13));
  const first = Object.entries(data.starterEpisodes).find(([, m]) => m.split === "train"), other = Object.entries(data.starterEpisodes).find(([, m]) => m.split === "validation");
  assert.ok(first && other);
  const twin = { ...first[1], split: "validation", lateral: first[1].lateral + .01 };
  assert.throws(() => T.validateEpisodeStarts({ ...data.starterEpisodes, twin }), /near|leak|similar/i);
});
test("The saved reference coverage run reports one value per seed for every recipe", { skip: onlyV2 || (data.coverageReference ? false : "run tools/coverage_reference.cjs, then student_render.py") }, () => {
  const reference = data.coverageReference;
  assert.equal(reference.settings.seeds, 10); assert.equal(reference.arms.length, 4);
  for (const arm of reference.arms) for (const key of ["nominal", "recovery", "hold", "overall"]) assert.equal(arm[key].values.length, reference.settings.seeds);
  assert.ok(reference.comparisons.every(c => c.overall.wins + c.overall.losses + c.overall.ties === reference.settings.seeds));
});
// ---- module 12: reinforcement learning and the zero-to-a-lap journey -------------------------------------------------------------------
const R = require("../student_rl.js");
const J = require("../student_journey.js");
const ctx = { cfg: data.config, tracks: new Map(tracks.map(t => [t.id, t])) };
test("PPO gradients match finite differences of the loss for the actor, the critic and the exploration noise", () => {
  const ac = new R.ActorCritic(5, { outputScale: 1 }), rng = new E.RNG(11), n = 24;
  const batch = { obs: new Float64Array(n * 18), actions: new Float64Array(n * 2), logp: new Float64Array(n), advantages: new Float64Array(n), returns: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    const x = Array.from({ length: 18 }, () => rng.normal() * .5), sample = ac.act(x, rng);
    batch.obs.set(x, i * 18); batch.actions.set(sample.action, i * 2);
    // Old log-probabilities are shifted so some ratios sit well inside the clip range and others well outside it, away from the kinks.
    batch.logp[i] = sample.logp + [0, .6, -.6, .02][i % 4]; batch.advantages[i] = rng.normal(); batch.returns[i] = rng.normal();
  }
  const indices = Array.from({ length: n }, (_, i) => i), options = { clipRange: .2, valueCoef: .5, entropyCoef: .01 };
  const loss = () => { const s = ac.ppoGradient(batch, indices, options); return s.policyLoss + options.valueCoef * s.valueLoss - options.entropyCoef * s.entropy; };
  loss();
  const analytic = ac.gradients.map(g => Float64Array.from(g));
  // Parameter groups: actor W/b x3, critic W/b x3, then the shared log standard deviations.
  for (const [group, index] of [[0, 40], [1, 3], [2, 10], [3, 5], [4, 7], [5, 1], [6, 50], [7, 2], [8, 30], [9, 4], [10, 3], [11, 0], [12, 0], [12, 1]]) {
    const parameter = ac.parameters[group], old = parameter[index], epsilon = 1e-6;
    parameter[index] = old + epsilon; const plus = loss();
    parameter[index] = old - epsilon; const minus = loss();
    parameter[index] = old;
    const numeric = (plus - minus) / (2 * epsilon), expected = analytic[group][index];
    assert.ok(Math.abs(numeric - expected) < 1e-6 * Math.max(1, Math.abs(expected)), `group ${group} index ${index}: finite difference ${numeric} vs backpropagation ${expected}`);
  }
});
test("Generalised advantage estimation matches a hand calculation, including an episode end", () => {
  const gamma = .9, lambda = .8, rewards = [1, 0, 2], values = [.5, .4, .3], done = [0, 0, 1], lastValue = 7;
  const { advantages, returns } = R.gae({ rewards, values, done, lastValue, gamma, lambda });
  const delta2 = 2 - .3, delta1 = 0 + gamma * .3 - .4, delta0 = 1 + gamma * .4 - .5;   // the last step ends the episode, so lastValue is not bootstrapped
  const a2 = delta2, a1 = delta1 + gamma * lambda * a2, a0 = delta0 + gamma * lambda * a1;
  [a0, a1, a2].forEach((expected, i) => close(advantages[i], expected, 1e-12, `advantage ${i}`));
  [a0 + .5, a1 + .4, a2 + .3].forEach((expected, i) => close(returns[i], expected, 1e-12, `return ${i}`));
  const alive = R.gae({ rewards, values, done: [0, 0, 0], lastValue, gamma, lambda }).advantages;
  close(alive[2], 2 + gamma * lastValue - .3, 1e-12, "bootstrapped last advantage");
});
test("The reward pays distance, charges for crashing and changes in the commands, and starts each drive without a previous command", () => {
  const track = tracks.find(t => t.id === 1), only = reward => new R.DrivingEnv({ track, cfg: data.config, seed: 3, reward: { distance: 0, centering: 0, failure: 0, smoothness: 0, ...reward } });
  const smooth = only({ smoothness: 1 }); smooth.reset();
  close(smooth.step([.2, .5]).reward, 0, 1e-15, "no previous command on the first step");
  close(smooth.step([-.4, .5]).reward, -((-.4 - .2) ** 2 + 0) / 2, 1e-12, "squared change of the two commands, averaged");
  smooth.reset(); close(smooth.step([.9, -.9]).reward, 0, 1e-15, "a reset forgets the previous command");
  const distance = only({ distance: 1 }); distance.reset();
  const moved = distance.step([0, 1]); assert.ok(moved.advanced > 0); close(moved.reward, moved.advanced, 1e-12, "distance reward is 1 per metre when weighted 1");
  const crash = only({ failure: 5 }); crash.reset();
  let result; for (let i = 0; i < 400 && !result?.terminated; i++) result = crash.step([1, 1]);
  assert.ok(result.terminated && result.reason !== ""); close(result.reward, -5, 1e-12, "the failure charge");
});
test("The reinforcement actor starts exactly where an imitation network of the same seed starts and exports as an ordinary network", () => {
  const agent = new R.ActorCritic(7, { outputScale: 1 }), exported = agent.actorNetwork(), reference = new E.Network(7);
  assert.deepEqual(exported.weights, reference.weights); assert.deepEqual(exported.biases, reference.biases);
  const x = Array.from({ length: 18 }, (_, i) => Math.sin(i)), mean = agent.forwardActor(x);
  exported.predict(x).forEach((v, i) => close(v, mean[i], 1e-12, `output ${i}`));
  assert.equal(E.fingerprint(exported), E.fingerprint(E.Network.fromJSON(exported.serialize())));
});
test("PPO training is deterministic for a seed, differs between seeds, and rejects unknown settings", async () => {
  const track = tracks.find(t => t.id === 1), run = seed => R.ppoTrain({ track, cfg: data.config, seed, iterations: 2, envs: 2, rolloutSteps: 64, minibatch: 64 });
  const [a, b, c] = [await run(1), await run(1), await run(2)];
  assert.deepEqual(a.history, b.history); assert.deepEqual(a.agent.actorNetwork().weights, b.agent.actorNetwork().weights);
  assert.notDeepEqual(a.agent.actorNetwork().weights, c.agent.actorNetwork().weights);
  assert.equal(a.steps, 2 * 2 * 64); assert.ok(a.history.every(h => Number.isFinite(h.policyLoss) && Number.isFinite(h.valueLoss) && h.kl >= 0));
  await assert.rejects(R.ppoTrain({ track, cfg: data.config, iterations: 1, lr: 1 }), /Unknown PPO setting/);
});
test("Run summaries separate the first lap from the lap that stays, and flag runs that lose it or flick the steering", () => {
  const row = (steps, done, flicker = .1) => [steps, done ? 100 : 40, done ? 1 : 0, done ? 130 : 20, flicker];
  const lucky = [row(0, 0), row(300, 1), row(600, 0), row(900, 1), row(1200, 1)], lost = [row(0, 0), row(300, 1), row(600, 0)], flicking = [row(0, 0), row(300, 1, 4.5)];
  assert.deepEqual(J.runSummary(lucky, 100), { firstLapSteps: 300, stableLapSteps: 900, finalCompleted: true, ratio: 1.3, regressed: false, lostLap: true, flicker: .1, flicks: false });
  assert.equal(J.runSummary(flicking, 100).flicks, true); assert.equal(J.runSummary([row(0, 0), [150, 0, 0, 3, 4.5]], 100).flicks, false, "a drive that ends within 10 s is too short to judge");
  assert.deepEqual([J.runSummary(lost, 100).firstLapSteps, J.runSummary(lost, 100).stableLapSteps, J.runSummary(lost, 100).regressed], [300, null, true]);
  const summary = J.aggregate([{ stages: lucky }, { stages: lost }, { stages: flicking }], 100);
  assert.deepEqual([summary.seeds, summary.finalLaps, summary.regressed, summary.lostLap, summary.flickering], [3, 2, 1, 2, 1]);
  assert.equal(summary.firstLapSteps.median, 300); assert.equal(summary.stableLapSteps.n, 2);
  assert.equal(J.lapPercent({ completed: false, progress: .9996 }), 99); assert.equal(J.lapPercent({ completed: true, progress: 1 }), 100);
  assert.deepEqual(J.rlMarks(100).slice(-3), [64, 80, 100]); assert.equal(J.rlMarks(7).at(-1), 7);
});
test("The imitation journey starts from the untrained network and its lap test is a real whole lap", async () => {
  const track = tracks.find(t => t.id === 1), stages = [];
  const run = await J.imitationJourney({ track, ctx, seed: 1, rounds: 6, onStage: stage => stages.push(stage) });
  assert.equal(stages.length, run.stages.length); assert.equal(run.stages.length, 8);
  assert.deepEqual(run.stages[0].model.weights, new E.Network(1).serialize().weights); assert.equal(run.stages[0].steps, 0);
  assert.ok(run.stages.every((s, i) => i === 0 || s.steps > run.stages[i - 1].steps));
  assert.ok(!run.stages[0].lap.completed && run.stages[0].lap.progress < .05, "an untrained car fails at once");
  const laps = run.stages.filter(s => s.lap.completed);
  assert.ok(laps.length > 0, "some trained stage completes the lap"); assert.ok(laps.every(s => s.lap.progress === 1 && s.lap.seconds > 90 && s.lap.seconds < 400));
  const expert = J.expertLap(track, data.config); assert.ok(expert.completed && expert.seconds > 100 && expert.seconds < 120);
});
test("The reinforcement journey learns to lap from a reward alone, smoothly, and a saved stage restores exactly", async () => {
  const track = tracks.find(t => t.id === 1), run = await J.reinforcementJourney({ track, ctx, seed: 1, iterations: 40 });
  assert.deepEqual(run.stages.map(s => s.iteration), [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40]);
  assert.ok(!run.stages[0].lap.completed && run.stages[0].lap.progress < .05, "an untrained car fails at once");
  const final = run.stages.at(-1); assert.ok(final.lap.completed, "the final car completes the lap"); assert.ok(final.lap.flicker < J.FLICKER_LIMIT, "and does not flick the steering");
  assert.equal(final.steps, 40 * 2048); assert.equal(run.history.length, 40);
  const saved = { index: final.index, iteration: final.iteration, steps: final.steps, updates: final.updates, model: final.model, row: J.compactStage(final) };
  assert.equal(J.restoreStage("reinforcement", track, data.config, saved).reproduced, true);
  assert.equal(J.restoreStage("reinforcement", track, data.config, { ...saved, row: [...saved.row.slice(0, 3), saved.row[3] + .1, saved.row[4]] }).reproduced, false);
});
test("Cancelling a reinforcement journey keeps the car as it stands", async () => {
  const track = tracks.find(t => t.id === 1); let polls = 0;
  const run = await J.reinforcementJourney({ track, ctx, seed: 1, iterations: 20, shouldCancel: () => ++polls > 2 });
  assert.equal(run.stopped, "cancelled"); assert.equal(run.history.length, 2); assert.equal(run.stages.at(-1).iteration, 2);
});
const savedJourney = data.journeyReference ? false : "run tools/journey_reference.cjs, then student_render.py";
test("The saved journey example reproduces in this engine stage by stage, and its summary is what the code computes from the raw runs", { skip: onlyV2 || savedJourney }, () => {
  const file = path.join(output, "data/journey_reference.json"), reference = JSON.parse(fs.readFileSync(file, "utf8")), page = data.journeyReference;
  assert.deepEqual(Object.keys(page).sort(), Object.keys(reference).filter(k => !["runs", "markdown"].includes(k)).sort());
  assert.equal(page.settings.seeds, 10); assert.equal(Object.keys(page.circuits).length, data.dataGeneration.valid_training_tracks.length);
  const featured = page.featured, track = tracks.find(t => t.id === featured.trackId);
  for (const method of ["imitation", "reinforcement"]) for (const saved of featured[method].stages) assert.equal(J.restoreStage(method, track, data.config, saved).reproduced, true, `${method} stage ${saved.index}`);
  for (const id of data.dataGeneration.valid_training_tracks) {
    const expert = J.expertLap(tracks.find(t => t.id === id), data.config).seconds, circuit = reference.circuits[id];
    close(expert, circuit.expertSeconds, .051, `expert lap ${id}`);
    for (const method of ["imitation", "reinforcement"]) assert.deepEqual(JSON.parse(JSON.stringify(J.aggregate(reference.runs[method][id], expert))), JSON.parse(JSON.stringify(circuit[method])), `${method} ${id}`);
  }
});
