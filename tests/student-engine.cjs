"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const E = require("../student_engine.js");
const output = path.resolve(process.env.STUDENT_OUTPUT || "student_artifacts");
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
