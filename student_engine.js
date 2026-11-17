(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.StudentEngine = factory();
})(globalThis, function () {
  "use strict";
  const version = "student-lab-2";
  // v1 datasets and models only lack the optional stress split and training options, so they stay importable.
  const compatibleVersions = ["student-lab-1", "student-lab-2"];
  const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const mod = (v, n) => ((v % n) + n) % n;
  const wrap = v => mod(v + Math.PI, 2 * Math.PI) - Math.PI;
  const copy = value => JSON.parse(JSON.stringify(value));
  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be finite`);
    return value;
  }
  class RNG {
    constructor(seed) {
      if (!Number.isInteger(seed)) throw new Error("Seed must be an integer");
      this.state = seed >>> 0;
    }
    uniform() {
      let t = this.state += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
    normal() {
      return Math.sqrt(-2 * Math.log(Math.max(this.uniform(), 1e-15))) * Math.cos(2 * Math.PI * this.uniform());
    }
    index(n) { if (n < 1) throw new Error("Cannot sample an empty dataset"); return Math.floor(this.uniform() * n); }
    range(low, high) { return low + (high - low) * this.uniform(); }
    shuffle(values) {
      for (let i = values.length - 1; i > 0; i--) { const j = this.index(i + 1); [values[i], values[j]] = [values[j], values[i]]; }
      return values;
    }
  }
  const zeros = (rows, columns) => Array.from({ length: rows }, () => Array(columns).fill(0));
  const featureMasks = {
    full: [],
    no_future_geometry: Array.from({ length: 11 }, (_, i) => i + 5),
    no_heading: [1, 2, 16],
    no_margin: [16],
  };
  function maskFeatures(features, mask) {
    if (!Object.hasOwn(featureMasks, mask)) throw new Error("Unknown feature experiment");
    const result = features.slice();
    for (const i of featureMasks[mask]) result[i] = 0;
    return result;
  }
  class Network {
    constructor(seed = 1, sizes = [18, 24, 16, 2], mask = "full") {
      if (sizes.length !== 4 || sizes[0] !== 18 || sizes[3] !== 2 || !sizes.every(n => Number.isInteger(n) && n > 0))
        throw new Error("Expected an 18-input, two-hidden-layer, two-output network");
      if (!Object.hasOwn(featureMasks, mask)) throw new Error("Unknown feature experiment");
      this.sizes = sizes.slice();
      this.mask = mask;
      const rng = new RNG(seed);
      this.weights = sizes.slice(0, -1).map((n, l) => Array.from({ length: n }, () =>
        Array.from({ length: sizes[l + 1] }, () => rng.normal() / Math.sqrt(n))));
      this.biases = sizes.slice(1).map(n => Array(n).fill(0));
      this.resetOptimizer();
    }
    resetOptimizer() {
      this.updates = 0;
      this.mW = this.weights.map(w => zeros(w.length, w[0].length));
      this.vW = this.weights.map(w => zeros(w.length, w[0].length));
      this.mB = this.biases.map(b => b.map(() => 0));
      this.vB = this.biases.map(b => b.map(() => 0));
    }
    static fromJSON(data) {
      if (!data || !Array.isArray(data.weights) || data.weights.length !== 3
          || !Array.isArray(data.biases) || data.biases.length !== 3
          || data.weights.some(w => !Array.isArray(w) || !w.length || w.some(row => !Array.isArray(row)))
          || data.biases.some(b => !Array.isArray(b))) throw new Error("Invalid model file");
      const sizes = [data.weights[0]?.length, ...data.biases.map(b => b.length)];
      const net = new Network(1, sizes, data.mask ?? "full");
      for (let l = 0; l < 3; l++) {
        if (data.weights[l].length !== sizes[l] || data.biases[l].length !== sizes[l + 1]
            || data.weights[l].some(row => row.length !== sizes[l + 1])) throw new Error("Invalid model dimensions");
        data.weights[l].flat().forEach(v => finite(v, "Weight"));
        data.biases[l].forEach(v => finite(v, "Bias"));
      }
      net.weights = copy(data.weights);
      net.biases = copy(data.biases);
      net.resetOptimizer();
      return net;
    }
    serialize() { return { version, sizes: this.sizes, mask: this.mask, weights: copy(this.weights), biases: copy(this.biases), updates: this.updates }; }
    clone() {
      const net = Network.fromJSON(this.serialize());
      for (const field of ["mW", "vW", "mB", "vB"]) net[field] = copy(this[field]);
      net.updates = this.updates;
      return net;
    }
    forward(features) {
      if (!Array.isArray(features) || features.length !== 18) throw new Error("Expected 18 observation values");
      features.forEach(v => finite(v, "Observation"));
      const activations = [maskFeatures(features, this.mask)];
      for (let l = 0; l < 3; l++) {
        const next = this.biases[l].slice();
        for (let j = 0; j < next.length; j++) {
          for (let i = 0; i < activations[l].length; i++) next[j] += activations[l][i] * this.weights[l][i][j];
          next[j] = Math.tanh(next[j]);
        }
        activations.push(next);
      }
      return activations;
    }
    predict(features) { return this.forward(features)[3]; }
    loss(rows) {
      validateBatch(rows);
      let total = 0;
      for (const row of rows) {
        const prediction = this.predict(row.features);
        total += 4 * (prediction[0] - row.targets[0]) ** 2 + (prediction[1] - row.targets[1]) ** 2;
      }
      return total / (2 * rows.length);
    }
    gradients(rows) {
      validateBatch(rows);
      const gw = this.weights.map(w => zeros(w.length, w[0].length));
      const gb = this.biases.map(b => b.map(() => 0));
      let loss = 0;
      const predictions = [], firstActivations = this.forward(rows[0].features);
      for (const row of rows) {
        const a = this.forward(row.features), residual = a[3].map((v, i) => v - row.targets[i]);
        predictions.push(a[3]);
        loss += residual[0] ** 2 * 4 + residual[1] ** 2;
        let gradient = residual.map((v, i) => v * [4, 1][i] / rows.length * (1 - a[3][i] ** 2));
        for (let l = 2; l >= 0; l--) {
          for (let j = 0; j < gradient.length; j++) {
            gb[l][j] += gradient[j];
            for (let i = 0; i < a[l].length; i++) gw[l][i][j] += a[l][i] * gradient[j];
          }
          if (l > 0) {
            const previous = Array(a[l].length).fill(0);
            for (let i = 0; i < previous.length; i++) {
              for (let j = 0; j < gradient.length; j++) previous[i] += this.weights[l][i][j] * gradient[j];
              previous[i] *= 1 - a[l][i] ** 2;
            }
            gradient = previous;
          }
        }
      }
      return { gw, gb, loss: loss / (2 * rows.length), predictions, activations: firstActivations };
    }
    trainBatch(rows, learningRate = .001, options = {}) {
      finite(learningRate, "Learning rate");
      if (learningRate <= 0 || learningRate > .05) throw new Error("Learning rate must be >0 and <=0.05");
      const weightDecay = options.weightDecay ?? 0, maxGradNorm = options.maxGradNorm ?? null;
      finite(weightDecay, "Weight decay");
      if (weightDecay < 0 || weightDecay > 1) throw new Error("Weight decay must be between 0 and 1");
      if (maxGradNorm !== null) { finite(maxGradNorm, "Gradient-norm limit"); if (maxGradNorm <= 0) throw new Error("Gradient-norm limit must be positive"); }
      const result = this.gradients(rows), changes = [];
      let squares = 0;
      for (const group of [result.gw, result.gb]) for (const layer of group) for (const entry of layer)
        if (Array.isArray(entry)) for (const v of entry) squares += v * v; else squares += entry * entry;
      const gradNorm = Math.sqrt(squares);
      if (!Number.isFinite(result.loss) || !Number.isFinite(gradNorm))
        throw new Error("numerical_divergence: non-finite loss or gradient; parameters were left unchanged. Lower the learning rate or check the data.");
      const clipped = maxGradNorm !== null && gradNorm > maxGradNorm, scale = clipped ? maxGradNorm / gradNorm : 1;
      this.updates++;
      const c1 = 1 - .9 ** this.updates, c2 = 1 - .999 ** this.updates, keep = options.trackChanges !== false;
      for (let l = 0; l < 3; l++) {
        for (let i = 0; i < this.weights[l].length; i++) {
          for (let j = 0; j < this.weights[l][i].length; j++) {
            const gradient = result.gw[l][i][j], applied = gradient * scale, before = this.weights[l][i][j];
            this.mW[l][i][j] = .9 * this.mW[l][i][j] + .1 * applied;
            this.vW[l][i][j] = .999 * this.vW[l][i][j] + .001 * applied ** 2;
            const decay = -learningRate * weightDecay * before;
            const delta = -learningRate * (this.mW[l][i][j] / c1) / (Math.sqrt(this.vW[l][i][j] / c2) + 1e-8);
            this.weights[l][i][j] = before + decay;
            this.weights[l][i][j] += delta;
            if (keep) changes.push({ parameter: `W${l}[${i},${j}]`, layer: l, i, j, before, gradient, applied, delta, decay, after: this.weights[l][i][j] });
          }
        }
        for (let j = 0; j < this.biases[l].length; j++) {
          const gradient = result.gb[l][j], applied = gradient * scale, before = this.biases[l][j];
          this.mB[l][j] = .9 * this.mB[l][j] + .1 * applied;
          this.vB[l][j] = .999 * this.vB[l][j] + .001 * applied ** 2;
          const delta = -learningRate * (this.mB[l][j] / c1) / (Math.sqrt(this.vB[l][j] / c2) + 1e-8);
          this.biases[l][j] += delta;
          if (keep) changes.push({ parameter: `b${l}[${j}]`, layer: l, j, before, gradient, applied, delta, decay: 0, after: this.biases[l][j] });
        }
      }
      result.changes = keep ? changes.sort((a, b) => Math.abs(b.gradient) - Math.abs(a.gradient)).slice(0, 6) : [];
      result.lossAfter = options.skipLossAfter ? null : this.loss(rows);
      result.updates = this.updates;
      Object.assign(result, { gradNorm, clipped, learningRate, weightDecay, maxGradNorm });
      return result;
    }
  }
  function validateBatch(rows) {
    if (!Array.isArray(rows) || !rows.length) throw new Error("Training and loss require examples");
    for (const row of rows) {
      if (!row || !Array.isArray(row.features) || row.features.length !== 18
          || !Array.isArray(row.targets) || row.targets.length !== 2) throw new Error("Examples require 18 inputs and two labels");
      row.targets.forEach(v => { finite(v, "Label"); if (Math.abs(v) > 1) throw new Error("Labels must be in [-1,1]"); });
    }
  }
  function validateRows(rows, allowLeakage = false) {
    if (!Array.isArray(rows) || rows.length > 20000) throw new Error("Dataset must contain at most 20,000 rows");
    const ids = new Set(), episodes = new Map();
    for (const row of rows) {
      if (!row || typeof row.id !== "string" || !row.id.length || ids.has(row.id)) throw new Error("Every example needs a unique ID");
      ids.add(row.id);
      if (!Number.isInteger(row.track_id) || row.track_id < 0 || row.track_id > 15) throw new Error("Unknown track ID");
      if (row.track_id >= 11 && !allowLeakage) throw new Error("Training/validation examples must come from the eleven training circuits");
      if (typeof row.source !== "string" || !row.source.length) throw new Error("Example source must be identified");
      if (typeof row.episode !== "string" || !row.episode.length) throw new Error("Examples must identify their driving episode");
      if (!["train", "validation", "test", "stress"].includes(row.split)) throw new Error("Unknown dataset split");
      if (row.split === "test" && !allowLeakage) throw new Error("Test examples cannot be trained on outside the explicit leakage lesson");
      if (!Array.isArray(row.features) || row.features.length !== 18 || !Array.isArray(row.targets) || row.targets.length !== 2)
        throw new Error("Examples require 18 inputs and two labels");
      row.features.forEach(v => finite(v, "Feature"));
      row.targets.forEach(v => { finite(v, "Label"); if (Math.abs(v) > 1) throw new Error("Labels must be in [-1,1]"); });
      if (episodes.has(row.episode) && episodes.get(row.episode) !== row.split) throw new Error("A driving episode cannot cross training/validation/test splits");
      episodes.set(row.episode, row.split);
    }
    return true;
  }
  class Track {
    constructor(data) {
      this.data = data;
      for (const key of ["xy", "left", "right", "tangent", "normal", "heading", "curvature"]) this[key] = data[key];
      this.n = this.xy.length;
      this.ds = data.ds;
      this.length = data.length;
      this.id = data.id;
      this.name = data.name;
    }
    point(distance) {
      const fractional = mod(distance, this.length) / this.ds, index = Math.floor(fractional), w = fractional - index;
      return this.xy[index].map((v, i) => v * (1 - w) + this.xy[(index + 1) % this.n][i] * w);
    }
    nearest(x, y) {
      let best = Infinity, index = 0;
      for (let i = 0; i < this.n; i++) {
        const d = (this.xy[i][0] - x) ** 2 + (this.xy[i][1] - y) ** 2;
        if (d < best) { best = d; index = i; }
      }
      return index;
    }
    locate(x, y) {
      const nearest = this.nearest(x, y);
      let best = Infinity, chosen;
      for (const index of [mod(nearest - 1, this.n), nearest]) {
        const next = (index + 1) % this.n, start = this.xy[index];
        const delta = this.xy[next].map((v, k) => v - start[k]);
        const weight = clip(((x - start[0]) * delta[0] + (y - start[1]) * delta[1]) / (delta[0] ** 2 + delta[1] ** 2), 0, 1);
        const p = start.map((v, k) => v + weight * delta[k]), distance = (x - p[0]) ** 2 + (y - p[1]) ** 2;
        if (distance < best) {
          best = distance;
          const tangent = this.tangent[index].map((v, k) => v * (1 - weight) + this.tangent[next][k] * weight);
          const length = Math.hypot(...tangent);
          chosen = [mod((index + weight) * this.ds, this.length),
            (x - p[0]) * -tangent[1] / length + (y - p[1]) * tangent[0] / length,
            Math.atan2(tangent[1], tangent[0])];
        }
      }
      return chosen;
    }
  }
  const ratios = [7.5, 5, 3.6, 2.7, 2.1, 1.65, 1.30, 1.05];
  const rpm = (speed, gear) => clip(speed * 60 / (2 * Math.PI * .33) * 3.8 * ratios[gear - 1], 3000, 12000);
  function derived(features, cfg) {
    const x = features.slice(0, 16);
    return [...x, 1 - Math.abs(x[0]) - (2.4 * Math.abs(x[1]) + cfg.car_width / 2 * Math.abs(x[2])) / (cfg.track_width / 2),
      Math.atan(7.5 * cfg.wheelbase / Math.max((x[3] * cfg.max_speed) ** 2, 1)) / cfg.max_steer];
  }
  function footprint(state, width = 1.8) {
    const c = Math.cos(state.heading), s = Math.sin(state.heading);
    return [[2.4, width / 2], [2.4, -width / 2], [-2.4, -width / 2], [-2.4, width / 2]].map(([x, y]) =>
      [state.x + x * c - y * s, state.y + x * s + y * c]);
  }
  const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
  const minus = (a, b) => [a[0] - b[0], a[1] - b[1]];
  function segmentDistance(p, a, b) {
    const d = minus(b, a), weight = clip(((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / Math.max(d[0] ** 2 + d[1] ** 2, 1e-15), 0, 1);
    return Math.hypot(p[0] - a[0] - weight * d[0], p[1] - a[1] - weight * d[1]);
  }
  class Environment {
    constructor(track, config, grip = 1) {
      finite(grip, "Grip");
      if (grip < .2 || grip > 1.2) throw new Error("Grip must be between 0.2 and 1.2");
      this.track = track;
      this.config = { ...config, max_braking: config.max_braking * grip, max_lateral_acceleration: config.max_lateral_acceleration * grip };
      this.grip = grip;
      this.reset({ fraction: 0, lateral: 0, heading_error: 0, speed: 12 });
    }
    makeState(scenario) {
      const cfg = this.config, distance = scenario.fraction * this.track.length;
      for (const key of ["fraction", "lateral", "heading_error", "speed"]) finite(scenario[key], key);
      if (scenario.speed < 0 || scenario.speed > cfg.max_speed) throw new Error("Starting speed is outside simulator bounds");
      const index = mod(Math.round(distance / this.track.ds), this.track.n), p = this.track.point(distance);
      let gear = 1;
      while (gear < 8 && rpm(scenario.speed, gear) > 9000) gear++;
      return { x: p[0] + this.track.normal[index][0] * scenario.lateral, y: p[1] + this.track.normal[index][1] * scenario.lateral,
        heading: this.track.heading[index] + scenario.heading_error, speed: scenario.speed, steering: 0, gear,
        rpm: rpm(scenario.speed, gear), throttle: 0, brake: 0, acceleration: 0, targetSpeed: scenario.speed,
        requestedSteering: 0, lateralAcceleration: 0, shiftActive: false, shiftRemaining: 0, shiftCooldown: 0 };
    }
    reset(scenario) {
      this.state = this.makeState(scenario);
      this.margin = this.clearance(this.state);
      this.contact = this.margin <= 1e-8;
      this.time = 0;
      this.progress = 0;
      this.previous = this.track.locate(this.state.x, this.state.y)[0];
    }
    observe(state = this.state) {
      const cfg = this.config, location = this.track.locate(state.x, state.y), [distance, lateral, roadHeading] = location;
      const error = wrap(state.heading - roadHeading), c = Math.cos(state.heading), s = Math.sin(state.heading);
      const features = [lateral / (cfg.track_width / 2), Math.sin(error), Math.cos(error), state.speed / cfg.max_speed, state.steering / cfg.max_steer];
      for (const ahead of [10, 20, 40, 70]) {
        const p = this.track.point(distance + ahead), dx = p[0] - state.x, dy = p[1] - state.y;
        features.push((dx * c + dy * s) / ahead, (-dx * s + dy * c) / ahead);
      }
      const indices = Array.from({ length: 22 }, (_, i) => Math.floor(mod(distance + i * 3, this.track.length) / this.track.ds));
      const lookahead = Math.floor(mod(distance + 8 + .45 * state.speed, this.track.length) / this.track.ds);
      features.push(this.track.curvature[indices[0]] / .05,
        Math.max(...indices.map(i => Math.abs(this.track.curvature[i]))) / .05, this.track.curvature[lookahead] / .05);
      return { features: derived(features, cfg), location };
    }
    teacher() {
      const state = this.state, cfg = this.config, { location } = this.observe(), distance = location[0];
      const steeringAt = lookahead => {
        const p = this.track.point(distance + lookahead), dx = p[0] - state.x, dy = p[1] - state.y;
        const left = -Math.sin(state.heading) * dx + Math.cos(state.heading) * dy;
        return clip(Math.atan2(2 * cfg.wheelbase * left, dx ** 2 + dy ** 2), -cfg.max_steer, cfg.max_steer);
      };
      let steer = steeringAt(8 + .45 * state.speed);
      const curvature = Math.max(...Array.from({ length: 22 }, (_, i) => Math.abs(this.track.curvature[Math.floor(mod(distance + i * 3, this.track.length) / this.track.ds)])));
      let target = clip(Math.min(Math.sqrt(3.5 / Math.max(curvature, .003)),
        Math.sqrt(4.5 * cfg.wheelbase / Math.max(Math.abs(Math.tan(steer)), .005)), 32), 6, cfg.max_speed);
      const edge = Math.abs(location[1]), error = wrap(state.heading - location[2]);
      if (edge > 3) {
        const limit = Math.atan(7.5 * cfg.wheelbase / Math.max(state.speed ** 2, 1));
        steer = clip(steeringAt(Math.max(6, 12 - edge)), -limit, limit);
        target = Math.min(target, Math.max(6, 15 - edge));
        if (location[1] * error > 0) target = Math.min(target, 7);
      }
      return [steer / cfg.max_steer, 2 * target / cfg.max_speed - 1];
    }
    clearance(state) {
      const polygon = footprint(state, this.config.car_width), nearest = this.track.nearest(state.x, state.y);
      let minimum = Infinity;
      for (const wall of [this.track.left, this.track.right]) {
        for (let offset = -7; offset <= 7; offset++) {
          const index = mod(nearest + offset, this.track.n), c = wall[index], d = wall[(index + 1) % this.track.n];
          const cd = minus(d, c);
          for (let edge = 0; edge < 4; edge++) {
            const a = polygon[edge], b = polygon[(edge + 1) % 4], ab = minus(b, a), denominator = cross(ab, cd);
            if (Math.abs(denominator) > 1e-12) {
              const t = cross(minus(c, a), cd) / denominator, u = cross(minus(c, a), ab) / denominator;
              if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return 0;
            }
            minimum = Math.min(minimum, segmentDistance(a, c, d), segmentDistance(c, a, b), segmentDistance(d, a, b));
          }
        }
      }
      return Math.abs(this.track.locate(state.x, state.y)[1]) > this.config.track_width / 2 ? -minimum : minimum;
    }
    step(action) {
      if (!Array.isArray(action) || action.length !== 2) throw new Error("Expected steering and target-speed commands");
      action.forEach(v => finite(v, "Action"));
      const before = this.state, cfg = this.config, dt = cfg.dt;
      let gear = before.gear, shiftRemaining = Math.max(0, before.shiftRemaining - dt), cooldown = Math.max(0, before.shiftCooldown - dt);
      const initialRPM = rpm(before.speed, gear);
      if (cooldown === 0) {
        if (initialRPM > 9000 && gear < 8) { gear++; shiftRemaining = .2; cooldown = .6; }
        else if (initialRPM < 4800 && gear > 1) { gear--; shiftRemaining = .2; cooldown = .6; }
      }
      const shiftActive = shiftRemaining > 0, currentRPM = rpm(before.speed, gear);
      const requestedSteering = clip(action[0], -1, 1) * cfg.max_steer;
      const targetSpeed = clip((action[1] + 1) * cfg.max_speed / 2, 0, cfg.max_speed);
      const request = 1.8 * (targetSpeed - before.speed);
      const throttle = clip(request / cfg.max_acceleration, 0, 1), brake = clip(-request / cfg.max_braking, 0, 1);
      const torque = 150 * (.75 + .25 * Math.exp(-(((currentRPM - 7000) / 3000) ** 2)));
      const engineAcceleration = Math.min(cfg.max_acceleration, torque * ratios[gear - 1] * 3.8 * .88 / (.33 * 950));
      const acceleration = throttle * engineAcceleration * (shiftActive ? 0 : 1) - brake * cfg.max_braking - .002 * before.speed ** 2;
      const speed = clip(before.speed + acceleration * dt, 0, cfg.max_speed);
      const steering = before.steering + clip(requestedSteering - before.steering, -cfg.steering_rate * dt, cfg.steering_rate * dt);
      const mean = (before.speed + speed) / 2, angle = mean / cfg.wheelbase * Math.tan(steering) * dt;
      const midpoint = before.heading + angle / 2;
      const after = { x: before.x + mean * Math.cos(midpoint) * dt, y: before.y + mean * Math.sin(midpoint) * dt,
        heading: wrap(before.heading + angle), speed, steering, gear, rpm: rpm(speed, gear), throttle, brake, acceleration,
        targetSpeed, requestedSteering, lateralAcceleration: speed ** 2 * Math.abs(Math.tan(steering)) / cfg.wheelbase,
        shiftActive, shiftRemaining, shiftCooldown: cooldown };
      const angleDelta = wrap(after.heading - before.heading);
      const count = Math.max(1, Math.ceil(Math.max(Math.hypot(after.x - before.x, after.y - before.y) / .5, Math.abs(angleDelta) / .025)));
      let margin = Infinity;
      for (let i = 0; i <= count; i++) {
        const f = i / count;
        margin = Math.min(margin, this.clearance({ ...after, x: before.x + f * (after.x - before.x),
          y: before.y + f * (after.y - before.y), heading: before.heading + f * angleDelta }));
      }
      this.state = after;
      this.margin = margin;
      this.contact = margin <= 1e-8;
      const location = this.track.locate(after.x, after.y);
      this.progress += wrapDistance(location[0] - this.previous, this.track.length);
      this.previous = location[0];
      this.time += dt;
      let reason = "";
      if (this.contact) reason = "wall_contact";
      else if (Math.abs(location[1]) + cfg.car_width / 2 > cfg.track_width / 2) reason = "off_track";
      else if (after.lateralAcceleration > cfg.max_lateral_acceleration) reason = "lateral_acceleration_limit";
      return { reason, location };
    }
    record() {
      const location = this.track.locate(this.state.x, this.state.y);
      return { ...this.state, time: this.time, margin: this.margin, contact: this.contact, lateral: location[1],
        headingError: wrap(this.state.heading - location[2]), progress: this.progress };
    }
    snapshot() {
      return { state: { ...this.state }, margin: this.margin, contact: this.contact, time: this.time,
        progress: this.progress, previous: this.previous };
    }
    restore(snapshot) {
      this.state = { ...snapshot.state };
      this.margin = snapshot.margin; this.contact = snapshot.contact; this.time = snapshot.time;
      this.progress = snapshot.progress; this.previous = snapshot.previous;
    }
  }
  const wrapDistance = (v, length) => mod(v + length / 2, length) - length / 2;
  function noisyFeatures(features, cfg, rng, noise = 0) {
    finite(noise, "Sensor noise");
    if (noise < 0 || noise > 3) throw new Error("Noise must be between 0 and 3");
    const x = features.slice(0, 16), lateralError = rng.normal(), headingError = rng.normal(), speedError = rng.normal();
    x[0] += noise * lateralError / (cfg.track_width / 2);
    const heading = Math.atan2(x[1], x[2]) + noise * .05 * headingError;
    x[1] = Math.sin(heading); x[2] = Math.cos(heading);
    x[3] = clip(x[3] + noise * .6 * speedError / cfg.max_speed, 0, 1);
    return derived(x, cfg);
  }
  function rule(features, cfg) {
    const steering = clip(-.08 * features[0] * cfg.track_width / 2 - .9 * Math.atan2(features[1], features[2]), -cfg.max_steer, cfg.max_steer);
    return [steering / cfg.max_steer, 2 * 14 / cfg.max_speed - 1];
  }
  function simulate(track, cfg, model, scenario, options = {}) {
    const env = new Environment(track, cfg, options.grip ?? 1), rng = options.rng ?? new RNG(options.seed ?? 1);
    const delay = options.delay ?? 0;
    if (!Number.isInteger(delay) || delay < 0 || delay > 20) throw new Error("Delay must be 0–20 observation steps");
    if (options.snapshot) env.restore(options.snapshot); else env.reset(scenario);
    const trace = [env.record()], history = [];
    let reason = env.contact ? "wall_contact" : "timeout";
    const goal = options.goal ?? 150, record = options.record === true;
    if (!env.contact) {
      for (let step = 0; step < (options.maxSteps ?? 350); step++) {
        const observed = noisyFeatures(env.observe().features, cfg, rng, options.noise ?? 0);
        history.push(observed);
        const features = history[Math.max(0, history.length - 1 - delay)];
        const action = model === "expert" ? env.teacher() : model === "rules" ? rule(features, cfg) : model.predict(features);
        if (!action.every(Number.isFinite)) { reason = "numerical_divergence"; break; }
        // Recorded on the pre-step row: what the policy saw, what it did, and what the expert would do at the true state.
        if (record) Object.assign(trace.at(-1), { features, action: action.slice(), expert: model === "expert" ? action.slice() : env.teacher() });
        const result = env.step(action);
        trace.push(env.record());
        if (result.reason || env.progress >= goal) { reason = result.reason || "section_complete"; break; }
      }
    }
    // A forked drive keeps the earlier states in the exit-centering window, exactly as the original drive would.
    const tail = (options.history ? [...options.history, ...trace] : trace).slice(-20), exitError = tail.reduce((v, s) => v + Math.abs(s.lateral), 0) / tail.length;
    return { trace, metric: { success: Number(reason === "section_complete" && exitError < 1), end_reason: reason,
      wall_contacts: Number(trace.some(s => s.contact)), duration_s: env.time, progress: clip(env.progress / goal, 0, 1),
      min_clearance_m: Math.min(...trace.map(s => s.margin)), exit_mean_lateral_m: exitError,
      peak_speed_kmh: Math.max(...trace.map(s => s.speed * 3.6)), braking_s: trace.filter(s => s.brake > .01).length * cfg.dt } };
  }
  function meanSD(values) {
    if (!values.length) throw new Error("Statistics require trials");
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return { mean, sd: values.length > 1 ? Math.sqrt(values.reduce((v, x) => v + (x - mean) ** 2, 0) / (values.length - 1)) : null };
  }
  function fingerprint(model) {
    const text = JSON.stringify(model.serialize());
    let hash = 2166136261;
    for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
    return (hash >>> 0).toString(16).padStart(8, "0");
  }
  return { version, compatibleVersions, RNG, Network, Track, Environment, featureMasks, maskFeatures, derived, validateRows,
    simulate, noisyFeatures, rule, footprint, clip, wrap, meanSD, fingerprint };
});
