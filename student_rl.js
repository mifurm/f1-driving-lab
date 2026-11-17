(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./student_engine.js"), require("./student_tools.js"));
  else root.StudentRL = factory(root.StudentEngine, root.StudentTools);
})(globalThis, function (E, T) {
  "use strict";
  // Reinforcement learning for the same car, the same circuits and the same 18 inputs as the imitation lessons.
  // There are no expert labels here. The only teaching signal is a reward score the car earns by driving, and the learner is trained by
  // proximal policy optimisation (PPO): a Gaussian policy whose mean is the same 18-24-16-2 tanh network the imitation lessons train,
  // plus a separate critic that estimates how much reward is still to come from a state.
  const { RNG, Environment, Network, clip } = E;
  const LOG_2PI = Math.log(2 * Math.PI);

  // What the car is paid for. Per control step (0.1 s):
  //   + `distance` x metres advanced along the centre line   (the point of the lap)
  //   - `centering` x (offset / half road width)^2           (a small nudge to stay near the middle)
  //   - `smoothness` x mean squared change of the two commands since the previous step
  //   - `failure` once, when the step ends the drive         (wall contact, leaving the road or exceeding the grip limit)
  // A failed drive also forfeits every later step's distance reward, which is what teaches the car to slow down for corners.
  // Without the smoothness term every final car we trained flicked its steering command back and forth on every step. Flicking costs no lap time, so nothing
  // in the reward removed it. 0.3 sits inside the window that does (about 0.2 to 0.4): lower weights let the flicker back and higher ones slow learning badly.
  const defaultReward = { distance: .1, centering: .02, smoothness: .3, failure: 1 };

  // One driving episode at a time on one circuit, started from the same "typical" random states the imitation lessons collect from.
  class DrivingEnv {
    constructor({ track, cfg, seed = 1, starts = T.defaultStarts.typical, horizon = 400, reward = defaultReward }) {
      this.track = track; this.cfg = cfg; this.rng = new RNG(seed); this.starts = starts; this.horizon = horizon;
      this.reward = { ...defaultReward, ...reward }; this.env = new Environment(track, cfg); this.half = cfg.track_width / 2; this.steps = 0; this.previous = null;
    }
    reset() {
      for (let attempt = 0; attempt < 60; attempt++) { this.env.reset(T.sampleStart(this.rng, this.starts)); if (!this.env.contact) break; }
      if (this.env.contact) throw new Error("Could not find a starting state clear of the walls");
      this.steps = 0; this.previous = null;
      return this.env.observe().features;
    }
    // Returns the next observation, the reward and whether the episode ended (terminated: the car failed; truncated: the time limit ran out).
    step(action) {
      const before = this.env.progress, result = this.env.step(action);
      this.steps++;
      const advanced = this.env.progress - before, offset = result.location[1] / this.half, failed = result.reason !== "";
      let reward = this.reward.distance * advanced - this.reward.centering * offset * offset;
      if (this.previous) reward -= this.reward.smoothness * ((action[0] - this.previous[0]) ** 2 + (action[1] - this.previous[1]) ** 2) / 2;
      this.previous = [action[0], action[1]];
      if (failed) reward -= this.reward.failure;
      return { features: this.env.observe().features, reward, advanced, terminated: failed, truncated: !failed && this.steps >= this.horizon, reason: result.reason };
    }
  }

  // ---- networks ---------------------------------------------------------------------------------------------------------------
  // Weights are stored output-major (W[j * nIn + i]) so the inner loops run over contiguous memory.
  function dense(rng, nIn, nOut, scale = 1) {
    const W = new Float64Array(nIn * nOut);
    // Drawn input-major, exactly like E.Network, so an actor with scale 1 starts from the same weights as new E.Network(seed).
    for (let i = 0; i < nIn; i++) for (let j = 0; j < nOut; j++) W[j * nIn + i] = rng.normal() / Math.sqrt(nIn) * scale;
    return { nIn, nOut, W, b: new Float64Array(nOut), gW: new Float64Array(nIn * nOut), gb: new Float64Array(nOut) };
  }
  function forwardLayer(layer, x, out, squash) {
    const { nIn, nOut, W, b } = layer;
    for (let j = 0; j < nOut; j++) {
      let sum = b[j];
      const row = j * nIn;
      for (let i = 0; i < nIn; i++) sum += x[i] * W[row + i];
      out[j] = squash ? Math.tanh(sum) : sum;
    }
  }
  // Accumulates the layer's parameter gradients and (when `dx` is given) the gradient with respect to its input.
  // `dOut` is the gradient with respect to the layer's output after the activation.
  function backwardLayer(layer, x, out, dOut, dx, squash) {
    const { nIn, nOut, W, gW, gb } = layer;
    if (dx) dx.fill(0);
    for (let j = 0; j < nOut; j++) {
      const dz = squash ? dOut[j] * (1 - out[j] * out[j]) : dOut[j], row = j * nIn;
      gb[j] += dz;
      for (let i = 0; i < nIn; i++) { gW[row + i] += x[i] * dz; if (dx) dx[i] += W[row + i] * dz; }
    }
  }

  class ActorCritic {
    constructor(seed = 1, { logStd = Math.log(.4), outputScale = .1 } = {}) {
      const rng = new RNG(seed);
      this.actor = [dense(rng, 18, 24), dense(rng, 24, 16), dense(rng, 16, 2, outputScale)];
      this.critic = [dense(rng, 18, 24), dense(rng, 24, 16), dense(rng, 16, 1)];
      this.logStd = new Float64Array(2).fill(logStd); this.gLogStd = new Float64Array(2);
      this.scratch = { a1: new Float64Array(24), a2: new Float64Array(16), mean: new Float64Array(2), c1: new Float64Array(24), c2: new Float64Array(16), value: new Float64Array(1),
        d1: new Float64Array(24), d2: new Float64Array(16), dIn: new Float64Array(18), dMean: new Float64Array(2), dValue: new Float64Array(1) };
      this.updates = 0;
      const arrays = [];
      for (const layer of [...this.actor, ...this.critic]) arrays.push([layer.W, layer.gW], [layer.b, layer.gb]);
      arrays.push([this.logStd, this.gLogStd]);
      this.parameters = arrays.map(([p]) => p); this.gradients = arrays.map(([, g]) => g);
      this.m = this.parameters.map(p => new Float64Array(p.length)); this.v = this.parameters.map(p => new Float64Array(p.length));
    }
    forwardActor(x) {
      const s = this.scratch;
      forwardLayer(this.actor[0], x, s.a1, true); forwardLayer(this.actor[1], s.a1, s.a2, true); forwardLayer(this.actor[2], s.a2, s.mean, true);
      return s.mean;
    }
    forwardCritic(x) {
      const s = this.scratch;
      forwardLayer(this.critic[0], x, s.c1, true); forwardLayer(this.critic[1], s.c1, s.c2, true); forwardLayer(this.critic[2], s.c2, s.value, false);
      return s.value[0];
    }
    value(x) { return this.forwardCritic(x); }
    // Samples an action around the policy mean. The environment clips actions to [-1, 1]; the log-probability is of the unclipped sample.
    act(x, rng) {
      const mean = this.forwardActor(x), value = this.forwardCritic(x), action = [0, 0];
      let logp = 0;
      for (let d = 0; d < 2; d++) {
        const sigma = Math.exp(this.logStd[d]), noise = rng.normal();
        action[d] = mean[d] + sigma * noise;
        logp += -.5 * noise * noise - this.logStd[d] - .5 * LOG_2PI;
      }
      return { action, logp, value, mean: [mean[0], mean[1]] };
    }
    // The driver the lessons replay: the policy mean (no exploration noise) as an ordinary E.Network, so it can be driven, fingerprinted and exported like any other.
    actorNetwork() {
      const net = new Network(1);
      this.actor.forEach((layer, l) => {
        for (let i = 0; i < layer.nIn; i++) for (let j = 0; j < layer.nOut; j++) net.weights[l][i][j] = layer.W[j * layer.nIn + i];
        for (let j = 0; j < layer.nOut; j++) net.biases[l][j] = layer.b[j];
      });
      net.resetOptimizer(); net.updates = this.updates;
      return net;
    }
    zeroGradients() { for (const g of this.gradients) g.fill(0); }
    gradientNorm() { let s = 0; for (const g of this.gradients) for (let i = 0; i < g.length; i++) s += g[i] * g[i]; return Math.sqrt(s); }
    // Adam over every actor, log-std and critic parameter, with the usual global gradient-norm limit.
    applyAdam(learningRate, maxGradNorm = .5) {
      const norm = this.gradientNorm();
      if (!Number.isFinite(norm)) throw new Error("numerical_divergence: non-finite gradient; parameters were left unchanged.");
      const scale = maxGradNorm && norm > maxGradNorm ? maxGradNorm / norm : 1;
      this.updates++;
      const c1 = 1 - .9 ** this.updates, c2 = 1 - .999 ** this.updates;
      this.parameters.forEach((p, k) => {
        const g = this.gradients[k], m = this.m[k], v = this.v[k];
        for (let i = 0; i < p.length; i++) {
          const gi = g[i] * scale;
          m[i] = .9 * m[i] + .1 * gi; v[i] = .999 * v[i] + .001 * gi * gi;
          p[i] -= learningRate * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
        }
      });
      for (let d = 0; d < 2; d++) this.logStd[d] = clip(this.logStd[d], Math.log(.05), Math.log(1));
      return norm;
    }
    // Gradient of the PPO loss over `indices` of `batch`:  L = clipped policy loss + valueCoef * 0.5 (V - return)^2 - entropyCoef * entropy.
    // Fills this.gradients and returns the loss pieces plus the approximate KL divergence (ratio - 1 - log ratio, always >= 0) from the policy that collected the data.
    ppoGradient(batch, indices, { clipRange = .2, valueCoef = .5, entropyCoef = 0 } = {}) {
      this.zeroGradients();
      const s = this.scratch, B = indices.length, x = new Float64Array(18), sigma = [Math.exp(this.logStd[0]), Math.exp(this.logStd[1])];
      let policyLoss = 0, valueLoss = 0, kl = 0, clipped = 0;
      for (const n of indices) {
        for (let i = 0; i < 18; i++) x[i] = batch.obs[n * 18 + i];
        const mean = this.forwardActor(x), value = this.forwardCritic(x), adv = batch.advantages[n];
        let logp = 0; const z = [0, 0];
        for (let d = 0; d < 2; d++) { z[d] = (batch.actions[n * 2 + d] - mean[d]) / sigma[d]; logp += -.5 * z[d] * z[d] - this.logStd[d] - .5 * LOG_2PI; }
        const ratio = Math.exp(logp - batch.logp[n]), low = 1 - clipRange, high = 1 + clipRange;
        policyLoss += -Math.min(ratio * adv, clip(ratio, low, high) * adv);
        kl += ratio - 1 - Math.log(ratio);
        const active = !((adv > 0 && ratio > high) || (adv < 0 && ratio < low));
        if (!active) clipped++;
        const dLogp = active ? -adv * ratio / B : 0;
        for (let d = 0; d < 2; d++) {
          s.dMean[d] = dLogp * z[d] / sigma[d];
          this.gLogStd[d] += dLogp * (z[d] * z[d] - 1) - entropyCoef / B;
        }
        backwardLayer(this.actor[2], s.a2, s.mean, s.dMean, s.d2, true);
        backwardLayer(this.actor[1], s.a1, s.a2, s.d2, s.d1, true);
        backwardLayer(this.actor[0], x, s.a1, s.d1, null, true);
        const error = value - batch.returns[n];
        valueLoss += .5 * error * error;
        s.dValue[0] = valueCoef * error / B;
        backwardLayer(this.critic[2], s.c2, s.value, s.dValue, s.d2, false);
        backwardLayer(this.critic[1], s.c1, s.c2, s.d2, s.d1, true);
        backwardLayer(this.critic[0], x, s.c1, s.d1, null, true);
      }
      const entropy = this.logStd[0] + this.logStd[1] + LOG_2PI + 1;
      return { policyLoss: policyLoss / B, valueLoss: valueLoss / B, entropy, kl: kl / B, clipFraction: clipped / B };
    }
    serialize() { return { actor: this.actorNetwork().serialize(), logStd: Array.from(this.logStd), updates: this.updates }; }
  }

  // ---- advantage estimation -----------------------------------------------------------------------------------------------------
  // Generalised advantage estimation for one trajectory of `length` steps. `done[t]` marks the last step of an episode (failure or time limit; a
  // time-limit step has already had gamma * V(next state) added to its reward). `lastValue` is V of the state after the final step.
  function gae({ rewards, values, done, lastValue, gamma = .99, lambda = .95, offset = 0, length = rewards.length }) {
    const advantages = new Float64Array(length), returns = new Float64Array(length);
    let next = 0;
    for (let t = length - 1; t >= 0; t--) {
      const live = 1 - done[offset + t], nextValue = t === length - 1 ? lastValue : values[offset + t + 1];
      const delta = rewards[offset + t] + gamma * nextValue * live - values[offset + t];
      next = delta + gamma * lambda * live * next;
      advantages[t] = next; returns[t] = next + values[offset + t];
    }
    return { advantages, returns };
  }

  // ---- the training loop ---------------------------------------------------------------------------------------------------------
  // `envs` independent drives of `rolloutSteps` control steps each are collected with the current policy, advantages are estimated, and the policy and critic
  // are improved for `epochs` passes over that fresh data. Every iteration therefore uses `envs * rolloutSteps` simulated driving steps.
  // The settings behind every number in module 12. Each iteration collects envs x rolloutSteps simulated steps; the learning rate falls linearly to zero over the run.
  const ppoDefaults = { iterations: 100, envs: 8, rolloutSteps: 256, epochs: 4, minibatch: 256, learningRate: 3e-4, anneal: true, gamma: .99, lambda: .95, clipRange: .2,
    entropyCoef: 0, valueCoef: .5, maxGradNorm: .5, horizon: 400, logStd: Math.log(.4) };
  async function ppoTrain({ track, cfg, seed = 1, reward = defaultReward, starts = T.defaultStarts.typical, agent = null, onIteration = null, shouldCancel = null, yieldFn = async () => {}, ...options }) {
    for (const key of Object.keys(options)) if (!Object.hasOwn(ppoDefaults, key)) throw new Error(`Unknown PPO setting "${key}"`);
    const { iterations, envs, rolloutSteps, epochs, minibatch, learningRate, anneal, gamma, lambda, clipRange, entropyCoef, valueCoef, maxGradNorm, horizon, logStd } = { ...ppoDefaults, ...options };
    const ac = agent ?? new ActorCritic(seed, { logStd });
    const rngAct = new RNG(seed * 31 + 7), rngShuffle = new RNG(seed * 57 + 13);
    const drivers = Array.from({ length: envs }, (_, k) => new DrivingEnv({ track, cfg, seed: seed * 1009 + k * 101 + 3, starts, horizon, reward }));
    let obs = drivers.map(d => Float64Array.from(d.reset()));
    const N = envs * rolloutSteps, batch = { obs: new Float64Array(N * 18), actions: new Float64Array(N * 2), logp: new Float64Array(N), advantages: new Float64Array(N), returns: new Float64Array(N) };
    const rewards = new Float64Array(N), values = new Float64Array(N), done = new Float64Array(N), history = [];
    const running = new Float64Array(envs), runningLength = new Float64Array(envs), runningDistance = new Float64Array(envs);
    let steps = 0, stopped = "completed";
    for (let iteration = 1; iteration <= iterations; iteration++) {
      if (shouldCancel?.()) { stopped = "cancelled"; break; }
      const finished = { returns: [], lengths: [], distances: [], failures: 0 };
      for (let t = 0; t < rolloutSteps; t++) {
        for (let e = 0; e < envs; e++) {
          const n = e * rolloutSteps + t, driver = drivers[e];
          const sample = ac.act(obs[e], rngAct);
          batch.obs.set(obs[e], n * 18); batch.actions[n * 2] = sample.action[0]; batch.actions[n * 2 + 1] = sample.action[1];
          batch.logp[n] = sample.logp; values[n] = sample.value;
          const result = driver.step([clip(sample.action[0], -1, 1), clip(sample.action[1], -1, 1)]);
          let r = result.reward;
          running[e] += result.reward; runningLength[e]++; runningDistance[e] += result.advanced;
          if (result.truncated) r += gamma * ac.value(result.features);
          rewards[n] = r; done[n] = result.terminated || result.truncated ? 1 : 0;
          if (result.terminated || result.truncated) {
            finished.returns.push(running[e]); finished.lengths.push(runningLength[e]); finished.distances.push(runningDistance[e]); if (result.terminated) finished.failures++;
            running[e] = 0; runningLength[e] = 0; runningDistance[e] = 0;
            obs[e] = Float64Array.from(driver.reset());
          } else obs[e] = Float64Array.from(result.features);
        }
      }
      steps += N;
      let total = 0, totalSquares = 0;
      for (let e = 0; e < envs; e++) {
        const { advantages, returns } = gae({ rewards, values, done, lastValue: ac.value(obs[e]), gamma, lambda, offset: e * rolloutSteps, length: rolloutSteps });
        batch.advantages.set(advantages, e * rolloutSteps); batch.returns.set(returns, e * rolloutSteps);
      }
      for (let n = 0; n < N; n++) { total += batch.advantages[n]; totalSquares += batch.advantages[n] ** 2; }
      const mean = total / N, spread = Math.sqrt(Math.max(totalSquares / N - mean * mean, 0)) + 1e-8;
      for (let n = 0; n < N; n++) batch.advantages[n] = (batch.advantages[n] - mean) / spread;
      const rate = anneal ? learningRate * (1 - (iteration - 1) / iterations) : learningRate;
      let stats = null;
      const order = Array.from({ length: N }, (_, i) => i);
      for (let epoch = 0; epoch < epochs; epoch++) {
        rngShuffle.shuffle(order);
        for (let start = 0; start + minibatch <= N; start += minibatch) {
          stats = ac.ppoGradient(batch, order.slice(start, start + minibatch), { clipRange, valueCoef, entropyCoef });
          ac.applyAdam(rate, maxGradNorm);
        }
      }
      const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
      const point = { iteration, steps, episodes: finished.returns.length, meanReturn: average(finished.returns), meanLength: average(finished.lengths), meanDistance: average(finished.distances),
        failureRate: finished.returns.length ? finished.failures / finished.returns.length : null, std: Math.exp(ac.logStd[0]), learningRate: rate, ...stats };
      history.push(point);
      await onIteration?.({ iteration, steps, agent: ac, stats: point, history });
      await yieldFn();
    }
    return { agent: ac, history, steps, stopped };
  }

  return { defaultReward, ppoDefaults, DrivingEnv, ActorCritic, gae, ppoTrain };
});
