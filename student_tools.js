(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./student_engine.js"));
  else root.StudentTools = factory(root.StudentEngine);
})(globalThis, function (E) {
  "use strict";
  const { RNG, Network, Environment, clip } = E;
  const Z95 = 1.959963984540054, CURVATURE_SCALE = 0.05;
  const average = values => values.reduce((a, b) => a + b, 0) / values.length;
  const noYield = async () => {};

  // ---- statistics -------------------------------------------------------------------------------------------------
  function wilson(k, n, z = Z95) {
    if (!Number.isInteger(k) || !Number.isInteger(n) || n < 1 || k < 0 || k > n) throw new Error("Wilson interval needs integers with 0 <= k <= n and n >= 1");
    const p = k / n, z2 = z * z, denominator = 1 + z2 / n, centre = p + z2 / (2 * n);
    const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
    return { k, n, p, lo: Math.max(0, (centre - half) / denominator), hi: Math.min(1, (centre + half) / denominator) };
  }
  // Linear interpolation between order statistics (NumPy's default percentile method).
  function quantile(sorted, q) {
    if (!sorted.length) throw new Error("Quantile of an empty sample");
    const position = (sorted.length - 1) * q, lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  }
  function bootstrapMean(values, rng, { iterations = 2000, alpha = .05, resamples = null } = {}) {
    if (!values.length) throw new Error("Bootstrap needs at least one value");
    const mean = average(values);
    if (values.length === 1) return { n: 1, mean, lo: mean, hi: mean };
    const means = [];
    for (let b = 0; b < (resamples ? resamples.length : iterations); b++) {
      let sum = 0;
      for (let i = 0; i < values.length; i++) sum += values[resamples ? resamples[b][i] : rng.index(values.length)];
      means.push(sum / values.length);
    }
    means.sort((a, b) => a - b);
    return { n: values.length, mean, lo: quantile(means, alpha / 2), hi: quantile(means, 1 - alpha / 2) };
  }
  // Two-sided exact binomial (p = 0.5) test on the discordant pairs of a paired comparison.
  function signTest(wins, losses) {
    const n = wins + losses;
    if (n === 0) return { n, p: 1 };
    const logFactorial = [0];
    for (let i = 1; i <= n; i++) logFactorial.push(logFactorial[i - 1] + Math.log(i));
    let tail = 0;
    for (let i = 0; i <= Math.min(wins, losses); i++)
      tail += Math.exp(logFactorial[n] - logFactorial[i] - logFactorial[n - i] - n * Math.LN2);
    return { n, p: Math.min(1, 2 * tail) };
  }
  // Paired comparison of two equally long outcome lists (per seed rates, or 0/1 successes per paired trial).
  function pairedDifference(a, b, rng, options = {}) {
    if (a.length !== b.length || !a.length) throw new Error("Paired outcomes need equal, non-empty lists");
    const differences = a.map((v, i) => v - b[i]);
    const wins = differences.filter(d => d > 0).length, losses = differences.filter(d => d < 0).length;
    return { ...bootstrapMean(differences, rng, options), wins, losses, ties: differences.length - wins - losses, p: signTest(wins, losses).p };
  }
  function summarizeSeeds(values, rng, options = {}) {
    const stats = bootstrapMean(values, rng, options), mean = stats.mean;
    return { ...stats, sd: values.length > 1 ? Math.sqrt(values.reduce((v, x) => v + (x - mean) ** 2, 0) / (values.length - 1)) : null,
      min: Math.min(...values), max: Math.max(...values), values: values.slice() };
  }

  // ---- training ----------------------------------------------------------------------------------------------------
  function learningRateAt(schedule, base, update, total) {
    const progress = total > 1 ? (update - 1) / (total - 1) : 0;
    if (schedule === "constant") return base;
    if (schedule === "linear") return base * (1 - .9 * progress);
    if (schedule === "cosine") return base * (.1 + .9 * .5 * (1 + Math.cos(Math.PI * progress)));
    if (schedule === "step") return base * .5 ** Math.min(2, Math.floor(3 * progress));
    throw new Error(`Unknown learning-rate schedule "${schedule}"`);
  }
  function modelLoss(model, rows) {
    if (!rows.length) return null;
    let total = 0;
    for (const row of rows) {
      const p = model.predict(row.features);
      total += 4 * (p[0] - row.targets[0]) ** 2 + (p[1] - row.targets[1]) ** 2;
    }
    return total / (2 * rows.length);
  }
  const subsample = (rows, limit) => rows.length <= limit ? rows : rows.filter((_, i) => i % Math.ceil(rows.length / limit) === 0);
  // Mini-batch Adam with optional schedule, decay, clipping, early stopping and a closed-loop probe.
  async function fit(options) {
    const o = { seed: 1, sizes: [18, 24, 16, 2], mask: "full", updates: 80, batchSize: 64, learningRate: .001, schedule: "constant",
      weightDecay: 0, maxGradNorm: null, evalEvery: 4, patience: 0, probe: null, probeEvery: 20, lossSample: 1024,
      stress: [], onProgress: null, shouldCancel: null, yieldFn: noYield, rngSeed: null, initial: null, ...options };
    const { train, validation = [] } = o;
    if (!train?.length) throw new Error("Training needs at least one example");
    if (!Number.isInteger(o.batchSize) || o.batchSize < 1) throw new Error("Batch size must be a positive integer");
    const net = o.initial ?? new Network(o.seed, o.sizes, o.mask), rng = new RNG(o.rngSeed ?? o.seed + 100);
    const trainSample = subsample(train, o.lossSample), stress = o.stress ?? [];
    const point = (update, rate) => ({ update, train: modelLoss(net, trainSample), validation: modelLoss(net, validation),
      stress: modelLoss(net, stress), learningRate: rate });
    const history = [point(net.updates, null)], first = net.updates;
    if (o.probe) history[0].probe = await o.probe(net, 0);
    let best = net.clone(), bestLoss = history[0].validation ?? Infinity, bestUpdate = first, sinceBest = 0, stopped = "completed", divergence = null;
    for (let step = 1; step <= o.updates; step++) {
      if (o.shouldCancel?.()) { stopped = "cancelled"; break; }
      const rate = learningRateAt(o.schedule, o.learningRate, step, o.updates);
      const batch = Array.from({ length: o.batchSize }, () => train[rng.index(train.length)]);
      try { net.trainBatch(batch, rate, { weightDecay: o.weightDecay, maxGradNorm: o.maxGradNorm, trackChanges: false, skipLossAfter: true }); }
      catch (error) {
        if (!/numerical_divergence/.test(error.message)) throw error;
        stopped = "diverged"; divergence = { update: net.updates + 1, message: error.message }; break;
      }
      const evaluateNow = step % o.evalEvery === 0 || step === o.updates;
      const probeNow = o.probe && (step % o.probeEvery === 0 || step === o.updates);
      if (evaluateNow || probeNow) {
        const p = point(net.updates, rate);
        if (probeNow) p.probe = await o.probe(net, net.updates);
        history.push(p);
        if (p.validation !== null && p.validation < bestLoss) { bestLoss = p.validation; best = net.clone(); bestUpdate = net.updates; sinceBest = 0; }
        else if (p.validation !== null) sinceBest++;
        o.onProgress?.(net, p, history);
        await o.yieldFn();
        if (o.patience > 0 && sinceBest >= o.patience) { stopped = "early_stop"; break; }
      }
    }
    if (!validation.length) { best = net.clone(); bestUpdate = net.updates; }
    return { net, last: net, best, bestUpdate, history, stopped, divergence };
  }

  // ---- closed-loop evaluation ---------------------------------------------------------------------------------------
  const kindOf = name => /recovery/i.test(name) ? "recovery" : "nominal";
  function sectionsFor(data, trackIds) {
    return trackIds.flatMap(id => data.scenarios[id].map((scenario, index) => ({ track_id: id, index, scenario, kind: kindOf(scenario.name) })));
  }
  function tally(rows) {
    const count = predicate => { const subset = rows.filter(predicate); return { k: subset.reduce((n, r) => n + r.success, 0), n: subset.length }; };
    return { ...count(() => true), nominal: count(r => r.kind === "nominal"), recovery: count(r => r.kind === "recovery"),
      contacts: rows.reduce((n, r) => n + r.wall_contacts, 0), handling: rows.filter(r => r.end_reason === "lateral_acceleration_limit").length,
      // A start whose footprint already touches a wall fails at t = 0 for every policy; it is counted, never silently dropped.
      invalid: rows.filter(r => r.end_reason === "wall_contact" && r.duration_s === 0).length,
      progress: rows.length ? average(rows.map(r => r.progress)) : 0 };
  }
  // ctx = { cfg, tracks: Map(id -> Track) }; options.visited collects the states the policy actually drove through.
  function evaluateSections(model, sections, ctx, options = {}) {
    const rows = [], visited = options.visited ? [] : null;
    for (const section of sections) {
      const track = ctx.tracks.get(section.track_id), result = E.simulate(track, ctx.cfg, model, section.scenario, options.sim ?? {});
      rows.push({ track_id: section.track_id, name: section.scenario.name, kind: section.kind, ...result.metric });
      if (visited) for (const s of result.trace) {
        const [distance] = track.locate(s.x, s.y);
        visited.push({ lateral: s.lateral, heading: s.headingError, speed: s.speed, kind: section.kind,
          curvature: track.curvature[E.sampleIndex(distance / track.ds, track.n)] / CURVATURE_SCALE });
      }
    }
    return { rows, visited, ...tally(rows) };
  }

  // ---- baselines -----------------------------------------------------------------------------------------------------
  function solve(matrix, rhs) {
    const n = matrix.length, a = matrix.map((row, i) => [...row, ...rhs[i]]);
    for (let c = 0; c < n; c++) {
      let pivot = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
      if (Math.abs(a[pivot][c]) < 1e-14) throw new Error("Singular system in the linear baseline; raise the ridge penalty");
      [a[c], a[pivot]] = [a[pivot], a[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const factor = a[r][c] / a[c][c];
        for (let k = c; k < a[r].length; k++) a[r][k] -= factor * a[c][k];
      }
    }
    return a.map((row, i) => row.slice(n).map(v => v / row[i]));
  }
  class RidgeModel {
    constructor(weights, mask, lambda) { this.weights = weights; this.mask = mask; this.lambda = lambda; this.updates = 0; this.kind = "ridge"; }
    static fit(rows, { lambda = .01, mask = "full" } = {}) {
      if (!rows.length) throw new Error("The linear baseline needs examples");
      if (!(lambda > 0)) throw new Error("Ridge penalty must be positive");
      const d = 18, size = d + 1, gram = Array.from({ length: size }, () => Array(size).fill(0)), cross = Array.from({ length: size }, () => [0, 0]);
      for (const row of rows) {
        const x = [...E.maskFeatures(row.features, mask), 1];
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) gram[i][j] += x[i] * x[j];
          cross[i][0] += x[i] * row.targets[0]; cross[i][1] += x[i] * row.targets[1];
        }
      }
      const m = rows.length;
      for (let i = 0; i < size; i++) { for (let j = 0; j < size; j++) gram[i][j] /= m; cross[i][0] /= m; cross[i][1] /= m; if (i < d) gram[i][i] += lambda; }
      return new RidgeModel(solve(gram, cross), mask, lambda);
    }
    predict(features) {
      const x = [...E.maskFeatures(features, this.mask), 1];
      return [0, 1].map(j => clip(x.reduce((sum, v, i) => sum + v * this.weights[i][j], 0), -1, 1));
    }
    serialize() { return { version: E.version, kind: this.kind, mask: this.mask, lambda: this.lambda, weights: this.weights.map(r => r.slice()), updates: 0 }; }
    clone() { return new RidgeModel(this.weights.map(r => r.slice()), this.mask, this.lambda); }
  }
  class KNNModel {
    constructor(points, targets, k, mask) { this.points = points; this.targets = targets; this.k = k; this.mask = mask; this.updates = 0; this.kind = "knn"; this.count = targets.length / 2; }
    static fit(rows, { k = 5, mask = "full" } = {}) {
      if (!rows.length) throw new Error("The nearest-neighbour baseline needs examples");
      if (!Number.isInteger(k) || k < 1 || k > rows.length) throw new Error("k must be an integer between 1 and the number of examples");
      const points = new Float64Array(rows.length * 18), targets = new Float64Array(rows.length * 2);
      rows.forEach((row, i) => { points.set(E.maskFeatures(row.features, mask), i * 18); targets.set(row.targets, i * 2); });
      return new KNNModel(points, targets, k, mask);
    }
    predict(features) {
      const x = E.maskFeatures(features, this.mask), best = [];
      for (let i = 0; i < this.count; i++) {
        let distance = 0;
        for (let d = 0; d < 18; d++) { const delta = this.points[i * 18 + d] - x[d]; distance += delta * delta; }
        if (best.length < this.k || distance < best[best.length - 1][0]) {
          let at = best.length;
          while (at > 0 && best[at - 1][0] > distance) at--;
          best.splice(at, 0, [distance, i]);
          if (best.length > this.k) best.pop();
        }
      }
      return [0, 1].map(j => best.reduce((sum, [, i]) => sum + this.targets[i * 2 + j], 0) / best.length);
    }
    serialize() {
      let checksum = 0;
      for (let i = 0; i < this.points.length; i++) checksum += this.points[i] * ((i % 7) + 1);
      return { version: E.version, kind: this.kind, mask: this.mask, k: this.k, examples: this.count, checksum, updates: 0 };
    }
    clone() { return new KNNModel(this.points, this.targets, this.k, this.mask); }
  }

  // ---- data generation and the coverage lab -------------------------------------------------------------------------
  const defaultStarts = {
    narrow: { lateral: 1, heading: .04, speed: [8, 14] },
    typical: { lateral: 3, heading: .12, speed: [8, 16] },
    broad: { lateral: 4, heading: .16, speed: [8, 18] },
    stress: { outward: true, lateral: [5.5, 6.6], heading: [.18, .30], speed: [8, 16] },
  };
  function sampleStart(rng, spec) {
    const speed = rng.range(spec.speed[0], spec.speed[1]), fraction = rng.uniform();
    if (spec.outward) {
      const side = rng.uniform() < .5 ? -1 : 1;
      return { fraction, lateral: side * rng.range(...spec.lateral), heading_error: side * rng.range(...spec.heading), speed };
    }
    return { fraction, lateral: rng.range(-spec.lateral, spec.lateral), heading_error: rng.range(-spec.heading, spec.heading), speed };
  }
  // Returns how many sampled starts were rejected because the car footprint already touched a wall.
  function startWithoutContact(env, rng, spec) {
    for (let attempt = 0; attempt < 60; attempt++) { env.reset(sampleStart(rng, spec)); if (!env.contact) return attempt; }
    throw new Error("Could not find a starting state clear of the walls");
  }
  // Label every visited state with the expert; `driver` ("expert" or a model) decides what the car actually does.
  function collect({ track, cfg, driver, rng, steps, spec, episodeSteps = 150, prefix = "ep", firstEpisode = 0, source = "expert demonstration", grip = 1 }) {
    const env = new Environment(track, cfg, grip), rows = [];
    let episode = firstEpisode, inEpisode = 0, failures = 0;
    let rejected = startWithoutContact(env, rng, spec);
    while (rows.length < steps) {
      const { features } = env.observe(), label = env.teacher();
      rows.push({ id: `${prefix}-t${track.id}-e${episode}-s${inEpisode}`, episode: `${prefix}-t${track.id}-e${episode}`, features, targets: label,
        split: "train", track_id: track.id, source });
      const action = driver === "expert" ? label : driver.predict(features);
      let ended = ++inEpisode >= episodeSteps;
      if (!action.every(Number.isFinite)) { failures++; ended = true; }
      else if (env.step(action).reason) { failures++; ended = true; }
      if (ended && rows.length < steps) { episode++; inEpisode = 0; rejected += startWithoutContact(env, rng, spec); }
    }
    return { rows, failures, rejected, episodes: episode - firstEpisode + 1 };
  }

  // "Projected occupancy": a cell of (offset, heading error, speed, signed curvature) counts as supported when at least
  // `minRows` examples from at least `minEpisodes` distinct episodes fall in it. It ignores previews and previous steering.
  const cell = (lateral, heading, speed, curvature) =>
    `${Math.floor(lateral / .75)}|${Math.floor(heading / .05)}|${Math.floor(speed / 3)}|${clip(Math.floor(curvature * 2), -4, 3)}`;
  function supportIndex(rows, cfg, { minRows = 3, minEpisodes = 2 } = {}) {
    const cells = new Map();
    for (const row of rows) {
      const key = cell(row.features[0] * cfg.track_width / 2, Math.atan2(row.features[1], row.features[2]), row.features[3] * cfg.max_speed, row.features[13]);
      const entry = cells.get(key) ?? { rows: 0, episodes: new Set() };
      entry.rows++; entry.episodes.add(row.episode); cells.set(key, entry);
    }
    return new Set([...cells].filter(([, e]) => e.rows >= minRows && e.episodes.size >= minEpisodes).map(([key]) => key));
  }
  function supportFraction(visited, support, kind = null) {
    const states = kind ? visited.filter(s => s.kind === kind) : visited;
    return states.length ? states.filter(s => support.has(cell(s.lateral, s.heading, s.speed, s.curvature))).length / states.length : null;
  }
  const inSupport = (support, state) => support.has(cell(state.lateral, state.heading, state.speed, state.curvature));
  function coverageGrid(points, { xRange, yRange, xBins, yBins }) {
    const counts = new Uint32Array(xBins * yBins);
    for (const [x, y] of points) {
      const i = Math.floor((x - xRange[0]) / (xRange[1] - xRange[0]) * xBins), j = Math.floor((y - yRange[0]) / (yRange[1] - yRange[0]) * yBins);
      if (i >= 0 && i < xBins && j >= 0 && j < yBins) counts[j * xBins + i]++;
    }
    return { counts, xBins, yBins, xRange, yRange, max: Math.max(0, ...counts) };
  }
  const featurePoint = (cfg, row) => [row.features[0] * cfg.track_width / 2, Math.atan2(row.features[1], row.features[2])];

  const arms = {
    bc_narrow: "Expert demonstrations from near-centreline starts",
    bc_broad: "Expert demonstrations from broad starts (includes recoveries)",
    bc_staged: "Staged expert control: DAgger's schedule and start distribution, but the expert drives",
    dagger: "DAgger: narrow expert demos, then the learner drives and the expert relabels",
  };
  // One arm for one seed. Every arm gets the same labelled-state budget per track and the same number of Adam updates.
  // Random streams are separate by role: network initialisation (seed), mini-batch order (seed + 100 + updates so far),
  // data collection (seed and arm). Evaluation uses fixed sections, so it consumes no randomness.
  async function runArm({ arm, seed, ctx, trackIds, budget = 100, rounds = 4, updatesPerRound = 120, starts = defaultStarts, learningRate = .001,
      batchSize = 64, sizes = [18, 24, 16, 2], mask = "full", yieldFn = noYield, shouldCancel = null, onRound = null }) {
    if (!Object.hasOwn(arms, arm)) throw new Error(`Unknown data strategy "${arm}"`);
    const rng = new RNG(seed * 7919 + Object.keys(arms).indexOf(arm) * 104729 + 17), rows = [], log = [];
    let episode = 0;
    const gather = async (driver, spec, steps, tag, source) => {
      let failed = 0, added = 0, rejected = 0;
      for (const id of trackIds) {
        const part = collect({ track: ctx.tracks.get(id), cfg: ctx.cfg, driver, rng, steps, spec, prefix: tag, firstEpisode: episode, source });
        episode += part.episodes; failed += part.failures; rejected += part.rejected; added += part.rows.length; rows.push(...part.rows);
        await yieldFn();
      }
      return { added, failed, rejected };
    };
    let diverged = false;
    const train = async (net, updates) => {
      const result = await fit({ seed, mask, train: rows, updates, batchSize, learningRate, initial: net, evalEvery: updates, lossSample: 256,
        rngSeed: seed + 100 + net.updates, shouldCancel });
      if (result.stopped === "diverged") diverged = true;
      return result;
    };
    let net = new Network(seed, sizes, mask);
    if (arm === "dagger" || arm === "bc_staged") {
      const initial = await gather("expert", starts.narrow, budget, "r0", "expert demonstration (round 0)");
      net = (await train(net, updatesPerRound)).net;
      log.push({ round: 0, ...initial, size: rows.length });
      onRound?.({ round: 0, net, rows, ...initial });
      for (let round = 1; round <= rounds; round++) {
        if (shouldCancel?.()) break;
        const got = arm === "dagger"
          ? await gather(net, starts.typical, budget, `r${round}`, `learner rollout, expert label (round ${round})`)
          : await gather("expert", starts.typical, budget, `r${round}`, `expert demonstration, typical starts (round ${round})`);
        net = (await train(net, updatesPerRound)).net;
        log.push({ round, ...got, size: rows.length });
        onRound?.({ round, net, rows, ...got });
      }
    } else {
      const got = await gather("expert", arm === "bc_narrow" ? starts.narrow : starts.broad, budget * (rounds + 1), "bc",
        arm === "bc_narrow" ? "expert demonstration (narrow)" : "expert demonstration (broad)");
      net = (await train(net, updatesPerRound * (rounds + 1))).net;
      log.push({ round: 0, ...got, size: rows.length });
    }
    return { arm, seed, net, rows, rounds: log, diverged };
  }

  // The whole coverage-lab experiment as one function, so the browser and the headless reference run share exactly the same code.
  // Every recipe gets the same labelled-state budget per track and the same number of updates; nominal and recovery sections stay separate.
  async function coverageExperiment({ data, ctx, arms, seedBase = 601, seedCount, budget, rounds, updatesPerRound, trainIds, validIds, holdoutIds,
      yieldFn = noYield, shouldCancel = null, onRun = null }) {
    const trainSections = sectionsFor(data, validIds), holdSections = sectionsFor(data, holdoutIds);
    const expertTrain = evaluateSections("expert", trainSections, ctx, { visited: true }), expertHold = evaluateSections("expert", holdSections, ctx);
    const results = Object.fromEntries(arms.map(arm => [arm, []])), keep = {}, rng = new RNG(41);
    for (const arm of arms) for (let i = 0; i < seedCount; i++) {
      const seed = seedBase + i;
      const run = await runArm({ arm, seed, ctx, trackIds: trainIds, budget, rounds, updatesPerRound, yieldFn, shouldCancel });
      if (shouldCancel?.()) throw new Error("Cancelled: no result was recorded.");
      const ev = evaluateSections(run.net, trainSections, ctx, { visited: true }), hold = evaluateSections(run.net, holdSections, ctx), support = supportIndex(run.rows, ctx.cfg);
      results[arm].push({ seed, nominal: ev.nominal.k / ev.nominal.n, recovery: ev.recovery.k / ev.recovery.n, hold: hold.k / hold.n, overall: ev.k / ev.n,
        supportReference: supportFraction(expertTrain.visited, support), supportVisited: supportFraction(ev.visited, support),
        failed: run.rounds.reduce((n, r) => n + r.failed, 0), rejected: run.rounds.reduce((n, r) => n + r.rejected, 0), diverged: run.diverged, labelled: run.rows.length, updates: run.net.updates });
      if (i === 0) keep[arm] = { rows: run.rows, visited: ev.visited.filter((_, k) => k % 6 === 0).map(s => ({ ...s, supported: inSupport(support, s) })) };
      onRun?.({ arm, seed, ev, hold });
      await yieldFn();
    }
    const summaries = arms.map(arm => {
      const r = results[arm], s = key => summarizeSeeds(r.map(x => x[key]), rng);
      return { arm, nominal: s("nominal"), recovery: s("recovery"), hold: s("hold"), overall: s("overall"), supportReference: average(r.map(x => x.supportReference)),
        supportVisited: average(r.map(x => x.supportVisited)), failed: r.reduce((n, x) => n + x.failed, 0), rejected: r.reduce((n, x) => n + x.rejected, 0), diverged: r.filter(x => x.diverged).length };
    });
    const comparisons = [["dagger", "bc_staged"], ["dagger", "bc_narrow"], ["bc_broad", "bc_narrow"], ["bc_staged", "bc_narrow"]].filter(([a, b]) => results[a] && results[b]).map(([a, b]) => ({
      a, b, overall: pairedDifference(results[a].map(x => x.overall), results[b].map(x => x.overall), rng), recovery: pairedDifference(results[a].map(x => x.recovery), results[b].map(x => x.recovery), rng) }));
    return { settings: { arms, budget, rounds, updatesPerRound, seeds: seedCount, seedBase }, results, summaries, comparisons, trainSections, holdSections,
      expert: { train: expertTrain, hold: expertHold }, keep };
  }

  // ---- paired experiments (controllers, robustness, worksheets) ---------------------------------------------------------
  function pairedStart(base, seed) {
    const rng = new RNG(seed);
    return { ...base, lateral: base.lateral + (rng.uniform() - .5) * .3, heading_error: base.heading_error + (rng.uniform() - .5) * .02,
      speed: base.speed + (rng.uniform() - .5) * .4 };
  }
  function randomSections(rng, trackIds, count, spec, label = "Random start") {
    return Array.from({ length: count }, (_, index) => {
      const start = sampleStart(rng, spec);
      return { track_id: trackIds[index % trackIds.length], index, kind: spec.outward ? "recovery" : "nominal",
        scenario: { name: `${label} ${index + 1}`, ...start } };
    });
  }
  // Every condition sees exactly the same (section, seed) starts and noise streams, so outcomes can be paired.
  // A condition may carry a live policy (a network or baseline). Rows and exports keep only its name, so a download never embeds weights or a training set.
  function describeCondition(condition) {
    const { policy, ...rest } = condition;
    return policy === undefined ? rest : { ...rest, policy: typeof policy === "string" ? policy : "model" };
  }
  async function runConditions({ conditions, sections, seeds, model, ctx, yieldFn = noYield, shouldCancel = null, onProgress = null }) {
    const rows = [];
    for (const [sectionIndex, section] of sections.entries()) {
      for (const seed of seeds) {
        const initial = pairedStart(section.scenario, seed * 1009 + sectionIndex);
        for (const condition of conditions) {
          const result = E.simulate(ctx.tracks.get(section.track_id), ctx.cfg, condition.policy ?? model, initial,
            { noise: condition.noise ?? 0, delay: condition.delay ?? 0, grip: condition.grip ?? 1, seed });
          rows.push({ condition: condition.name, seed, section: sectionIndex, track_id: section.track_id, kind: section.kind, initial, settings: describeCondition(condition), metric: result.metric });
        }
      }
      onProgress?.(sectionIndex + 1, sections.length);
      await yieldFn();
      if (shouldCancel?.()) break;
    }
    return rows;
  }
  // Trials inside one section share a track and a start family, so they are not independent: the section is the unit of
  // replication. Each section's success rate over the seeds is one observation, and comparisons are paired by section
  // (cluster bootstrap and an exact sign test over sections). A Wilson interval is added only when the caller states that
  // every trial is its own independent random start (`independentStarts`).
  function summarizeConditions(rows, rng, { baseline = null, iterations = 2000, independentStarts = false } = {}) {
    const names = [...new Set(rows.map(r => r.condition))], base = baseline ?? names[0];
    const rates = new Map(names.map(name => {
      const bySection = new Map();
      for (const r of rows) if (r.condition === name) bySection.set(r.section, [...(bySection.get(r.section) ?? []), r.metric.success]);
      return [name, new Map([...bySection].map(([section, values]) => [section, average(values)]))];
    }));
    return names.map(name => {
      const trials = rows.filter(r => r.condition === name), k = trials.reduce((n, r) => n + r.metric.success, 0);
      const perSection = [...rates.get(name).values()], interval = bootstrapMean(perSection, rng, { iterations });
      const summary = { condition: name, trials: trials.length, k, sections: perSection.length, rate: interval.mean, lo: interval.lo, hi: interval.hi,
        wilson: independentStarts ? wilson(k, trials.length) : null,
        contacts: trials.reduce((n, r) => n + r.metric.wall_contacts, 0),
        handling: trials.filter(r => r.metric.end_reason === "lateral_acceleration_limit").length,
        invalid: trials.filter(r => r.metric.end_reason === "wall_contact" && r.metric.duration_s === 0).length,
        progress: average(trials.map(r => r.metric.progress)), clearance: average(trials.map(r => r.metric.min_clearance_m)) };
      if (name !== base) {
        const shared = [...rates.get(name).keys()].filter(section => rates.get(base).has(section));
        summary.versus = { baseline: base, sections: shared.length,
          ...pairedDifference(shared.map(s => rates.get(name).get(s)), shared.map(s => rates.get(base).get(s)), rng, { iterations }) };
      }
      return summary;
    });
  }

  // ---- split independence ------------------------------------------------------------------------------------------------
  // Distinct episode IDs do not make two drives independent: identical starts give near-identical frames. Episode starts are
  // recorded as { split, track_id, fraction, lateral, heading_error, speed } and compared across splits.
  function startsAreNearDuplicates(a, b) {
    return a.track_id === b.track_id && Math.abs(a.fraction - b.fraction) < .02 && Math.abs(a.lateral - b.lateral) < .3
      && Math.abs(a.heading_error - b.heading_error) < .02 && Math.abs(a.speed - b.speed) < .5;
  }
  function findSplitLeak(episodes, candidate) {
    for (const [id, start] of Object.entries(episodes))
      if (start.split !== candidate.split && startsAreNearDuplicates(start, candidate)) return id;
    return null;
  }
  function validateEpisodeStarts(episodes) {
    const entries = Object.entries(episodes);
    for (let i = 0; i < entries.length; i++) for (let j = i + 1; j < entries.length; j++)
      if (entries[i][1].split !== entries[j][1].split && startsAreNearDuplicates(entries[i][1], entries[j][1]))
        throw new Error(`Episodes ${entries[i][0]} (${entries[i][1].split}) and ${entries[j][0]} (${entries[j][1].split}) start from nearly the same state, so they leak across splits`);
    return true;
  }

  // ---- diagnosis ---------------------------------------------------------------------------------------------------------
  function actionErrors(trace) {
    const errors = [];
    trace.forEach((row, index) => {
      if (!row.action || !row.expert) return;
      const steering = row.action[0] - row.expert[0], speed = row.action[1] - row.expert[1];
      errors.push({ index, time: row.time, steering, speed, error: (4 * steering ** 2 + speed ** 2) / 2 });
    });
    return errors;
  }
  // First step where the policy's weighted action error stays above the threshold for `sustain` consecutive steps.
  function firstDivergence(errors, { threshold = .02, sustain = 3 } = {}) {
    if (!errors.length) return null;
    const need = Math.min(sustain, errors.length);
    for (let i = 0; i + need <= errors.length; i++)
      if (errors.slice(i, i + need).every(e => e.error > threshold)) return errors[i];
    return null;
  }
  function snapshotFromRow(row, track) {
    const keys = ["x", "y", "heading", "speed", "steering", "gear", "rpm", "throttle", "brake", "acceleration", "targetSpeed", "requestedSteering",
      "lateralAcceleration", "shiftActive", "shiftRemaining", "shiftCooldown"];
    return { state: Object.fromEntries(keys.map(k => [k, row[k]])), margin: row.margin, contact: row.contact, time: row.time,
      progress: row.progress, previous: track.locate(row.x, row.y)[0] };
  }
  // "Expert recovery from this state": fork the recorded drive at evenly spaced times and let the expert finish from that exact
  // state (steering, drivetrain, clock, progress and exit window preserved). It is a counterfactual about the state, not proof
  // that an earlier action caused the failure, and outcomes need not switch only once, so every fork is returned.
  async function expertRecovery({ track, cfg, trace, strideSteps = 10, maxSteps = 350, goal = 150, sim = {}, yieldFn = noYield }) {
    const forks = [], last = trace.length - 1, stride = Math.max(1, Math.min(strideSteps, Math.floor(last / 8)));
    for (let index = 0; index < last; index += stride) {
      const result = E.simulate(track, cfg, "expert", null, { ...sim, snapshot: snapshotFromRow(trace[index], track), maxSteps: maxSteps - index, goal,
        history: trace.slice(Math.max(0, index - 19), index) });
      forks.push({ index, time: trace[index].time, success: result.metric.success === 1, end_reason: result.metric.end_reason,
        min_clearance_m: result.metric.min_clearance_m });
      await yieldFn();
    }
    const firstFailure = forks.findIndex(f => !f.success);
    return { forks, firstFailure: firstFailure < 0 ? null : forks[firstFailure],
      lastRecoverable: firstFailure < 0 ? forks.at(-1) ?? null : firstFailure > 0 ? forks[firstFailure - 1] : null,
      monotonic: forks.every((f, i) => i === 0 || f.success <= forks[i - 1].success) };
  }
  const defaultFeatureGroups = names => [
    { name: "lane offset", indices: [0] }, { name: "heading error", indices: [1, 2] }, { name: "speed", indices: [3] },
    { name: "previous steering", indices: [4] }, { name: "road previews (10-70 m)", indices: [5, 6, 7, 8, 9, 10, 11, 12] },
    { name: "curvature inputs", indices: [13, 14, 15] }, { name: "wall margin (derived)", indices: [16], derived: true },
    { name: "steering budget (derived)", indices: [17], derived: true },
  ].map(g => ({ ...g, kind: "group", detail: g.indices.map(i => names?.[i] ?? `input ${i}`) }));
  // Increase in the weighted action loss when a group of inputs is shuffled across examples (breaking its link to the labels).
  // With `cfg`, the derived wall-margin and steering-budget inputs are recomputed from the shuffled base inputs, so the shuffled
  // examples stay physically consistent; the two derived inputs themselves are shuffled on their own.
  function permutationImportance(model, rows, rng, { repeats = 5, groups = null, names = null, maxRows = 600, permutation = null, cfg = null } = {}) {
    const used = rows.length > maxRows ? rng.shuffle(rows.slice()).slice(0, maxRows) : rows;
    const baseline = modelLoss(model, used), entries = groups ?? [
      ...defaultFeatureGroups(names), ...Array.from({ length: 18 }, (_, i) => ({ name: names?.[i] ?? `input ${i}`, indices: [i], kind: i > 15 ? "derived" : "feature" }))];
    return entries.map(group => {
      const increases = [], recompute = cfg !== null && group.indices.every(i => i < 16);
      for (let r = 0; r < repeats; r++) {
        const order = permutation ? permutation(used.length, group, r) : rng.shuffle(Array.from({ length: used.length }, (_, i) => i));
        const shuffled = used.map((row, i) => {
          const features = row.features.slice();
          for (const column of group.indices) features[column] = used[order[i]].features[column];
          return { features: recompute ? E.derived(features, cfg) : features, targets: row.targets };
        });
        increases.push(modelLoss(model, shuffled) - baseline);
      }
      const mean = average(increases);
      return { ...group, baseline, mean, sd: repeats > 1 ? Math.sqrt(increases.reduce((v, x) => v + (x - mean) ** 2, 0) / (repeats - 1)) : null, increases };
    });
  }

  return { Z95, wilson, quantile, bootstrapMean, signTest, pairedDifference, summarizeSeeds, learningRateAt, modelLoss, fit,
    sectionsFor, evaluateSections, tally, RidgeModel, KNNModel, defaultStarts, sampleStart, collect, supportIndex, supportFraction,
    coverageGrid, featurePoint, arms, runArm, coverageExperiment, pairedStart, randomSections, describeCondition, runConditions, summarizeConditions, actionErrors,
    firstDivergence, snapshotFromRow, expertRecovery, defaultFeatureGroups, permutationImportance,
    startsAreNearDuplicates, findSplitLeak, validateEpisodeStarts, inSupport };
});
