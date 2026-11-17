(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./student_engine.js"), require("./student_tools.js"), require("./student_rl.js"));
  else root.StudentJourney = factory(root.StudentEngine, root.StudentTools, root.StudentRL);
})(globalThis, function (E, T, R) {
  "use strict";
  // Module 12, "From zero to a lap". Both learners start as a car that cannot drive and are trained on ONE circuit. After every stage of training the car as it
  // is at that moment is put on the start line and drives a whole lap with no help, so every stage faces the same test: does it get round, and how fast?
  //   imitation      DAgger (module 7). Stage 0 is the untrained network, stage 1 follows the expert demonstrations, stage k follows DAgger round k - 1.
  //   reinforcement  PPO (student_rl.js). Stage 0 is the untrained policy, then snapshots of the policy at chosen training iterations.
  // A stage's lap is driven with the policy mean (no exploration noise), so its result is a property of that snapshot alone.
  const { Network } = E;
  const lapStart = { fraction: 0, lateral: 0, heading_error: 0, speed: 12 };
  const LAP_STEPS = 4000;
  const defaults = { imitation: { rounds: 16, budget: 150, updatesPerRound: 25 }, reinforcement: { iterations: 300 } };
  const markList = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500, 600];
  // Fine early, when the car changes quickly, then roughly doubling; the last iteration is always kept.
  const rlMarks = iterations => [...markList.filter(m => m < iterations), iterations];

  const round = (value, places) => Math.round(value * 10 ** places) / 10 ** places;
  const lapRun = (track, cfg, model) => E.simulate(track, cfg, model, lapStart, { goal: track.length, maxSteps: LAP_STEPS });
  // What the gallery and tables need from a lap; the full trace is regenerated on demand for the replay.
  // `flicker` is how far the steering angle moves per 0.1 s step on average, in degrees: about 0.1 for smooth driving, 4.5 when the car flicks it at the actuator's speed limit.
  function summarizeLap(run) {
    const { trace, metric } = run, stride = Math.max(1, Math.ceil(trace.length / 150));
    const path = trace.filter((_, i) => i % stride === 0 || i === trace.length - 1).map(s => [round(s.x, 1), round(s.y, 1)]);
    const moved = trace.slice(1).reduce((sum, s, i) => sum + Math.abs(s.steering - trace[i].steering), 0);
    return { completed: metric.end_reason === "section_complete", reason: metric.end_reason, progress: metric.progress, seconds: metric.duration_s, steps: trace.length - 1, path,
      flicker: trace.length > 1 ? moved / (trace.length - 1) * 180 / Math.PI : 0 };
  }
  const expertLap = (track, cfg) => summarizeLap(lapRun(track, cfg, "expert"));

  function makeStage(method, track, cfg, net, fields) {
    return { method, ...fields, minutes: fields.steps * cfg.dt / 60, model: net.serialize(), lap: summarizeLap(lapRun(track, cfg, net)) };
  }

  // The expert labels states the learner reaches (DAgger). Each labelled state is one simulated control step of 0.1 s.
  async function imitationJourney({ track, ctx, seed = 1, rounds = defaults.imitation.rounds, budget = defaults.imitation.budget, updatesPerRound = defaults.imitation.updatesPerRound,
      onStage = null, shouldCancel = null, yieldFn }) {
    const stages = [];
    const push = (net, fields) => { const stage = makeStage("imitation", track, ctx.cfg, net, { index: stages.length, ...fields }); stages.push(stage); onStage?.(stage, stages); };
    push(new Network(seed), { steps: 0, updates: 0 });
    const result = await T.runArm({ arm: "dagger", seed, ctx, trackIds: [track.id], budget, rounds, updatesPerRound, yieldFn, shouldCancel,
      onRound: ({ round: r, net, rows }) => push(net, { round: r, steps: rows.length, updates: net.updates }) });
    return { method: "imitation", seed, trackId: track.id, settings: { rounds, budget, updatesPerRound }, stages, diverged: result.diverged };
  }

  // The car earns a reward for distance and pays for crashing and for jerky commands; nothing tells it what the right steering angle is.
  async function reinforcementJourney({ track, ctx, seed = 1, iterations = defaults.reinforcement.iterations, marks = rlMarks(iterations), reward = R.defaultReward, ppo = {},
      onStage = null, onIteration = null, shouldCancel = null, yieldFn }) {
    const agent = new R.ActorCritic(seed), wanted = new Set(marks), stages = [];
    const push = (iteration, steps) => {
      const stage = makeStage("reinforcement", track, ctx.cfg, agent.actorNetwork(), { index: stages.length, iteration, steps, updates: agent.updates });
      stages.push(stage); onStage?.(stage, stages);
    };
    push(0, 0);
    const result = await R.ppoTrain({ track, cfg: ctx.cfg, seed, iterations, agent, reward, yieldFn, shouldCancel, ...ppo,
      onIteration: info => { onIteration?.(info); if (wanted.has(info.iteration)) push(info.iteration, info.steps); } });
    // Cancelled part-way: keep the car as it stands.
    const last = result.history.at(-1);
    if (last && stages.at(-1).iteration !== last.iteration) push(last.iteration, last.steps);
    return { method: "reinforcement", seed, trackId: track.id, settings: { iterations, reward, ...ppo }, stages, history: result.history, stopped: result.stopped };
  }

  // ---- summaries ---------------------------------------------------------------------------------------------------------------
  // Whole percent of the lap driven; a car that crashed a few metres short of the line reads 99%, never 100%.
  const lapPercent = lap => lap.completed ? 100 : Math.min(99, Math.floor(lap.progress * 100));
  // A compact row per stage for the saved reference: [driving steps, % of the lap completed, lap completed (0/1), lap time in s, steering flicker in degrees per step].
  const compactStage = s => [s.steps, lapPercent(s.lap), s.lap.completed ? 1 : 0, round(s.lap.seconds, 1), round(s.lap.flicker, 2)];
  // A car whose steering moves more than this many degrees per 0.1 s step, on average over the lap, is flicking it; a drive that ends within 10 s is too short to judge.
  const FLICKER_LIMIT = 1, FLICKER_MIN_SECONDS = 10;
  const isFlicking = (seconds, flicker) => seconds >= FLICKER_MIN_SECONDS && flicker > FLICKER_LIMIT;
  const sorted = values => values.slice().sort((a, b) => a - b);
  const median = values => T.quantile(sorted(values), .5);
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;

  // How one run went: when it first got round, from which stage it keeps getting round at every later stage, and how fast and how smoothly the final car drives.
  function runSummary(rows, expertSeconds) {
    const first = rows.find(r => r[2] === 1), final = rows.at(-1);
    let from = rows.length;
    while (from > 0 && rows[from - 1][2] === 1) from--;
    return { firstLapSteps: first ? first[0] : null, stableLapSteps: from < rows.length ? rows[from][0] : null, finalCompleted: final[2] === 1,
      ratio: final[2] === 1 ? final[3] / expertSeconds : null, regressed: Boolean(first) && final[2] === 0, lostLap: Boolean(first) && rows.slice(rows.indexOf(first)).some(r => r[2] === 0),
      flicker: final[4], flicks: isFlicking(final[3], final[4]) };
  }
  // Median and range across seeds, stage by stage. The stage schedule is the same for every seed.
  function curve(runs) {
    const count = Math.min(...runs.map(r => r.stages.length));
    return Array.from({ length: count }, (_, k) => {
      const progress = sorted(runs.map(r => r.stages[k][1])), done = runs.filter(r => r.stages[k][2] === 1);
      return { steps: median(runs.map(r => r.stages[k][0])), median: T.quantile(progress, .5), lo: progress[0], hi: progress.at(-1), laps: done.length,
        seconds: done.length ? median(done.map(r => r.stages[k][3])) : null };
    });
  }
  const spread = values => values.length ? { median: T.quantile(sorted(values), .5), q1: T.quantile(sorted(values), .25), q3: T.quantile(sorted(values), .75), n: values.length } : null;
  function aggregate(runs, expertSeconds) {
    const per = runs.map(r => runSummary(r.stages, expertSeconds)), ratios = per.filter(p => p.ratio !== null).map(p => p.ratio);
    return { seeds: runs.length, firstLapSteps: spread(per.filter(p => p.firstLapSteps !== null).map(p => p.firstLapSteps)),
      stableLapSteps: spread(per.filter(p => p.stableLapSteps !== null).map(p => p.stableLapSteps)), finalLaps: per.filter(p => p.finalCompleted).length,
      regressed: per.filter(p => p.regressed).length, lostLap: per.filter(p => p.lostLap).length, ratio: ratios.length ? { mean: mean(ratios), best: Math.min(...ratios), worst: Math.max(...ratios) } : null,
      flickering: per.filter(p => p.flicks).length, flicker: mean(per.map(p => p.flicker)), curve: curve(runs) };
  }
  // Rebuilds a saved stage from its model with the code in front of the student, and records whether the lap came out exactly as saved.
  function restoreStage(method, track, cfg, saved) {
    const stage = makeStage(method, track, cfg, Network.fromJSON(saved.model), { index: saved.index, round: saved.round, iteration: saved.iteration, steps: saved.steps, updates: saved.updates });
    stage.reproduced = JSON.stringify(compactStage(stage)) === JSON.stringify(saved.row);
    return stage;
  }
  function describeMinutes(minutes) {
    if (minutes < 1) return `${Math.round(minutes * 60)} s`;
    if (minutes < 120) return `${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} min`;
    return `${(minutes / 60).toFixed(minutes < 1200 ? 1 : 0)} h`;
  }

  return { lapStart, LAP_STEPS, FLICKER_LIMIT, FLICKER_MIN_SECONDS, isFlicking, defaults, rlMarks, lapRun, summarizeLap, expertLap, imitationJourney, reinforcementJourney, restoreStage, lapPercent,
    compactStage, runSummary, spread, curve, aggregate, describeMinutes };
});
