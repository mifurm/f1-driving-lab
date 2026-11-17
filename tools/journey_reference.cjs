#!/usr/bin/env node
"use strict";
// Headless reference for the browser's "From zero to a lap" lab. It runs the same StudentJourney functions the page runs, for every valid circuit and
// seed, so the numbers students see next to their own training come from exactly the code they run. Every run is independent and deterministic, so
// the result does not depend on how many worker processes share the work.
//   node tools/journey_reference.cjs --artifact student_v2_artifacts [--seeds 10] [--iterations 300] [--featured 9] [--workers 12]
const fs = require("fs");
const os = require("os");
const path = require("path");
const { fork } = require("child_process");
const E = require("../student_engine.js");
const R = require("../student_rl.js");
const J = require("../student_journey.js");

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, token, i, all) => token.startsWith("--") ? [...pairs, [token.slice(2), all[i + 1]]] : pairs, []));
const artifact = path.resolve(args.artifact ?? "student_v2_artifacts");
function load() {
  const data = JSON.parse(fs.readFileSync(path.join(artifact, "data/lab.json"), "utf8"));
  return { data, ctx: { cfg: data.config, tracks: new Map(data.tracks.map(track => [track.id, new E.Track(track)])) } };
}
const round = (value, places) => Math.round(value * 10 ** places) / 10 ** places;

if (process.argv.includes("--worker")) {
  const { ctx } = load();
  process.on("message", async task => {
    try {
      const track = ctx.tracks.get(task.trackId), started = Date.now();
      const run = task.method === "imitation" ? await J.imitationJourney({ track, ctx, seed: task.seed })
        : await J.reinforcementJourney({ track, ctx, seed: task.seed, iterations: task.iterations, reward: { ...R.defaultReward, ...task.reward } });
      const result = { variant: task.variant, method: task.method, trackId: task.trackId, seed: task.seed, rows: run.stages.map(J.compactStage), seconds: (Date.now() - started) / 1000, diverged: Boolean(run.diverged) };
      if (task.keep) {
        result.stages = run.stages.map(s => ({ index: s.index, round: s.round, iteration: s.iteration, steps: s.steps, updates: s.updates, model: s.model }));
        if (run.history) result.history = run.history.map(h => ({ iteration: h.iteration, steps: h.steps, meanReturn: round(h.meanReturn ?? 0, 3), meanDistance: round(h.meanDistance ?? 0, 1),
          failureRate: round(h.failureRate ?? 0, 3), std: round(h.std, 4) }));
      }
      process.send(result);
    } catch (error) {
      process.send({ failed: `${task.variant} circuit ${task.trackId} seed ${task.seed}: ${error.stack ?? error}` });
    }
  });
  process.send({ ready: true });
  return;
}

const steps = v => v === null || v === undefined ? "never" : Math.round(v).toLocaleString("en-US");
async function main() {
  const started = Date.now(), { data, ctx } = load(), ids = data.dataGeneration.valid_training_tracks;
  const seedCount = Number(args.seeds ?? 10), iterations = Number(args.iterations ?? J.defaults.reinforcement.iterations), featured = Number(args.featured ?? 9);
  const tasks = [];
  // Reinforcement runs are the slow ones, so they go first. The ablation trains the featured circuit again with the steering-smoothness reward term switched off.
  for (const trackId of ids) for (let seed = 1; seed <= seedCount; seed++) tasks.push({ variant: "reinforcement", method: "reinforcement", trackId, seed, iterations, keep: trackId === featured && seed === 1 });
  for (let seed = 1; seed <= seedCount; seed++) tasks.push({ variant: "reinforcement_no_smoothness", method: "reinforcement", trackId: featured, seed, iterations, reward: { smoothness: 0 }, keep: false });
  for (const trackId of ids) for (let seed = 1; seed <= seedCount; seed++) tasks.push({ variant: "imitation", method: "imitation", trackId, seed, keep: trackId === featured && seed === 1 });
  const results = [], children = [], workers = Math.max(1, Math.min(Number(args.workers ?? os.cpus().length - 2), tasks.length));
  const pool = new Promise((resolve, reject) => {
    let next = 0, running = 0;
    const feed = child => {
      if (next >= tasks.length) { child.finished = true; child.kill(); if (--running === 0) resolve(); return; }
      child.send(tasks[next++]);
    };
    for (let w = 0; w < workers; w++) {
      const child = fork(__filename, ["--worker", "--artifact", artifact], { stdio: ["inherit", "inherit", "inherit", "ipc"] });
      children.push(child); running++;
      child.on("error", reject);
      // A worker that dies on its own, or reports a failed task, must stop the whole run instead of leaving a stale reference behind.
      child.on("exit", code => { if (!child.finished) reject(new Error(`A worker exited early (code ${code}) while ${results.length} of ${tasks.length} runs were done`)); });
      child.on("message", message => {
        if (message.failed) { reject(new Error(message.failed)); return; }
        if (!message.ready) {
          results.push(message);
          process.stderr.write(`${message.variant} circuit ${message.trackId} seed ${message.seed}: ${J.lapPercent({ completed: message.rows.at(-1)[2] === 1, progress: message.rows.at(-1)[1] / 100 })}% in ${message.seconds.toFixed(1)} s (${results.length}/${tasks.length})\n`);
        }
        feed(child);
      });
    }
  });
  // On a failure stop the other workers too, so their half-finished runs do not print errors after this one.
  await pool.catch(error => { for (const child of children) { child.finished = true; child.kill(); } throw error; });

  const runs = { imitation: {}, reinforcement: {}, reinforcement_no_smoothness: {} };
  for (const r of results.sort((a, b) => a.seed - b.seed)) (runs[r.variant][r.trackId] ??= []).push({ seed: r.seed, stages: r.rows });
  const circuits = {};
  for (const id of ids) {
    const track = ctx.tracks.get(id), expert = J.expertLap(track, ctx.cfg);
    circuits[id] = { name: track.name, length: round(track.length, 0), expertSeconds: round(expert.seconds, 1),
      imitation: J.aggregate(runs.imitation[id], expert.seconds), reinforcement: J.aggregate(runs.reinforcement[id], expert.seconds) };
  }
  const ablation = { trackId: featured, smoothness: R.defaultReward.smoothness, with: circuits[featured].reinforcement,
    without: J.aggregate(runs.reinforcement_no_smoothness[featured], circuits[featured].expertSeconds) };
  const keep = variant => results.find(r => r.variant === variant && r.trackId === featured && r.seed === 1);
  const imitationKept = keep("imitation"), reinforcementKept = keep("reinforcement");
  const settings = { seeds: seedCount, circuits: ids, imitation: J.defaults.imitation, reinforcement: { ...R.ppoDefaults, iterations, marks: J.rlMarks(iterations), reward: R.defaultReward },
    lap: { start: J.lapStart, maxSteps: J.LAP_STEPS, flickerLimit: J.FLICKER_LIMIT, flickerMinSeconds: J.FLICKER_MIN_SECONDS } };
  const featuredJourney = { trackId: featured, seed: 1,
    imitation: { settings: J.defaults.imitation, stages: imitationKept.stages.map((s, i) => ({ ...s, row: imitationKept.rows[i] })) },
    reinforcement: { settings: { iterations, marks: J.rlMarks(iterations), reward: R.defaultReward }, stages: reinforcementKept.stages.map((s, i) => ({ ...s, row: reinforcementKept.rows[i] })),
      history: reinforcementKept.history } };

  const time = value => value === null || value === undefined ? "—" : `${value.toFixed(2)}×`;
  const both = s => `${steps(s.firstLapSteps?.median)} / ${steps(s.stableLapSteps?.median)}`;
  // Pooled over every circuit and seed.
  const overall = Object.fromEntries(["imitation", "reinforcement"].map(method => {
    const per = ids.flatMap(id => runs[method][id].map(r => J.runSummary(r.stages, circuits[id].expertSeconds))), ratios = per.filter(p => p.ratio !== null).map(p => p.ratio);
    return [method, { runs: per.length, firstLapSteps: J.spread(per.filter(p => p.firstLapSteps !== null).map(p => p.firstLapSteps)), stableLapSteps: J.spread(per.filter(p => p.stableLapSteps !== null).map(p => p.stableLapSteps)),
      finalLaps: per.filter(p => p.finalCompleted).length, regressed: per.filter(p => p.regressed).length, lostLap: per.filter(p => p.lostLap).length, flickering: per.filter(p => p.flicks).length,
      ratio: ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : null }];
  }));
  const pooled = (name, o) => `${name}: first lap ${steps(o.firstLapSteps?.median)} steps, stays lapping from ${steps(o.stableLapSteps?.median)}, final car laps in ${o.finalLaps}/${o.runs} runs, `
    + `${o.lostLap}/${o.runs} runs failed at some stage after they had first completed the lap (${o.regressed} in the final car), mean lap time ${time(o.ratio)} the expert's, ${o.flickering}/${o.runs} final cars flick the steering`;
  const lines = [
    `Reference run: ${ids.length} circuits × ${seedCount} seeds (1–${seedCount}) × 2 learners. Each learner trains on one circuit, then drives one full lap from the start line with no help.`,
    `Imitation (DAgger): ${J.defaults.imitation.rounds} rounds after the expert demonstrations, ${J.defaults.imitation.budget} labelled steps and ${J.defaults.imitation.updatesPerRound} Adam updates per round.`,
    `Reinforcement (PPO): ${iterations} iterations × ${R.ppoDefaults.envs} drives × ${R.ppoDefaults.rolloutSteps} steps = ${(iterations * R.ppoDefaults.envs * R.ppoDefaults.rolloutSteps).toLocaleString("en-US")} simulated steps, no expert labels. One simulated step is 0.1 s of driving.`,
    "First lap = the first snapshot whose car completes the lap. Stays lapping = the first snapshot from which it and every later snapshot complete the lap. Both are medians across seeds, in driving steps.",
    "Lost a lap = runs that fail the lap at some snapshot after having completed it. Final car = the last snapshot. Lap time / expert = the final car's lap time divided by the expert's on that circuit, mean over the seeds whose final car completes the lap.", "",
    "| Circuit | Expert lap | Imitation: first lap / stays lapping (steps) | Imitation: lost a lap at some stage | Imitation: final cars that lap | Imitation: lap time / expert | RL: first lap / stays lapping (steps) | RL: lost a lap at some stage | RL: final cars that lap | RL: lap time / expert |",
    "|---|---|---|---|---|---|---|---|---|---|",
    ...ids.map(id => {
      const c = circuits[id], i = c.imitation, r = c.reinforcement;
      return `| ${id} ${c.name} | ${c.expertSeconds.toFixed(0)} s | ${both(i)} | ${i.lostLap}/${i.seeds} | ${i.finalLaps}/${i.seeds} | ${time(i.ratio?.mean)} | ${both(r)} | ${r.lostLap}/${r.seeds} | ${r.finalLaps}/${r.seeds} | ${time(r.ratio?.mean)} |`;
    }), "",
    pooled("Imitation, all runs", overall.imitation), pooled("Reinforcement, all runs", overall.reinforcement), "",
    `Reward ablation on circuit ${featured} (${ablation.with.seeds} seeds each): with the steering-smoothness term (weight ${ablation.smoothness}) ${ablation.with.flickering}/${ablation.with.seeds} final cars flick the steering, `
    + `mean ${ablation.with.flicker.toFixed(2)}° per step, lap time ${time(ablation.with.ratio?.mean)} the expert's; without it ${ablation.without.flickering}/${ablation.without.seeds} flick it, `
    + `mean ${ablation.without.flicker.toFixed(2)}° per step, lap time ${time(ablation.without.ratio?.mean)} the expert's.`,
  ];
  const output = path.join(artifact, "data/journey_reference.json");
  fs.writeFileSync(output, JSON.stringify({ version: "journey-reference-1", engine: E.version, settings, circuits, overall, ablation, featured: featuredJourney, runs, markdown: lines.join("\n"),
    seconds: (Date.now() - started) / 1000, cpuSeconds: round(results.reduce((v, r) => v + r.seconds, 0), 0) }));
  process.stdout.write(`${lines.join("\n")}\n\nWrote ${output} in ${((Date.now() - started) / 1000).toFixed(1)} s\n`);
}
main().catch(error => { console.error(error); process.exit(1); });
