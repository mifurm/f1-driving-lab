#!/usr/bin/env node
"use strict";
// Headless reference for the browser's "Data coverage and DAgger" lab. It calls the same StudentTools.coverageExperiment
// that the page calls, so the reference table students see is produced by exactly the code they run themselves.
//   node tools/coverage_reference.cjs --artifact student_v2_artifacts [--seeds 10] [--budget 100] [--rounds 4] [--updates 120]
const fs = require("fs");
const path = require("path");
const E = require("../student_engine.js");
const T = require("../student_tools.js");

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, token, i, all) => token.startsWith("--") ? [...pairs, [token.slice(2), all[i + 1]]] : pairs, []));
const artifact = path.resolve(args.artifact ?? "student_v2_artifacts");
const settings = { seedCount: Number(args.seeds ?? 10), budget: Number(args.budget ?? 100), rounds: Number(args.rounds ?? 4), updatesPerRound: Number(args.updates ?? 120) };
const data = JSON.parse(fs.readFileSync(path.join(artifact, "data/lab.json"), "utf8"));
const ctx = { cfg: data.config, tracks: new Map(data.tracks.map(track => [track.id, new E.Track(track)])) };
const names = { bc_narrow: "Narrow expert demonstrations", bc_broad: "Broad expert demonstrations", bc_staged: "Staged expert control", dagger: "DAgger" };
const pct = v => `${(100 * v).toFixed(0)}%`, signed = v => `${v >= 0 ? "+" : "−"}${Math.abs(100 * v).toFixed(0)} pts`;
const interval = s => `${pct(s.mean)} [${pct(s.lo)}, ${pct(s.hi)}]`;
const trim = s => ({ n: s.n, mean: s.mean, lo: s.lo, hi: s.hi, sd: s.sd, min: s.min, max: s.max, values: s.values });

(async () => {
  const started = Date.now();
  const run = await T.coverageExperiment({
    data, ctx, arms: Object.keys(names), ...settings, trainIds: Array.from({ length: 11 }, (_, i) => i),
    validIds: data.dataGeneration.valid_training_tracks, holdoutIds: data.dataGeneration.legacy_holdout_tracks,
    onRun: ({ arm, seed, ev }) => process.stderr.write(`${arm} seed ${seed}: ${ev.k}/${ev.n}\n`),
  });
  const expert = { nominal: run.expert.train.nominal, recovery: run.expert.train.recovery, hold: { k: run.expert.hold.k, n: run.expert.hold.n } };
  const summary = {
    version: "coverage-reference-1", engine: E.version, settings: run.settings, sections: { training: run.trainSections.length, holdout: run.holdSections.length },
    expert: { nominal: { k: expert.nominal.k, n: expert.nominal.n }, recovery: { k: expert.recovery.k, n: expert.recovery.n }, hold: expert.hold },
    arms: run.summaries.map(s => ({ arm: s.arm, name: names[s.arm], nominal: trim(s.nominal), recovery: trim(s.recovery), hold: trim(s.hold), overall: trim(s.overall),
      supportReference: s.supportReference, supportVisited: s.supportVisited, failed: s.failed, rejected: s.rejected, diverged: s.diverged })),
    comparisons: run.comparisons.map(c => ({ a: c.a, b: c.b, overall: c.overall, recovery: c.recovery })),
  };
  const lines = [
    `Reference run: ${summary.arms.length} recipes × ${summary.settings.seeds} training seeds (${summary.settings.seedBase}–${summary.settings.seedBase + summary.settings.seeds - 1}), ${summary.settings.budget} labelled states per track per round,`,
    `${summary.settings.rounds} rounds after the first, ${summary.settings.updatesPerRound} Adam updates per round. Seed = unit of replication; intervals are 95% percentile bootstrap over seeds.`, "",
    "| Recipe | Nominal sections | Recovery sections | Legacy holdout | Reference occupancy | Visited occupancy | Failed collections | Diverged |", "|---|---|---|---|---|---|---|---|",
    ...summary.arms.map(a => `| ${a.name} | ${interval(a.nominal)} | ${interval(a.recovery)} | ${interval(a.hold)} | ${pct(a.supportReference)} | ${pct(a.supportVisited)} | ${a.failed} | ${a.diverged} |`),
    `| Expert (ceiling) | ${expert.nominal.k}/${expert.nominal.n} | ${expert.recovery.k}/${expert.recovery.n} | ${expert.hold.k}/${expert.hold.n} | 100% | n/a | n/a | n/a |`, "",
    "| Paired by seed | Overall difference [95% bootstrap] | Recovery-only difference | Seeds better/worse/equal | Sign test p |", "|---|---|---|---|---|",
    ...summary.comparisons.map(c => `| ${names[c.a]} − ${names[c.b]} | ${signed(c.overall.mean)} [${signed(c.overall.lo)}, ${signed(c.overall.hi)}] | ${signed(c.recovery.mean)} [${signed(c.recovery.lo)}, ${signed(c.recovery.hi)}] | ${c.overall.wins}/${c.overall.losses}/${c.overall.ties} | ${c.overall.p.toFixed(3)} |`),
  ];
  const output = path.join(artifact, "data/coverage_reference.json");
  fs.writeFileSync(output, JSON.stringify({ summary, markdown: lines.join("\n"), seconds: (Date.now() - started) / 1000 }, null, 1));
  process.stdout.write(`${lines.join("\n")}\n\nWrote ${output} in ${((Date.now() - started) / 1000).toFixed(1)} s\n`);
})().catch(error => { console.error(error); process.exit(1); });
