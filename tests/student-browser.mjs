// Real-browser verification of the student lab (version 2): all twelve modules, the teacher tools and the failure paths.
//   PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" node tests/student-browser.mjs [student_v2_artifacts]
// Every check runs the real engine in Chromium; nothing is mocked. Writes browser_verification.json, teacher_guide.pdf and screenshots.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] || "student_v2_artifacts");
const kit = JSON.parse(await fs.readFile("teaching_kit.json", "utf8"));
const python = process.env.PYTHON || (existsSync(".venv/bin/python") ? ".venv/bin/python" : "python3");
const errors = [], checks = {}, exercised = [], downloadNames = [], failedChecks = [], scratch = await fs.mkdtemp(path.join(os.tmpdir(), "f1-lab-"));
// What the lab downloads during the run is kept apart from the lab's real data files.
const downloads = path.join(output, "browser_downloads");
await fs.mkdir(downloads, { recursive: true });
// KEEP_GOING=1 collects every failed check instead of stopping at the first (used to show that a new check fails on code without its fix).
const keepGoing = process.env.KEEP_GOING === "1";
const check = (name, condition, detail = "") => {
  if (!condition && keepGoing) { failedChecks.push(`${name} ${detail}`.trim()); checks[name] = false; return; }
  assert.ok(condition, `Check failed: ${name} ${detail}`); checks[name] = true;
};
// The page logs every handled error to the console. These come from the failure paths this test triggers on purpose; anything else is a bug.
const intended = [/Make both predictions/, /nearly identical/, /does not match this lab/, /Learning rate: choose a number between/, /not a fresh test pack/, /Answer every question/,
  /has no valid sections/, /no built-in circuit uses/, /check is already recorded/, /Cancelled: no DAgger session was started/, /Cancelled: the DAgger session was not changed/,
  /Cancelled: nothing was added to the DAgger session/];
const unexpected = text => !intended.some(pattern => pattern.test(text));
// The report is only evidence for the exact build it exercised, so it carries the hashes of the page files; student_verify.py compares them with the files on disk.
const BUILD_FILES = ["index.html", "app.js", "engine.js", "tools.js", "rl.js", "journey.js", "labs.js", "journey_ui.js", "class_dashboard.html", "teacher_guide.html"];
const buildHashes = async () => Object.fromEntries(await Promise.all(BUILD_FILES.map(async name => [name, createHash("sha256").update(await fs.readFile(path.join(output, name))).digest("hex")])));
const testedBuild = await buildHashes();
// The headless counterpart of module 12, from the same source files the page was built from.
const require = createRequire(import.meta.url);
const journeyApi = require("../student_journey.js"), engineApi = require("../student_engine.js");
const labData = JSON.parse(await fs.readFile(path.join(output, "data/lab.json"), "utf8"));
const nodeContext = { cfg: labData.config, tracks: new Map(labData.tracks.map(track => [track.id, new engineApi.Track(track)])) };
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });

async function setRange(page, id, value) {
  await page.locator(`#${id}`).evaluate((input, next) => { input.value = next; input.dispatchEvent(new Event("input")); }, String(value));
}
const idle = (page, timeout = 240000) => page.waitForFunction(() => !studentLab.busy, null, { timeout });
// Makes every yield inside the lab take `ms` milliseconds (60 by default), so a Cancel click, or another tab, lands inside a long action instead of racing it.
const slowFrames = (page, ms = 60) => page.evaluate(delay => { window.__raf = window.requestAnimationFrame; window.requestAnimationFrame = callback => window.setTimeout(() => callback(performance.now()), delay); }, ms);
const normalFrames = page => page.evaluate(() => { window.requestAnimationFrame = window.__raf; });
// Presses Cancel while a long action runs; false means the action had already finished (or never started) and there was nothing to cancel.
const cancelRunning = page => page.locator("#cancelRun").click({ timeout: 5000 }).then(() => true, () => false);
const error = page => page.locator("#error").innerText();
async function visit(page, id) { await page.locator(`button[data-module="${id}"]`).click(); await page.waitForFunction(m => studentLab.module === m, id); }
async function shot(page, name) { await page.screenshot({ path: path.join(output, "visuals", `${name}.png`), fullPage: true }); }
async function finish(page, id, name) { await shot(page, name); exercised.push(id); }
async function rowCount(page, id) { return page.locator(`#${id} tbody tr`).count(); }
async function paintedColors(page, id) {
  return page.evaluate(canvasId => {
    const canvas = document.getElementById(canvasId), image = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
    return new Set(Array.from({ length: image.data.length / 4 }, (_, i) => `${image.data[i * 4]},${image.data[i * 4 + 1]},${image.data[i * 4 + 2]}`)).size;
  }, id);
}

try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 }, acceptDownloads: true });
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", message => { if (message.type() === "error" && unexpected(message.text())) errors.push(message.text()); });
  page.on("download", async download => { downloadNames.push(download.suggestedFilename()); await download.saveAs(path.join(downloads, download.suggestedFilename())); });
  await page.goto(pathToFileURL(path.join(output, "index.html")).href);
  await page.waitForFunction(() => !!window.studentLab);

  // ---- 1. predict ------------------------------------------------------------------------------------------------------------------
  check("sealed_circuits_hidden_before_freezing", await page.locator("#track option").count() === 13);
  check("all_eighteen_inputs_listed", await rowCount(page, "inputTable") === 18);
  check("default_track_is_a_valid_monitor_circuit", await page.locator("#track").inputValue() === "3");
  check("legacy_circuits_are_flagged", (await page.locator('#track option[value="0"]').innerText()).includes("legacy geometry"));
  await page.locator("#track").selectOption("0");
  check("legacy_note_explains_the_inner_wall_fold", (await page.locator("#legacyNote").innerText()).includes("inner wall folds"));
  await page.locator("#track").selectOption("3");
  check("reference_ablation_table_has_four_variants", await rowCount(page, "referenceAblationTable") === 4);
  check("sealed_results_hidden_until_exposure", !(await page.locator("#referenceAblationTable").innerText()).includes("pristine_test"));
  check("prediction_hidden_until_revealed", !(await page.locator("#predictionResult").isVisible()));
  await page.locator("#revealPrediction").click();
  check("reveal_requires_a_prediction", (await error(page)).includes("Make both predictions"));
  await page.locator("#guessDirection").selectOption("right"); await page.locator("#guessBrake").selectOption("yes");
  await page.locator("#revealPrediction").click();
  await page.waitForFunction(() => studentLab.runs[0]?.trace.length > 50);
  check("prediction_revealed_from_the_actual_policy", await page.locator("#predictionResult").isVisible() && (await page.locator("#predictionResult").innerText()).includes("Actual model output"));
  check("canvas_is_painted", await paintedColors(page, "view") > 100);
  const before = await page.evaluate(() => studentLab.replayTime); await page.waitForTimeout(150);
  check("replay_advances", await page.evaluate(() => studentLab.replayTime) > before);
  await finish(page, "predict", "01_predict");

  // ---- 2. dataset, training and cancellation ---------------------------------------------------------------------------------------
  await visit(page, "dataset"); await page.locator("#newDrive").click();
  const observation = await page.evaluate(() => studentLab.manual.env.observe().features);
  await setRange(page, "manualSteer", -10); await setRange(page, "manualTarget", 25);
  await page.locator("#recordStep").click();
  check("recorded_inputs_are_the_actual_observation", JSON.stringify(await page.evaluate(() => studentLab.rows[0].features)) === JSON.stringify(observation));
  check("recorded_label_is_the_students_command", Math.abs(await page.evaluate(() => studentLab.rows[0].targets[0]) - (-10 * Math.PI / 180 / .5)) < 1e-10);
  await page.locator("#recordSplit").selectOption("validation"); await page.locator("#recordStep").click();
  check("an_existing_drive_cannot_change_split", await page.evaluate(() => studentLab.rows[1].split) === "train");
  await page.locator("#track").selectOption("1"); await page.locator("#newDrive").click(); await page.locator("#recordStep").click();
  check("a_new_drive_takes_the_selected_split", await page.evaluate(() => studentLab.rows[2].split) === "validation");
  const rowsBefore = await page.evaluate(() => studentLab.rows.length);
  await page.locator("#recordSplit").selectOption("train"); await page.locator("#newDrive").click();
  check("near_duplicate_start_in_the_other_split_is_rejected", (await error(page)).includes("nearly identical") && await page.evaluate(() => studentLab.rows.length) === rowsBefore);
  await setRange(page, "startLateral", 1.5); await page.locator("#newDrive").click();
  check("a_clearly_different_start_is_accepted", (await error(page)) === "");
  await page.locator("#recordPlay").click(); await page.waitForFunction(() => studentLab.rows.length >= 6); await page.locator("#recordPlay").click();
  await page.locator("#loadStarter").click();
  const counts = await page.locator("#datasetCounts").innerText();
  check("starter_set_loaded_with_both_splits", /train/i.test(counts) && /validation/i.test(counts), counts);
  await page.locator("#downloadDataset").click(); await page.locator("#downloadCSV").click(); await page.waitForTimeout(250);
  check("dataset_downloads_exist", downloadNames.includes("student-driving-dataset.json") && downloadNames.includes("student-driving-dataset.csv"));
  const csv = await fs.readFile(path.join(downloads, "student-driving-dataset.csv"), "utf8");
  check("csv_has_labels_and_all_inputs", csv.split("\r\n")[0].includes("label_steering_fraction") && csv.split("\r\n")[0].split(",").length === 5 + 18 + 2);
  const savedDataset = await fs.readFile(path.join(downloads, "student-driving-dataset.json"), "utf8");
  await page.locator("#importDataset").setInputFiles({ name: "wrong.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ version: "other", rows: [] })) });
  await page.waitForFunction(() => document.getElementById("error").textContent.length > 0);
  check("dataset_from_another_version_is_rejected", (await error(page)).includes("does not match"));
  await page.locator("#importDataset").setInputFiles({ name: "import.json", mimeType: "application/json", buffer: Buffer.from(savedDataset) });
  await page.waitForFunction(() => document.getElementById("status").textContent.includes("Imported"));
  check("file_input_is_emptied_so_the_same_file_can_be_chosen_again", await page.locator("#importDataset").evaluate(input => input.files.length === 0 && input.value === ""));
  await page.locator("#clearDataset").click();
  check("clearing_removes_every_example", await page.evaluate(() => studentLab.rows.length === 0));
  await page.locator("#importDataset").setInputFiles({ name: "import.json", mimeType: "application/json", buffer: Buffer.from(savedDataset) });
  await page.waitForFunction(() => studentLab.rows.length > 0);
  check("the_same_dataset_can_be_imported_again_after_clearing", await page.evaluate(() => studentLab.rows.length) === JSON.parse(savedDataset).rows.length);
  await page.locator("#updates").fill("400"); await page.locator("#trainStudent").click();
  await page.waitForFunction(() => studentLab.losses.length >= 3);
  check("cancel_button_appears_while_training", await page.locator("#cancelRun").isVisible());
  await page.locator("#importDataset").setInputFiles({ name: "busy.json", mimeType: "application/json", buffer: Buffer.from(savedDataset) });
  check("a_file_chosen_while_the_lab_is_busy_is_refused_and_the_input_still_emptied", (await error(page)).includes("Wait for the current experiment") && await page.locator("#importDataset").evaluate(input => input.files.length === 0));
  await page.locator("#cancelRun").click(); await idle(page);
  check("cancelling_stops_training_early_and_says_so", await page.evaluate(() => studentLab.student.updates < 400 && studentLab.lastFit.stopped === "cancelled"));
  await page.locator("#updates").fill("40"); await page.locator("#trainStudent").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.student?.updates === 40 && studentLab.lastFit.stopped !== "cancelled", null, { timeout: 180000 });
  check("training_lowers_training_loss", await page.evaluate(() => studentLab.losses.at(-1).train < studentLab.losses[0].train));
  check("closed_loop_probes_recorded_beside_the_loss", await page.evaluate(() => studentLab.probe.length >= 2 && studentLab.probe.every(p => p.nominal.n === 8 && p.recovery.n === 8)));
  check("loss_and_probe_canvases_painted", await paintedColors(page, "lossCanvas") > 20 && await paintedColors(page, "probeCanvas") > 20);
  await page.locator("#animateStudentLearning").click(); await page.waitForFunction(() => studentLab.runs.length === 3);
  check("replays_use_recorded_weight_snapshots", JSON.stringify(await page.evaluate(() => studentLab.trainingSnapshots.map(s => s.update))) === JSON.stringify([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]));
  await page.locator("#exportModel").click(); await page.waitForTimeout(200);
  check("model_export_downloaded", downloadNames.includes("student-driving-model.json"));
  await finish(page, "dataset", "02_dataset_training");

  // ---- 3. one training step ----------------------------------------------------------------------------------------------------------
  await visit(page, "step"); await page.locator("#inspectStep").click();
  await page.waitForFunction(() => studentLab.packet?.updates === 41);
  check("backprop_equals_finite_difference", await page.evaluate(() => Math.abs(studentLab.packet.numericGradient - studentLab.packet.changes[0].gradient) < 1e-7));
  for (let i = 0; i < 4; i++) await page.locator("#stepNext").click();
  check("adam_stage_reached", (await page.locator("#stepStage").innerText()).includes("Adam"));
  check("weight_and_batch_tables_have_real_rows", await rowCount(page, "weightTable") === 6 && await rowCount(page, "batchTable") === 8);
  const shown = await page.locator("#batchTable tbody tr").evaluateAll(rows => rows.reduce((sum, row) => sum + Number(row.lastElementChild.textContent), 0) / rows.length);
  check("displayed_per_example_errors_average_to_the_loss", Math.abs(shown - await page.evaluate(() => studentLab.packet.loss)) < 1e-6);
  await page.locator("#driveAfterStep").click();
  check("before_and_after_update_drives_replayed", await page.evaluate(() => studentLab.runs.length) === 2);
  await finish(page, "step", "03_training_step");

  // ---- 4. rules versus learning ------------------------------------------------------------------------------------------------------
  await visit(page, "controllers"); await page.locator("#model").selectOption("pretrained"); await page.locator("#controllerSeeds").selectOption("3");
  await page.locator("#compareControllers").click(); await idle(page);
  check("three_controllers_compared", await rowCount(page, "controllerTable") === 3);
  check("small_seed_counts_are_labelled_exploratory", (await page.locator("#statsNote").innerText()).length > 0 && await page.evaluate(() => studentLab.comparisons.exploratory === true));
  const starts = await page.evaluate(() => studentLab.comparisons.rows.filter(r => r.seed === studentLab.comparisons.rows[0].seed && r.section === studentLab.comparisons.rows[0].section).map(r => JSON.stringify(r.initial)));
  check("controllers_share_identical_starts", starts.length === 3 && new Set(starts).size === 1, JSON.stringify(starts));
  const outcomes = await page.evaluate(() => Object.fromEntries(["Rules", "Expert", "Neural"].map(name => [name, studentLab.comparisons.rows.filter(r => r.condition === name).reduce((n, r) => n + r.metric.success, 0)])));
  check("expert_beats_rules_in_the_same_sections", outcomes.Expert > outcomes.Rules, JSON.stringify(outcomes));
  await page.locator("#downloadComparison").click(); await page.waitForTimeout(200);
  check("comparison_download_exists", downloadNames.includes("controller-comparison.json"));
  const comparisonFile = await fs.readFile(path.join(downloads, "controller-comparison.json"), "utf8");
  check("comparison_export_names_its_policies_but_never_embeds_weights", !/"weights"|"mW"|"biases"/.test(comparisonFile) && comparisonFile.includes('"policy": "model"') && comparisonFile.length < 600000, `${comparisonFile.length} bytes`);
  await finish(page, "controllers", "04_controllers");

  // ---- 5. remove inputs and retrain --------------------------------------------------------------------------------------------------
  await visit(page, "features"); await page.locator("#ablationSeeds").selectOption("3");
  await visit(page, "dataset"); await page.locator("#updates").fill("12"); await visit(page, "features");
  await page.locator("#localAblation").click(); await idle(page, 300000);
  check("twelve_models_really_retrained", await page.evaluate(() => studentLab.localAblations.models.length === 12 && studentLab.localAblations.models.every(m => m.actual_updates === 12)));
  check("small_ablation_marked_exploratory", (await page.locator("#ablationStatus").innerText()).includes("EXPLORATORY"));
  check("ablation_table_and_strips_drawn", await rowCount(page, "ablationTable") === 4 && await paintedColors(page, "ablationCanvas") > 20);
  check("reference_curves_drawn", await paintedColors(page, "referenceCurves") > 20 && await rowCount(page, "ablationHistories") === 4);
  await page.locator("#animateReferenceLearning").click();
  check("reference_snapshots_replayed", await page.evaluate(() => studentLab.runs.length) === 3);
  await finish(page, "features", "05_feature_retraining");

  // ---- 6. training playground -------------------------------------------------------------------------------------------------------
  await visit(page, "playground"); await page.locator("#presetOverfit").click(); await page.locator("#playSeeds").selectOption("3"); await page.locator("#runPlayground").click(); await idle(page);
  const overfit = await page.evaluate(() => studentLab.playground.summary[0]);
  check("overfitting_preset_overfits", overfit.validation > 1.5 * overfit.train && overfit.bestUpdate < 300, JSON.stringify({ train: overfit.train, validation: overfit.validation, best: overfit.bestUpdate }));
  await page.locator("#presetRegularize").click(); await page.locator("#playSeeds").selectOption("3"); await page.locator("#runPlayground").click(); await idle(page);
  const regularised = await page.evaluate(() => studentLab.playground.summary[0]);
  check("regularising_preset_lowers_validation_loss", regularised.validation < 0.6 * overfit.validation && regularised.early > 0, JSON.stringify({ before: overfit.validation, after: regularised.validation }));
  await page.locator("#presetBaselines").click(); await idle(page);
  check("baselines_trained_and_selectable", await page.evaluate(() => ["ridge", "knn"].every(k => studentLab.baselines[k]) && !document.querySelector('#model option[value="ridge"]').disabled));
  await page.locator("#sweepParameter").selectOption("hidden"); await page.locator("#playSeeds").selectOption("3");
  await page.locator("#presetOverfit").click(); await page.locator("#playUpdates").fill("60"); await page.locator("#runSweep").click(); await idle(page);
  check("sweep_compares_three_settings_with_one_factor_changed", await page.evaluate(() => studentLab.playground.results.length === 3 && new Set(studentLab.playground.results.map(r => r.settings.hidden1)).size === 3));
  await page.locator("#playRate").fill("0.5"); await page.locator("#runPlayground").click();
  check("out_of_range_learning_rate_rejected_in_words_not_element_ids", (await error(page)).startsWith("Learning rate: choose a number between"));
  await page.locator("#playRate").fill("0.002");
  await finish(page, "playground", "06_playground");

  // ---- 7. data coverage and DAgger --------------------------------------------------------------------------------------------------
  await visit(page, "coverage");
  check("saved_ten_seed_reference_displayed", await page.locator("#coverageReferenceBox").isVisible());
  await page.locator("#covBudget").selectOption("50"); await page.locator("#covRounds").selectOption("2"); await page.locator("#covUpdates").selectOption("60"); await page.locator("#covSeeds").selectOption("3");
  await page.locator("#runCoverage").click(); await idle(page, 300000);
  check("four_recipes_run_with_equal_budgets", await page.evaluate(() => Object.values(studentLab.coverage.results).every(rs => rs.length === 3 && rs.every(r => r.labelled === 50 * 11 * 3 && r.updates === 60 * 3))));
  check("nominal_and_recovery_reported_separately", await rowCount(page, "coverageTable") === 5 && (await page.locator("#coverageTable").innerText()).includes("Recovery sections"));
  check("paired_comparisons_use_seeds_as_the_unit", await rowCount(page, "coveragePairs") === 4 && await page.evaluate(() => studentLab.coverage.comparisons.every(c => c.overall.wins + c.overall.losses + c.overall.ties === 3)));
  check("coverage_maps_painted", await paintedColors(page, "coverageMaps") > 30);
  await page.locator("#startDagger").click(); await idle(page);
  await page.locator("#daggerRollout").click(); await idle(page);
  const snapshot = () => page.evaluate(() => { const d = studentLab.dagger; return JSON.stringify({ rows: d.rows.length, round: d.round, pending: d.pending?.rows.length ?? null, updates: d.net.updates, weights: StudentEngine.fingerprint(d.net) }); });
  const beforeCancel = await snapshot(), pendingRows = await page.evaluate(() => { window.__session = studentLab.dagger; return studentLab.dagger.pending.rows.length; });
  await slowFrames(page); await page.locator("#daggerTrain").click(); const cancelledRetrain = await cancelRunning(page); await idle(page); await normalFrames(page);
  check("cancelling_dagger_retraining_leaves_the_session_exactly_as_it_was", cancelledRetrain && (await error(page)).startsWith("Cancelled") && await snapshot() === beforeCancel, `${await error(page)} ${await snapshot()} vs ${beforeCancel}`);
  if (await page.locator("#daggerTrain").isEnabled()) { await page.locator("#daggerTrain").click(); await idle(page); }
  check("dagger_session_adds_learner_visited_states", await page.evaluate(() => studentLab.dagger.round === 1 && studentLab.dagger.rows.some(r => /learner rollout/.test(r.source)) && studentLab.dagger.log.length >= 2));
  check("retraining_after_a_cancel_adds_each_state_exactly_once", await page.evaluate(n => { const d = studentLab.dagger; return d.rows.length === n && new Set(d.rows.map(r => r.id)).size === d.rows.length; }, JSON.parse(beforeCancel).rows + pendingRows));
  await slowFrames(page); await page.locator("#startDagger").click(); const cancelledStart = await cancelRunning(page); await idle(page); await normalFrames(page);
  check("cancelling_a_new_dagger_session_keeps_the_existing_one", cancelledStart && (await error(page)).startsWith("Cancelled") && await page.evaluate(() => studentLab.dagger === window.__session && studentLab.dagger.round === 1));
  // A cancelled rollout must leave no trace: a fresh, identical session gives the same rollout whether or not one was cancelled first.
  const rolloutOf = () => page.evaluate(() => { const p = studentLab.dagger.pending; return JSON.stringify({ loss: p.newStateLoss, first: p.rows[0].id, last: p.rows.at(-1).id, rows: p.rows.length }); });
  await page.locator("#startDagger").click(); await idle(page);
  await page.locator("#daggerRollout").click(); await idle(page);
  const uninterrupted = await rolloutOf();
  await page.locator("#startDagger").click(); await idle(page);
  const started = await page.evaluate(() => ({ rng: studentLab.dagger.rng.state, episode: studentLab.dagger.episode }));
  await slowFrames(page, 100); await page.locator("#daggerRollout").click(); const cancelledRollout = await cancelRunning(page); await idle(page); await normalFrames(page);
  const afterCancel = await page.evaluate(() => ({ rng: studentLab.dagger.rng.state, episode: studentLab.dagger.episode, pending: studentLab.dagger.pending }));
  check("cancelling_a_dagger_rollout_gives_back_its_random_numbers_and_episode_numbers", cancelledRollout && (await error(page)).startsWith("Cancelled: nothing was added") && afterCancel.pending === null
    && afterCancel.rng === started.rng && afterCancel.episode === started.episode, JSON.stringify({ started, afterCancel }));
  await page.locator("#daggerRollout").click(); await idle(page);
  check("a_rollout_after_a_cancelled_one_equals_an_uninterrupted_rollout", await rolloutOf() === uninterrupted, `${await rolloutOf()} vs ${uninterrupted}`);
  await finish(page, "coverage", "07_coverage");

  // ---- 8. diagnose a failure ---------------------------------------------------------------------------------------------------------
  await visit(page, "diagnose"); await page.locator("#model").selectOption("student"); await page.locator("#diagnoseRun").click(); await idle(page);
  check("failure_recorded_with_expert_labels_at_the_true_states", await page.evaluate(() => studentLab.diagnosis.errors.length >= 1 && studentLab.diagnosis.errors.every(e => Number.isFinite(e.error))));
  await setRange(page, "divergeThreshold", 50);
  check("threshold_slider_updates_label", (await page.locator("#divergeLabel").innerText()).includes("0.050"));
  await page.locator("#recoveryRun").click(); await idle(page);
  check("expert_recovery_forks_reported", await rowCount(page, "recoveryTable") >= 1 && await page.evaluate(() => studentLab.diagnosis.recovery.forks.length >= 1));
  await page.locator("#importanceRun").click(); await idle(page);
  check("permutation_importance_for_groups_and_features", await page.evaluate(() => studentLab.importance.results.some(r => r.kind === "group") && studentLab.importance.results.some(r => r.kind === "derived") && studentLab.importance.results.some(r => r.kind === "feature")));
  check("overlay_and_importance_charts_painted", await paintedColors(page, "overlayCanvas") > 20 && await paintedColors(page, "importanceCanvas") > 20);
  await finish(page, "diagnose", "08_diagnose");

  // ---- 9. generalization, leakage and a teacher's fresh test pack ----------------------------------------------------------------------
  await visit(page, "generalization"); await page.locator("#model").selectOption("student");
  check("fresh_pack_evaluation_is_disabled_until_a_model_is_frozen", await page.locator("#evaluateFresh").isDisabled());
  await page.locator("#freezeReveal").click(); await idle(page);
  check("sealed_circuits_exposed_after_freezing", await page.locator("#track option").count() === 16 && await rowCount(page, "testTable") === 12);
  const fingerprint = await page.evaluate(() => StudentEngine.fingerprint(studentLab.frozen));
  const packPath = path.join(scratch, "fresh_pack.json");
  execFileSync(python, ["tools/make_fresh_pack.py", "--seed", "20260601", "--circuits", "2", "--output", packPath], { stdio: "pipe" });
  await page.locator("#freshPack").setInputFiles(packPath); await page.waitForFunction(() => studentLab.fresh);
  await page.locator("#evaluateFresh").click(); await idle(page);
  check("fresh_pack_evaluated_once_on_the_frozen_model", await rowCount(page, "freshTable") === 8 && (await page.locator("#freshStatus").innerText()).includes("Frozen model"));
  const goodPack = JSON.parse(await fs.readFile(packPath, "utf8")), loadedPack = await page.evaluate(() => studentLab.fresh.fingerprint);
  const rejectPack = async (name, mutate, expected) => {
    const pack = JSON.parse(JSON.stringify(goodPack)); mutate(pack);
    await page.locator("#freshPack").setInputFiles({ name, mimeType: "application/json", buffer: Buffer.from(JSON.stringify(pack)) });
    const rejected = await page.waitForFunction(text => document.getElementById("error").textContent.includes(text), expected, { timeout: 5000 }).then(() => true, () => false);
    return rejected && await page.evaluate(() => studentLab.fresh.fingerprint) === loadedPack;
  };
  check("a_foreign_file_is_not_accepted_as_a_fresh_pack", await rejectPack("foreign.json", pack => { pack.schema = "other"; }, "not a fresh test pack"));
  check("a_pack_circuit_without_sections_is_rejected_up_front", await rejectPack("no-sections.json", pack => { delete pack.scenarios[pack.tracks[0].id]; }, "has no valid sections"));
  check("a_pack_circuit_that_reuses_a_built_in_id_is_rejected", await rejectPack("clash.json", pack => { pack.tracks[0].id = 3; }, "no built-in circuit uses"));
  check("a_pack_with_a_malformed_section_is_rejected", await rejectPack("bad-section.json", pack => { pack.scenarios[pack.tracks[0].id][0].speed = "fast"; }, "has no valid sections"));
  check("file_input_is_emptied_after_a_rejected_pack", await page.locator("#freshPack").evaluate(input => input.files.length === 0));
  await page.locator("#demonstrateLeak").click(); await idle(page);
  check("leak_demonstration_trains_a_clone_on_test_labels", await page.evaluate(() => studentLab.leakage.valid_generalization === false && StudentEngine.fingerprint(studentLab.frozen)) === fingerprint);
  check("leakage_declared_invalid", (await page.locator("#leakStatus").innerText()).includes("INVALID"));
  await finish(page, "generalization", "09_generalization");

  // ---- 10. robustness -----------------------------------------------------------------------------------------------------------------
  await visit(page, "robustness"); await page.locator("#model").selectOption("pretrained"); await page.locator("#robustSeeds").selectOption("3"); await page.locator("#runRobustness").click(); await idle(page);
  check("clean_and_shifted_conditions_compared", await rowCount(page, "robustnessTable") === 2 && await page.evaluate(() => studentLab.robustness.conditions.length === 2));
  await page.locator("#downloadRobustness").click(); await page.waitForTimeout(200);
  check("robustness_download_exists", downloadNames.includes("robustness-experiment.json"));
  await finish(page, "robustness", "10_robustness");

  // ---- 12. from zero to a lap (run before module 11 so that the results card lists it; recorded after module 11 to keep the kit's order) ----------------------
  await visit(page, "journey");
  check("journey_lists_every_valid_circuit_and_opens_on_the_saved_one", await page.locator("#journeyCircuit option").count() === 9 && await page.locator("#journeyCircuit").inputValue() === "9");
  check("journey_starts_empty_with_nothing_to_replay", await page.locator("#journeyReplay").isDisabled() && await page.locator("#journeyMovie").isDisabled() && await page.locator("#journeyImitationStage").isDisabled()
    && await page.locator('#model option[value="journeyReinforcement"]').isDisabled() && await page.locator('#model option[value="journeyImitation"]').isDisabled());
  check("journey_reference_table_covers_every_circuit_and_the_reward_ablation", await rowCount(page, "journeyReferenceTable") === 9 && (await page.locator("#journeyAblation").textContent()).includes("Without it, 10/10 flick"));
  await page.locator("#journeySaved").click(); await idle(page);
  const savedExample = await page.evaluate(() => {
    const j = studentLab.journey, both = ["imitation", "reinforcement"];
    return { stages: both.map(m => j[m].stages.length), reproduced: both.every(m => j[m].stages.every(s => s.reproduced)), untrainedFails: both.every(m => !j[m].stages[0].lap.completed && j[m].stages[0].lap.progress < .05),
      finalLaps: both.every(m => j[m].stages.at(-1).lap.completed), smooth: both.every(m => j[m].stages.at(-1).lap.flicker < StudentJourney.FLICKER_LIMIT), source: both.map(m => j[m].source) };
  });
  check("journey_saved_example_has_every_stage_of_both_learners", savedExample.stages.join() === "18,24" && savedExample.source.join() === "saved,saved");
  check("journey_saved_example_is_driven_again_in_the_page_and_matches_what_was_saved", savedExample.reproduced && (await page.locator("#journeyStatus").innerText()).includes("matches the saved numbers"));
  check("journey_untrained_cars_fail_at_once_and_the_final_cars_lap_smoothly", savedExample.untrainedFails && savedExample.finalLaps && savedExample.smooth);
  check("journey_filmstrip_is_painted_and_has_a_picture_per_key_stage", await paintedColors(page, "journeyGallery") > 20 && await page.evaluate(() => studentLab.journeyCells.length) === 20);
  check("journey_curves_show_the_ten_seed_reference_bands", (await page.locator("#journeyBandNote").innerText()).includes("Shaded bands") && await paintedColors(page, "journeyProgress") > 10 && await paintedColors(page, "journeyLapTime") > 10);
  check("journey_stage_table_has_a_row_per_stage_and_the_summary_names_both_learners", await rowCount(page, "journeyTable") === 42
    && (await page.locator("#journeyCompare").innerText()).includes("Reinforcement needed") && (await page.locator("#journeyCompare").innerText()).includes("Imitation"));
  check("journey_final_cars_join_the_policy_menu", await page.evaluate(() => ["journeyImitation", "journeyReinforcement"].every(v => !document.querySelector(`#model option[value="${v}"]`).disabled)));
  // Replaying: three whole laps side by side, at the faster journey speed, which is put back when the module is left.
  await page.locator("#journeyReplay").click();
  const replayed = await page.evaluate(() => ({ names: studentLab.runs.map(r => r.name), laps: studentLab.runs.map(r => r.metric.end_reason), longest: Math.min(...studentLab.runs.map(r => r.trace.at(-1).time)),
    speed: document.getElementById("replaySpeed").value, camera: document.getElementById("camera").value }));
  check("journey_replay_shows_the_two_final_cars_and_the_expert_on_a_whole_lap", replayed.names.length === 3 && replayed.names[0].startsWith("Imitation") && replayed.names[1].startsWith("Reinforcement") && replayed.names[2] === "Expert"
    && replayed.laps.every(r => r === "section_complete") && replayed.longest > 100, JSON.stringify(replayed));
  check("journey_replay_raises_the_speed_and_shows_the_whole_circuit", replayed.speed === "16" && replayed.camera === "overview");
  check("journey_replay_panels_keep_one_colour_per_learner", await page.evaluate(() => studentLab.runs.map(r => r.color).join()) === "#ffc56b,#76a9ff,#4de2c2");
  await page.waitForFunction(() => studentLab.replayTime > 8, null, { timeout: 30000 });
  check("journey_replay_advances_at_the_chosen_speed", await page.evaluate(() => studentLab.replayPlaying && studentLab.replayTime > 8));
  // Picking an early stage from the filmstrip replays a car that fails, with the failure marked. A slower speed the student chose is kept.
  await page.locator("#replaySpeed").selectOption("2");
  const pick = await page.evaluate(() => { const cell = studentLab.journeyCells.find(c => c.method === "reinforcement" && c.index === 2), canvas = document.getElementById("journeyGallery"), scale = canvas.clientWidth / canvas.width;
    return { x: (cell.x + cell.size / 2) * scale, y: (cell.y + cell.size / 2) * scale }; });
  await page.locator("#journeyGallery").click({ position: pick });
  check("journey_a_later_replay_keeps_the_speed_the_student_chose", await page.locator("#replaySpeed").inputValue() === "2");
  const early = await page.evaluate(() => ({ chosen: studentLab.journey.selected.reinforcement, reason: studentLab.runs[1].metric.end_reason, marked: !!studentLab.runs[1].marker, name: studentLab.runs[1].name }));
  check("journey_clicking_a_picture_replays_that_stage_and_marks_where_it_failed", early.chosen === 2 && early.reason !== "section_complete" && early.marked && early.name.includes("Iteration 2"), JSON.stringify(early));
  await setRange(page, "journeyImitationStage", 3);
  check("journey_slider_selects_a_stage_and_describes_it", await page.evaluate(() => studentLab.journey.selected.imitation) === 3 && (await page.locator("#journeyImitationDetail").innerText()).includes("DAgger round 2"));
  // The final cars drive in the other lessons through the policy menu.
  const fromMenu = await page.evaluate(() => {
    const final = method => StudentEngine.fingerprint(StudentEngine.Network.fromJSON(studentLab.journey[method].stages.at(-1).model));
    const result = {};
    for (const [value, method] of [["journeyImitation", "imitation"], ["journeyReinforcement", "reinforcement"]]) { document.getElementById("model").value = value; result[method] = StudentEngine.fingerprint(selectedModel()) === final(method); }
    return result;
  });
  check("journey_policy_menu_hands_the_final_cars_to_the_other_lessons", fromMenu.imitation && fromMenu.reinforcement);
  await page.locator("#model").selectOption("pretrained");
  await visit(page, "predict");
  check("journey_replay_speed_returns_to_the_lessons_default_when_the_module_is_left", await page.locator("#replaySpeed").inputValue() === "4");
  check("journey_camera_returns_to_the_close_up_when_the_module_is_left", await page.locator("#camera").inputValue() === "follow");
  check("journey_replay_view_goes_back_to_the_top_of_the_page_when_the_module_is_left", await page.evaluate(() => document.getElementById("view").parentNode.id !== "journeyViewHost"
    && document.getElementById("view").nextElementSibling.id === "replayControls" && !!document.getElementById("view").offsetParent
    && !!document.getElementById("camera").closest(".controls") && !document.getElementById("camera").closest("#journeyViewHost")));
  await visit(page, "journey");
  check("journey_replay_view_and_camera_menu_are_docked_next_to_the_filmstrip", await page.evaluate(() => document.getElementById("view").parentNode.id === "journeyViewHost" && document.getElementById("replayControls").parentNode.id === "journeyViewHost"
    && !!document.getElementById("camera").closest("#journeyViewHost") && !document.getElementById("camera").closest(".controls")));
  // The learning movie plays every picture of the filmstrip in turn; a cancel stops it, and so does leaving the module (which also puts the speed and camera back for good).
  await page.locator("#replaySpeed").selectOption("32");
  await page.locator("#journeyMovie").click();
  await page.waitForFunction(() => document.getElementById("journeyStatus").textContent.includes("Learning movie: picture 2"), null, { timeout: 60000 });
  await visit(page, "predict"); await idle(page, 60000); await page.waitForTimeout(1500);
  check("journey_leaving_the_module_stops_the_movie_and_it_does_not_take_the_view_back", (await page.locator("#journeyStatus").innerText()).includes("stopped")
    && await page.locator("#replaySpeed").inputValue() === "4" && await page.locator("#camera").inputValue() === "follow");
  await visit(page, "journey");
  await page.locator("#replaySpeed").selectOption("32");
  await page.locator("#journeyMovie").click();
  await page.waitForFunction(() => document.getElementById("journeyStatus").textContent.includes("Learning movie: picture 2"), null, { timeout: 60000 });
  check("journey_movie_is_cancellable_while_it_plays", await cancelRunning(page));
  await idle(page, 60000);
  check("journey_cancelled_movie_says_so", (await page.locator("#journeyStatus").innerText()).includes("stopped") && (await error(page)) === "");
  await page.locator("#journeyMovie").click(); await idle(page, 240000);
  const movieStatus = await page.locator("#journeyStatus").innerText(), movieEnd = await page.evaluate(() => ({ imitation: studentLab.journey.selected.imitation, reinforcement: studentLab.journey.selected.reinforcement,
    last: [studentLab.journey.imitation.stages.length - 1, studentLab.journey.reinforcement.stages.length - 1] }));
  check("journey_movie_plays_every_picture_and_ends_on_the_final_cars", movieStatus.includes("finished") && movieEnd.imitation === movieEnd.last[0] && movieEnd.reinforcement === movieEnd.last[1], movieStatus);
  await shot(page, "12_journey_saved_example");

  // Live training with small budgets (extra menu entries keep the test quick): the page must produce exactly what the headless code produces.
  await page.evaluate(() => { for (const [id, value] of [["journeyRounds", "8"], ["journeyIterations", "40"], ["journeyIterations", "12"]]) { const select = document.getElementById(id); if (![...select.options].some(o => o.value === value)) select.add(new Option(value, value)); } });
  await page.locator("#journeyCircuit").selectOption("1"); await page.locator("#journeySeed").fill("1");
  await page.locator("#journeyRounds").selectOption("8"); await page.locator("#journeyIterations").selectOption("40");
  await slowFrames(page, 30);
  await page.locator("#journeyBoth").click();
  await page.waitForFunction(() => document.getElementById("journeyStatus").textContent.includes("Reinforcement: iteration"), null, { timeout: 60000 });
  check("journey_reports_live_progress_and_blocks_a_second_action_while_training", await page.evaluate(() => studentLab.busy) && await page.locator("#cancelRun").isVisible());
  await page.locator("#journeyReplay").click();
  check("journey_replay_waits_for_a_running_experiment", (await error(page)).includes("Wait for the current experiment"));
  await idle(page, 240000); await normalFrames(page);
  const trained = await page.evaluate(() => ({ imitation: studentLab.journey.imitation.stages.map(StudentJourney.compactStage), reinforcement: studentLab.journey.reinforcement.stages.map(StudentJourney.compactStage),
    circuit: studentLab.journey.trackId, sources: [studentLab.journey.imitation.source, studentLab.journey.reinforcement.source], band: document.getElementById("journeyBandNote").textContent }));
  const headless = { imitation: (await journeyApi.imitationJourney({ track: nodeContext.tracks.get(1), ctx: nodeContext, seed: 1, rounds: 8 })).stages.map(journeyApi.compactStage),
    reinforcement: (await journeyApi.reinforcementJourney({ track: nodeContext.tracks.get(1), ctx: nodeContext, seed: 1, iterations: 40 })).stages.map(journeyApi.compactStage) };
  check("journey_training_in_the_browser_equals_the_headless_run_bit_for_bit", JSON.stringify(trained.imitation) === JSON.stringify(headless.imitation) && JSON.stringify(trained.reinforcement) === JSON.stringify(headless.reinforcement)
    && trained.circuit === 1 && trained.sources.join() === "trained,trained");
  check("journey_live_run_learns_to_lap_from_the_untrained_car", trained.reinforcement[0][2] === 0 && trained.reinforcement.at(-1)[2] === 1 && trained.imitation[0][2] === 0 && trained.imitation.some(row => row[2] === 1));
  check("journey_bands_are_withheld_when_the_budget_differs_from_the_reference", trained.band.includes("appear when the budgets are the default"));
  // Cancelling keeps the stages so far; the reward menu switches the smoothness term off and the page records it.
  await page.locator("#journeyIterations").selectOption("100"); await slowFrames(page, 40);
  await page.locator("#journeyReinforcement").click();
  await page.waitForFunction(() => document.getElementById("journeyStatus").textContent.includes("iteration 3 "), null, { timeout: 60000 });
  check("journey_reinforcement_run_is_cancellable", await cancelRunning(page));
  await idle(page, 60000); await normalFrames(page);
  const cancelled = await page.evaluate(() => { const r = studentLab.journey.reinforcement; return { stopped: r.stopped, last: r.stages.at(-1).iteration, count: r.stages.length, sliderMax: Number(document.getElementById("journeyReinforcementStage").max) }; });
  check("journey_cancelled_run_keeps_the_car_as_it_stands", cancelled.stopped === "cancelled" && cancelled.last >= 3 && cancelled.last < 100 && cancelled.sliderMax === cancelled.count - 1 && (await error(page)) === "", JSON.stringify(cancelled));
  await page.locator("#journeyIterations").selectOption("12"); await page.locator("#journeyReward").selectOption("0");
  await page.locator("#journeyReinforcement").click(); await idle(page, 120000);
  check("journey_reward_menu_switches_the_smoothness_term_off", await page.evaluate(() => studentLab.journey.reinforcement.settings.reward.smoothness === 0 && studentLab.journey.reinforcement.stages.at(-1).iteration === 12));
  await page.locator("#journeyReward").selectOption("0.3"); await page.locator("#replaySpeed").selectOption("4");
  // With only one learner trained, the replay keeps that learner's colour and the other one's absence is not papered over.
  await page.locator("#journeyCircuit").selectOption("2"); await page.locator("#journeyReinforcement").click(); await idle(page, 120000);
  await page.locator("#journeyReplay").click();
  check("journey_one_learner_replay_keeps_that_learners_colour", await page.evaluate(() => studentLab.journey.trackId === 2 && studentLab.journey.imitation === null && studentLab.runs.map(r => r.color).join() === "#76a9ff,#4de2c2"
    && document.getElementById("journeyImitationStage").disabled && document.getElementById("journeyImitationDetail").textContent === "Not trained yet."));
  check("journey_counts_its_runs_for_the_results_card", await page.evaluate(() => ["journey_saved_loads", "journey_imitation_runs", "journey_reinforcement_runs", "journey_replays", "journey_movies"].every(key => studentLab.log.counts[key] > 0)));
  await shot(page, "12_journey");

  // ---- 11. worksheets, quiz and results card -----------------------------------------------------------------------------------------------
  await visit(page, "worksheets");
  check("every_worksheet_listed", await page.locator("#worksheet option").count() === kit.worksheets.length);
  for (const [index, worksheet] of kit.worksheets.entries()) {
    await page.locator("#worksheet").selectOption(String(index)); await page.locator("#hypothesis").fill(`Prediction for ${worksheet.title}: it will change.`);
    for (const id of ["wsSeeds"]) if (await page.locator(`#${id}`).count()) await page.locator(`#${id}`).selectOption("3");
    await page.locator("#runWorksheet").click(); await idle(page, 300000);
    check(`worksheet_${worksheet.id}_produced_results`, (await error(page)) === "" && await rowCount(page, "worksheetResults") >= 1, `${worksheet.id}: ${await error(page)}`);
  }
  await page.locator("#worksheet").selectOption(String(kit.worksheets.findIndex(w => w.runner === "controllers"))); await page.locator("#wsSeeds").selectOption("3");
  await page.locator("#hypothesis").fill("The expert will complete more sections than the rules."); await page.locator("#runWorksheet").click(); await idle(page, 300000);
  await page.locator("#exportWorksheet").click(); await page.waitForTimeout(300);
  const controllersExport = await fs.readFile(path.join(downloads, "ml-driving-worksheet.json"), "utf8");
  check("worksheet_export_of_raw_trials_never_embeds_weights", !/"weights"|"mW"|"biases"/.test(controllersExport) && controllersExport.includes('"policy": "model"') && controllersExport.length < 1500000, `${controllersExport.length} bytes`);
  await page.locator("#worksheet").selectOption(String(kit.worksheets.findIndex(w => w.runner === "shift")));
  await page.locator("#runWorksheet").click(); await idle(page);
  await page.locator("#evidence").fill("Compared identical weights over matched seeds."); await page.locator("#conclusion").fill("A toy-model result is not a real-car safety guarantee.");
  await page.locator("#exportWorksheet").click(); await page.waitForTimeout(200);
  const worksheetExport = JSON.parse(await fs.readFile(path.join(downloads, "ml-driving-worksheet.json"), "utf8"));
  check("worksheet_export_has_hypothesis_evidence_and_rubric", worksheetExport.hypothesis.length > 0 && worksheetExport.evidence.length > 0 && worksheetExport.rubric.criteria.length === kit.rubric.criteria.length);
  const firstOrder = await page.evaluate(() => JSON.stringify(studentLab.quizOrder));
  check("quiz_choices_are_shuffled", await page.evaluate(() => studentLab.quizOrder.some(order => order[0] !== 0)));
  await page.locator("#gradeQuiz").click();
  check("quiz_requires_every_answer", (await error(page)).includes("Answer every question"));
  await page.locator("#quizPhase").selectOption("pre");
  for (let q = 0; q < kit.quiz.length; q++) await page.locator(`input[name="quiz${q}"][value="1"]`).check();
  await page.locator("#gradeQuiz").click();
  check("wrong_answers_are_explained_and_mapped_to_misconceptions", (await page.locator("#quizReview").innerText()).includes("Common misconception") || (await page.locator("#quiz").innerText()).includes("Common misconception"));
  check("a_graded_check_is_locked", await page.evaluate(() => [...document.querySelectorAll("#quiz input")].every(input => input.disabled)) && await page.locator("#gradeQuiz").isDisabled());
  await page.locator("#quizPhase").selectOption("post"); await page.locator("#quizPhase").selectOption("pre");
  check("returning_to_a_graded_phase_does_not_offer_a_second_baseline", await page.locator("#gradeQuiz").isDisabled() && (await page.locator("#quizFeedback").innerText()).includes("already recorded"));
  await page.locator("#gradeQuiz").evaluate(button => { button.disabled = false; button.click(); });
  check("the_handler_refuses_a_second_attempt_even_with_the_button_forced", (await error(page)).includes("already recorded") && await page.evaluate(() => studentLab.quizAttempts.length) === 1);
  await page.locator("#quizPhase").selectOption("post");
  check("a_new_attempt_reshuffles_the_choices", await page.evaluate(() => JSON.stringify(studentLab.quizOrder)) !== firstOrder);
  for (let q = 0; q < kit.quiz.length; q++) await page.locator(`input[name="quiz${q}"][value="0"]`).check();
  await page.locator("#gradeQuiz").click();
  check("correct_answers_score_fully", (await page.locator("#quizFeedback").innerText()).startsWith(`${kit.quiz.length}/${kit.quiz.length}`));
  check("pre_and_post_attempts_recorded", await page.evaluate(() => studentLab.quizAttempts.map(a => a.phase).join()) === "pre,post");
  await page.locator("#nickname").fill("Ada <script>");
  await page.locator("#downloadCard").click(); await page.waitForTimeout(250);
  const card = JSON.parse(await fs.readFile(path.join(downloads, "ml-lab-results-card.json"), "utf8"));
  check("results_card_has_no_free_text_or_timestamps", card.schema === "f1-ml-lab-results-card-v1" && !JSON.stringify(card).includes("<script") && !/\d{4}-\d{2}-\d{2}T/.test(JSON.stringify(card)) && !("evidence" in card) && !("hypothesis" in card));
  check("results_card_records_what_was_done", card.modules_visited.length === kit.modules.length && card.quiz.length === 2 && card.worksheets_completed.length >= kit.worksheets.length - 1 && card.exposure.test_exposed === true);
  await finish(page, "worksheets", "11_worksheets");
  exercised.push("journey");

  // ---- persistence, printable guide and the class dashboard -------------------------------------------------------------------------------
  await visit(page, "dataset"); await page.locator("#clearDataset").click();
  check("clearing_data_does_not_unexpose_the_test_set", await page.evaluate(() => studentLab.testExposed && studentLab.testLabelsExposed));
  await page.reload(); await page.waitForFunction(() => !!window.studentLab);
  check("test_exposure_persists_across_reloads", await page.evaluate(() => studentLab.testExposed && studentLab.testLabelsExposed));
  await page.goto(pathToFileURL(path.join(output, "teacher_guide.html")).href);
  check("teacher_guide_lists_every_objective_and_rubric_criterion", await page.evaluate(([objectives, criteria]) => objectives.every(o => document.body.innerText.includes(o)) && criteria.every(c => document.body.innerText.includes(c)),
    [kit.objectives.map(o => o.id), kit.rubric.criteria.map(c => c.name)]));
  await page.pdf({ path: path.join(output, "teacher_guide.pdf"), format: "A4", printBackground: false, margin: { top: "12mm", bottom: "12mm", left: "10mm", right: "10mm" } });
  check("teacher_guide_pdf_written", (await fs.stat(path.join(output, "teacher_guide.pdf"))).size > 5000);

  const dashboard = await browser.newPage({ viewport: { width: 1300, height: 1000 }, acceptDownloads: true });
  dashboard.on("pageerror", e => errors.push(`dashboard: ${e.message}`));
  dashboard.on("console", message => { if (message.type() === "error" && !/Failed to load resource/.test(message.text())) errors.push(`dashboard: ${message.text()}`); });
  dashboard.on("download", async download => { downloadNames.push(download.suggestedFilename()); await download.saveAs(path.join(downloads, download.suggestedFilename())); });
  await dashboard.goto(pathToFileURL(path.join(output, "class_dashboard.html")).href);
  const attemptOf = (phase, score, misconceptions = []) => ({ phase, score, total: 16, by_objective: { O1: [score > 8 ? 1 : 0, 1] }, misconceptions });
  const synthetic = (modules, quiz) => ({ ...card, nickname: "x", modules_visited: modules, quiz });
  // Card 2 holds two before-lesson attempts (6 then a perfect 16) and two after-lesson attempts (10 then 12): the baseline must stay the first, the final score the last.
  const files = [card, synthetic(["predict", "dataset"], [attemptOf("pre", 6), attemptOf("pre", 16), attemptOf("post", 10, ["M1", "M5"]), attemptOf("post", 12, ["M1", "M5"])]),
    synthetic(kit.modules.map(m => m.id), [attemptOf("pre", 9), attemptOf("post", 15, ["M1"])])]
    .map((value, i) => ({ name: `card${i}.json`, mimeType: "application/json", buffer: Buffer.from(JSON.stringify(value)) }));
  files.push({ name: "notes.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ hello: "world" })) });
  await dashboard.locator("#cards").setInputFiles(files);
  await dashboard.waitForFunction(() => document.getElementById("loaded").textContent.length > 0);
  check("dashboard_aggregates_valid_cards_and_rejects_others", (await dashboard.locator("#loaded").innerText()).includes("3 cards loaded, 1 rejected")
    && (await dashboard.locator("#error").innerText()).includes("notes.json") && await dashboard.locator("#cardsTable tbody tr").count() === 3);
  const secondCard = await dashboard.locator("#cardsTable tbody tr:nth-child(2) td").allInnerTexts();
  check("dashboard_baseline_is_the_first_before_attempt_and_the_final_is_the_last_after_attempt", secondCard[2] === "6/16" && secondCard[3] === "12/16", JSON.stringify(secondCard));
  const misconceptionText = await dashboard.locator("#misconceptionsTable").innerText();
  check("dashboard_counts_misconceptions_from_the_latest_attempt", misconceptionText.includes("M1") && misconceptionText.includes("2/3") && misconceptionText.includes("M5"));
  check("dashboard_reports_a_descriptive_paired_gain", (await dashboard.locator("#quizSummary").innerText()).includes("not evidence of learning"));
  check("dashboard_hides_nicknames_by_default", !(await dashboard.locator("#cardsTable").innerText()).includes("Ada"));
  await dashboard.locator("#exportCsv").click(); await dashboard.waitForTimeout(300);
  const aggregate = (await fs.readFile(path.join(downloads, "class-aggregate.csv"), "utf8")).split("\r\n");
  check("dashboard_exports_an_aggregate_csv", aggregate.length === 4 && aggregate[0].startsWith("card,modules_visited"));
  await dashboard.screenshot({ path: path.join(output, "visuals/12_class_dashboard.png"), fullPage: true });

  // ---- the sealed circuits stay unsealed: exposure is shared between tabs and can only grow ---------------------------------------------------
  const pageUrl = pathToFileURL(path.join(output, "index.html")).href, EXPOSURE_KEY = "f1-student-lab-exposure-v1";
  const openTab = async context => {
    const tab = await context.newPage();
    tab.on("pageerror", e => errors.push(`tab: ${e.message}`));
    tab.on("console", message => { if (message.type() === "error" && unexpected(message.text())) errors.push(`tab: ${message.text()}`); });
    await tab.goto(pageUrl); await tab.waitForFunction(() => !!window.studentLab);
    return tab;
  };
  const freshContext = () => browser.newContext({ viewport: { width: 1300, height: 1000 } });
  const shared = await freshContext(), tabA = await openTab(shared), tabB = await openTab(shared);
  check("a_fresh_browser_profile_starts_sealed", await tabA.evaluate(() => !studentLab.testExposed) && await tabA.locator("#track option").count() === 13);
  await visit(tabB, "generalization"); await tabB.locator("#model").selectOption("pretrained"); await tabB.locator("#freezeReveal").click(); await idle(tabB);
  const heard = await tabA.waitForFunction(() => studentLab.testExposed, null, { timeout: 10000 }).then(() => true, () => false);
  check("a_reveal_in_one_tab_unseals_every_open_tab", heard && await tabA.locator("#track option").count() === 16 && (await tabA.locator("#testExposure").innerText()).includes("exposed"));
  // A tab that has not heard about a reveal (simulated by writing the record behind its back) must neither undo it nor call its own freeze the first reveal.
  const staleTab = async () => {
    const tab = await openTab(await freshContext());
    await tab.evaluate(key => localStorage.setItem(key, JSON.stringify({ testExposed: true, testLabelsExposed: false })), EXPOSURE_KEY);
    assert.ok(await tab.evaluate(() => !studentLab.testExposed), "the tab must still believe the circuits are sealed");
    return tab;
  };
  const stale1 = await staleTab(); await visit(stale1, "dataset"); await stale1.locator("#clearDataset").click();
  check("a_stale_tab_cannot_write_exposure_back_to_false", await stale1.evaluate(key => JSON.parse(localStorage.getItem(key)).testExposed === true && studentLab.testExposed === true, EXPOSURE_KEY));
  check("a_tab_that_missed_the_event_still_unseals_its_track_list_and_tables", await stale1.locator("#track option").count() === 16
    && !(await stale1.locator("#referenceControllers tbody td:last-child").allInnerTexts()).includes("Sealed"));
  const stale2 = await staleTab(); await stale2.locator("#model").selectOption("pretrained"); await visit(stale2, "generalization");
  await stale2.locator("#freezeReveal").click(); await idle(stale2);
  check("a_freeze_in_a_stale_tab_is_not_called_the_first_reveal", (await stale2.locator("#freezeStatus").innerText()).includes("already exposed"));
  // Two tabs freezing at once: the one that finishes second must not claim the first reveal, even though it started before the other had revealed anything.
  const raceContext = await freshContext(), slowTab = await openTab(raceContext), fastTab = await openTab(raceContext);
  for (const tab of [slowTab, fastTab]) { await tab.locator("#model").selectOption("pretrained"); await visit(tab, "generalization"); }
  await slowFrames(slowTab, 150); await slowTab.locator("#freezeReveal").click();
  await fastTab.locator("#freezeReveal").click(); await idle(fastTab); await idle(slowTab); await normalFrames(slowTab);
  check("a_reveal_completed_elsewhere_while_a_tab_was_evaluating_is_not_called_the_first", (await fastTab.locator("#freezeStatus").innerText()).includes("first UI reveal")
    && (await slowTab.locator("#freezeStatus").innerText()).includes("already exposed"), `${await fastTab.locator("#freezeStatus").innerText()} | ${await slowTab.locator("#freezeStatus").innerText()}`);
  await tabB.reload(); await tabB.waitForFunction(() => !!window.studentLab);
  check("exposure_survives_a_reload_of_the_revealing_tab", await tabB.evaluate(() => studentLab.testExposed));
  // A damaged record must not stop the page from loading: it is read conservatively, as "already exposed".
  const damaged = await freshContext(), probe = await damaged.newPage();
  probe.on("pageerror", e => errors.push(`damaged-record tab: ${e.message}`));
  await probe.goto(pageUrl); await probe.waitForFunction(() => !!window.studentLab);
  const notHandled = [];
  for (const garbage of ["null", "[]", "42", "\"text\"", "{\"testExposed\":1}", "not json"]) {
    await probe.evaluate(([key, value]) => localStorage.setItem(key, value), [EXPOSURE_KEY, garbage]);
    await probe.reload();
    const booted = await probe.waitForFunction(() => document.querySelectorAll("#track option").length > 0 && document.querySelectorAll("#scenario option").length > 0, null, { timeout: 10000 }).then(() => true, () => false);
    if (!(booted && await probe.evaluate(() => studentLab.testExposed === true) && (await probe.locator("#error").innerText()).includes("cannot be verified"))) notHandled.push(garbage);
  }
  check("a_damaged_exposure_record_never_stops_the_page_and_counts_as_exposed", notHandled.length === 0, JSON.stringify(notHandled));

  if (keepGoing) failedChecks.push(...errors.map(text => `console or page error: ${text.slice(0, 160)}`)); else assert.deepEqual(errors, []);
  assert.deepEqual(await buildHashes(), testedBuild, "the page files changed while they were being tested");
  for (const expected of ["student-driving-dataset.json", "student-driving-dataset.csv", "student-driving-model.json", "controller-comparison.json", "robustness-experiment.json", "ml-driving-worksheet.json", "ml-lab-results-card.json"])
    check(`download_${expected}`, downloadNames.includes(expected));
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({ success: failedChecks.length === 0, modules: exercised, checks, build_sha256: testedBuild, downloads: downloadNames, errors }, null, 2));
  if (failedChecks.length) { console.log(`${failedChecks.length} FAILED CHECKS:\n- ${failedChecks.join("\n- ")}`); process.exitCode = 1; }
  else console.log(`Verified ${exercised.length} modules and ${Object.keys(checks).length} browser checks: training, cancellation, near-duplicate rejection, playground, coverage/DAgger, diagnosis, fresh pack, leakage, worksheets, quiz, results card, cross-tab exposure and class dashboard.`);
} finally { await browser.close(); await fs.rm(scratch, { recursive: true, force: true }); }
