// Real-browser verification of the student lab (version 2): all eleven modules, the teacher tools and the failure paths.
//   PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" node tests/student-browser.mjs [student_v2_artifacts]
// Every check runs the real engine in Chromium; nothing is mocked. Writes browser_verification.json, teacher_guide.pdf and screenshots.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] || "student_v2_artifacts");
const kit = JSON.parse(await fs.readFile("teaching_kit.json", "utf8"));
const python = process.env.PYTHON || (existsSync(".venv/bin/python") ? ".venv/bin/python" : "python3");
const errors = [], checks = {}, exercised = [], downloadNames = [], scratch = await fs.mkdtemp(path.join(os.tmpdir(), "f1-lab-"));
const check = (name, condition, detail = "") => { assert.ok(condition, `Check failed: ${name} ${detail}`); checks[name] = true; };
// The page logs every handled error to the console. These come from the failure paths this test triggers on purpose; anything else is a bug.
const intended = [/Make both predictions/, /nearly identical/, /does not match this lab/, /Learning rate: choose a number between/, /not a fresh test pack/, /Answer every question/];
const unexpected = text => !intended.some(pattern => pattern.test(text));
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });

async function setRange(page, id, value) {
  await page.locator(`#${id}`).evaluate((input, next) => { input.value = next; input.dispatchEvent(new Event("input")); }, String(value));
}
const idle = (page, timeout = 240000) => page.waitForFunction(() => !studentLab.busy, null, { timeout });
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
  page.on("download", async download => { downloadNames.push(download.suggestedFilename()); await download.saveAs(path.join(output, "data", download.suggestedFilename())); });
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
  const csv = await fs.readFile(path.join(output, "data/student-driving-dataset.csv"), "utf8");
  check("csv_has_labels_and_all_inputs", csv.split("\r\n")[0].includes("label_steering_fraction") && csv.split("\r\n")[0].split(",").length === 5 + 18 + 2);
  const savedDataset = await fs.readFile(path.join(output, "data/student-driving-dataset.json"), "utf8");
  await page.locator("#importDataset").setInputFiles({ name: "wrong.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ version: "other", rows: [] })) });
  await page.waitForFunction(() => document.getElementById("error").textContent.length > 0);
  check("dataset_from_another_version_is_rejected", (await error(page)).includes("does not match"));
  await page.locator("#importDataset").setInputFiles({ name: "import.json", mimeType: "application/json", buffer: Buffer.from(savedDataset) });
  await page.waitForFunction(() => document.getElementById("status").textContent.includes("Imported"));
  await page.locator("#updates").fill("400"); await page.locator("#trainStudent").click();
  await page.waitForFunction(() => studentLab.losses.length >= 3);
  check("cancel_button_appears_while_training", await page.locator("#cancelRun").isVisible());
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
  await page.locator("#daggerRollout").click(); await idle(page); await page.locator("#daggerTrain").click(); await idle(page);
  check("dagger_session_adds_learner_visited_states", await page.evaluate(() => studentLab.dagger.round === 1 && studentLab.dagger.rows.some(r => /learner rollout/.test(r.source)) && studentLab.dagger.log.length >= 2));
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
  await page.locator("#freshPack").setInputFiles({ name: "bad.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ schema: "other" })) });
  await page.waitForFunction(() => document.getElementById("error").textContent.includes("not a fresh test pack"));
  check("a_foreign_file_is_not_accepted_as_a_fresh_pack", true);
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

  // ---- 11. worksheets, quiz and results card -----------------------------------------------------------------------------------------------
  await visit(page, "worksheets");
  check("every_worksheet_listed", await page.locator("#worksheet option").count() === kit.worksheets.length);
  for (const [index, worksheet] of kit.worksheets.entries()) {
    await page.locator("#worksheet").selectOption(String(index)); await page.locator("#hypothesis").fill(`Prediction for ${worksheet.title}: it will change.`);
    for (const id of ["wsSeeds"]) if (await page.locator(`#${id}`).count()) await page.locator(`#${id}`).selectOption("3");
    await page.locator("#runWorksheet").click(); await idle(page, 300000);
    check(`worksheet_${worksheet.id}_produced_results`, (await error(page)) === "" && await rowCount(page, "worksheetResults") >= 1, `${worksheet.id}: ${await error(page)}`);
  }
  await page.locator("#worksheet").selectOption(String(kit.worksheets.findIndex(w => w.runner === "shift")));
  await page.locator("#runWorksheet").click(); await idle(page);
  await page.locator("#evidence").fill("Compared identical weights over matched seeds."); await page.locator("#conclusion").fill("A toy-model result is not a real-car safety guarantee.");
  await page.locator("#exportWorksheet").click(); await page.waitForTimeout(200);
  const worksheetExport = JSON.parse(await fs.readFile(path.join(output, "data/ml-driving-worksheet.json"), "utf8"));
  check("worksheet_export_has_hypothesis_evidence_and_rubric", worksheetExport.hypothesis.length > 0 && worksheetExport.evidence.length > 0 && worksheetExport.rubric.criteria.length === kit.rubric.criteria.length);
  const firstOrder = await page.evaluate(() => JSON.stringify(studentLab.quizOrder));
  check("quiz_choices_are_shuffled", await page.evaluate(() => studentLab.quizOrder.some(order => order[0] !== 0)));
  await page.locator("#gradeQuiz").click();
  check("quiz_requires_every_answer", (await error(page)).includes("Answer every question"));
  await page.locator("#quizPhase").selectOption("pre");
  for (let q = 0; q < kit.quiz.length; q++) await page.locator(`input[name="quiz${q}"][value="1"]`).check();
  await page.locator("#gradeQuiz").click();
  check("wrong_answers_are_explained_and_mapped_to_misconceptions", (await page.locator("#quizReview").innerText()).includes("Common misconception") || (await page.locator("#quiz").innerText()).includes("Common misconception"));
  await page.locator("#quizPhase").selectOption("post");
  check("a_new_attempt_reshuffles_the_choices", await page.evaluate(() => JSON.stringify(studentLab.quizOrder)) !== firstOrder);
  for (let q = 0; q < kit.quiz.length; q++) await page.locator(`input[name="quiz${q}"][value="0"]`).check();
  await page.locator("#gradeQuiz").click();
  check("correct_answers_score_fully", (await page.locator("#quizFeedback").innerText()).startsWith(`${kit.quiz.length}/${kit.quiz.length}`));
  check("pre_and_post_attempts_recorded", await page.evaluate(() => studentLab.quizAttempts.map(a => a.phase).join()) === "pre,post");
  await page.locator("#nickname").fill("Ada <script>");
  await page.locator("#downloadCard").click(); await page.waitForTimeout(250);
  const card = JSON.parse(await fs.readFile(path.join(output, "data/ml-lab-results-card.json"), "utf8"));
  check("results_card_has_no_free_text_or_timestamps", card.schema === "f1-ml-lab-results-card-v1" && !JSON.stringify(card).includes("<script") && !/\d{4}-\d{2}-\d{2}T/.test(JSON.stringify(card)) && !("evidence" in card) && !("hypothesis" in card));
  check("results_card_records_what_was_done", card.modules_visited.length === kit.modules.length && card.quiz.length === 2 && card.worksheets_completed.length >= kit.worksheets.length - 1 && card.exposure.test_exposed === true);
  await finish(page, "worksheets", "11_worksheets");

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
  dashboard.on("download", async download => { downloadNames.push(download.suggestedFilename()); await download.saveAs(path.join(output, "data", download.suggestedFilename())); });
  await dashboard.goto(pathToFileURL(path.join(output, "class_dashboard.html")).href);
  const synthetic = (modules, pre, post, misconceptions) => ({ ...card, nickname: "x", modules_visited: modules, quiz: [
    { phase: "pre", score: pre, total: 16, by_objective: { O1: [pre > 8 ? 1 : 0, 1] }, misconceptions: [] },
    { phase: "post", score: post, total: 16, by_objective: { O1: [1, 1] }, misconceptions }] });
  const files = [card, synthetic(["predict", "dataset"], 6, 12, ["M1", "M5"]), synthetic(kit.modules.map(m => m.id), 9, 15, ["M1"])]
    .map((value, i) => ({ name: `card${i}.json`, mimeType: "application/json", buffer: Buffer.from(JSON.stringify(value)) }));
  files.push({ name: "notes.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ hello: "world" })) });
  await dashboard.locator("#cards").setInputFiles(files);
  await dashboard.waitForFunction(() => document.getElementById("loaded").textContent.length > 0);
  check("dashboard_aggregates_valid_cards_and_rejects_others", (await dashboard.locator("#loaded").innerText()).includes("3 cards loaded, 1 rejected")
    && (await dashboard.locator("#error").innerText()).includes("notes.json") && await dashboard.locator("#cardsTable tbody tr").count() === 3);
  const misconceptionText = await dashboard.locator("#misconceptionsTable").innerText();
  check("dashboard_counts_misconceptions_from_the_latest_attempt", misconceptionText.includes("M1") && misconceptionText.includes("2/3") && misconceptionText.includes("M5"));
  check("dashboard_reports_a_descriptive_paired_gain", (await dashboard.locator("#quizSummary").innerText()).includes("not evidence of learning"));
  check("dashboard_hides_nicknames_by_default", !(await dashboard.locator("#cardsTable").innerText()).includes("Ada"));
  await dashboard.locator("#exportCsv").click(); await dashboard.waitForTimeout(300);
  const aggregate = (await fs.readFile(path.join(output, "data/class-aggregate.csv"), "utf8")).split("\r\n");
  check("dashboard_exports_an_aggregate_csv", aggregate.length === 4 && aggregate[0].startsWith("card,modules_visited"));
  await dashboard.screenshot({ path: path.join(output, "visuals/12_class_dashboard.png"), fullPage: true });

  assert.deepEqual(errors, []);
  for (const expected of ["student-driving-dataset.json", "student-driving-dataset.csv", "student-driving-model.json", "controller-comparison.json", "robustness-experiment.json", "ml-driving-worksheet.json", "ml-lab-results-card.json"])
    check(`download_${expected}`, downloadNames.includes(expected));
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({ success: true, modules: exercised, checks, downloads: downloadNames, errors }, null, 2));
  console.log(`Verified ${exercised.length} modules and ${Object.keys(checks).length} browser checks: training, cancellation, near-duplicate rejection, playground, coverage/DAgger, diagnosis, fresh pack, leakage, worksheets, quiz, results card and class dashboard.`);
} finally { await browser.close(); await fs.rm(scratch, { recursive: true, force: true }); }
