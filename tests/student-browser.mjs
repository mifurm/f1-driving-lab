import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] || "student_artifacts"), errors = [];
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
const downloadNames = [], exercised = [];
async function setRange(page, id, value) {
  await page.locator(`#${id}`).evaluate((input, next) => { input.value = next; input.dispatchEvent(new Event("input")); }, String(value));
}
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 }, acceptDownloads: true });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  page.on("download", async download => { downloadNames.push(download.suggestedFilename()); await download.saveAs(path.join(output, "data", download.suggestedFilename())); });
  await page.goto(pathToFileURL(path.join(output, "index.html")).href);
  await page.waitForFunction(() => !!window.studentLab);
  assert.equal(await page.locator("#track option").count(), 13);
  assert.equal(await page.locator("#inputTable tbody tr").count(), 18);
  assert.equal(await page.locator("#referenceControllers").innerText().then(text => text.includes("Sealed")), true);
  assert.equal(await page.locator("#ablationTable").innerText().then(text => text.includes("pristine_test")), false);
  await page.locator("#guessDirection").selectOption("right"); await page.locator("#guessBrake").selectOption("yes");
  await page.locator("#revealPrediction").click();
  await page.waitForFunction(() => studentLab.runs[0].trace.length > 50);
  assert.equal(await page.locator("#predictionResult").isVisible(), true);
  const painted = await page.evaluate(() => {
    const canvas = document.getElementById("view"), image = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
    return new Set(Array.from({ length: image.data.length / 4 }, (_, i) => `${image.data[i * 4]},${image.data[i * 4 + 1]},${image.data[i * 4 + 2]}`)).size;
  });
  assert.ok(painted > 100, "Canvas must actually be painted");
  await page.screenshot({ path: path.join(output, "visuals/01_prediction.png"), fullPage: true }); exercised.push("prediction");
  const poseBefore = await page.evaluate(() => studentLab.replayTime);
  await page.waitForTimeout(100); assert.ok(await page.evaluate(() => studentLab.replayTime) > poseBefore);

  await page.locator("#tabDataset").click(); await page.locator("#newDrive").click();
  const observationBefore = await page.evaluate(() => studentLab.manual.env.observe().features);
  await setRange(page, "manualSteer", -10); await setRange(page, "manualTarget", 25);
  await page.locator("#recordStep").click();
  assert.deepEqual(await page.evaluate(() => studentLab.rows[0].features), observationBefore);
  const expectedSteer = -10 * Math.PI / 180 / .5;
  assert.ok(Math.abs((await page.evaluate(() => studentLab.rows[0].targets[0])) - expectedSteer) < 1e-10);
  await page.locator("#recordSplit").selectOption("validation"); await page.locator("#recordStep").click();
  assert.equal(await page.evaluate(() => studentLab.rows[1].split), "train", "An existing drive cannot change split");
  await page.locator("#track").selectOption("1"); await page.locator("#newDrive").click(); await page.locator("#recordStep").click();
  assert.equal(await page.evaluate(() => studentLab.rows[2].split), "validation");
  await page.locator("#recordPlay").click();
  await page.waitForFunction(() => studentLab.rows.length >= 5);
  await page.locator("#recordPlay").click();
  await page.locator("#loadStarter").click();
  await page.locator("#downloadDataset").click(); await page.locator("#downloadCSV").click(); await page.waitForFunction(() => !studentLab.busy);
  await page.waitForTimeout(200);
  assert.ok(downloadNames.includes("student-driving-dataset.json"));
  const savedDataset = await fs.readFile(path.join(output, "data/student-driving-dataset.json"), "utf8");
  await page.locator("#importDataset").setInputFiles({ name: "import.json", mimeType: "application/json", buffer: Buffer.from(savedDataset) });
  await page.waitForFunction(() => document.getElementById("status").textContent.includes("Imported"));
  await page.locator("#updates").fill("40"); await page.locator("#trainStudent").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.student?.updates === 40, null, { timeout: 120000 });
  assert.ok(await page.evaluate(() => studentLab.losses.at(-1).train < studentLab.losses[0].train));
  await page.locator("#animateStudentLearning").click();
  await page.waitForFunction(() => studentLab.runs.length === 3);
  assert.deepEqual(await page.evaluate(() => studentLab.trainingSnapshots.map(s => s.update)), [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]);
  await setRange(page, "learningUpdate", 3); await page.locator("#exportModel").click();
  await page.screenshot({ path: path.join(output, "visuals/02_dataset_training.png"), fullPage: true }); exercised.push("dataset_training_and_import_export");

  await page.locator("#tabStep").click(); await page.locator("#inspectStep").click();
  await page.waitForFunction(() => studentLab.packet?.updates === 41);
  assert.ok(await page.evaluate(() => Math.abs(studentLab.packet.numericGradient - studentLab.packet.changes[0].gradient) < 1e-7));
  for (let i = 0; i < 4; i++) await page.locator("#stepNext").click();
  assert.ok((await page.locator("#stepStage").innerText()).includes("Adam"));
  assert.equal(await page.locator("#weightTable tbody tr").count(), 6);
  assert.equal(await page.locator("#batchTable tbody tr").count(), 8);
  const displayedLoss = await page.locator("#batchTable tbody tr").evaluateAll(rows =>
    rows.reduce((sum, row) => sum + Number(row.lastElementChild.textContent), 0) / rows.length);
  assert.ok(Math.abs(displayedLoss - await page.evaluate(() => studentLab.packet.loss)) < 1e-7);
  await page.locator("#driveAfterStep").click(); assert.equal(await page.evaluate(() => studentLab.runs.length), 2);
  await page.screenshot({ path: path.join(output, "visuals/03_actual_gradient.png"), fullPage: true }); exercised.push("real_minibatch_gradient");

  await page.locator("#tabControllers").click(); await page.locator("#model").selectOption("pretrained");
  await page.locator("#compareControllers").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.comparisons?.rows.length === 9, null, { timeout: 120000 });
  assert.equal(await page.locator("#controllerTable tbody tr").count(), 3);
  const paired = await page.evaluate(() => studentLab.comparisons.rows.filter(row => row.seed === 401).map(row => row.initial));
  assert.deepEqual(paired[0], paired[1]); assert.deepEqual(paired[1], paired[2]);
  await page.locator("#downloadComparison").click();
  await page.screenshot({ path: path.join(output, "visuals/04_controllers.png"), fullPage: true }); exercised.push("three_paired_controllers");

  await page.locator("#tabFeatures").click(); await page.locator("#updates").evaluate(input => { input.value = 8; });
  await page.locator("#localAblation").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.localAblations?.models.length === 12, null, { timeout: 120000 });
  assert.ok(await page.evaluate(() => studentLab.localAblations.models.every(m => m.actual_updates === 8)));
  await page.locator("#animateReferenceLearning").click(); assert.equal(await page.evaluate(() => studentLab.runs.length), 3);
  await page.screenshot({ path: path.join(output, "visuals/05_feature_retraining.png"), fullPage: true }); exercised.push("12_actual_local_ablations_and_reference_snapshots");

  await page.locator("#tabGeneralization").click(); await page.locator("#freezeReveal").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.testExposed, null, { timeout: 120000 });
  assert.equal(await page.locator("#track option").count(), 16); assert.equal(await page.locator("#testTable tbody tr").count(), 12);
  const frozen = await page.evaluate(() => StudentEngine.fingerprint(studentLab.frozen));
  await page.locator("#demonstrateLeak").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.testLabelsExposed, null, { timeout: 120000 });
  assert.equal(await page.evaluate(() => StudentEngine.fingerprint(studentLab.frozen)), frozen);
  assert.equal(await page.evaluate(() => studentLab.leakage.valid_generalization), false);
  assert.ok((await page.locator("#leakStatus").innerText()).includes("INVALID"));
  await page.screenshot({ path: path.join(output, "visuals/06_generalization_leakage.png"), fullPage: true }); exercised.push("freeze_test_and_genuine_leakage");

  await page.locator("#tabRobustness").click(); await page.locator("#runRobustness").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.robustness?.rows.length === 6, null, { timeout: 120000 });
  assert.equal(await page.locator("#robustnessTable tbody tr").count(), 2);
  assert.equal(await page.evaluate(() => studentLab.robustness.rows[3].settings.grip), .65);
  await page.locator("#downloadRobustness").click();
  await page.screenshot({ path: path.join(output, "visuals/07_robustness.png"), fullPage: true }); exercised.push("genuine_noise_delay_and_grip");

  await page.locator("#tabWorksheets").click(); assert.equal(await page.locator("#worksheet option").count(), 8);
  await page.locator("#hypothesis").fill("Observation noise will reduce the fraction of safe centered exits.");
  await page.locator("#runWorksheet").click();
  await page.waitForFunction(() => !studentLab.busy && studentLab.worksheet?.rows.length === 9, null, { timeout: 120000 });
  assert.equal(await page.locator("#worksheetResults tbody tr").count(), 3);
  const settings = await page.evaluate(() => studentLab.worksheet.conditions);
  assert.ok(settings.every(condition => condition.delay === 0 && condition.grip === 1));
  const answers = [1, 1, 0, 2, 0];
  await page.locator("#quizPhase").selectOption("post");
  for (const [i, answer] of answers.entries()) await page.locator(`input[name="quiz${i}"][value="${answer}"]`).check();
  await page.locator("#gradeQuiz").click(); assert.ok((await page.locator("#quizFeedback").innerText()).startsWith("5/5"));
  assert.equal(await page.evaluate(() => studentLab.quizAttempts.at(-1).phase), "post");
  await page.locator("#evidence").fill("Compared identical model weights and three paired seeds across noise values.");
  await page.locator("#conclusion").fill("A toy-model result is not a real-car safety guarantee.");
  await page.locator("#exportWorksheet").click();
  await page.screenshot({ path: path.join(output, "visuals/08_worksheet.png"), fullPage: true }); exercised.push("one_factor_repeats_quiz_and_export");
  await page.locator("#tabDataset").click(); await page.locator("#clearDataset").click();
  assert.equal(await page.evaluate(() => studentLab.testExposed && studentLab.testLabelsExposed), true);
  await page.reload(); await page.waitForFunction(() => !!window.studentLab);
  assert.equal(await page.evaluate(() => studentLab.testExposed && studentLab.testLabelsExposed), true, "Exposure must persist across reloads");
  await page.goto(pathToFileURL(path.join(output, "teacher_guide.html")).href);
  await page.pdf({ path: path.join(output, "teacher_guide.pdf"), format: "A4", printBackground: false, margin: { top: "12mm", bottom: "12mm", left: "10mm", right: "10mm" } });
  await page.waitForTimeout(250);
  assert.deepEqual(errors, []);
  for (const expected of ["student-driving-dataset.json", "student-driving-dataset.csv", "student-driving-model.json", "controller-comparison.json", "robustness-experiment.json", "ml-driving-worksheet.json"])
    assert.ok(downloadNames.includes(expected), `Missing actual download ${expected}`);
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({ success: true, modules: exercised, downloads: downloadNames, painted_colors: painted, errors }, null, 2));
  console.log(`Verified all eight modules, manual recording, genuine training/gradients, 12 live ablations, frozen tests, leakage and downloads.`);
} finally { await browser.close(); }
