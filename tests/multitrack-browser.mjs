import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] ?? "multitrack_artifacts");
const errors = [];
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(pathToFileURL(path.join(output, "replay.html")).href);
  await page.locator("#play").click();
  const history = JSON.parse(await fs.readFile(path.join(output, "history.json"), "utf8"));
  const learningErrors = await page.evaluate(expected => {
    const failures = [];
    for (let round = 0; round <= 120; round++) {
      const control = document.getElementById("round");
      control.value = round;
      control.dispatchEvent(new Event("input"));
      const progress = parseFloat(document.getElementById("progress").textContent);
      if (Math.abs(progress - expected[round].train_progress_pct) > 0.06)
        failures.push({ round, progress });
      if (document.getElementById("primary").textContent !== `${expected[round].train_completed}/33`)
        failures.push({ round, missingCompletion: true });
    }
    return failures;
  }, history);
  assert.deepEqual(learningErrors, [], "Training animation UI mismatches saved metrics");
  await page.screenshot({ path: path.join(output, "visuals/training_replay_screenshot.png"), fullPage: true });
  await page.locator("#mode").selectOption("drive");
  assert.equal(await page.locator("#track option").count(), 13);
  const metrics = JSON.parse(await fs.readFile(path.join(output, "scenario_metrics.json"), "utf8"));
  const drivingErrors = await page.evaluate(expected => {
    const failures = [];
    const names = ["Start straight", "Corner entry", "Recovery start"];
    for (const metric of expected) {
      for (const [id, value, event] of [
        ["round", metric.round, "input"],
        ["track", metric.track_id, "change"],
        ["scenario", names.indexOf(metric.scenario), "change"],
        ["scrub", 1000, "input"],
      ]) {
        const control = document.getElementById(id);
        control.value = value;
        control.dispatchEvent(new Event(event));
      }
      const progress = parseFloat(document.getElementById("progress").textContent);
      const result = document.getElementById("result").textContent;
      if (Math.abs(progress - metric.progress * 100) > 0.06
          || !result.includes(metric.end_reason.replaceAll("_", " ")))
        failures.push({ round: metric.round, track: metric.track_id, scenario: metric.scenario, progress, result });
    }
    return failures;
  }, metrics);
  assert.deepEqual(drivingErrors, [], "Driving replay mismatches stored outcomes");
  const pixels = await page.locator("#view").evaluate(canvas => {
    const context = canvas.getContext("2d");
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i]) painted++;
    return { lost: context.isContextLost(), painted };
  });
  assert.equal(pixels.lost, false);
  assert.ok(pixels.painted > 10000, "Canvas is blank");
  await page.locator("#round").fill("120");
  await page.locator("#track").selectOption("9");
  await page.locator("#scenario").selectOption("0");
  await page.locator("#scrub").fill("1000");
  await page.screenshot({ path: path.join(output, "visuals/driving_replay_screenshot.png"), fullPage: true });
  await page.locator("#restart").click();
  const before = await page.locator("#time").textContent();
  await page.waitForTimeout(500);
  assert.notEqual(await page.locator("#time").textContent(), before, "Driving clock did not advance");
  await page.locator("#play").click();
  await page.goto(pathToFileURL(path.join(output, "training_deck.html")).href);
  assert.equal(await page.locator(".slide").count(), 8);
  for (let slide = 1; slide <= 8; slide++) {
    assert.equal(await page.locator("#count").textContent(), `${slide} / 8`);
    assert.ok(await page.locator(".slide.active img").evaluate(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({ path: path.join(output, "visuals", `slide_${String(slide).padStart(2, "0")}.png`), fullPage: true });
    if (slide < 8) await page.keyboard.press("ArrowRight");
  }
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({
    passed: true, learning_rounds_tested: 121, drive_states_tested: 4719,
    canvas_rendering_tested: true, playback_clock_tested: true,
    presentation_slides_tested: 8, console_errors: errors,
  }, null, 2) + "\n");
  console.log("PASS: all 121 learning rounds, 4,719 driving states, canvas rendering, playback and 8 slides.");
} finally {
  await browser.close();
}
