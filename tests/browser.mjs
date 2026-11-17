import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] ?? "artifacts");
const errors = [];
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(pathToFileURL(path.join(output, "replay.html")).href);
  await page.locator("#play").click();
  const metrics = JSON.parse(await fs.readFile(path.join(output, "scenario_metrics.json"), "utf8"));
  for (let round = 0; round <= 120; round++) {
    for (let scenario = 0; scenario < 3; scenario++) {
      await page.evaluate(({ round, scenario }) => {
        const roundControl = document.getElementById("round");
        roundControl.value = round;
        roundControl.dispatchEvent(new Event("input"));
        const scenarioControl = document.getElementById("scenario");
        scenarioControl.value = scenario;
        scenarioControl.dispatchEvent(new Event("change"));
        const scrub = document.getElementById("scrub");
        scrub.value = 1000;
        scrub.dispatchEvent(new Event("input"));
      }, { round, scenario });
      const label = await page.locator("#roundLabel").textContent();
      assert.equal(label, String(round));
      const progress = await page.locator("#progress").textContent();
      const expected = metrics[round * 3 + scenario];
      assert.ok(Math.abs(parseFloat(progress) - expected.progress * 100) < 0.06,
        `Replay progress mismatch at round ${round}, scenario ${scenario}`);
      const result = await page.locator("#result").textContent();
      assert.ok(result.includes(expected.end_reason.replaceAll("_", " ")));
    }
  }
  await page.evaluate(() => {
    document.getElementById("round").value = 120;
    document.getElementById("round").dispatchEvent(new Event("input"));
    document.getElementById("scenario").value = 0;
    document.getElementById("scenario").dispatchEvent(new Event("change"));
    document.getElementById("scrub").value = 1000;
    document.getElementById("scrub").dispatchEvent(new Event("input"));
  });
  const canvasPixels = await page.locator("#view").evaluate(canvas => {
    const context = canvas.getContext("2d");
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let index = 3; index < pixels.length; index += 4) {
      if (pixels[index] > 0) painted++;
    }
    return { contextLost: context.isContextLost(), painted };
  });
  assert.equal(canvasPixels.contextLost, false, "Canvas context was lost");
  assert.ok(canvasPixels.painted > 10000, "Track canvas did not render");
  await page.screenshot({ path: path.join(output, "visuals/replay_screenshot.png"), fullPage: true });
  await page.locator("#restart").click();
  const startTime = await page.locator("#time").textContent();
  await page.waitForTimeout(500);
  const endTime = await page.locator("#time").textContent();
  assert.notEqual(startTime, endTime, "Playback clock is not advancing");
  await page.locator("#play").click();
  await page.goto(pathToFileURL(path.join(output, "training_deck.html")).href);
  assert.equal(await page.locator(".slide").count(), 15);
  for (let slide = 1; slide <= 15; slide++) {
    assert.equal(await page.locator(".slide.active").count(), 1);
    assert.equal(await page.locator("#count").textContent(), `${slide} / 15`);
    const imagesReady = await page.locator(".slide.active img").evaluateAll(
      images => images.every(image => image.complete && image.naturalWidth > 0)
    );
    assert.ok(imagesReady, `Broken image on slide ${slide}`);
    await page.screenshot({
      path: path.join(output, "visuals", `slide_${String(slide).padStart(2, "0")}.png`),
      fullPage: true,
    });
    if (slide < 15) await page.keyboard.press("ArrowRight");
  }
  await page.keyboard.press("ArrowLeft");
  assert.equal(await page.locator("#count").textContent(), "14 / 15");
  assert.deepEqual(errors, [], "Browser raised errors");
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({
    passed: true,
    replay_rounds_tested: 121,
    replay_scenarios_tested: 363,
    slide_images_and_navigation_tested: 15,
    canvas_rendering_tested: true,
    playback_clock_tested: true,
    console_errors: errors,
  }, null, 2) + "\n");
  console.log("PASS: all 363 replay states, playback controls, 15 slides, images, and keyboard navigation.");
} finally {
  await browser.close();
}
