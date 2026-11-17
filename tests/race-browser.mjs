import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] ?? "race_artifacts");
const errors = [];
const catalog = JSON.parse(await fs.readFile(path.join(output, "data/circuits.json"), "utf8"));
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
let attemptsTested = 0, rolloutsTested = 0, brakeLightsTested = 0;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  for (const circuit of catalog) {
    await page.goto(pathToFileURL(path.join(output, `tracks/track_${String(circuit.id).padStart(2, "0")}.html`)).href);
    await page.locator("#play").click();
    await page.locator("#mode").selectOption("single");
    const checked = await page.evaluate(() => {
      const failures = [];
      const set = (id, value, event) => {
        $(id).value = value;
        $(id).dispatchEvent(new Event(event));
      };
      function assertHud(trace, label) {
        const state = trace.at(-1);
        const expectations = {
          velocity: (state[4] * 3.6).toFixed(0) + " km/h",
          target: (state[13] * 3.6).toFixed(0) + " km/h",
          gear: "G" + state[10], rpm: state[11].toFixed(0),
          throttleText: (state[8] * 100).toFixed(0) + "%",
          brakeText: (state[9] * 100).toFixed(0) + "%",
          progress: (Math.min(Math.max(state[7], 0), 1) * 100).toFixed(1) + "%",
          time: state[0].toFixed(1) + " s",
        };
        for (const [id, value] of Object.entries(expectations))
          if ($(id).textContent !== value) failures.push(`${label}: ${id}`);
        if ($("gears").querySelectorAll(".active").length !== 1
            || $("gears").querySelector(".active").textContent !== String(state[10]))
          failures.push(`${label}: gear badge`);
      }
      for (const attempt of data.attempts) {
        set("round", attempt.round, "input");
        set("scrub", 1000, "input");
        assertHud(attempt.trace, `attempt ${attempt.round}`);
        if (!$("result").textContent.includes(attempt.metric.end_reason.replaceAll("_", " ")))
          failures.push(`attempt ${attempt.round}: outcome`);
      }
      const count = data.attempts.length, rolloutCount = data.rollouts.length;
      if (rolloutCount) {
        set("mode", "rollout", "change");
        for (let index = 0; index < rolloutCount; index++) {
          set("round", index, "input");
          set("scrub", 1000, "input");
          assertHud(data.rollouts[index].trace, `collection ${index + 1}`);
          if ($("roundLabel").textContent !== `${index + 1} (policy ${index})`
              || $("roundKind").textContent !== "Training collection")
            failures.push(`collection ${index + 1}: wrong policy`);
        }
      } else if (!$("rolloutOption").disabled) failures.push("Unseen circuit offers training footage");
      set("mode", "single", "change");
      set("round", 120, "input");
      const trace = data.attempts[120].trace;
      const peak = trace.reduce((a, b) => a[9] > b[9] ? a : b);
      time = peak[0];
      update();
      if (peak[9] <= .02) failures.push("No braking in final drive");
      const [minX, maxX, minY, maxY] = bounds;
      const scale = Math.min((1100 - 44) / (maxX - minX), (650 - 45 - 22) / (maxY - minY));
      const ox = (1100 - scale * (maxX - minX)) / 2;
      const oy = 45 + (650 - 45 - scale * (maxY - minY)) / 2;
      const x = ox + (peak[1] - minX) * scale, y = oy + (maxY - peak[2]) * scale;
      const lx = -8 * 1.15, ly = 4.1 * 1.15;
      const rx = Math.round(x + lx * Math.cos(peak[3]) + ly * Math.sin(peak[3]));
      const ry = Math.round(y - lx * Math.sin(peak[3]) + ly * Math.cos(peak[3]));
      const pixels = ctx.getImageData(rx - 2, ry - 2, 5, 5).data;
      let redLight = false;
      for (let i = 0; i < pixels.length; i += 4)
        if (pixels[i] > 180 && pixels[i] > pixels[i + 1] * 1.4) redLight = true;
      if (!redLight) failures.push("Recorded brake input did not paint a red rear lamp");
      for (const id of ["view", "telemetry", "learning"]) {
        const cv = $(id), c = cv.getContext("2d"), image = c.getImageData(0, 0, cv.width, cv.height).data;
        let painted = 0;
        for (let i = 3; i < image.length; i += 4) if (image[i]) painted++;
        if (c.isContextLost() || painted < 1000) failures.push(`Blank ${id} canvas`);
      }
      if ($("attemptTable").querySelectorAll("tbody tr").length !== 121) failures.push("Attempt table incomplete");
      return { failures, count, rolloutCount, redLight };
    });
    assert.deepEqual(checked.failures, [], `${circuit.name}: per-track replay verification failed`);
    attemptsTested += checked.count;
    rolloutsTested += checked.rolloutCount;
    brakeLightsTested += Number(checked.redLight);
    await page.screenshot({
      path: path.join(output, `visuals/track_${String(circuit.id).padStart(2, "0")}_cockpit.png`), fullPage: true,
    });
    await page.locator("#mode").selectOption("compare");
    await page.locator("#round").fill("60");
    await page.locator("#scrub").fill("500");
    assert.match(await page.locator("#sourceCaption").textContent(), /FIXED TEST/);
    await page.locator("#mode").selectOption("single");
    await page.locator("#round").fill("120");
    await page.locator("#ghost").check();
    await page.locator("#ghost").uncheck();
    if (circuit.id === 0) {
      await page.locator("#speed").selectOption("1");
      await page.locator("#restart").click();
      const before = await page.evaluate(() => ({ pose: sample(active().trace, time).state.slice(1, 3), image: canvas.toDataURL() }));
      await page.waitForTimeout(350);
      const after = await page.evaluate(() => ({ pose: sample(active().trace, time).state.slice(1, 3), image: canvas.toDataURL() }));
      assert.ok(Math.hypot(after.pose[0] - before.pose[0], after.pose[1] - before.pose[1]) > .1, "Car did not move");
      assert.notEqual(after.image, before.image, "Moving car did not repaint the canvas");
      await page.locator("#play").click();
      await page.locator("#mode").selectOption("sequence");
      await page.locator("#round").fill("0");
      await page.locator("#speed").selectOption("32");
      await page.locator("#play").click();
      await page.waitForFunction(() => +document.getElementById("round").value >= 1, null, { timeout: 10000 });
      await page.locator("#play").click();
      const prior = Number(await page.locator("#round").inputValue());
      await page.locator("#next").click();
      assert.equal(Number(await page.locator("#round").inputValue()), prior + 1);
    }
    console.log(`PASS ${circuit.name}: 121 attempts, ${checked.rolloutCount} training episodes and visible brake lights`);
  }
  await page.goto(pathToFileURL(path.join(output, "index.html")).href);
  assert.equal(await page.locator("#trackSelect option").count(), 13);
  assert.equal(await page.locator(".gallery article").count(), 13);
  await page.locator("#trackSelect").selectOption("12");
  await page.frameLocator("#cockpit").locator("#title").filter({ hasText: catalog[12].name }).waitFor();
  assert.equal(await page.frameLocator("#cockpit").locator("#title").textContent(), catalog[12].name);
  await page.goto(pathToFileURL(path.join(output, "training_deck.html")).href);
  assert.equal(await page.locator(".slide").count(), 8);
  for (let slide = 1; slide <= 8; slide++) {
    assert.equal(await page.locator("#count").textContent(), `${slide} / 8`);
    assert.ok(await page.locator(".slide.active img").evaluate(image => image.complete && image.naturalWidth > 0));
    await page.screenshot({ path: path.join(output, `visuals/deck_${String(slide).padStart(2, "0")}.png`), fullPage: true });
    if (slide < 8) await page.keyboard.press("ArrowRight");
  }
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({
    passed: true, attempts_tested: attemptsTested, training_rollouts_tested: rolloutsTested,
    brake_lights_tested: brakeLightsTested, movement_tested: true, sequence_advance_tested: true,
    presentation_slides_tested: 8, console_errors: errors,
  }, null, 2) + "\n");
} finally {
  await browser.close();
}
