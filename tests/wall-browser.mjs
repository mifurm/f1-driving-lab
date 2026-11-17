import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const output = path.resolve(process.argv[2] ?? "wall_artifacts");
const catalog = JSON.parse(await fs.readFile(path.join(output, "data/circuits.json"), "utf8"));
const manifest = JSON.parse(await fs.readFile(path.join(output, "manifest.json"), "utf8"));
const errors = [];
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
let attempts = 0, rollouts = 0, wallDemos = 0;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  for (const circuit of catalog) {
    await page.goto(pathToFileURL(path.join(output, `tracks/track_${String(circuit.id).padStart(2, "0")}.html`)).href);
    await page.locator("#play").click();
    await page.locator("#mode").selectOption("single");
    const result = await page.evaluate(() => {
      const failures = [];
      function set(id, value, event) { $(id).value = value; $(id).dispatchEvent(new Event(event)); }
      function hud(trace, label) {
        const s = trace.at(-1);
        for (const [id, expected] of Object.entries({
          margin: s[17].toFixed(2) + " m", lateral: s[6].toFixed(2) + " m",
          velocity: (s[4] * 3.6).toFixed(0) + " km/h", gear: `G${s[10]} / ${s[11].toFixed(0)}`,
          heading: (s[18] * 180 / Math.PI).toFixed(1) + "°",
          progress: (s[19] * 100).toFixed(1) + "%", brakeText: (s[9] * 100).toFixed(0) + "%",
          steering: (s[5] * 180 / Math.PI).toFixed(1) + "°", target: (s[13] * 3.6).toFixed(0) + " km/h",
        })) if ($(id).textContent !== expected) failures.push(`${label}: wrong ${id}`);
      }
      for (let number = 0; number <= data.rounds; number++) {
        for (let scenario = 0; scenario < 4; scenario++) {
          set("scenario", scenario, "change");
          set("round", number, "input");
          set("scrub", 1000, "input");
          hud(data.attempts[number][scenario].trace, `${number}/${scenario}`);
          if (!$("result").textContent.includes(data.attempts[number][scenario].metric.end_reason.replaceAll("_", " ")))
            failures.push(`${number}/${scenario}: wrong failure reason`);
        }
      }
      if (data.rollouts.length) {
        set("mode", "rollout", "change");
        for (let number = 0; number < data.rounds; number++) {
          set("round", number, "input");
          set("scrub", 1000, "input");
          hud(data.rollouts[number].trace, `collection ${number + 1}`);
          if ($("roundLabel").textContent !== `${number + 1} (policy ${number})`) failures.push("Incorrect collection/policy label");
        }
      } else if (!$("rolloutOption").disabled) failures.push("Unseen circuit offers training footage");
      set("mode", "unsafe", "change");
      set("scrub", 1000, "input");
      hud(data.unsafe, "unsafe demo");
      if (!$("contact").textContent.includes("WALL CONTACT") || !$("source").textContent.includes("not counted"))
        failures.push("Scripted collision not explicitly distinguished from ML");
      if (!$("round").disabled) failures.push("Scripted demo offers a misleading model round");
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let red = 0;
      for (let i = 0; i < pixels.length; i += 4)
        if (pixels[i] > 180 && pixels[i] > 1.4 * pixels[i + 1]) red++;
      if (red < 100) failures.push("Wall/contact canvas is blank");
      for (const id of ["learning", "telemetry"]) {
        const cv = $(id), c = cv.getContext("2d"), image = c.getImageData(0, 0, cv.width, cv.height).data;
        let painted = 0;
        for (let i = 3; i < image.length; i += 4) if (image[i]) painted++;
        if (painted < 1000 || c.isContextLost()) failures.push(`Blank ${id} canvas`);
      }
      if ($("attemptTable").querySelectorAll("tbody tr").length !== 484) failures.push("Attempt table incomplete");
      return { failures, attempts: data.attempts.length * 4, rollouts: data.rollouts.length };
    });
    assert.deepEqual(result.failures, [], circuit.name);
    attempts += result.attempts;
    rollouts += result.rollouts;
    wallDemos++;
    await page.screenshot({ path: path.join(output, `visuals/track_${String(circuit.id).padStart(2, "0")}_wall_contact.png`), fullPage: true });
    await page.locator("#mode").selectOption("single");
    await page.locator("#round").fill(String(manifest.selected_round));
    await page.locator("#scenario").selectOption("3");
    await page.locator("#scrub").fill("150");
    await page.screenshot({ path: path.join(output, `visuals/track_${String(circuit.id).padStart(2, "0")}_corner_cockpit.png`), fullPage: true });
    const beforeCamera = await page.locator("#view").evaluate(canvas => canvas.toDataURL());
    await page.locator("#camera").selectOption("overview");
    assert.notEqual(await page.locator("#view").evaluate(canvas => canvas.toDataURL()), beforeCamera);
    await page.locator("#camera").selectOption("follow");
    await page.locator("#sensors").uncheck();
    await page.locator("#sensors").check();
    await page.locator("#ghost").check();
    await page.locator("#ghost").uncheck();
    await page.locator("#mode").selectOption("compare");
    await page.locator("#scrub").fill("500");
    if (circuit.id === 0) {
      await page.locator("#mode").selectOption("single");
      await page.locator("#speed").selectOption("1");
      await page.locator("#restart").click();
      const before = await page.evaluate(() => ({ x: sample(active().trace, time).state[1], y: sample(active().trace, time).state[2], image: canvas.toDataURL() }));
      await page.waitForTimeout(350);
      const after = await page.evaluate(() => ({ x: sample(active().trace, time).state[1], y: sample(active().trace, time).state[2], image: canvas.toDataURL() }));
      assert.ok(Math.hypot(after.x - before.x, after.y - before.y) > .1);
      assert.notEqual(after.image, before.image);
      await page.locator("#play").click();
      await page.locator("#mode").selectOption("sequence");
      await page.locator("#round").fill("0");
      await page.locator("#scrub").fill("1000");
      await page.locator("#play").click();
      await page.waitForFunction(() => +document.getElementById("round").value >= 1, null, { timeout: 10000 });
      await page.locator("#play").click();
      const number = Number(await page.locator("#round").inputValue());
      await page.locator("#next").click();
      assert.equal(Number(await page.locator("#round").inputValue()), number + 1);
    }
    console.log(`PASS ${circuit.name}: 484 corner/recovery tests, ${result.rollouts} collections, wall-contact demo`);
  }
  await page.goto(pathToFileURL(path.join(output, "index.html")).href);
  assert.equal(await page.locator("#trackSelect option").count(), 13);
  assert.equal(await page.locator(".gallery article").count(), 13);
  await page.locator("#trackSelect").selectOption("12");
  await page.frameLocator("#cockpit").locator("#title").filter({ hasText: catalog[12].name }).waitFor();
  await page.goto(pathToFileURL(path.join(output, "training_deck.html")).href);
  assert.equal(await page.locator(".slide").count(), 6);
  for (let slide = 1; slide <= 6; slide++) {
    assert.equal(await page.locator("#count").textContent(), `${slide} / 6`);
    assert.ok(await page.locator(".slide.active img").evaluate(image => image.complete && image.naturalWidth > 0));
    await page.screenshot({ path: path.join(output, `visuals/deck_${String(slide).padStart(2, "0")}.png`), fullPage: true });
    if (slide < 6) await page.keyboard.press("ArrowRight");
  }
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(output, "browser_verification.json"), JSON.stringify({
    passed: true, attempts, rollouts, wall_demos: wallDemos, slides: 6,
    movement: true, camera: true, sensors: true, sequence: true, console_errors: errors,
  }, null, 2) + "\n");
} finally {
  await browser.close();
}
