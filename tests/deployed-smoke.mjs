// Smoke test for a deployed copy of the labs, in a real browser and over HTTP.
//   PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" node tests/deployed-smoke.mjs https://<name>.azurestaticapps.net
// It opens the start page, the student lab (running module 12 end to end), the teacher pages and the earlier labs. None may raise a console
// error, a page error or a failed request. It also checks the headers, file types and 404 page that infra's site configuration promises.
// Nothing is written to disk.
import { chromium } from "playwright";

const base = (process.argv[2] || "").replace(/\/+$/, "");
if (!/^https?:\/\/[^/]+$/.test(base)) { console.error("usage: node tests/deployed-smoke.mjs https://<host>"); process.exit(2); }

const failures = [];
let count = 0;
const check = (name, condition, detail = "") => {
  count++;
  console.log(`${condition ? "ok  " : "FAIL"} ${name}${condition || !detail ? "" : `: ${detail}`}`);
  if (!condition) failures.push(name);
};

const SECURITY_HEADERS = { "x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "x-frame-options": "SAMEORIGIN", "x-robots-tag": "noindex, nofollow" };
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1360, height: 900 } });

async function open(path, ready) {
  const page = await context.newPage();
  const problems = [];
  const isFavicon = url => /\/favicon\.ico$/.test(url);
  page.on("pageerror", error => problems.push(`page error: ${error.message}`));
  page.on("console", message => { if (message.type() === "error" && !isFavicon(message.location().url || "")) problems.push(`console: ${message.text()}`); });
  page.on("requestfailed", request => problems.push(`request failed: ${request.url()} (${request.failure()?.errorText})`));
  page.on("response", response => { if (response.status() >= 400 && !isFavicon(response.url())) problems.push(`HTTP ${response.status()}: ${response.url()}`); });
  const response = await page.goto(base + path, { waitUntil: "load", timeout: 240000 });
  if (ready) await ready(page);
  return { page, response, problems };
}

try {
  // ---- start page -------------------------------------------------------------------------------------------------------------------
  {
    const { page, response, problems } = await open("/");
    const headers = response.headers();
    check("start page is served as HTML", response.status() === 200 && headers["content-type"].startsWith("text/html"), `${response.status()} ${headers["content-type"]}`);
    check("start page carries the configured security headers", Object.entries(SECURITY_HEADERS).every(([name, value]) => headers[name] === value), JSON.stringify(headers));
    check("start page names the lab and offers the student lab first", (await page.locator("h1").innerText()) === "F1 Driving Lab"
      && (await page.locator("a.button").first().getAttribute("href")) === "student_v2_artifacts/index.html");
    check("start page shows the learning-journey filmstrip", await page.evaluate(() => document.images.length > 0 && [...document.images].every(image => image.complete && image.naturalWidth > 0)));
    const links = [...new Set(await page.$$eval("a[href]", anchors => anchors.map(anchor => anchor.href)))];
    const broken = [];
    for (const url of links) {
      let answer = await context.request.head(url);
      if (answer.status() === 405) answer = await context.request.get(url);
      if (answer.status() !== 200) broken.push(`${answer.status()} ${url}`);
    }
    check(`all ${links.length} links on the start page answer 200`, links.length >= 20 && broken.length === 0, broken.join("; "));
    check("start page raised no errors", problems.length === 0, problems.join("; "));
    await page.close();
  }

  // ---- the student lab, module 12 end to end ------------------------------------------------------------------------------------------
  {
    const { page, problems } = await open("/student_v2_artifacts/index.html", ready => ready.waitForFunction(() => !!window.studentLab, null, { timeout: 120000 }));
    check("student lab starts with its engine and journey library loaded", await page.evaluate(() => typeof StudentEngine === "object" && typeof StudentJourney === "object"));
    await page.locator('button[data-module="journey"]').click();
    await page.waitForFunction(() => studentLab.module === "journey");
    await page.locator("#journeySaved").click();
    await page.waitForFunction(() => !studentLab.busy, null, { timeout: 300000 });
    const saved = await page.evaluate(() => {
      const journey = studentLab.journey, both = ["imitation", "reinforcement"];
      return { stages: both.map(method => journey[method].stages.length), reproduced: both.every(method => journey[method].stages.every(stage => stage.reproduced)),
        finalLaps: both.every(method => journey[method].stages.at(-1).lap.completed), untrainedFails: both.every(method => !journey[method].stages[0].lap.completed), status: document.getElementById("journeyStatus").innerText };
    });
    check("module 12 re-drives all 42 stages of the saved example in the browser and matches the saved numbers", saved.stages.join() === "18,24" && saved.reproduced && saved.status.includes("matches the saved numbers"), JSON.stringify(saved));
    check("module 12: untrained cars fail and both final cars complete the lap", saved.untrainedFails && saved.finalLaps);
    await page.locator("#journeyReplay").click();
    await page.waitForFunction(() => studentLab.replayTime > 3, null, { timeout: 60000 });
    check("module 12 replays the two final cars and the expert side by side", await page.evaluate(() => studentLab.runs.map(run => run.name.split(" ")[0]).join()) === "Imitation,Reinforcement,Expert");
    check("student lab raised no errors", problems.length === 0, problems.join("; "));
    await page.close();
  }

  // ---- teacher pages and downloads ----------------------------------------------------------------------------------------------------
  for (const [path, heading] of [["/student_v2_artifacts/teacher_guide.html", "Teacher Guide"], ["/student_v2_artifacts/class_dashboard.html", "Class dashboard"], ["/student_v2_artifacts/training_deck.html", ""]]) {
    const { page, problems } = await open(path);
    check(`${path} loads without errors`, problems.length === 0 && (!heading || (await page.locator("h1").first().innerText()).startsWith(heading)), problems.join("; "));
    await page.close();
  }
  const pdf = await context.request.get(`${base}/student_v2_artifacts/teacher_guide.pdf`);
  check("teacher guide PDF is a PDF", pdf.status() === 200 && pdf.headers()["content-type"] === "application/pdf" && (await pdf.body()).subarray(0, 5).toString() === "%PDF-");

  // ---- file types and the 404 page that staticwebapp.config.json sets ------------------------------------------------------------------
  const types = { "/STUDENT_V2_RESULTS.md": "text/plain", "/artifacts/metrics.csv": "text/csv", "/student_v2_artifacts/training_deck.pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "/student_v2_artifacts/visuals/journey_learning.gif": "image/gif" };
  for (const [path, wanted] of Object.entries(types)) {
    const answer = await context.request.head(base + path);
    check(`${path} is served as ${wanted}`, answer.status() === 200 && (answer.headers()["content-type"] || "").startsWith(wanted), `${answer.status()} ${answer.headers()["content-type"]}`);
  }
  const missing = await context.request.get(`${base}/no-such-page.html`);
  check("an unknown address gets the friendly 404 page with a 404 status", missing.status() === 404 && (await missing.text()).includes("Page not found"), `${missing.status()}`);
  const compressed = await context.request.get(`${base}/student_v2_artifacts/index.html`, { headers: { "accept-encoding": "gzip, br" } });
  console.log(`info compression of the 11 MB lab page: content-encoding=${compressed.headers()["content-encoding"] || "none"}, ${(await compressed.body()).length} bytes after decoding`);

  // ---- the earlier labs --------------------------------------------------------------------------------------------------------------------
  const earlier = [["/artifacts/replay.html", "120-round replay"], ["/multitrack_artifacts/replay.html", "Multi-track learning"], ["/race_artifacts/index.html", "GP-X Driver Lab"],
    ["/wall_artifacts/index.html", "Learning corners and wall avoidance"], ["/student_artifacts/index.html", "Student ML Driving Laboratory"]];
  for (const [path, title] of earlier) {
    const { page, problems } = await open(path);
    check(`${path} loads without errors`, (await page.title()).includes(title) && problems.length === 0, `${await page.title()} ${problems.join("; ")}`);
    await page.close();
  }
} finally {
  await browser.close();
}

if (failures.length) { console.error(`\n${failures.length} of ${count} checks failed:\n  ${failures.join("\n  ")}`); process.exit(1); }
console.log(`\nAll ${count} checks passed against ${base}`);
