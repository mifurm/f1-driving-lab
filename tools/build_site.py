"""Stage the F1 Driving Lab as one static website, ready to upload to Azure Static Web Apps.

The labs are plain files that run in the browser, and every lab page links to what it needs: scripts, images, downloads,
the other labs and the result notes. The site is built by following those links from each lab's entry pages and copying
only what they reach. Models, training data, test screenshots, zip bundles and the template pages in the repository root
are never published. Files keep their place in the repository (student_v2_artifacts/index.html stays at that path), so
the links between labs keep working unchanged.

    python tools/build_site.py                          # every lab, about 410 MB: needs the Standard plan
    python tools/build_site.py --labs v2 --sku Free     # the student lab only, about 20 MB: fits the Free plan

Besides the copied files the script writes a landing page (index.html), a 404 page and staticwebapp.config.json (security
headers, MIME types for the notes and the PowerPoint decks). A manifest with the size and SHA-256 of every file goes to
site-manifest.json, so a deployment can be checked byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]

# Per-environment limits from https://learn.microsoft.com/azure/static-web-apps/quotas (the check uses decimal MB).
SIZE_LIMIT = {"Free": 250_000_000, "Standard": 500_000_000}
FILE_LIMIT = 15_000

GENERATED = ("index.html", "404.html", "staticwebapp.config.json")
LINK = re.compile(r"""(?:href|src|poster)\s*=\s*["']([^"'<>{}$`]+)["']""", re.IGNORECASE)
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)  # http:, data:, mailto:, javascript:, //host
PRIVATE_SUFFIXES = {".env", ".pem", ".key", ".pfx", ".npz", ".pkl"}
PRIVATE_DIRECTORIES = {"node_modules", "__pycache__", "classic_ml_samples"}


class SiteError(Exception):
    """The site cannot be published as asked."""


@dataclass(frozen=True)
class Figure:
    path: str
    alt: str
    caption: str


@dataclass(frozen=True)
class Lab:
    key: str
    title: str
    summary: str
    entry: str  # opened by the card's button
    links: tuple[tuple[str, str], ...] = ()  # more pages and downloads listed on the card
    figure: Figure | None = None  # shown under the first card on the landing page


LABS = (
    Lab(
        "v2",
        "Student ML Driving Lab",
        "The main classroom lab. Twelve hands-on modules take a student from labelling driving examples to training a "
        "network, finding where it fails and fixing it. Module 12 shows a whole lap from zero: an imitation learner and "
        "a reinforcement learner each go from a car that crashes in its first seconds to one that laps the circuit.",
        "student_v2_artifacts/index.html",
        (
            ("Teacher guide", "student_v2_artifacts/teacher_guide.html"),
            ("Teacher guide (PDF)", "student_v2_artifacts/teacher_guide.pdf"),
            ("Class dashboard", "student_v2_artifacts/class_dashboard.html"),
            ("Classroom deck", "student_v2_artifacts/training_deck.html"),
            ("Deck (PowerPoint)", "student_v2_artifacts/training_deck.pptx"),
            ("Learning movie (GIF)", "student_v2_artifacts/visuals/journey_learning.gif"),
            ("Measured results", "STUDENT_V2_RESULTS.md"),
        ),
        Figure(
            "student_v2_artifacts/visuals/journey_filmstrip.png",
            "Two rows of ten snapshots of one lap attempt on the same circuit. In the top row the imitation learner "
            "crashes at the start until its fourth round of expert corrections, after which every snapshot completes "
            "the lap. In the bottom row the reinforcement learner crashes until iteration 16, after which every "
            "snapshot completes the lap.",
            "Module 12 on one circuit: how a car goes from zero to a lap. The imitation learner (top) stays on the "
            "lap from 750 labelled steps; the reinforcement learner (bottom) needs 32,768 simulated steps.",
        ),
    ),
    Lab(
        "v1",
        "First student lab",
        "The first student lab, kept for reference: eight modules on collecting data by hand, training and gradients in "
        "the browser, controller and feature experiments, new test circuits and leakage lessons.",
        "student_artifacts/index.html",
        (
            ("Teacher guide", "student_artifacts/teacher_guide.html"),
            ("Classroom deck", "student_artifacts/training_deck.html"),
            ("Measured experiments", "STUDENT_RESULTS.md"),
        ),
    ),
    Lab(
        "wall",
        "Corner and wall learning",
        "Close-up simulations of every circuit with oriented-car wall contact, near-wall recovery, 120 further "
        "training rounds and a safety-selected checkpoint.",
        "wall_artifacts/index.html",
        (("Training deck", "wall_artifacts/training_deck.html"), ("Measured results", "WALL_RESULTS.md")),
    ),
    Lab(
        "race",
        "GP-X driver lab",
        "Moving cars on harder circuits with full telemetry (speed, braking, throttle, gears and engine speed) and a "
        "replay of the actual training episodes.",
        "race_artifacts/index.html",
        (("Training deck", "race_artifacts/training_deck.html"), ("Measured results", "RACE_RESULTS.md")),
    ),
    Lab(
        "multitrack",
        "Multi-track learning",
        "Eleven training circuits, two circuits the driver has never seen and 120 more training rounds, with animations.",
        "multitrack_artifacts/replay.html",
        (("Training deck", "multitrack_artifacts/training_deck.html"), ("Measured results", "MULTITRACK_RESULTS.md")),
    ),
    Lab(
        "original",
        "Original 120-round replay",
        "The first experiment: a small neural network learns to drive a fictional car around one circuit. Pick any of "
        "the 120 training rounds and replay it.",
        "artifacts/replay.html",
        (
            ("Training deck", "artifacts/training_deck.html"),
            ("Deck (PowerPoint)", "artifacts/training_deck.pptx"),
            ("Improvement plan", "IMPROVEMENT_PLAN.md"),
        ),
    ),
)

HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "SAMEORIGIN",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    # The teacher guide carries the answer key; keep the site out of search results (this is a hint, not access control).
    "X-Robots-Tag": "noindex, nofollow",
}
MIME_TYPES = {
    ".md": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def megabytes(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{max(size, 1) / 1e3:.0f} KB"


def collect(root: Path, starts: Iterable[str]) -> tuple[set[Path], list[str]]:
    """Every file reachable from the start pages by relative links, and a description of each link that cannot be served."""
    root = root.resolve()
    found: set[Path] = set()
    problems: list[str] = []
    listings: dict[Path, set[str]] = {}

    def listing(directory: Path) -> set[str]:
        if directory not in listings:
            listings[directory] = {entry.name for entry in directory.iterdir()} if directory.is_dir() else set()
        return listings[directory]

    def exists_with_exact_case(path: Path) -> bool:
        # Azure serves from a case-sensitive file system; macOS and Windows hide a wrongly cased link until it is live.
        current = root
        for part in path.relative_to(root).parts:
            if part not in listing(current):
                return False
            current = current / part
        return path.is_file()

    def visit(source: Path | None, target: str) -> None:
        origin = f"{source.relative_to(root).as_posix()}" if source else "build_site.py"
        base = root if source is None or target.startswith("/") else source.parent
        path = (base / unquote(target).lstrip("/")).resolve()
        if root not in path.parents and path != root:
            problems.append(f"{origin}: {target!r} points outside the repository")
            return
        if path.is_dir():
            path = path / "index.html"
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in PRIVATE_DIRECTORIES for part in relative.parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
            problems.append(f"{origin}: {target!r} is a private or development file and is never published")
        elif not exists_with_exact_case(path):
            problems.append(f"{origin}: {target!r} does not exist (names are case-sensitive on Azure)")
        elif path not in found:
            found.add(path)
            if path.suffix.lower() in {".html", ".htm"}:
                for raw in LINK.findall(path.read_text(encoding="utf-8", errors="ignore")):
                    link = raw.strip().split("#", 1)[0].split("?", 1)[0]
                    if link and not EXTERNAL.match(link):
                        visit(path, link)

    for start in starts:
        visit(None, start)
    return found, problems


def available(labs: Sequence[Lab], spec: str, root: Path) -> list[Lab]:
    """The labs asked for. 'all' skips labs that have not been generated; naming a lab that is missing is an error."""
    if spec == "all":
        present = [lab for lab in labs if (root / lab.entry).is_file()]
        for lab in labs:
            if lab not in present:
                print(f"warning: skipping {lab.key}: {lab.entry} has not been generated", file=sys.stderr)
    else:
        keys = [key.strip() for key in spec.split(",") if key.strip()]
        unknown = sorted(set(keys) - {lab.key for lab in labs})
        if unknown:
            raise SiteError(f"unknown lab {', '.join(unknown)}; choose from {', '.join(lab.key for lab in labs)} or 'all'")
        present = [lab for lab in labs if lab.key in keys]
        missing = [lab.entry for lab in present if not (root / lab.entry).is_file()]
        if missing:
            raise SiteError(f"not generated yet: {', '.join(missing)} (see 'Reproduce training and visuals' in README.md)")
    if not present:
        raise SiteError("no lab has been generated yet; see 'Reproduce training and visuals' in README.md")
    return present


def link(path: str) -> str:
    return html.escape(quote(path), quote=True)


def card(lab: Lab, size: int, primary: bool) -> str:
    extras = "".join(f'<li><a href="{link(path)}">{html.escape(label)}</a></li>' for label, path in lab.links)
    return (
        f'<article class="card{" primary" if primary else ""}">'
        f"<h3>{html.escape(lab.title)}</h3><p>{html.escape(lab.summary)}</p>"
        f'<p class="actions"><a class="button" href="{link(lab.entry)}">Open {html.escape(lab.title)}</a>'
        f'<span class="meta">pages up to {megabytes(size)}</span></p>'
        f'<ul class="links" aria-label="More from {html.escape(lab.title)}">{extras}</ul></article>'
    )


PAGE_STYLE = """:root{color-scheme:dark;--bg:#101722;--panel:#182233;--line:#334358;--text:#ecf2fa;--muted:#a9b8ca;--accent:#5eead4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header,main,footer{max-width:1040px;margin:0 auto;padding:0 24px}header{padding-top:48px}
.eyebrow{color:var(--accent);font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:13px;margin:0}
h1{font-size:clamp(34px,6vw,56px);line-height:1.05;margin:8px 0 16px}h2{font-size:22px;margin:40px 0 12px}
.lead{font-size:18px;color:var(--muted);max-width:68ch}a{color:#8ecbff}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px}.card h3{margin:0 0 6px;font-size:20px}
.card p{margin:0 0 14px;color:var(--muted)}.primary{border-color:var(--accent)}.primary h3{font-size:24px}
.button{display:inline-block;background:var(--accent);color:#06241f;font-weight:700;text-decoration:none;padding:10px 18px;border-radius:10px}
.button:hover,.button:focus-visible{outline:3px solid #fff;outline-offset:2px}.meta{font-size:14px;white-space:nowrap}
.actions{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px}
.links{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:6px 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;list-style:none;padding:0;margin:0}
figure{margin:24px 0 0}figure img{width:100%;height:auto;border-radius:12px;border:1px solid var(--line);background:#000}
figcaption{color:var(--muted);font-size:14px;margin-top:8px}footer{color:var(--muted);font-size:14px;padding-bottom:48px;margin-top:40px}"""


def page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="robots" content="noindex,nofollow">'
        f"<title>{html.escape(title)}</title><style>{PAGE_STYLE}</style></head><body>{body}</body></html>\n"
    )


def render_index(labs: Sequence[Lab], sizes: dict[str, int]) -> str:
    first, *rest = labs
    figure = ""
    if first.figure:
        figure = (
            f'<figure><img src="{link(first.figure.path)}" alt="{html.escape(first.figure.alt)}">'
            f"<figcaption>{html.escape(first.figure.caption)}</figcaption></figure>"
        )
    more = ""
    if rest:
        cards = "".join(f"<li>{card(lab, sizes[lab.key], False)}</li>" for lab in rest)
        more = f'<section aria-labelledby="more"><h2 id="more">More labs</h2><ul class="grid">{cards}</ul></section>'
    body = (
        '<header><p class="eyebrow">Teaching lab · runs in the browser</p><h1>F1 Driving Lab</h1>'
        '<p class="lead">A car that starts by driving off the track, and the machine learning that teaches it to lap a '
        "circuit. Students collect examples, train models, watch them fail, find out why and repair them.</p></header>"
        f'<main><section aria-labelledby="start"><h2 id="start">Start here</h2>{card(first, sizes[first.key], True)}{figure}</section>'
        f"{more}</main>"
        "<footer>Everything runs in the browser tab: the pages contain no analytics and the labs never upload student work. "
        "The teacher guide includes the answer key.</footer>"
    )
    return page("F1 Driving Lab: how a machine learns to drive", body)


def render_not_found() -> str:
    return page(
        "Page not found: F1 Driving Lab",
        '<header><h1>Page not found</h1><p class="lead">That address is not part of the F1 Driving Lab. '
        '<a href="/">Go to the start page</a>.</p></header>',
    )


def render_config() -> str:
    config = {
        "$schema": "https://json.schemastore.org/staticwebapp.config.json",
        "responseOverrides": {"404": {"rewrite": "/404.html"}},
        "globalHeaders": HEADERS,
        "mimeTypes": MIME_TYPES,
    }
    return json.dumps(config, indent=2) + "\n"


def prepare(root: Path, out: Path) -> None:
    out = out.resolve()
    if out == root or out in root.parents:
        raise SiteError(f"{out} contains the repository; choose another --out")
    if out.exists():
        if not all((out / name).is_file() for name in GENERATED):
            raise SiteError(f"{out} exists and is not a site made by this script; remove it or choose another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def build(root: Path, out: Path, manifest: Path, labs: Sequence[Lab], sku: str, limit: int | None = None) -> dict:
    """Stage the site for the given labs in 'out' and return the manifest that was written to 'manifest'."""
    if not labs:
        raise SiteError("no labs to publish")
    root = root.resolve()
    sources: set[Path] = set()
    problems: list[str] = []
    lab_files: dict[str, set[Path]] = {}
    for lab in labs:
        starts = [lab.entry, *(path for _, path in lab.links), *([lab.figure.path] if lab.figure else [])]
        reached, issues = collect(root, starts)
        lab_files[lab.key] = reached
        problems += issues
        sources |= reached
    if problems:
        raise SiteError("links that cannot be served:\n  " + "\n  ".join(dict.fromkeys(problems)))
    # What opening a lab costs: some entry pages are small launchers for pages of several MB, so measure the pages they lead to.
    sizes = {lab.key: max(path.stat().st_size for path in collect(root, [lab.entry])[0] if path.suffix.lower() in {".html", ".htm"}) for lab in labs}

    clashes = [name for name in GENERATED if (root / name).resolve() in sources]
    if clashes:
        raise SiteError(f"the site's own pages would overwrite repository files: {', '.join(clashes)}")
    total = sum(path.stat().st_size for path in sources)
    limit = SIZE_LIMIT[sku] if limit is None else limit
    if total > limit:
        raise SiteError(
            f"{megabytes(total)} of content is too big for the {sku} plan ({megabytes(limit)} per environment). "
            "Publish fewer labs (--labs v2) or use --sku Standard."
        )
    if len(sources) + len(GENERATED) > FILE_LIMIT:
        raise SiteError(f"{len(sources)} files exceed the limit of {FILE_LIMIT} per app")

    leaks = (str(root).encode(), root.as_posix().encode())
    digests: dict[Path, str] = {}
    leaking: list[str] = []
    for path in sorted(sources):
        data = path.read_bytes()
        digests[path] = hashlib.sha256(data).hexdigest()
        if any(needle in data for needle in leaks):
            leaking.append(path.relative_to(root).as_posix())
    if leaking:
        raise SiteError("these files contain this machine's repository path and would publish it:\n  " + "\n  ".join(leaking))

    prepare(root, out)
    for path in sorted(sources):
        target = out / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    files = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": digests[path]} for path in sources}
    pages = {"index.html": render_index(labs, sizes), "404.html": render_not_found(), "staticwebapp.config.json": render_config()}
    for name, text in pages.items():
        data = text.encode("utf-8")
        (out / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    files = dict(sorted(files.items()))
    summary = {
        "sku": sku,
        "labs": [lab.key for lab in labs],
        "totalBytes": sum(item["bytes"] for item in files.values()),
        "labBytes": {key: sum(path.stat().st_size for path in paths) for key, paths in lab_files.items()},
        "files": files,
    }
    manifest.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage the F1 Driving Lab as a static website for Azure Static Web Apps.")
    parser.add_argument("--labs", default="all", help="'all' (default) or a comma-separated list of: " + ", ".join(lab.key for lab in LABS))
    parser.add_argument("--sku", choices=sorted(SIZE_LIMIT), default="Standard", help="plan the site must fit (default Standard)")
    parser.add_argument("--out", type=Path, default=ROOT / "site", help="directory to stage into; replaced if it holds an earlier site")
    parser.add_argument("--manifest", type=Path, default=ROOT / "site-manifest.json")
    args = parser.parse_args(argv)
    try:
        labs = available(LABS, args.labs, ROOT)
        summary = build(ROOT, args.out, args.manifest, labs, args.sku)
    except SiteError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Staged {len(summary['files'])} files, {megabytes(summary['totalBytes'])}, in {args.out} for the {args.sku} plan.")
    print("Content reached from each lab's pages (labs share files, so the rows do not add up):")
    for lab in labs:
        print(f"  {lab.key:<11} {megabytes(summary['labBytes'][lab.key]):>9}  {lab.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
