"""The Azure site stager publishes exactly what the labs' pages reach and refuses what must not go on a public website.

Everything here runs on a small fake repository in a temporary directory, so it needs none of the generated artifacts.
"""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from tools import build_site
from tools.build_site import GENERATED, Figure, Lab, SiteError, available, build

ALPHA = (
    '<script src="app.js"></script><img src="img/car.png"><a href="../notes.md">notes</a><a href="#top">top</a>'
    '<a href="page.html?x=1#y">page</a><a href="https://example.com/x">external</a><a href="data:text/plain,hi">data</a>'
    '<a href="sub/">directory</a>'
)
LABS = (
    Lab("alpha", "Alpha lab", "The first lab.", "alpha/index.html", (("Notes", "notes.md"),), Figure("alpha/img/car.png", "A car", "Caption")),
    Lab("beta", "Beta <lab>", "Fast & furious.", "beta/index.html"),
)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        self.found += [value for name, value in attrs if name in ("href", "src", "poster") and value]


def write(root, name, content="x"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


class SiteBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "repo"
        self.out = self.root / "site"
        self.manifest = self.base / "manifest.json"
        write(self.root, "alpha/index.html", ALPHA)
        write(self.root, "alpha/app.js", "console.log(1)")
        write(self.root, "alpha/img/car.png", b"\x89PNG")
        write(self.root, "alpha/page.html", '<a href="index.html">back</a>')
        write(self.root, "alpha/sub/index.html", "sub")
        write(self.root, "alpha/unlinked.bin", "never linked")
        write(self.root, "notes.md", "# notes")
        write(self.root, "models/net.npz", "weights")
        write(self.root, "beta/index.html", '<a href="../alpha/app.js">shared script</a>')

    def build(self, labs=LABS, sku="Standard", **options):
        return build(self.root, self.out, self.manifest, labs, sku, **options)

    def staged(self):
        return {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}

    def test_publishes_exactly_what_the_pages_reach_through_queries_fragments_and_directories(self):
        self.build()
        self.assertEqual(
            self.staged() - set(GENERATED),
            {"alpha/index.html", "alpha/app.js", "alpha/img/car.png", "alpha/page.html", "alpha/sub/index.html", "notes.md", "beta/index.html"},
        )
        self.assertTrue(set(GENERATED) <= self.staged())

    def test_every_link_in_the_staged_site_resolves_to_a_staged_file(self):
        self.build()
        checked = 0
        for page in self.out.rglob("*.html"):
            parser = Links()
            parser.feed(page.read_text())
            for value in parser.found:
                target = unquote(value.split("#")[0].split("?")[0])
                if not target or target.startswith(("http:", "https:", "data:")):
                    continue
                path = self.out / target.lstrip("/") if target.startswith("/") else page.parent / target
                path = path / "index.html" if path.is_dir() else path
                self.assertTrue(path.is_file(), f"{page.relative_to(self.out)} links to {value}")
                checked += 1
        self.assertGreater(checked, 8)

    def test_manifest_matches_the_staged_files(self):
        summary = self.build()
        self.assertEqual(json.loads(self.manifest.read_text()), summary)
        self.assertEqual(set(summary["files"]), self.staged())
        for name, entry in summary["files"].items():
            data = (self.out / name).read_bytes()
            self.assertEqual((entry["bytes"], entry["sha256"]), (len(data), hashlib.sha256(data).hexdigest()), name)
        self.assertEqual(summary["totalBytes"], sum(entry["bytes"] for entry in summary["files"].values()))

    def test_a_broken_link_is_reported_with_the_page_that_holds_it(self):
        write(self.root, "alpha/page.html", '<a href="gone.html">gone</a>')
        with self.assertRaisesRegex(SiteError, r"alpha/page\.html: 'gone\.html' does not exist"):
            self.build()

    def test_a_link_with_the_wrong_case_is_reported_because_azure_is_case_sensitive(self):
        write(self.root, "alpha/page.html", '<img src="IMG/car.png">')
        with self.assertRaisesRegex(SiteError, r"'IMG/car\.png' does not exist"):
            self.build()

    def test_a_link_leaving_the_repository_is_refused(self):
        write(self.base, "outside.txt", "secret")
        write(self.root, "alpha/page.html", '<a href="../../outside.txt">out</a>')
        with self.assertRaisesRegex(SiteError, "outside the repository"):
            self.build()

    def test_private_and_hidden_files_are_never_published(self):
        for target in (".env", "keys/server.pem", "models/net.npz", ".git/config", "node_modules/x/index.js"):
            with self.subTest(target=target):
                write(self.root, target, "private")
                write(self.root, "alpha/page.html", f'<a href="../{target}">x</a>')
                with self.assertRaisesRegex(SiteError, "private or development file"):
                    self.build()

    def test_a_file_that_names_this_machines_repository_path_is_refused_and_the_old_site_survives(self):
        self.build()
        write(self.root, "alpha/app.js", f'link("{self.root.as_uri()}/alpha/index.html")')
        with self.assertRaisesRegex(SiteError, r"alpha/app\.js"):
            self.build()
        self.assertIn("alpha/app.js", self.staged())
        self.assertNotIn(self.root.as_posix(), (self.out / "alpha/app.js").read_text())

    def test_content_over_the_plan_limit_or_the_file_limit_is_refused(self):
        with self.assertRaisesRegex(SiteError, "too big for the Free plan"):
            self.build(sku="Free", limit=10)
        with mock.patch.object(build_site, "FILE_LIMIT", 5), self.assertRaisesRegex(SiteError, "exceed the limit"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_a_rebuild_replaces_the_old_site_but_only_ever_deletes_a_site(self):
        self.build()
        write(self.out, "stale.txt", "left over")
        self.build()
        self.assertNotIn("stale.txt", self.staged())

        for refused in (self.root, self.root / "alpha", self.base):
            with self.subTest(out=refused):
                before = {p for p in self.root.rglob("*") if p.is_file()}
                with self.assertRaises(SiteError):
                    build(self.root, refused, self.manifest, LABS, "Standard")
                self.assertEqual({p for p in self.root.rglob("*") if p.is_file()}, before)

    def test_site_pages_may_not_overwrite_repository_files(self):
        write(self.root, "index.html", "mine")
        write(self.root, "beta/index.html", '<a href="../index.html">home</a>')
        with self.assertRaisesRegex(SiteError, "would overwrite"):
            self.build()

    def test_lab_selection_skips_missing_labs_only_when_asked_for_all(self):
        ghost = Lab("ghost", "Ghost", "Not generated.", "ghost/index.html")
        with contextlib.redirect_stderr(io.StringIO()) as warnings:
            self.assertEqual([lab.key for lab in available((*LABS, ghost), "all", self.root)], ["alpha", "beta"])
        self.assertIn("ghost", warnings.getvalue())
        self.assertEqual([lab.key for lab in available(LABS, "beta,alpha", self.root)], ["alpha", "beta"])
        with self.assertRaisesRegex(SiteError, "not generated yet"):
            available((*LABS, ghost), "ghost", self.root)
        with self.assertRaisesRegex(SiteError, "unknown lab nope"):
            available(LABS, "alpha,nope", self.root)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SiteError, "no lab has been generated"):
            available((ghost,), "all", self.root)

    def test_landing_page_lists_every_lab_escapes_their_text_and_leads_with_the_first(self):
        self.build()
        index = (self.out / "index.html").read_text()
        self.assertIn("Beta &lt;lab&gt;", index)
        self.assertIn("Fast &amp; furious.", index)
        self.assertNotIn("Beta <lab>", index)
        for target in ("alpha/index.html", "beta/index.html", "notes.md", "alpha/img/car.png"):
            self.assertIn(f'href="{target}"' if target.endswith(("html", "md")) else f'src="{target}"', index)
        self.assertLess(index.index("Alpha lab"), index.index("Beta &lt;lab&gt;"))
        self.assertIn('alt="A car"', index)

    def test_config_is_valid_json_that_sets_security_headers_and_serves_notes_as_text(self):
        self.build()
        config = json.loads((self.out / "staticwebapp.config.json").read_text())
        self.assertEqual(config["globalHeaders"]["X-Content-Type-Options"], "nosniff")
        self.assertEqual(config["globalHeaders"]["X-Robots-Tag"], "noindex, nofollow")
        self.assertTrue(config["mimeTypes"][".md"].startswith("text/plain"))
        self.assertEqual(config["responseOverrides"]["404"]["rewrite"], "/404.html")
        self.assertTrue((self.out / "404.html").is_file())


if __name__ == "__main__":
    unittest.main()
