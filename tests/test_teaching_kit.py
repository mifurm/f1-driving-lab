"""Static cross-reference checks for the student lab: teaching kit, page and scripts must agree.

These need no generated artifacts, so they run in CI. They catch the failures a student would hit first: a quiz
item pointing at a missing objective, a worksheet whose runner does not exist, a tab with no lesson, or a script
looking up a page element that is not there.
"""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
KIT = json.loads((ROOT / "teaching_kit.json").read_text())
HTML = (ROOT / "student_lab.html").read_text()
SCRIPTS = {name: (ROOT / name).read_text() for name in ("student_app.js", "student_labs.js", "student_journey_ui.js")}
DYNAMIC_IDS = {"wsFactor", "wsValues", "wsSeeds", "wsPool", "wsParams", "wsCircuit", "wsReward"}  # worksheet controls built at run time


class TeachingKitTests(unittest.TestCase):
    def test_every_objective_is_taught_by_a_module_and_checked_by_a_quiz_item(self):
        objectives = {o["id"] for o in KIT["objectives"]}
        self.assertEqual(len(objectives), len(KIT["objectives"]), "duplicate objective id")
        taught = {o for module in KIT["modules"] for o in module["objectives"]}
        checked = {item["objective"] for item in KIT["quiz"]}
        self.assertEqual(objectives - taught, set(), "objectives no module teaches")
        self.assertEqual(objectives - checked, set(), "objectives no quiz item checks")
        self.assertLessEqual(taught | checked, objectives)

    def test_quiz_items_have_one_first_correct_choice_and_explained_distractors(self):
        misconceptions = {m["id"] for m in KIT["misconceptions"]}
        ids = [item["id"] for item in KIT["quiz"]]
        self.assertEqual(len(ids), len(set(ids)))
        for item in KIT["quiz"]:
            choices = item["choices"]
            self.assertGreaterEqual(len(choices), 3, item["id"])
            self.assertEqual([c.get("correct", False) for c in choices].count(True), 1, item["id"])
            self.assertTrue(choices[0].get("correct"), f"{item['id']}: the page shuffles, so the stored correct choice must be first")
            for choice in choices:
                self.assertTrue(choice.get("why"), f"{item['id']}: every choice needs an explanation")
                if choice.get("misconception"):
                    self.assertIn(choice["misconception"], misconceptions, f"{item['id']}: unknown misconception")
            mapped = [c for c in choices if not c.get("correct") and c.get("misconception")]
            self.assertTrue(mapped, f"{item['id']}: at least one wrong answer must point to a catalogued misconception")

    def test_every_misconception_is_addressed_and_has_a_remedy(self):
        used = {m for module in KIT["modules"] for m in module["misconceptions"]}
        used |= {c["misconception"] for item in KIT["quiz"] for c in item["choices"] if c.get("misconception")}
        for misconception in KIT["misconceptions"]:
            self.assertTrue(misconception["remedy"], misconception["id"])
            self.assertIn(misconception["id"], used, f"{misconception['id']} is catalogued but never taught or tested")

    def test_modules_match_the_page_tabs_in_order(self):
        tabs = re.findall(r'<button id="tab\w+" data-module="(\w+)"', HTML)
        self.assertEqual([m["id"] for m in KIT["modules"]], tabs)
        for tab in tabs:
            self.assertIn(f'<section id="{tab}"', HTML)
        objectives = {o["id"] for o in KIT["objectives"]}
        for module in KIT["modules"]:
            self.assertGreater(module["minutes"], 0)
            self.assertLessEqual(set(module["objectives"]), objectives)

    def test_worksheets_reference_real_modules_objectives_and_runners(self):
        modules = {m["id"] for m in KIT["modules"]}
        objectives = {o["id"] for o in KIT["objectives"]}
        registry = SCRIPTS["student_labs.js"].split("const wsRunners = {", 1)[1].split("\n};", 1)[0]
        runners = set(re.findall(r"^  (?:async )?(\w+)\(", registry, re.M))
        self.assertTrue(runners, "could not find the worksheet runner registry")
        for worksheet in KIT["worksheets"]:
            self.assertIn(worksheet["module"], modules, worksheet["id"])
            self.assertLessEqual(set(worksheet["objectives"]), objectives, worksheet["id"])
            self.assertIn(worksheet["runner"], runners, worksheet["id"])
        self.assertEqual({w["module"] for w in KIT["worksheets"]}, modules - {"worksheets"}, "every lesson except the worksheet module itself needs a worksheet")

    def test_rubric_has_a_descriptor_for_every_criterion_at_every_level(self):
        levels = KIT["rubric"]["levels"]
        self.assertGreaterEqual(len(levels), 3)
        for criterion in KIT["rubric"]["criteria"]:
            self.assertEqual(len(criterion["descriptors"]), len(levels), criterion["name"])
            self.assertTrue(all(criterion["descriptors"]))


class PageScriptTests(unittest.TestCase):
    def test_every_element_the_scripts_look_up_exists_in_the_page(self):
        page_ids = set(re.findall(r'\bid="([^"]+)"', HTML))
        missing = {}
        for name, source in SCRIPTS.items():
            for element_id in re.findall(r'(?:\$|document\.getElementById)\("([A-Za-z][\w-]*)"\)', source):
                if element_id not in page_ids and element_id not in DYNAMIC_IDS:
                    missing.setdefault(element_id, set()).add(name)
        self.assertEqual(missing, {})

    def test_page_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', HTML)
        self.assertEqual(sorted({i for i in ids if ids.count(i) > 1}), [])


if __name__ == "__main__":
    unittest.main()
