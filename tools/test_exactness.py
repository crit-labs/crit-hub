"""The exactness on cards: a result may state the levels its engine asked of its rows, the check holds them to crit's
scale, the README shows them beside the quality mode, and the exporter writes them from the candidate's package.

    python -B -m unittest discover -s tools

The card tests work on a copy of a fixture engine folder (tools/fixtures/engines/crit-labs/), whose result states no
level.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

HUB = Path(__file__).resolve().parents[1]
FIXTURES = HUB / "tools" / "fixtures" / "engines" / "crit-labs"
sys.path.insert(0, str(HUB / "tools"))
import card as cardlib  # noqa: E402
import check_engines  # noqa: E402
import export_engine  # noqa: E402

NAME = "mtp-adaptive-width"
ASKED = {"decode": "row-exact", "prompt": "exact"}


class CardExactnessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.folder = self.root / "engines" / "crit-labs" / NAME
        shutil.copytree(FIXTURES / NAME, self.folder)

    def with_exactness(self, value):
        """The copy's card with its result stating `value` as its exactness, written and rendered."""
        card, lock = cardlib.load(self.folder), cardlib.load_lock(self.folder)
        card["results"][0]["exactness"] = value
        cardlib.write(self.folder, card, lock)
        cardlib.main([self.folder], False)
        return check_engines.check(self.folder)

    def test_a_result_states_its_levels_and_the_readme_shows_them_beside_the_quality_mode(self):
        self.assertEqual(self.with_exactness(ASKED), [])
        readme = (self.folder / "README.md").read_text(encoding="utf-8")
        self.assertIn("| preserve (rows: decode row-exact, prompt exact) | **", readme)
        self.assertIn(f"; rows: decode row-exact, prompt exact ([exactness]({cardlib.EXACTNESS_URL}))", readme)
        self.assertIn("exactness:\n    decode: row-exact\n    prompt: exact\n",
                      (self.folder / "card.yaml").read_text(encoding="utf-8"))

    def test_a_malformed_exactness_is_refused_naming_the_field(self):
        for wrong in ("exact", {"prompt": "exact"}, {"decode": "exact", "prompt": "exact"},
                      {"decode": "row-exact", "prompt": "bitwise"}, {**ASKED, "verify": "exact"}, None):
            with self.subTest(wrong=wrong):
                problems = "\n".join(self.with_exactness(wrong))
                self.assertIn("exactness is {decode: LEVEL, prompt: LEVEL}", problems)

    def test_a_card_without_it_renders_as_before(self):
        before = {name: (self.folder / name).read_bytes() for name in ("README.md", "card.yaml")}
        cardlib.main([self.folder], False)
        self.assertEqual({name: (self.folder / name).read_bytes() for name in before}, before)
        self.assertEqual(check_engines.check(self.folder), [])
        self.assertIsNone(cardlib.rows_text(cardlib.load(self.folder)["results"][0]))


class ExportExactnessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def candidate(self, configuration):
        """A candidate folder whose package holds `configuration` (None: no configuration.json)."""
        folder = self.root / "candidate"
        (folder / "package").mkdir(parents=True, exist_ok=True)
        path = folder / "package" / "configuration.json"
        if configuration is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(configuration), encoding="utf-8")
        return folder

    def test_the_exporter_writes_the_levels_the_package_asks(self):
        found = export_engine.asked_exactness(self.candidate({"plugins": {}, "exactness": ASKED}), [])
        self.assertEqual(found, ASKED)
        # One bare level is written per phase, the decode rows' exact as row-exact.
        self.assertEqual(export_engine.asked_exactness(self.candidate({"exactness": "exact"}), []), ASKED)
        with self.assertRaisesRegex(ValueError, "exactness is"):
            export_engine.asked_exactness(self.candidate({"exactness": {"prompt": "exact"}}), [])
        # The result gets them right after its quality mode.
        args = SimpleNamespace(eval=[], gpu="rtx5090", model="qwen3.8-27b@NVFP4", candidate="candidate-0001")
        result = export_engine.measured(args, {"files": []}, self.candidate({"exactness": ASKED}), self.root / "entry")
        self.assertEqual(result["exactness"], ASKED)

    def test_without_a_level_in_the_package_it_writes_none_unless_eval_judged_by_one(self):
        self.assertIsNone(export_engine.asked_exactness(self.candidate({"plugins": {}}), []))
        self.assertIsNone(export_engine.asked_exactness(self.candidate(None), [{"quality": {"honesty": {}}}]))
        judged = {"quality": {"honesty": {"judged_by": {"exactness": ASKED}}}}
        self.assertEqual(export_engine.asked_exactness(self.candidate(None), [judged, judged]), ASKED)
        mode_only = {"quality": {"honesty": {"judged_by": {"quality_mode": "preserve"}}}}
        self.assertIsNone(export_engine.asked_exactness(self.candidate(None), [mode_only]))   # never from the mode
        args = SimpleNamespace(eval=[], gpu="rtx5090", model="qwen3.8-27b@NVFP4", candidate="candidate-0001")
        result = export_engine.measured(args, {"files": []}, self.candidate({"plugins": {}}), self.root / "entry")
        self.assertNotIn("exactness", result)


if __name__ == "__main__":
    unittest.main()
