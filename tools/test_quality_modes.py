"""Quality modes on cards: a reproduction is judged in its result's mode, and the cards show each row's mode.

    python -B -m unittest discover -s tools

Each test works on a copy of a fixture engine folder (tools/fixtures/engines/crit-labs/, two engines crit-hub listed before
the plugin architecture): mtp-adaptive-width's result was judged in preserve, mtp4-nvfp4-mma's in speed, and each is reproduced here from its own result file judged in the other mode.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HUB = Path(__file__).resolve().parents[1]
FIXTURES = HUB / "tools" / "fixtures" / "engines" / "crit-labs"
sys.path.insert(0, str(HUB / "tools"))
import add_result  # noqa: E402
import card as cardlib  # noqa: E402
import check_engines  # noqa: E402
import export_engine  # noqa: E402

# The engines and the mode their result was judged in.
ENGINES = {"mtp-adaptive-width": "preserve", "mtp4-nvfp4-mma": "speed"}
OTHER = {"preserve": "speed", "speed": "preserve"}
BY = "someone-else"


class ReproductionModeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def engine(self, name):
        """A copy of the engine's folder at engines/crit-labs/<name> under a temporary root."""
        folder = self.root / "engines" / "crit-labs" / name
        shutil.copytree(FIXTURES / name, folder)
        return folder

    def result_file(self, folder, mode):
        """The engine's own result file as one judged in `mode`, outside its folder."""
        data = json.loads((folder / "evidence" / "crit-eval.json").read_text(encoding="utf-8"))
        data["requirements"]["quality_mode"] = mode
        path = self.root / f"result-{mode}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, data

    def test_the_cards_state_the_modes_these_tests_rely_on(self):
        for name, mode in ENGINES.items():
            self.assertEqual({cardlib.quality_mode(r) for r in cardlib.load(FIXTURES / name)["results"]},
                             {mode})

    def test_add_result_reproduces_a_result_only_in_its_mode(self):
        for name, mode in ENGINES.items():
            with self.subTest(engine=name):
                folder = self.engine(name)
                # The result files name the checkpoint's revision but not its repository: the model is named.
                model = cardlib.load(folder)["results"][0]["model"]
                other, _ = self.result_file(folder, OTHER[mode])
                with self.assertRaisesRegex(SystemExit, f"judged in quality mode {mode}, these result files in "
                                                        f"{OTHER[mode]}: a reproduction measures the result again"):
                    add_result.add(folder, [other], BY, model=model)
                same, _ = self.result_file(folder, mode)
                found = add_result.add(folder, [same], BY, model=model)
                self.assertEqual((found["added"], found["problems"]), ("reproduction", []))
                readme = (folder / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"reproduced by {BY} | {mode} |", readme)
                self.assertIn(f"in quality mode {mode}, crit Eval", readme)

    def test_the_check_refuses_a_reproduction_judged_in_another_mode(self):
        for name, mode in ENGINES.items():
            with self.subTest(engine=name):
                folder = self.engine(name)
                path, data = self.result_file(folder, OTHER[mode])
                card, lock = cardlib.load(folder), cardlib.load_lock(folder)
                entry = export_engine.result_entry([data])
                entry.pop("track")
                if not entry.get("more"):
                    entry.pop("more", None)
                shutil.copyfile(path, folder / "evidence" / "crit-eval-2.json")
                record = {"by": BY, **entry, "evidence": [
                    {"file": "evidence/crit-eval-2.json", "sha256": export_engine.digest(folder / "evidence/crit-eval-2.json")}]}
                self.assertEqual(record["quality_mode"], OTHER[mode])
                card = copy.deepcopy(card)
                card["results"][0]["reproductions"] = [record]
                cardlib.write(folder, card, lock)
                cardlib.main([folder], False)
                problems = check_engines.check(folder)
                self.assertIn(f"judged in quality mode {OTHER[mode]}, its result in {mode}: a reproduction measures "
                              "the result again in the result's mode", "\n".join(problems))


class StatsTableTests(unittest.TestCase):
    def test_every_row_names_the_quality_mode_its_figures_were_judged_in(self):
        for name, mode in ENGINES.items():
            with self.subTest(engine=name):
                readme = (FIXTURES / name / "README.md").read_text(encoding="utf-8")
                header = next(line for line in readme.splitlines() if line.startswith("| ") and "Decode" in line)
                self.assertTrue(header.startswith("| Quality mode | Decode (sampled) |"), header)
                row = readme.splitlines()[readme.splitlines().index(header) + 2]
                self.assertTrue(row.startswith(f"| {mode} | **"), row)


if __name__ == "__main__":
    unittest.main()
