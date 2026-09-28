import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load():
    for name in (
        "vlmeval",
        "vlmeval.smp",
        "vlmeval.utils",
        "vlmeval.dataset",
        "vlmeval.dataset.utils",
    ):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        sys.modules[name] = mod
    sys.modules["vlmeval.utils"].can_infer = lambda answer, choices: None

    path = (
        Path(__file__).resolve().parents[1]
        / "eval_mm"
        / "vlmevalkit"
        / "vlmeval"
        / "dataset"
        / "utils"
        / "mathvista.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vlmeval.dataset.utils.mathvista", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


class MathVistaTextTest(unittest.TestCase):
    def test_a_text_answer_matches_the_reply(self):
        line = {
            "question_type": "free_form",
            "answer_type": "text",
            "answer": "yes",
            "res": "yes",
            "prediction": "yes",
        }
        self.assertTrue(mod.post_check(line, prefetch=False))

    def test_an_integer_answer_still_matches(self):
        line = {
            "question_type": "free_form",
            "answer_type": "integer",
            "answer": "14",
            "res": "14",
            "prediction": "14",
        }
        self.assertTrue(mod.post_check(line, prefetch=False))


if __name__ == "__main__":
    unittest.main()
