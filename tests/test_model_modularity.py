"""A custom architecture lives in demo_scale_model.hpp and writes its prompt, checkpoint, and journal."""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from typing import Any

from tools import tensor_engine

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PROMPT = ARTIFACTS / "demo_scale_prompt.txt"
CHECKPOINT = ARTIFACTS / "demo_scale_checkpoint.json"
JOURNAL = ARTIFACTS / "demo_scale_journal.json"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


class ModelModularityTest(unittest.TestCase):
    def test_custom_model_compiles_and_saves_its_prompt(self) -> None:
        blob = f"demo_{uuid.uuid4().hex[:8]}.json"
        journal = f"demo_{uuid.uuid4().hex[:8]}.json"
        model = tensor_engine.DemoScaleModel()
        seeded = _plain(model.seed_to_storage(blob, journal, 5))
        self.assertEqual(seeded["architecture"], "demo_scale")
        self.assertEqual(seeded["origin"], "demo_scale")
        self.assertEqual(seeded["directives"]["model_type"], "DemoScaleModel")
        self.assertEqual(list(seeded["layers"]), ["mix", "head"])

        prompt = tensor_engine.compile_prompt(blob, journal, "mix", 3, 25, "")
        journal_text = tensor_engine.read_journal_text(journal)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        PROMPT.write_text(prompt, encoding="utf-8")
        CHECKPOINT.write_text(json.dumps(seeded, indent=2) + "\n", encoding="utf-8")
        JOURNAL.write_text(journal_text if journal_text.endswith("\n") else journal_text + "\n", encoding="utf-8")

        saved = PROMPT.read_text(encoding="utf-8")
        self.assertEqual(saved, prompt)
        self.assertIn("Model: demo_scale", saved)
        self.assertIn("Journal scheme: task scale.", saved)
        self.assertIn("origin=demo_scale", saved)
        self.assertNotIn("winner=seed", saved)
        self.assertIn("[scale_span]:", saved)
        self.assertIn("Encoding: span_decimal", saved)
        self.assertIn("span = highest weight - lowest weight.", saved)
        self.assertIn("[mix_sign]:", saved)
        self.assertIn("Encoding: mix_sign", saved)
        self.assertIn("+1 means hold can keep the mix.", saved)
        self.assertIn("Weight Scheme: totr_rank1", saved)
        self.assertIn("W[row][col] = (A[row] * gamma) * (B[col] * gamma).", saved)
        self.assertIn("scale_output", saved)
        self.assertNotIn("regression_output", saved)
        self.assertIn("Probe Data: mix:+1", saved)
        self.assertIn('"hold"', saved)
        self.assertIn('"shift"', saved)

        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["origin"], "demo_scale")
        self.assertEqual(checkpoint["status"], "seeded")
        journal_saved = JOURNAL.read_text(encoding="utf-8")
        self.assertIn('"task": "scale"', journal_saved)
        self.assertIn('"name": "origin"', journal_saved)
        self.assertIn("demo_scale", journal_saved)
        self.assertNotIn("winner", journal_saved)


if __name__ == "__main__":
    unittest.main()
