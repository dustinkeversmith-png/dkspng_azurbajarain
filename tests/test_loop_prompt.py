"""The loop prompt and the model JSON parser stay free of storage."""

from __future__ import annotations

import unittest

from orchestrators.loop_optimizer import candidate_from_text, synthesis_prompt


class LoopPromptTest(unittest.TestCase):
    def test_prompt_carries_tokens_and_skips_a_formula(self) -> None:
        text = synthesis_prompt(
            {
                "loss": 0.5913,
                "max_residual": 1.7765,
                "l1_w": "r:1|g:0.25|A:3 -4|B:5 2 -3 1",
                "l2_w": "r:1|g:0.25|A:2 -1 1 3|B:4",
                "l1_b": "0.25 0.15 0.15 0.25",
                "l2_b": "0.15",
                "history": [
                    {"trial": 1, "loss": 4.0, "status": "rejected"},
                    {"trial": 2, "loss": 12.0, "status": "rejected_divergent"},
                    {"trial": 3, "loss": 2.0, "status": "rejected"},
                    {"trial": 4, "loss": 1.2, "status": "rejected"},
                ],
            }
        )
        self.assertIn("- T2: loss=12.0000, st=rejected_divergent", text)
        self.assertNotIn("- T1:", text)
        self.assertIn("r:1|g:0.25|A:3 -4|B:5 2 -3 1", text)
        self.assertIn("recovery", text)
        self.assertNotIn("sin", text.lower())
        self.assertLess(len(text.split()), 180)

    def test_parser_keeps_the_models_gamma(self) -> None:
        parsed = candidate_from_text(
            '{"l1_w":"r:1|g:0.18|A:10 -8|B:12 4 -6 3","l2_w":"r:1|g:0.18|A:4 -2 3 1|B:6",'
            '"l1_b":"0.2 0.2 0.2 0.2","l2_b":"0.1"}'
        )
        self.assertIn("g:0.18", parsed["l1_w"])
        self.assertIn("g:0.18", parsed["l2_w"])

    def test_parser_accepts_fenced_json(self) -> None:
        parsed = candidate_from_text(
            '```json\n{"l1_w":"r:1|g:0.2|A:1 -1|B:2 3 -4 5","l2_w":"r:1|g:0.2|A:1 2 3 4|B:5",'
            '"l1_b":"0.2 0.2 0.2 0.2","l2_b":"0.1"}\n```'
        )
        self.assertTrue(parsed["l1_w"].startswith("r:1|"))
        self.assertEqual(parsed["l2_b"], "0.1")

    def test_parser_rejects_a_missing_token(self) -> None:
        with self.assertRaises(ValueError):
            candidate_from_text('{"l1_w":"r:1|g:0.2|A:1 -1|B:1 1 1 1"}')

    def test_parser_fills_a_missing_bias_from_the_checkpoint(self) -> None:
        parsed = candidate_from_text(
            '{"l1_w":"r:1|g:0.2|A:1 -1|B:2 3 -4 5","l2_w":"r:1|g:0.2|A:1 2 3 4|B:5"}',
            fallback={"l1_b": "0.2 0.2 0.2 0.2", "l2_b": "0.1"},
        )
        self.assertEqual(parsed["l1_b"], "0.2 0.2 0.2 0.2")
        self.assertEqual(parsed["l2_b"], "0.1")


if __name__ == "__main__":
    unittest.main()
