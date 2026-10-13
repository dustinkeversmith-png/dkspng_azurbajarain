"""Slice completions use the model's candidate scheme. The loop does not write the prompt."""

from __future__ import annotations

import unittest

from orchestrators.tournament_evaluator import completion_context


class LoopPromptTest(unittest.TestCase):
    def test_context_reads_the_model_parameters(self) -> None:
        context = completion_context(
            {
                "architecture": "mlp_regressor",
                "in_dim": 4,
                "out_dim": 4,
                "parameters": [
                    {"key": "w", "role": "weights", "scheme": "totr_rank1", "token": "r:1|g:0.02|A:1|B:2"},
                    {"key": "b", "role": "bias", "scheme": "bias_delta", "token": "q:0.01|D:10"},
                ],
                "candidate_scheme": {"slots": ["c1", "c2", "c3"], "fields": [{"key": "w", "role": "weights"}]},
                "probe": {"key": "gradient", "token": "A:[+1] B:[0]"},
            }
        )
        self.assertEqual(context["architecture"], "mlp_regressor")
        self.assertEqual(context["rows"], 4)
        self.assertEqual(context["active"]["w"], "r:1|g:0.02|A:1|B:2")
        self.assertEqual(context["active"]["b"], "q:0.01|D:10")
        self.assertEqual(context["probe"]["key"], "gradient")
        self.assertEqual(context["scheme"]["slots"], ["c1", "c2", "c3"])
        self.assertNotIn("totr_w", context)
        self.assertNotIn("gradient_hint", context)


if __name__ == "__main__":
    unittest.main()
