"""2-4-1 ReLU regression. Scoring, storage, and argument parsing stay on the native path."""

from __future__ import annotations

import unittest
import uuid

from tools import tensor_engine
from tools.agents.native_storage_tools import NativeStorageTools
from tools.core.host import argument_object


def _column(scale: float) -> list[list[float]]:
    return [[scale], [scale], [scale], [scale]]


class RegressionPipelineTest(unittest.TestCase):
    def test_relu_forward_and_mse_are_finite(self) -> None:
        model = tensor_engine.RegressionMLPModel(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0, 0.0],
            [[1.0], [0.0], [0.0], [0.0]],
            [0.0],
        )
        self.assertEqual(model.forward([[-2.0, 0.5]])[0][0], 0.0)
        self.assertEqual(model.forward([[2.0, 0.5]])[0][0], 2.0)
        score = model.evaluate_mse(50)
        self.assertGreater(score["loss"], 0.0)
        self.assertGreaterEqual(score["max_residual"], 0.0)
        self.assertGreaterEqual(score["std_residual"], 0.0)

    def test_alternating_matrix_stays_a_totr_token(self) -> None:
        encoded = tensor_engine.TotrEngine.encode_matrix(
            [[1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]], 1
        )
        decoded = tensor_engine.TotrEngine.decode_matrix(encoded, 2, 4)
        self.assertAlmostEqual(decoded[0][0], 1.0, delta=0.05)
        self.assertAlmostEqual(decoded[0][1], -1.0, delta=0.05)

    def test_string_arguments_and_sequential_journal(self) -> None:
        blob = f"reg_{uuid.uuid4().hex[:8]}.json"
        journal_name = f"reg_{uuid.uuid4().hex[:8]}.json"
        tools = NativeStorageTools()
        zeros = tensor_engine.TotrEngine.encode_matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 1)
        layer2 = tensor_engine.TotrEngine.encode_matrix(_column(0.0), 1)
        biased = tensor_engine.RegressionMLPModel(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0, 0.0],
            _column(0.0),
            [1.0],
        ).evaluate_mse(50)
        zero_score = tensor_engine.RegressionMLPModel(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0, 0.0],
            _column(0.0),
            [0.0],
        ).evaluate_mse(50)
        worse_bias, better_bias = ("0", "1") if zero_score["loss"] > biased["loss"] else ("1", "0")

        first = tools.commit_winner_native(
            {
                "l1_w": zeros,
                "l2_w": layer2,
                "l1_b": "0 0 0 0",
                "l2_b": worse_bias,
                "target_blob": blob,
                "journal_blob": journal_name,
            }
        )
        self.assertEqual(first["status"], "committed_natively")
        self.assertNotIn("weights", first)

        state = tools.read_totr_state_native(
            '{"checkpoint_blob":"%s","journal_blob":"%s"}' % (blob, journal_name)
        )
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["l1_w"].startswith("r:1|g:"))
        self.assertNotIn("weights", state)

        broken = argument_object(
            '{"checkpoint_blob","%s","journal_blob","%s"}' % (blob, journal_name)
        )
        self.assertEqual(broken["checkpoint_blob"], blob)
        self.assertEqual(argument_object("not-an-object"), {})
        scored = tensor_engine.score_totr_candidate(zeros, layer2, "0 0 0 0", worse_bias)
        self.assertEqual(scored["status"], "scored")
        self.assertAlmostEqual(scored["loss"], first["loss"], places=6)
        self.assertNotIn("weights", scored)

        second = tools.commit_winner_native(
            {
                "l1_w": zeros,
                "l2_w": layer2,
                "l1_b": "0 0 0 0",
                "l2_b": better_bias,
                "target_blob": blob,
                "journal_blob": journal_name,
            }
        )
        self.assertEqual(second["status"], "committed_natively")
        self.assertLess(second["loss"], first["loss"])

        rejected = tools.commit_winner_native(
            {
                "l1_w": zeros,
                "l2_w": layer2,
                "l1_b": "0 0 0 0",
                "l2_b": worse_bias,
                "target_blob": blob,
                "journal_blob": journal_name,
            }
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["checksum"], second["checksum"])

        journal = tensor_engine.SlidingJournal(5)
        for index in range(6):
            journal.add_entry(float(index), 0.1, 0.01, f"chk_{index}")
        restored = tensor_engine.SlidingJournal(5)
        restored.deserialize(journal.serialize())
        self.assertEqual(restored.size(), 5)
        self.assertNotIn("chk_0", journal.serialize())
        self.assertIn("chk_5", journal.serialize())


if __name__ == "__main__":
    unittest.main()
