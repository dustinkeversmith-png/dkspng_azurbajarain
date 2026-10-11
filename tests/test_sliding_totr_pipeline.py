"""Rank-1 TOTR tokens and the sliding journal stay in the compiled engine."""

from __future__ import annotations

import unittest
import uuid

from tools import tensor_engine


class TestSlidingTotrPipeline(unittest.TestCase):
    def test_totr_matrix_reconstruction(self):
        engine = tensor_engine
        original = [[0.05, -0.02, 0.08, 0.01], [0.03, -0.01, 0.04, 0.02]]
        encoded = engine.TotrEngine.encode_matrix(original, 1)
        self.assertTrue(encoded.startswith("r:1|g:"))
        self.assertIn("|A:", encoded)
        self.assertIn("|B:", encoded)

        decoded = engine.TotrEngine.decode_matrix(encoded, 2, 4)
        self.assertEqual(len(decoded), 2)
        self.assertEqual(len(decoded[0]), 4)
        self.assertAlmostEqual(decoded[0][2], 0.08, delta=0.03)

    def test_sliding_journal_ring_buffer(self):
        engine = tensor_engine
        journal = engine.SlidingJournal(3)
        journal.add_entry(1.15, 0.01, 0.005, "chk_1")
        journal.add_entry(0.85, 0.25, 0.050, "chk_2")
        journal.add_entry(0.60, 0.55, 0.080, "chk_3")
        journal.add_entry(0.35, 0.95, 0.120, "chk_4")

        self.assertEqual(journal.size(), 3)
        serialized = journal.serialize()
        self.assertNotIn("chk_1", serialized)
        self.assertIn("chk_4", serialized)

        restored = engine.SlidingJournal(3)
        restored.deserialize(serialized)
        self.assertEqual(restored.size(), 3)

    def test_native_azurite_receipt_has_no_tensors(self):
        engine = tensor_engine
        layer1 = engine.TotrEngine.encode_matrix([[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]], 1)
        layer2 = engine.TotrEngine.encode_matrix([[0.05], [0.05], [0.05], [0.05]], 1)
        blob = f"slide_{uuid.uuid4().hex[:8]}.json"
        journal = f"slide_{uuid.uuid4().hex[:8]}.json"
        receipt = engine.commit_winner_native(layer1, layer2, "0 0 0 0", "0", blob, journal)
        self.assertEqual(receipt["status"], "committed_natively")
        self.assertIsInstance(receipt["loss"], float)
        self.assertIsInstance(receipt["max_residual"], float)
        self.assertIsInstance(receipt["checksum"], str)
        self.assertGreater(receipt["loss"], 0.0)
        for value in receipt.values():
            self.assertNotIsInstance(value, list)

        state = engine.read_totr_state_native(blob, journal)
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["l1_w"].startswith("r:1|g:"))
        self.assertIsInstance(state["loss"], float)
        self.assertIsInstance(state["max_residual"], float)
        self.assertEqual(state["journal_checksum"], receipt["checksum"])
        self.assertNotIn("weights", state)


if __name__ == "__main__":
    unittest.main()
