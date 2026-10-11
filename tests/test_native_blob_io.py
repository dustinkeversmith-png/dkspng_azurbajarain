"""Azurite reads and writes go through the compiled client."""

from __future__ import annotations

import unittest
import uuid

from tools import tensor_engine
from tools.agents.native_storage_tools import NativeStorageTools


class NativeBlobIoTest(unittest.TestCase):
    def test_put_and_get_match(self) -> None:
        connection = "UseDevelopmentStorage=true"
        payload = b"native-bytes-check"
        tensor_engine.blob_exchange(connection, "PUT", "/models", {"restype": "container"}, b"", "")
        tensor_engine.blob_exchange(
            connection, "PUT", "/models/native_bytes.bin", {}, payload, "application/octet-stream"
        )
        loaded = tensor_engine.blob_exchange(connection, "GET", "/models/native_bytes.bin", {}, b"", "")
        self.assertEqual(loaded, payload)

    def test_commit_and_read_round_trip_without_python_bytes(self) -> None:
        tools = NativeStorageTools()
        layer1 = tensor_engine.TotrEngine.encode_matrix([[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]], 1)
        layer2 = tensor_engine.TotrEngine.encode_matrix([[0.05], [0.05], [0.05], [0.05]], 1)
        blob = f"round_{uuid.uuid4().hex[:8]}.json"
        journal = f"round_{uuid.uuid4().hex[:8]}.json"
        first = tools.commit_winner_native(
            {
                "winning_compressed": {
                    "l1_w": layer1,
                    "l2_w": layer2,
                    "l1_b": "0 0 0 0",
                    "l2_b": "0",
                },
                "target_blob": blob,
                "journal_blob": journal,
            }
        )
        self.assertEqual(first["status"], "committed_natively")
        self.assertGreater(first["loss"], 0.0)
        self.assertEqual(first["blob"], "models/" + blob)
        self.assertNotIn("l1_w", first)
        state = tools.read_totr_state_native(
            {"checkpoint_blob": blob, "journal_blob": journal}
        )
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["l1_w"].startswith("r:1|g:"))
        self.assertNotIn("weights", state)
        second = tools.commit_winner_native(
            {
                "winning_compressed": {
                    "l1_w": state["l1_w"],
                    "l2_w": state["l2_w"],
                    "l1_b": state["l1_b"],
                    "l2_b": state["l2_b"],
                },
                "target_blob": blob,
                "journal_blob": journal,
            }
        )
        third = tools.commit_winner_native(
            {
                "winning_compressed": {
                    "l1_w": state["l1_w"],
                    "l2_w": state["l2_w"],
                    "l1_b": state["l1_b"],
                    "l2_b": state["l2_b"],
                },
                "target_blob": blob,
                "journal_blob": journal,
            }
        )
        self.assertEqual(second["checksum"], third["checksum"])
        self.assertEqual(second["bytes"], third["bytes"])


if __name__ == "__main__":
    unittest.main()
