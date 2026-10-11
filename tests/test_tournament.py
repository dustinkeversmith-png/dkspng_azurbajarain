"""Tournament scoring stays in C++ and receipts stay free of weight matrices."""

from __future__ import annotations

import re
import unittest
import uuid

from tools import tensor_engine

_HINT = re.compile(r"A:\[(?:0|\+1|\+2|-1|-2)(?:, (?:0|\+1|\+2|-1|-2))*\] B:\[(?:0|\+1|\+2|-1|-2)(?:, (?:0|\+1|\+2|-1|-2))*\]")


def _no_matrix(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"weights", "w"} and isinstance(item, list):
                raise AssertionError("weight matrix leaked")
            _no_matrix(item)
    elif isinstance(value, list):
        if value and isinstance(value[0], list):
            raise AssertionError("nested matrix leaked")
        for item in value:
            _no_matrix(item)


class TournamentGateTest(unittest.TestCase):
    def test_cpp_keeps_the_lower_loss_candidate(self) -> None:
        blob = f"tour_{uuid.uuid4().hex[:8]}.json"
        journal = f"tour_{uuid.uuid4().hex[:8]}.json"
        seeded = tensor_engine.MlpRegressor().seed_to_storage(blob, journal, 10)
        _no_matrix(dict(seeded))
        self.assertEqual(seeded["directives"]["model_type"], "MlpRegressor")
        slice_state = tensor_engine.read_architecture_slice(blob, journal, list(seeded["layers"])[0])
        _no_matrix(dict(slice_state))
        self.assertRegex(slice_state["gradient_hint"], _HINT)
        self.assertIn("sparsity", dict(slice_state["telemetry"]))
        self.assertLessEqual(int(slice_state["in_dim"]), 4)
        receipt = tensor_engine.commit_tournament(
            list(seeded["layers"])[0],
            [
                {"name": "c1", "w": slice_state["totr_w"], "b": slice_state["biases"]},
                {"name": "c2", "w": "r:1|g:0.9|A:90 90 90 90|B:90 90 90 90", "b": "1 1 1 1"},
                {"name": "c3", "w": "r:1|g:0.02|A:1 -1 1 -1|B:1 1 1 1", "b": "0.1 0.1 0.1 0.1"},
            ],
            blob,
            journal,
            10,
        )
        _no_matrix(dict(receipt))
        scored = [dict(row) for row in receipt["candidates"] if row["status"] == "scored"]
        self.assertTrue(scored)
        best = min(scored, key=lambda row: float(row["loss"]))
        self.assertEqual(receipt["winner"], best["name"])
        if receipt["status"] == "committed_natively":
            self.assertLess(float(receipt["loss"]), float(seeded["loss"]))
            self.assertNotEqual(receipt["checksum"], seeded["checksum"])
        else:
            self.assertEqual(receipt["checksum"], seeded["checksum"])

    def test_copied_active_slice_is_not_committed(self) -> None:
        blob = f"tour_{uuid.uuid4().hex[:8]}.json"
        journal = f"tour_{uuid.uuid4().hex[:8]}.json"
        seeded = tensor_engine.MlpRegressor().seed_to_storage(blob, journal, 10)
        layer = list(seeded["layers"])[0]
        slice_state = tensor_engine.read_architecture_slice(blob, journal, layer)
        copy = {"w": slice_state["totr_w"], "b": slice_state["biases"]}
        receipt = tensor_engine.commit_tournament(
            layer,
            [
                {"name": "c1", **copy},
                {"name": "c2", **copy},
                {"name": "c3", **copy},
            ],
            blob,
            journal,
            10,
        )
        self.assertEqual(receipt["status"], "rejected")
        self.assertEqual(receipt["winner"], "none")
        self.assertEqual(receipt["checksum"], seeded["checksum"])
        self.assertEqual([dict(row)["status"] for row in receipt["candidates"]], ["unchanged", "unchanged", "unchanged"])

    def test_drop_smaller_than_one_ten_thousandth_is_rejected(self) -> None:
        blob = f"tour_{uuid.uuid4().hex[:8]}.json"
        journal = f"tour_{uuid.uuid4().hex[:8]}.json"
        seeded = tensor_engine.MlpRegressor().seed_to_storage(blob, journal, 10)
        layer = list(seeded["layers"])[0]
        slice_state = tensor_engine.read_architecture_slice(blob, journal, layer)
        moved = slice_state["totr_w"].replace("g:0.002", "g:0.002001")
        opened = tensor_engine.commit_tournament(
            layer,
            [{"name": "c1", "w": moved, "b": slice_state["biases"]}] * 3,
            blob,
            journal,
            10,
        )
        self.assertEqual(opened["status"], "committed_natively")
        micro = "r:1|g:0.0022|A:-99 -33 33 99|B:74 74 74 74"
        receipt = tensor_engine.commit_tournament(
            layer,
            [{"name": "c1", "w": micro, "b": slice_state["biases"]}] * 3,
            blob,
            journal,
            10,
        )
        self.assertEqual(receipt["status"], "rejected")
        self.assertGreater(float(receipt["delta"]), -1e-4)
        self.assertLess(float(receipt["delta"]), 0.0)
        self.assertEqual(receipt["checksum"], opened["checksum"])

    def test_autoencoder_slice_is_reconstruction_telemetry(self) -> None:
        blob = f"ae_{uuid.uuid4().hex[:8]}.json"
        journal = f"ae_{uuid.uuid4().hex[:8]}.json"
        seeded = tensor_engine.BottleneckAutoencoder().seed_to_storage(blob, journal, 10)
        self.assertEqual(seeded["architecture"], "bottleneck_ae")
        self.assertIn("bottleneck", seeded["directives"]["task"].lower())
        meta = tensor_engine.read_architecture_meta(blob, journal)
        _no_matrix(dict(meta))
        self.assertEqual(list(meta["layers"]), ["encode", "decode"])
        first = tensor_engine.read_architecture_slice(blob, journal, "encode")
        self.assertEqual(int(first["in_dim"]), 8)
        self.assertEqual(int(first["out_dim"]), 2)
        self.assertIn("sparsity", dict(meta["telemetry"]))


if __name__ == "__main__":
    unittest.main()
