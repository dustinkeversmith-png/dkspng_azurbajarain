"""Slice reads stay on TOTR strings. The full network is scored in C++."""

from __future__ import annotations

import unittest
import uuid

from tools import tensor_engine


CONFIGS = {
    "A": [[2, 4], [4, 1]],
    "B": [[4, 8], [8, 4], [4, 1]],
    "C": [[8, 16], [16, 1]],
}


def _no_matrix(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "weights":
                raise AssertionError("weight matrix leaked")
            _no_matrix(item)
    elif isinstance(value, list):
        if value and isinstance(value[0], list):
            raise AssertionError("nested matrix leaked")
        for item in value:
            _no_matrix(item)


class ModularSliceTest(unittest.TestCase):
    def test_three_shapes_round_trip_one_slice(self) -> None:
        for name, shapes in CONFIGS.items():
            blob = f"mod_{name}_{uuid.uuid4().hex[:8]}.json"
            journal = f"mod_{name}_{uuid.uuid4().hex[:8]}.json"
            seeded = tensor_engine.seed_stack_native(shapes, blob, journal)
            _no_matrix(dict(seeded))
            self.assertEqual(seeded["status"], "seeded")
            self.assertLess(float(seeded["loss"]), 100.0)
            meta = tensor_engine.read_model_meta(blob, journal)
            _no_matrix(dict(meta))
            self.assertGreaterEqual(len(list(meta["layers"])), len(shapes))
            widest = tensor_engine.read_layer_slice(blob, journal, list(meta["layers"])[0])
            _no_matrix(dict(widest))
            self.assertTrue(str(widest["totr_w"]).startswith("r:1|g:"))
            self.assertNotIn("000000", str(widest["biases"]))
            self.assertLessEqual(int(widest["in_dim"]), 4)
            self.assertLessEqual(int(widest["out_dim"]), 4)
            receipt = tensor_engine.commit_slice_native(
                list(meta["layers"])[0],
                "r:1|g:0.9|A:"
                + " ".join(["90"] * int(widest["in_dim"]))
                + "|B:"
                + " ".join(["90"] * int(widest["out_dim"])),
                " ".join(["1"] * int(widest["out_dim"])),
                blob,
                journal,
            )
            _no_matrix(dict(receipt))
            self.assertIn(receipt["status"], ("rejected", "rejected_divergent", "committed_natively"))
            self.assertEqual(receipt["checksum"], seeded["checksum"])

    def test_neural_model_chains_layers_and_seeds(self) -> None:
        model = tensor_engine.NeuralModel("CustomDeepClassifier")
        model.add_layer("stem", 8, 16, tensor_engine.ActivationType.ReLU).add_layer(
            "hidden", 16, 8, tensor_engine.ActivationType.Tanh
        ).add_layer("head", 8, 1, tensor_engine.ActivationType.Linear).set_loss(
            tensor_engine.LossType.MeanSquaredError
        )
        model.initialize_default_weights()
        blob = f"custom_{uuid.uuid4().hex[:8]}.json"
        journal = f"custom_{uuid.uuid4().hex[:8]}.json"
        receipt = model.seed_to_storage(blob, journal)
        _no_matrix(dict(receipt))
        self.assertEqual(receipt["status"], "seeded")
        self.assertGreater(len(list(receipt["layers"])), 3)
        meta = tensor_engine.read_model_meta(blob, journal)
        self.assertEqual(list(meta["layers"]), list(receipt["layers"]))
        slice_state = tensor_engine.read_layer_slice(blob, journal, list(meta["layers"])[0])
        _no_matrix(dict(slice_state))
        self.assertLessEqual(int(slice_state["in_dim"]), 4)
        self.assertLessEqual(int(slice_state["out_dim"]), 4)


if __name__ == "__main__":
    unittest.main()
