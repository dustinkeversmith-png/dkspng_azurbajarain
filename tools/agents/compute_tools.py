"""C++ perceptron update. Numeric arrays stay in blob storage."""

from __future__ import annotations

from typing import Any

from tools.core.config import MAX_LEARNING_RATE
from tools.core.helpers import json_object_from_text
from tools.storage.blob_client import BlobClient


class ComputeTools:
    """Native diagnostics, one gradient step, and a packed forward pass."""

    def __init__(self, blobs: BlobClient | None = None) -> None:
        self.blobs = blobs or BlobClient()

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def dampening(self, args: dict[str, Any]) -> float:
        if "dampening" in args and not isinstance(args["dampening"], str):
            value = args["dampening"]
        else:
            raw = args.get("dampening_text", args.get("dampening", ""))
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = raw
            else:
                value = json_object_from_text(str(raw)).get("dampening")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("dampening must be a number from 0.1 to 1")
        dampening = float(value)
        if dampening < 0.1 or dampening > 1.0:
            raise ValueError("dampening must be from 0.1 to 1")
        return dampening

    def gradients(self, args: dict[str, Any]) -> dict[str, Any]:
        if isinstance(args.get("gradients"), dict):
            return args["gradients"]
        text = args.get("gradients_text", "")
        if not isinstance(text, str) or "{" not in text:
            raise ValueError("gradients_text must contain the specialist gradient JSON")
        parsed = json_object_from_text(text)
        gradients = parsed.get("gradients", parsed)
        if not isinstance(gradients, dict):
            raise ValueError("gradients must be an object keyed by layer")
        return gradients

    def calculate_matrix_diagnostics(self, args: dict[str, Any]) -> dict[str, Any]:
        """Summarize each layer in models/<weights_blob> using the C++ engine."""
        if not isinstance(args, dict):
            raise TypeError("calculate_matrix_diagnostics expects an arguments object")
        if "weights_matrix" in args or "weights" in args:
            raise ValueError("Pass weights_blob. Raw matrices stay in blob storage.")
        checkpoint = self.blobs.read_model(str(args.get("weights_blob", "")))
        engine = self.engine()
        return {
            "layer_diagnostics": engine.layer_diagnostics(checkpoint),
            **self.blobs.summary(checkpoint),
        }

    def execute_native_update(self, args: dict[str, Any]) -> dict[str, Any]:
        """Update models/<weights_blob> into models/<target_blob> with a capped learning rate."""
        if not isinstance(args, dict):
            raise TypeError("execute_native_update expects an arguments object")
        if "weights" in args or "updates" in args:
            raise ValueError("Pass weights_blob and gradients_text. Raw tensors stay in blob storage.")
        source = str(args.get("weights_blob", ""))
        target = str(args.get("target_blob", ""))
        learning_rate = args.get("learning_rate", 0.01)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise ValueError("learning_rate must be a number")
        if learning_rate < 0 or learning_rate > MAX_LEARNING_RATE:
            raise ValueError(f"learning_rate must be from 0 to {MAX_LEARNING_RATE}")
        decay = args.get("weight_decay", 0.0001)
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or decay < 0:
            raise ValueError("weight_decay must be zero or positive")

        checkpoint = self.blobs.read_model(source)
        gradients = self.gradients(args)
        dampening = self.dampening(args)
        engine = self.engine()
        updated = engine.update_checkpoint(
            checkpoint, gradients, float(learning_rate), float(decay), dampening
        )
        self.blobs.parameter_count(updated)
        stored = self.blobs.write_model(target, updated)
        return {
            "status": "updated",
            "source_blob": source if source.startswith("models/") else f"models/{source}",
            "target_blob": stored["blob_name"],
            "learning_rate": float(learning_rate) * dampening,
            "bytes_written": stored["bytes"],
            **engine.omit(stored, ["bytes", "weight_mean", "weight_min", "weight_max", "parameter_count"]),
            **self.blobs.summary(updated),
        }

    def evaluate_packed_outputs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Pack y = bias + weights · inputs into models/<target_blob>."""
        if not isinstance(args, dict):
            raise TypeError("evaluate_packed_outputs expects an arguments object")
        engine = self.engine()
        inputs = engine.coerce_matrix(args.get("inputs"), "inputs", True)
        engine.validate_tiny_inputs(inputs, 8, 8)
        checkpoint = self.blobs.read_model(str(args.get("weights_blob", "")))
        weights, biases = engine.single_layer(checkpoint)
        outputs = engine.linear_outputs(inputs, weights, biases)
        stored = self.blobs.write_model(str(args.get("target_blob", "")), checkpoint, outputs)
        return {
            "status": "evaluated",
            "bytes_written": stored["bytes"],
            **engine.omit(stored, ["bytes"]),
        }
