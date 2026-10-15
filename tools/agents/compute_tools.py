"""C++ perceptron update. Argument validation stays here; format/encoding parsing lives in the model definition."""

from __future__ import annotations

from typing import Any

from tools.storage.blob_client import BlobClient


class ComputeTools:
    """Native diagnostics, one gradient step, and a packed forward pass delegated to the model."""

    def __init__(
        self,
        blobs: BlobClient | None = None,
        config: Any | None = None,
        architecture: str = "mlp_regressor",
    ) -> None:
        self._architecture = architecture
        self.blobs = blobs or BlobClient(config=config, architecture=architecture)
        if config is not None:
            self.blobs.set_config(config)

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def _arch(self, checkpoint: dict[str, Any] | None = None) -> str:
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("architecture"), str):
            return checkpoint["architecture"]
        return self._architecture

    def dampening(self, args: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> float:
        if not isinstance(args, dict):
            raise TypeError("dampening expects an arguments object")
        return float(self.engine().parse_dampening(args, self._arch(checkpoint)))

    def gradients(self, args: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise TypeError("gradients expects an arguments object")
        return dict(self.engine().parse_gradients(args, checkpoint, self._arch(checkpoint)))

    def learning_rate(self, args: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> float:
        if not isinstance(args, dict):
            raise TypeError("learning_rate expects an arguments object")
        lr = args.get("learning_rate")
        if lr is not None and (isinstance(lr, bool) or not isinstance(lr, (int, float))):
            raise ValueError("learning_rate must be a number")
        cfg = self.blobs.config(checkpoint)
        resolved = float(cfg.default_learning_rate if lr is None else lr)
        if resolved <= 0 or resolved > cfg.max_learning_rate:
            raise ValueError(f"learning_rate must be in (0, {cfg.max_learning_rate}]")
        return float(self.engine().parse_learning_rate(args, self._arch(checkpoint)))

    def weight_decay(self, args: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> float:
        if not isinstance(args, dict):
            raise TypeError("weight_decay expects an arguments object")
        decay = args.get("weight_decay")
        if decay is not None and (isinstance(decay, bool) or not isinstance(decay, (int, float)) or decay < 0):
            raise ValueError("weight_decay must be zero or positive")
        return float(self.engine().parse_weight_decay(args, self._arch(checkpoint)))

    def calculate_matrix_diagnostics(self, args: dict[str, Any]) -> dict[str, Any]:
        """Summarize each layer in models/<weights_blob> using the model's layer stats in C++."""
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
        """Update models/<weights_blob> into models/<target_blob> using model-parsed gradients and limits."""
        if not isinstance(args, dict):
            raise TypeError("execute_native_update expects an arguments object")
        if "weights" in args or "updates" in args:
            raise ValueError("Pass weights_blob and gradients_text. Raw tensors stay in blob storage.")
        source = str(args.get("weights_blob", ""))
        target = str(args.get("target_blob", ""))
        checkpoint = self.blobs.read_model(source)
        learning_rate = self.learning_rate(args, checkpoint)
        decay = self.weight_decay(args, checkpoint)
        gradients = self.gradients(args, checkpoint)
        dampening = self.dampening(args, checkpoint)
        engine = self.engine()
        updated = engine.update_checkpoint(
            checkpoint, gradients, learning_rate, decay, dampening
        )
        self.blobs.parameter_count(updated)
        stored = self.blobs.write_model(target, updated)
        return {
            "status": "updated",
            "source_blob": source if source.startswith("models/") else f"models/{source}",
            "target_blob": stored["blob_name"],
            "learning_rate": learning_rate * dampening,
            "bytes_written": stored["bytes"],
            **engine.omit(stored, ["bytes", "weight_mean", "weight_min", "weight_max", "parameter_count"]),
            **self.blobs.summary(updated),
        }

    def evaluate_packed_outputs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Evaluate inputs through the model's layer evaluator and pack into models/<target_blob>."""
        if not isinstance(args, dict):
            raise TypeError("evaluate_packed_outputs expects an arguments object")
        engine = self.engine()
        checkpoint = self.blobs.read_model(str(args.get("weights_blob", "")))
        outputs = engine.evaluate_checkpoint_inputs(checkpoint, args.get("inputs"))
        stored = self.blobs.write_model(str(args.get("target_blob", "")), checkpoint, outputs)
        return {
            "status": "evaluated",
            "bytes_written": stored["bytes"],
            **engine.omit(stored, ["bytes"]),
        }
