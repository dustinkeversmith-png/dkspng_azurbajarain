"""Azurite client. Sockets, signatures, and checkpoint bytes stay in the native engine."""

from __future__ import annotations

import os
from typing import Any

from tools.messaging.compressor import Compressor


class BlobClient:
    """HMAC-SHA256 client for local Azurite or Azure Blob Storage."""

    def __init__(
        self,
        transfer: Any | None = None,
        config: Any | None = None,
        architecture: str = "mlp_regressor",
    ) -> None:
        self._transfer = transfer
        self._config = config
        self._architecture = architecture

    def transfer(self):
        if self._transfer is None:
            from tools.storage.transfer import CheckpointTransfer

            self._transfer = CheckpointTransfer(self)
        return self._transfer

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def config(self, checkpoint_or_arch: dict[str, Any] | str | None = None):
        if self._config is not None:
            return self._config
        arch = self._architecture
        if isinstance(checkpoint_or_arch, str) and checkpoint_or_arch:
            arch = checkpoint_or_arch
        elif isinstance(checkpoint_or_arch, dict) and isinstance(checkpoint_or_arch.get("architecture"), str):
            arch = checkpoint_or_arch["architecture"]
        return self.engine().model_config(arch)

    def set_config(self, config: Any) -> None:
        self._config = config

    def connection_string(self) -> str:
        return os.environ.get("BLOB_STORAGE_CONNECTION", "UseDevelopmentStorage=true")

    def blob_name(self, value: str, container: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{container} blob name is required")
        name = value.strip().replace("\\", "/").lstrip("/")
        prefix = container + "/"
        if name.startswith(prefix):
            name = name[len(prefix) :]
        if not name or "/" in name or ".." in name:
            raise ValueError("blob name must be a single file, optionally prefixed by its container")
        return name

    def request(
        self,
        method: str,
        path: str,
        query: dict[str, str] | None = None,
        body: bytes = b"",
        content_type: str = "",
    ) -> bytes:
        raw = self.engine().blob_exchange(
            self.connection_string(),
            method,
            path,
            query or {},
            body or b"",
            content_type,
        )
        return bytes(raw)

    def ensure_container(self, container: str) -> None:
        self.request("PUT", f"/{container}", {"restype": "container"})

    def read_bytes(self, container: str, blob_name: str) -> bytes:
        self.ensure_container(container)
        payload = self.request("GET", f"/{container}/{blob_name}")
        cfg = self.config()
        if len(payload) > cfg.max_blob_bytes:
            raise ValueError(f"{container}/{blob_name} is too large for this tiny run")
        return payload

    def write_bytes(self, container: str, blob_name: str, payload: bytes, content_type: str) -> int:
        cfg = self.config()
        if len(payload) > cfg.max_blob_bytes:
            raise ValueError(f"{container}/{blob_name} would exceed the tiny-checkpoint size cap")
        self.ensure_container(container)
        self.request("PUT", f"/{container}/{blob_name}", body=payload, content_type=content_type)
        return len(payload)

    def read_json(self, container: str, blob_name: str) -> dict[str, Any]:
        parsed = self.engine().storage_json_load(self.read_bytes(container, blob_name))
        if not isinstance(parsed, dict):
            raise ValueError(f"{container}/{blob_name} must be a JSON object")
        return parsed

    def write_json(self, container: str, blob_name: str, payload: dict[str, Any]) -> int:
        encoded = bytes(self.engine().storage_json_dump(payload))
        return self.write_bytes(container, blob_name, encoded, "application/json")

    def parameter_count(self, checkpoint: dict[str, Any]) -> int:
        cfg = self.config(checkpoint)
        return int(self.engine().parameter_count(checkpoint, cfg.max_parameters))

    def summary(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config(checkpoint)
        return self.engine().checkpoint_summary(checkpoint, cfg.max_parameters)

    def eval_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config(payload)
        return self.engine().eval_summary(payload, cfg.max_residuals, cfg.max_anomalies)

    def read_model(self, blob_name: str) -> dict[str, Any]:
        return self.transfer().load_document(blob_name)["layers"]

    def write_model(
        self,
        blob_name: str,
        checkpoint: dict[str, Any],
        outputs: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        return self.transfer().store_document(blob_name, checkpoint, outputs)

    def fetch_model_state(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read models/<weights_blob> and evals/<eval_blob>. Return a summary only."""
        if not isinstance(args, dict):
            raise TypeError("fetch_model_state expects an arguments object")
        weights_blob = self.blob_name(args.get("weights_blob", ""), "models")
        eval_blob = self.blob_name(args.get("eval_blob", ""), "evals")
        document = self.transfer().load_document(weights_blob)
        evaluation = self.eval_summary(self.read_json("evals", eval_blob))
        return {
            "status": "loaded",
            "weights_blob": document["blob_name"],
            "eval_blob": f"evals/{eval_blob}",
            "format": document.get("format", Compressor.FORMAT),
            "compressed": bool(document.get("compressed")),
            "checksum": document["checksum"],
            "bytes": document["bytes"],
            "output_count": self.engine().output_count(document.get("outputs") or []),
            **self.summary(document["layers"]),
            **evaluation,
        }

    def persist_optimized_checkpoint(self, args: dict[str, Any]) -> dict[str, Any]:
        """Overwrite models/<target_blob> with the bytes already stored at source_blob."""
        if not isinstance(args, dict):
            raise TypeError("persist_optimized_checkpoint expects an arguments object")
        if "weights_payload" in args or "weights" in args:
            raise ValueError("Pass source_blob. Raw weights stay in blob storage.")
        source = self.blob_name(args.get("source_blob", ""), "models")
        target = self.blob_name(args.get("target_blob", ""), "models")
        document = self.transfer().load_document(source)
        stored = self.write_model(target, document["layers"], document.get("outputs") or [])
        return {
            "status": "persisted",
            "bytes_written": stored["bytes"],
            **self.engine().omit(stored, ["bytes"]),
        }
