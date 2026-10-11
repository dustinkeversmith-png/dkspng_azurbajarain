"""Move a packed checkpoint between tools without returning the numeric arrays."""

from __future__ import annotations

from typing import Any

from tools.messaging.compressor import Compressor


class CheckpointTransfer:
    """Read and write SJC1 checkpoints through a blob client."""

    def __init__(self, blobs: Any | None = None, compressor: Compressor | None = None) -> None:
        if blobs is None:
            from tools.storage.blob_client import BlobClient

            blobs = BlobClient(transfer=self)
        self.blobs = blobs
        self.blobs._transfer = self
        self.compressor = compressor or Compressor()

    def store_document(
        self,
        blob_name: str,
        layers: dict[str, Any],
        outputs: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        name = self.blobs.blob_name(blob_name, "models")
        self.blobs.parameter_count(layers)
        sealed = self.compressor.seal(layers, outputs or [])
        self.blobs.write_bytes("models", name, sealed, "application/octet-stream")
        document = self.compressor.open_seal(sealed)
        return self.compressor.receipt(f"models/{name}", sealed, document)

    def load_document(self, blob_name: str) -> dict[str, Any]:
        name = self.blobs.blob_name(blob_name, "models")
        raw = self.blobs.read_bytes("models", name)
        if not raw.startswith(self.compressor.FORMAT.encode("ascii")):
            raise ValueError("checkpoint must be an SJC1 container")
        document = self.compressor.open_seal(raw)
        self.blobs.parameter_count(document["layers"])
        document["blob_name"] = f"models/{name}"
        document["bytes"] = len(raw)
        document["format"] = self.compressor.FORMAT
        return document
