"""SJC1 container. Packing, summaries, and checksums run in the compiled engine."""

from __future__ import annotations

from typing import Any


class Compressor:
    """Pack and open checkpoint containers without exposing raw arrays to the model."""

    FORMAT = "SJC1"

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def seal(self, layers: dict[str, Any], outputs: list[list[float]] | None = None) -> bytes:
        raw = self.engine().pack_checkpoint(layers, outputs or [])
        if isinstance(raw, str):
            raw = raw.encode("latin1")
        return bytes(raw)

    def open_seal(self, blob: bytes) -> dict[str, Any]:
        document = self.engine().SJC1Compressor.unpack(blob)
        document["format"] = self.FORMAT
        document["bytes"] = len(blob)
        document["checksum"] = self.engine().content_checksum(blob)
        return document

    def receipt(self, blob_name: str, sealed: bytes, document: dict[str, Any]) -> dict[str, Any]:
        summary = self.engine().receipt_stats(document["layers"], document.get("outputs") or [])
        summary["format"] = self.FORMAT
        summary["compressed"] = bool(document.get("compressed"))
        summary["blob_name"] = blob_name
        summary["bytes"] = len(sealed)
        summary["checksum"] = self.engine().content_checksum(sealed)
        return summary
