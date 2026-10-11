"""Pass blob names into the native Azurite client. Checkpoint bytes stay in C++."""

from __future__ import annotations

from typing import Any

from tools.core.host import argument_object


def _payload(value: dict[str, Any] | str) -> dict[str, Any]:
    return argument_object(value)


class NativeStorageTools:
    """Read state and commit candidate updates. Bytes stay in C++."""

    HOST_TOOLS = (
        (
            "read_totr_state_native",
            "Read current checkpoint and journal. Pass checkpoint_blob and journal_blob.",
        ),
        (
            "commit_winner_native",
            "Score candidate in C++ and commit if loss decreases. Pass l1_w, l2_w, l1_b, l2_b, target_blob, journal_blob.",
        ),
    )

    def engine(self):
        from tools import tensor_engine

        return tensor_engine

    def read_totr_state_native(self, args: dict[str, Any] | str) -> dict[str, Any]:
        """Download checkpoint and journal inside the native client."""
        payload = _payload(args)
        checkpoint = payload.get("checkpoint_blob", "sliding_totr.json")
        journal = payload.get("journal_blob", "sliding_journal.json")
        res = self.engine().read_state_native(checkpoint, journal)
        res["checkpoint_uri"] = f"models/{checkpoint}"
        res["journal_uri"] = f"evals/{journal}"
        return res

    def commit_winner_native(self, args: dict[str, Any] | str) -> dict[str, Any]:
        """Decode winning TOTR text, score it, and write to Azurite only if loss drops."""
        payload = _payload(args)
        compressed = payload.get("winning_compressed", payload)
        if isinstance(compressed, str):
            compressed = _payload(compressed)
        if not isinstance(compressed, dict):
            raise ValueError("winning_compressed must be an object")
        target = payload.get("target_blob", "sliding_totr.json")
        journal = payload.get("journal_blob", "sliding_journal.json")
        for key in ("l1_w", "l2_w", "l1_b", "l2_b"):
            if not isinstance(compressed.get(key), str):
                raise ValueError(f"{key} must be a TOTR or bias string")
        return self.engine().commit_winner_native(
            compressed["l1_w"],
            compressed["l2_w"],
            compressed["l1_b"],
            compressed["l2_b"],
            target,
            journal,
        )
