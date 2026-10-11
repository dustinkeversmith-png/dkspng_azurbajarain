"""Upload the tiny perceptron fixtures into local Azurite."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.storage.blob_client import BlobClient
from tools.storage.transfer import CheckpointTransfer

client = BlobClient()
transfer = CheckpointTransfer(client)
weights = json.loads((ROOT / "artifacts" / "storage" / "sample_weights.json").read_text(encoding="utf-8"))
evaluation = json.loads((ROOT / "artifacts" / "storage" / "sample_eval.json").read_text(encoding="utf-8"))
transfer.store_document("checkpoint_v1.json", weights)
client.write_json("evals", "batch_error_run42.json", evaluation)
print("seeded models/checkpoint_v1.json and evals/batch_error_run42.json")
