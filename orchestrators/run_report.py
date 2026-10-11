"""Write tournament reports into the repo's reports folder."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
JSONL_NAME = "tournament_completions.jsonl"
LATEST_NAME = "latest_report.md"


def _directory(directory: Path | None) -> Path:
    folder = REPORT_DIR if directory is None else directory
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(source: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")
    return cleaned[:80] or "run"


def append_attempt(record: dict[str, Any], directory: Path | None = None) -> Path:
    """Append one raw completion. This file is the durable log of every model reply."""
    folder = _directory(directory)
    path = folder / JSONL_NAME
    row = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def publish_report(source: str, body: str, directory: Path | None = None) -> Path:
    """Overwrite latest_report.md and keep a timestamped copy beside it."""
    folder = _directory(directory)
    stamp = _stamp()
    text = f"# {source}\n\nWritten {stamp} UTC\n\n{body.rstrip()}\n"
    (folder / LATEST_NAME).write_text(text, encoding="utf-8")
    path = folder / f"{stamp}_{_slug(source)}.md"
    path.write_text(text, encoding="utf-8")
    print(f"[report] {path}")
    return path
