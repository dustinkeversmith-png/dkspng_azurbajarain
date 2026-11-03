"""Bake off Burgers online setups: base vs breakwall vs compact.

Reports best loss, threshold hits, and token usage per setup.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SETUPS = ("burger_base", "burger_breakwall", "burger_compact")
THRESHOLDS = (0.10, 0.05, 0.03, 0.026, 0.02, 0.01)


def _load_settings() -> None:
    path = ROOT / "local.settings.json"
    if not path.exists():
        return
    values = json.loads(path.read_text(encoding="utf-8")).get("Values") or {}
    for key, value in values.items():
        os.environ.setdefault(str(key), str(value))


def _first_trial_at(summary: dict, thr: float) -> int | None:
    best = float(summary["initial_loss"])
    if best <= thr:
        return 0
    for row in summary.get("trials") or []:
        best = min(best, float(row["best_loss"]))
        if best <= thr:
            return int(row["trial"])
    return None


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    setups = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT_SETUPS
    _load_settings()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = ROOT / "reports" / "burger_runs" / f"{stamp}_setups"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for setup in setups:
        print(f"\n=== SETUP {setup} trials={trials} target={target} ===", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_single_model.py"),
            f"--model={setup}",
            f"--trials={trials}",
            f"--target-loss={target}",
            f"--out-dir={out_root / setup}",
            f"--run-name={setup}",
            "--visible-history=1" if setup != "burger_base" else "--visible-history=3",
            "--min-commit-drop=0.00001" if setup != "burger_base" else "--min-commit-drop=0.0001",
        ]
        # Base keeps defaults from its model config via args override above.
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            print(f"[fail] {setup} exit={completed.returncode}", flush=True)
            rows.append({"setup": setup, "status": "failed", "exit_code": completed.returncode})
            continue

        summary_path = out_root / setup / setup / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tokens = summary["token_usage"]
        row = {
            "setup": setup,
            "status": "ok",
            "initial_loss": summary["initial_loss"],
            "best_loss": summary["best_loss"],
            "trials_run": summary["trials_run"],
            "commits": summary["commits"],
            "rejected": summary["rejected"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
            "avg_tokens_per_trial": tokens["avg_total_tokens_per_trial"],
            "broke_026": float(summary["best_loss"]) < 0.026,
            "broke_02": float(summary["best_loss"]) < 0.02,
            "broke_01": float(summary["best_loss"]) <= 0.01,
        }
        for thr in THRESHOLDS:
            key = f"trials_to_{str(thr).replace('.', 'p')}"
            hit = _first_trial_at(summary, thr)
            row[key] = hit if hit is not None else ""
        rows.append(row)
        print(
            f"[result] {setup}: best={row['best_loss']:.6f} commits={row['commits']}/{row['trials_run']} "
            f"tokens={row['total_tokens']} avg/trial={row['avg_tokens_per_trial']:.1f} "
            f"broke_0.026={row['broke_026']} broke_0.02={row['broke_02']}",
            flush=True,
        )

    csv_path = out_root / "setup_comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Burgers online setup bakeoff",
        "",
        f"Trials/setup: `{trials}` | target loss: `{target}`",
        "",
        "| Setup | Best loss | Commits | Total tokens | Avg tok/trial | <0.026 | <0.02 | <=0.01 |",
        "| :--- | ---: | ---: | ---: | ---: | :---: | :---: | :---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| `{row['setup']}` | FAIL | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| `{row['setup']}` | {float(row['best_loss']):.6f} | {row['commits']}/{row['trials_run']} | "
            f"{row['total_tokens']} | {float(row['avg_tokens_per_trial']):.1f} | "
            f"{'Y' if row['broke_026'] else 'N'} | {'Y' if row['broke_02'] else 'N'} | "
            f"{'Y' if row['broke_01'] else 'N'} |"
        )
    lines.extend(["", f"CSV: `{csv_path}`", ""])
    report = "\n".join(lines)
    (out_root / "setup_comparison.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
