"""Burgers ~10K-parameter learning-speed comparison: Adam baseline vs online tournament.

Primary question: which method improves loss faster (epochs/trials and wall time per advance),
not only final accuracy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
PY = sys.executable


def _load_settings() -> None:
    settings_path = ROOT / "local.settings.json"
    if not settings_path.exists():
        return
    values = json.loads(settings_path.read_text(encoding="utf-8")).get("Values") or {}
    for key, value in values.items():
        os.environ.setdefault(str(key), str(value))


def _threshold_table(hits: list[dict[str, Any]], loss_history: list[float] | None = None) -> list[dict[str, Any]]:
    if hits:
        return hits
    # Derive from loss history if needed (online path).
    out: list[dict[str, Any]] = []
    if not loss_history:
        return out
    for thr in THRESHOLDS:
        epoch = -1
        for i, loss in enumerate(loss_history, start=1):
            if loss <= thr:
                epoch = i
                break
        out.append({"threshold": thr, "epoch": epoch, "wall_seconds": -1.0})
    return out


def _advancements(loss_curve: list[float], labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Record each new best loss as a learning advancement."""
    rows: list[dict[str, Any]] = []
    if not loss_curve:
        return rows
    best = loss_curve[0]
    rows.append({"step": 1, "label": (labels[0] if labels else "1"), "best_loss": best, "delta": 0.0})
    for i in range(1, len(loss_curve)):
        if loss_curve[i] + 1e-12 < best:
            delta = loss_curve[i] - best
            best = loss_curve[i]
            rows.append(
                {
                    "step": i + 1,
                    "label": (labels[i] if labels and i < len(labels) else str(i + 1)),
                    "best_loss": best,
                    "delta": delta,
                }
            )
    return rows


def _write_comparison(out_dir: Path, baseline: dict[str, Any], online: dict[str, Any] | None) -> str:
    b_hits = _threshold_table(list(baseline.get("threshold_hits") or []))
    b_adv = _advancements([float(x) for x in baseline.get("loss_history") or []])
    lines = [
        "# Burgers Base (~10K) Learning-Speed Comparison",
        "",
        "Question: does online tournament learning improve loss **faster** than Adam (epochs/trials and wall time per advance)?",
        "",
        "## Adam baseline (SiLU MLP, analytical backprop)",
        "",
        f"- Parameters: `{baseline['parameter_count']}`",
        f"- Train MSE: `{baseline['train_mse_initial']:.6f}` -> best `{baseline['train_mse_best']:.6f}` "
        f"(final `{baseline['train_mse_final']:.6f}`)",
        f"- Extrapolation MSE at best: `{baseline.get('extrap_mse_at_best', baseline['extrap_mse_final']):.6f}`",
        f"- Extrapolation MSE final: `{baseline['extrap_mse_final']:.6f}`",
        f"- PDE residual (mean grid): `{baseline['mean_pde_residual']:.6f}`",
        f"- Epochs run: `{baseline['epochs_completed']}` (best at epoch `{baseline['best_epoch']}`)",
        f"- Mean epoch wall time: `{baseline['mean_epoch_seconds']:.4f}s`",
        f"- Total wall time: `{baseline['elapsed_wall_seconds']:.2f}s`",
        f"- Converged: `{baseline['converged']}` (stop: `{baseline['stop_reason']}`)",
        "",
        "### Epochs / wall time to train-MSE thresholds",
        "",
        "| Threshold | Epoch first hit | Wall seconds |",
        "| ---: | ---: | ---: |",
    ]
    for hit in b_hits:
        ep = hit.get("epoch", -1)
        wall = hit.get("wall_seconds", -1.0)
        ep_s = "never" if ep is None or int(ep) < 0 else str(int(ep))
        wall_s = "—" if wall is None or float(wall) < 0 else f"{float(wall):.3f}"
        lines.append(f"| {float(hit['threshold']):.2f} | {ep_s} | {wall_s} |")

    lines.extend(
        [
            "",
            f"### Learning advancements (new best train MSE): `{len(b_adv)}` total",
            "",
            "| Epoch | Best train MSE | Delta |",
            "| ---: | ---: | ---: |",
        ]
    )
    # Condense: first, every N, and last few if long
    show = b_adv
    if len(show) > 24:
        head = show[:8]
        tail = show[-8:]
        mid = show[8:-8]
        stride = max(1, len(mid) // 8)
        show = head + mid[::stride][:8] + tail
    for row in show:
        lines.append(f"| {row['step']} | {row['best_loss']:.6f} | {row['delta']:+.6f} |")

    if online:
        trials = list(online.get("trials") or [])
        best_curve = [float(online["initial_loss"])]
        labels = ["seed"]
        for t in trials:
            best_curve.append(float(t["best_loss"]))
            labels.append(f"t{t['trial']}:{t['status'][:3]}")
        o_adv = _advancements(best_curve, labels)
        commits = [t for t in trials if t.get("status") == "committed_natively"]
        lines.extend(
            [
                "",
                "## Online tournament (`BurgerOnlineModel` / Azure OpenAI)",
                "",
                f"- Parameters: ~`{baseline['parameter_count']}` (same topology family)",
                f"- Train MSE: `{online['initial_loss']:.6f}` -> best `{online['best_loss']:.6f}` "
                f"(`{online['loss_improvement_pct']:+.2f}%`)",
                f"- Trials run: `{online['trials_run']}` | commits `{online['commits']}` | "
                f"rejected `{online['rejected']}` | diverged `{online.get('diverged', 0)}`",
                f"- Total wall time: `{online['elapsed_seconds']:.2f}s` "
                f"(~`{online['elapsed_seconds'] / max(1, online['trials_run']):.2f}s` / trial)",
                f"- Tokens: `{online['token_usage']['total_tokens']}` "
                f"(prompt `{online['token_usage']['prompt_tokens']}`, "
                f"completion `{online['token_usage']['completion_tokens']}`)",
                "",
                "### Condensed trial log (commits + every 5th trial)",
                "",
                "| Trial | Slice | Status | Loss | Best | Delta | Trial tokens |",
                "| ---: | :--- | :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for t in trials:
            keep = t.get("status") == "committed_natively" or int(t["trial"]) == 1 or int(t["trial"]) % 5 == 0 or int(
                t["trial"]
            ) == len(trials)
            if not keep:
                continue
            lines.append(
                f"| {t['trial']} | `{t['slice']}` | `{t['status']}` | {float(t['loss']):.6f} | "
                f"{float(t['best_loss']):.6f} | {float(t['delta']):+.6f} | {t['total_tokens']} |"
            )

        lines.extend(
            [
                "",
                f"### Learning advancements (new best after a trial): `{max(0, len(o_adv) - 1)}` improvements",
                "",
                "| After | Best train MSE | Delta |",
                "| :--- | ---: | ---: |",
            ]
        )
        for row in o_adv:
            lines.append(f"| {row['label']} | {row['best_loss']:.6f} | {row['delta']:+.6f} |")

        # Speed comparison at shared thresholds reached by both
        lines.extend(
            [
                "",
                "## Head-to-head: effort to reach shared train-MSE thresholds",
                "",
                "| Threshold | Adam epochs | Adam seconds | Online trials to best<=thr | Online seconds (approx) |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        mean_trial_s = online["elapsed_seconds"] / max(1, online["trials_run"])
        for hit in b_hits:
            thr = float(hit["threshold"])
            adam_ep = int(hit.get("epoch", -1))
            adam_s = float(hit.get("wall_seconds", -1.0))
            online_trial = -1
            running_best = float(online["initial_loss"])
            if running_best <= thr:
                online_trial = 0
            else:
                for t in trials:
                    running_best = min(running_best, float(t["best_loss"]))
                    if running_best <= thr:
                        online_trial = int(t["trial"])
                        break
            adam_ep_s = "never" if adam_ep < 0 else str(adam_ep)
            adam_sec_s = "-" if adam_s < 0 else f"{adam_s:.2f}"
            on_trial_s = "never" if online_trial < 0 else str(online_trial)
            on_sec_s = "-" if online_trial < 0 else f"{online_trial * mean_trial_s:.2f}"
            lines.append(f"| {thr:.2f} | {adam_ep_s} | {adam_sec_s} | {on_trial_s} | {on_sec_s} |")

        lines.extend(
            [
                "",
                "## Verdict sketch",
                "",
                f"- Adam advancements: `{len(b_adv)}` new-bests over `{baseline['epochs_completed']}` epochs "
                f"(`{baseline['elapsed_wall_seconds']:.1f}s`).",
                f"- Online advancements: `{max(0, len(o_adv) - 1)}` new-bests over `{online['trials_run']}` trials "
                f"(`{online['elapsed_seconds']:.1f}s`).",
                f"- Adam final/best train MSE `{baseline['train_mse_best']:.6f}` vs online best `{online['best_loss']:.6f}`.",
                f"- Adam extrapolation `{baseline['extrap_mse_final']:.6f}` (online eval set not recomputed here).",
                "",
            ]
        )

    text = "\n".join(lines) + "\n"
    (out_dir / "comparison_report.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Burgers learning-speed comparison runner")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--online-trials", type=int, default=40)
    parser.add_argument("--max-usd", type=float, default=10.0)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.baseline_only:
        args.skip_online = True

    _load_settings()
    from tools import tensor_engine

    out_dir = ROOT / "reports" / "burger_runs" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[baseline] Adam ~10K Burgers to near-convergence...", flush=True)
    t0 = time.time()
    baseline = dict(
        tensor_engine.run_burger_baseline(
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            learning_rate=0.001,
            batch_size=64,
            r_enc=8,
            width=68,
            seed=2026,
            patience=args.patience,
            min_delta=1e-4,
            track_extrap_each_epoch=True,
        )
    )
    print(
        f"[baseline] done in {time.time() - t0:.1f}s | epochs={baseline['epochs_completed']} "
        f"best={baseline['train_mse_best']:.6f} @ep{baseline['best_epoch']} "
        f"extrap={baseline['extrap_mse_final']:.6f} converged={baseline['converged']} "
        f"({baseline['stop_reason']}) mean_epoch={baseline['mean_epoch_seconds']:.4f}s",
        flush=True,
    )
    (out_dir / "benchmark_artifact.json").write_text(str(baseline["json"]), encoding="utf-8")
    (out_dir / "trajectory.csv").write_text(str(baseline["trajectory_csv"]), encoding="utf-8")
    baseline_summary = {
        k: baseline[k]
        for k in (
            "parameter_count",
            "train_mse_initial",
            "train_mse_final",
            "train_mse_best",
            "extrap_mse_final",
            "extrap_mse_at_best",
            "mean_pde_residual",
            "elapsed_wall_seconds",
            "mean_epoch_seconds",
            "epochs_completed",
            "best_epoch",
            "converged",
            "stop_reason",
            "threshold_hits",
        )
    }
    (out_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")

    online_summary = None
    if not args.skip_online and args.online_trials > 0:
        # Empirically ~2.5k tokens/trial on this model; ~$0.01–0.05/trial on gpt-5-mini → 40 trials << $10.
        est_per_trial = 0.05
        if args.online_trials * est_per_trial > args.max_usd:
            capped = max(1, int(args.max_usd / est_per_trial))
            print(f"[online] capping trials {args.online_trials} -> {capped} for ${args.max_usd} budget", flush=True)
            args.online_trials = capped

        print(f"[online] BurgerOnlineModel for up to {args.online_trials} trials...", flush=True)
        cmd = [
            PY,
            str(ROOT / "scripts" / "run_single_model.py"),
            "--model=burger_base",
            f"--trials={args.online_trials}",
            f"--target-loss={args.target_loss}",
            f"--out-dir={out_dir / 'online'}",
            "--run-name=burger_base_online",
        ]
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        summary_path = out_dir / "online" / "burger_base_online" / "run_summary.json"
        online_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    report = _write_comparison(out_dir, baseline, online_summary)
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"[done] {out_dir / 'comparison_report.md'}", flush=True)


if __name__ == "__main__":
    main()
