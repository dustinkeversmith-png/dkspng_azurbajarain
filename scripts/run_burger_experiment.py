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


def _level_bands(curve: list[float], step_size: int) -> list[dict[str, Any]]:
    """Summarize mean/end loss and per-step improvement over fixed epoch/trial bands."""
    if not curve or step_size < 1:
        return []
    rows: list[dict[str, Any]] = []
    for start in range(0, len(curve), step_size):
        end = min(len(curve), start + step_size)
        chunk = curve[start:end]
        if not chunk:
            continue
        start_loss = float(chunk[0])
        end_loss = float(chunk[-1])
        # Absolute drop across the band (positive => improved).
        drop = start_loss - end_loss
        per_step = drop / max(1, end - start)
        rows.append(
            {
                "start": start + 1,
                "end": end,
                "start_loss": start_loss,
                "end_loss": end_loss,
                "mean_loss": sum(chunk) / len(chunk),
                "drop": drop,
                "per_step": per_step,
            }
        )
    return rows


def _mean_drop_per_step(curve: list[float]) -> float:
    if len(curve) < 2:
        return 0.0
    return (float(curve[0]) - float(curve[-1])) / float(len(curve) - 1)


def _write_comparison(out_dir: Path, baseline: dict[str, Any], online: dict[str, Any] | None) -> str:
    """Compare learning by epochs/trials and convergence-per-step, not wall time."""
    b_hits = _threshold_table(list(baseline.get("threshold_hits") or []))
    b_loss = [float(x) for x in baseline.get("loss_history") or []]
    b_adv = _advancements(b_loss)
    b_bands = _level_bands(b_loss, step_size=25)
    b_per_epoch = _mean_drop_per_step(b_loss)

    lines = [
        "# Burgers Base (~10K) Epoch-Level Convergence Comparison",
        "",
        "Metric: **epochs / trials** and **loss drop per epoch-level step**, not wall-clock time.",
        "One Adam epoch = one full pass over the train set. One online trial = one tournament slice update.",
        "",
        "## Adam baseline (SiLU MLP)",
        "",
        f"- Parameters: `{baseline['parameter_count']}`",
        f"- Train MSE: `{baseline['train_mse_initial']:.6f}` -> best `{baseline['train_mse_best']:.6f}` "
        f"(final `{baseline['train_mse_final']:.6f}`)",
        f"- Extrapolation MSE final: `{baseline['extrap_mse_final']:.6f}`",
        f"- PDE residual: `{baseline['mean_pde_residual']:.6f}`",
        f"- Epochs to best / stop: `{baseline['best_epoch']}` / `{baseline['epochs_completed']}`",
        f"- Converged: `{baseline['converged']}` (`{baseline['stop_reason']}`)",
        f"- Mean train-MSE drop per epoch: `{b_per_epoch:.6f}`",
        f"- New-best advancements: `{len(b_adv)}` over `{baseline['epochs_completed']}` epochs "
        f"({(len(b_adv) / max(1, int(baseline['epochs_completed']))):.3f} advances/epoch)",
        "",
        "### Epochs to train-MSE convergence levels",
        "",
        "| Level (train MSE) | First epoch hit | Epochs from start |",
        "| ---: | ---: | ---: |",
    ]
    for hit in b_hits:
        ep = int(hit.get("epoch", -1))
        ep_s = "never" if ep < 0 else str(ep)
        from_s = "never" if ep < 0 else str(ep)
        lines.append(f"| <= {float(hit['threshold']):.2f} | {ep_s} | {from_s} |")

    lines.extend(
        [
            "",
            "### Convergence per epoch band (25-epoch levels)",
            "",
            "| Epochs | Start MSE | End MSE | Band drop | Drop / epoch |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for band in b_bands:
        lines.append(
            f"| {band['start']}-{band['end']} | {band['start_loss']:.6f} | {band['end_loss']:.6f} | "
            f"{band['drop']:+.6f} | {band['per_step']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "### Condensed new-best trail (epoch-level)",
            "",
            "| Epoch | Best train MSE | Delta |",
            "| ---: | ---: | ---: |",
        ]
    )
    show = b_adv
    if len(show) > 24:
        head, mid, tail = show[:8], show[8:-8], show[-8:]
        stride = max(1, len(mid) // 8)
        show = head + mid[::stride][:8] + tail
    for row in show:
        lines.append(f"| {row['step']} | {row['best_loss']:.6f} | {row['delta']:+.6f} |")

    if online:
        trials = list(online.get("trials") or [])
        # Epoch-analog curve: best loss after each trial (seed = step 0).
        o_best_curve = [float(online["initial_loss"])]
        o_trial_loss = [float(online["initial_loss"])]
        labels = ["seed"]
        for t in trials:
            o_best_curve.append(float(t["best_loss"]))
            o_trial_loss.append(float(t["loss"]))
            labels.append(f"t{int(t['trial'])}")
        o_adv = _advancements(o_best_curve, labels)
        o_bands = _level_bands(o_best_curve[1:], step_size=5)  # bands over trials 1..N
        # Prefer best-curve drop per trial for convergence rate.
        o_per_trial = _mean_drop_per_step(o_best_curve)
        commits = sum(1 for t in trials if t.get("status") == "committed_natively")

        lines.extend(
            [
                "",
                "## Online tournament (1 trial = 1 epoch-level update)",
                "",
                f"- Train MSE: `{online['initial_loss']:.6f}` -> best `{online['best_loss']:.6f}` "
                f"({online['loss_improvement_pct']:+.2f}%)",
                f"- Trials: `{online['trials_run']}` | commits `{commits}` | rejected `{online['rejected']}`",
                f"- Mean best-MSE drop per trial: `{o_per_trial:.6f}`",
                f"- New-best advancements: `{max(0, len(o_adv) - 1)}` over `{online['trials_run']}` trials "
                f"({(max(0, len(o_adv) - 1) / max(1, int(online['trials_run']))):.3f} advances/trial)",
                "",
                "### Trials to train-MSE convergence levels",
                "",
                "| Level (train MSE) | First trial with best<=level |",
                "| ---: | ---: |",
            ]
        )
        for thr in THRESHOLDS:
            hit_trial = -1
            running = float(online["initial_loss"])
            if running <= thr:
                hit_trial = 0
            else:
                for t in trials:
                    running = min(running, float(t["best_loss"]))
                    if running <= thr:
                        hit_trial = int(t["trial"])
                        break
            lines.append(f"| <= {thr:.2f} | {'never' if hit_trial < 0 else hit_trial} |")

        lines.extend(
            [
                "",
                "### Convergence per trial band (5-trial levels)",
                "",
                "| Trials | Start best MSE | End best MSE | Band drop | Drop / trial |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for band in o_bands:
            lines.append(
                f"| {band['start']}-{band['end']} | {band['start_loss']:.6f} | {band['end_loss']:.6f} | "
                f"{band['drop']:+.6f} | {band['per_step']:+.6f} |"
            )

        lines.extend(
            [
                "",
                "### Condensed trial log (commits + checkpoints)",
                "",
                "| Trial | Slice | Status | Loss | Best | Delta vs prior best |",
                "| ---: | :--- | :--- | ---: | ---: | ---: |",
            ]
        )
        for t in trials:
            keep = (
                t.get("status") == "committed_natively"
                or int(t["trial"]) == 1
                or int(t["trial"]) % 5 == 0
                or int(t["trial"]) == len(trials)
            )
            if not keep:
                continue
            lines.append(
                f"| {t['trial']} | `{t['slice']}` | `{t['status']}` | {float(t['loss']):.6f} | "
                f"{float(t['best_loss']):.6f} | {float(t['delta']):+.6f} |"
            )

        lines.extend(
            [
                "",
                "## Head-to-head (epoch-level effort only)",
                "",
                "| Convergence level | Adam epochs | Online trials | Winner (fewer steps) |",
                "| ---: | ---: | ---: | :--- |",
            ]
        )
        for hit in b_hits:
            thr = float(hit["threshold"])
            adam_ep = int(hit.get("epoch", -1))
            online_trial = -1
            running = float(online["initial_loss"])
            if running <= thr:
                online_trial = 0
            else:
                for t in trials:
                    running = min(running, float(t["best_loss"]))
                    if running <= thr:
                        online_trial = int(t["trial"])
                        break
            adam_s = "never" if adam_ep < 0 else str(adam_ep)
            on_s = "never" if online_trial < 0 else str(online_trial)
            if adam_ep < 0 and online_trial < 0:
                winner = "neither"
            elif adam_ep < 0:
                winner = "online"
            elif online_trial < 0:
                winner = "adam"
            elif adam_ep < online_trial:
                winner = "adam"
            elif online_trial < adam_ep:
                winner = "online"
            else:
                winner = "tie"
            lines.append(f"| <= {thr:.2f} | {adam_s} | {on_s} | {winner} |")

        lines.extend(
            [
                "",
                "## Convergence rate (per epoch-level step)",
                "",
                f"- Adam mean drop / epoch: `{b_per_epoch:.6f}` over `{len(b_loss)}` epochs",
                f"- Online mean drop / trial: `{o_per_trial:.6f}` over `{len(o_best_curve) - 1}` trials",
                f"- Advances / epoch (Adam): `{(len(b_adv) / max(1, int(baseline['epochs_completed']))):.3f}`",
                f"- Advances / trial (Online): `{(max(0, len(o_adv) - 1) / max(1, int(online['trials_run']))):.3f}`",
                "",
                "## Verdict (epoch/trial units)",
                "",
                f"- Adam reached best train MSE `{baseline['train_mse_best']:.6f}` in `{baseline['best_epoch']}` epochs.",
                f"- Online reached best train MSE `{online['best_loss']:.6f}` in `{online['trials_run']}` trials.",
            ]
        )
        online_best = float(online["best_loss"])
        adam_best = float(baseline["train_mse_best"])
        if online_best <= 0.01:
            lines.append("- Online matched Adam's 0.01 convergence level within the trial budget.")
        elif online_best <= 0.02:
            lines.append("- Online reached the 0.02 band but not 0.01 within the matched trial/epoch budget.")
        elif online_best <= 0.05:
            lines.append(
                f"- Online improved into the 0.05 band (best `{online_best:.4f}`) but did not reach 0.02/0.01 "
                "within the matched trial/epoch budget."
            )
        else:
            lines.append(
                f"- Online best `{online_best:.4f}` stayed above 0.05; Adam still converges further "
                f"(to `{adam_best:.4f}`)."
            )
        lines.append("")

    text = "\n".join(lines) + "\n"
    (out_dir / "comparison_report.md").write_text(text, encoding="utf-8")
    (out_dir / "comparison_report_epochs.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Burgers learning-speed comparison runner")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument(
        "--online-trials",
        type=int,
        default=205,
        help="Online tournament trials (default 205 to match Adam epochs-to-0.01).",
    )
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.01,
        help="Early-stop when best committed train MSE <= this (default 0.01, Adam convergence level).",
    )
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
        # Empirically ~1.8k tokens/trial on burger_base; gpt-5-mini is cheap (~$0.02–0.05/trial).
        est_per_trial = 0.04
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
