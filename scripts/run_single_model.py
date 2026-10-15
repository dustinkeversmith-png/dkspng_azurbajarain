"""Run a single online model through tournament optimization with configurable settings and full artifact output.

Supports running any canonical online model (ForwardAnalyzedModel, MlpRegressor, BottleneckAutoencoder, DemoScaleModel)
for a configurable number of trials, per-learning LearningConfig overrides, and writes a complete training artifact
bundle (run_summary.json, initial/final checkpoints, journal.json, compiled prompts, raw LLM completions, trial metrics
CSV, and a comprehensive Markdown report with token usage and forward_analysis diagnostics).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrators.run_report import append_attempt, publish_report
from orchestrators.tournament_evaluator import collect_tournament, completion_context
from tools import tensor_engine

MODEL_FACTORIES: dict[str, str] = {
    "forward_analyzed_mlp": "ForwardAnalyzedModel",
    "forward_analyzed": "ForwardAnalyzedModel",
    "ForwardAnalyzedModel": "ForwardAnalyzedModel",
    "mlp_regressor": "MlpRegressor",
    "MlpRegressor": "MlpRegressor",
    "bottleneck_ae": "BottleneckAutoencoder",
    "BottleneckAutoencoder": "BottleneckAutoencoder",
    "demo_scale": "DemoScaleModel",
    "DemoScaleModel": "DemoScaleModel",
}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


def _load_settings() -> None:
    settings_path = ROOT / "local.settings.json"
    if not settings_path.exists():
        return
    values = json.loads(settings_path.read_text(encoding="utf-8")).get("Values") or {}
    for key, value in values.items():
        os.environ.setdefault(str(key), str(value))


def _resolve_factory(model_arg: str) -> tuple[str, Any]:
    factory_name = MODEL_FACTORIES.get(model_arg, model_arg)
    if not hasattr(tensor_engine, factory_name):
        available = sorted(set(MODEL_FACTORIES.values()))
        raise SystemExit(f"Unknown model '{model_arg}'. Available models: {', '.join(available)}")
    return factory_name, getattr(tensor_engine, factory_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single online model with configurable trials and full training artifact output."
    )
    parser.add_argument(
        "--model",
        default="forward_analyzed_mlp",
        help="Model architecture or class name (default: forward_analyzed_mlp).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Maximum number of tournament slice trials to run (default: 5).",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.01,
        help="Stop early when best committed loss <= target-loss (default: 0.01).",
    )
    parser.add_argument(
        "--journal-capacity",
        type=int,
        default=10,
        help="Sliding journal ring buffer capacity (default: 10).",
    )
    parser.add_argument(
        "--visible-history",
        type=int,
        default=3,
        help="Number of journal entries rendered into each compiled prompt (default: 3).",
    )
    parser.add_argument(
        "--int-bound",
        type=int,
        default=25,
        help="Integer code bound advertised in prompt constraints (default: 25).",
    )
    parser.add_argument(
        "--min-commit-drop",
        type=float,
        default=1e-4,
        help="Minimum loss improvement required to commit a checkpoint (default: 1e-4).",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=100.0,
        help="Loss threshold above which a candidate is rejected as divergent (default: 100.0).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "reports" / "single_model_runs"),
        help="Root directory where the timestamped training artifact bundle is written.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional custom name suffix for the artifact folder.",
    )
    parser.add_argument(
        "--deployment",
        default="",
        help="Optional Azure OpenAI deployment override (defaults to AZURE_OPENAI_DEPLOYMENT).",
    )
    return parser.parse_args()


def _build_markdown_report(summary: dict[str, Any]) -> str:
    cfg = summary["learning_config"]
    tokens = summary["token_usage"]
    lines = [
        f"# Single Model Training Artifact Report: `{summary['architecture']}` (`{summary['factory']}`)",
        "",
        "## 1. Run Settings & Per-Learning Configuration",
        "",
        f"- **Run ID**: `{summary['run_id']}`",
        f"- **Timestamp (UTC)**: `{summary['timestamp_utc']}`",
        f"- **Architecture**: `{summary['architecture']}` (`{summary['factory']}`)",
        f"- **Deployment**: `{summary['deployment']}`",
        f"- **Checkpoint Blob**: `{summary['checkpoint_blob']}`",
        f"- **Journal Blob**: `{summary['journal_blob']}`",
        f"- **Configured Trials**: `{summary['configured_trials']}` (executed: `{summary['trials_run']}`)",
        f"- **Target Loss**: `{summary['target_loss']:.4f}`",
        f"- **Journal Capacity**: `{summary['journal_capacity']}`",
        f"- **Visible History Count**: `{cfg['visible_history_count']}`",
        f"- **Integer Code Bound (`int_bound`)**: `{cfg['int_bound']}`",
        f"- **Min Commit Drop**: `{cfg['min_commit_drop']}`",
        f"- **Divergence Threshold**: `{cfg['divergence_threshold']}`",
        f"- **Elapsed Wall Time**: `{summary['elapsed_seconds']:.2f}s`",
        "",
        "## 2. Training Performance & Token Usage Summary",
        "",
        f"- **Initial Seed Loss**: `{summary['initial_loss']:.4f}`",
        f"- **Final Best Loss**: `{summary['best_loss']:.4f}`",
        f"- **Absolute Loss Delta**: `{summary['loss_delta']:+.4f}` (`{summary['loss_improvement_pct']:+.2f}%`)",
        f"- **Commits**: `{summary['commits']} / {summary['trials_run']}`",
        f"- **Rejected Trials**: `{summary['rejected']} / {summary['trials_run']}`",
        f"- **Diverged Trials**: `{summary['diverged']} / {summary['trials_run']}`",
        f"- **Mutation Failures**: `{summary['mutation_failed']} / {summary['trials_run']}`",
        f"- **Winning Slot Breakdown**: `{json.dumps(summary['winner_counts'])}`",
        f"- **Total Prompt Tokens**: `{tokens['prompt_tokens']}` (avg `{tokens['avg_prompt_tokens_per_trial']:.1f}` / trial)",
        f"- **Total Completion Tokens**: `{tokens['completion_tokens']}` (avg `{tokens['avg_completion_tokens_per_trial']:.1f}` / trial)",
        f"- **Total Billed Tokens**: `{tokens['total_tokens']}` (avg `{tokens['avg_total_tokens_per_trial']:.1f}` / trial)",
        "",
    ]

    if summary.get("initial_forward_analysis"):
        fa_init = summary["initial_forward_analysis"]
        fa_final = summary.get("final_forward_analysis") or {}
        lines.extend(
            [
                "## 3. Offline Forward-Analysis Diagnostics (Seed vs Final Checkpoint)",
                "",
                "### Initial Seed Forward Analysis",
                f"- **Summary Token**: `{fa_init.get('summary_token', '')}`",
                f"- **Dead Units (L1 / L2 / Total)**: `{fa_init.get('dead_units_l1', 0)} / {fa_init.get('dead_units_l2', 0)} / {fa_init.get('total_dead_units', 0)}`",
                f"- **Numerical Ranks (L1 / L2)**: `{fa_init.get('numerical_rank_l1', 0)} / {fa_init.get('numerical_rank_l2', 0)}` (Mean Effective Rank: `{fa_init.get('mean_effective_rank', 0.0):.3f}`)",
                f"- **Target Correlation**: `{fa_init.get('target_correlation', 0.0):.4f}` | **Sign Agreement**: `{fa_init.get('sign_agreement_ratio', 0.0):.4f}` | **Range Ratio**: `{fa_init.get('range_ratio', 0.0):.4f}`",
                f"- **Pathway Activity**: {fa_init.get('pathway_activity', '')}",
                f"- **Feature Sensitivity**: {fa_init.get('feature_sensitivity', '')}",
                f"- **Representational Structure**: {fa_init.get('representational_structure', '')}",
                f"- **Functional Mapping**: {fa_init.get('functional_mapping', '')}",
                "",
                "### Final Journal Forward Analysis",
                f"- **Summary Token**: `{fa_final.get('fa_summary', '')}`",
                f"- **Dead Units**: `{fa_final.get('fa_dead_units', 0.0)}` | **Mean Effective Rank**: `{fa_final.get('fa_effective_rank', 0.0)}`",
                f"- **Target Correlation**: `{fa_final.get('fa_target_corr', 0.0):.4f}` | **Sign Agreement**: `{fa_final.get('fa_sign_agreement', 0.0):.4f}`",
                f"- **Pathway Activity**: {fa_final.get('fa_pathway', '')}",
                f"- **Functional Mapping**: {fa_final.get('fa_mapping', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Per-Trial Trajectory & Token Breakdown",
            "",
            "| Trial | Slice | Attempts | Prompt Tok | Comp Tok | Total Tok | Status | Winner | Step | Loss | Delta |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: |",
        ]
    )
    for row in summary["trials"]:
        lines.append(
            f"| {row['trial']} | `{row['slice']}` | {row['attempts']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['total_tokens']} | `{row['status']}` | "
            f"`{row['winner'] or '-'}` | `{row['winner_step'] or '-'}` | {row['loss']:.4f} | {row['delta']:+.4f} |"
        )

    lines.extend(["", "## 5. Artifact Files Written", ""])
    for name, path in summary["artifact_files"].items():
        lines.append(f"- **{name}**: `{path}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    _load_settings()

    factory_name, factory_cls = _resolve_factory(args.model)
    model = factory_cls()

    cfg = model.config
    cfg.visible_history_count = args.visible_history
    cfg.int_bound = args.int_bound
    cfg.min_commit_drop = args.min_commit_drop
    cfg.divergence_threshold = args.divergence_threshold
    model.set_config(cfg)

    run_token = uuid.uuid4().hex[:8]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_blob = f"single_{factory_name.lower()}_{run_token}.json"
    journal_blob = f"journal_{factory_name.lower()}_{run_token}.json"

    seeded = _plain(model.seed_to_storage(checkpoint_blob, journal_blob, args.journal_capacity))
    architecture = str(seeded.get("architecture") or args.model)
    slug = args.run_name.strip() or f"{timestamp}_{architecture}_{run_token}"
    artifact_dir = Path(args.out_dir).resolve() / slug
    prompts_dir = artifact_dir / "prompts"
    completions_dir = artifact_dir / "completions"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    completions_dir.mkdir(parents=True, exist_ok=True)

    initial_fa: dict[str, Any] = {}
    if hasattr(model, "analyze_network"):
        initial_fa = _plain(model.analyze_network())

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    deployment = args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    meta = _plain(tensor_engine.read_architecture_meta(checkpoint_blob, journal_blob))
    layer_names = list(meta.get("layers") or [])
    initial_loss = float(seeded.get("loss", meta.get("loss", 999.0)))
    best_loss = initial_loss

    print(
        f"[start] model={architecture} ({factory_name}) blob={checkpoint_blob} "
        f"initial_loss={initial_loss:.4f} slices={layer_names} trials={args.trials}",
        flush=True,
    )

    started_wall = time.time()
    trial_rows: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []

    for trial_idx in range(1, args.trials + 1):
        if best_loss <= args.target_loss or not layer_names:
            break
        active_slice = layer_names[(trial_idx - 1) % len(layer_names)]
        slice_state = _plain(tensor_engine.read_architecture_slice(checkpoint_blob, journal_blob, active_slice))
        current_loss = float(slice_state.get("loss", best_loss))
        context = completion_context(slice_state)

        trial_prompts: list[str] = []
        trial_attempts: list[dict[str, Any]] = []

        def compile_prompt(note: str) -> str:
            compiled = tensor_engine.compile_prompt(
                checkpoint_blob,
                journal_blob,
                active_slice,
                args.visible_history,
                args.int_bound,
                note,
            )
            attempt_no = len(trial_prompts) + 1
            prompt_path = prompts_dir / f"trial_{trial_idx:02d}_attempt_{attempt_no}_{active_slice.replace('@', '_')}.txt"
            prompt_path.write_text(compiled, encoding="utf-8")
            trial_prompts.append(str(prompt_path))
            return compiled

        def complete(prompt_text: str) -> tuple[str, int, int, int]:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                reasoning_effort="none",
            )
            usage = response.usage
            content = response.choices[0].message.content or ""
            p_tok = int(usage.prompt_tokens) if usage is not None else 0
            c_tok = int(usage.completion_tokens) if usage is not None else 0
            t_tok = int(usage.total_tokens) if usage is not None else 0
            return content, p_tok, c_tok, t_tok

        def commit(candidates: list[dict[str, str]]) -> dict[str, Any]:
            return _plain(
                tensor_engine.commit_tournament(
                    active_slice,
                    candidates,
                    checkpoint_blob,
                    journal_blob,
                    args.journal_capacity,
                )
            )

        def on_attempt(record: dict[str, Any]) -> None:
            enriched = {
                "trial": trial_idx,
                "slice": active_slice,
                **record,
            }
            trial_attempts.append(enriched)
            all_attempts.append(enriched)
            comp_path = (
                completions_dir
                / f"trial_{trial_idx:02d}_attempt_{record['attempt']}_{active_slice.replace('@', '_')}.json"
            )
            comp_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            append_attempt(
                {
                    "source": "run_single_model",
                    "architecture": architecture,
                    "layer": active_slice,
                    **enriched,
                }
            )

        synthesized = collect_tournament(compile_prompt, complete, commit, context, on_attempt)
        receipt = _plain(synthesized.get("receipt") or {})
        status = "mutation_failed" if synthesized.get("mutation_failed") else str(receipt.get("status") or "rejected")
        loss = float(receipt.get("loss", current_loss)) if not synthesized.get("mutation_failed") else current_loss
        delta = float(receipt.get("delta", loss - best_loss)) if not synthesized.get("mutation_failed") else 0.0
        winner = str(receipt.get("winner") or "") if not synthesized.get("mutation_failed") else ""
        winner_step = str(receipt.get("winner_step") or "") if not synthesized.get("mutation_failed") else ""

        if status == "committed_natively":
            best_loss = min(best_loss, loss)

        row = {
            "trial": trial_idx,
            "slice": active_slice,
            "attempts": int(synthesized.get("attempts") or len(trial_attempts)),
            "status": status,
            "winner": winner,
            "winner_step": winner_step,
            "loss": loss,
            "best_loss": best_loss,
            "delta": delta,
            "checksum": str(receipt.get("checksum") or seeded.get("checksum") or ""),
            "probe": context.get("probe", {}).get("token", ""),
            "reasoning": str(synthesized.get("reasoning") or ""),
            "confidence": synthesized.get("confidence", 0.5),
            "prompt_tokens": int(synthesized.get("prompt_tokens") or 0),
            "completion_tokens": int(synthesized.get("completion_tokens") or 0),
            "total_tokens": int(synthesized.get("total_tokens") or 0),
            "candidates": _plain(receipt.get("candidates") or []),
            "prompt_files": trial_prompts,
            "error": str(synthesized.get("error") or ""),
        }
        trial_rows.append(row)
        print(
            f"[trial {trial_idx:02d}/{args.trials:02d}] slice={active_slice:<12} "
            f"status={status:<20} winner={winner or '-':<4} loss={loss:8.4f} "
            f"best={best_loss:8.4f} delta={delta:+8.4f} tokens={row['total_tokens']} "
            f"(p={row['prompt_tokens']}, c={row['completion_tokens']})",
            flush=True,
        )

    elapsed_s = time.time() - started_wall

    final_checkpoint = _plain(tensor_engine.read_state_native(checkpoint_blob, journal_blob))
    final_meta = _plain(tensor_engine.read_architecture_meta(checkpoint_blob, journal_blob))
    journal_text = tensor_engine.read_journal_text(journal_blob)
    journal_json = json.loads(journal_text) if journal_text.strip() else {}
    final_fa: dict[str, Any] = {}
    if journal_json.get("history"):
        latest_metrics = {
            m["name"]: m["value"]
            for m in (journal_json["history"][-1].get("metrics") or [])
            if isinstance(m, dict) and "name" in m
        }
        final_fa = {k: v for k, v in latest_metrics.items() if k.startswith("fa_")}

    prompt_total = sum(r["prompt_tokens"] for r in trial_rows)
    comp_total = sum(r["completion_tokens"] for r in trial_rows)
    billed_total = sum(r["total_tokens"] for r in trial_rows)
    n_trials = len(trial_rows)

    winner_counts: dict[str, int] = {}
    for r in trial_rows:
        if r["status"] == "committed_natively" and r["winner"]:
            winner_counts[r["winner"]] = winner_counts.get(r["winner"], 0) + 1

    seed_ckpt_path = artifact_dir / "seed_checkpoint.json"
    final_ckpt_path = artifact_dir / "final_checkpoint.json"
    journal_path = artifact_dir / "journal.json"
    csv_path = artifact_dir / "trials.csv"
    summary_path = artifact_dir / "run_summary.json"
    report_path = artifact_dir / "report.md"

    seed_ckpt_path.write_text(json.dumps(seeded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final_ckpt_path.write_text(
        json.dumps({"checkpoint": final_checkpoint, "meta": final_meta}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    journal_path.write_text(
        journal_text if journal_text.endswith("\n") else journal_text + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "trial",
                "slice",
                "attempts",
                "status",
                "winner",
                "winner_step",
                "loss",
                "best_loss",
                "delta",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "checksum",
            ]
        )
        for r in trial_rows:
            writer.writerow(
                [
                    r["trial"],
                    r["slice"],
                    r["attempts"],
                    r["status"],
                    r["winner"],
                    r["winner_step"],
                    f"{r['loss']:.6f}",
                    f"{r['best_loss']:.6f}",
                    f"{r['delta']:.6f}",
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["total_tokens"],
                    r["checksum"],
                ]
            )

    loss_delta = initial_loss - best_loss
    loss_pct = (loss_delta / initial_loss * 100.0) if initial_loss != 0.0 else 0.0

    summary: dict[str, Any] = {
        "run_id": slug,
        "timestamp_utc": timestamp,
        "architecture": architecture,
        "factory": factory_name,
        "deployment": deployment,
        "checkpoint_blob": checkpoint_blob,
        "journal_blob": journal_blob,
        "configured_trials": args.trials,
        "trials_run": n_trials,
        "target_loss": args.target_loss,
        "journal_capacity": args.journal_capacity,
        "elapsed_seconds": elapsed_s,
        "learning_config": _plain(model.config.to_dict()),
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "loss_delta": loss_delta,
        "loss_improvement_pct": loss_pct,
        "commits": sum(1 for r in trial_rows if r["status"] == "committed_natively"),
        "rejected": sum(1 for r in trial_rows if r["status"] == "rejected"),
        "diverged": sum(1 for r in trial_rows if r["status"] == "rejected_divergent"),
        "mutation_failed": sum(1 for r in trial_rows if r["status"] == "mutation_failed"),
        "winner_counts": winner_counts,
        "token_usage": {
            "prompt_tokens": prompt_total,
            "completion_tokens": comp_total,
            "total_tokens": billed_total,
            "avg_prompt_tokens_per_trial": (prompt_total / n_trials) if n_trials else 0.0,
            "avg_completion_tokens_per_trial": (comp_total / n_trials) if n_trials else 0.0,
            "avg_total_tokens_per_trial": (billed_total / n_trials) if n_trials else 0.0,
        },
        "initial_forward_analysis": initial_fa,
        "final_forward_analysis": final_fa,
        "seed_checkpoint": seeded,
        "final_meta": final_meta,
        "journal": journal_json,
        "trials": trial_rows,
        "artifact_files": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "trials_csv": str(csv_path),
            "seed_checkpoint_json": str(seed_ckpt_path),
            "final_checkpoint_json": str(final_ckpt_path),
            "journal_json": str(journal_path),
            "prompts_dir": str(prompts_dir),
            "completions_dir": str(completions_dir),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_report = _build_markdown_report(summary)
    report_path.write_text(md_report, encoding="utf-8")
    publish_report(f"single model {architecture}", md_report)

    print(
        f"\n[done] architecture={architecture} initial_loss={initial_loss:.4f} best_loss={best_loss:.4f} "
        f"delta={loss_delta:+.4f} ({loss_pct:+.1f}%) commits={summary['commits']}/{n_trials} "
        f"total_tokens={billed_total} (avg={summary['token_usage']['avg_total_tokens_per_trial']:.1f}/trial)",
        flush=True,
    )
    print(f"[artifacts] {artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
