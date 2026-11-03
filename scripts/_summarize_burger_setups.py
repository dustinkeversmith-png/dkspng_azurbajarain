import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / "reports" / "burger_runs" / "20261102T224159Z_setups"
comp = json.loads((root / "burger_compact" / "burger_compact" / "run_summary.json").read_text(encoding="utf-8"))
br = json.loads((root / "burger_breakwall" / "burger_breakwall" / "run_summary.json").read_text(encoding="utf-8"))
base_path = ROOT / "reports" / "burger_runs" / "20261102T221709Z_setups" / "burger_base" / "burger_base" / "run_summary.json"
base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None


def hits(summary: dict) -> dict[float, int | None]:
    out: dict[float, int | None] = {}
    for thr in [0.10, 0.05, 0.026, 0.02, 0.01]:
        hit = None
        best = float(summary["initial_loss"])
        if best <= thr:
            hit = 0
        else:
            for trial in summary["trials"]:
                best = min(best, float(trial["best_loss"]))
                if best <= thr:
                    hit = int(trial["trial"])
                    break
        out[thr] = hit
    return out


rows = [("burger_base", base), ("burger_breakwall", br), ("burger_compact", comp)]
lines = [
    "# Burgers setup bakeoff (token + convergence)",
    "",
    "| Setup | Best MSE | Trials | Commits | Total tokens | Avg tok/trial | Prompt toks | Completion toks | <0.026 @ trial | <0.02 @ trial |",
    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name, summary in rows:
    if not summary:
        continue
    tok = summary["token_usage"]
    h = hits(summary)
    lines.append(
        f"| `{name}` | {summary['best_loss']:.6f} | {summary['trials_run']} | {summary['commits']} | "
        f"{tok['total_tokens']} | {tok['avg_total_tokens_per_trial']:.1f} | {tok['prompt_tokens']} | "
        f"{tok['completion_tokens']} | "
        f"{h[0.026] if h[0.026] is not None else 'never'} | "
        f"{h[0.02] if h[0.02] is not None else 'never'} |"
    )

lines.extend(
    [
        "",
        "## Setup differences",
        "",
        "- `burger_base`: 3 candidates, history=3, fixed gamma bands.",
        "- `burger_breakwall`: 2 candidates, history=1, phase-adaptive gamma, hotspot + PDE/extrap telemetry, min_commit_drop=1e-5.",
        "- `burger_compact`: same polish policy, shorter prompts (no PDE/extrap blocks), 2 candidates.",
        "",
        "## Finding",
        "",
        "- **Compact broke the prior ~0.026 wall** (best **0.0234** in 100 trials).",
        "- Breakwall stayed near the old mid-loss plateau (~0.068) in this bakeoff.",
        "- None reached 0.02 / 0.01 yet; compact is closest with similar total tokens to breakwall.",
        "",
    ]
)
text = "\n".join(lines)
out = root / "setup_comparison_final.md"
out.write_text(text, encoding="utf-8")
print(text)
print(f"wrote {out}")
