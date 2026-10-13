"""Ten-trial tournament of the regressor and the autoencoder.

Starts the local Functions host when port 7071 is closed, seeds a fresh checkpoint
per model, and polls each orchestration to completion. Completions have no token cap.
A trial is committed only when the three candidates change the slice.
Gamma bands and journal fields come from the model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "http://localhost:7071"
STARTER = f"{HOST}/api/orchestrators/modular_model_orchestrator"
PORT = 7071
ARM_TIMEOUT_S = 12 * 60

ARMS: list[dict[str, Any]] = [
    {
        "name": "mlp_regressor",
        "factory": "MlpRegressor",
        "int_bound": 25,
    },
    {
        "name": "bottleneck_ae",
        "factory": "BottleneckAutoencoder",
        "int_bound": 25,
    },
]


def _listening_pids(port: int) -> list[int]:
    output = subprocess.check_output(["netstat", "-ano"], text=True, errors="replace")
    found: list[int] = []
    needle = f":{port}"
    for line in output.splitlines():
        if needle not in line or "LISTENING" not in line:
            continue
        pid = int(line.split()[-1])
        if pid != os.getpid() and pid not in found:
            found.append(pid)
    return found


def _stop_listener() -> None:
    for pid in _listening_pids(PORT):
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)


def _host_ready(log_text: str) -> bool:
    return (
        "modular_model_orchestrator: orchestrationTrigger" in log_text
        and "Host lock lease acquired" in log_text
    )


def _start_host() -> tuple[subprocess.Popen[str], Any]:
    _stop_listener()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / ".venv" / "Lib" / "site-packages")
    log_handle = open(ROOT / "scripts" / "shootout_host.log", "w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        "func start",
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        shell=True,
    )
    log_path = ROOT / "scripts" / "shootout_host.log"
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"Functions host exited early with code {process.returncode}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if _host_ready(text):
            return process, log_handle
        time.sleep(1)
    raise SystemExit("Functions host did not list modular_model_orchestrator")


def _stop_host(process: subprocess.Popen[str] | None) -> None:
    if process is not None and process.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], check=False, capture_output=True)
    _stop_listener()


def _request(url: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    loaded = json.loads(raw) if raw else {}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _seed(factory_name: str, checkpoint: str, journal: str) -> float:
    from tools import tensor_engine

    factory = getattr(tensor_engine, factory_name)
    receipt = factory().seed_to_storage(checkpoint, journal, 10)
    return float(receipt["loss"])


def _poll(status_url: str) -> dict[str, Any]:
    deadline = time.time() + ARM_TIMEOUT_S
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = _request(status_url)
        runtime = str(last.get("runtimeStatus") or "")
        if runtime in {"Completed", "Failed", "Terminated"}:
            return last
        time.sleep(3)
    last["runtimeStatus"] = last.get("runtimeStatus") or "TimedOut"
    return last


def _history(status: dict[str, Any]) -> list[dict[str, Any]]:
    output = status.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = {}
    if not isinstance(output, dict):
        return []
    rows = output.get("history") or []
    return [row for row in rows if isinstance(row, dict)]


def _output_dict(status: dict[str, Any]) -> dict[str, Any]:
    output = status.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return {"error": output}
    return output if isinstance(output, dict) else {}


def _run_arm(arm: dict[str, Any]) -> dict[str, Any]:
    token = uuid.uuid4().hex[:8]
    slug = arm["name"].lower().replace("-", "_")
    checkpoint = f"bench_{slug}_{token}.json"
    journal = f"journal_{slug}_{token}.json"
    initial = _seed(arm["factory"], checkpoint, journal)
    started = _request(
        STARTER,
        {
            "max_trials": 10,
            "target_loss": 0.01,
            "checkpoint_blob": checkpoint,
            "journal_blob": journal,
            "journal_capacity": 10,
            "rolling_config": {"visible_history_count": 3},
            "training_params": {
                "arm_name": arm["name"],
                "int_bound": arm["int_bound"],
            },
        },
    )
    status_url = str(started.get("statusQueryGetUri") or "")
    if not status_url:
        raise SystemExit(f"{arm['name']} did not return a status url")
    print(f"started {arm['name']} instance={started.get('id')} blob={checkpoint}", flush=True)
    status = _poll(status_url)
    output = _output_dict(status)
    rows = _history(status)
    runtime = str(status.get("runtimeStatus") or "")
    final = float(output.get("best_loss", initial)) if runtime == "Completed" else initial
    commits = sum(1 for row in rows if row.get("status") == "committed_natively")
    diverged = sum(1 for row in rows if row.get("status") == "rejected_divergent")
    wins = {"c1": 0, "c2": 0, "c3": 0}
    for row in rows:
        if row.get("status") != "committed_natively":
            continue
        winner = str(row.get("winner") or "")
        if winner in wins:
            wins[winner] += 1
    token_total = sum(int(row.get("total_tokens") or 0) for row in rows)
    trial_count = len(rows) or int(output.get("trials") or 0)
    return {
        "name": arm["name"],
        "runtime": runtime,
        "initial": float(output.get("initial_loss", initial)) if runtime == "Completed" else initial,
        "final": final,
        "commits": commits,
        "diverged": diverged,
        "wins": wins,
        "token_total": token_total,
        "trial_count": trial_count,
        "rows": rows,
        "error": "" if runtime == "Completed" else str(output.get("error") or status.get("output") or runtime)[:240],
    }


def _redact(text: str) -> str:
    settings = ROOT / "local.settings.json"
    if settings.exists():
        values = json.loads(settings.read_text(encoding="utf-8")).get("Values") or {}
        secret = str(values.get("AZURE_OPENAI_API_KEY") or "")
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _pct(initial: float, final: float) -> float:
    if initial == 0:
        return 0.0
    return (initial - final) / initial * 100.0


def _scoreboard(results: list[dict[str, Any]]) -> str:
    lines = [
        "========================================================================================",
        "TOURNAMENT RESULTS (10 TRIALS: mlp_regressor, bottleneck_ae)",
        "========================================================================================",
        f"{'Arm Name':<18} {'L_init':>8} {'L_final':>8} {'Delta L (%)':>16} {'Commits':>9} {'Diverged':>10} {'Win (c1/c2/c3)':>16} {'Avg Tokens':>12}",
        "----------------------------------------------------------------------------------------",
    ]
    for row in results:
        if row["runtime"] != "Completed":
            lines.append(f"{row['name']:<18} FAILED {_redact(row['error'])}")
            continue
        delta = row["initial"] - row["final"]
        pct = _pct(row["initial"], row["final"])
        trials = row["trial_count"] or 10
        avg = (row["token_total"] / trials) if trials else 0.0
        wins = row["wins"]
        lines.append(
            f"{row['name']:<18} {row['initial']:8.4f} {row['final']:8.4f} "
            f"{delta:+7.4f} ({pct:+5.1f}%) {row['commits']:>3}/{trials:<3} {row['diverged']:>6}/{trials:<3} "
            f"({wins['c1']} / {wins['c2']} / {wins['c3']}) {avg:11.1f}"
        )
    lines.append("========================================================================================")
    return "\n".join(lines)


def _trial_report(results: list[dict[str, Any]]) -> str:
    lines = ["", "PER-TRIAL TOKENS (uncapped completion)", "Arm                Trial  Prompt  Completion   Total  Status               Winner  Loss"]
    for result in results:
        if not result["rows"]:
            lines.append(f"{result['name']:<18} no trials ({result['runtime']})")
            continue
        for row in result["rows"]:
            lines.append(
                f"{result['name']:<18} {int(row.get('trial') or 0):5d} "
                f"{int(row.get('prompt_tokens') or 0):7d} {int(row.get('completion_tokens') or 0):11d} "
                f"{int(row.get('total_tokens') or 0):7d}  {str(row.get('status') or ''):<20} "
                f"{str(row.get('winner') or ''):<6}  {float(row.get('loss') or 0.0):.4f}"
            )
        lines.append(
            f"{result['name']:<18} TOTAL {result['token_total']:7d}  avg "
            f"{(result['token_total'] / result['trial_count']) if result['trial_count'] else 0.0:.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    if not _listening_pids(10000):
        raise SystemExit("Azurite is not listening on port 10000")
    process, log_handle = _start_host()
    results: list[dict[str, Any]] = []
    try:
        for arm in ARMS:
            try:
                results.append(_run_arm(arm))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                results.append(
                    {
                        "name": arm["name"],
                        "runtime": "Failed",
                        "initial": 0.0,
                        "final": 0.0,
                        "commits": 0,
                        "diverged": 0,
                        "wins": {"c1": 0, "c2": 0, "c3": 0},
                        "token_total": 0,
                        "trial_count": 0,
                        "rows": [],
                        "error": str(exc)[:240],
                    }
                )
        board = _scoreboard(results)
        trials = _trial_report(results)
        print(board, flush=True)
        print(trials, flush=True)
        from orchestrators.run_report import publish_report

        publish_report("tournament shootout", board + "\n" + trials)
        (ROOT / "reports").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports" / "shootout_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        _stop_host(process)
        log_handle.close()


if __name__ == "__main__":
    main()
