import glob
import json
import os
import signal
import socket
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List


def _ensure_output_dir(dataset: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, f"../temp/{dataset}")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    return out_dir


def _run_id() -> str:
    """Identifies one pipeline run so every stage in it (each its own `python` process) lands in
    the same timings file, while different runs get separate files instead of piling into one
    ever-growing shared timings.json. Under SLURM this is the job ID; agent_auditor.sh also exports
    AGENTAUDITOR_RUN_ID explicitly so stages still group together when run outside SLURM. Falls back
    to a per-process id so at minimum, ad-hoc/concurrent runs don't collide."""
    return (
        os.environ.get("AGENTAUDITOR_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOBID")
        or f"pid{os.getpid()}_{int(time.time())}"
    )


def _timing_file_path(dataset: str) -> str:
    out_dir = _ensure_output_dir(dataset)
    return os.path.join(out_dir, f"timings_{_run_id()}.json")


# HiPerGator's burst QOS preempts by killing and requeuing the whole sbatch job (SIGTERM, then a
# grace period, then SIGKILL) rather than suspending it in place - so there's no SIGSTOP/SIGCONT gap
# to subtract from wall time. The risk is instead that a preempted stage still manages to hit the
# `finally` block below within the grace period and gets recorded with a normal-looking (but
# truncated) duration. Catching SIGTERM here lets time_and_record() flag that entry as unreliable
# instead of silently trusting it.
_preempt_signal_state: Dict[str, Any] = {"received": False, "signal": None}


def _handle_termination_signal(signum, frame):
    _preempt_signal_state["received"] = True
    _preempt_signal_state["signal"] = signum
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _handle_termination_signal)


def _slurm_context() -> Dict[str, Any]:
    """Best-effort SLURM/HiPerGator job metadata, so latency entries can be correlated with
    burst-QOS requeues and node placement instead of trusting raw wall-clock time in isolation."""
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", 0) or 0),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }


def time_and_record(stage_name: str, func: Callable, dataset: str, *args, **kwargs) -> Any:
    """Run `func(*args, **kwargs)` while timing it and record metrics under
    `../temp/{dataset}/timings_<run_id>.json` (see `_run_id`).

    Returns the original function return value. On exception, records error details
    and re-raises the exception.
    """
    start_mono = time.monotonic()
    start_iso = datetime.utcnow().isoformat() + "Z"
    ctx = _slurm_context()
    _preempt_signal_state["received"] = False
    exc_info = None
    preempted = False
    success = False
    try:
        result = func(*args, **kwargs)
        success = True
        return result
    except SystemExit:
        preempted = True
        success = False
        raise
    except Exception as e:
        exc_info = str(e)
        success = False
        raise
    finally:
        end_iso = datetime.utcnow().isoformat() + "Z"
        duration = time.monotonic() - start_mono

        unreliable_reasons: List[str] = []
        if preempted or _preempt_signal_state["received"]:
            unreliable_reasons.append("sigterm_during_stage")
        if ctx["slurm_restart_count"] > 0:
            unreliable_reasons.append("job_requeued")

        timing_entry: Dict[str, Any] = {
            "stage": stage_name,
            "start": start_iso,
            "end": end_iso,
            "duration_seconds": duration,
            "success": success,
            "reliable": len(unreliable_reasons) == 0,
            "unreliable_reasons": unreliable_reasons,
            "slurm": ctx,
        }
        if exc_info is not None:
            timing_entry["error"] = exc_info

        # Load existing timings and update
        path = _timing_file_path(dataset)
        timings: Dict[str, Any] = {}
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    timings = json.load(f) or {}
        except Exception:
            timings = {}

        # store under stage_name with timestamp suffix if collision
        key = stage_name
        if key in timings:
            # ensure uniqueness
            suffix = 1
            while f"{key}_{suffix}" in timings:
                suffix += 1
            key = f"{key}_{suffix}"

        timings[key] = timing_entry

        try:
            # Write to a temp file and rename over the target rather than truncating it in place -
            # an abrupt SIGKILL (grace-period timeout, or a cgroup OOM-kill on a contended burst
            # node, neither of which SIGTERM handling below can catch) mid-write would otherwise
            # corrupt/truncate every previously recorded stage's history, not just this entry.
            tmp_path = f"{path}.tmp{os.getpid()}"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(timings, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort: do not let timing write errors mask the original error
            pass


def summarize_dataset_timings(dataset: str) -> Dict[str, Any]:
    """Aggregate per-stage latency stats across every `../temp/{dataset}/timings_*.json` run file.

    Entries that failed, or were flagged unreliable (SIGTERM'd mid-stage, or written during a
    SLURM job that had already been requeued at least once - see `_slurm_context`), are excluded
    from the aggregates so burst-QOS preemption noise doesn't get baked into reported latency.
    """
    out_dir = _ensure_output_dir(dataset)
    paths = sorted(glob.glob(os.path.join(out_dir, "timings_*.json")))

    by_stage: Dict[str, List[float]] = {}
    excluded = 0
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                timings = json.load(f) or {}
        except Exception:
            continue
        for entry in timings.values():
            if not entry.get("success", False) or not entry.get("reliable", True):
                excluded += 1
                continue
            by_stage.setdefault(entry.get("stage", "unknown"), []).append(entry["duration_seconds"])

    stages: Dict[str, Any] = {}
    for stage, durations in by_stage.items():
        durations.sort()
        n = len(durations)
        stages[stage] = {
            "n": n,
            "mean_seconds": statistics.mean(durations),
            "median_seconds": statistics.median(durations),
            "min_seconds": durations[0],
            "max_seconds": durations[-1],
            "p95_seconds": durations[min(n - 1, int(n * 0.95))],
        }

    return {"runs_included": len(paths), "excluded_entries": excluded, "stages": stages}
