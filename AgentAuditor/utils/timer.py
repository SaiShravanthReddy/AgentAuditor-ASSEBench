import json
import os
import signal
import socket
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List


def _ensure_output_dir(dataset: str) -> str:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../temp", dataset)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _run_id() -> str:
    return (
        os.environ.get("AGENTAUDITOR_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOBID")
        or f"pid{os.getpid()}_{int(time.time())}"
    )


def _state_file_path(dataset: str, run_id: str | None = None) -> str:
    effective_run_id = run_id or _run_id()
    return os.path.join(_ensure_output_dir(dataset), f"timings_{effective_run_id}.state.json")


def _timing_file_path(dataset: str, run_id: str | None = None) -> str:
    effective_run_id = run_id or _run_id()
    return os.path.join(_ensure_output_dir(dataset), f"timings_{effective_run_id}.json")


_preempt_signal_state: Dict[str, Any] = {"received": False, "signal": None}


def _handle_termination_signal(signum, frame):
    _preempt_signal_state["received"] = True
    _preempt_signal_state["signal"] = signum
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _handle_termination_signal)


def _slurm_context() -> Dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", 0) or 0),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }


def _empty_run_state(dataset: str, run_id: str | None = None) -> Dict[str, Any]:
    return {
        "run": {
            "dataset": dataset,
            "model": os.environ.get("AGENTAUDITOR_MODEL_INFER") or os.environ.get("AGENTAUDITOR_MODEL") or "unknown",
            "job_id": run_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "unknown",
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "started_at": None,
            "ended_at": None,
        },
        "stage_summary": {},
        "conversations": {},
    }


def _load_run_state(dataset: str, run_id: str | None = None) -> Dict[str, Any]:
    effective_run_id = run_id or _run_id()
    path = _state_file_path(dataset, effective_run_id)
    if not os.path.exists(path):
        return _empty_run_state(dataset, effective_run_id)

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict) and "run" in state and "stage_summary" in state and "conversations" in state:
            return state
    except Exception:
        pass

    return _empty_run_state(dataset, effective_run_id)


def _save_run_state(state: Dict[str, Any], dataset: str, run_id: str | None = None) -> None:
    effective_run_id = run_id or _run_id()
    path = _state_file_path(dataset, effective_run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _update_stage_summary(state: Dict[str, Any], stage_name: str, duration: float, start: str | None, end: str | None, success: bool, reliable: bool) -> None:
    stage_bucket = state["stage_summary"].setdefault(
        stage_name,
        {"duration_seconds": 0.0, "success": True, "reliable": True, "start": None, "end": None},
    )
    stage_bucket["duration_seconds"] += duration
    stage_bucket["success"] = bool(stage_bucket["success"] and success)
    stage_bucket["reliable"] = bool(stage_bucket["reliable"] and reliable)

    if start is not None and (stage_bucket["start"] is None or start < stage_bucket["start"]):
        stage_bucket["start"] = start
    if end is not None and (stage_bucket["end"] is None or end > stage_bucket["end"]):
        stage_bucket["end"] = end


def _update_conversation_summary(state: Dict[str, Any], conversation_id: str, round_index: int | None, stage_name: str, duration: float) -> None:
    if conversation_id is None:
        return

    convo = state["conversations"].setdefault(
        str(conversation_id),
        {"conversation_id": str(conversation_id), "stages": {}, "rounds": {}, "total_seconds": 0.0},
    )
    convo["stages"][stage_name] = convo["stages"].get(stage_name, 0.0) + duration
    convo["total_seconds"] += duration

    if round_index is not None:
        round_key = str(round_index)
        round_entry = convo["rounds"].setdefault(
            round_key,
            {"round_index": round_index, "stages": {}, "total_seconds": 0.0},
        )
        round_entry["stages"][stage_name] = round_entry["stages"].get(stage_name, 0.0) + duration
        round_entry["total_seconds"] += duration


def _write_summary_to_disk(dataset: str, summary: Dict[str, Any], run_id: str | None = None) -> None:
    path = _timing_file_path(dataset, run_id=run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _delete_state_file(dataset: str, run_id: str | None = None) -> None:
    path = _state_file_path(dataset, run_id)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def time_and_record(stage_name: str, func: Callable, dataset: str, *args, **kwargs) -> Any:
    """Measure a stage and persist the live aggregate state to disk so separate Python invocations share one run."""
    conversation_id = kwargs.pop("conversation_id", None)
    round_index = kwargs.pop("round_index", None)
    run_id = _run_id()
    state = _load_run_state(dataset, run_id)

    start_mono = time.monotonic()
    start_iso = datetime.utcnow().isoformat() + "Z"
    slurm = _slurm_context()
    _preempt_signal_state["received"] = False
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
    except Exception:
        success = False
        raise
    finally:
        end_iso = datetime.utcnow().isoformat() + "Z"
        duration = time.monotonic() - start_mono

        unreliable_reasons: List[str] = []
        if preempted or _preempt_signal_state["received"]:
            unreliable_reasons.append("sigterm_during_stage")
        if slurm["slurm_restart_count"] > 0:
            unreliable_reasons.append("job_requeued")

        if state["run"]["started_at"] is None or start_iso < state["run"]["started_at"]:
            state["run"]["started_at"] = start_iso
        if end_iso > (state["run"]["ended_at"] or end_iso):
            state["run"]["ended_at"] = end_iso

        _update_stage_summary(state, stage_name, duration, start_iso, end_iso, success, len(unreliable_reasons) == 0)
        _update_conversation_summary(state, conversation_id, round_index, stage_name, duration)
        _save_run_state(state, dataset, run_id)


def build_nested_run_timing_structure(dataset: str, run_id: str | None = None) -> Dict[str, Any]:
    """Assemble the final clean summary from the persisted run state and write it once."""
    effective_run_id = run_id or _run_id()
    state = _load_run_state(dataset, effective_run_id)

    stage_summary = {
        stage: {
            "duration_seconds": round(float(details["duration_seconds"]), 2),
            "success": details["success"],
            "reliable": details["reliable"],
            "start": details["start"],
            "end": details["end"],
        }
        for stage, details in state["stage_summary"].items()
    }

    conversations = []
    for conversation_id, convo in sorted(state["conversations"].items(), key=lambda item: str(item[0])):
        round_rows = []
        for round_key, round_entry in sorted(convo["rounds"].items(), key=lambda item: int(item[0])):
            round_rows.append({
                "round_index": int(round_key),
                "total_seconds": round(float(round_entry["total_seconds"]), 2),
                "stages": {k: round(float(v), 2) for k, v in round_entry["stages"].items()},
            })

        conversations.append({
            "conversation_id": conversation_id,
            "total_seconds": round(float(convo["total_seconds"]), 2),
            "stages": {k: round(float(v), 2) for k, v in convo["stages"].items()},
            "rounds": round_rows,
        })

    summary = {
        "run": {
            "dataset": dataset,
            "model": state["run"]["model"],
            "job_id": effective_run_id,
            "node": state["run"]["node"],
            "started_at": state["run"]["started_at"],
            "ended_at": state["run"]["ended_at"],
        },
        "stage_summary": stage_summary,
        "conversations": conversations,
    }

    _write_summary_to_disk(dataset, summary, effective_run_id)
    # _delete_state_file(dataset, effective_run_id)
    return summary

