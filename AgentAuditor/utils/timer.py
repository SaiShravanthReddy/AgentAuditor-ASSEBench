import glob
import json
import os
import signal
import socket
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
    """Return the identifier for one run.

    All stages from the same job should write into the same timing file, while different runs get
    separate files. The value is taken from AGENTAUDITOR_RUN_ID when present, otherwise SLURM's job
    ID, and finally a process-local fallback for ad hoc runs.
    """
    return (
        os.environ.get("AGENTAUDITOR_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOBID")
        or f"pid{os.getpid()}_{int(time.time())}"
    )


def _timing_file_path(dataset: str) -> str:
    out_dir = _ensure_output_dir(dataset)
    return os.path.join(out_dir, f"timings_{_run_id()}.json")


# Track whether a stage was interrupted by a termination signal so the timing record can be marked
# as unreliable instead of silently treating a truncated run as valid.
_preempt_signal_state: Dict[str, Any] = {"received": False, "signal": None}


def _handle_termination_signal(signum, frame):
    _preempt_signal_state["received"] = True
    _preempt_signal_state["signal"] = signum
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _handle_termination_signal)


def _slurm_context() -> Dict[str, Any]:
    """Collect the available scheduler and host metadata for each timed stage."""
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", 0) or 0),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }


def time_and_record(stage_name: str, func: Callable, dataset: str, *args, **kwargs) -> Any:
    """Execute a callable while measuring its runtime and persisting a timing record.

    The wrapped function is called exactly as provided. Metadata fields such as conversation_id,
    item_id, round_index, and run_metadata are stripped out before invocation so they only affect
    the timing record and do not leak into the function call itself.
    """
    conversation_id = kwargs.pop("conversation_id", None)
    item_id = kwargs.pop("item_id", None)
    round_index = kwargs.pop("round_index", None)
    metadata = kwargs.pop("metadata", None)
    run_metadata = kwargs.pop("run_metadata", None)

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
        if conversation_id is not None:
            timing_entry["conversation_id"] = conversation_id
        if item_id is not None:
            timing_entry["item_id"] = item_id
        if round_index is not None:
            timing_entry["round_index"] = round_index
        if metadata is not None:
            timing_entry["metadata"] = metadata
        if run_metadata is not None:
            timing_entry["run_metadata"] = run_metadata
        if exc_info is not None:
            timing_entry["error"] = exc_info

        # Load the existing timing file for the run and append this stage entry.
        path = _timing_file_path(dataset)
        timings: Dict[str, Any] = {}
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    timings = json.load(f) or {}
        except Exception:
            timings = {}

        # Use the stage name as the top-level key, then add a suffix only if the same stage appears
        # more than once in the same run file.
        key = stage_name
        if key in timings:
            suffix = 1
            while f"{key}_{suffix}" in timings:
                suffix += 1
            key = f"{key}_{suffix}"

        timings[key] = timing_entry

        try:
            # Write to a temporary file and atomically replace the target so a partial write cannot
            # corrupt the full run history.
            tmp_path = f"{path}.tmp{os.getpid()}"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(timings, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            # Timing metadata is auxiliary; a write failure should not hide the original execution
            # failure from the caller.
            pass


def build_nested_run_timing_structure(dataset: str, run_id: str | None = None) -> Dict[str, Any]:
    """Build the canonical per-run summary structure.

    The format is:
      {
        "run": {...},
        "stage_summary": {...},
        "conversations": [{...}]
      }

    Stage totals are accumulated across all raw timing records in the run. The run-level metadata is
    kept separate from stage metadata, and round identity remains in conversation_id + round_index.
    """

    def empty_run_summary() -> Dict[str, Any]:
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
            "conversations": [],
        }

    def rounded_value(value: float) -> float:
        return round(float(value), 2)

    def rounded_map(values: Dict[str, float]) -> Dict[str, float]:
        return {key: rounded_value(value) for key, value in values.items()}

    out_dir = _ensure_output_dir(dataset)
    paths = sorted(glob.glob(os.path.join(out_dir, "timings_*.json")))
    if run_id is not None:
        paths = [p for p in paths if f"timings_{run_id}" in p]

    if not paths:
        return empty_run_summary()

    path = paths[-1]

    try:
        with open(path, 'r', encoding='utf-8') as f:
            timings = json.load(f) or {}
    except Exception:
        return empty_run_summary()

    stage_summary: Dict[str, Any] = {}
    conversations: Dict[str, Dict[str, Any]] = {}
    started_at = None
    ended_at = None

    for entry in timings.values():
        stage_name = entry.get("stage", "unknown")
        stage_duration = float(entry.get("duration_seconds", 0.0) or 0.0)

        stage_bucket = stage_summary.setdefault(stage_name, {
            "duration_seconds": 0.0,
            "success": True,
            "reliable": True,
        })
        stage_bucket["duration_seconds"] += stage_duration
        stage_bucket["success"] = bool(stage_bucket["success"] and entry.get("success", True))
        stage_bucket["reliable"] = bool(stage_bucket["reliable"] and entry.get("reliable", True))

        start_value = entry.get("start")
        end_value = entry.get("end")
        if start_value is not None and (started_at is None or start_value < started_at):
            started_at = start_value
        if end_value is not None and (ended_at is None or end_value > ended_at):
            ended_at = end_value

        conversation_id = entry.get("conversation_id")
        if conversation_id is None:
            continue

        convo = conversations.setdefault(
            str(conversation_id),
            {"conversation_id": str(conversation_id), "stages": {}, "rounds": {}, "total_seconds": 0.0},
        )
        convo["stages"][stage_name] = convo["stages"].get(stage_name, 0.0) + stage_duration
        convo["total_seconds"] += stage_duration

        round_index = entry.get("round_index")
        if round_index is not None:
            round_key = str(round_index)
            round_entry = convo["rounds"].setdefault(
                round_key,
                {"round_index": round_index, "stages": {}, "total_seconds": 0.0},
            )
            round_entry["stages"][stage_name] = round_entry["stages"].get(stage_name, 0.0) + stage_duration
            round_entry["total_seconds"] += stage_duration

    stage_summary = {
        stage: {
            "duration_seconds": rounded_value(details["duration_seconds"]),
            "success": details["success"],
            "reliable": details["reliable"],
        }
        for stage, details in stage_summary.items()
    }

    conversations_list = []
    for conversation_id, convo in sorted(conversations.items(), key=lambda item: str(item[0])):
        rounds_list = []
        for round_index, round_entry in sorted(convo["rounds"].items(), key=lambda item: int(item[0])):
            rounds_list.append({
                "round_index": int(round_index),
                "total_seconds": rounded_value(round_entry["total_seconds"]),
                "stages": rounded_map(round_entry["stages"]),
            })

        conversations_list.append({
            "conversation_id": conversation_id,
            "total_seconds": rounded_value(convo["total_seconds"]),
            "stages": rounded_map(convo["stages"]),
            "rounds": rounds_list,
        })

    return {
        "run": {
            "dataset": dataset,
            "model": os.environ.get("AGENTAUDITOR_MODEL_INFER") or os.environ.get("AGENTAUDITOR_MODEL") or "unknown",
            "job_id": run_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or os.path.splitext(os.path.basename(path))[0].replace("timings_", ""),
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "stage_summary": stage_summary,
        "conversations": conversations_list,
    }
