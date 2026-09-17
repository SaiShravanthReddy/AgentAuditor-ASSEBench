"""Low-overhead timing for AgentAuditor pipeline stages and inference items.

Each top-level stage runs in a separate Python process. A shared run ID lets those processes
aggregate into one state file. Nested timers within a stage share in-memory state and write it only
when the outermost timer exits, avoiding per-item filesystem overhead.

Timing calls for a given dataset/run are expected to be serial within one process.
"""

import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict, cast


class RunDetails(TypedDict):
    dataset: str
    job_id: str
    node: str
    started_at: str | None
    ended_at: str | None


class StageDetails(TypedDict):
    duration_seconds: float
    success: bool
    reliable: bool
    start: str | None
    end: str | None


class ConversationDetails(TypedDict):
    conversation_id: str
    round_count: int
    stages: dict[str, float]
    total_seconds: float


class RunState(TypedDict):
    run: RunDetails
    stage_summary: dict[str, StageDetails]
    conversations: dict[str, ConversationDetails]


@dataclass(frozen=True)
class TimingMetadata:
    """Metadata consumed by the timer rather than forwarded to the wrapped function."""

    conversation_id: object | None = None
    round_count: int | None = None
    record_stage_summary: bool = True


_TEMP_ROOT = Path(__file__).resolve().parent.parent / "temp"
_active_states: dict[tuple[str, str], RunState] = {}
_active_depths: dict[tuple[str, str], int] = {}
_termination_signal: int | None = None
_signal_handlers_installed = False


def install_signal_handlers() -> None:
    """Install AgentAuditor's SIGTERM handler explicitly and at most once per process."""
    global _signal_handlers_installed
    if not _signal_handlers_installed:
        signal.signal(signal.SIGTERM, _handle_termination_signal)
        _signal_handlers_installed = True


def _handle_termination_signal(signum: int, _frame: Any) -> None:
    global _termination_signal
    _termination_signal = signum
    raise SystemExit(128 + signum)


def _output_dir(dataset: str) -> Path:
    path = _TEMP_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_id() -> str:
    return (
        os.environ.get("AGENTAUDITOR_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOBID")
        or f"pid{os.getpid()}_{int(time.time())}"
    )


def _state_file_path(dataset: str, run_id: str) -> Path:
    return _output_dir(dataset) / f"timings_{run_id}.state.json"


def _timing_file_path(dataset: str, run_id: str) -> Path:
    return _output_dir(dataset) / f"timings_{run_id}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slurm_restart_count() -> int:
    return int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)


def _empty_run_state(dataset: str, run_id: str) -> RunState:
    return {
        "run": {
            "dataset": dataset,
            "job_id": run_id,
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "started_at": None,
            "ended_at": None,
        },
        "stage_summary": {},
        "conversations": {},
    }


def _is_valid_state(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("run"), dict)
        and isinstance(value.get("stage_summary"), dict)
        and isinstance(value.get("conversations"), dict)
    )


def _load_run_state(dataset: str, run_id: str) -> RunState:
    path = _state_file_path(dataset, run_id)
    if not path.exists():
        return _empty_run_state(dataset, run_id)

    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read timing state {path}: {exc}") from exc

    if not _is_valid_state(state):
        raise RuntimeError(
            f"Timing state {path} has an invalid structure; refusing to overwrite it"
        )
    return cast(RunState, state)


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _save_run_state(state: RunState, dataset: str, run_id: str) -> None:
    _atomic_json_write(_state_file_path(dataset, run_id), state)


def _enter_timing_scope(dataset: str, run_id: str) -> tuple[tuple[str, str], RunState, bool]:
    key = (dataset, run_id)
    is_outermost = key not in _active_states
    if is_outermost:
        _active_states[key] = _load_run_state(dataset, run_id)
        _active_depths[key] = 0
    _active_depths[key] += 1
    return key, _active_states[key], is_outermost


def _leave_timing_scope(key: tuple[str, str], dataset: str, run_id: str) -> None:
    remaining_depth = _active_depths[key] - 1
    if remaining_depth > 0:
        _active_depths[key] = remaining_depth
        return

    state = _active_states[key]
    try:
        _save_run_state(state, dataset, run_id)
        _write_summary(state, dataset, run_id)
    finally:
        _active_depths.pop(key, None)
        _active_states.pop(key, None)


def _update_stage_summary(
    state: RunState,
    stage_name: str,
    duration: float,
    start: str,
    end: str,
    success: bool,
    reliable: bool,
) -> None:
    details = state["stage_summary"].setdefault(
        stage_name,
        {
            "duration_seconds": 0.0,
            "success": True,
            "reliable": True,
            "start": None,
            "end": None,
        },
    )
    details["duration_seconds"] += duration
    details["success"] = details["success"] and success
    details["reliable"] = details["reliable"] and reliable

    if details["start"] is None or start < details["start"]:
        details["start"] = start
    if details["end"] is None or end > details["end"]:
        details["end"] = end


def _update_conversation_summary(
    state: RunState,
    conversation_id: object | None,
    round_count: int | None,
    stage_name: str,
    duration: float,
) -> None:
    if conversation_id is None:
        return

    conversation_key = str(conversation_id)
    conversation = state["conversations"].setdefault(
        conversation_key,
        {
            "conversation_id": conversation_key,
            "round_count": round_count or 0,
            "stages": {},
            "total_seconds": 0.0,
        },
    )
    if round_count is not None:
        conversation["round_count"] = round_count
    elif "round_count" not in conversation:
        legacy_rounds = cast(dict[str, Any], conversation).get("rounds", {})
        conversation["round_count"] = len(legacy_rounds)
    conversation["stages"][stage_name] = conversation["stages"].get(stage_name, 0.0) + duration
    conversation["total_seconds"] += duration


def time_and_record(
    stage_name: str,
    func: Callable[..., Any],
    dataset: str,
    *args: Any,
    timing: TimingMetadata | None = None,
    **func_kwargs: Any,
) -> Any:
    """Run ``func``, record its duration, and return its result.

    ``timing`` is consumed here; all remaining positional and keyword arguments are forwarded to
    ``func`` unchanged.
    """
    global _termination_signal

    metadata = timing or TimingMetadata()
    run_id = _run_id()
    scope_key, state, is_outermost = _enter_timing_scope(dataset, run_id)
    if is_outermost:
        _termination_signal = None

    start_monotonic = time.monotonic()
    start_timestamp = _utc_now()
    success = False

    try:
        result = func(*args, **func_kwargs)
        success = True
        return result
    finally:
        end_timestamp = _utc_now()
        duration = time.monotonic() - start_monotonic
        try:
            preempted = _termination_signal is not None
            reliable = not preempted and _slurm_restart_count() == 0

            run = state["run"]
            if run["started_at"] is None or start_timestamp < run["started_at"]:
                run["started_at"] = start_timestamp
            if run["ended_at"] is None or end_timestamp > run["ended_at"]:
                run["ended_at"] = end_timestamp

            if metadata.record_stage_summary:
                _update_stage_summary(
                    state,
                    stage_name,
                    duration,
                    start_timestamp,
                    end_timestamp,
                    success,
                    reliable,
                )
            _update_conversation_summary(
                state,
                metadata.conversation_id,
                metadata.round_count,
                stage_name,
                duration,
            )
        finally:
            _leave_timing_scope(scope_key, dataset, run_id)


def _rounded_stages(stages: dict[str, float]) -> dict[str, float]:
    return {name: round(float(duration), 2) for name, duration in stages.items()}


def _build_summary(state: RunState, dataset: str, run_id: str) -> dict[str, Any]:
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
    for conversation_id, conversation in sorted(state["conversations"].items()):
        legacy_rounds = cast(dict[str, Any], conversation).get("rounds", {})
        round_count = conversation.get("round_count", len(legacy_rounds))
        conversations.append(
            {
                "conversation_id": conversation_id,
                "round_count": round_count,
                "total_seconds": round(float(conversation["total_seconds"]), 2),
                "stages": _rounded_stages(conversation["stages"]),
            }
        )

    summary = {
        "run": {
            "dataset": dataset,
            "job_id": run_id,
            "node": state["run"]["node"],
            "started_at": state["run"]["started_at"],
            "ended_at": state["run"]["ended_at"],
        },
        "stage_summary": stage_summary,
        "conversations": conversations,
    }
    return summary


def _write_summary(state: RunState, dataset: str, run_id: str) -> dict[str, Any]:
    summary = _build_summary(state, dataset, run_id)
    _atomic_json_write(_timing_file_path(dataset, run_id), summary)
    return summary


def build_nested_run_timing_structure(dataset: str, run_id: str | None = None) -> dict[str, Any]:
    """Rebuild and persist the human-readable timing summary for one run."""
    effective_run_id = run_id or _run_id()
    state = _load_run_state(dataset, effective_run_id)
    return _write_summary(state, dataset, effective_run_id)
