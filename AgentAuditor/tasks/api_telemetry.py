"""Per-call API telemetry: token usage and latency, captured at the same instrumentation point
(wrapping the raw API call) since both are properties of a single request/response.

Neither was tracked anywhere in this pipeline before - LLMHandler.call_llm_api() (infer.py) and
its near-duplicates in preprocess.py/demo.py/infer_llm_repair.py only ever extracted
choices[0].message.content from the response, discarding everything else. The AI gateway
(api.ai.it.ufl.edu) DOES return a standard OpenAI-shaped "usage" field - confirmed via live curl
calls this session - it was just never read.

Wired into infer.py's LLMHandler first (the stage PROFILING_RESULTS.md identifies as the
pipeline's actual bottleneck, and the one most directly tied to cost/latency tradeoffs between
models) as the reference implementation; the other LLM-calling stages could adopt the same pattern
later - "decide later how to use them" per the brief this was built under.

Pure functions only (no I/O beyond write_api_telemetry's sidecar file).
"""
import json
import os
from typing import Any, Dict, List, Optional


def extract_usage(response_json: Any) -> Optional[Dict[str, int]]:
    """Pulls the standard OpenAI-shaped usage block (prompt_tokens/completion_tokens/total_tokens)
    from a raw API response dict, if present. Returns None rather than a zero-filled dict when
    absent, so callers can distinguish "no usage reported" from "zero tokens used"."""
    if not isinstance(response_json, dict):
        return None
    usage = response_json.get('usage')
    if not isinstance(usage, dict):
        return None
    try:
        return {
            'prompt_tokens': int(usage['prompt_tokens']),
            'completion_tokens': int(usage['completion_tokens']),
            'total_tokens': int(usage['total_tokens']),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile - no numpy dependency, matching this pipeline's existing
    dependency-light style (eval.py, calibration_metrics.py)."""
    k = max(0, min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1)))))
    return sorted_values[k]


def summarize_calls(call_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """call_records: list of {'elapsed_seconds': float, 'usage': Optional[dict], 'success': bool} -
    one entry per API call ATTEMPT (including failed/retried attempts, so this also surfaces retry
    overhead, not just successful-call stats - a failed call that still consumed tokens or time
    before erroring is a real cost).

    Latency is reported as percentiles (P50/P95/P99), not just mean, specifically because
    PROFILING_RESULTS.md already flagged an unexplained wide variance in infer_emb's 120b timing
    (3.6s to 119.8s across similar datasets) that a mean alone would hide.
    """
    total = len(call_records)
    if total == 0:
        return {'total_calls': 0, 'successful_calls': 0, 'failed_calls': 0,
                'latency': None, 'tokens': None}

    successful = [c for c in call_records if c.get('success')]
    failed = [c for c in call_records if not c.get('success')]

    latencies = sorted(c['elapsed_seconds'] for c in call_records if c.get('elapsed_seconds') is not None)
    latency_stats = None
    if latencies:
        latency_stats = {
            'mean': sum(latencies) / len(latencies),
            'p50': _percentile(latencies, 50),
            'p95': _percentile(latencies, 95),
            'p99': _percentile(latencies, 99),
            'min': latencies[0],
            'max': latencies[-1],
        }

    usages = [c['usage'] for c in call_records if c.get('usage')]
    token_stats = None
    if usages:
        total_prompt = sum(u['prompt_tokens'] for u in usages)
        total_completion = sum(u['completion_tokens'] for u in usages)
        total_tokens = sum(u['total_tokens'] for u in usages)
        token_stats = {
            'calls_with_usage': len(usages),
            'total_prompt_tokens': total_prompt,
            'total_completion_tokens': total_completion,
            'total_tokens': total_tokens,
            'mean_total_tokens_per_call': total_tokens / len(usages),
        }

    return {
        'total_calls': total,
        'successful_calls': len(successful),
        'failed_calls': len(failed),
        'latency': latency_stats,
        'tokens': token_stats,
    }


def write_api_telemetry(output_dir: str, stage: str, metrics: Dict[str, Any]) -> str:
    """Writes to <output_dir>/api_telemetry_<stage>.json - one file per stage since multiple
    LLM-calling stages could eventually report here."""
    telemetry_path = os.path.join(output_dir, f'api_telemetry_{stage}.json')
    with open(telemetry_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return telemetry_path
