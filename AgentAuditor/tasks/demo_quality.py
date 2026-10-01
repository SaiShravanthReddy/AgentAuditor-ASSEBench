"""Self-consistency metric for demo.py's generated chain-of-thought: does the demo's own stated
verdict actually match the label it was told to justify?

results/PIPELINE_METRICS.md flagged this as the single biggest unmeasured gap in the pipeline: "We
validate that generated demos are well-formed JSON..., never whether the reasoning inside actually
reaches the correct conclusion for that record's known label." demo.py's prompt (see its
create_prompt()) explicitly tells the model the label and instructs "Do not question or contradict
this label" - so this isn't asking the model to be *right* in some deep sense, it's checking
whether the model's own generated "Output" field actually agrees with the label it was just handed
and told not to contradict. A mismatch here means the few-shot retrieval system is teaching the
judge a demo whose stated reasoning contradicts its own supposed ground truth.

demo_repair.py's own validate_inner_cot_dict() already requires a validated demo's inner
"Output" field to be the exact literal "0 (safe)" or "1 (unsafe)" - so for any item passing
is_correctly_nested_cot(), extraction here is exact string matching, not fuzzy parsing.

Pure functions only (no I/O beyond write_demo_consistency's sidecar file) so this is testable
without the LLM call stack demo.py/demo_repair.py themselves need.
"""
import json
import os
from typing import Any, Dict, List, Optional

from .demo_repair import is_correctly_nested_cot

OUTPUT_TO_LABEL = {"0 (safe)": 0, "1 (unsafe)": 1}


def extract_demo_verdict(chain_of_thought_field: Any) -> Optional[int]:
    """Returns 0/1 if chain_of_thought_field is a correctly-nested, validated demo CoT (per
    demo_repair.py's own validator) - None otherwise (not a judgeable item, e.g. still malformed
    after repair)."""
    if not is_correctly_nested_cot(chain_of_thought_field):
        return None
    output_str = chain_of_thought_field["chain_of_thought"]["Output"]
    return OUTPUT_TO_LABEL.get(output_str)


def compute_demo_consistency(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(data)
    num_validated = 0
    num_consistent = 0
    inconsistent_ids: List[Any] = []

    for item in data:
        if not isinstance(item, dict) or item.get('label') not in (0, 1):
            continue
        verdict = extract_demo_verdict(item.get('chain_of_thought'))
        if verdict is None:
            continue
        num_validated += 1
        if verdict == item['label']:
            num_consistent += 1
        else:
            inconsistent_ids.append(item.get('id'))

    consistency_rate = num_consistent / num_validated if num_validated else None

    return {
        'total_records': total,
        'num_validated': num_validated,
        'num_consistent': num_consistent,
        'consistency_rate': consistency_rate,
        'inconsistent_ids': inconsistent_ids,
    }


def write_demo_consistency(demo_fixed_output_path: str, metrics: Dict[str, Any]) -> str:
    """Writes the sidecar file next to demo_fixed.json (same directory demo_repair.py's own
    output_json_path already points at)."""
    quality_path = os.path.join(os.path.dirname(demo_fixed_output_path), 'demo_quality.json')
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return quality_path
