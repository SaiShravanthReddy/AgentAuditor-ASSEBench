"""Per-category accuracy breakdown - e.g. CNFinBench's MT_App/MT_Cog/MT_Inter scenario types, or
FinVault's numbered attack categories (01-attack-case, etc). Previously only done ad-hoc by
manually reading records whenever a regression came up (e.g. the MT_App false-negative clustering
in results/AGENTAUDITOR_DIAGNOSIS.md, or the v3-vs-v5 case-category comparison) - this makes it a
standing, automatic metric instead of a one-off investigation each time.

preprocess.py drops the dataset's own `scenario` field (only keeps id/profile/contents/label - see
CNFinBench/cnfinbench_to_agentauditor.py's docstring), so category can't be read from a dedicated
field downstream. Both CNFinBench and FinVault encode it directly in the record `id` instead
(e.g. "harmless-MT_App-12", "finvault-v5-fixed-01-attack-case_fixed_0089"), which does survive -
extraction here is regex over the id string, not a new field.
"""
import re
from typing import Any, Dict, List, Optional

from .eval import calculate_metrics

CNFINBENCH_SCENARIO_RE = re.compile(r'(MT_App|MT_Cog|MT_Inter)')
FINVAULT_CATEGORY_RE = re.compile(r'(\d+-[a-z]+-case)')


def extract_category_from_id(item_id: Any) -> Optional[str]:
    """Returns the category token found in item_id, or None if neither known pattern matches
    (e.g. rjudge/agentharm/AgentJudge ids, which don't carry a scenario category this way)."""
    if not isinstance(item_id, str):
        return None
    m = CNFINBENCH_SCENARIO_RE.search(item_id)
    if m:
        return m.group(1)
    m = FINVAULT_CATEGORY_RE.search(item_id)
    if m:
        return m.group(1)
    return None


def compute_category_breakdown(
    ids: List[Any], true_labels: List[int], predicted_labels: List[int]
) -> Dict[str, Dict[str, Any]]:
    """Groups items by extract_category_from_id() and computes accuracy/precision/recall/F1 within
    each group, the same way eval.py's overall metrics are computed - just scoped to one category
    instead of the whole dataset. Items whose id doesn't match a known category are grouped under
    'uncategorized' rather than silently dropped, so counts still sum to the input length.
    """
    groups: Dict[str, Dict[str, List[int]]] = {}
    for item_id, true_label, predicted_label in zip(ids, true_labels, predicted_labels):
        category = extract_category_from_id(item_id) or 'uncategorized'
        groups.setdefault(category, {'true': [], 'predicted': []})
        groups[category]['true'].append(true_label)
        groups[category]['predicted'].append(predicted_label)

    breakdown = {}
    for category, data in groups.items():
        accuracy, precision, recall, f1 = calculate_metrics(data['true'], data['predicted'])
        breakdown[category] = {
            'count': len(data['true']),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return breakdown
