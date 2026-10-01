"""Demo CoT reasoning specificity: does a generated demo's reasoning actually reference the real
conversation it's supposedly explaining, or is it generic/templated text that could apply to any
case?

demo_quality.py's self-consistency check catches a demo whose stated verdict contradicts its own
label - but a demo could state the "correct" verdict for the wrong reason, with reasoning that's
vague boilerplate rather than grounded in the specific case. This is the exact failure pattern
independently confirmed elsewhere in this pipeline: infer_llm_repair.py's correction LLM, when
given a shape it didn't recognize, produced "Received input JSON with verdict and confidence.
Verdict '0' indicates safe content." - reasoning that doesn't reference the actual case at all (see
that module's git history). This metric exists so that failure mode gets caught automatically here
too, rather than needing another manual incident to notice it.

Heuristic: fraction of distinctive (non-stopword) words in the reasoning text that also appear
somewhere in the actual conversation content. Low overlap suggests generic reasoning not grounded
in the specific case; it's not a perfect measure of REASONING QUALITY, only of whether the text is
case-specific - a demo can score high here and still reason badly, but a demo that scores low is
very likely not engaging with the actual content at all.

Pure functions only (no I/O beyond write_demo_specificity's sidecar file).
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that', 'these',
    'those', 'it', 'its', 'as', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'and', 'or',
    'but', 'not', 'no', 'does', 'do', 'did', 'has', 'have', 'had', 'will', 'would', 'should',
    'could', 'can', 'may', 'might', 'based', 'provides', 'provide', 'provided', 'any', 'all',
    'each', 'both', 'such', 'what', 'which', 'who', 'whom', 'their', 'there', 'here', 'from',
    'into', 'than', 'then', 'also', 'if', 'so', 'about', 'because', 'while', 'during', 'between',
}

LOW_SPECIFICITY_THRESHOLD = 0.1


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def compute_specificity_score(reasoning_text: str, conversation_text: str) -> Optional[float]:
    """Fraction of distinctive reasoning_text tokens that also appear in conversation_text. None
    if reasoning_text has no distinctive tokens at all (e.g. empty or pure stopwords/punctuation)."""
    reasoning_tokens = _tokenize(reasoning_text)
    if not reasoning_tokens:
        return None
    conversation_tokens = _tokenize(conversation_text)
    overlap = reasoning_tokens & conversation_tokens
    return len(overlap) / len(reasoning_tokens)


def _extract_conversation_text(contents: Any) -> str:
    """Pulls just the actual dialogue text out of the contents structure (a list wrapping a list
    of {"role": ..., "content": ...} turns - see CLAUDE.md's contents schema note). Deliberately
    NOT a raw json.dumps() of the structure: that would include JSON key names like "role" and
    "content" as tokens, which can spuriously match reasoning text (e.g. the word "content"
    appearing in both a JSON key and in unrelated reasoning prose) without the actual dialogue
    sharing anything real - confirmed causing a false negative in this module's own tests before
    this fix."""
    texts = []
    turns = contents[0] if isinstance(contents, list) and contents and isinstance(contents[0], list) else contents
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict) and isinstance(turn.get('content'), str):
                texts.append(turn['content'])
    return ' '.join(texts)


def compute_demo_specificity(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores every validated demo (reuses demo_repair.py's own is_correctly_nested_cot - only a
    structurally valid demo has reasoning text worth scoring in the first place) and reports the
    mean plus which ones fell below LOW_SPECIFICITY_THRESHOLD, for direct inspection."""
    from .demo_repair import is_correctly_nested_cot

    scores = []
    low_specificity_ids = []

    for item in data:
        if not isinstance(item, dict):
            continue
        cot = item.get('chain_of_thought')
        if not is_correctly_nested_cot(cot):
            continue

        inner = cot['chain_of_thought']
        reasoning_text = ' '.join(str(v) for k, v in inner.items() if k != 'Output')
        conversation_text = _extract_conversation_text(item.get('contents'))

        score = compute_specificity_score(reasoning_text, conversation_text)
        if score is None:
            continue
        scores.append(score)
        if score < LOW_SPECIFICITY_THRESHOLD:
            low_specificity_ids.append(item.get('id'))

    if not scores:
        return {'num_scored': 0, 'mean_specificity': None, 'low_specificity_count': 0,
                'low_specificity_ids': []}

    return {
        'num_scored': len(scores),
        'mean_specificity': sum(scores) / len(scores),
        'low_specificity_count': len(low_specificity_ids),
        'low_specificity_ids': low_specificity_ids,
    }


def write_demo_specificity(demo_fixed_output_path: str, metrics: Dict[str, Any]) -> str:
    """Writes the sidecar file next to demo_fixed.json (same directory demo_repair.py's own
    output_json_path already points at)."""
    quality_path = os.path.join(os.path.dirname(demo_fixed_output_path), 'demo_specificity.json')
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return quality_path
