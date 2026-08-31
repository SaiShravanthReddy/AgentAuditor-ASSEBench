#!/usr/bin/env python3
"""Independently verify FinVault v3's anonymization actually removes case_id/attack-type leakage.

the v3 data provider's export (finvault_output_full1064_v3_anonymized/) replaces case_id everywhere with opaque
tokens (case_0347 etc.) and keeps the real case_id in a separate case_id_mapping.json, not
embedded in the trajectory data. This checks that claim against the raw trajectories.jsonl rather
than trusting it, per the leakage mechanism found in v2: case_id (which embeds the attack
technique, e.g. "ATTACK_V1_001_authority_override") appeared in the task_prompt's
"[Case ID: ...]" line, in 74% of environment/tool observations, and in 6% of the agent's own
tool-call actions (as a case_id argument) - confirmed structural (present in both v1 and v2).

Checks every task_prompt/observation/action string in v3 for:
  - the real case_id (from the mapping file)
  - the real attack_type / synthesis_technique substrings
  - the literal "[Case ID: ...]" tag pattern
  - any leftover "ATTACK_V<n>_..." style substring not caught by the above

Usage:
    python FinVault/verify_v3_anonymization.py <v3_dir> [--mapping case_id_mapping.json]

    <v3_dir> should contain trajectories.jsonl and (unless --mapping is given explicitly)
    case_id_mapping.json.
"""
import argparse
import json
import os
import re
import sys

ATTACK_ID_RE = re.compile(r"ATTACK_V\d+_\d+\w*", re.IGNORECASE)
CASE_ID_TAG_RE = re.compile(r"\[Case ID:\s*[^\]]+\]", re.IGNORECASE)


def all_text_fields(record):
    """Yield (field_path, text) for every string the judge/converter would see."""
    if record.get("task_prompt"):
        yield "task_prompt", record["task_prompt"]
    for i, step in enumerate(record.get("steps", [])):
        if step.get("observation"):
            yield f"steps[{i}].observation", step["observation"]
        if step.get("action"):
            yield f"steps[{i}].action", step["action"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v3_dir", help="Path to finvault_output_full1064_v3_anonymized/")
    parser.add_argument("--mapping", default=None,
                         help="Path to case_id_mapping.json (default: <v3_dir>/case_id_mapping.json)")
    args = parser.parse_args()

    traj_path = os.path.join(args.v3_dir, "trajectories.jsonl")
    mapping_path = args.mapping or os.path.join(args.v3_dir, "case_id_mapping.json")

    if not os.path.exists(traj_path):
        print(f"ERROR: {traj_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Support either {opaque: real} or {real: opaque}; detect by which side looks like "case_NNNN".
    sample_keys = list(mapping.keys())[:5]
    opaque_is_key = all(re.match(r"^case_\d+$", k) for k in sample_keys) if sample_keys else True
    real_ids = set(mapping.values()) if opaque_is_key else set(mapping.keys())

    records = [json.loads(l) for l in open(traj_path, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} v3 records, {len(mapping)} mapping entries")

    leaked = []
    leftover_attack_pattern = []
    leftover_case_tag = []

    for record in records:
        opaque_id = record.get("case_id", "")
        real_id = mapping.get(opaque_id) if opaque_is_key else next(
            (k for k, v in mapping.items() if v == opaque_id), None
        )
        attack_type = record.get("attack_type") or ""
        synthesis_technique = record.get("synthesis_technique") or ""

        needles = [n for n in (real_id, attack_type, synthesis_technique) if n]

        for field, text in all_text_fields(record):
            for needle in needles:
                if needle and needle.lower() in text.lower():
                    leaked.append((opaque_id, field, needle))
            if CASE_ID_TAG_RE.search(text):
                leftover_case_tag.append((opaque_id, field))
            m = ATTACK_ID_RE.search(text)
            if m and m.group(0) != opaque_id:
                leftover_attack_pattern.append((opaque_id, field, m.group(0)))

    print(f"\nDirect real-id/attack_type/synthesis_technique leaks: {len(leaked)}")
    for opaque_id, field, needle in leaked[:20]:
        print(f"  {opaque_id} :: {field} contains {needle!r}")

    print(f"\nLeftover '[Case ID: ...]' tags: {len(leftover_case_tag)}")
    for opaque_id, field in leftover_case_tag[:20]:
        print(f"  {opaque_id} :: {field}")

    print(f"\nLeftover ATTACK_V<n>_... substrings not matching this record's own opaque id: {len(leftover_attack_pattern)}")
    for opaque_id, field, match in leftover_attack_pattern[:20]:
        print(f"  {opaque_id} :: {field} contains {match!r}")

    total_issues = len(leaked) + len(leftover_case_tag) + len(leftover_attack_pattern)
    print(f"\n{'FAIL' if total_issues else 'PASS'}: {total_issues} total leakage findings across {len(records)} records")
    sys.exit(1 if total_issues else 0)


if __name__ == "__main__":
    main()
