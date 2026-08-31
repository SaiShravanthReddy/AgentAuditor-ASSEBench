#!/usr/bin/env python3
"""Fix the two mechanical case_id leaks found in FinVault v3 ("anonymized") that the data provider's own
anonymization pass missed - NOT the in-band tool-verdict leak (e.g. "detected potential
violation: Transaction Structuring"), which is a separate, judgment-call question (is that
legitimate simulated environment feedback, or a scripted label-echo?) deliberately left alone here.

Two issues fixed:

1. The top-level `case_id` field itself was never anonymized in v3 - 1064/1064 records still have
   the real, technique-revealing string (e.g. "ATTACK_V1_001_authority_override"). Only text
   *embedded inside* task_prompt/observation/action was scrubbed.

2. 64/1064 records still have a real case_id baked into a tool observation's JSON payload (e.g.
   `steps[0].observation` containing `"case_id": "ATTACK_V1_001_amortized_cost"`), including cases
   where the leaked id belongs to a *different* record (the shared base scenario before a
   synthesis technique was layered on top) - i.e. this isn't only "does a record leak its own id,"
   it's "does any record's text contain any real id from the whole dataset."

IMPORTANT - v3's own case_id_mapping.json is NOT used as the source of new ids here, because it has
a real correctness bug: it's keyed by the bare `case_id` string, but 43 case_id strings (e.g.
"ATTACK_V1_001") are shared by multiple genuinely distinct records (different scenario_id,
different task_prompt, different outcome - verified, not assumed) across different scenarios/case
types. the original mapping collapses all of them onto the same opaque token (e.g. all 4
"ATTACK_V1_001" records -> "case_0744"), which would silently merge distinct records' identities if
reused directly as our unique id. Instead, this script:
  - Assigns each output record its own fresh, guaranteed-unique id (by line position), independent
    of FinVault's mapping.
  - Still uses v3's case_id_mapping.json, but only as a source of "which strings count as real case
    ids that must be scrubbed from text" (mapping.keys()) - collisions don't matter for that use,
    since we're removing identifying substrings, not preserving a stable per-case identity in text.

Output: a new sibling folder (default finvault_output_full1064_v3_fixed/) containing a fixed
trajectories.jsonl, plus a local-only debug_id_mapping.json (new_id -> real original case_id, for
our own troubleshooting - never fed into the AgentAuditor pipeline, and this whole output dir is
under the gitignored FinVault/data/).

Usage:
    python FinVault/fix_v3_leakage.py <v3_dir> [<output_dir>]
"""
import argparse
import json
import os
import re
import sys

REDACTED = "[REDACTED_CASE_ID]"

# Catches case_id-shaped strings the exact-match pass (below) can't, because they aren't a literal
# copy of any real id in case_id_mapping.json - verified empirically that the agent sometimes
# generates a near-duplicate id itself (e.g. as a tool-call argument), such as
# "ATTACK_V1_000001_authority_override" vs. the real "ATTACK_V1_001_authority_override" (extra
# padding digit). Also catches an optional leading "<technique>__" prefix, since that's part of the
# real id shape too (e.g. "roleplay_induction__ATTACK_V1_001_...").
GENERIC_CASE_ID_PATTERN = re.compile(
    r"(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)*__)?(?:ATTACK|NORMAL)_V\d+_\d+(?:_[A-Za-z0-9]+)*",
    re.IGNORECASE,
)


def build_redaction_pattern(mapping):
    # Real ids can be substrings of each other (e.g. "ATTACK_V1_001" is a prefix of
    # "ATTACK_V1_001_amortized_cost") - sort longest-first so the full id is matched and replaced
    # before any shorter overlapping fragment, avoiding a partial replacement that still leaks the
    # descriptive suffix.
    real_ids = sorted(set(mapping.keys()), key=len, reverse=True)
    escaped = [re.escape(rid) for rid in real_ids if rid]
    pattern = re.compile("|".join(escaped))
    return pattern


def scrub_text(text, pattern):
    if not text:
        return text
    text = pattern.sub(REDACTED, text)
    # Second pass: catch case_id-shaped strings that weren't an exact match to any known real id
    # (e.g. agent-hallucinated near-duplicates) but still visibly encode ATTACK_V*/NORMAL_V*.
    text = GENERIC_CASE_ID_PATTERN.sub(REDACTED, text)
    return text


def fix_record(record, idx, pattern):
    original_case_id = record.get("case_id", "")

    fixed = dict(record)
    fixed["case_id"] = f"case_fixed_{idx:04d}"

    if fixed.get("task_prompt"):
        fixed["task_prompt"] = scrub_text(fixed["task_prompt"], pattern)

    fixed_steps = []
    for step in fixed.get("steps", []):
        fixed_step = dict(step)
        if fixed_step.get("observation"):
            fixed_step["observation"] = scrub_text(fixed_step["observation"], pattern)
        if fixed_step.get("action"):
            fixed_step["action"] = scrub_text(fixed_step["action"], pattern)
        fixed_steps.append(fixed_step)
    fixed["steps"] = fixed_steps

    return fixed, original_case_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v3_dir", help="Path to finvault_output_full1064_v3_anonymized/")
    parser.add_argument("output_dir", nargs="?", default=None,
                         help="Output dir (default: <v3_dir>/../finvault_output_full1064_v3_fixed)")
    args = parser.parse_args()

    traj_path = os.path.join(args.v3_dir, "trajectories.jsonl")
    mapping_path = os.path.join(args.v3_dir, "case_id_mapping.json")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.v3_dir)), "finvault_output_full1064_v3_fixed"
    )

    if not os.path.exists(traj_path):
        print(f"ERROR: {traj_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    pattern = build_redaction_pattern(mapping)

    records = [json.loads(l) for l in open(traj_path, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} v3 records, {len(mapping)} real case_id strings to scrub")

    os.makedirs(output_dir, exist_ok=True)
    out_traj_path = os.path.join(output_dir, "trajectories.jsonl")
    debug_mapping_path = os.path.join(output_dir, "debug_id_mapping.json")

    debug_mapping = {}
    scrub_count = 0
    with open(out_traj_path, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            fixed, original_case_id = fix_record(record, idx, pattern)
            debug_mapping[fixed["case_id"]] = original_case_id
            out_f.write(json.dumps(fixed, ensure_ascii=False) + "\n")

    with open(debug_mapping_path, "w", encoding="utf-8") as f:
        json.dump(debug_mapping, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} fixed records -> {out_traj_path}")
    print(f"Wrote debug id mapping (local reference only, never touches the pipeline) -> {debug_mapping_path}")


if __name__ == "__main__":
    main()
