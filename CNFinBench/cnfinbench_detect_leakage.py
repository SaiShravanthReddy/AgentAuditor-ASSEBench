#!/usr/bin/env python3
"""Detect items whose own dialogue was retrieved as one of their own few-shot demonstrations.

AgentAuditor's infer_emb.py (`create_fewshot_dataset`) searches every item in the full dataset
against a demo pool built from FINCH cluster representatives, with no id-based self-exclusion. If
a cluster representative's own embedding is (near-)identical between the query pass and the demo
pool, it can retrieve itself as a "prior example" - showing the model its own dialogue with its
own true-label-justified reasoning already spelled out, right before asking it to judge that same
dialogue.

This is NOT guaranteed to happen for every representative (verified empirically, not assumed):
whether self-similarity survives depends on whether the reference-side content (demo_fixed.json,
built during the demo/demo_repair stages) still matches the query-side content (the original
converted dataset) closely enough. In practice this varied a lot between conditions in the
CNFinBench run - severe in one condition, partial in another, absent in a third - so each item is
checked individually here rather than assuming "representative -> leaked".

Detection method: for each item, compare its own dialogue turns against each of its retrieved
fewshot_demos' Q text. A turn "matches" if a 150-character prefix of its content appears verbatim
in the demo Q. If >=80% of an item's own turns match within a single demo, that item is flagged as
genuinely self-leaked.

Usage (run from the repo root):
    python CNFinBench/cnfinbench_detect_leakage.py AgentAuditor/temp/<dataset>/k3.json leaked_ids_<name>.json
"""
import argparse
import json

MATCH_THRESHOLD = 0.8
PREFIX_LEN = 150
MIN_TURN_LEN = 40


def genuinely_leaked_ids(k3_path):
    with open(k3_path, "r", encoding="utf-8") as f:
        k3 = json.load(f)

    leaked = set()
    for item in k3:
        item_id = item.get("id")
        turns = item["contents"][0] if item.get("contents") else []
        if not turns:
            continue
        for demo in item.get("fewshot_demos", []):
            q = demo.get("Q", "")
            matches = sum(
                1 for t in turns
                if t.get("content") and len(t["content"]) > MIN_TURN_LEN and t["content"][:PREFIX_LEN] in q
            )
            if matches / len(turns) >= MATCH_THRESHOLD:
                leaked.add(item_id)
                break
    return leaked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("k3_json", help="Path to AgentAuditor's temp/<dataset>/k3.json")
    parser.add_argument("output", help="Path to write the JSON array of genuinely self-leaked ids")
    args = parser.parse_args()

    leaked = genuinely_leaked_ids(args.k3_json)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sorted(leaked), f, indent=2)
    print(f"{len(leaked)} genuinely self-leaked items found in {args.k3_json} -> {args.output}")


if __name__ == "__main__":
    main()
