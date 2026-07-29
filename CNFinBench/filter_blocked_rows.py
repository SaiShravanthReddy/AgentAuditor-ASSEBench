#!/usr/bin/env python3
"""Filter guardrail-blocked conversations out of a CNFinBench evaluation.json, per Anirudhh's
row-index lists (one JSON array of ints per condition x subset, e.g. harmful_MT_App_blocked_rows.json).

Ivan wants AgentAuditor metrics specifically on the conversations that do NOT get blocked by
target-model guardrails, for the Citibank presentation - this produces the filtered input for that.

Join key: each record in the source evaluation.json has explicit `row_index` and `dataset`
(MT_App/MT_Cog/MT_Inter) fields - a record is dropped iff its row_index appears in the blocked-rows
file matching its own dataset. This is NOT positional filtering (list index) - deliberately, since
row_index is an explicit field and using it directly avoids any risk of a record-ordering mismatch
between this source file and whatever Anirudhh indexed against.

Refuses to run (loudly, not silently) if any source record is missing `row_index` - this exact
failure mode was found empirically for one CNFinBench condition's local data before this script was
written; better to fail here than silently do positional filtering against the wrong records.

Usage:
    python CNFinBench/filter_blocked_rows.py <source_evaluation.json> <blocked_rows_dir> <condition> <output.json>

    <condition> must be "harmful" or "harmless" - selects which 3 of the 6 blocked_rows files to use
    (e.g. condition=harmful -> harmful_MT_App_blocked_rows.json, harmful_MT_Cog_blocked_rows.json,
    harmful_MT_Inter_blocked_rows.json).
"""
import argparse
import json
import os
import sys

SUBSETS = ["MT_App", "MT_Cog", "MT_Inter"]


def load_blocked_sets(blocked_rows_dir, condition):
    blocked = {}
    for subset in SUBSETS:
        path = os.path.join(blocked_rows_dir, f"{condition}_{subset}_blocked_rows.json")
        if not os.path.exists(path):
            print(f"ERROR: expected blocked-rows file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            indices = json.load(f)
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            print(f"ERROR: {path} is not a flat JSON array of ints", file=sys.stderr)
            sys.exit(1)
        blocked[subset] = set(indices)
    return blocked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Path to source evaluation.json (must have row_index + dataset fields)")
    parser.add_argument("blocked_rows_dir", help="Directory containing the 6 *_blocked_rows.json files")
    parser.add_argument("condition", choices=["harmful", "harmless"])
    parser.add_argument("output", help="Path to write the filtered evaluation.json")
    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        print(f"ERROR: expected a JSON array in {args.source}, got {type(records)}", file=sys.stderr)
        sys.exit(1)

    missing_row_index = [r.get("id", "?") for r in records if "row_index" not in r]
    if missing_row_index:
        print(f"ERROR: {len(missing_row_index)}/{len(records)} records in {args.source} are missing "
              f"'row_index' - refusing to filter, since that would require unreliable positional "
              f"matching instead. Fetch a source file that has this field for every record. "
              f"Example missing ids: {missing_row_index[:5]}", file=sys.stderr)
        sys.exit(1)

    unknown_subset = [r.get("id", "?") for r in records if r.get("dataset") not in SUBSETS]
    if unknown_subset:
        print(f"ERROR: {len(unknown_subset)} records have a 'dataset' value outside {SUBSETS} - "
              f"can't map them to a blocked-rows file. Example ids: {unknown_subset[:5]}", file=sys.stderr)
        sys.exit(1)

    blocked = load_blocked_sets(args.blocked_rows_dir, args.condition)

    kept = []
    dropped_by_subset = {s: 0 for s in SUBSETS}
    for r in records:
        subset = r["dataset"]
        if r["row_index"] in blocked[subset]:
            dropped_by_subset[subset] += 1
            continue
        kept.append(r)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    total_in = len(records)
    total_dropped = sum(dropped_by_subset.values())
    print(f"{args.condition}: {total_in} records in -> {len(kept)} kept, {total_dropped} dropped (blocked by guardrails)")
    for subset in SUBSETS:
        subset_total = sum(1 for r in records if r["dataset"] == subset)
        print(f"  {subset}: {subset_total} -> {subset_total - dropped_by_subset[subset]} "
              f"({dropped_by_subset[subset]} dropped, {len(blocked[subset])} in blocked-rows file)")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
