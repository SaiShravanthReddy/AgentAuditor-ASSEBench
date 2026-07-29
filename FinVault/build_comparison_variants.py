#!/usr/bin/env python3
"""Build the 4 pairwise FinVault comparison datasets from the already leak-fixed, converted
finvault-v3-fixed.json + its metadata sidecar - no need to re-run the leak fix or converter, since
that data's `label` field already encodes `outcome == "attack_success"` correctly for the 3 variants
that involve attack_success.

Variants:
  - full              : all 1064 records, label as already converted (benign+defended=0, attack_success=1)
  - defended_v_attack  : defended + attack_success only (957 records), label unchanged (0/1 already correct)
  - benign_v_attack    : benign + attack_success only (643 records), label unchanged (0/1 already correct)
  - benign_v_defended  : benign + defended only (528 records) - NEITHER outcome involves real harm, so
    there is no natural safe/unsafe mapping here (see FinVault/README.md's "Benign vs Defended framing"
    note). Built with defended=1/benign=0 (attack-attempt-detection framing) as a default, but NOT
    intended to be run until Prof. Ivan confirms the framing - see --variant benign_v_defended's
    output filename, which is suffixed _PENDING_CONFIRMATION to make this unmistakable rather than
    relying on someone remembering not to launch it.

Usage:
    python FinVault/build_comparison_variants.py
    (reads AgentAuditor/data/finvault-v3-fixed.json + .meta.json, writes 4 new datasets under
    AgentAuditor/data/)
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "AgentAuditor", "data")

SOURCE_JSON = os.path.join(DATA_DIR, "finvault-v3-fixed.json")
SOURCE_META = os.path.join(DATA_DIR, "finvault-v3-fixed.json.meta.json")


def load_source():
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(SOURCE_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return records, meta


def write_variant(name, records, meta, filtered_ids, relabel_fn=None):
    out_records = []
    out_meta = {}
    for r in records:
        if r["id"] not in filtered_ids:
            continue
        rec = dict(r)
        if relabel_fn is not None:
            rec["label"] = relabel_fn(meta[r["id"]])
        out_records.append(rec)
        out_meta[r["id"]] = meta[r["id"]]

    out_path = os.path.join(DATA_DIR, f"{name}.json")
    meta_path = os.path.join(DATA_DIR, f"{name}.json.meta.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=False)

    n_unsafe = sum(1 for r in out_records if r["label"] == 1)
    print(f"{name}: {len(out_records)} records -> {out_path}")
    print(f"  label=1: {n_unsafe}/{len(out_records)} ({100*n_unsafe/len(out_records):.1f}%)" if out_records else "  (empty!)")


def main():
    records, meta = load_source()
    outcome_of = {rid: m["outcome"] for rid, m in meta.items()}

    ids_by_outcome = {"benign": set(), "defended": set(), "attack_success": set()}
    for rid, outcome in outcome_of.items():
        ids_by_outcome.setdefault(outcome, set()).add(rid)

    print(f"Source: {len(records)} records "
          f"(benign={len(ids_by_outcome['benign'])}, defended={len(ids_by_outcome['defended'])}, "
          f"attack_success={len(ids_by_outcome['attack_success'])})\n")

    # 1. full: benign+defended=0 vs attack_success=1 - identical to source, just copied under an
    #    explicit name so it sits alongside the other 3 variants consistently.
    all_ids = set(outcome_of.keys())
    write_variant("finvault-v3-fixed-full", records, meta, all_ids)

    # 2. defended vs attack_success - label already correct (0/1), just filter.
    write_variant(
        "finvault-v3-fixed-defended-v-attack",
        records, meta,
        ids_by_outcome["defended"] | ids_by_outcome["attack_success"],
    )

    # 3. benign vs attack_success - label already correct (0/1), just filter.
    write_variant(
        "finvault-v3-fixed-benign-v-attack",
        records, meta,
        ids_by_outcome["benign"] | ids_by_outcome["attack_success"],
    )

    # 4. benign vs defended - NEITHER outcome is real harm; default framing (defended=1) applied but
    #    filename flags this as pending confirmation, not ready to launch.
    write_variant(
        "finvault-v3-fixed-benign-v-defended_PENDING_CONFIRMATION",
        records, meta,
        ids_by_outcome["benign"] | ids_by_outcome["defended"],
        relabel_fn=lambda m: 1 if m["outcome"] == "defended" else 0,
    )
    print("\nNOTE: benign-v-defended uses defended=1/benign=0 (attack-attempt-detection framing) as a "
          "DEFAULT ONLY - do not register or run this dataset until Prof. Ivan confirms the framing "
          "(see the question drafted in conversation). Filename is suffixed accordingly.")


if __name__ == "__main__":
    main()
