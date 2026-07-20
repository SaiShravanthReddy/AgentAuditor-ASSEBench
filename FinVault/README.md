# FinVault

Running AgentAuditor against [FinVault](https://github.com/aifinlab/FinVault), an
execution-grounded financial-agent safety benchmark (31 sandbox scenarios, tool-calling attack
traces, judged by observable execution outcomes rather than text alone) — structurally quite
different from CNFinBench's plain multi-turn Q&A dialogues.

- `data/` — raw FinVault source data (gitignored). Expects a `trajectories.jsonl` (e.g. from
  HiPerGator's `finvault_output_full1064/`) plus a clone of the public FinVault repo's `sandbox/`
  directory (needed for prompt text — see below).
- `finvault_to_agentauditor.py` — converter from FinVault's schema to AgentAuditor's ASSEBench
  input schema (`id`/`profile`/`contents`/`label`), analogous to
  `CNFinBench/cnfinbench_to_agentauditor.py`. Validated at 1064/1064 (100%) against a real
  HiPerGator export.
- **Metrics and leak detection reuse `CNFinBench/cnfinbench_metrics.py` and
  `CNFinBench/cnfinbench_detect_leakage.py` directly** — both are already dataset-agnostic (their
  CLI takes any `output-k3_corrected.json`/`k3.json` path, nothing CNFinBench-specific is
  hardcoded). No FinVault-specific fork exists yet; only create one if a real FinVault run
  surfaces judge-output-parsing quirks the current extractor doesn't handle (as happened
  iteratively for CNFinBench — see `CNFinBench/RESULTS.md`).
- `RESULTS.md` — results write-up, once a run completes

## Why FinVault's saved output needs a two-source join

FinVault's `trajectories.jsonl` (the actual judged output) has the outcome/label and the agent's
tool-call actions per step, but **not the case's initial prompt text**. That has to be joined back
in from the public FinVault repo's `sandbox/attack_datasets/`,
`sandbox/attack_datasets_synthesis/<technique>/`, and `sandbox/normal_datasets/` files, keyed on
`(scenario_id, case_id, synthesis_technique)`. Getting this join right took three empirically-found
fixes (see the converter's docstring for detail):

1. Synthesized cases have their `case_id` prefixed with `"{technique}__"` in `trajectories.jsonl`
   but not in the dataset file itself.
2. A few scenarios' original attack files use a `"scenarios"` key instead of `"attacks"` for the
   case list (schema isn't 100% consistent across all 31 scenario files).
3. Some scenario files store `"scenario_id"` as a bare int (`19`) instead of a zero-padded string
   (`"09"`) — the join must always derive `scenario_id` from the filename, never from that field,
   or joins silently drop ~10% of records.

## Label mapping

`outcome == "attack_success"` maps to AgentAuditor's unsafe label (1); `"defended"`/`"benign"`
map to safe (0). **Not** the raw `attack_success` boolean field, despite that looking like the
obvious choice: that field is computed generically off "were any vulnerabilities triggered" and
is unreliable specifically for `case_type="normal"` records — verified empirically, 5/107 normal
(benign) cases have `attack_success=True` despite `outcome="benign"` and the raw ternary
`label=0`, which is FinVault's own harness noise on a benign request, not a real attack outcome.
`outcome` agreed with the raw ternary `label` field on all 1064 records with zero exceptions, so
it's the verified-authoritative source. No threshold to choose here, unlike CNFinBench's
continuous HICS score. `is_benign_false_positive` (15/1064 cases where the agent wrongly refused a
*legitimate* request — an over-refusal/utility failure, not a safety failure) is deliberately kept
out of the safety label and preserved in the metadata sidecar instead, for a separate
over-refusal-rate analysis.

## What's in `contents`

Each case becomes one user turn (the joined `attack_prompt`/`query_prompt`) followed by one agent
turn per trajectory step (that step's raw tool-call `action` text). FinVault's saved output does
not include tool observations or a separate reasoning/thought field, so those are not fabricated —
only what's actually present is carried over. This is an execution-grounded sandbox trace (one
prompt, then a sequence of the agent's own tool calls), not an alternating multi-round dialogue
like CNFinBench.

Uses the same shared `AgentAuditor/` pipeline at the repo root (dataset keys prefixed
`finvault-*`, alongside CNFinBench's `cnfinbench-*` keys, per `AgentAuditor/__main__.py`'s
`dataset_fullname` map) — not a separate clone.

**Note on where converted dataset files live:** `finvault_to_agentauditor.py`'s output (the
AgentAuditor-schema JSON) goes to `AgentAuditor/data/finvault-*.json`, *not*
`FinVault/data/`. This isn't an oversight — `AgentAuditor/tasks/*.py` hardcodes its data/temp
paths relative to the package itself (`../data`, `../temp`, resolved from each script's own file
location) in ~22 places across 10 files, with no configurable override. Making that configurable
or symlinking around it was considered and explicitly declined (see project history) in favor of
just keeping `AgentAuditor/data/` and `AgentAuditor/temp/` as the shared location for every
dataset's converted/intermediate files, relying on the `cnfinbench-`/`finvault-` filename prefix
for separation rather than physical folder location. `FinVault/data/` holds only the raw,
unconverted source data.
