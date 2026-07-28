# Results: AgentAuditor (gpt-oss-20b) on FinVault v3-fixed

## Setup

- **Model**: `gpt-oss-20b`, served via UF's AI gateway (OpenAI-compatible endpoint), used for every
  LLM-calling pipeline stage (preprocess, demo, demo_repair, infer, infer_fix2).
- **Data**: `finvault_output_full1064_v3_fixed` — see `FinVault/README.md`'s "Data versions" and
  "Leakage investigation" sections for how this was derived from Xiaoyu's v3 export and what was
  fixed. 1064 records (536 unsafe / 528 safe by the label definition below).
- **Ground truth**: `outcome == "attack_success"` → unsafe (label 1); `"defended"`/`"benign"` →
  safe (label 0). Not the raw `attack_success` boolean — see `FinVault/README.md`'s "Label mapping"
  section for why.
- **Conversion**: `FinVault/finvault_to_agentauditor.py`, run against the leak-fixed data with its
  built-in guard confirming no real case_id survived.
- **Pipeline runtime**: all 5 LLM-calling stages completed in 94.6 minutes total (preprocess 23.1m,
  cluster 12.6m, demo 5.4m, infer_emb 10.0m, infer 43.5m) — well under the ~286m estimate
  extrapolated from CNFinBench's per-record rates, run via `FinVault/run_finvault_pipeline.sh`.

## Validity notes (read before trusting the numbers below)

### 1. Output-parsing coverage — a real bug, found and fixed, not a data problem

The first `eval` run reported **1004/1064 (94%) "processing errors"** — alarming enough to
investigate before reporting anything. Root cause, confirmed by running `eval.py`'s
`extract_output()` directly against every record's real judge output rather than guessing: the
judge's verdict was present in every case, just under one of **50+ different key names**
(`final_decision`, `result`, `decision`, `safety`, `safe`, `unsafe`, `conclusion`,
`safety_assessment`, etc.) that `eval.py`'s hardcoded key list didn't recognize — `gpt-oss-20b`
did not reliably follow the requested output schema, matching the same failure mode CNFinBench hit
(see `CNFinBench/RESULTS.md`'s "Output-parsing coverage" section).

Fixed in `AgentAuditor/tasks/eval.py`'s `extract_output()`:
- Added the confirmed-real key names to the recognized list (purely additive — the original list is
  untouched, so this can't change behavior for other datasets' eval runs).
- Fixed a logic gap: the original code returned on the *first* matching key name even if that key's
  *value* didn't actually parse (e.g. a record with both a working `safety: 0` and a non-numeric
  prose `final_decision` sentence would previously fail entirely, because `final_decision` was
  checked first and its value didn't parse). Now falls through to the next candidate key instead of
  giving up on the whole record.
- Before trusting any of this, spot-checked ambiguous cases directly — e.g. confirmed `"safe": 0`
  co-occurring with reasoning that concluded "the operation is safe" (so `0` means what the
  pipeline's convention says it means, not an inverted boolean) — rather than assuming key-name
  similarity implies matching semantics.

Result: **1058/1064 (99.44%) now parse successfully.** The remaining 6 are genuinely malformed
outputs (raw markdown text instead of JSON, or a nested dict/garbled tool-call-shaped payload
instead of a scalar verdict) — correctly excluded, not a parser gap. Inspected individually to
confirm before excluding.

### 2. Few-shot self-leakage — not yet checked (open item)

`cnfinbench_detect_leakage.py` (checks whether a cluster representative retrieves its own dialogue
as a few-shot demo during `infer_emb`, inflating its own score) is dataset-agnostic and applies here
too, but **has not been run against this FinVault run yet**. CNFinBench's own numbers (see its
RESULTS.md) show this leakage was severe in one condition (94% of representatives) and entirely
absent in another — not something to assume either way. Treat the numbers below as **preliminary
until this check runs.**

### 3. Other limitations not corrected for (disclosed, not fixed)

- **No baseline comparison**: `direct_eval` (zero-shot, no memory/demos) has not been run against
  this dataset, so it's unknown whether AgentAuditor's memory/RAG apparatus helps, hurts, or is
  neutral here.
- **Single run, temperature 0, no repeats**: no variance/confidence interval on any number below.
- **FinVault's own label has an inherent noise floor**: cross-checking v1 vs. v2 (independent
  rollouts of the same cases) showed ~2.5% of attack-case labels flip between runs due to genuine
  agent behavioral nondeterminism, not scoring noise (see `FinVault/README.md`). No detector's
  accuracy against this ground truth can be expected to exceed that floor.

## Results

| n | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| 1058 | 0.674 | 0.740 | 0.538 | 0.623 |

**Confusion matrix:**

| | Predicted Safe | Predicted Unsafe |
|---|---|---|
| **Actually Safe** | 428 (TN) | 100 (FP) |
| **Actually Unsafe** | 245 (FN) | 285 (TP) |

## Key takeaways

1. **The judge is conservative, not aggressive.** Precision (74.0%) is meaningfully higher than
   recall (53.8%) — when it flags something unsafe, it's usually right, but it misses close to half
   of the actually-unsafe cases (245 false negatives) rather than over-flagging safe ones (100 false
   positives). Opposite failure mode from CNFinBench's harmless condition, where the model
   effectively never predicted unsafe at all (0% recall) — here it does flag unsafe cases, just not
   enough of them.
2. **Not close to a class-imbalance artifact.** Labels are near-balanced (536 unsafe / 528 safe),
   so the 67.4% accuracy reflects real discriminative signal, not a "predict majority class" trick
   the way CNFinBench's harmless-condition accuracy did.
3. **Two known open items before this is a final number**: self-leakage check (validity note #2)
   and a `direct_eval` baseline comparison (validity note #3) — recommend running both before citing
   this as a finished result.

## Reproducing this

```bash
python3 FinVault/fix_v3_leakage.py FinVault/data/finvault_output_full1064_v3_anonymized FinVault/data/finvault_output_full1064_v3_fixed
python3 FinVault/finvault_to_agentauditor.py FinVault/data/finvault_output_full1064_v3_fixed/trajectories.jsonl AgentAuditor/data/finvault-v3-fixed.json --run-name finvault-v3-fixed
python3.11 -m AgentAuditor finvault-v3-fixed preprocess
python3.11 -m AgentAuditor finvault-v3-fixed cluster
python3.11 -m AgentAuditor finvault-v3-fixed demo
python3.11 -m AgentAuditor finvault-v3-fixed infer_emb
python3.11 -m AgentAuditor finvault-v3-fixed infer
python3.11 -m AgentAuditor finvault-v3-fixed eval
```

Requires Python ≥3.10 (repo uses `match`/`case` in `AgentAuditor/__main__.py`) and
`AGENTAUDITOR_API_KEY`/`AGENTAUDITOR_API_BASE`/`AGENTAUDITOR_MODEL_*` set via `.env`. On a machine
where the default `python3`/`python` resolves to <3.10, use an explicit `python3.11` (or newer)
interpreter for every step above, as done here.
