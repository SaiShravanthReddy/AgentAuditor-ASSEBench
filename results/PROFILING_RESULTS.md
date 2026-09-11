# Profiling Results — CNFinBench + FinVault v5 (gpt-oss-20b vs. gpt-oss-120b)

Batch run on HiPerGator (`hpg-turin`, per-stage timing via `AgentAuditor/utils/timer.py`) across
all 5 registered CNFinBench variants and 2 priority FinVault v5 comparisons, on both
`gpt-oss-20b` and `gpt-oss-120b`.

**Data-completeness note (read before using the CNFinBench `20b` numbers below)**: `infer.py` has
no resume mechanism — if an API call exhausts its retries, that record is silently dropped rather
than retried, and a full rerun reprocesses everything from scratch rather than filling gaps. All 5
CNFinBench `20b` runs lost records this way (63/13/55/40/16 respectively) during a period of heavy
concurrent load (14 jobs sharing one API key at once). A full-dataset rerun after the load cleared
recovered most but not all of them (46/42/20/11/9 still missing) — there's a baseline API failure
rate even without contention that the fixed 5-retry logic doesn't fully absorb. **A new
`infer_retry` pipeline stage** (`python -m AgentAuditor <dataset> infer_retry`) now exists to
retry just the missing records and merge them in, rather than reprocessing everything again — not
yet run to completion at time of writing. Treat the 5 CNFinBench `20b` rows as directionally
correct but not final. FinVault v5 and all `120b` CNFinBench runs are fully complete.

## Results

"vs. trivial baseline" = accuracy minus what a baseline that always guesses the majority class
would score — the right way to judge whether a number reflects real signal or just follows the
class imbalance.

| Dataset | Question | Model | Accuracy | vs. trivial | Precision | Recall | F1 | AUROC | AUPRC | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| `finvault-v5-fixed-defended-v-attack` | Q1: Did the attack succeed? | 20b | 80.9% | +25.2 | 87.2% | 76.9% | 81.8% | 0.835 | 0.825 | 957/957 |
| `finvault-v5-fixed-defended-v-attack` | Q1 | 120b | **83.6%** | **+27.9** | 87.9% | 81.9% | 84.8% | 0.893 | 0.890 | 952/957 |
| `finvault-v5-fixed-benign-v-malicious` | Q2: Was it malicious at all? | 20b | 46.1% | −45.7 | 100% | 41.3% | 58.4% | 0.735 | 0.956 | 1043/1043 |
| `finvault-v5-fixed-benign-v-malicious` | Q2 | 120b | **81.1%** | −10.6 | 99.2% | **80.0%** | 88.6% | **0.920** | 0.989 | 1040/1043 |
| `cnfinbench-pooled` | Pooled harmful+harmless | 20b⚠️ | 72.8% | +4.5 | 66.7% | 28.6% | 40.0% | 0.674 | 0.464 | 596/642 |
| `cnfinbench-pooled` | Pooled | 120b | 73.4% | +5.6 | 65.5% | 36.7% | 47.1% | 0.721 | 0.539 | 642/642 |
| `cnfinbench-harmful` | Harmful subset | 20b⚠️ | 55.0% | −0.4 | 71.6% | 31.2% | 43.4% | 0.618 | 0.633 | 278/321 |
| `cnfinbench-harmful` | Harmful | 120b | **64.2%** | **+10.9** | 88.9% | 37.4% | 52.7% | 0.723 | 0.739 | 321/321 |
| `cnfinbench-harmful-unblocked` | Harmful, guardrail-filtered | 20b⚠️ | 58.7% | ~0.0 | 82.2% | 37.8% | 51.8% | 0.632 | 0.696 | 167/169 |
| `cnfinbench-harmful-unblocked` | Harmful-unblocked | 120b | 57.2% | −1.7 | 82.2% | 34.9% | 49.0% | 0.725 | 0.756 | 180/180 |
| `cnfinbench-harmless` | Harmless subset | 20b⚠️ | 89.1% | 0.0 | 0.0% | 0.0% | 0.0% | 0.635 | 0.158 | 312/312 |
| `cnfinbench-harmless` | Harmless | 120b | 84.7% | −4.1 | 27.6% | 22.2% | 24.6% | 0.596 | 0.168 | 321/321 |
| `cnfinbench-harmless-unblocked` | Harmless, guardrail-filtered | 20b⚠️ | 88.3% | −0.7 | 25.0% | 3.2% | 5.7% | 0.550 | 0.132 | 281/281 |
| `cnfinbench-harmless-unblocked` | Harmless-unblocked | 120b | 74.1% | **−14.3** | 13.6% | 22.9% | 17.0% | 0.566 | 0.139 | 301/301 |

⚠️ = incomplete data, see the note above.

### Reading these results

- **120b beats 20b almost everywhere, most dramatically on Q2**: recall 41.3%→80.0%, AUROC
  0.735→0.920 — the single biggest jump in the whole batch, and it moves Q2 from "badly
  miscalibrated" to genuinely strong. The "judge defaults to safe/benign" bias present on `20b`
  (see `results/AGENTAUDITOR_DIAGNOSIS.md` and recent investigation) is measurably weaker on `120b`
  in most cases.
- **One exception worth flagging**: `harmless-unblocked` on `120b` is *worse* than `20b` relative
  to its trivial baseline (−14.3 vs. −0.7 pts), despite better recall — `120b` produced far more
  false positives here (51 vs. 3), trading precision for recall in a way that hurt overall accuracy
  on this specific dataset. Model upgrade isn't a uniform win; worth checking per-dataset, not just
  in aggregate.
- **Q1 (`defended-v-attack`) remains the strongest, most reliable result on both models** — the one
  comparison AgentAuditor has done well on consistently across every run to date (v3, v5, both
  models).
- **CNFinBench harmless/harmless-unblocked are still weak on both models**, though `120b` shows
  *some* real signal on `harmless` (22.2% recall vs. 0.0% on `20b`) — the near-total "always predict
  safe" behavior isn't universal to the model, just more pronounced on `20b`.

## A separate, likely long-standing bug found while investigating this batch

**`infer_llm_repair.py`'s `fix2_main()` has a hardcoded target-ID list**:
```python
ERROR_IDS_TO_FIX = [83, 102, 163, 189, 415, 648, 1438, 1492]
```
These are bare integers, leftover from some earlier dev/test dataset. Every dataset in this repo
uses string IDs (e.g. `"harmless-MT_App-1"`), so `item_id in error_id_set` is essentially always
`False` — **this LLM-based repair fallback has likely never actually corrected anything, on any
run, ever.** Circumstantial evidence: `infer_fix2` consistently takes 0.06–2.9 seconds across every
run in this batch regardless of dataset size — too fast for a stage meant to make real LLM repair
calls. Not yet fixed — flagging here since it affects every dataset this repo has ever evaluated,
not just this batch. The fix is straightforward: compute the target IDs dynamically from whichever
items are still malformed after `infer_json_repair.py`'s mechanical pass, instead of the hardcoded
list.

## Timing (mean seconds per stage, `hpg-turin`)

`preprocess`/`demo_generation`/`infer` are LLM-API-bound; `cluster`/`infer_emb` are GPU-accelerated
local embedding stages; `demo_repair` mixes local validation + LLM repair calls.

| Dataset | Model | preprocess | cluster | demo_gen | demo_repair | infer_emb | infer |
|---|---|---|---|---|---|---|---|
| `defended-v-attack` | 20b | 4696.8 | 146.1 | 438.4 | 394.6 | 103.9 | 2281.3 |
| `defended-v-attack` | 120b | 5615.1 | 144.8 | 430.7 | 357.3 | 3.6 | 3479.0 |
| `benign-v-malicious` | 20b | ~1080¹ | 151.2 | 2093.6 | 190.5 | 3.7 | 3511.1 |
| `benign-v-malicious` | 120b | 5929.4 | 152.8 | 386.3 | 368.2 | 3.7 | 3130.1 |
| `pooled` | 20b | 1778.6 | 411.9 | 2204.6 | 1010.0 | 479.4 | ~4254¹ |
| `pooled` | 120b | 3576.3 | 409.0 | 1102.7 | 508.4 | 5.2 | 2965.4 |
| `harmful` | 20b | 812.2 | 211.0 | 1061.6 | 239.3 | 240.5 | ~3223¹ |
| `harmful` | 120b | 1830.5 | 205.6 | 472.5 | 461.4 | 3.7 | 2403.3 |
| `harmful-unblocked` | 20b | ~449¹ | 119.4 | 880.9 | 206.6 | 3.3 | ~2521¹ |
| `harmful-unblocked` | 120b | 961.6 | 131.2 | 315.1 | 232.5 | 119.8 | 1821.4 |
| `harmless` | 20b | 560.9 | 207.8 | 184.7 | 221.4 | 245.9 | ~1067¹ |
| `harmless` | 120b | 1334.3 | 208.5 | 491.0 | 485.7 | 3.6 | 2430.3 |
| `harmless-unblocked` | 20b | ~255¹ | 194.0 | 1854.5 | 330.1 | 3.6 | ~1222¹ |
| `harmless-unblocked` | 120b | 1250.8 | 193.5 | 463.4 | 506.5 | 3.5 | 2130.4 |

¹ Averaged across the original run + the incomplete-data correction rerun (`infer_retry` will give
a final single-run number once used).

**Bottleneck pattern**: `infer` (the actual judging calls) dominates wall time everywhere —
1000–3500+ seconds regardless of model — this is the real target for speed optimization, not the
embedding/clustering stages, which are consistently under 500 seconds. `infer_emb`'s wide variance
on `120b` (3.6s to 119.8s across otherwise-similar datasets) is worth a closer look — possibly some
`120b` jobs didn't land on GPU nodes.

## Known issues / next steps

1. **Run `infer_retry` on the 5 incomplete CNFinBench `20b` datasets** to finalize the comparison.
2. **Fix `infer_llm_repair.py`'s hardcoded `ERROR_IDS_TO_FIX`** — likely improves every dataset's
   completeness, not just this batch's.
3. **Investigate why `finvault-v5-fixed-benign-v-malicious` (Q2) `20b` recall (41.3%) sits below the
   validated post-fix baseline (52.1%)** while `120b` on the same dataset (80.0%) far exceeds it —
   still open whether this is model-capability-driven or something dataset-specific to this v5 run.
4. **`harmless-unblocked`'s `120b` regression relative to trivial baseline** — check whether this
   is a one-off or a real precision/recall tradeoff pattern worth understanding before treating
   `120b` as a strict upgrade.
