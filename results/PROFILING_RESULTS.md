# Profiling Results — CNFinBench + FinVault v5 (gpt-oss-20b)

Batch run on HiPerGator (`hpg-turin`, per-stage timing via the new `AgentAuditor/utils/timer.py`
instrumentation) across all 5 registered CNFinBench variants and 2 priority FinVault v5 comparisons,
on `gpt-oss-20b`.

## Results

Base rate = the accuracy a trivial "always guess the majority class" baseline would score — the
right way to judge whether these numbers reflect real signal or just following the class imbalance.

| Dataset | Question | Accuracy | vs. trivial baseline | Precision | Recall | F1 | AUROC | AUPRC | n (valid/total) |
|---|---|---|---|---|---|---|---|---|---|
| `finvault-v5-fixed-defended-v-attack` | Q1: Did the attack succeed? | **80.9%** | +25.2 pts | 87.2% | 76.9% | 81.8% | 0.835 | 0.825 | 957/957 |
| `finvault-v5-fixed-benign-v-defended`* | Fix 1/2/3 generalization check | **58.9%** | -24.2 pts | 99.1% | 51.1% | 67.4% | 0.756 | 0.917 | 509/510 |
| `finvault-v5-fixed-benign-v-malicious` | Q2: Was it malicious at all? | **46.1%** | -45.6 pts | 100% | 41.3% | 58.4% | 0.735 | 0.956 | 1043/1043 |
| `cnfinbench-pooled` | Pooled harmful+harmless | **73.1%** | +4.8 pts | 66.3% | 31.0% | 42.2% | 0.705 | 0.486 | 579/579 |
| `cnfinbench-harmful` | Harmful subset | **56.0%** | -0.8 pts | 75.8% | 33.1% | 46.1% | 0.617 | 0.652 | 266/266 |
| `cnfinbench-harmful-unblocked` | Harmful, guardrail-filtered | **55.8%** | -3.7 pts | 80.5% | 34.0% | 47.8% | 0.653 | 0.697 | 163/164 |
| `cnfinbench-harmless` | Harmless subset | **89.0%** | +0.0 pts | 0.0% | 0.0% | 0.0% | 0.601 | 0.144 | 308/308 |
| `cnfinbench-harmless-unblocked` | Harmless, guardrail-filtered | **88.1%** | -0.4 pts | 33.3% | 3.3% | 6.1% | 0.562 | 0.139 | 260/261 |

*`finvault-v5-fixed-benign-v-defended` was run separately (burst QoS, ahead of the other 13) as a
targeted check of whether Fix 1/2/3 generalize beyond the dataset they were validated on — not part
of the original 7-dataset scope, included here since the result is relevant to the same question.

### Reading these results

- **Q1 (`defended-v-attack`) is the strongest result here** — 25 points above trivial, consistent
  with FinVault Q1 being the one comparison AgentAuditor has reliably done well on across every
  prior run (v3 and v5 alike).
- **Q2 (`benign-v-malicious`) is the weakest result in this batch, and a regression worth flagging.**
  91.8% of this dataset is actually "malicious," so guessing malicious every time beats this run's
  46.1% by 45.6 points. Recall (41.3%) is close to the *pre-Fix-1/2/3* v3 baseline (43.4%), not the
  post-fix validated number (52.1%) — despite Fix 1/2/3 being merged into `main` and active for this
  run. Worth a closer look at why this specific v5 dataset didn't see the same improvement v3 and
  `benign-v-defended` did.
- **`benign-v-defended` (the Fix 1/2/3 generalization check) landed close to the v3 post-fix
  numbers** (51.1% recall vs. 52.1% validated on v3; 0.756 AUROC vs. 0.764) — good evidence the
  fixes work correctly on a second, previously-untested comparison, even though this dataset also
  scores well below its own trivial baseline.
- **CNFinBench harmless / harmless-unblocked are still essentially at their trivial baseline**
  (0.0 and -0.4 pts) with near-random AUROC (0.601, 0.562) — confirms and extends the previously
  known "near-random ranking on harmless data" finding. Precision/recall of 0% on `harmless` means
  the judge predicted "safe" for every single item in that run.
- **CNFinBench harmful / harmful-unblocked / pooled are all weak-to-marginal** — `pooled` is the
  only one that meaningfully beats its trivial baseline (+4.8 pts); `harmful` and
  `harmful-unblocked` are at or slightly below trivial. This is a new observation: the earlier open
  question was specifically about `harmless`'s near-random AUROC, but this data suggests the weak
  signal isn't isolated to the harmless subset.
- **Processing errors were rare and not a data-quality concern**: 3 items total across all 7 runs
  (2 CNFinBench parsing failures, 1 FinVault judge refusal on adversarial content — investigated
  separately, confirmed to be the judge model declining to engage with an attack-simulation prompt,
  not a pipeline bug).

## Timing (per stage, `hpg-turin` GPU)

All times in seconds unless noted. `preprocess`/`demo_generation`/`infer` are LLM-API-bound;
`cluster`/`infer_emb` are the GPU-accelerated local embedding stages; `demo_repair` is a mix of
local validation + LLM repair calls; `infer_fix1`/`infer_fix2`/`eval` are all sub-second.

| Dataset | preprocess | cluster | demo_gen | demo_repair | infer_emb | infer | Total (approx) |
|---|---|---|---|---|---|---|---|
| `finvault-v5-fixed-defended-v-attack` | 4696.8 | 146.1 | 438.4 | 394.6 | 103.9 | 2281.3 | ~2h 14m |
| `finvault-v5-fixed-benign-v-malicious` | 0.2 | 151.2 | 2093.6 | 190.5 | 3.7 | 3511.1 | ~1h 39m |
| `cnfinbench-pooled` | 1778.6 | 411.9 | 2204.6 | 1010.0 | 479.4 | 4618.8 | ~2h 55m |
| `cnfinbench-harmful` | 812.2 | 211.0 | 1061.6 | 239.3 | 240.5 | 3376.8 | ~1h 39m |
| `cnfinbench-harmful-unblocked` | 0.2 | 119.4 | 880.9 | 206.6 | 3.3 | 2252.8 | ~57m |
| `cnfinbench-harmless` | 560.9 | 207.8 | 184.7 | 221.4 | 245.9 | 851.6 | ~38m |
| `cnfinbench-harmless-unblocked` | 0.3 | 194.0 | 1854.5 | 330.1 | 3.6 | 1084.2 | ~58m |

Several `preprocess` times near-zero (`benign-v-malicious`, `harmful-unblocked`,
`harmless-unblocked`) reflect `preprocess.py`'s caching behavior — it skips the LLM call entirely
when `memory.json` already has every entry from a prior run, which was the case for these reused
repo copies.

## Known issues / next steps

1. **Investigate why `finvault-v5-fixed-benign-v-malicious` (Q2) didn't see the Fix 1/2/3
   improvement** that both v3 `benign-v-malicious` and v5 `benign-v-defended` showed — recall
   (41.3%) is closer to the pre-fix baseline than the validated post-fix number.
2. **CNFinBench's weak-signal problem extends beyond `harmless`** — `harmful`, `harmful-unblocked`,
   and `pooled` are all weak or at-trivial too, not just the previously-flagged `harmless` subset.
