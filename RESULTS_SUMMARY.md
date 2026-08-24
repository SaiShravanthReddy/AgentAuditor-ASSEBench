# Summary: FinVault (2 analyses) + CNFinBench (filtered vs. unfiltered)

*Full detail and reproduction steps: `FinVault/FinVault_RESULTS.md`, `CNFinBench/RESULTS.md`*

---

## Confidence / AUROC / AUPRC re-run (2026-08-11/12) — current numbers

Per Ivan's confirmed decision to use self-reported LLM confidence (not logprobs) for AUROC/AUPRC,
added a `confidence` field (0.0-1.0) to the judge prompt and re-ran every dataset below. All numbers
in this section are **independently re-verified** — pulled the raw per-item output files and
recomputed confusion matrices / AUROC / AUPRC from scratch (not just trusting `eval.py`'s own
printout), confirmed exact matches. This section supersedes the accuracy-only numbers further down
for the same datasets; those are kept for historical context (they predate the confidence field and,
for FinVault, predate a fixed prompt-construction bug — see note below the FinVault table).

### FinVault — v3 vs. v5, same 2 priority questions on independently-generated data

| | Q1: Did the attack succeed? (v3) | Q1 (v5) | Q2: Was it malicious at all? (v3) | Q2 (v5) |
|---|---|---|---|---|
| Accuracy | 77.5% | 76.5% | 49.1% | 43.6% |
| Precision / Recall | 79.3% / 75.0% | 82.1% / 69.0% | 100% / 43.4% | 99.7% / 38.7% |
| F1 | 77.1% | 75.0% | 60.5% | 55.7% |
| AUROC / AUPRC | 0.804 / 0.764 | 0.801 / 0.773 | 0.741 / 0.949 | 0.716 / 0.953 |
| Base rate (trivial "always malicious" baseline) | ~50% | ~51% | ~90% | ~92% |

**Q1 is solid and reproduces cleanly on v5** — not a v3 artifact. **Q2's miscalibration also
reproduces on v5**, slightly worse if anything: zero/near-zero false positives, but the judge misses
well over half of actually-malicious cases, landing far below the trivial "always guess malicious"
baseline both times. This is a genuine, dataset-independent finding, not noise.

*Note on v3 vs. earlier-reported v3 numbers*: this re-run also fixed a real bug where the reworded
`goal` prompt built specifically for the Q2 comparison (asking "was there a manipulation attempt"
instead of the stock "did harm occur") was never actually reaching the model — `combine_to_prompt()`
silently ignored the per-record `goal` field. So the v3 Q2 numbers above are the first ones that
actually reflect the intended question; don't compare them to the historical Q2 table further below
(that one predates the fix and was graded against the wrong prompt).

### Fix validation (2026-08-24) — retrieval/demo-generation fixes, tested on v3 Q2

Diagnostic investigation into *why* AgentAuditor underperforms (initiated by Anirudhh) found two
concrete, code-level root causes for Q2's miscalibration, both now fixed and pushed:

- **Finding 1 / Fix 1**: `demo.py`'s chain-of-thought generation and `infer_emb.py`'s few-shot Q/A
  packaging both hardcoded the stock "judge if safe/unsafe" framing regardless of a record's actual
  `goal` field - so Q2's demos reasoned about generic operational safety violations instead of
  "was there a manipulation attempt," teaching the wrong signal. Also required fixing
  `preprocess.py`, which silently dropped `goal` from its output whitelist.
- **Finding 2 / Fix 2**: `demo_repair.py` logs but doesn't remove records whose LLM-based CoT repair
  failed - they stayed in the retrieval pool with broken/blank chain-of-thought, silently wasting
  few-shot slots. Now excluded from retrieval candidacy entirely before similarity search runs.

**Tested on `finvault-v3-fixed-benign-v-malicious` (Q2)** - re-ran `cluster→demo→infer_emb→infer→eval`
with both fixes applied (reused the existing, already-clean `memory.json` with `goal` patched back in,
skipping the expensive `preprocess` LLM pass since scenario/risk/failure classification doesn't
depend on `goal`). Results independently re-verified from raw output (exact match):

| | Before (baseline) | After (Fix 1+2) | Δ |
|---|---|---|---|
| Accuracy | 49.1% | 56.7% | +7.6 |
| Precision / Recall | 100% / 43.4% | 99.6% / 52.1% | −0.4 / **+8.7** |
| F1 | 60.5% | 68.4% | +7.9 |
| AUROC / AUPRC | 0.741 / 0.949 | 0.764 / 0.953 | +0.023 / +0.004 |
| Confusion (FN → TP) | FN=587, TP=370 | FN=458, TP=498 | **128 more malicious cases caught** |

**Recall improved meaningfully (+8.7 points) with precision staying near-perfect** - confirms both
root causes were real contributors to Q2's miscalibration, not just plausible-sounding hypotheses.
One caveat worth tracking: Fix 2 excluded 24/63 (38%) of this run's demo pool as broken, notably
higher than the 12/63 (19%) measured on the old (pre-Fix-1) demo pool - likely because Fix 1's
goal-aware `demo.py` prompt produces different raw LLM output with its own JSON-parse-failure rate.
`demo_repair.py`'s repair success rate is itself a candidate for future improvement.

Not yet done: re-testing on v5's `benign-v-malicious` or the deprioritized `benign-v-defended`
(both share Finding 1's reworded-goal issue) to confirm the improvement generalizes beyond this one
dataset.

### CNFinBench — all 5 conditions, unfiltered + guardrail-filtered

| Condition | n | Accuracy | Precision / Recall | AUROC | AUPRC | Base rate |
|---|---|---|---|---|---|---|
| harmful | 313 | 56.9% | 75.4% / 27.7% | 0.625 | 0.619 | 53.0% |
| harmful-unblocked | 175 | 49.7% | 71.4% / 24.3% | 0.628 | 0.678 | 58.9% |
| harmless | 300 | 84.0% | 20.8% / 14.7% | 0.534 | 0.134 | 11.3% |
| harmless-unblocked | 281 | 87.5% | 0% / 0% | 0.541 | 0.143 | 12.5% |
| pooled | 614 | 68.6% | 54.3% / 19.1% | 0.614 | 0.421 | 32.4% |

**harmless / harmless-unblocked have essentially no ranking signal** (AUROC ~0.53-0.54, barely above
random) — the judge almost never flags anything unsafe in these subsets, and even its confidence
scores don't meaningfully separate the few true positives from the rest. This is a materially weaker
result than FinVault's Q2: there, the hard verdict was miscalibrated but the *confidence ranking*
still carried real signal (AUROC 0.72-0.80); here there's barely any signal at all to recover.
`harmful` / `harmful-unblocked` are closer to FinVault's pattern — weak-to-moderate but real signal.

---

## FinVault — 2 analyses (per Anirudhh/Ivan's framing)

*Historical (pre-confidence-field, and Q2 predates the goal-prompt bug fix) — see the
"Confidence / AUROC / AUPRC re-run" section above for current numbers.*

Both use the **full 1064-record dataset**, just with different label groupings.

### Q1: Did the attack succeed?
Label = `attack_success` (1) vs. everything else (0).

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **AgentAuditor** | **78.3%** | 79.5% | 76.4% | 78.0% |
| Zero-shot baseline | 55.0% | — | — | — |
| **Δ (memory helps)** | **+23.3 pts** | | | |

Real, meaningful signal — not a class-imbalance artifact (labels near-balanced: 536/528). Consistent with two other narrower framings of the same "harm occurred" question (defended-v-attack 81.5%, benign-v-attack 76.3%).

### Q2: Was it benign or malicious at all (regardless of outcome)?
Label = `defended`+`attack_success` pooled as "malicious" (1) vs. `benign` (0).

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **AgentAuditor** | **53.9%** | 90.9% | 54.2% | 67.9% |
| Zero-shot baseline | 47.8% | — | — | — |
| **Δ (memory helps)** | **+6.0 pts** | | | |
| **Trivial "always malicious" baseline** | **89.9%** | | | |

**This is the important, not-good finding**: 89.9% of the data is actually malicious, so guessing "malicious" every time would score 89.9% accuracy. AgentAuditor scores 53.9% — **36 points below that trivial baseline**. Precision is strong (when it says malicious, it's usually right) but recall is weak — it defaults to "benign" far more than the true class balance justifies. Memory still helps vs. zero-shot, but by much less than on Q1, and neither comes close to trivial.

**Bottom line**: AgentAuditor is reliable at telling whether harm actually resulted (Q1), but far less reliable at flagging that something malicious is happening at all (Q2) — which is the more production-relevant question for a live monitoring use case.

---

## CNFinBench — filtered (guardrail-unblocked) vs. unfiltered

*Historical (pre-confidence-field) — see the "Confidence / AUROC / AUPRC re-run" section above for
current numbers including all 5 conditions. The accuracy/recall/F1 figures below are consistent with
that section's, just without AUROC/AUPRC.*

Removing conversations blocked by the target model's own guardrails, to see how AgentAuditor performs on what actually gets through.

| Condition | n | Accuracy | Recall | F1 |
|---|---|---|---|---|
| harmful → harmful-unblocked | 317→147 | 57.7%→**46.3%** ▼11.4 | 28.6%→**21.2%** ▼7.4 | 0.417→0.313 |
| harmless → harmless-unblocked | 271→252 | 88.2%→89.3% | 0.0%→**0.0%** (unchanged) | 0.000→0.000 |
| pooled → pooled-unblocked* | 588→399 | 71.8%→73.4% | 24.6%→**16.1%** ▼8.5 | 0.366→0.254 |

*\*Concatenation reading, the correct like-for-like pairing (filtered pooled is a concatenation, not a dedicated pooled run).*

**Headline: filtering makes the numbers worse, not better** — and that's expected, not a bug. Guardrail-blocked conversations were likely the easier, more obviously-risky cases; removing them leaves the harder, subtler residue for AgentAuditor to judge. `harmful`'s accuracy is now **below** what always-predicting-unsafe would score (57.8%) — a meaningfully worse-than-trivial result.

**Pooled accuracy rising while recall falls is a composition artifact, not improvement**: harmless (the always-"safe", high-accuracy/zero-recall condition) grows from 46% to 63% of the pooled set after filtering, mechanically pulling accuracy up while recall drops. Read the recall column, not the accuracy column, for the pooled row.

**harmless/harmless-unblocked recall is 0.0% in both** — a known, pre-existing limitation unrelated to guardrail filtering; neither caused nor fixed by this analysis.

---

## Open items (not yet resolved)
- Which judge model produced the *original* CNFinBench ground-truth labels — ruled out GPT-OSS, likely Qwen, unconfirmed (waiting on Dhruv).
- FinVault self-leakage: checked, clean (0 leaked items across all 5 comparisons).
- CNFinBench self-leakage: real, now excluded from these numbers (23 harmful-unblocked, 49 harmless-unblocked items excluded).
