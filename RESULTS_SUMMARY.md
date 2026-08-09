# Summary: FinVault (2 analyses) + CNFinBench (filtered vs. unfiltered)

*Full detail and reproduction steps: `FinVault/FinVault_RESULTS.md`, `CNFinBench/RESULTS.md`*

---

## FinVault — 2 analyses (per Anirudhh/Ivan's framing)

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
