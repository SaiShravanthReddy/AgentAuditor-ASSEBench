# gpt-oss-20b vs. gpt-oss-120b — Summary for the Team

Headline: **120b is a real, substantial upgrade — but not a clean one.** Full data and per-dataset
tables in `PROFILING_RESULTS.md`; this is the shareable summary.

## The numbers that matter most

- **Biggest win**: FinVault Q2 (`benign-v-malicious`, "was it malicious at all?") — accuracy
  46.1%→81.1%, recall 41.3%→80.0%, AUROC 0.735→0.920. The single largest jump in the whole batch.
- **Q1 stays strong on both models** (80.9%→83.6%) — the one comparison AgentAuditor has reliably
  done well on regardless of model choice.
- **One real regression**: CNFinBench `harmless-unblocked` got *worse* relative to its trivial
  baseline on 120b (−0.7 pts → −14.3 pts), driven by false positives jumping from 3 to 51. Model
  upgrade is not a uniform win — check per-dataset, not just the average.

## Full results table

"vs. trivial" = accuracy minus what always-guess-the-majority-class would score — the honest way
to tell real signal from just following class imbalance. ⚠️ = CNFinBench `20b` rows still had some
records missing from API failures during a high-contention batch run at time of measurement (see
`PROFILING_RESULTS.md` for the full data-completeness note) — directionally correct, not final.

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


## Why: 120b learned a real pattern, and over-applies it

Both the win and the regression trace back to the same mechanism. Reading real records showed
120b learned to recognize **"cumulative disclosure escalation"** — a long conversation (24 turns)
that starts with generic questions and steadily narrows into specific technical/security detail —
as a risk signal. That's genuine progress: it catches `MT_App` true positives that `20b` mostly
misses (both models had confidently missed these before). But it over-applies the same pattern to
`MT_Inter` conversations that are structurally similar yet ground-truth-labeled benign, which is
what drives the `harmless-unblocked` false-positive surge.

**Open question for the team**: is the `MT_App`/`MT_Inter` distinction a real, learnable model
capability gap, or is the label boundary itself hard to draw consistently? Worth a conversation
before assuming this is purely a model-quality issue.

## Confidence and threshold findings

- Q2's self-reported confidence score is a strong, genuine signal on **both** models (mean
  confidence for correct vs. wrong predictions: 0.78 vs. 0.05 on 20b, 0.88 vs. 0.09 on 120b) — but
  the deployed hard-verdict threshold is badly miscalibrated on 20b specifically. Recalibrating the
  threshold (using data already on disk, zero new API cost) recovers most of the 46%→92% gap.
- CNFinBench `harmless`'s confidence signal is **near-uninformative on 20b** (0.02 vs. 0.03 —
  essentially no separation between correct and wrong) but **genuinely useful on 120b** (0.04 vs.
  0.42) — meaning 120b has real signal there even where its hard accuracy looks unremarkable.

## Timing (mean seconds per stage)

`infer` (the actual judgment calls) dominates wall time on **both** models regardless of size —
1000-4500+ seconds per dataset. No clean "120b is just slower" story. `cluster`/`infer_emb` stay
fast on both (under 500s). One open oddity: `infer_emb` timing on 120b runs varies widely
(3.6s-119.8s across similar-sized datasets) — possibly inconsistent GPU-node placement, not yet
root-caused.

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

¹ Averaged across the original run + the incomplete-data correction rerun.

## Bottom line

120b is worth the switch for FinVault-style outcome/intent questions (Q1, Q2). CNFinBench
`harmless`-type tasks need more than a model upgrade — the underlying signal is weak-to-moderate
even on the bigger model, and the new false-positive tradeoff on `harmless-unblocked` needs
understanding before treating 120b as a strict improvement everywhere.
