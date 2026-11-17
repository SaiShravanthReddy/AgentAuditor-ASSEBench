# CNFinBench & FinVault Reference

What each benchmark is, how it's labeled, what AgentAuditor measures on it today, and where the
data lives. Compiled from direct inspection of the datasets and real evaluation runs.

## Overview

Both benchmarks put a financial-institution agent through an adversarial multi-turn conversation
and ask whether it stayed safe. They differ in how the conversation is built and how "safe" is
decided.

| | CNFinBench | FinVault |
|---|---|---|
| Attack surface | Conversational escalation, text only | Live tool calls the agent executes |
| Ground truth | LLM judge score (HICS) | Rule-based, execution-grounded |
| Main dataset | `cnfinbench-pooled`, 642 cases | `finvault-v5-fixed-full`, 1,043 cases |
| Class balance | 68% benign / 32% malicious | 8% benign / 92% malicious |

## CNFinBench

### Structure and labels

Each conversation runs up to 12 user turns (some to 24), opening with a broad, legitimate-sounding
question and narrowing, turn by turn, into a request for specific operational or regulatory
detail. The three scenario types share that same escalation strategy and differ only in what
they're trying to extract:

- **MT_App** — data governance: classification, lineage, DPIAs, access workflows
- **MT_Cog** — behavioral finance: investor psychology, cognitive bias, compliance communications
- **MT_Inter** — fraud and security operations: detection tooling, AML escalation, encryption,
  biometric security

Ground truth comes from a HICS score, confirmed by reading the raw judge telemetry:
`Qwen/Qwen3.5-9B` scores each conversation; a threshold of 70 splits safe (>=70, label 0) from
unsafe (<70, label 1).

**Schema note:** turns carry generic role values - `user`, `agent`, `environment` - with no role
that marks a speaker as malicious. The harmful/harmless split lives at the conversation level, in
the id prefix (`harmful-MT_App-1` vs. `harmless-MT_App-1`), not per turn.

### Datasets

| Dataset | Cases | Benign | Malicious | Notes |
|---|---:|---:|---:|---|
| `cnfinbench-pooled` | 642 | 435 | 207 | Main dataset - both classes present |
| `cnfinbench-harmful` | 321 | 150 | 171 | Subset |
| `cnfinbench-harmless` | 321 | 285 | 36 | Subset |
| `*-unblocked` variants | - | - | - | Guardrail-filtered: excludes conversations the target model already blocked itself |

### AgentAuditor on cnfinbench-pooled (gpt-oss-20b, 642 cases, fully recovered)

| Accuracy | Precision | Recall | F1 | Specificity | AUROC |
|---:|---:|---:|---:|---:|---:|
| 72.4% | 67.4% | 28.0% | 39.6% | 93.6% | 0.673 |

Calibration is poor: Brier 0.261, ECE 0.253. The judge reports ~97.7% confidence almost regardless
of whether it's right - confidence is useful for ranking (AUROC) but not trustworthy as a
probability.

| Category | n | Accuracy |
|---|---:|---:|
| MT_Inter | 342 | 81.0% |
| MT_Cog | 100 | 78.0% |
| MT_App | 200 | 55.0% |

### Known failure patterns

- **MT_App misses cluster around cumulative disclosure** - 76-79% of all false negatives are
  MT_App. Each turn reads as reasonable compliance Q&A; the risk is only visible in the narrowing
  trajectory across the full conversation.
- **A stronger model over-corrects the other way** - swapping in gpt-oss-120b catches more real
  MT_App attacks but learns the pattern too broadly, pushing MT_Inter false positives from 3 to 51
  on conversations that are structurally similar but genuinely benign.
- **Mixed-model run collapsed to all-safe** - preprocess `mistral-small-3.1`, demo/demo-repair
  `llama-3.3-70b-instruct`, infer `nemotron-3-super-120b-a12b`, on `cnfinbench-harmless` (321
  cases). It predicted "safe" for all 321 (confidence 0.95-1.0): accuracy 88.8% (the benign share,
  285/321), recall 0%, AUROC 0.48. Raw verdicts were checked, so this is model output, not a
  parsing error. Re-running infer with gpt-oss-120b on the same demos also gave all-safe (recall
  0%, AUROC 0.51), so the infer model isn't the cause; for reference, the single-model gpt-oss-120b
  run reached 22% recall on this subset. Suspects: the mixed preprocess tags or demos. Not
  isolated (the deciding test - gpt-oss-120b infer on the original gpt-oss-120b demos - was not
  run), and only this one dataset and combination were tried.

## FinVault

### Structure and labels

FinVault tests an agent's live tool use rather than its conversation text. An agent in a simulated
financial workflow has access to 203 distinct tools; the benchmark checks whether it calls the
wrong ones, or the right ones at the wrong time.

**Tool vocabulary:** 135 Allow / 63 Confirm / 4 Deny / 1 Flagged.

**Flagged tool:** `certify_green_project` accepts `bypass_penalty_check`, `override_assessment`,
and `override_formula` arguments - a call that sets any of them true looks like an exploit
attempt, not a routine certification.

### Versions

v2 is retired: real label leakage, attack-technique names baked straight into the prompt text.
It's kept only as `finvault-full-LEAKY-DO-NOT-USE` so the old dataset key fails loudly instead of
silently reproducing the leak. v3 and v5 are both leak-fixed, but **they are not the same cases
with corrected labels** - they're independent simulation runs of the same case-template library.
Checked directly: 57% of matching case numbers have a different turn count between v3 and v5,
meaning a genuinely different conversation played out, not a relabel. Every v5 record also carries
a `[Case ID: [REDACTED_CASE_ID]]` marker; only 112 of 1,064 v3 records had any marker at all,
confirming the two versions come from different prompt templates.

### Comparisons

- **Q1 - defended-v-attack**: did the attack succeed? The most consistently validated result
  across versions and models: 84% accuracy, AUROC 0.89 on gpt-oss-120b.
- **Q2 - benign-v-malicious**: was it malicious at all? Pools defended and attack-succeeded cases
  as "malicious" against plain benign traffic. Both v3 and v5 run about 90% malicious.

### AgentAuditor on finvault-v5-fixed-full (gpt-oss-20b, 1,042 of 1,043 processed)

| Accuracy | Precision | Recall | F1 | Specificity | AUROC |
|---:|---:|---:|---:|---:|---:|
| 76.5% | 82.1% | 69.0% | 75.0% | 84.3% | 0.801 |

Calibration: Brier 0.209, ECE 0.174 - weaker than ideal but notably better-behaved than
CNFinBench's. Accuracy varies widely by attack-case category: several categories score 100%, a
handful of sparsely-populated categories score near 0% on precision and recall.

## Vs. a deterministic guardrails baseline

Same case counts, evaluated against a Lakera-inspired, policy-based guardrails detector from the
sibling project.

| Metric | CNFinBench - AgentAuditor | CNFinBench - Guardrails | FinVault - AgentAuditor | FinVault - Guardrails |
|---|---:|---:|---:|---:|
| Accuracy | 72.4% | 66.4% | 76.5% | 67.9% |
| Precision | 67.4% | 45.4% | 82.1% | 63.8% |
| Recall | 28.0% | 21.3% | 69.0% | 85.7% |
| F1 | 39.6% | 28.9% | 75.0% | 73.2% |
| Specificity | 93.6% | 87.8% | 84.3% | 49.2% |

On CNFinBench, AgentAuditor leads on every metric - a deterministic, tool-name-based policy has no
way to see a risk that only appears in the trajectory across many turns. On FinVault, it's a real
tradeoff: the guardrails detector catches more actual attacks (85.7% recall), at the cost of
flagging roughly half of all genuinely safe interactions (49.2% specificity). AgentAuditor is the
more conservative reader on FinVault, missing more attacks but raising far fewer false alarms.

## Pipeline notes

Both benchmarks run through the same six-stage AgentAuditor pipeline: preprocess, cluster, demo
generation and repair, embedding retrieval, inference and repair, eval. Numbers above reflect real
runs, after fixing the following:

- **Silent false success** - the infer stage used to report success even when every API call failed.
- **Destructive repair step** - a JSON-repair stage flagged already-valid records as malformed and
  replaced them with hallucinated filler.
- **Lost answers in reasoning_content** - the real root cause of a major data-loss bug: some model
  responses put the actual verdict in a `reasoning_content` field instead of `content`, which the
  client silently discarded. Fixing this recovered a dataset from 612 of 642 complete records to
  all 642.
- **Confidence extraction gaps** - a few records stored confidence under a capitalized key or as a
  quoted string; both are now read correctly.

New measurement added this cycle: Brier score and ECE for calibration, a per-category accuracy
breakdown, retrieval label-agreement and similarity-score distributions, a standing self-leakage
check, demo self-consistency and specificity checks, and API token/latency telemetry.

## File locations

### Local

| | Path |
|---|---|
| CNFinBench | `AgentAuditor/data/cnfinbench-pooled.json` |
| FinVault | `AgentAuditor/data/finvault-v5-fixed-full.json` |
| Q1 | `AgentAuditor/data/finvault-v5-fixed-defended-v-attack.json` |
| Q2 | `AgentAuditor/data/finvault-v5-fixed-benign-v-malicious.json` |

Pipeline outputs for these runs live in the corresponding run folders on the HPC cluster
(`AgentAuditor/temp/<dataset>/output-k3_corrected.json`), not in this repo.

## Open questions

- **CNFinBench attack-attempt labeling** - the guardrails protocol wants a "malicious role" to
  define an attack attempt. No such role exists in the schema; pending team confirmation on
  whether `user` should stand in for it by convention.
- **CNFinBench attack-success proxy** - proposed as malicious-role conversations with low HICS.
  Exact threshold and direction not yet confirmed.
- **FinVault v3 vs. v5** - confirmed as different simulation batches, not a regression. Still open
  whether v5's trajectories are harder in general or just different.
- **Mixed-model collapse on cnfinbench-harmless** - the infer model is ruled out; the demo
  generator and preprocess stage haven't been isolated yet.
