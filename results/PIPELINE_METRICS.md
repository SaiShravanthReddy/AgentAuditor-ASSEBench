# AgentAuditor Pipeline Metrics — Performance, Inference, and Standalone-Tool Readiness

Real, measured performance for each of AgentAuditor's 6 pipeline stages, what each number implies,
and — the actual question this document exists to answer — which stages are good enough,
independently, to extract as standalone tools. Written with a second baseline (TRACES, in
development by another teammate) in mind as a reason this matters: a validated standalone
component is reusable regardless of which baseline ends up winning the comparison it's built for.
No claims about TRACES' own design are made here — this document only covers what's actually been
measured in AgentAuditor.

**Known open issue affecting this data**: after the new `infer_retry` stage runs and recovers
missing records, `AUROC`/`AUPRC` become uncomputable (0 usable confidence values) for whichever
dataset it touched — confirmed on 4/5 CNFinBench `20b` datasets, still under investigation. Ranking
metrics quoted below for those 4 use the last known-good measurement (pre-`infer_retry`), explicitly
marked. Hard accuracy/precision/recall/F1 numbers are unaffected — they don't depend on confidence.

## 1. `preprocess` — semantic annotation

**Performance:**
- Duration: 0.2s (cached) to ~4700s (cold, largest dataset) — see `PROFILING_RESULTS.md`'s timing table
- Reliability: correctly and loudly fails via `LLM_GENERATION_ERROR` sentinels rather than silently succeeding — confirmed at 642/642 (100%) sentinel rate during the dead-`120b`-key incident, 0% once the key was fixed

**Inference:** Does its one job (LLM-annotate + resume-on-crash) reliably. The failure-sentinel design is a genuine strength — it's how the dead-key bug was caught at all rather than silently corrupting downstream stages. No unique differentiated capability, though — it's a generic "call an LLM, annotate a record" pattern with an AgentAuditor-specific output schema (`scenario`/`risk_type`/`failure_mode`).

**Standalone readiness: Low.** Reliable, but not differentiated — nothing here outperforms a generic LLM-classification call. The main reusable idea (fail loudly via sentinel values, not silently) is a design pattern worth carrying forward, not a component worth extracting as-is.

## 2. `cluster` — representative selection

**Performance:**
- Duration: 119-479s, GPU-accelerated (`nomic-embed-text-v1.5` + FINCH)
- Representative count: consistently ~10-15% of input size across datasets (target `len(data)/10`)
- **No quality metric exists yet** — no silhouette score, no coverage check has been run. This is a real gap, not a zero score.

**Inference:** Structurally sound and fast, but we cannot currently make a *quality* claim — only a *speed and completion* claim. It's plausible this works well (nothing downstream has pointed at bad clustering as a root cause of any diagnosed failure), but that's an absence of evidence, not evidence of quality.

**Standalone readiness: Technically easiest to extract, quality unproven.** Of all 6 stages, this has the least coupling to AgentAuditor's judging logic — no API key, no prompt construction, no judge-specific schema, just embeddings + clustering. That makes it the *cheapest* to pull out. But extracting an unvalidated component risks shipping something whose actual quality is unknown — recommend building the silhouette/coverage metric first, even a quick one, before calling this ready.

## 3. `demo` (generation) vs. `demo_repair` (validation/repair) — split verdict

**Performance — `demo_repair`'s repair logic specifically:**
- Pre-fix: 12/63 (19%) of a real demo pool silently broken (blank few-shot slots)
- Post-Fix-1 (before the brace-repair fix): 24/63 (38%) broken — a regression caused by Fix 1's new prompt having its own JSON-failure mode
- **Post-brace-fix: 63/63 (100%) valid** — verified against all 24 real historical failures *and* a fresh live re-run

**Performance — `demo` (CoT generation) itself:** No quality metric exists. We validate that generated demos are *well-formed JSON* (that's what `demo_repair` checks), never whether the reasoning inside actually reaches the *correct* conclusion for that record's known label. This is the single biggest gap in this whole document — flagged in the previous conversation as the highest-priority metric to build, still not built.

**Inference:** `demo_repair`'s repair logic is the most rigorously validated component in the entire pipeline — a real, measured 100% recovery rate, not an estimate. `demo.py`'s generation quality is completely unmeasured; we don't know if the demos it produces are teaching good or bad reasoning.

**Standalone readiness: `demo_repair`'s repair logic — High, strongest candidate in the pipeline.** Fully generic (fixes any LLM JSON output missing closing braces, nothing safety-judging-specific), pure function on a string, and has the best hard evidence of any component here. **`demo.py`'s generation — cannot assess**, no quality data exists.

## 4. `infer_emb` — few-shot retrieval

**Performance:**
- Duration: 3.6-480s
- Self-leakage: a real, confirmed bug existed (a cluster representative retrieving itself as a "prior example," breaking 1 of 3 demo slots) — fixed via `exclude_id`, but **not independently re-measured on current data**; we know it was fixed, not its current rate
- End-to-end impact of the retrieval fixes (Fix 2 exclusion + Fix 3 self-exclusion, combined with Fix 1): +8.7pt recall, independently verified — but this is an *attributed* effect from a combined test, not a retrieval-quality score in isolation

**Inference:** We have strong evidence the retrieval *fixes* mattered, but no standing metric for retrieval quality itself (e.g. "% of retrieved demos sharing the query's true label"). The self-leakage detector (`cnfinbench_detect_leakage.py`) exists and is dataset-agnostic, but hasn't been rerun against the current batch to get a current number.

**Standalone readiness: Medium.** The safe-retrieval *logic* (self-exclusion + broken-demo filtering) is a clean, generically-useful pattern for any RAG-style few-shot system, with real attributed value — but it's more coupled to the surrounding embedding/demo-pool infrastructure than `demo_repair`'s pure-function repair logic, and its own isolated quality still needs the leakage-rate metric run fresh.

## 5. `infer` (+ `infer_json_repair`/`infer_llm_repair`) — the actual judgment

**Performance:**
- Complete API-failure rate: `gpt-oss-20b` under the original 14-job concurrent batch, 2.2%-9.8% (16/164 to 63/642); still 2.2%-10.9% remaining even in relative isolation after `infer_retry`. `gpt-oss-120b`: 0% for all 5 CNFinBench datasets, 0.3-0.5% parse-error rate (different failure category) on FinVault.
- Confidence calibration gap (mean positive-class score, correct vs. wrong predictions):
  - FinVault Q2 (`benign-v-malicious`): **0.78 vs. 0.05** (`20b`), **0.88 vs. 0.09** (`120b`) — strong, real signal
  - FinVault Q1 (`defended-v-attack`): 0.51 vs. 0.34 (`20b`), 0.53 vs. 0.39 (`120b`) — moderate
  - CNFinBench `harmless`: **0.02 vs. 0.03** (`20b`, near-zero — essentially no signal), 0.04 vs. 0.42 (`120b`, real signal)
  - CNFinBench `harmless-unblocked`: 0.02 vs. 0.11 (`20b`, weak), 0.05 vs. 0.64 (`120b`, strong)
- `infer_llm_repair`'s correction stage: was **completely non-functional** until the hardcoded-ID bug fix (2026-09) — every dataset ever run before that fix relied on mechanical repair only

**Inference:** This is the most inconsistent stage in the pipeline, and inconsistent in an interesting way — its judgment quality is *task-dependent*, not uniformly good or bad. It's genuinely strong on outcome-based questions (Q1, Q2) and genuinely weak on intent-based ones (CNFinBench `harmless`, especially on `20b`), and its reliability (API completion rate) is sensitive to shared-resource contention in a way none of the other stages are.

**Standalone readiness: Low — this is the piece under active repair, not ready for extraction.** It's also the most safety-judging-specific stage (confidence-extraction, prompt construction), so even a well-performing version would need real decoupling work. Given its quality is still task-dependent and partially unexplained (the CNFinBench `harmless` gap), extracting it now would mean shipping a known-inconsistent component.

## 6. `eval` — scoring

**Performance:** Accuracy/precision/recall/F1/AUROC/AUPRC/confusion matrix, computed correctly across every dataset in this project, including correct handling of real-world LLM-judge failure sentinels (excludes them rather than corrupting metrics).

**Inference:** This is the one stage with an existing, real proof of standalone reusability: `CLAUDE.md` documents that FinVault reuses CNFinBench's own `cnfinbench_metrics.py` *directly, unforked* — it was already generic enough to not need dataset-specific changes.

**Standalone readiness: High — already proven, not just plausible.** Pure JSON-in/metrics-out, zero coupling to the judge model, retrieval logic, or even which benchmark produced the data.

## Standalone-tool ranking, by actual evidence

| Rank | Component | Evidence |
|---|---|---|
| 1 | `eval`'s scoring logic | Already reused across benchmarks unforked — proven, not projected |
| 2 | `demo_repair`'s JSON repair | 100% measured recovery rate (24/24 historical + 63/63 live), fully generic |
| 3 | `cluster`'s embedding+FINCH selection | Least coupled technically, but no quality metric exists yet — build one before extracting |
| 4 | `infer_emb`'s safe-retrieval logic | Real attributed value (+8.7pt recall) but more infrastructure-coupled; needs a fresh isolated leakage-rate measurement |
| 5 | `preprocess`'s annotation call | Reliable but not differentiated from a generic LLM-classification call |
| 6 | `infer`'s judgment logic | Most valuable capability in principle, least ready in practice — task-dependent quality, under active investigation, most domain-coupled |

**Bottom line**: the two strongest, most defensible standalone candidates right now are `eval`'s
scoring (already proven) and `demo_repair`'s JSON repair logic (best-measured performance in the
pipeline). The component most people would assume is "the AgentAuditor tool" — `infer`, the actual
judgment step — is the *least* ready, precisely because it's still being actively diagnosed and
fixed.
