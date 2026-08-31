# Why AgentAuditor Underperforms — Diagnosis & Fixes


## Fix 1: Demos reasoned about the wrong question

`demo.py` (generates few-shot chain-of-thought) and `infer_emb.py` (packages demos for the judge)
both hardcoded the stock "safe/unsafe" framing, ignoring each record's actual `goal` field.
FinVault's Q2 ("was there a manipulation attempt, regardless of outcome") uses a different question
with remapped labels — but its demos were still generated as "explain why this is
safe/unsafe," teaching the wrong signal. Explains Q2's shape: ~100% precision (attack_success
cases still look like normal "safety violations") but ~40% recall (defended cases got no relevant
guidance).

Also required fixing `preprocess.py`, which silently dropped `goal` from its output fields —
`demo.py` never had a goal to read even before this fix.

**Fix:** both functions now use each record's `goal` when present, falling back to the old
behavior otherwise (byte-identical output verified for unaffected datasets).
**Commit:** `7adab5d`

## Fix 2: ~7-38% of demos were silently blank

`demo_repair.py` logs but doesn't remove records whose LLM-based JSON repair failed — they stayed
in the retrieval pool with broken chain-of-thought, silently becoming blank few-shot demos and
wasting a retrieval slot.

**Fix:** `infer_emb.py` now excludes broken records from retrieval candidacy before similarity
search runs, using `demo_repair.py`'s own validity check.
**Commit:** `bbec8ee`

## Validation (FinVault v3, Q2)

Re-ran `cluster→demo→infer_emb→infer→eval` with both fixes. Independently re-verified from raw
output (exact match):

| | Before | After | Δ |
|---|---|---|---|
| Accuracy | 49.1% | 56.7% | +7.6 |
| Precision / Recall | 100% / 43.4% | 99.6% / 52.1% | −0.4 / **+8.7** |
| AUROC / AUPRC | 0.741 / 0.949 | 0.764 / 0.953 | +0.023 / +0.004 |

128 more malicious cases caught (FN 587→458, TP 370→498). Both fixes confirmed as real
contributors, not just hypotheses.

*Caveat: Fix 2 excluded 38% of demos this run vs. 19% pre-fix — Fix 1's new prompt likely has its
own JSON-parse-failure rate. `demo_repair.py`'s repair quality is now its own improvement target.*

## Fix 3: A cluster representative could retrieve itself as its own demo

Reference (demo pool) and query files overlap — representatives are drawn from the same full
dataset every query comes from — so a query whose own dialogue was chosen as a representative
would very likely retrieve itself (near-perfect content similarity). Confirmed happening:
`harmless-MT_App-16` retrieved itself, and that self-match was also broken by Fix 2's bug — wasting
1 of 3 demo slots on a blank self-referential example. `find_most_similar_two_stage()` now takes
`exclude_id`, applied before similarity ranking. Not yet re-tested with a pipeline re-run.
**Commit:** `6373dcf`

## CNFinBench harmless (AUROC 0.53, unresolved)

Not explained by either fix above (its `goal` field is uniform, unaffected by Fix 1). Reading real
false negatives found: confirmed self-leakage (a query retrieved itself as a demo — that demo was
broken by Fix 2's exact bug, now excluded going forward); confidence is near-uninformative here
(false negatives and true negatives show the same 0.99/0.95 confidence distribution); 86% of false
negatives cluster in one scenario subtype (`MT_App`), suggesting a capacity or task-framing gap
rather than a simple bug.

## Next steps

1. Re-test Fix 1+2 on v5's `benign-v-malicious` to confirm generalization
2. Improve `demo_repair.py`'s repair success rate
3. Try a different judge model on CNFinBench harmless (`gpt-oss-120b` first — easy swap, cheap to
   test: only needs `infer_emb→infer→eval` re-run)
4. Confirm with the team whether "cumulative disclosure escalation" is the intended risk signal for
   `MT_App` — determines if this needs a prompt fix or is genuine task ambiguity
5. Still open: which judge model produced CNFinBench's original labels (tracked in
   `RESULTS_SUMMARY.md`)
