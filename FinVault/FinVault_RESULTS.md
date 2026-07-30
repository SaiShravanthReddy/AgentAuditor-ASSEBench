# Results: AgentAuditor (gpt-oss-20b) on FinVault v3-fixed

## Note for whoever generates slides/summaries from this file

**`benign-v-defended`'s 34.9% accuracy is not a weaker version of the other three comparisons — it
answers a genuinely different question and must be presented that way.** The other three
comparisons ("full," "defended-v-attack," "benign-v-attack") all ask *"did harm occur?"* and score
76-81%, consistently. `benign-v-defended` instead asks *"was there a manipulation attempt at all,
regardless of outcome"* — neither of its two classes involves real harm. Do not put all 4 numbers
in one undifferentiated row/list implying they're the same kind of measurement. If summarizing in
one line: *"AgentAuditor reliably detects when harm actually occurred (76-81% across three
framings), but cannot reliably tell an attack attempt from an ordinary interaction when no harm
resulted (35%, close to random)."* See Key Takeaway #2 below for the full reasoning (surface-level
manipulation language appears at nearly identical rates in benign vs. defended prompts, so there's
no easy cue to exploit — this is a hard task, not a broken one).

**Second headline finding, now that the `direct_eval` zero-shot baseline has been run: memory/
retrieval provides a large, consistent benefit specifically on the "did harm occur" question, and
that same mechanism is the one place it actively hurts.** On `full`, `defended-v-attack`, and
`benign-v-attack`, AgentAuditor's memory-augmented judge beats a zero-shot baseline by
**+22 to +25 points of accuracy** (e.g. 78.3% vs. 55.0% on `full`) — a large, consistent effect
across all three framings of the harm question, not a one-off. On `benign-v-defended`, the pattern
flips: the zero-shot baseline *outperforms* AgentAuditor by 7 points (41.9% vs. 34.9%). Both halves
of this story belong together on a slide — "memory helps a lot for detecting real harm, but doesn't
transfer to the attack-attempt-detection question" is the accurate one-line summary, not just the
positive half. See "Results: baseline comparison" below for the full table.

## Superseded — earlier numbers in this file were from a broken run

An earlier version of this file reported a single pooled result (n=1058, accuracy 0.674). Those
numbers came from a run that was later discovered to be **accidentally zero-shot** — a cache-key
collision bug in `infer_emb.py` (fixed; see validity note #1) silently caused every record to
receive 0 few-shot demonstrations instead of AgentAuditor's intended memory-augmented retrieval, so
that run measured a plain zero-shot judge while being labeled as AgentAuditor's actual approach.
**Retracted, not reported below.** The results below are from 4 separate HiPerGator runs (one per
pairwise comparison) with that bug fixed and independently verified (0.0% zero-demo records across
all 4 — see validity note #1).

## Setup

- **Model**: `gpt-oss-20b`, served via UF's AI gateway (OpenAI-compatible endpoint), used for every
  LLM-calling pipeline stage (preprocess, demo, demo_repair, infer, infer_fix2).
- **Data**: `finvault_output_full1064_v3_fixed` — see `FinVault/README.md`'s "Data versions" and
  "Leakage investigation" sections for how this was derived from Xiaoyu's v3 export and what was
  fixed.
- **Ground truth**: `outcome == "attack_success"` → unsafe (label 1); `"defended"`/`"benign"` →
  safe (label 0), except the `benign-v-defended` comparison, which uses a different framing
  entirely (see below).
- **4 comparisons**, built by `FinVault/build_comparison_variants.py` from the same leak-fixed
  source:
  - **full** (1064 records): pooled `benign`+`defended` (0) vs. `attack_success` (1) — "did harm occur"
  - **defended-v-attack** (957 records): `defended` (0) vs. `attack_success` (1) — "did harm occur," isolating attack-attempted cases
  - **benign-v-attack** (643 records): `benign` (0) vs. `attack_success` (1) — "did harm occur," cleanest separation
  - **benign-v-defended** (528 records): `benign` (0) vs. `defended` (1) — a **different question**, "was there a manipulation attempt at all, regardless of outcome" (neither class involves real harm, so this uses a reworded `goal` prompt, not the stock safety-issue one — see the script's docstring for why reusing the stock prompt would be invalid here)
- **Infrastructure**: all 4 ran on HiPerGator as separate SLURM jobs / repo copies (per this repo's
  documented parallelization pattern), with 64GB memory allocated to `infer_emb` (16GB was
  insufficient and caused 3/4 initial attempts to be OOM-killed — see validity note #1).

## Validity notes (read before trusting the numbers below)

### 1. Two real bugs found and fixed during this work, both verified against real data before trusting results

**Bug A — silent zero-shot degeneration.** `infer_emb.py`'s embedding cache was keyed only on the
bare input filename (e.g. `demo_fixed.json`), identical across every dataset sharing this repo's
`embedding_cache_structured/` directory. A stale cache from the very first dataset ever run got
silently reused by every later dataset, making few-shot retrieval return ids that didn't exist in
the *current* dataset's reference data — collapsing to 0 demos for 100% of records, with no
exception raised anywhere. This affected the original (now-retracted) FinVault run above, and was
also found to have already affected a previously-published CNFinBench condition. **Fixed**
(dataset-qualified cache keys) and a **hard guard added**: `infer_emb.py` now aborts if ≥50% of
records would get zero demos, rather than silently writing a degenerate dataset. Verified on the
current runs: 0.0% zero-demo records across all 4 comparisons.

**Bug B — output-parsing gap, in two rounds.** `eval.py`'s `extract_output()` didn't recognize the
many different key names `gpt-oss-20b` actually used for its verdict — first found on the original
(retracted) run (50+ missing top-level key names → 94% "processing errors" → fixed → 99.44%
success). Once bug A was fixed and real few-shot demos started working, a **second, related gap**
surfaced: `gpt-oss-20b` far more often nested its verdict inside `chain_of_thought` (matching the
demos' own CoT-nested style) than it had under the accidentally-zero-shot run, but
`extract_output_from_chain_of_thought()` had its own separate, never-updated list of only 7 keys —
causing success rates to drop back to 51-70% on the first post-fix runs. Root-caused by inspecting
the actual failing records' key structure directly (not guessing): verdict keys like
`Final Judgment`, `final_decision`, `Conclusion` were present but unrecognized. **Fixed** by
unifying the top-level and CoT-nested key lists into one shared, case/spacing-insensitive lookup,
tested against synthetic cases matching the exact observed failure patterns plus a regression check
against previously-working formats (both clean) before pushing. Verified against real data:
success rate recovered to 96.6-97.9% across all 4 comparisons (individual counts in the Results
table below).

### 2. Few-shot self-leakage — checked, clean

`cnfinbench_detect_leakage.py` (checks whether a cluster representative retrieves its own dialogue
as a few-shot demo during `infer_emb`, inflating its own score) is dataset-agnostic and applies
here too. CNFinBench's own numbers (see its RESULTS.md) showed this leakage was severe in one
condition (94% of representatives) and entirely absent in another, so this was not something to
assume either way for FinVault. Run against all 4 comparisons' `k3.json`: **0 genuinely self-leaked
items found in every single one** (full, defended-v-attack, benign-v-attack, benign-v-defended).
Not a concern for these results.

### 3. `full` comparison: 1 record (1063, not the full 1064) dropped by a request timeout — understood, not a mystery

`full`'s input dataset has 1064 records, but `eval.py` reports processing 1063 — traced end-to-end
through every pipeline stage (`memory.json` → `k3.json` → `output-k3.json` →
`output-k3_corrected.json`) rather than assumed: the record (`...case_fixed_0525`) is present
through `infer_emb`, then vanishes starting at `infer`'s output. `infer.py` sends failed-after-all-
retries items to a separate `failed.json` rather than including them in the main output — confirmed
via that file this was the only such case for `full`.

Root cause confirmed from the SLURM log (not inferred): all 3 retry attempts failed identically with
`Read timed out (read timeout=30)` — the same failure every time, not a transient network blip (a
real blip wouldn't fail 3/3 identically). This record's prompt happened to retrieve 3 few-shot
demonstrations in this particular run (each a full conversation + detailed chain-of-thought
example), making it one of the longest prompts in the run — plausibly too long for `gpt-oss-20b` to
finish generating within `infer.py`'s 30-second per-attempt timeout. Consistent with this: the same
underlying record also appears in the `defended-v-attack` and `benign-v-attack` comparisons and
succeeded in both — each comparison runs its own separate clustering/demo-retrieval pipeline, so the
same query record draws a different (and in those runs, shorter) set of demos. Isolated to this one
record in this one comparison (0.09% of `full`); not indicative of a systemic issue.

### 4. `direct_eval` baseline — run, and it needed 3 of its own bug fixes first

`direct_eval` (zero-shot, no memory/demos) had never actually been run before — attempting it
surfaced that it was completely non-functional, for three independent reasons, each confirmed
against a real crash/traceback before fixing (not guessed):

- `ratelimiter`'s bare import crashes on Python ≥3.11 (calls the removed `asyncio.coroutine` at
  class-definition time) — this repo already requires ≥3.11 for `match`/`case` in `__main__.py`, so
  `direct_eval` could never have run on a correctly-set-up environment at all. Shimmed (`RateLimiter`
  is only ever used synchronously here, so the broken async path is never actually exercised).
- `__main__.py`'s `direct_eval` case called `direct_metric_main` (metrics-only, reads an
  already-existing output file) with 2 positional arguments, but that function only accepts 1 — a
  `TypeError` on every invocation, meaning the actual LLM-querying stage (`direct_eval_main`, in
  `direct_eval.py`) was never reached even after fixing the import. Fixed to call
  `direct_eval_main` first, then `direct_metric_main` on its output.
- `completion.choices[0].message.content` can legitimately be `None` from the API itself (e.g. an
  empty/filtered response) without raising an exception — so it wasn't caught by the `backoff`
  retry decorator, and crashed `evaluate_single_file` on `pred.lower()` partway through a live run
  (confirmed via the actual traceback, not reasoned about in the abstract). Treated the same as any
  other unparseable response instead of crashing the whole run over one record.

All three fixed, verified against real data (not just re-reading the diff), committed, and pushed
before trusting the numbers below. Results in "Results: baseline comparison" below.

### 5. Other limitations not corrected for (disclosed, not fixed)

- **Single run, temperature 0, no repeats**: no variance/confidence interval on any number below.
- **FinVault's own label has an inherent noise floor**: cross-checking v1 vs. v2 (independent
  rollouts of the same cases) showed ~2.5% of attack-case labels flip between runs due to genuine
  agent behavioral nondeterminism, not scoring noise (see `FinVault/README.md`). No detector's
  accuracy against this ground truth can be expected to exceed that floor.
- **`benign-v-defended`'s remaining ~2-3% parsing failures** (and similar small residuals on the
  other 3) were not individually inspected this round the way the original 6 stragglers were —
  assumed genuinely malformed based on the pattern established earlier, not re-verified per-record.

## Results

| Comparison | n (successfully processed) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| full (pooled, "did harm occur") | 1041 / 1063 (97.9%) | 0.7829 | 0.7944 | 0.7639 | 0.7789 |
| defended-v-attack ("did harm occur," attack-attempted only) | 933 / 957 (97.5%) | 0.8146 | 0.8927 | 0.7563 | 0.8188 |
| benign-v-attack ("did harm occur," cleanest separation) | 624 / 643 (97.1%) | 0.7628 | 0.8970 | 0.8069 | 0.8496 |
| benign-v-defended ("was there a manipulation attempt") | 510 / 528 (96.6%) | 0.3490 | 0.7305 | 0.2983 | 0.4236 |

**Confusion matrices:**

**full**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 417 (TN) | 103 (FP) |
| **Actual 1** | 123 (FN) | 398 (TP) |

**defended-v-attack**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 369 (TN) | 47 (FP) |
| **Actual 1** | 126 (FN) | 391 (TP) |

**benign-v-attack**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 58 (TN) | 48 (FP) |
| **Actual 1** | 100 (FN) | 418 (TP) |

**benign-v-defended**

| | Predicted 0 (benign) | Predicted 1 (defended) |
|---|---|---|
| **Actual 0 (benign)** | 56 (TN) | 45 (FP) |
| **Actual 1 (defended)** | 287 (FN) | 122 (TP) |

## Results: baseline comparison (AgentAuditor vs. `direct_eval` zero-shot)

Same `gpt-oss-20b` model, same 4 datasets, only difference is whether the judge gets AgentAuditor's
retrieved few-shot demonstrations or nothing. `direct_eval`'s own metrics script (`direct_metric.py`)
computes accuracy slightly differently from AgentAuditor's `eval.py` — it includes unparseable
predictions in the accuracy denominator as automatically-wrong (`eval.py` excludes them entirely
from both numerator and denominator). `direct_eval`'s parse-failure rate is small (2.5-3.1%,
"Validity" column below), so this doesn't change any conclusion, but the two accuracy numbers aren't
computed by identical methodology — flagging rather than silently presenting them as equivalent.

| Comparison | AgentAuditor Accuracy | direct_eval Accuracy | Δ | AgentAuditor F1 | direct_eval F1 |
|---|---|---|---|---|---|
| full | 0.7829 | 0.5498 | **+23.3 pts** | 0.7789 | 0.5637 |
| defended-v-attack | 0.8146 | 0.5611 | **+25.3 pts** | 0.8188 | 0.6004 |
| benign-v-attack | 0.7628 | 0.5428 | **+22.0 pts** | 0.8496 | 0.6813 |
| benign-v-defended | 0.3490 | 0.4186 | **−7.0 pts** | 0.4236 | 0.5318 |

`direct_eval`'s validity (parse success) rate was 97.5%, 96.9%, 97.2%, 97.5% respectively — all
comparable to AgentAuditor's own 96.6-97.9%, so the gap isn't a parsing artifact on either side.

## Key takeaways

1. **The three "did harm occur" comparisons (full, defended-v-attack, benign-v-attack) all show
   real, meaningful discriminative signal** — accuracy 76-81%, precision consistently strong
   (79-90%), recall moderate (76-81%). Roughly consistent across all three framings of the harm
   question, which is reassuring: the judge isn't just exploiting one particular class composition.
2. **`benign-v-defended` is genuinely hard, as predicted before running it.** Accuracy (34.9%) is
   *far below* what a trivial "always predict defended" baseline would get — 409/510 (80.2%) of the
   successfully-processed items are actually `defended`, so a majority-class baseline beats this
   judge by over 45 points. Recall is only 29.8% —
   the judge overwhelmingly under-flags genuine attack-attempts as "no manipulation occurred." This
   matches the surface-marker analysis done before running it: manipulation-style language
   (`[System Notification]`, `CASE-REF`, etc.) appears at nearly identical rates in benign (68.2%)
   and defended (65.6%) prompts, so there's no easy surface cue to exploit, and the judge evidently
   isn't picking up on subtler signals either. **This result should be presented as "the judge
   cannot reliably distinguish an attack attempt from an ordinary interaction when no harm results,"
   not glossed over as a weak version of the other three comparisons** — it's answering a genuinely
   different, harder question.
3. **Two pipeline bugs (validity note #1) were found and fixed specifically because these results
   were checked against raw data rather than trusted at face value** — both silently produced
   plausible-looking but wrong numbers (a real zero-shot run masquerading as memory-augmented; a
   ~35-48% data-loss rate masquerading as normal parsing noise). Worth stating explicitly for
   anything downstream (e.g. Ivan's presentation) that cites these numbers: they reflect a
   corrected, verified pipeline, not the first run's output.
4. **Memory/retrieval provides a large, consistent benefit specifically on the "did harm occur"
   question — and is the one place it actively hurts on the different question.** AgentAuditor beats
   the zero-shot `direct_eval` baseline by +22 to +25 points of accuracy on `full`,
   `defended-v-attack`, and `benign-v-attack` alike (not a one-off — consistent across all three
   framings). On `benign-v-defended`, that flips: zero-shot *outperforms* AgentAuditor by 7 points.
   Read together with takeaway #2: the few-shot demonstrations AgentAuditor retrieves are apparently
   well-suited to reasoning about *whether harm resulted*, but that same mechanism doesn't transfer
   to (and may actively interfere with) the categorically different *"was this an attack attempt at
   all"* question — plausibly because the retrieved examples are themselves built around
   harm-outcome reasoning, priming the judge toward the wrong kind of analysis for this comparison.
   This answers the open question from earlier in this project ("how much does AgentAuditor's memory
   actually help versus just asking the model directly?") with real data rather than leaving it
   unaddressed.
5. **Getting to this baseline required fixing 3 more bugs in code that had never actually been run
   before** (validity note #4) — `direct_eval` was completely non-functional prior to this session,
   for three independent reasons, each confirmed via a real crash before fixing. Self-leakage
   (previously an open item) has also since been checked and is clean across all 4 comparisons
   (validity note #2).

## Reproducing this

```bash
python3 FinVault/fix_v3_leakage.py FinVault/data/finvault_output_full1064_v3_anonymized FinVault/data/finvault_output_full1064_v3_fixed
python3 FinVault/finvault_to_agentauditor.py FinVault/data/finvault_output_full1064_v3_fixed/trajectories.jsonl AgentAuditor/data/finvault-v3-fixed.json --run-name finvault-v3-fixed
python3 FinVault/build_comparison_variants.py   # produces the 4 comparison datasets in AgentAuditor/data/

# per comparison (dataset keys: finvault-v3-fixed-full, -defended-v-attack, -benign-v-attack, -benign-v-defended):
python3.11 -m AgentAuditor <dataset> preprocess
python3.11 -m AgentAuditor <dataset> cluster
python3.11 -m AgentAuditor <dataset> demo
python3.11 -m AgentAuditor <dataset> infer_emb   # needs ~64GB memory if running as a SLURM job; 16GB was insufficient
python3.11 -m AgentAuditor <dataset> infer
python3.11 -m AgentAuditor <dataset> eval

# baseline comparison (no clustering/demo/embedding needed - much lighter weight):
python3.11 -m AgentAuditor <dataset> direct_eval
```

Requires Python ≥3.10 (repo uses `match`/`case` in `AgentAuditor/__main__.py`) and
`AGENTAUDITOR_API_KEY`/`AGENTAUDITOR_API_BASE`/`AGENTAUDITOR_MODEL_*` set via `.env`. On a machine
where the default `python3`/`python` resolves to <3.10, use an explicit `python3.11` (or newer)
interpreter for every step above.
