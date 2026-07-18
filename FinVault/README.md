# FinVault

Scaffold for running AgentAuditor against [FinVault](https://github.com/aifinlab/FinVault).

Mirrors `CNFinBench/`'s structure:

- `data/` — raw FinVault source data (gitignored, not yet populated)
- `finvault_to_agentauditor.py` — converter from FinVault's schema to AgentAuditor's ASSEBench
  input schema (`id`/`profile`/`contents`/`label`), analogous to
  `CNFinBench/cnfinbench_to_agentauditor.py`
- `finvault_metrics.py` — metrics computation, analogous to `CNFinBench/cnfinbench_metrics.py`
- `finvault_detect_leakage.py` — self-leakage detection, analogous to
  `CNFinBench/cnfinbench_detect_leakage.py` (reusable close to as-is, since the leakage mechanism
  is in AgentAuditor's shared `infer_emb.py`, not benchmark-specific)
- `RESULTS.md` — results write-up, once a run completes

Uses the same shared `AgentAuditor/` pipeline at the repo root (dataset keys prefixed
`finvault-*`, alongside CNFinBench's `cnfinbench-*` keys, per `AgentAuditor/__main__.py`'s
`dataset_fullname` map) — not a separate clone.

**Note on where converted dataset files live:** `finvault_to_agentauditor.py`'s output (the
AgentAuditor-schema JSON) goes to `AgentAuditor/data/finvault-*.json`, *not*
`FinVault/data/`. This isn't an oversight — `AgentAuditor/tasks/*.py` hardcodes its data/temp
paths relative to the package itself (`../data`, `../temp`, resolved from each script's own file
location) in ~22 places across 10 files, with no configurable override. Making that configurable
or symlinking around it was considered and explicitly declined (see project history) in favor of
just keeping `AgentAuditor/data/` and `AgentAuditor/temp/` as the shared location for every
dataset's converted/intermediate files, relying on the `cnfinbench-`/`finvault-` filename prefix
for separation rather than physical folder location. `FinVault/data/` holds only the raw,
unconverted source data.
