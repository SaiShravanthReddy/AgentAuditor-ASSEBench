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
