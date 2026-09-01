import os
import sys

from dotenv import load_dotenv

# Load AGENTAUDITOR_API_KEY / AGENTAUDITOR_API_BASE / AGENTAUDITOR_MODEL_* from the repo-root
# .env before any task module (which reads them at GPTConfig() construction time) is imported.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

dataset_fullname = {
    'rjudge': 'rjudge',
    'agentharm': 'agentharm',
    'asb-sa': 'asb-safety',
    'asb-se': 'asb-security',
    'aj-l': 'AgentJudge-loose',
    'aj-s': 'AgentJudge-strict',
    'aj-sa': 'AgentJudge-safety',
    'aj-se': 'AgentJudge-security',
    'cnfinbench-pooled': 'cnfinbench-pooled',
    'cnfinbench-harmless': 'cnfinbench-harmless',
    'cnfinbench-harmful': 'cnfinbench-harmful',
    # Guardrail-filtered variants (excludes conversations identified as blocked by the
    # target model's own guardrails, per CNFinBench/filter_blocked_rows.py) - for a stakeholder
    # presentation. Harmless's local source data was missing row_index; resolved by verifying `id`
    # forms an exact 1..N sequence per subset (no gaps/duplicates) and deriving row_index from it
    # (--derive-row-index-from-id) rather than waiting on a fresh data pull - 301 kept records,
    # independently cross-checked against a separately-computed filtered count (same 301) before trusting it.
    'cnfinbench-harmful-unblocked': 'cnfinbench-harmful-unblocked',
    'cnfinbench-harmless-unblocked': 'cnfinbench-harmless-unblocked',
    # 'finvault-full' (v2-based) is deliberately NOT registered here anymore - its converted file
    # still had real, technique-revealing case_ids leaking into judge input (see FinVault/README.md
    # and fix_v3_leakage.py's docstring). Renamed to finvault-full-LEAKY-DO-NOT-USE.json so it fails
    # loudly (FileNotFoundError) instead of silently producing a leaky baseline if someone still
    # types the old dataset key. Use finvault-v3-fixed for real runs; if you specifically need the
    # leaky version for a before/after leakage-impact comparison, add a temporary entry pointing at
    # 'finvault-full-LEAKY-DO-NOT-USE' rather than restoring this key.
    'finvault-v3-fixed': 'finvault-v3-fixed',
    # The 5 pairwise comparison variants built by FinVault/build_comparison_variants.py, all derived
    # from the same leak-fixed source. 'benign-v-defended' and 'benign-v-malicious' both use a
    # reworded `goal` prompt (attack-attempt detection, not "did harm occur") since neither
    # comparison's classes involve real harm uniformly - see the script's docstring for why reusing
    # the stock safety-issue prompt would be invalid for those two.
    'finvault-v3-fixed-full': 'finvault-v3-fixed-full',
    'finvault-v3-fixed-defended-v-attack': 'finvault-v3-fixed-defended-v-attack',
    'finvault-v3-fixed-benign-v-attack': 'finvault-v3-fixed-benign-v-attack',
    'finvault-v3-fixed-benign-v-defended': 'finvault-v3-fixed-benign-v-defended',
    # The team's actual "was this malicious at all" framing (confirmed 2026-07-30):
    # full 1064-record dataset, defended+attack_success pooled as "malicious" (1) vs benign (0) - NOT
    # the same as benign-v-defended above, which excludes attack_success entirely.
    'finvault-v3-fixed-benign-v-malicious': 'finvault-v3-fixed-benign-v-malicious',
    # v5 (finvault_output_v5_final, leak-fixed via the same fix_v3_leakage.py mechanism -
    # see FinVault/README.md), priority-scoped to just the 2 questions the team confirmed matter
    # (2026-08-10): "did the attack succeed" and "was it malicious at all". Built by
    # FinVault/build_comparison_variants.py --source-name finvault-v5-fixed.
    'finvault-v5-fixed-full': 'finvault-v5-fixed-full',
    'finvault-v5-fixed-benign-v-malicious': 'finvault-v5-fixed-benign-v-malicious',
}

if __name__ == "__main__":
    # Execute the main script
    if len(sys.argv) >= 3:
        dataset = sys.argv[1]  # First argument
        choice = sys.argv[2]  # Second argument
        
        print(f"Dataset: {dataset}")
        print(f"Task: {choice}")
    else:
        raise ValueError("Please provide at least two command-line arguments")
    
    match choice:
        case 'preprocess':
            from .tasks.preprocess import preprocess_main
            preprocess_main(dataset, dataset_fullname[dataset])
        case 'cluster':
            from .tasks.cluster import cluster_main
            cluster_main(dataset)
        case 'demo':
            from .tasks.demo import demo_main
            from .tasks.demo_repair import demo_repair_main
            demo_main(dataset)
            demo_repair_main(dataset)
        case 'infer_emb':
            from .tasks.infer_emb import infer_emb_main
            infer_emb_main(dataset, dataset_fullname[dataset])
            pass
        case 'infer':
            from .tasks.infer import infer_main
            from .tasks.infer_json_repair import fix1_main
            from .tasks.infer_llm_repair import fix2_main
            infer_main(dataset)
            fix1_main(dataset)
            fix2_main(dataset)
        case 'eval':
            from .tasks.eval import eval_main
            eval_main(dataset)
        case 'direct_eval':
            # Was wrongly calling direct_metric_main (metrics-only, reads an already-existing
            # output file) with 2 args when it only accepts 1 - direct_eval_main (the actual
            # zero-shot LLM-querying stage, in direct_eval.py) was never being called at all, so
            # this crashed with a TypeError before ever reaching the LLM. Fixed: run the querying
            # stage first, then compute metrics on its output, mirroring every other multi-step
            # case above (e.g. 'demo').
            from .tasks.direct_eval import direct_eval_main
            from .tasks.direct_metric import direct_metric_main
            direct_eval_main(dataset, dataset_fullname[dataset])
            direct_metric_main(dataset)
        case 'direct_metric':
            from .tasks.direct_metric import direct_metric_main
            direct_metric_main(dataset)
        case _:
            raise ValueError("Invalid choice..")
