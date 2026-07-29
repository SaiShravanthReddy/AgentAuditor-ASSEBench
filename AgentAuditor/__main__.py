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
    # Guardrail-filtered variants (excludes conversations Anirudhh identified as blocked by the
    # target model's own guardrails, per CNFinBench/filter_blocked_rows.py) - for Ivan's Citibank
    # presentation. 'cnfinbench-harmless-unblocked' is NOT registered yet - blocked pending
    # resolving why the local harmless source data is missing the row_index field the filter
    # requires (see CNFinBench/README.md's guardrail-filtering note).
    'cnfinbench-harmful-unblocked': 'cnfinbench-harmful-unblocked',
    # 'finvault-full' (v2-based) is deliberately NOT registered here anymore - its converted file
    # still had real, technique-revealing case_ids leaking into judge input (see FinVault/README.md
    # and fix_v3_leakage.py's docstring). Renamed to finvault-full-LEAKY-DO-NOT-USE.json so it fails
    # loudly (FileNotFoundError) instead of silently producing a leaky baseline if someone still
    # types the old dataset key. Use finvault-v3-fixed for real runs; if you specifically need the
    # leaky version for a before/after leakage-impact comparison, add a temporary entry pointing at
    # 'finvault-full-LEAKY-DO-NOT-USE' rather than restoring this key.
    'finvault-v3-fixed': 'finvault-v3-fixed',
    # The 4 pairwise comparison variants built by FinVault/build_comparison_variants.py, all derived
    # from the same leak-fixed source. 'benign-v-defended' uses a reworded `goal` prompt (attack-
    # attempt detection, not "did harm occur") since neither of its two classes involves real harm -
    # see the script's docstring for why reusing the stock safety-issue prompt would be invalid here.
    'finvault-v3-fixed-full': 'finvault-v3-fixed-full',
    'finvault-v3-fixed-defended-v-attack': 'finvault-v3-fixed-defended-v-attack',
    'finvault-v3-fixed-benign-v-attack': 'finvault-v3-fixed-benign-v-attack',
    'finvault-v3-fixed-benign-v-defended': 'finvault-v3-fixed-benign-v-defended',
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
            from .tasks.infer_fix1 import fix1_main
            from .tasks.infer_fix2 import fix2_main
            infer_main(dataset)
            fix1_main(dataset)
            fix2_main(dataset)
        case 'eval':
            from .tasks.eval import eval_main
            eval_main(dataset)
        case 'direct_eval':
            from .tasks.direct_metric import direct_metric_main
            direct_metric_main(dataset, dataset_fullname[dataset])
        case 'direct_metric':
            from .tasks.direct_metric import direct_metric_main
            direct_metric_main(dataset)
        case _:
            raise ValueError("Invalid choice..")
