#!/bin/bash
# Before running the script, configure LLM API keys and endpoints in .py files under AgentAuditor/tasks, along with model names
# To run an experiement, take rjudge as an example:
# --output must be declared before any non-#SBATCH executable line in the script. The `logs/`
# directory has to exist BEFORE you submit - SLURM opens this file the instant the job starts, not
# when the script gets to running, so `mkdir -p logs` inside the script itself is too late.
#SBATCH --job-name=agent-auditor
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email-address>
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=hpg-turin

# Sequentially run the following commands, remember to check successful completion of each step
python -m AgentAuditor rjudge preprocess
python -m AgentAuditor rjudge cluster
python -m AgentAuditor rjudge demo
python -m AgentAuditor rjudge infer_emb
python -m AgentAuditor rjudge infer
python -m AgentAuditor rjudge eval

# Notes: Only one model and one dataset can be used at a time. If you want to parallelize the process,
# just make a copy of the repo.
