#!/bin/bash
# Runs the AgentAuditor pipeline for a dataset (default: finvault-full), logging a timestamped
# milestone per stage plus a running time estimate, since this is meant to run unattended as a
# SLURM batch job with no one watching it live.
#
# Baseline per-record/per-stage rates below are measured from the actual CNFinBench run (see
# CNFinBench/RESULTS.md) on gpt-oss-20b, not guessed: preprocess ~1.4s/record, cluster ~4.1s/record,
# demo ~23s/representative (~10% of records), infer_emb ~5.2s/record, infer ~3.5s/record (on the
# ~90% of records that aren't cluster representatives). These are a starting estimate only - FinVault's
# per-record content is meaningfully larger (median ~4.5KB vs CNFinBench's dialogues) so actual
# durations may run longer, especially for cluster/infer_emb (embedding-bound) and infer
# (prompt-length-bound). The script re-estimates remaining time using each stage's *actual*
# observed duration once it completes, rather than trusting the static baseline throughout.
#
# Usage: run_finvault_pipeline.sh [dataset_key] [total_record_count]
#   e.g. run_finvault_pipeline.sh finvault-full 1064
#
# Optional phone push notifications via ntfy.sh (free, no signup): set NTFY_TOPIC to a private,
# hard-to-guess string (ntfy topics are public to anyone who knows the name - don't use something
# guessable like "finvault"), install the ntfy app (iOS/Android) and subscribe to that same topic
# name, then export NTFY_TOPIC=<your-topic> before running this script. Leave it unset to disable -
# everything else works identically either way. Only fires at real milestones (pipeline start, each
# stage's completion, pipeline done/failed) - not the frequent "elapsed so far" lines, to avoid
# spamming your phone. Verify HiPerGator's compute nodes actually have outbound HTTPS access before
# relying on this for a long unattended run (same caveat as the LLM API egress check) - a failed
# notification never affects the pipeline's own success/failure, but a silently-undelivered
# notification is still worth knowing about ahead of time, not discovering mid-run.

set -uo pipefail

DATASET="${1:-finvault-full}"
TOTAL_RECORDS="${2:-1064}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
REP_FRACTION="0.10"   # FINCH target_n_clusters = len(data)/10, per AgentAuditor/tasks/cluster.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${SCRIPT_DIR}/${DATASET}_milestones.log"

REPS=$(python3 -c "print(int(${TOTAL_RECORDS} * ${REP_FRACTION}))")
NON_REPS=$(( TOTAL_RECORDS - REPS ))

# Baseline seconds-per-unit, from CNFinBench actuals.
RATE_PREPROCESS=1.4
RATE_CLUSTER=4.1
RATE_DEMO=23
RATE_INFER_EMB=5.2
RATE_INFER=3.5

est_preprocess=$(python3 -c "print(round(${TOTAL_RECORDS} * ${RATE_PREPROCESS} / 60, 1))")
est_cluster=$(python3 -c "print(round(${TOTAL_RECORDS} * ${RATE_CLUSTER} / 60, 1))")
est_demo=$(python3 -c "print(round(${REPS} * ${RATE_DEMO} / 60, 1))")
est_infer_emb=$(python3 -c "print(round(${TOTAL_RECORDS} * ${RATE_INFER_EMB} / 60, 1))")
est_infer=$(python3 -c "print(round(${NON_REPS} * ${RATE_INFER} / 60, 1))")

log() {
    # `|| true` is load-bearing: with `pipefail` set, a tee failure (e.g. a bad LOG path) would
    # otherwise leak into this function's exit status - and since the script's last executed
    # command is always a log() call, that would make the *entire pipeline* falsely report
    # failure to SLURM even when every actual stage succeeded. Logging must never be able to do
    # that - it's diagnostic, not part of the pipeline's real success/failure signal.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" || true
}

notify() {
    # Real milestones only (start/stage-done/failed/pipeline-done) - always logs, and additionally
    # pushes to ntfy.sh if NTFY_TOPIC is set. `|| true` on the curl call for the same reason as
    # log()'s tee: a failed/slow notification must never be able to affect the pipeline's actual
    # exit status - it's a side channel, not part of the real success/failure signal.
    local message="$*"
    log "$message"
    if [ -n "$NTFY_TOPIC" ]; then
        curl -s -m 10 -o /dev/null -X POST "https://ntfy.sh/${NTFY_TOPIC}" -d "[${DATASET}] ${message}" || true
    fi
}

notify "########## PIPELINE START: ${DATASET} (${TOTAL_RECORDS} records, ~${REPS} cluster reps) ##########"
log "Baseline estimates (from CNFinBench actuals, scaled to this record count - refined below as stages complete):"
log "  preprocess ~${est_preprocess}m | cluster ~${est_cluster}m | demo ~${est_demo}m | infer_emb ~${est_infer_emb}m | infer ~${est_infer}m"
TOTAL_EST=$(python3 -c "print(round(${est_preprocess}+${est_cluster}+${est_demo}+${est_infer_emb}+${est_infer}, 1))")
log "  TOTAL ESTIMATE: ~${TOTAL_EST} minutes (~$(python3 -c "print(round(${TOTAL_EST}/60,1))") hours) - unverified until stages actually run"

# Sets the global STAGE_ACTUAL_MINUTES rather than returning via stdout/command-substitution:
# `$(run_stage ... | tail -1)` would silently swallow every log() line this function prints (they'd
# go into the pipe to `tail -1` instead of the terminal/SLURM stdout log), so the SLURM job's own
# output file would be missing every per-stage START/DONE line even though the separate
# milestones.log file had them - the two logs would silently disagree. Calling run_stage directly
# (no command substitution) lets its log() output flow straight through to both.
STAGE_ACTUAL_MINUTES=0
run_stage() {
    local stage="$1"
    local est_minutes="$2"
    log "=== START ${stage} (estimated ~${est_minutes}m) ==="
    local t0=$(date +%s)
    python -m AgentAuditor "$DATASET" "$stage"
    local status=$?
    local t1=$(date +%s)
    STAGE_ACTUAL_MINUTES=$(python3 -c "print(round((${t1}-${t0})/60, 1))")
    if [ $status -eq 0 ]; then
        notify "=== DONE ${stage}: took ${STAGE_ACTUAL_MINUTES}m (estimated ${est_minutes}m) ==="
    else
        notify "=== FAILED ${stage} after ${STAGE_ACTUAL_MINUTES}m (exit code ${status}) - stopping pipeline ==="
        exit $status
    fi
}

remaining_est=$TOTAL_EST
elapsed_total=0

run_stage preprocess "$est_preprocess"
actual_preprocess=$STAGE_ACTUAL_MINUTES
elapsed_total=$(python3 -c "print(round(${elapsed_total}+${actual_preprocess}, 1))")
remaining_est=$(python3 -c "print(round(${remaining_est} - ${est_preprocess}, 1))")
log ">>> elapsed so far: ${elapsed_total}m | remaining stages estimated: ~${remaining_est}m <<<"

run_stage cluster "$est_cluster"
actual_cluster=$STAGE_ACTUAL_MINUTES
elapsed_total=$(python3 -c "print(round(${elapsed_total}+${actual_cluster}, 1))")
remaining_est=$(python3 -c "print(round(${remaining_est} - ${est_cluster}, 1))")
log ">>> elapsed so far: ${elapsed_total}m | remaining stages estimated: ~${remaining_est}m <<<"

run_stage demo "$est_demo"
actual_demo=$STAGE_ACTUAL_MINUTES
elapsed_total=$(python3 -c "print(round(${elapsed_total}+${actual_demo}, 1))")
remaining_est=$(python3 -c "print(round(${remaining_est} - ${est_demo}, 1))")
log ">>> elapsed so far: ${elapsed_total}m | remaining stages estimated: ~${remaining_est}m <<<"

run_stage infer_emb "$est_infer_emb"
actual_infer_emb=$STAGE_ACTUAL_MINUTES
elapsed_total=$(python3 -c "print(round(${elapsed_total}+${actual_infer_emb}, 1))")
remaining_est=$(python3 -c "print(round(${remaining_est} - ${est_infer_emb}, 1))")
log ">>> elapsed so far: ${elapsed_total}m | remaining stages estimated: ~${remaining_est}m <<<"

run_stage infer "$est_infer"
actual_infer=$STAGE_ACTUAL_MINUTES
elapsed_total=$(python3 -c "print(round(${elapsed_total}+${actual_infer}, 1))")

notify "########## PIPELINE DONE: ${DATASET} — total actual runtime: ${elapsed_total}m (estimated ${TOTAL_EST}m) ##########"
