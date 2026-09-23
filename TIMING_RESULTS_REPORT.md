# AgentAuditor Latency Report

**Report date:** 2026-09-23  
**Runs analyzed:** 5 completed Slurm jobs covering 2,166 conversations  

This report contains wall-clock pipeline timings and conversation-level inference latency only. Job 43051574 is used for `cnfinbench-harmful-unblocked`; the earlier zero-conversation attempt is excluded.

## Run-level results

| Job | Dataset | Node | Conversations | Wall time | Recorded-stage time | Unrecorded gaps | Result |
|---|---|---:|---:|---:|---:|---:|---|
| 42455730 | `cnfinbench-harmless` | c0603a-s8 | 321 | 54 m 27.6 s | 54 m 4.0 s | 23.6 s | Complete |
| 42461503 | `cnfinbench-harmful` | c0603a-s18 | 321 | 58 m 7.0 s | 57 m 36.4 s | 30.6 s | Complete |
| 43051574 | `cnfinbench-harmful-unblocked` | c0602a-s8 | 180 | 27 m 21.4 s | 27 m 1.3 s | 20.1 s | Complete |
| 42703810 | `cnfinbench-harmless-unblocked` | c0602a-s7 | 301 | 1 h 22 m 44.8 s | 1 h 22 m 21.0 s | 23.8 s | Complete |
| 42792253 | `finvault-v5-fixed-benign-v-malicious` | c1105a-s21 | 1,043 | 3 h 12 m 28.5 s | 3 h 11 m 57.7 s | 30.8 s | Complete |

Wall time is the difference between `run.started_at` and `run.ended_at`. Recorded-stage time is the sum of all stage durations. The small difference includes process startup, imports, inter-stage launcher overhead, and other work outside timed scopes.

## Stage durations

| Dataset | Preprocess | Cluster | Demo generation | Demo repair | Retrieval (`infer_emb`) | Inference | Fix 1 | Fix 2 | Evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnfinbench-harmless` | 14 m 15.3 s | 3 m 26.8 s | 4 m 33.7 s | 3 m 41.9 s | 4 m 2.8 s | 24 m 0.9 s | 0.50 s | 0.14 s | 1.87 s |
| `cnfinbench-harmful` | 17 m 20.0 s | 3 m 31.8 s | 4 m 32.3 s | 3 m 31.6 s | 3 m 59.3 s | 24 m 38.8 s | 0.45 s | 0.13 s | 2.10 s |
| `cnfinbench-harmful-unblocked` | 8 m 12.4 s | 2 m 1.7 s | 2 m 4.9 s | 1 m 21.1 s | 2 m 0.4 s | 11 m 18.9 s | 0.26 s | 0.07 s | 1.67 s |
| `cnfinbench-harmless-unblocked` | 24 m 32.9 s | 3 m 17.6 s | 5 m 56.9 s | 9 m 44.0 s | 3 m 54.5 s | 34 m 52.7 s | 0.51 s | 0.15 s | 1.81 s |
| `finvault-v5-fixed-benign-v-malicious` | 1 h 42 m 59.4 s | 2 m 30.0 s | 7 m 33.3 s | 10 m 7.4 s | 3.95 s | 1 h 8 m 12.5 s | 0.21 s | 0.08 s | 30.86 s |

### Stage distribution observations

- The two full 321-record CNFinBench runs are closely matched. Harmful took 3 m 39.5 s longer overall, primarily because preprocessing was 3 m 4.7 s slower.
- The completed harmful-unblocked rerun was the fastest end-to-end job and had the highest inference throughput. Its 180 conversations completed inference at 15.91 conversations/minute.
- The 301-record harmless-unblocked run was substantially slower than both 321-record runs. Its preprocessing was 24 m 32.9 s, demo repair was 9 m 44.0 s, and inference was 34 m 52.7 s. This indicates higher model/API latency or more expensive outputs, not simply a larger input count.
- FinVault retrieval took only 3.95 s, compared with roughly two to four minutes for the complete CNFinBench runs. This is consistent with a warm embedding cache or materially different cached retrieval workload and should not be interpreted as an uncached benchmark comparison without checking cache state.
- The JSON repair stages after inference were negligible in every complete run. This likely means few or no outputs required the LLM-based second repair pass.

## Conversation-level inference latency

All recorded conversations have `round_count=1`; therefore these files do not support a latency-versus-round-count analysis.

| Dataset | N | API-time sum | Mean | Median | P90 | P95 | P99 | Min | Max | Inference throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnfinbench-harmless` | 321 | 23 m 3.1 s | 4.31 s | 3.81 s | 6.48 s | 7.82 s | 11.73 s | 1.25 s | 24.18 s | 13.37 conv/min |
| `cnfinbench-harmful` | 321 | 23 m 40.8 s | 4.43 s | 4.02 s | 6.63 s | 7.57 s | 9.86 s | 1.51 s | 18.35 s | 13.02 conv/min |
| `cnfinbench-harmful-unblocked` | 180 | 10 m 59.6 s | 3.66 s | 3.38 s | 4.89 s | 5.50 s | 8.66 s | 1.74 s | 17.96 s | 15.91 conv/min |
| `cnfinbench-harmless-unblocked` | 301 | 34 m 0.1 s | 6.78 s | 5.85 s | 10.84 s | 12.91 s | 19.50 s | 2.69 s | 27.32 s | 8.63 conv/min |
| `finvault-v5-fixed-benign-v-malicious` | 1,043 | 1 h 6 m 38.6 s | 3.83 s | 3.52 s | 5.93 s | 7.29 s | 8.91 s | 0.96 s | 14.47 s | 15.29 conv/min |

Percentiles use linear interpolation over the recorded per-conversation `total_seconds` values. Throughput uses the full `infer` stage duration, not only the sum of API calls.

### Slowest recorded conversations

| Dataset | Conversation ID | Duration |
|---|---|---:|
| `cnfinbench-harmless` | `harmless-MT_Cog-50` | 24.18 s |
| `cnfinbench-harmful` | `harmful-MT_Inter-129` | 18.35 s |
| `cnfinbench-harmful-unblocked` | `harmful-unblocked-MT_Inter-83` | 17.96 s |
| `cnfinbench-harmless-unblocked` | `harmless-unblocked-MT_App-61` | 27.32 s |
| `finvault-v5-fixed-benign-v-malicious` | `finvault-v5-fixed-00-attack-case_fixed_0039` | 14.47 s |

## Aggregate accounting

The five completed jobs occupied **6 h 55 m 9.3 s** of wall time. Their timed stages sum to **6 h 53 m 0.4 s**, leaving **2 m 8.9 s** outside timed scopes.

| Stage | Total across five completed jobs | Share of total wall time |
|---|---:|---:|
| Preprocess | 2 h 47 m 20.0 s | 40.3% |
| Cluster | 14 m 48.0 s | 3.6% |
| Demo generation | 24 m 41.1 s | 5.9% |
| Demo repair | 28 m 26.0 s | 6.8% |
| Retrieval (`infer_emb`) | 14 m 0.9 s | 3.4% |
| Inference | 2 h 43 m 3.7 s | 39.3% |
| Fix 1 | 1.93 s | <0.1% |
| Fix 2 | 0.58 s | <0.1% |
| Evaluation | 38.31 s | 0.2% |

The five completed runs contain **2,166 conversations** and **2 h 38 m 22.2 s** of recorded per-conversation API time.
