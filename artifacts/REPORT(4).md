# C1 overlapping-pool headroom diagnostic

## Scope

Offline diagnostic of the original 10-qubit overlapping target, seed 13, and its frozen 128-tree pool. Existing completed records are reused; each remaining candidate receives one evaluation with the original internal precision retries and fidelity requirement 0.999. New evaluations have a 60-second per-candidate ceiling. No new candidate generator, score, target, or compiler is introduced.

Record selection is frozen before new compilation: the lexicographically first existing completed record for each tree is used, independent of its cost. All duplicate records are checked for status/cost agreement. Completed artifacts from deadline runs may be reused here even if they were not credited online; this does not alter the original benchmark.

## Headroom

- Status counts: {'success': 128}.
- SCENT: **1163 CNOTs**.
- Best verified candidate: **1163 CNOTs**.
- SCENT gap to best verified: **0 CNOTs (0.00%)**.
- Strict improvements over SCENT: **0/128**.
- Verified ties with SCENT: **1**.

All candidates have completed measured evaluations under the fixed policy; the best feasible value is the policy-defined pool optimum.

## Score retrieval

Ranks are one-based; ties use the original repetition-0 random permutation, without cost labels. Best-of-K is an offline replay, not a fresh timing experiment.

| Score | SCENT rank | First improvement rank | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 | K=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | 1 | none | 1163 | 1163 | 1163 | 1163 | 1163 | 1163 | 1163 | 1163 |
| padded | 2 | none | 1171 | 1163 | 1163 | 1163 | 1163 | 1163 | 1163 | 1163 |

## Improving candidates

| Candidate | CNOT | Gain over SCENT | Parameter rank | Padded rank |
|---|---:|---:|---:|---:|

No evaluated candidate strictly improves SCENT. This pool provides no observed improvement opportunity; it does not show whether better trees exist outside the pool.

## Cost and limitations

Reused 58 records; newly evaluated 70 candidates. New controlled evaluation wall time totals 152.11 seconds. New synthesis starts, including interrupted attempts: 71.

New synthesis attempts are flushed and fsynced before compilation. JSON results use the existing atomic temporary-file replacement helper. Offline reuse and candidate time ceilings make these costs unsuitable as online speedup evidence. A single target cannot establish size-dependent behavior or compiler independence. Larger/mixed GNN pretraining remains on hold.
