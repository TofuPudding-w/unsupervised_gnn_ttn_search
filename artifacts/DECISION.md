# Decision after the repaired evaluator and construction-cost check

The repaired evaluator completed 105 online runs and exhaustive evaluation of 768 pool candidates. All candidates pass the common policy. All 105 historical final CNOT/depth endpoints are unchanged. Padded screening finds the unique pool Pareto candidate first in all six targets; SCENT is that candidate in five pools. Only seed 15 improves over SCENT (955→805 CNOT, 795→742 D2). These are pool-specific conclusions, not global optimality claims.

The exact subtree-cache experiment preserves all 2,304 paired representations and score orders. However, it saves only about 0.06–0.15 seconds per pool at 8–10 qubits in median timings. Twelve-qubit median savings are about 2–4% and are not consistently positive in every paired repetition.

| Qubits | Family | Independent seconds, min–max | Shared seconds, min–max | Faster shared repetitions / 3 |
|---:|---|---|---|---:|
| 8 | hierarchical | 0.278–0.328 | 0.242–0.268 | 3 |
| 8 | overlapping | 0.295–0.362 | 0.231–0.257 | 3 |
| 10 | hierarchical | 0.657–0.750 | 0.548–0.659 | 2 |
| 10 | overlapping | 0.666–0.964 | 0.610–0.642 | 3 |
| 12 | hierarchical | 3.718–7.900 | 3.727–4.863 | 2 |
| 12 | overlapping | 4.656–4.963 | 4.459–4.895 | 2 |

Do not integrate this cache into the benchmark yet. This is a builder-cost probe: the independent pass retains reconstructed states and descriptors but releases node graphs as it proceeds, while the shared pass retains cached nodes. Production screeners retain prepared TTNs for subsequent compilation. The probe therefore does not establish production memory savings or end-to-end speedup. Retained-array counts also omit temporary Kronecker matrices, Python overhead and possible backing storage of views.

Keep GNN training paused. Existing pools provide no remaining shortlist-quality improvement over padded screening, and this experiment does not demonstrate a major construction bottleneck that learning must solve. The next scientific experiment should broaden a prespecified benchmark (target structure and/or candidate generation) and ask whether useful selection headroom exists. Do not select only instances where SCENT or padded screening loses. Keep the repaired compiler, error policy and fixed-score baselines. No such expanded benchmark or training run was launched here.
