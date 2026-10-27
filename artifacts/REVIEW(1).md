# P1 completed: interpretation and direction decision

The frozen source/input audit and resource recount were rerun successfully.
All 878 accepted QPY circuits passed recorded-hash, CX and logical D2 checks.
Stored target, representation and circuit-output vectors were independently
rechecked for all 878 accepted results; their fidelities agree with the records.
This review did not resimulate all QPY circuits or perform additional synthesis.

## Completeness and correctness

- 802 main candidate/reference evaluations + 12 sensitivity evaluations + 64
  conditional stronger-binary evaluations = 878 completed records.
- All 878 succeeded; 879 actual synthesis starts. Nine candidates used retries;
  not every retry involved synthesis because representation checks come first.
- Minimum final fidelity: 0.9999006908384327.
- Maximum representation infidelity: 9.930916156708491e-05.
- Maximum compilation infidelity: 8.881784197001252e-16.
- Recorded evaluation activity: 1810.10 seconds, approximately 30.2 minutes;
  this is offline experiment activity, not an online selector speedup and does
  not include every preparation/testing activity before launch.

## The only preliminary gate pass did not survive the practical threshold

`generated2_701_a0.3_g0.15` triggered the prespecified 64-tree stronger binary
search. The relevant actual circuits are:

| Circuit | CX | D2 |
|---|---:|---:|
| Preliminary minimum-CX binary anchor | 789 | 685 |
| Winning ordinary mixed candidate | 748 | 685 |
| Strengthened minimum-CX binary anchor | 767 | 700 |
| Strengthened minimum-D2 binary anchor | 803 | 616 |

Initially the mixed candidate saved 41 CX without increasing D2, exceeding
max(20, 5% of 789). After strengthening it still improves the count anchor by
19 CX and 15 D2, but 19 is below max(20, 5% of 767) = 38.35 CX.
It does not pass the depth route against the 616-depth anchor either.
Thus the benefit did not become zero or reverse; it became insufficient for
the frozen continuation criterion. No base group passes the final criterion.
The other targets did not trigger strengthening; their complete sets of
81 binary refinements were NOT all compiled. Do not claim exhaustive binary
optimality or universal failure of ternary representations.

## Ordinary mixed pools and resource tradeoffs

Across the 20 settings, M has a lower minimum CX than B in 15 and a higher
minimum CX in five. This does not imply 15 Pareto improvements: the minima for
CX and D2 can refer to different circuits, and lower count can trade for depth.
For example, `generated3_701_a1.2_g0.15` has binary 197/97 and mixed 191/152:
six fewer CX with substantially greater depth.
B and M differ in generation strategy as well as arity, so this comparison
alone does not identify an arity-specific causal effect.

## Sensitivity and selection

All twelve common-local-tolerance controls return the same CX and D2 as their
main-policy counterparts. Eight controls are binary, for which the main
nonroot-node denominator is already eight; only the four mixed controls change
the initial denominator. Therefore this is limited robustness evidence, not
proof that error allocation never affects arity comparisons. Both policies
retain the original global accuracy requirements.

Fixed-pool retrieval over 20 targets x 2 pools x 3 tie seeds gives:

| Score | Exact pool minimum at K=1 | Maximum K=1 CX regret | Exact pool minimum at K=8 |
|---|---:|---:|---:|
| Padded size | 110/120 | 3 | 120/120 |
| Parameter count | 91/120 | 67 | 120/120 |

The 120 comparisons are not independent targets. K counts shortlisted
candidates in this offline retrieval analysis, with no free SCENT initialization;
these are not measured end-to-end runtime comparisons. The simple padded score
remains strong, and there is no demonstrated need for a learned selector here.

## Hidden generating references

Some generating ternary trees improve count but trade against depth. For
`generated3_701_a0.3_g0.0`, the hidden generating reference is 168/119 whereas
an observed binary circuit is 197/96. This is a resource tradeoff, not binary
inability to prepare the state. Other generating references can be worse than
ordinary candidates. These references are diagnostic and cannot be credited
as algorithmically discovered structures.

## Decision

Stop at P1 as agreed. Do not launch P2a/P2b, expand GNN pretraining, or add new
ternary target sweeps merely to obtain a passing case. The supported outcome is:
independent mixed-arity construction works correctly, some small or tradeoff
benefits exist, but the frozen practical continuation criterion is not met and
simple selection already performs well in these pools.

A possible future main direction remains circuit-cost-aware binary candidate
generation with simple screening and optional local compiler regrouping. That
would require its own concrete method and bounded evaluation proposal, rather
than automatic continuation of this mixed-arity experiment. Only two generated
base seeds were used; negative conclusions remain local to this development
cohort, candidate policies, error policy and repaired reference compiler.
