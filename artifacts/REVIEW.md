# Completed ternary preflight: independent review

The frozen-source/input report audit was rerun successfully. All 80 saved QPY
circuits were then loaded and independently simulated; CX and D2 were recounted,
and local tensors were reconstructed against the stored common states.

- 16 paired cases on four prespecified targets; 80/80 successful compilations.
- Exactly 80 synthesis starts; no interrupted runs or additional attempts.
- Minimum final fidelity: 0.9999030469106125 (required 0.999).
- Maximum common-state amplitude deviation: 3.9104838266427664e-16.
- Maximum independently recomputed compilation infidelity: 8.881784197001252e-16.
- All original representation and compilation sub-budgets pass.

## Main comparison

Against the best of the three binary refinements, ternary uses fewer CX in
13/16 cases: twelve save six CX, one saves four. All thirteen tie the lowest
binary D2. In the other three cases one actual binary refinement dominates
ternary in both resources:

| Case | Best binary CX / D2 | Ternary CX / D2 |
|---|---:|---:|
| hierarchical_21_comb | 156 / 155 | 197 / 195 |
| overlapping_21_comb | 1262 / 1238 | 1468 / 1441 |
| ttn_101_a0.3_g0.15_comb | 1262 / 1238 | 1468 / 1441 |

The small CX benefit occurs on all four targets, but there is **no D2 improvement
beyond the best binary refinement in any case**. Sixteen cases are not sixteen
independent targets. The fixed starting trees and single edge per tree limit
coverage; no general statement about all ternary TTNs follows.

Original-tree comparisons can exaggerate the ternary-specific contribution.
For ttn_102_a0.3_g0.15_scent, original 662/480 becomes 621/439 after merging,
but binary_01 already gives 627/439: ternary's incremental benefit over that
refinement is only six CX and zero D2. For ttn_101_a0.3_g0.15_comb, merging
improves the original 1686/1654 to 1468/1441, yet binary_02 reaches 1262/1238.
These comparisons do not prove a causal mechanism for each gate saving;
regrouping and SVD gauge changes remain part of the binary refinements.

## Descriptive padded-score check (post-experiment)

Using the existing sum of padded local matrix sizes, the minimum-score variant
among ternary and all three refinements is unique in every case. It selects
ternary in the thirteen winning cases and the dominating binary refinement in
the other three. Thus it attains the observed minimum CX in all sixteen cases,
and also the observed minimum D2. Including the original diagnostic introduces
only harmless ties in these results.

This is a post-experiment diagnostic using an existing score, not an independent
held-out selection benchmark or a new preregistered success criterion. Computing
the refinements and their scores costs work; no online speedup is established.
The pattern is consistent with matrix-size effects, but a detailed attribution
of the repeated six-CX savings would require block-level synthesis analysis.

## Direction decision

Retain local merge/refinement as a small optional compiler optimization, with
simple padded-size selection as the baseline. Do not expand to a multifurcating
GNN or independently optimized high-arity TTNs on this evidence. The observed
increment beyond the best binary refinement is small, supplies no additional
depth reduction, and is selected correctly by the existing simple score here.

The main research direction remains circuit-cost-aware binary candidate
generation plus screening. The known missing good binary candidates provide
more direct evidence for that gap. If local regrouping is integrated later,
freeze its rule and measure total construction/refinement/scoring/compilation
cost on untouched targets. Do not claim timing or memory gains from the current
single repetitions and fixed execution order.

Raw recomputed checks and score comparisons: `independent_review.json`.
Original complete resource tables: `REPORT.md`. No new synthesis was performed
for this review; frozen code, tensors, input selection and evaluation policy
were unchanged.
