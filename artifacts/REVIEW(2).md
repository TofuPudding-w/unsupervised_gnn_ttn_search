# Completed exploratory replication: persistence, magnitude and limits

## Audit

All 80 paired online targets and all 80 offline diagnostic targets completed.
There are 3359 fresh candidate records and 3359 synthesis calls; every record
succeeded. Recorded evaluation activity is 7110.52 seconds (about 1.98 hours),
not a single-method search time. Frozen source/input fingerprints and all saved
QPY CX/D2 checks were rerun. Eighteen distinct selected-positive/anchor circuits
were additionally reloaded and independently simulated; all passed original
representation, compilation and final-fidelity requirements. This review did
not resimulate all 3359 circuits or run new synthesis.

Minimum recorded final fidelity: 0.9999004510435686. Maximum recorded
representation infidelity: 9.954895643049522e-05. Maximum recorded compilation
infidelity: 9.992007221626409e-16.

## Actual method results: some benefits replicate, no uniform improvement

The primary endpoint is each method's actual returned K=4 circuit, chosen by
(CX,D2,hash), after constructing 24 candidates. On 80 settings:

| Paired result | Count |
|---|---:|
| Mixed no worse in either resource and strictly better in at least one | 32 |
| Binary no worse in either resource and strictly better in at least one | 20 |
| Resource tradeoff | 25 |
| Exact resource tie | 3 |

These are 10 independent base groups, not 80 independent samples. Averaging
relative changes within each seed and then across the ten seeds gives CX
-0.30095% and D2 +10.98682%. Three seeds have negative mean CX change; two have
negative mean D2 change; only seed1105 improves both seed means. The median
per-setting CX change is -1.7016%, but occasional regressions offset frequent
small gains. No statistical significance or universal advantage is claimed.

The policy optimizes CX first and does not enforce a depth cap. Thus increased
returned depth cannot be interpreted as an intrinsic theorem about ternary
circuits or as a depth-constrained experiment.

Mean measured search times are 6.7383s for B and 6.6981s for M; median paired
M/B time ratio is 0.99844. With one timing sample per method/target and startup
variation, this does not establish a speedup. Both methods pay all generation,
construction, ranking, fresh shortlist reconstruction/compilation and verification.

## Practical gains survive measured stronger binary controls on some targets

Thirteen actual selected mixed circuits meet the original practical count/depth
margin against their budget-matched selected B circuit. Rechecking those actual
selected circuits against every measured binary diagnostic yields seven passing
settings across six seeds (1102,1103,1104,1105,1108,1109). This differs from the
eight passing offline M-pool findings: an unreturned candidate is not an actual
method success.

| Target | Selected mixed CX/D2 | Minimum-CX measured binary CX/D2 | Minimum-D2 measured binary CX/D2 | Passing route |
|---|---:|---:|---:|---|
| g3_1102_a0.3_g0.0 | 186/119 | 188/143 | 197/129 | depth |
| g2_1102_a1.2_g0.15 | 273/159 | 279/177 | 279/177 | depth |
| g2_1103_a1.2_g0.0 | 49/23 | 94/75 | 105/55 | count and depth |
| g2_1104_a1.2_g0.15 | 97/65 | 147/103 | 158/90 | count and depth |
| g2_1105_a1.2_g0.15 | 171/111 | 173/133 | 173/133 | depth |
| g3_1108_a1.2_g0.15 | 384/303 | 425/361 | 460/305 | count |
| g3_1109_a0.3_g0.15 | 522/513 | 627/558 | 627/558 | count and depth |

These are genuine selected-circuit signals, not only hidden-reference examples.
For example the last target improves on B's original selected 783/714, but the
more appropriate strengthened count anchor is 627/558: the remaining gain is
105 CX (16.75%) and 45 D2, rather than attributing all 261 CX to an arity effect.

Every target received the same stronger diagnostic procedure: generate 64 extra
binary candidates, compile the union of padded-top4 and parameter-top4. The
64 were not exhaustively compiled; their screening/construction is an extra
paid diagnostic, not a free equal-budget algorithm. Nor were all binary
refinements of hidden generating structures enumerated here. Conclusions are
limited to measured alternatives and do not overturn P1's frozen decision.

## What selection and hidden references reveal

Padded K=4 attains the observed pool minimum CX in 80/80 B pools and 79/80 M
pools. The remaining M case has seven-CX regret. This is strong count retrieval,
not proof that every useful Pareto point is retrieved. There is still little
observed room for a more complicated CX-ranking model within these pools.

Thirteen of 40 hidden ternary references meet the practical measured-binary
comparison, spanning nine seeds. They remain existence diagnostics, excluded
from algorithmic success. The ordinary algorithms discovered useful alternatives
on some targets, but do not generally recover the hidden structure.

Of the 80 actually returned M candidates, 74 come from fixed local merges of
SCENT-perturbed trees, five are shared binary candidates, and one is random
mixed arity. Six of the seven strengthened positive selections come from local
merges; the seventh (g3_1109_a0.3_g0.15) is the sole selected random mixed tree.
These source counts are descriptive, not a controlled attribution of causality:
B and M also differ in candidate generation and representation truncation.

## Direction decision

The small-benefit question has a qualified positive answer: useful automatically
found mixed structures recur, including several substantial successes beyond
measured stronger binary controls. The stronger claim of a broadly preferable
default mixed generator is not supported: mean CX is almost unchanged, depth
is often worse, and important losses remain.

Retain mixed-arity generation as a potentially useful optional candidate source,
not a replacement for binary search. Do not resume unsupervised GNN+BO, announce
P1 passed retrospectively, or launch more target/parameter sweeps automatically.
The experiment ends as promised. Any later proposal should explain which
observable target/structure properties distinguish successful candidates from
regressions and include simple rules and budget-matched generation baselines.
An enlarged union pool or selecting the better B/M outcome is not free and has
not been established as an equal-budget online method by this experiment.

Complete paired tables: REPORT.md. Derived selected-vs-strengthened comparisons:
review_statistics.json. Original paired trajectories, offline pool labels and
seed summaries remain separate and unchanged.
