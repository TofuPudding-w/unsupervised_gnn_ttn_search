# Versioned compiler repair validation

Compiler: `verified-uc-diagonal-v1-qiskit-2.6.0`. Installed Qiskit remains unchanged. The frozen `compile_reference` remains available and unchanged.

## Mechanism and repair

The saved UCGate circuit implements the intended gate up to a diagonal, but the returned diagonal has incorrect relative signs. Independently reconstructing the control-branch 2×2 matrices reveals the mismatch. The adapter checks D U = target, recomputes row phases only when needed, and uses a local uniformly controlled Euler-rotation fallback if the residual exceeds 1e-8 per matrix entry. The fallback includes control-dependent phases and is independently verified as a full operator; it is rejected if verification fails. Exact sign corrections are retained when consistent with that threshold. The diagonal correction adds no gates; the Euler fallback can increase the local gate count.

The adapter freezes each checked circuit and uses its verified diagonal in the existing Isometry elimination algorithm, including multi-controlled gate steps. It does not alter input tensors, truncation tolerances, or fidelity requirements. Both ordinary and repaired circuits undergo independent statevector verification. The precise internal Rust numerical cause of the bad signs is not claimed to be repaired upstream.

Verification stores one 2×2 matrix per control assignment, rather than a full multiplexer operator. It costs O(branches × emitted gates), so the extra work is real and may limit scaling. Private Qiskit interfaces are version-gated to 2.6.0. The rare Euler fallback uses a full local operator for verification and therefore has higher memory cost. Unexpected circuit structure, unsupported versions and failed fallback verification fail explicitly. No global monkeypatches or vendor edits are used.

## Paired direct 3× checks

28 cases: original hierarchical failure, every unique retry candidate in the six-pool audit, SCENT/padded-first controls and one additional successful candidate per pool. All are development cases, not new held-out tests. 12 direct frozen compilations fail the compilation budget; all repaired compilations have fidelity greater than 1−1e-8 to the reconstructed TTN.

| Case | Role | Frozen Fcomp | Repaired Fcomp | CX old/new | D2 old/new | Corrected diagonals / Euler fallbacks | Compile seconds old/new |
|---|---|---:|---:|---|---|---:|---|
| hierarchical_13_random | original_failure | 0.0327020652 | 1.0000000000 | 1417/1417 | 1374/1374 | 1/0 | 1.245/1.088 |
| overlapping_13_0 | successful_control | 1.0000000000 | 1.0000000000 | 592/592 | 436/436 | 0/0 | 0.626/0.555 |
| overlapping_13_43 | historical_retry | 0.9406410897 | 1.0000000000 | 2662/3043 | 2390/2767 | 0/1 | 1.918/2.868 |
| overlapping_13_51 | historical_retry | 1.0000000000 | 1.0000000000 | 1867/1867 | 1042/1042 | 0/0 | 1.452/1.400 |
| overlapping_13_54 | historical_retry | 0.5584780534 | 1.0000000000 | 3556/3556 | 2281/2281 | 1/0 | 2.418/2.649 |
| overlapping_13_62 | historical_retry | 0.5606533478 | 1.0000000000 | 1676/1721 | 1386/1430 | 0/1 | 1.247/1.182 |
| overlapping_13_92 | historical_retry | 0.3619298295 | 1.0000000000 | 1869/1869 | 1577/1577 | 1/0 | 1.545/1.377 |
| overlapping_13_1 | successful_control | 1.0000000000 | 1.0000000000 | 1196/1196 | 1158/1158 | 0/0 | 1.015/0.898 |
| overlapping_14_0 | successful_control | 1.0000000000 | 1.0000000000 | 592/592 | 436/436 | 0/0 | 0.521/0.655 |
| overlapping_14_117 | historical_retry | 1.0000000000 | 1.0000000000 | 1867/1867 | 1575/1575 | 0/0 | 1.464/1.403 |
| overlapping_14_1 | successful_control | 1.0000000000 | 1.0000000000 | 1743/1743 | 1665/1665 | 0/0 | 1.396/1.478 |
| overlapping_15_0 | successful_control | 1.0000000000 | 1.0000000000 | 955/955 | 795/795 | 0/0 | 0.783/0.806 |
| overlapping_15_3 | historical_retry | 0.2889388758 | 1.0000000000 | 3169/3169 | 2069/2069 | 2/0 | 2.093/2.336 |
| overlapping_15_14 | historical_retry | 0.9745570340 | 1.0000000000 | 2874/2874 | 2603/2603 | 2/0 | 2.263/2.057 |
| overlapping_15_39 | historical_retry | 0.6903252576 | 1.0000000000 | 1888/1933 | 1599/1643 | 0/1 | 1.414/1.351 |
| overlapping_15_77 | successful_control | 1.0000000000 | 1.0000000000 | 805/805 | 742/742 | 0/0 | 0.790/0.714 |
| overlapping_15_1 | successful_control | 1.0000000000 | 1.0000000000 | 2531/2531 | 2255/2255 | 0/0 | 1.771/1.872 |
| overlapping_16_0 | successful_control | 1.0000000000 | 1.0000000000 | 588/588 | 435/435 | 0/0 | 0.556/0.503 |
| overlapping_16_8 | historical_retry | 0.9287277093 | 1.0000000000 | 2874/2874 | 2603/2603 | 1/0 | 2.185/1.981 |
| overlapping_16_11 | historical_retry | 0.6472487153 | 1.0000000000 | 1890/1890 | 1601/1601 | 1/0 | 1.405/1.518 |
| overlapping_16_1 | successful_control | 1.0000000000 | 1.0000000000 | 2239/2239 | 2155/2155 | 0/0 | 1.586/1.530 |
| overlapping_17_0 | successful_control | 1.0000000000 | 1.0000000000 | 592/592 | 436/436 | 0/0 | 0.607/0.523 |
| overlapping_17_20 | historical_retry | 1.0000000000 | 1.0000000000 | 1832/1832 | 1529/1529 | 0/0 | 1.450/1.424 |
| overlapping_17_39 | historical_retry | 0.2713801570 | 1.0000000000 | 1711/1711 | 1388/1388 | 1/0 | 1.174/1.423 |
| overlapping_17_1 | successful_control | 1.0000000000 | 1.0000000000 | 1596/1596 | 1546/1546 | 0/0 | 1.302/1.162 |
| overlapping_18_0 | successful_control | 1.0000000000 | 1.0000000000 | 592/592 | 439/439 | 0/0 | 0.694/0.548 |
| overlapping_18_58 | historical_retry | 0.7407727790 | 1.0000000000 | 3169/3262 | 2069/2160 | 0/1 | 2.128/2.089 |
| overlapping_18_1 | successful_control | 1.0000000000 | 1.0000000000 | 1467/1467 | 1359/1359 | 0/0 | 0.942/1.167 |

Representation-budget failures at direct 3×: ['overlapping_13_51', 'overlapping_14_117', 'overlapping_17_20']. These still require the frozen tolerance fallback despite correct compilation. Direct synthesis here is diagnostic; the production evaluator must continue to reject representation-infeasible attempts before synthesis.


Of 13 successful controls, 0 change CNOT or two-qubit depth. Mean compilation time across all cases: frozen 1.357 s, repaired 1.377 s. These are single paired measurements with alternating order, not a repeated timing benchmark.

## Status and migration

The opt-in repaired compiler is implemented and validated on this bounded cohort. This does not certify arbitrary matrices or larger systems. Keep final-fidelity and sub-budget checks and the common fallback policy when integrating it into evaluations.

Use `from ttn_search.compilers.verified import compile_verified, COMPILER_VERSION`. The existing reference compiler is not redirected. Historical labels, reports and fallback outcomes remain untouched. This report compares direct 3× attempts; corrected 3× costs are not interchangeable with historical accepted 1× fallback costs. A full pool or method comparison requires a separately versioned evaluator run, rebuilding outcomes for every compared method. That refresh is not performed here.
