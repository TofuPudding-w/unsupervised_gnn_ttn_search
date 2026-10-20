# Learning-based TTN structure search

Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Original research design: [学习式TTN结构搜索_整体方案与实验流程.md](学习式TTN结构搜索_整体方案与实验流程.md).

## Current implementation

Stage A has started. The implemented path is normalized dense target → nested
hierarchical SVD → local isometries → Qiskit `u`/`cx` circuit → exact statevector
verification. Physical leaf labels are preserved throughout; complex states,
non-power-of-two bond ranks and output wire ordering are covered by tests.

Available methods are single SCENT, random search, raw LCA/leaf-depth-feature BO,
and PCA-feature BO. All search methods share a pool and initial observations
including SCENT. The PCA/scaler fit uses 512 independently generated target-free
random trees, not performance labels or the evaluation targets.

The pinned upstream source is in `vendor/treetensorcomp`; provenance and license
are documented in [vendor/SCENT_REVISION.md](vendor/SCENT_REVISION.md). The baseline
uses upstream recursive spectral partitioning with exact statevector mutual
information. It is labeled `SCENT_exact_state_mutual_information`, not a full
reproduction of the paper's default affinity estimation or approximate compiler.
The newest installed SciPy is used only for the dense reference path; upstream
TCI's older SciPy compatibility requirement remains a separate integration task.

## Environment and commands

The current environment was tested with Python 3.12. Exact installed dependencies
are captured in `requirements-stage-a.lock.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-stage-a.lock.txt
MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ttn_search.run_stage_a --config configs/stage_a_smoke.json --output artifacts/stage_a_smoke
```

The smoke configuration uses two 8-qubit targets, 32-candidate pools, four common
initial evaluations, eight evaluations per method and one search seed. These are
explicitly reduced engineering settings, not the final stage A protocol.

`configs/stage_a_pilot.json` describes the planned larger run: six targets,
256-candidate pools, eight initial observations, 64 evaluations and three seeds.
The full pilot has **not** been run. Change `--config` and `--output` to run it;
check the smoke timing and resource limits first. Trajectories expose the
16/32/64 checkpoints without three separate searches.

## First measured smoke results

All nine tests passed. Across both targets, 35 unique target–tree evaluations
met the final squared-fidelity threshold of 0.999.

| Target | Unique evaluations | SCENT CX | Best seen CX | Worst seen CX |
|---|---:|---:|---:|---:|
| Hierarchical, seed 0 | 16 | 75 | 75 | 777 |
| Overlapping, seed 0 | 19 | 504 | 362 | 777 |

All three searches finished at the same best cost on each target. The 362-CX
candidate was already in the shared initial observations. This establishes
circuit-cost variation and some candidate headroom on the overlapping target;
it supplies **no evidence of a BO or learned-representation advantage**.

The generated [smoke report](artifacts/stage_a_smoke/REPORT.md), manifests,
candidate pools, traces, per-attempt fidelity records, QPY circuits and NPZ
verification states are in `artifacts/stage_a_smoke/`. Generated artifacts are
ignored by git; preserve or copy them explicitly when sharing results.

## Evaluation and accounting

The initial reference compiler uses one exact synthesis per tolerance and stops
at the first feasible tolerance. It has no stochastic restart search. Every
attempt is recorded, and fidelity and CX refer to the same circuit. Global
representation and compilation errors are checked separately from final fidelity.

Cache identity includes normalized target content/metadata, rooted labeled tree,
precision/compiler settings and seed, numerical software versions, and local
source fingerprint. QPY outputs can be reloaded with `qiskit.qpy.load`.

Shared-cache experiments report both actual elapsed time and charged evaluation
replay time. The latter is not measured independent online speed. Full online
accounting additionally includes recorded SCENT discovery and pool generation.
No hard per-candidate wall-clock or memory limit is enforced in this first
small-system implementation; production resource enforcement is still pending.

## Next stage A work

1. Complete balanced diagnostic sampling and the six-target pilot after resource
   calibration; report pool headroom separately from search quality.
2. Add enforced per-candidate resource limits and independent compiler-seed
   validation before interpreting larger experiments.
3. Compare finite-budget raw/PCA BO with random search on common initial points;
   diagnose sparse feasibility and proposal overhead.
4. Examine the alternate precision allocation and exact-synthesis limitations.
5. Gate GNN training on credible search headroom and useful surrogate predictions.

TCI, the research approximate compiler, GNN training, mixed pretraining and final
independent experiments are not implemented by this initial stage A delivery.
