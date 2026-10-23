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

Qiskit synthesis failures (including `TranspilerError`) are recorded as
`compiler_error` attempts with their exception type, traceback and elapsed
compilation time. The evaluator proceeds to the next already configured
tolerance. If no attempt produces a verified feasible circuit, the candidate
remains unsuccessful and search continues; no fidelity or CX label is invented
for a compiler failure. Unexpected programming exceptions still propagate.

The larger pilot exposed this case on hierarchical target seed 1, candidate 79.
Its tree is `((5, (4, (1, 7))), (6, (2, (0, 3))))`. Regression tests now cover
that candidate, recovery after a synthesis failure, and exhausted failures
without aborting search. This change handles the backend failure; it does not
patch Qiskit's synthesis implementation or relax fidelity requirements.

For an interrupted pre-fix run, keep its artifacts and use a new output directory
such as `artifacts/stage_a_pilot_v2` for the corrected run. The changed evaluator
changes the source fingerprint, so old evaluation records will not be reused
automatically. The old measurements remain available for inspection.

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

Focused follow-up checks are implemented in `scripts/run_focused_checks.py` and
`scripts/check_compiler_anomaly.py`. They write to `artifacts/stage_a_focused/`
and leave the pilot artifacts and production compiler settings unchanged.
The comparison script requires the pilot's evaluator source/software identities
to match the current environment before reusing its records. The committed
scripts are outside `src/ttn_search`, so adding analysis code does not invalidate
evaluation caches. The report audit records the analysis script fingerprints.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_compiler_anomaly.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_focused_checks.py
PYTHONPATH=src .venv/bin/python scripts/report_focused_checks.py
```

The focused protocol uses all six development targets for the neighborhood and
held-out prediction checks, and targets `hierarchical_1` and `overlapping_1` for
the alternative error allocation. The neighborhood method visits one-hop SCENT
neighbors after common initialization, then samples randomly; it is not greedy
local descent. Historical and new cached trajectories are explicitly replay
comparisons, not measured independent online speedups. The target selection and
heuristic-heavy held-out pool limit generalization claims.

To evaluate complete small candidate pools before running the larger search:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ttn_search.run_stage_a --config configs/stage_a_diagnostics.json --output artifacts/stage_a_diagnostics
```

This configuration evaluates all 32 candidates for each of six targets (192
target–tree requests). `search_seeds: []` skips all search trajectories, so its
`budget` and `initial_count` fields are unused. The pools follow the existing
heuristic-first generator; they are not uniformly sampled or source-balanced.
The report's best value is the optimum of each fully evaluated 32-tree pool,
not an optimum over all trees. The completed development run is saved in `artifacts/stage_a_diagnostics`.

After inspecting feasibility, cost variation and timing, run the larger search:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ttn_search.run_stage_a --config configs/stage_a_pilot.json --output artifacts/stage_a_pilot
```

These output directories have separate caches. Rerunning an unchanged command
reuses its completed candidate records but replays the search from the start;
it does not resume a saved GP. Source edits change cache keys. The pilot's first
32 diagnostics are a prefix of its 256-tree pool, not a complete evaluation of
that pool. No hard candidate timeout or memory cap is implemented yet.

Pyright diagnostics are suppressed only within the unmodified upstream
`vendor/treetensorcomp` directory via `tool.pyright.ignore`. Import resolution
and checks of our `src`/`tests` remain enabled. This does not repair or certify
all upstream execution paths; Stage A uses the tested exact-MI clustering path.

1. Complete balanced diagnostic sampling and the six-target pilot after resource
   calibration; report pool headroom separately from search quality.
2. Add enforced per-candidate resource limits and independent compiler-seed
   validation before interpreting larger experiments.
3. Compare finite-budget raw/PCA BO with random search on common initial points;
   diagnose sparse feasibility and proposal overhead.
4. Examine the alternate precision allocation and exact-synthesis limitations.
5. Gate GNN training on credible search headroom and useful surrogate predictions.

TCI, the research approximate compiler, mixed pretraining and final independent
experiments remain outside the current implementation. Stage B is described below.

## Stage B: structure-only representation pilot

Stage B is implemented in `src/ttn_search/learning/encoder.py` and
`scripts/run_stage_b.py`. Install CPU PyTorch into the existing environment:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run from the repository root, after the Stage A pilot/focused artifacts exist:

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_stage_b.py
.venv/bin/python scripts/report_stage_b.py
```

The `GLIBC_TUNABLES` setting reserves additional loader TLS space for Qiskit and
PyTorch in this Linux process; without it this environment reports “cannot
allocate memory in static TLS block.” It does not change numerical precision.
Use the same setting when running the combined test suite:

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Results, checkpoints, corpus, provenance, and traces are written to
`artifacts/stage_b`; the report is `artifacts/stage_b/REPORT.md`. The runner
repeats training and search on rerun, reusing matching evaluation caches; it is
not a training checkpoint resume. Archive that output directory before changing
the protocol. `requirements-stage-b.lock.txt` records the installed environment;
CPU torch wheels require the CPU index above.

The pilot uses 2,000 random training trees, 250 validation and 250 test trees,
excluding all six original candidate pools. Three independent GNN training
seeds are reported separately. Raw/PCA features and an untrained GNN are controls.
The encoder has four direction-aware GIN layers, width 64, and an eight-value
embedding. Only tree features enter its reconstruction pretraining.

Each target uses SCENT plus 127 randomly selected candidates from its original
pool. All searches share eight initial observations and budgets 16/32/64.
Held-out prediction uses 8/16 known labels and five shared splits. These remain
development targets, with offline cache replay; the experiment does not measure
independent online speedups or justify moving automatically to mixed pretraining.
The Stage A evaluator is unchanged; Stage B code has separate provenance in its
protocol because the legacy evaluator fingerprint covers only direct package files.

After the run, audit the saved results:

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_stage_b.py
```

This checks corpus/candidate separation, shared initialization, no repeated search
queries, incumbent values against cached labels, and trained-encoder permutation
invariance, label sensitivity, finite outputs, and latent variation.

## New-target neighborhood validation

The fixed configuration is `configs/neighborhood_validation.json`: five new
instances per existing family, seeds 3–7, three search seeds, and 8/16/32
additional evaluations after a common SCENT start. There is no fixed candidate
pool. The iterative method accepts strict improvements immediately and uses
random fallback after exhausting unseen neighbors. Failed evaluations count.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_neighborhood_validation.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/analyze_neighborhood_validation.py
```

Artifacts and report are in `artifacts/neighborhood_validation`. The runner
refuses to mix changed protocols in that directory. Reruns replay trajectories
and reuse matching evaluations; they do not resume a partially completed search.
Actual wall time includes cache effects and is not an independent online timing
comparison. Mechanism analysis selects the first improving and first feasible
non-improving neighbor from iterative seed 0, and records partition changes,
node ranks and separately synthesized local costs alongside final circuit costs.

## Bounded decision experiment before Stage C expansion

The frozen settings are in `configs/decision_experiment.json`. Two independent
runners write to `artifacts/decision_experiment/cost` and `/gnn`:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/decision_cost.py
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/decision_gnn.py
.venv/bin/python scripts/decision_report.py
```

The cost runner reproduces a two-tree transition under five truncation tolerances,
then extracts construction-only descriptors from saved evaluations and tests
held-out ranking and static screening. No compiled counts or successful-retry
ranks enter descriptor inputs. Primary new-target tests use only the global-random
samples plus SCENT, not adaptively selected neighborhood trees. Feature-based
screening pays for construction of the whole candidate set.

The GNN runner restarts all three original seeds for 300 epochs on the unchanged
corpus, verifies reproduction of the original first 100 epochs, and evaluates
selected checkpoints with unchanged downstream settings. It does not resume Adam
from a weights-only checkpoint. Both runners reject changed protocols in the same
output directory; rerunning repeats computation. Stage C expansion remains on hold.

Audit the completed decision experiment with:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_decision_experiment.py
```

The audit verifies recommendation eligibility and regret, label-free score order,
whole-pool construction charging, screening endpoints, local dimensions, and
paired GNN initialization/incumbents. The fixed padded-size score is a reference-
compiler diagnostic, not a claim of transfer to the SCENT approximate compiler.

## Fresh end-to-end screening benchmark

`configs/screening_benchmark.json` freezes untouched target seeds 8–12, shared
128-tree pools (SCENT plus random trees), three repetitions, and candidate/call
budgets 1/2/4/8/16. Each method runs independently without outcome caches. Screeners
reuse only TTNs they constructed themselves. Method order is randomized, and
actual elapsed-time comparisons include generation, full prescreening, model
loading/inference where used, compilation, retries, verification and artifact I/O.

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_screening_benchmark.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/report_screening_benchmark.py
```

Outputs are in `artifacts/screening_benchmark`. Completed method runs may be
resumed by rerunning an unchanged protocol; an unfinished method reruns its
computations, never reuses partial circuit labels. Frozen GNN checkpoints and
scalers use only old training data. SCENT-only is a one-candidate baseline.
See `POST_STAGE_B_GUIDE.md` for the experiment history and detailed code walkthrough.

Audit the fresh benchmark after completion:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_screening_benchmark.py
```

## Revised Stage C1: scaling pilot

Run `PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_stage_c1.py`.
Settings are frozen in `configs/stage_c1.json`; results go to `artifacts/stage_c1`.
This starts with two 10-qubit targets and five non-GNN baselines. Independent
fixed-evaluation and hard-deadline experiments use the same 128-tree pools.
Time runs continue beyond 16 evaluations when possible. Interrupted computations
are excluded from verified result prefixes. The fixed-evaluation safety ceiling
is 300 seconds per method; the time budget is 30 seconds. Inspect `controlled.json`
for each method, `runs.json` for aggregate progress, and `completion.json` for
completion. A missing checkpoint is not a failed-fidelity measurement.
Twelve qubits and a different compiler are subsequent decisions; GNN expansion
remains on hold. The required fidelity is 0.999.

### Offline C1 pool-headroom diagnostic

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/diagnose_c1_pool.py
```

This completes the existing overlapping seed-13 pool without changing its trees.
Outputs and `REPORT.md` are in `artifacts/stage_c1_headroom`. Existing completed
records are reused under a frozen selection rule. Missing candidates receive the
original evaluation policy with a 60-second ceiling; timeouts and operational
errors remain separate from infeasibility. New compiler starts are flushed before
synthesis. This is offline headroom analysis, not a fresh online timing benchmark.

### Accuracy-budget and alternative-synthesis development probe

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compilation_probe.py
python3 scripts/report_compilation_probe.py
```

The fixed three-tree probe writes to `artifacts/compilation_probe`; completed
configurations resume only under identical provenance. It compares local tolerance
multipliers and full-unitary QSD completion, while checking final fidelity and
recording original sub-budget compliance separately. It does not change the
benchmark evaluator. Block-analysis timing includes extra synthesis work and
must not be interpreted as a production compiler speed comparison.

## Revised policy validation and matched-update GNN study

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_revised_policy.py
python3 scripts/report_revised_policy.py
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536 PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib NUMBA_CACHE_DIR=/tmp/ttn-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_revised_gnn.py
```

The evaluator requires all original global accuracy budgets and tries local
multipliers 3/1/0.1/0 in that order. Reports/artifacts are under
`artifacts/revised_policy` and `artifacts/revised_gnn`. Completed items resume
under unchanged provenance. The GNN study trains size-specific ten-qubit models
on 2000 and 10000 trees with matched updates, then measures fresh independent
fixed-evaluation search on five untouched targets. This is not mixed pretraining
or an uncapped fixed-time benchmark.

After the GNN study completes, generate its report and run its artifact audit:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ttn-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/report_revised_gnn.py
```

The report compares matched-update reconstruction, per-target CNOT/depth,
fixed-evaluation quality, measured online costs, and paired data-size effects.
It verifies source/model fingerprints, corpus separation, accuracy requirements,
incumbent histories and logged synthesis attempts before reporting results.
