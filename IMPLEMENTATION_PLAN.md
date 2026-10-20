# Learning-based TTN structure search — implementation plan

Updated: 2026-10-20. This is a research implementation plan, not a statement of results.

## Objective and scope

Given a classically accessible target state, a fidelity threshold and a classical search budget, return the lowest-CNOT verified feasible circuit found through TTN structure search. Optimize final squared fidelity F = |<psi|U|0>|² subject to F >= F0; report depth, runtime and failures separately. Keep physical bit labels fixed. Rank selection is controlled by a shared precision policy, not learned.

The first cycle uses fixed-size, nonnegative targets, frozen candidate pools and sequential constrained Bayesian optimization. It does not claim cross-size generalization. The structure encoder is self-supervised; the online performance surrogate learns from actual candidate evaluations.

## Execution order

1. Canonical labeled trees and small target states.
2. Dense hierarchical SVD → TTN → reference isometry circuits → exact verification.
3. Candidate headroom diagnostics and simple search (stage A).
4. Few-observation predictive diagnostics and random-pretrained GNN (stage B).
5. Mixed pretraining (stage C).
6. Factorial candidate/pretraining ablation (stage D).
7. Independent tests, TCI/research compiler, hardware and size extensions (stage E).

TCI and reproduction of the paper's approximate compiler must not block reference-backend experiments. Every backend needs its own validation and conclusions; success under the reference compiler does not establish success under SCENT's approximate compiler.

## Upstream SCENT

Pin treetensorcomp revision f05c28fbf959ecc023fd4ed89d1e00691d5ca564:
https://gitlab.com/gohmat/treetensorcomp

The inspected code provides QuanticsFunction, IndexID, TTNGraph, Clusterer, TreeTensorNetwork and Interpolator. Circuit synthesis was not found in the inspected release. Keep upstream source isolated and retain its AGPL-3.0-only metadata. Resolve its SciPy requirements in a separate compatible environment if necessary; do not assume the packaging constraints reproduce the example environment.

Keep the preparation root distinct from the backend's movable gauge center. Some upstream constructors suppress a degree-two root. The adapter must preserve labeled partitions and restore the preparation root explicitly. Treat graph reorientation as gauge bookkeeping, not as a new search candidate. Any adapted SCENT implementation must be identified and its affinity metric documented.

## Architecture and records

Use `src/ttn_search/` for trees, targets, reference/SCENT/TCI backends, verification, features, search, datasets, models and experiment orchestration. Keep configurations in `configs/`, correctness checks in `tests/`, and generated artifacts in `artifacts/`.

Core contracts:

- TargetSpec: target parameters, normalized amplitudes/function, grid, physical bit order and hash.
- TreeSpec: rooted labeled binary topology, canonical serialization and hash.
- PipelineConfig: construction, precision, synthesis, routing, resource and seed policies.
- BuildResult: normalized TTN, local isometries, rank profile, index maps and reconstruction evidence.
- CompileResult: circuit, physical wire map, ancilla policy and synthesis diagnostics.
- EvaluationRecord: status, all three fidelities, CNOT/depth, timings, configuration/software hashes and artifacts.
- SearchResult: best feasible evaluation or explicit no-feasible-result status, complete trajectory.

All methods call the same evaluator. Cache keys include target, rooted tree, pipeline, software/code version and evaluation seeds. Persist records atomically and support resumption. Separate actual evaluation time from cache lookup time.

## Stage A: reference pipeline and non-neural search

### Development configuration

- Fixed n=8; smaller states for correctness tests.
- Hierarchical and overlapping-correlation targets; initially three instances per family.
- Dense hierarchical SVD construction and exact isometry synthesis.
- Exact statevector verification; all-to-all connections; single-qubit rotations plus CX.
- Primary final fidelity 0.999.
- Diagnostic pools: 32 unique trees, optionally 64, evaluated completely.
- Search pools: 256 unique trees, budgets 16/32/64 including eight common initial evaluations.
- Three search seeds initially; measure evaluation cost before expanding.

Generate normalized amplitudes proportional to sqrt(p(x)) for probability encoding, or explicitly mark function-amplitude encoding. Fix finite grid, variable/bit mapping and domain. Build positive-definite covariances from factor loadings plus diagonal noise. Do not expose hidden generating groups to the model.

### Trees and circuits

Canonicalize sibling order and internal numbering while retaining leaf labels and preparation root. Validate connected acyclic binary trees and complete unique leaf coverage. Generate SCENT, balanced, comb, random-merge and legal local-reconnection candidates; deduplicate before evaluation. Random merging is not uniform sampling over all trees.

For the dense reference, recursively construct nested subspaces and local isometries, normalize the root, and reconstruct independently. Embed non-power-of-two bond spaces in qubit registers, complete unused columns isometrically, and explicitly account for all wires. Verify that circuit output matches the TTN and original state in a fixed physical bit order. Count gates after decomposition, not from tensor size. A layout relabeling is allowed on all-to-all hardware only when the full circuit is consistently mapped to fixed logical outputs; routing requires separate accounting.

### Evaluation policy

An evaluation is C(target, tree, precision policy, compiler budget, seed protocol), not an intrinsic cost of a tree. Freeze tolerance attempts, iteration limits, restart counts, seeds, time/memory limits and output-selection rules. Return the lowest-CX feasible circuit found by this policy, with deterministic tie handling. Charge all construction/compilation attempts. Keep cost and fidelity associated with the same circuit.

The first deterministic reference compiler can use a single synthesis attempt per construction tolerance; declare this explicitly. Approximate synthesis retries are a later extension. Timeouts and numerical failures are distinct from measured infeasibility and never get invented fidelity or cost labels.

Record F_repr, F_comp and F_final separately. Local SVD/TCI tolerances are not global fidelity guarantees. For E=1-F0, start with global representation/compilation error budgets 0.1E and 0.4E. Verify actual global errors. The bound is e_final <= (sqrt(e_repr)+sqrt(e_comp))². A predeclared robustness allocation is 0.2E/0.2E; rerun representative searches, not only winning trees. Exact synthesis may render this compilation allocation inactive; report that limitation.

### Baselines and initialization

Compare single SCENT, random search, raw relationship features + BO and PCA features + identical BO. Relationship features are upper-triangular LCA depths/n and leaf depths/n. Fit PCA on declared structural training data; any unlabeled pool preprocessing must be identified as transductive and shared across methods.

Every main search uses the same pool and eight initial points including the original SCENT tree. SCENT consumes one evaluation. Report total and additional evaluations beyond SCENT, and count tree-discovery overhead. The single-SCENT baseline does not artificially consume the search budget.

Use separate cost and fidelity models. After a feasible incumbent exists, score unseen candidates by expected cost improvement times feasibility probability. Before that, use feasibility and uncertainty exploration. Define fallback proposals when GP fitting lacks sufficient observations. Never encode operational failures as fidelity measurements. All attempted candidates consume budget; duplicates are excluded. Use independent deterministic seed streams.

Run a single trajectory to budget 64 and extract 16/32/64 checkpoints. A 64-tree fully evaluated pool only diagnoses headroom and discovery speed; its terminal optimum cannot distinguish methods. Frozen 256-tree pools need not be fully evaluated.

### Stage A deliverables and gate

Deliver a tested evaluator, reproducible tree–fidelity–CX–time records, SCENT/random/raw/PCA comparisons, and a report of headroom and feasibility. Initial smoke runs are engineering validation, not statistical evidence. Continue only if tree-dependent cost differences are credible and limited-budget search has useful room to improve.

## Stage B: test the representation hypothesis

Start with 2,000 unique random structures at fixed n. Direction-aware GIN: hidden dimension 64, four message layers, latent dimension eight, root/global readout. Inputs are node type, physical label, relative depth and subtree size. Reconstruct LCA and leaf depths with equal-weight MSE. Do not include target tensors, fitted ranks, source labels or circuit quality in encoder inputs.

Using complete small-pool development data, fit raw/PCA/GNN surrogates on shared subsets of 8 or 16 observations. Evaluate low-cost feasible candidate retrieval, feasibility, recommendation regret and feasible-only ranking on held-out candidates. Examine constraint calibration where data allow it. Then compare real BO trajectories. Reconstruction accuracy alone is insufficient.

Test internal-renumbering/sibling invariance, physical-label sensitivity, embedding collapse and long-comb behavior. If needed, replace shallow message passing with full upward/downward tree traversal. Do not expand to 10,000 trees without useful downstream evidence.

## Stage C: mixed structural pretraining (revised idea §6.3)

Target unique proportions are random/SCENT/SCENT-perturbation = 70/10/20. Pilot counts are 1,400/200/400; full counts 7,000/1,000/2,000. These are post-deduplication targets, not replication quotas. Bound generation effort; fill unresolved deficits with random structures and report actual proportions.

SCENT parents come from an isolated pretraining target library. Run only necessary affinity/tree discovery, not subsequent TCI or circuit evaluation. Account separately for expensive prerequisites. Descendants use 1–3 legal reconnections and a per-parent cap, preserve labels, and require no target refitting. SCENT provenance is not a quality label.

Split targets and related parameter variants before generating trees. Parents and all descendants share a lineage/split. Random ancestors and descendants follow the same rule. Globally eliminate cross-split canonical duplicates without breaking lineage assignments. Aim for 80/10/10 target-group splits; report actual tree counts. Use a common structural validation set excluded from both random-only and mixed training corpora.

Final state-preparation test targets never supply pretraining structures. Online SCENT on a test target is allowed but cannot feed back into training or tuning. Accidental topology overlap is not by itself test-target leakage; an unseen-topology claim requires its own hash-disjoint protocol.

Store tree hash/serialization, n, source, generator/seed, target/group IDs, parent hash, lineage, perturbation count, split, generation-run ID and SCENT-config hash. Record shared parent costs once in a generation-run table. Keep provenance for discarded duplicate attempts. Equal sample weights by default; source metadata stays outside the encoder. Match unique training counts, model, optimizer and update counts between corpora.

## Stage D: factorial ablation

Compare random-only versus mixed pretraining crossed with random-only versus mixed online pools. Within a pool condition share candidates and initial observations between encoders. Across pool conditions fix pool size and a replacement policy with a shared random component. The random-only pool condition does not force SCENT initialization. Count online SCENT time. Retain raw/PCA and random-search controls. Tune mixture only on development targets with a small predeclared comparison set.

## Stage E: independent validation and extensions

Freeze protocol before final testing. Expand to chain, hierarchical, overlapping and non-Gaussian target families, initially up to ten independent targets per family when resources allow. Aggregate search seeds within each target before paired bootstrap across targets. Final compiler validation gives every method the same independent seeds/opportunities; preserve original search observations and report all validation outcomes.

Add TCI, the paper's approximate compiler, n=12, hardware routing, dynamic pools and MPS comparisons only as separately validated extensions. Hardware runs count routing and SWAP decomposition. Larger systems require tractable exact verification or clearly labeled bounds/estimates; two compact TNs do not automatically have a cheap overlap contraction.

Report best feasible CX versus evaluations and actual time, success rates, paired differences, depth/fidelity tradeoffs and total costs. Include data generation, pretraining, affinity, proposal, evaluation/retries and final verification. Cached replay is a development tool, not evidence of online wall-clock speedup. Keep failed runs visible. Compute amortization only when measured savings justify it.

## Stop/redirect decisions

| Observation | Decision |
|---|---|
| Trees yield almost identical costs | Inspect targets, truncation and compiler; pause training |
| Random search already finds good trees quickly | Measure whether model overhead is justified; do not cherry-pick hard cases |
| Feature BO helps but GNN does not | Inspect representation/training objective before expanding data |
| Random pretraining helps, mixture does not | Keep random training and report the mixture ablation |
| Reference success does not transfer to research compiler | Explain backend dependence and narrow claims |
| Scaling fails | Separate representation, feasibility, runtime and compiler limitations |

## Validation priorities

Test canonical identity and leaf coverage; dense reconstruction and isometry direction; complex phases and non-power-of-two ranks; compiler bit order and padding; all three fidelity definitions; cache separation; failure/budget accounting; common initialization; lineage isolation and unique-source proportions. Avoid redundant tests of mere configuration values.

## Work estimate and status

A first research cycle may take 4–6 weeks if a usable compiler is available; reproducing approximate synthesis is separately estimated after its audit. The immediate deliverable is trustworthy small-system data and evidence of search headroom, followed by GNN and mixture validation. Execution status and measured results belong in artifact reports and README, not as assumed outcomes in this plan.
