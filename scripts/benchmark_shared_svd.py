"""Bounded cold-cache construction benchmark; no compilation or model training."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ttn_search.reference import build,fidelity
from ttn_search.screening.shared_svd import SharedSVD
from ttn_search.screening.evaluator import Prepared,scores
from ttn_search.scent import scent_tree
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree,tree_hash
from ttn_search.evaluate import write_json,software

OUT=Path('artifacts/shared_svd_feasibility')
CONFIG=dict(sizes=[8,10,12],families=['hierarchical','overlapping'],target_seed=19,pool_size=128,repeats=3,
    pool_seed=20261101,local_budget_numerator=.0003)


def one_pass(state,pool,tolerance,shared):
    started=time.perf_counter();cache=SharedSVD(state,tolerance) if shared else None
    rows=[];states=[]
    for tree in pool:
        rep=cache.build(tree) if cache is not None else build(state,tree,tolerance)
        fr=fidelity(state,rep.state);a,b=scores(Prepared('',rep,{}))
        rows.append(dict(ranks=rep.ranks,parameters=a,padded=b,repr_fidelity=fr,
            representation_feasible=1-fr<=.0001+1e-12,isometry_error=rep.isometry_error))
        states.append(rep.state)
    order=np.argsort([r['padded'] for r in rows],kind='stable').tolist()
    elapsed=time.perf_counter()-started
    return dict(elapsed_s=elapsed,svd_calls=cache.svd_calls if cache is not None else len(pool)*(int(np.log2(len(state)))-2),
        cache_hits=cache.hits if cache is not None else 0,retained_cache_array_bytes=cache.retained_array_bytes if cache is not None else 0,
        rows=rows,order=order),states


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[__file__,'src/ttn_search/screening/shared_svd.py','src/ttn_search/reference.py','src/ttn_search/targets.py','src/ttn_search/scent.py','src/ttn_search/trees.py']
    protocol=dict(config=CONFIG,software=software(),files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        rules=['One new instance per family/size; repetitions measure timing, not independent targets',
            '128 fixed candidates: SCENT plus unique random pair-merge trees; no outcome filtering',
            'Both methods build every candidate, reconstruct states, check representation fidelity, compute scores and rank',
            'Each shared pass starts a fresh target/tolerance cache; no reuse across repetitions or targets',
            'Execution order alternates; single-thread BLAS; state equality checks run outside both timers',
            'Same 3x first-attempt tolerance as frozen policy; representation-infeasible candidates remain in data',
            'No synthesis, final circuit fidelity, online speedup or new target search-quality claim'])
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    write_json(path,protocol);results=[]
    for n in CONFIG['sizes']:
        for family in CONFIG['families']:
            target=gaussian_target(n,family,CONFIG['target_seed']);started=time.perf_counter()
            scent,_=scent_tree(target.state,0);pool=[scent];seen={tree_hash(scent)}
            rng=np.random.default_rng(CONFIG['pool_seed']+n)
            while len(pool)<128:
                tree=random_tree(n,rng);key=tree_hash(tree)
                if key not in seen:pool.append(tree);seen.add(key)
            generation_s=time.perf_counter()-started
            write_json(OUT/f'{n}_{family}_pool.json',dict(trees=pool,target_hash=target.hash,generation_s=generation_s))
            for repeat in range(CONFIG['repeats']):
                path=OUT/f'{n}_{family}_{repeat}.json'
                if path.exists():row=json.loads(path.read_text())
                else:
                    passes={};outputs={}
                    order=['independent','shared'] if (repeat+n//2+(family=='overlapping'))%2==0 else ['shared','independent']
                    for method in order:
                        passes[method],outputs[method]=one_pass(target.state,pool,.0003/(n-1),method=='shared')
                    a,b=passes['independent'],passes['shared']
                    assert a['order']==b['order']
                    for x,y in zip(a['rows'],b['rows']):
                        for key in ['ranks','parameters','padded','representation_feasible']:assert x[key]==y[key],(n,family,key)
                    max_difference=max(float(np.max(np.abs(x-y))) for x,y in zip(outputs['independent'],outputs['shared']))
                    min_agreement=min(fidelity(x,y) for x,y in zip(outputs['independent'],outputs['shared']))
                    assert max_difference<1e-10 and min_agreement>1-1e-10
                    row=dict(n=n,family=family,repeat=repeat,execution_order=order,passes=passes,
                        max_state_difference=max_difference,min_state_agreement=min_agreement)
                    write_json(path,row)
                results.append(row);write_json(OUT/'results.json',results)
                print(n,family,repeat,'independent',round(row['passes']['independent']['elapsed_s'],3),'shared',round(row['passes']['shared']['elapsed_s'],3),flush=True)
    lines=['# Exact subtree-reuse feasibility check','',
        'The completed repaired-compiler benchmark exhaustively evaluated six 128-tree pools. Padded screening selects the minimum CNOT and minimum D2 candidate first in all six. SCENT is optimal in five pools; only overlapping seed 15 improves (955→805 CNOT, 795→742 D2). The search endpoints are unchanged by compiler repair. There is no demonstrated CNOT-ranking headroom for a new GNN on those pools.', '',
        'This follow-up measures whether repeated TTN construction can be made cheaper without learning or approximate descriptors. Identical labelled subtrees reuse their SVD basis only within the same normalized target and tolerance. Different topologies on the same leaf set do not share a cache entry. Original builders and evaluators remain unchanged.', '',
        '## Frozen construction experiment','',
        'Eight, ten and twelve qubits; hierarchical and overlapping seed 19; one target per family/size; 128 candidates each (SCENT + random pair merges); three alternating-order timing repetitions. Every shared pass starts a cold cache. Both timings include construction, independent local-tensor reconstruction, representation verification, scores and ordering. Pool generation is common and recorded separately. Equivalence assertions run outside timing.', '',
        '| Qubits | Family | Independent median seconds | Shared median seconds | Ratio independent/shared | Seconds saved | SVD calls independent/shared | Retained cache arrays MiB |',
        '|---:|---|---:|---:|---:|---:|---|---:|']
    groups=[]
    for n in CONFIG['sizes']:
        for family in CONFIG['families']:
            rows=[r for r in results if r['n']==n and r['family']==family]
            a=float(np.median([r['passes']['independent']['elapsed_s'] for r in rows]));b=float(np.median([r['passes']['shared']['elapsed_s'] for r in rows]))
            sample=rows[0];memory=sample['passes']['shared']['retained_cache_array_bytes']/2**20
            groups.append(dict(n=n,family=family,independent_s=a,shared_s=b,saved_s=a-b,ratio=a/b))
            lines.append(f"| {n} | {family} | {a:.4f} | {b:.4f} | {a/b:.3f} | {a-b:.4f} | {sample['passes']['independent']['svd_calls']}/{sample['passes']['shared']['svd_calls']} | {memory:.2f} |")
    lines+=['','## Verification and interpretation','',
        f"All {len(results)*128} paired constructions match ranks, parameter counts, padded scores, representation-feasibility decisions and stable score order. Maximum amplitude difference: {max(r['max_state_difference'] for r in results):.3g}. This establishes construction equivalence on this cohort, not new circuit-fidelity results.", '',
        'Saved time is construction/screening time only. No circuits were compiled, no fallback schedules executed, and no end-to-end or fixed-time search speedup is claimed. Twelve-qubit results are construction feasibility only. Retained cache arrays exclude Python objects, returned states and temporary Kronecker products; this is not peak process memory. Caching retains arrays and remains a dense-state algorithm, so it does not establish large-system scalability.', '',
        'Three repetitions characterize timing variation for six specific targets; they do not replace independent targets. This candidate generator produces repeated small subtrees, so benefits need not transfer to other pools. Shared nodes must be treated as immutable by callers; tensor arrays are read-only.', '',
        '## Decision gate','',
        'Use these measured savings to decide whether exact construction reuse warrants integration. Do not claim a learned descriptor shortcut is needed before comparing with this non-learning baseline. A future integration must carry the same target/tolerance cache boundaries through retries and charge cache initialization and memory. Circuit comparison labels remain under the frozen repaired evaluator until an explicitly versioned integration is validated.', '']
    write_json(OUT/'summary.json',groups)
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_json(OUT/'completion.json',dict(targets=6,paired_passes=len(results),paired_candidate_builds=len(results)*128,compilation_calls=0))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
