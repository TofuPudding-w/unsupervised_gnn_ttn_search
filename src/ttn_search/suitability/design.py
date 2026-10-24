from typing import TypedDict
import numpy as np
from ttn_search.targets import Target,gaussian_target
from ttn_search.trees import Tree,balanced,comb,canonical,leaves,random_tree,neighbors,tree_hash


def apply_zz(state,pairs,gamma):
    indices=np.arange(len(state));result=state.copy()
    for i,j in pairs:
        zi=1-2*((indices>>i)&1);zj=1-2*((indices>>j)&1)
        result*=np.exp(-.5j*gamma*zi*zj)
    return result


def generated_target(seed,alpha,gamma):
    n=10;rng=np.random.default_rng(seed);permutation=rng.permutation(n).tolist()
    tree=balanced(permutation) if seed==101 else comb(permutation)
    def visit(t,root=False):
        if isinstance(t,int):return np.eye(2,dtype=complex),[t]
        a,la=visit(t[0]);b,lb=visit(t[1]);labels=la+lb
        rows=a.shape[1]*b.shape[1]
        rank=1 if root else min(4,2**min(len(labels),n-len(labels)))
        raw=rng.normal(size=(rows,rank))+1j*rng.normal(size=(rows,rank))
        if root:local=raw/np.linalg.norm(raw)
        else:
            q,r=np.linalg.qr(raw,mode='reduced');d=np.diag(r)
            q=q*(d/np.abs(d))
            weights=np.exp(-alpha*np.arange(rank));weights/=np.linalg.norm(weights)
            local=q*weights
        return np.kron(a,b)@local,labels
    raw,labels=visit(tree,True)
    state=raw.reshape([2]*n).transpose(np.argsort(labels)).flatten(order='F');state/=np.linalg.norm(state)
    assert not isinstance(tree,int)
    left=[q for q in permutation if q in leaves(tree[0])];right=[q for q in permutation if q in leaves(tree[1])]
    pairs=[(left[i%len(left)],right[i%len(right)]) for i in range(2)]
    state=apply_zz(state,pairs,gamma)
    metadata=dict(n=n,family='generated_ttn',seed=seed,alpha=alpha,gamma=gamma,permutation=permutation,
        generating_tree=tree,zz_pairs=pairs,group=f'ttn_{seed}',spectrum_control='local_factor_weights_not_global_Schmidt_spectra')
    return Target(state,metadata),tree


def cohort():
    result=[]
    for family in ['hierarchical','overlapping']:
        for seed in [21,22]:
            target=gaussian_target(10,family,seed)
            result.append((f'{family}_{seed}',f'{family}_{seed}',target,None))
    for seed in [101,102]:
        for alpha in [.3,1.2]:
            for gamma in [0.,.15]:
                target,tree=generated_target(seed,alpha,gamma)
                result.append((f'ttn_{seed}_a{alpha}_g{gamma}',f'ttn_{seed}',target,tree))
    return result


class CandidateSets(TypedDict):
    R: list[str]
    M: list[str]
    trees: dict[str,Tree]
    perturbation_distance: dict[str,int]


def candidate_sets(scent,seed) -> CandidateSets:
    rng=np.random.default_rng(seed);scent=canonical(scent)
    random=[scent];seen={tree_hash(scent)}
    while len(random)<64:
        tree=random_tree(10,rng);key=tree_hash(tree)
        if key not in seen:random.append(tree);seen.add(key)
    one={tree_hash(t):t for t in neighbors(scent)}
    two={tree_hash(t):t for u in one.values() for t in neighbors(u)}
    two={k:t for k,t in two.items() if k not in one and k!=tree_hash(scent)}
    excluded={tree_hash(t) for t in random[:32]}
    def shuffled(mapping):
        keys=sorted(k for k in mapping if k not in excluded)
        return [mapping[keys[int(i)]] for i in rng.permutation(len(keys))]
    first=shuffled(one)[:16];second=shuffled(two)[:32-len(first)]
    if len(first)+len(second)!=32:raise ValueError('Insufficient local perturbations; no random substitution')
    mixed=random[:32]+first+second
    union={tree_hash(t):t for t in random+mixed}
    return {'R':[tree_hash(t) for t in random],'M':[tree_hash(t) for t in mixed],'trees':union,
        'perturbation_distance':{tree_hash(t):1 for t in first}|{tree_hash(t):2 for t in second}}
