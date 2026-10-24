"""Instrumented reference construction; no circuit labels or optimization."""
from typing import Any
import time
import numpy as np
from ttn_search.reference import Node,Representation,fidelity,nbits
from ttn_search.trees import canonical,validate,features

FEATURE_SETS=[['log_padded'],['log_padded','cut_affinity_mean','cut_affinity_max'],
    ['log_padded','cut_affinity_mean','cut_affinity_max','log_parameters','rank_mean','rank_max',
     'width_mean','width_max','cut_entropy_mean','cut_entropy_max','discarded_sum']]


def profile_build(state,tree,tolerance) -> tuple[Representation,dict[str,Any]]:
    started=time.perf_counter();timing={k:0. for k in ['setup','projection','svd','assembly','reconstruction','isometry_check','verification','descriptors']}
    before=time.perf_counter();state=np.asarray(state,dtype=complex);n=int(np.log2(len(state)))
    tree=canonical(tree);validate(tree,n);state=state/np.linalg.norm(state);tensor=state.reshape([2]*n,order='F')
    timing['setup']+=time.perf_counter()-before;nodes=[];spectra=[]
    def visit(t,root=False):
        if isinstance(t,int):node=Node(t,[t],np.eye(2,dtype=complex),np.eye(2,dtype=complex),None)
        else:
            a,b=visit(t[0]),visit(t[1]);labels=a.labels+b.labels
            before=time.perf_counter();other=[q for q in range(n) if q not in labels]
            unfolding=tensor.transpose(labels+other).reshape(2**len(labels),-1)
            product=np.kron(a.basis,b.basis);reduced=product.conj().T@unfolding
            timing['projection']+=time.perf_counter()-before
            if root:coeff=reduced/np.linalg.norm(reduced)
            else:
                before=time.perf_counter();u,s,_=np.linalg.svd(reduced,full_matrices=False,compute_uv=True)
                timing['svd']+=time.perf_counter()-before
                numerical_rank=max(1,int(np.count_nonzero(s>max(s[0]*1e-12,1e-15))))
                rank=numerical_rank
                for k in range(1,numerical_rank+1):
                    if np.sum(s[k:]**2)<=tolerance:rank=k;break
                coeff=u[:,:rank]
                spectra.append(dict(labels=labels,singular_values=s.tolist(),rank=rank,discarded_weight=float(np.sum(s[rank:]**2))))
            before=time.perf_counter();node=Node(t,labels,product@coeff,coeff,(a,b));timing['assembly']+=time.perf_counter()-before
        nodes.append(node);return node
    root=visit(tree,True)
    def reconstruct(node):
        if node.children is None:return node.local
        return np.kron(*(reconstruct(c) for c in node.children))@node.local
    before=time.perf_counter();dense=reconstruct(root).reshape([2]*n)
    ordered=dense.transpose(np.argsort(root.labels)).flatten(order='F');timing['reconstruction']+=time.perf_counter()-before
    before=time.perf_counter();error=max(float(np.linalg.norm(x.local.conj().T@x.local-np.eye(x.local.shape[1]))) for x in nodes)
    timing['isometry_check']+=time.perf_counter()-before
    rep=Representation(root,ordered,[x.rank for x in nodes if x is not root],sum(x.local.size for x in nodes),error)
    before=time.perf_counter();fr=fidelity(state,ordered);timing['verification']+=time.perf_counter()-before
    total=time.perf_counter()-started;timing['unclassified']=max(0.,total-sum(timing.values()));timing['total']=total
    return rep,dict(timing=timing,construction_spectra=spectra,representation_fidelity=fr)


def describe(state,rep,affinity,diagnostic) -> dict[str,Any]:
    start=time.perf_counter();n=int(np.log2(len(state)));cuts=[];widths=[];padded=0;local=[]
    tensor=state.reshape([2]*n,order='F');spectrum_s=0.;affinity_s=0.
    def visit(node,root=False):
        nonlocal padded,spectrum_s,affinity_s
        if not root:
            other=[q for q in range(n) if q not in node.labels]
            before=time.perf_counter()
            matrix=tensor.transpose(node.labels+other).reshape(2**len(node.labels),-1)
            s=np.linalg.svd(matrix,compute_uv=False);p=s*s;p=p[p>1e-15];p=p/p.sum()
            ent=float(-np.sum(p*np.log2(p)));spectrum_s+=time.perf_counter()-before
            before=time.perf_counter();weight=float(affinity[np.ix_(node.labels,other)].sum());affinity_s+=time.perf_counter()-before
            cuts.append(dict(labels=node.labels,schmidt_values=s.tolist(),entropy=ent,
                affinity=weight))
        if node.children is not None:
            output=sum(nbits(c.rank) for c in node.children);input_width=nbits(node.rank)
            entries=2**(output+input_width);padded+=entries;widths.append(output)
            local.append(dict(labels=node.labels,rank=node.rank,input_width=input_width,output_width=output,padded_entries=entries))
            for c in node.children:visit(c)
    visit(rep.root,True)
    entropies=[c['entropy'] for c in cuts];weights=[c['affinity'] for c in cuts]
    vector=dict(log_padded=float(np.log1p(padded)),cut_affinity_mean=float(np.mean(weights)),cut_affinity_max=max(weights),
        log_parameters=float(np.log1p(rep.parameter_count)),rank_mean=float(np.mean(rep.ranks)),rank_max=max(rep.ranks),
        width_mean=float(np.mean(widths)),width_max=max(widths),cut_entropy_mean=float(np.mean(entropies)),cut_entropy_max=max(entropies),
        discarded_sum=sum(r['discarded_weight'] for r in diagnostic['construction_spectra']))
    before=time.perf_counter();structural=features(rep.root.tree).tolist();structural_s=time.perf_counter()-before
    return dict(features=vector,structural_features=structural,cuts=cuts,local_blocks=local,
        spectrum_s=spectrum_s,affinity_aggregation_s=affinity_s,structural_s=structural_s,
        padded=padded,parameters=rep.parameter_count,descriptor_s=time.perf_counter()-start)
