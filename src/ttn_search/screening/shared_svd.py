"""Experimental exact subtree reuse for one fixed state and truncation tolerance.

This is a construction-cost diagnostic, not a replacement for the frozen
benchmark builder. It reuses only identical labelled subtree topologies, never
merely matching leaf sets. Returned tensor arrays are read-only.
"""
import numpy as np
from ttn_search.reference import Node,Representation
from ttn_search.trees import canonical,validate


class SharedSVD:
    def __init__(self,state,local_discarded_weight=0.):
        state=np.asarray(state,dtype=complex)
        if state.ndim!=1 or len(state)<2 or len(state)&(len(state)-1) or not np.isfinite(state).all() or np.linalg.norm(state)==0:
            raise ValueError('Expected finite nonzero power-of-two statevector')
        if not np.isfinite(local_discarded_weight) or local_discarded_weight<0:
            raise ValueError('Invalid truncation budget')
        self.n=(len(state)-1).bit_length()
        self._tensor=(state/np.linalg.norm(state)).reshape([2]*self.n,order='F')
        self._tolerance=local_discarded_weight
        self._nodes={}
        self.hits=0
        self.svd_calls=0

    def _visit(self,tree,root=False):
        key=(root,tree)
        if key in self._nodes:
            self.hits+=1
            return self._nodes[key]
        if isinstance(tree,int):
            node=Node(tree,[tree],np.eye(2,dtype=complex),np.eye(2,dtype=complex),None)
        else:
            a,b=(self._visit(t) for t in tree)
            labels=a.labels+b.labels
            other=[q for q in range(self.n) if q not in labels]
            unfolding=self._tensor.transpose(labels+other).reshape(2**len(labels),-1)
            product=np.kron(a.basis,b.basis)
            reduced=product.conj().T@unfolding
            if root:coeff=reduced/np.linalg.norm(reduced)
            else:
                u,s,_=np.linalg.svd(reduced,full_matrices=False,compute_uv=True)
                self.svd_calls+=1
                numerical_rank=max(1,int(np.count_nonzero(s>max(s[0]*1e-12,1e-15))))
                rank=numerical_rank
                for k in range(1,numerical_rank+1):
                    if np.sum(s[k:]**2)<=self._tolerance:
                        rank=k
                        break
                # Copy selected columns so a view cannot retain the full SVD U.
                coeff=u[:,:rank].copy()
            node=Node(tree,labels,product@coeff,coeff,(a,b))
        node.basis.flags.writeable=False
        node.local.flags.writeable=False
        self._nodes[key]=node
        return node

    def build(self,tree):
        tree=canonical(tree);validate(tree,self.n)
        root=self._visit(tree,True)
        nodes=[]
        def collect(node):
            if node.children is not None:
                for child in node.children:collect(child)
            nodes.append(node)
        collect(root)
        # Same independent local-tensor reconstruction as reference.build.
        def reconstruct(node):
            if node.children is None:return node.local
            return np.kron(*(reconstruct(c) for c in node.children))@node.local
        dense=reconstruct(root).reshape([2]*self.n)
        ordered=dense.transpose(np.argsort(root.labels)).flatten(order='F')
        error=max(float(np.linalg.norm(x.local.conj().T@x.local-np.eye(x.local.shape[1]))) for x in nodes)
        return Representation(root,ordered,[x.rank for x in nodes if x is not root],sum(x.local.size for x in nodes),error)

    @property
    def retained_array_bytes(self):
        """Retained numerical arrays, excluding outputs, temporaries and Python objects."""
        arrays={id(a):a for node in self._nodes.values() for a in [node.basis,node.local]}
        return self._tensor.nbytes+sum(a.nbytes for a in arrays.values())
