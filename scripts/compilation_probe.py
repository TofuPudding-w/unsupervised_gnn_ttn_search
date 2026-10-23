"""Frozen development tolerance and compiler experiment on three existing TTNs."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from scipy.linalg import null_space
from qiskit import QuantumCircuit,transpile,qpy
from qiskit.circuit.library import Isometry
from qiskit.synthesis import qs_decomposition
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import write_json,source_hash,software
from ttn_search.reference import build,padded_isometry,nbits,compile_reference,circuit_state,fidelity
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/compilation_probe')
CONFIG=dict(n=10,family='overlapping',seed=13,candidate_ids=[0,74,1],
    tolerance_multipliers=[1,3,10,30,100],alternative_multipliers=[1,10],
    final_fidelity=.999,local_base_tolerance=.0001/9,compiler_seed=17,optimization_level=1,timeout_s=120)

def compile_blocks(rep,backend):
    n=len(rep.root.labels);circuit=QuantumCircuit(n);free=list(range(n));mapping={};blocks=[];owners=[]
    def emit(node,incoming):
        if node.children is None:mapping[incoming[0]]=node.tree;return
        a,b=node.children;total=nbits(a.rank)+nbits(b.rank);need=total-len(incoming)
        assert 0<=need<=len(free)
        wires=incoming+free[:need];del free[:need]
        mat=padded_isometry(node);local=QuantumCircuit(total);start=time.perf_counter()
        if total:
            if backend=='unitary_qsd':
                if mat.shape[1]<mat.shape[0]:mat=np.column_stack((mat,null_space(mat.conj().T)))
                local=qs_decomposition(mat)
            else:local.append(Isometry(mat,0,0),range(total))
            local=transpile(local,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
        else:local.global_phase=float(np.angle(mat[0,0]))
        index=len(blocks)
        blocks.append(dict(leaves=sorted(node.labels),rank=node.rank,child_ranks=[a.rank,b.rank],
            input_bits=nbits(node.rank),output_bits=total,padded_entries=2**(nbits(node.rank)+total),
            cx=int(local.count_ops().get('cx',0)),d2=two_qubit_depth(local),compile_s=time.perf_counter()-start))
        circuit.compose(local,qubits=wires,inplace=True);owners.extend([index]*len(local.data))
        emit(a,wires[nbits(b.rank):]);emit(b,wires[:nbits(b.rank)])
    emit(rep.root,[])
    logical=QuantumCircuit(n);logical.global_phase=circuit.global_phase
    for inst in circuit.data:logical.append(inst.operation,[mapping[circuit.find_bit(q).index] for q in inst.qubits])
    # One critical CX chain in the independently synthesized block assembly.
    last=[None]*n;lengths=[];parents=[];gateowners=[]
    for inst,owner in zip(circuit.data,owners):
        if inst.operation.num_qubits!=2:continue
        qs=[circuit.find_bit(q).index for q in inst.qubits]
        predecessors=[last[q] for q in qs if last[q] is not None]
        parent=max(predecessors,key=lambda i:lengths[i]) if predecessors else None
        lengths.append(1+(lengths[parent] if parent is not None else 0));parents.append(parent);gateowners.append(owner)
        for q in qs:last[q]=len(lengths)-1
    contribution=[0]*len(blocks)
    cursor=max(range(len(lengths)),key=lambda i:lengths[i]) if lengths else None
    while cursor is not None:contribution[gateowners[cursor]]+=1;cursor=parents[cursor]
    for row,value in zip(blocks,contribution):row['critical_chain_cx']=value
    return logical,blocks

def worker(tree,candidate,factor,backend,directory):
    try:
        with threadpool_limits(limits=1):
            target=gaussian_target(10,'overlapping',13);start=time.perf_counter()
            rep=build(target.state,canonical(tree),CONFIG['local_base_tolerance']*factor)
            build_s=time.perf_counter()-start;fr=fidelity(target.state,rep.state)
            write_json(directory/'started.json',dict(phase='compilation',representation_fidelity=fr))
            start=time.perf_counter()
            assembly,blocks=compile_blocks(rep,backend)
            assembled_depth=two_qubit_depth(assembly)
            if backend=='reference':circuit=compile_reference(rep,17,1)
            else:circuit=transpile(assembly,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
            compile_and_analysis_s=time.perf_counter()-start
            start=time.perf_counter();output=circuit_state(circuit);ff=fidelity(target.state,output);fc=fidelity(rep.state,output)
            assembly_fidelity=fidelity(rep.state,circuit_state(assembly))
            assert assembly_fidelity>.999999,'Block allocation/compilation mismatch'
            result=dict(candidate=candidate,factor=factor,backend=backend,status='success',
                fidelity_repr=fr,fidelity_final=ff,fidelity_comp=fc,feasible=ff>=.999,
                original_budget_feasible=(ff>=.999 and 1-fr<=.0001+1e-12 and 1-fc<=.0004+1e-12),
                ranks=rep.ranks,parameters=rep.parameter_count,padded_entries=sum(b['padded_entries'] for b in blocks),
                cx=int(circuit.count_ops().get('cx',0)),d2=two_qubit_depth(circuit),total_depth=circuit.depth(),
                assembly_d2=assembled_depth,assembly_fidelity=assembly_fidelity,
                sum_block_depths=sum(b['d2'] for b in blocks),blocks=blocks,build_s=build_s,
                compile_and_analysis_s=compile_and_analysis_s,verification_s=time.perf_counter()-start)
            with (directory/'circuit.qpy').open('wb') as f:qpy.dump(circuit,f)
            write_json(directory/'result.json',result)
    except Exception as exc:write_json(directory/'result.json',dict(candidate=candidate,factor=factor,backend=backend,status='error',error=repr(exc)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    poolpath=Path('artifacts/stage_c1/overlapping/pool.json');pool=json.loads(poolpath.read_text())
    protocol=dict(config=CONFIG,core_hash=source_hash(),software=software(),
        runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pool_hash=hashlib.sha256(poolpath.read_bytes()).hexdigest(),
        policy='Development sweep: final F>=0.999 determines acceptance; original sub-budgets reported separately. No retries. QSD uses full unitary completion of each padded isometry. Trees: SCENT, original padded-first, fixed random pool candidate 1. Critical-chain attribution belongs to independently compiled block assembly, not globally optimized reference circuit.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    write_json(path,protocol);rows=[]
    for candidate in CONFIG['candidate_ids']:
        for backend in ['reference','unitary_qsd']:
            factors=CONFIG['tolerance_multipliers'] if backend=='reference' else CONFIG['alternative_multipliers']
            for factor in factors:
                directory=OUT/f'{candidate}_{backend}_{factor}';directory.mkdir(exist_ok=True)
                result=directory/'result.json'
                if not result.exists():
                    process=mp.Process(target=worker,args=(pool['trees'][candidate],candidate,factor,backend,directory));process.start();process.join(CONFIG['timeout_s'])
                    if process.is_alive():process.terminate();process.join();write_json(result,dict(candidate=candidate,factor=factor,backend=backend,status='timeout'))
                    elif not result.exists():write_json(result,dict(candidate=candidate,factor=factor,backend=backend,status='worker_error',exitcode=process.exitcode))
                row=json.loads(result.read_text());rows.append(row);write_json(OUT/'results.json',rows)
                print(candidate,backend,factor,row['status'],row.get('cx'),row.get('d2'),row.get('fidelity_final'),flush=True)
    write_json(OUT/'completion.json',dict(evaluations=len(rows)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
