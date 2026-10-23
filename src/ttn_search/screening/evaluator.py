"""Reuse an initial TTN while preserving the reference evaluator's retry policy.

This evaluator deliberately has no result cache. Every call synthesizes and
verifies its own circuits; reuse is limited to the explicitly supplied TTN.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import time
import traceback
from typing import Any
import numpy as np
from qiskit import qpy
from qiskit.exceptions import QiskitError
from ttn_search.evaluate import Pipeline,digest,write_json
from ttn_search.reference import Representation,build,compile_reference,circuit_state,fidelity,nbits
from ttn_search.trees import tree_hash,serialize


@dataclass
class Prepared:
    identity: str
    representation: Representation | None
    attempt: dict[str, Any]


def identity(target,tree,pipeline):
    return digest(dict(target=target.hash,tree=tree_hash(tree),pipeline=asdict(pipeline)))


def prepare(target,tree,pipeline: Pipeline,factor=1.) -> Prepared:
    """Construct and check representation fidelity; do not synthesize gates."""
    attempt: dict[str,Any]=dict(tolerance_factor=factor,time_build_s=0.,time_repr_verify_s=0.,
        time_compile_s=0.,time_verify_s=0.,time_io_s=0.,compile_attempted=False)
    rep=None;phase='build';start=time.perf_counter()
    try:
        rep=build(target.state,tree,(1-pipeline.fidelity)*pipeline.repr_fraction*factor/(target.metadata['n']-1))
        attempt['time_build_s']=time.perf_counter()-start
        phase='repr_verify';start=time.perf_counter()
        fr=fidelity(target.state,rep.state)
        attempt.update(fidelity_repr=fr,max_bond_dimension=max(rep.ranks),parameter_count=rep.parameter_count,
            isometry_error=rep.isometry_error,time_repr_verify_s=time.perf_counter()-start,
            status='ready' if 1-fr<=(1-pipeline.fidelity)*pipeline.repr_fraction+1e-12 else 'representation_infeasible')
    except (QiskitError,ValueError,np.linalg.LinAlgError) as exc:
        attempt[f'time_{phase}_s']=time.perf_counter()-start
        attempt.update(status='numerical_error',error=str(exc),error_type=type(exc).__name__,
            failed_phase=phase,traceback=traceback.format_exc())
    return Prepared(identity(target,tree,pipeline),rep,attempt)


def scores(prepared: Prepared):
    """Parameter count and padded matrix-entry count, with no circuit labels."""
    rep=prepared.representation
    if rep is None:return float('inf'),float('inf')
    padded=0
    def visit(node):
        nonlocal padded
        if node.children is None:return
        output_bits=sum(nbits(child.rank) for child in node.children)
        padded+=2**(output_bits+nbits(node.rank))
        for child in node.children:visit(child)
    visit(rep.root)
    return rep.parameter_count,padded


def evaluate_fresh(target,tree,pipeline: Pipeline,directory,prepared: Prepared | None=None):
    """Evaluate once with fixed retries; caller pays for a supplied initial TTN.

Initial construction timings remain in the attempt for provenance. The reused
flag prevents charging those timings a second time in incremental accounting.
"""
    started=time.perf_counter();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    key=identity(target,tree,pipeline)
    if prepared is not None:
        if prepared.identity!=key or prepared.attempt['tolerance_factor']!=pipeline.tolerance_factors[0]:
            raise ValueError('Prepared TTN does not match target/tree/pipeline/initial tolerance')
    attempts=[];best=None
    for j,factor in enumerate(pipeline.tolerance_factors):
        before=time.perf_counter()
        reused=prepared is not None and j==0
        item=prepared if reused else prepare(target,tree,pipeline,factor)
        assert item is not None
        rep=item.representation;attempt=dict(item.attempt,reused_initial=reused)
        if attempt['status']=='ready':
            assert rep is not None
            phase='compile';start=time.perf_counter();attempt['compile_attempted']=True
            try:
                circuit=compile_reference(rep,pipeline.compiler_seed,pipeline.optimization_level)
                attempt['time_compile_s']=time.perf_counter()-start
                phase='verify';start=time.perf_counter()
                output=circuit_state(circuit)
                fc,ff=fidelity(rep.state,output),fidelity(target.state,output)
                attempt.update(fidelity_comp=fc,fidelity_final=ff,cx_count=int(circuit.count_ops().get('cx',0)),
                    depth_total=circuit.depth(),time_verify_s=time.perf_counter()-start,
                    status='success' if ff>=pipeline.fidelity and 1-fc<=(1-pipeline.fidelity)*pipeline.comp_fraction+1e-12 else 'infeasible')
                io=time.perf_counter()
                path=directory/f'attempt_{j}.qpy'
                with path.open('wb') as f:qpy.dump(circuit,f)
                np.savez_compressed(directory/f'attempt_{j}.npz',target=target.state,representation=rep.state,circuit_state=output)
                attempt.update(circuit_path=str(path.resolve()),time_io_s=time.perf_counter()-io)
            except (QiskitError,ValueError,np.linalg.LinAlgError) as exc:
                attempt[f'time_{phase}_s']=time.perf_counter()-start
                attempt.update(status='compiler_error' if isinstance(exc,QiskitError) and phase=='compile' else 'numerical_error',
                    error=str(exc),error_type=type(exc).__name__,failed_phase=phase,traceback=traceback.format_exc())
        attempt['incremental_elapsed_s']=time.perf_counter()-before
        attempts.append(attempt)
        if attempt['status']=='success':best=attempt;break
    if best is None:
        measured=[a for a in attempts if 'fidelity_final' in a]
        best=max(measured,key=lambda a:a['fidelity_final']) if measured else attempts[-1]
    result=dict(**best,tree_hash=tree_hash(tree),tree_serialization=serialize(tree),target_hash=target.hash,
        pipeline=asdict(pipeline),attempts=attempts,cache_hit=False,initial_reused=prepared is not None,
        compiler_calls=sum(a['compile_attempted'] for a in attempts),
        incremental_build_s=sum(a['time_build_s'] for a in attempts if not a['reused_initial']),
        incremental_repr_verify_s=sum(a['time_repr_verify_s'] for a in attempts if not a['reused_initial']),
        compile_s=sum(a['time_compile_s'] for a in attempts),verify_s=sum(a['time_verify_s'] for a in attempts),
        retry_s=sum(a['incremental_elapsed_s'] for a in attempts[1:]),
        io_s=sum(a['time_io_s'] for a in attempts),evaluation_wall_s=time.perf_counter()-started)
    write_json(directory/'record.json',result)
    return result
