"""Versioned verified compiler evaluator. Frozen evaluators remain unchanged.

Preparation identities describe TTNs only. Outcome identities additionally bind
compiler version, compiler bytes and evaluator bytes. Every call is fresh.
"""
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.exceptions import QiskitError
from ttn_search.compilers import verified
from ttn_search.compilers.verified import compile_verified,COMPILER_VERSION
from ttn_search.evaluate import Pipeline,digest,write_json
from ttn_search.reference import circuit_state,fidelity
from ttn_search.trees import tree_hash,serialize
from .evaluator import Prepared,identity,prepare,scores

COMPILER_HASH=hashlib.sha256(Path(verified.__file__).read_bytes()).hexdigest()
EVALUATOR_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def evaluate_verified(target,tree,pipeline: Pipeline,directory,prepared: Prepared | None=None):
    """Evaluate once with fixed retries; caller pays for a supplied initial TTN.

Initial construction timings remain in the attempt for provenance. The reused
flag prevents charging those timings a second time in incremental accounting.
"""
    started=time.perf_counter();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'record.json').exists() or (directory/'synthesis_starts.jsonl').exists():
        raise FileExistsError('Refusing to overwrite an existing evaluation; use a fresh directory')
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
                attempt['compiler_stats']={}
                with (directory/'synthesis_starts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=j,tolerance_factor=factor,
                        compiler_version=COMPILER_VERSION,utc=time.time(),monotonic=time.monotonic()))+'\n')
                    stream.flush();os.fsync(stream.fileno())
                circuit=compile_verified(rep,pipeline.compiler_seed,pipeline.optimization_level,
                    stats=attempt['compiler_stats'])
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
        compiler_version=COMPILER_VERSION,compiler_hash=COMPILER_HASH,evaluator_hash=EVALUATOR_HASH,
        evaluation_identity=digest(dict(representation_identity=key,compiler_version=COMPILER_VERSION,
            compiler_hash=COMPILER_HASH,evaluator_hash=EVALUATOR_HASH)),
        pipeline=asdict(pipeline),attempts=attempts,cache_hit=False,initial_reused=prepared is not None,
        compiler_calls=sum(a['compile_attempted'] for a in attempts),
        incremental_build_s=sum(a['time_build_s'] for a in attempts if not a['reused_initial']),
        incremental_repr_verify_s=sum(a['time_repr_verify_s'] for a in attempts if not a['reused_initial']),
        compile_s=sum(a['time_compile_s'] for a in attempts),verify_s=sum(a['time_verify_s'] for a in attempts),
        retry_s=sum(a['incremental_elapsed_s'] for a in attempts[1:]),
        io_s=sum(a['time_io_s'] for a in attempts),evaluation_wall_s=time.perf_counter()-started)
    write_json(directory/'record.json',result)
    return result
