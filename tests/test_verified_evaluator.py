from dataclasses import asdict
import json
import numpy as np
import pytest
from qiskit import QuantumCircuit
from ttn_search.evaluate import Pipeline
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree
from ttn_search.screening import verified_evaluator as evaluator

POLICY=Pipeline(tolerance_factors=(3.,1.,.1,0.))


def test_actual_repaired_failure_is_versioned_and_verified(tmp_path):
    target=gaussian_target(10,'hierarchical',13)
    tree=random_tree(10,np.random.default_rng(2026))
    prepared=evaluator.prepare(target,tree,POLICY,3.)
    record=evaluator.evaluate_verified(target,tree,POLICY,tmp_path,prepared)
    assert record['status']=='success' and len(record['attempts'])==1
    assert record['compiler_version']==evaluator.COMPILER_VERSION
    assert record['compiler_hash']==evaluator.COMPILER_HASH
    assert record['evaluator_hash']==evaluator.EVALUATOR_HASH
    assert record['evaluation_identity']!=prepared.identity
    assert record['fidelity_comp']>1-1e-8
    assert record['compiler_stats']['diagonals_corrected']>=1
    assert record['initial_reused'] and record['incremental_build_s']==0
    starts=[json.loads(x) for x in (tmp_path/'synthesis_starts.jsonl').read_text().splitlines()]
    assert len(starts)==record['compiler_calls']==1
    assert starts[0]['tolerance_factor']==3
    assert json.loads((tmp_path/'record.json').read_text())==json.loads(json.dumps(record))
    with pytest.raises(FileExistsError):
        evaluator.evaluate_verified(target,tree,POLICY,tmp_path,prepared)


def test_failed_verification_is_charged_and_retried(tmp_path,monkeypatch):
    target=gaussian_target(4,'hierarchical',1)
    tree=random_tree(4,np.random.default_rng(1))
    original=evaluator.compile_verified
    calls=[]
    def fail_first(rep,*args,**kwargs):
        # Log must be persisted before compiler begins.
        calls.append(len((tmp_path/'synthesis_starts.jsonl').read_text().splitlines()))
        if len(calls)==1:
            c=QuantumCircuit(4);c.x(0)
            return c
        return original(rep,*args,**kwargs)
    monkeypatch.setattr(evaluator,'compile_verified',fail_first)
    record=evaluator.evaluate_verified(target,tree,POLICY,tmp_path)
    assert record['status']=='success' and record['tolerance_factor']==1
    assert calls==[1,2] and record['compiler_calls']==2
    assert record['attempts'][0]['status']=='infeasible'
    assert record['retry_s']>0 and record['compile_s']>0
    assert record['pipeline']==asdict(POLICY)
