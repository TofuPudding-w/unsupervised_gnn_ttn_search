"""Frozen configuration and construction-only descriptors for the decision pilot."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ttn_search.evaluate import write_json, software, source_hash
from ttn_search.reference import build, nbits, fidelity
from ttn_search.trees import canonical

OUT=Path('artifacts/decision_experiment')
STAGE_B=Path('artifacts/stage_b')
NEIGHBOR=Path('artifacts/neighborhood_validation')
CONFIG=Path('configs/decision_experiment.json')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(component):
    config=json.loads(CONFIG.read_text())
    paths=[Path('scripts')/f'decision_{s}.py' for s in ['common','cost','gnn']]
    protocol=dict(config=config,software=software(),current_evaluator_source_hash=source_hash(),
        script_hashes={str(p):file_hash(p) for p in paths},
        corpus_hash=file_hash(STAGE_B/'corpus.json'),
        old_stage_b_protocol_hash=file_hash(STAGE_B/'protocol.json'),
        neighborhood_protocol_hash=file_hash(NEIGHBOR/'protocol.json'),
        rules=['Stage C scaling and mixed pretraining on hold',
            'Descriptors come from the first nominal-tolerance TTN build; no final retry ranks or compiled gate counts as inputs',
            'Primary cohorts: complete Stage B pools and SCENT plus unique global-random samples on ten new targets',
            'Adaptive neighborhood union is extracted for completeness but excluded from primary prediction metrics',
            'GP scalers for descriptor/raw comparisons fit only observed training indices within each split',
            'Fixed screening scores require no circuit labels, but construct all held-out candidates; charge this work',
            'GNN restarts same seeds to 300 epochs because original checkpoints omit optimizer state; verify first 100 epoch histories',
            'GNN uses exact original corpus, pools, scalers, initialization, GP and validation-only selection',
            'Existing development targets and post-result mechanism case; no independent confirmation claim'])
    directory=OUT/component;directory.mkdir(parents=True,exist_ok=True)
    path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError('Changed frozen protocol; use a new output directory')
    write_json(path,protocol)
    return config,directory


def nodes(rep):
    result=[]
    def visit(node):
        if node.children is not None:
            result.append(node)
            for child in node.children:visit(child)
    visit(rep.root)
    return result


def descriptors(rep):
    internal=nodes(rep)
    inp=[nbits(node.rank) for node in internal]
    out=[sum(nbits(c.rank) for c in node.children) for node in internal]
    entries=[2**(i+o) for i,o in zip(inp,out)]
    return [max(rep.ranks),sum(rep.ranks),rep.parameter_count,sum(inp),sum(out),max(out),sum(entries),max(entries)]


def construct(target,tree,pipeline,multiplier=1.):
    before=time.perf_counter()
    n=target.metadata['n']
    tolerance=(1-pipeline['fidelity'])*pipeline['repr_fraction']*multiplier/(n-1)
    rep=build(target.state,canonical(tree),tolerance)
    x=descriptors(rep)
    elapsed=time.perf_counter()-before
    return rep,dict(features=x,construction_s=elapsed,local_tolerance=tolerance,
                    representation_fidelity=fidelity(target.state,rep.state))
