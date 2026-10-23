"""Shared frozen 3x-first policy, strict global budgets, persistent synthesis starts."""
import os
import json
import time
from pathlib import Path
from ttn_search.evaluate import Pipeline
from ttn_search.screening import evaluator

POLICY=Pipeline(tolerance_factors=(3.,1.,.1,0.))

def evaluate(target,tree,directory,prepared=None,pipeline=POLICY):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    original=evaluator.compile_reference
    def logged(*args,**kwargs):
        with (directory/'synthesis_starts.jsonl').open('a') as f:
            f.write(json.dumps(dict(monotonic=time.monotonic()))+'\n');f.flush();os.fsync(f.fileno())
        return original(*args,**kwargs)
    evaluator.compile_reference=logged
    try:return evaluator.evaluate_fresh(target,tree,pipeline,directory,prepared)
    finally:evaluator.compile_reference=original
