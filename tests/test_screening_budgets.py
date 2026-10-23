import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('screening_report',Path('scripts/report_screening_benchmark.py'))
assert spec is not None and spec.loader is not None
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_no_credit_before_verification_or_across_retry_boundary():
    run={'trace':[{'k':1,'compiler_calls':1,'elapsed_s':.8,'best_cx':None},
                  {'k':2,'compiler_calls':3,'elapsed_s':2.5,'best_cx':200},
                  {'k':3,'compiler_calls':4,'elapsed_s':3.1,'best_cx':100}]}
    assert module.at_budget(run,.7,'time') is None
    assert module.at_budget(run,2.4,'time') is None
    assert module.at_budget(run,2.5,'time')==200
    assert module.at_budget(run,2,'compiler') is None
    assert module.at_budget(run,3,'compiler')==200
    assert module.at_budget(run,2,'candidate')==200
    assert module.at_budget(run,10,'time')==100


def test_scent_stops_after_one_evaluation():
    run={'trace':[{'k':1,'compiler_calls':1,'elapsed_s':.2,'best_cx':75}]}
    assert module.at_budget(run,16,'candidate')==75
    assert module.at_budget(run,.1,'time') is None
    assert module.at_budget(run,.2,'time')==75
