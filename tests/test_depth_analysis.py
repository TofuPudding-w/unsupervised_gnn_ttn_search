import importlib.util
from pathlib import Path
from qiskit import QuantumCircuit

spec=importlib.util.spec_from_file_location('depth_analysis',Path('scripts/analyze_c1_depth.py'))
assert spec is not None and spec.loader is not None
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parallel_layers_and_ignored_single_qubit_gates():
    circuit=QuantumCircuit(4)
    circuit.cx(0,1)
    circuit.cx(2,3)
    for _ in range(10):circuit.u(.1,.2,.3,1)
    circuit.cx(1,2)
    assert module.two_qubit_depth(circuit)==2
    assert circuit.depth()>2


def test_empty_and_serial_circuits():
    circuit=QuantumCircuit(3)
    assert module.two_qubit_depth(circuit)==0
    circuit.cx(0,1);circuit.cx(1,2);circuit.cx(0,2)
    assert module.two_qubit_depth(circuit)==3
