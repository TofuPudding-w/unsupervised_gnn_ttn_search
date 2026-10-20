import numpy as np
from treetensorcomp.quantics.functions import QuanticsFunction
from treetensorcomp.quantics.index_settings import ExternalIndexSetting
from treetensorcomp.tensor_networks import tree_graphs, tree_tensor_network
from treetensorcomp.tci.interpolation import Interpolator


def _f(q, sigma):
    return np.exp(-0.5 * ((q - 0.3) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def test_tci_accurate_1D():
    """
    Test the TCI in 1D with a simple Gaussian function.
    """
    testfunc = QuanticsFunction(_f, ['q'], {'sigma': 0.1}) # Function to interpolate
    test_ttng = tree_graphs.TTNGraph.from_binary_tree(branching_ratio=2, depth=4) # Binary tree graph structure
    test_ttn = tree_tensor_network.TreeTensorNetwork.random_tree(test_ttng) # Random tree tensor network (initial values)
    test_interpolator = Interpolator(test_ttn, testfunc, error_tol=1e-10) # Interpolator structure
    # Points to test the interpolation
    q_vals = np.linspace(0,1,100)
    test_setting = ExternalIndexSetting.from_variable_values('q', 16, q_vals)

    test_interpolator.tensor_cross_interpolation(n_sweeps=5)

    errors = np.real(test_ttn.evaluate_index_setting(test_setting)) - testfunc.evaluate_index_setting(test_setting)
    if not np.allclose(errors, 0, atol=1e-8):
        raise AssertionError(f"TCI failed, with mean error {np.mean(np.abs(errors))} and max error {np.max(np.abs(errors))}. The maximum bond dimension was {test_ttn.get_max_bond_dimension()}.")

# TODO: check that the interpolative center is exact (to some machine tolerance)
# TODO: checks in higher dimensions