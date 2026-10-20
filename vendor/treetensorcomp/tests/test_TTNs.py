import numpy as np
from treetensorcomp.tensor_networks import tree_graphs, tree_tensor_network
from treetensorcomp.quantics.index_settings import ExternalIndexSetting

# TODO write test that sets gauge with ortho centre, then check everything is actually an isometry
# as well as checking the full tensor is the same before and after
# if generating from a random isometry tree, probably check the isometries of that first too

def test_interpolative_gauge():
    """
    Test that checks that interpolative gauge exactly interpolates the full tensor at the gauge centre.
    Checks this at two nodes: one internal and one with an external index.
    """
    # Create our test TTN with a binary tree topology
    test_ttng = tree_graphs.TTNGraph.from_binary_tree(branching_ratio=2, depth=4)
    test_ttn = tree_tensor_network.TreeTensorNetwork.random_tree(test_ttng)

    # Full tensor for later comparison
    full_tensor = test_ttn.full_tensor()

    # Try an internal node and one with an external index
    for test_node in [2, 11]:
        # Set interpolative gauge centred at the test node and pullout its tensor
        test_ttn.move_gauge_centre(test_node, gauge_type='interpolative', gauge_parameters={'relative_error': 1e-12})
        test_tensor_id = test_ttn.ttn_graph.digraph.nodes[test_node]['id']
        test_tensor = test_ttn.tensors[test_tensor_id]
        # Construct the index setting at the interpolative centre
        in_edges = list(test_ttn.ttn_graph.digraph.in_edges(test_node))
        edge_ids = [test_ttn.ttn_graph.digraph.get_edge_data(u,v)['id'] for u,v in in_edges]
        assert set(edge_ids) == set(test_tensor.indices)
        perm = [edge_ids.index(idx) for idx in test_tensor.indices]
        index_settings = [test_ttn.ttn_graph.digraph.get_edge_data(*in_edges[perm[i]])['external_index_setting'] for i in range(len(in_edges))]
        product_setting = ExternalIndexSetting.tensor_parent_index_settings(index_settings)

        # Get the full-tensor slice that should correspond to interpolated values
        slice = full_tensor.evaluate_index_setting(product_setting)

        # Compare to the tensor at the gauge centre
        assert np.allclose(slice, test_tensor.tensor_ndarray.flatten())

