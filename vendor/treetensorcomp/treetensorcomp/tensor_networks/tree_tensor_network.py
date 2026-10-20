import uuid
from typing import Self

import networkx as nx
import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray
from scipy.io import savemat

from treetensorcomp.helper.linalg import generate_random_isometry, sp_multi_tensor_prod
from treetensorcomp.helper.misc import get_nth_bit_of_binary_fraction
from treetensorcomp.quantics.functions import QuanticsFunction
from treetensorcomp.quantics.index_settings import ExternalIndexSetting
from treetensorcomp.tensor_networks.tensors import Tensor
from treetensorcomp.tensor_networks.tree_graphs import TTNGraph


# TODO: Need to add a real/complex parameter to the TTN. If it starts with real pivots
# and interpolates a real function, it will have no complex entries. So its evaluation
# can be assumed real and not have to be cast to real.
class TreeTensorNetwork:
    """A tree tensor network object including multiple tensors and a graph structure."""

    ttn_graph: TTNGraph
    name: str
    tensors: dict[str, Tensor]
    gauge: str

    # TODO: an initializer that chooses a single pivot and creates a trivial tree with that.

    @classmethod
    def random_tree(
        cls,
        ttn_graph: TTNGraph,
        max_bond_dimension: int = 4,
        saturate: bool = True,
        name: str | None = None,
    ) -> Self:
        """
        Initialize a TreeTensorNetwork for a given graph structure with random tensors.

        Parameters
        ----------
        ttn_graph : TTNGraph
            The graph structure for the tree tensor network.
        max_bond_dimension : int, optional
            Maximum bond dimension of the generated tree tensor network (default: 4).
        saturate : bool, optional
            Whether the bond dimension of each node matches the maximum bond dimension
            or a random number less than or equal to `max_bond_dimension` (default:
            True).
        name : str, optional
            An identifier for the TreeTensorNetwork object.

        Returns
        -------
        TreeTensorNetwork
            A randomly generated TreeTensorNetwork based off the given graph.

        """
        instance = cls()

        ttn_graph.sanity_check_graph_structure()
        instance.ttn_graph = ttn_graph

        if name is not None:
            instance.name = name
        else:
            instance.name = "TTN_" + str(uuid.uuid4())

        # Start from the leaf nodes and work towards the center, choosing dimension of each bond
        tensor_subgraph = instance.ttn_graph.digraph.subgraph(
            [
                node
                for node in instance.ttn_graph.digraph.nodes
                if instance.ttn_graph.digraph.nodes[node].get("node_type", "")
                != "physical_index"
            ]
        )  # subgraph without physical indices (tensors only)
        processing_order = list(nx.topological_sort(tensor_subgraph))

        # Now work up the tree hierarchy and choose dimensions for each bond
        for current_node in processing_order:
            outgoing_edges = list(instance.ttn_graph.digraph.out_edges(current_node))

            # Non-root tensors: set the dimension of the outgoing edge to be less than
            # or equal to the incoming, so that it's an isometry. (In general SVD
            # compression would enforce this)
            if len(outgoing_edges) > 0:
                incoming_edges = list(instance.ttn_graph.digraph.in_edges(current_node))
                incoming_dimension = np.prod(
                    [
                        instance.ttn_graph.digraph.edges[edge]["dimension"]
                        for edge in incoming_edges
                    ]
                )
                # 'saturate=True' means that we set the maximum possible bond dimension
                # up to the user-supplied limit. Otherwise random up to that. This is
                # used to stop TCI getting stuck on internal nodes.
                if saturate:
                    outgoing_dimension = min(incoming_dimension, max_bond_dimension)
                else:
                    # cap it at user-flagged max bond dimension
                    outgoing_dimension = min(
                        np.random.randint(1, incoming_dimension + 1), max_bond_dimension
                    )

                instance.ttn_graph.digraph.edges[outgoing_edges[0]]["dimension"] = (
                    outgoing_dimension
                )
                instance.ttn_graph.digraph.nodes[current_node]["node_type"] = "isometry"

            # Center node
            else:
                instance.ttn_graph.digraph.nodes[current_node]["node_type"] = (
                    "center_orthogonality"  # We are building from isometries, so this is an orthogonality center by construction
                )

        # Now generate random isometries according to the dimensions of the edges set above
        instance.tensors = {}
        for node in instance.ttn_graph.digraph.nodes:
            if (
                instance.ttn_graph.digraph.nodes[node].get("node_type", "")
                != "physical_index"
            ):
                # link the tensor to the node id in the graph
                tensor_id = instance.ttn_graph.digraph.nodes[node]["id"]

                # get the incoming dimensions
                incoming_edges = list(instance.ttn_graph.digraph.in_edges(node))
                incoming_dimensions = [
                    instance.ttn_graph.digraph.edges[edge]["dimension"]
                    for edge in incoming_edges
                ]
                n = np.prod(incoming_dimensions)

                # Orthogonality center (incoming edges only)
                if (
                    instance.ttn_graph.digraph.nodes[node].get("node_type", "")
                    == "center_orthogonality"
                ):
                    # generate and reshape the isometry
                    isometry_matrix = generate_random_isometry(n, 1)
                    isometry_tensor = isometry_matrix.reshape(incoming_dimensions)
                    tensor_indices = [
                        instance.ttn_graph.digraph.edges[edge]["id"]
                        for edge in incoming_edges
                    ]  # link the tensor to the edge ids in the graph

                # Non-central tensor (both incoming and outgoing edges)
                else:
                    outgoing_edges = list(instance.ttn_graph.digraph.out_edges(node))

                    # outgoing edges, exactly one outgoing bond expected
                    assert len(outgoing_edges) == 1

                    m = instance.ttn_graph.digraph.edges[outgoing_edges[0]]["dimension"]
                    isometry_matrix = generate_random_isometry(n, m)
                    isometry_tensor = isometry_matrix.reshape((*incoming_dimensions, m))

                    # link the tensor to the edge ids in the graph
                    tensor_indices = [
                        instance.ttn_graph.digraph.edges[edge]["id"]
                        for edge in incoming_edges
                    ] + [instance.ttn_graph.digraph.edges[outgoing_edges[0]]["id"]]

                # store the tensor in the instance
                new_tensor = Tensor(isometry_tensor, tensor_indices, id=tensor_id)
                instance.tensors[tensor_id] = new_tensor

        instance.gauge = "orthogonality"  # by construction
        return instance

    @classmethod
    def single_pivot(
        cls,
        ttn_graph: TTNGraph,
        quantics_function: QuanticsFunction,
    ) -> Self:
        """
        Initialize a TreeTensorNetwork for a given graph structure with a single pivot.

        The output TTN has a bond dimension of one and is in the interpolative gauge.

        Parameters
        ----------
        ttn_graph : TTNGraph
            The graph structure for the tree tensor network.
        quantics_function : QuanticsFunction
            The function used to evaluate the pivot.

        Returns
        -------
        TreeTensorNetwork
            A new TreeTensorNetwork based off the given graph.

        """
        if not isinstance(ttn_graph, TTNGraph):
            msg = "ttn_graph must be an instance of TTNGraph"
            raise TypeError(msg)

        if not isinstance(quantics_function, QuanticsFunction):
            msg = "quantics_function must be an instance of QuanticsFunction"
            raise TypeError(msg)

        instance = cls()
        ttn_graph.sanity_check_graph_structure()
        instance.ttn_graph = ttn_graph
        instance.name = "TTN_" + quantics_function.name
        instance.tensors = {}

        # Initialize with a good initial pivot and rescale it to [0,1]
        variable_setting = quantics_function.find_maximum_magnitude()
        variable_setting_rescaled = {
            var: (val - quantics_function.variable_domains[var][0])
            / (
                quantics_function.variable_domains[var][1]
                - quantics_function.variable_domains[var][0]
            )
            for var, val in variable_setting.items()
        }

        # Loop over the tensor nodes and build trivial interpolation tensors for a single pivot
        tensor_nodes = [
            node
            for node in instance.ttn_graph.digraph.nodes
            if instance.ttn_graph.digraph.nodes[node].get("node_type", "")
            != "physical_index"
        ]

        for node in tensor_nodes:
            in_edge_ids = [
                instance.ttn_graph.digraph.in_edges[edge]["id"]
                for edge in instance.ttn_graph.digraph.in_edges(node)
            ]
            out_edge_ids = [
                instance.ttn_graph.digraph.out_edges[edge]["id"]
                for edge in instance.ttn_graph.digraph.out_edges(node)
            ]

            index_names = in_edge_ids + out_edge_ids
            tensor_factors = []

            for index_name in index_names:
                if index_name.startswith("bond_"):
                    tensor_factor = np.array([1.0], dtype=float)
                else:
                    tensor_factor = np.zeros(2, dtype=float)
                    varname, bitnum = index_name.split("_")
                    bitnum = int(bitnum)
                    tensor_factor[
                        get_nth_bit_of_binary_fraction(
                            variable_setting_rescaled[varname], bitnum
                        )
                    ] = 1.0  # Choose the bit corresponding to the variable setting

                tensor_factors.append(tensor_factor)

            tensor = sp_multi_tensor_prod(tensor_factors)
            node_name = instance.ttn_graph.digraph.nodes[node]["id"]
            instance.tensors[node_name] = Tensor(tensor, index_names, id=node_name)

        # Now transform into interpolative gauge (this shouldn't change the tensors,
        # just give the correct index settings along the bonds, allowing one to set the
        # pivot at the centre)
        root_node = instance.ttn_graph.find_tree_center()
        instance.move_gauge_centre(
            root_node,
            gauge_type="interpolative",
            gauge_parameters={"relative_error": 1e-12},
        )

        # Get the external index setting for the root node's incoming edges
        root_in_edges = list(instance.ttn_graph.digraph.in_edges(root_node))
        root_index_setting = ExternalIndexSetting.tensor_parent_index_settings(
            [
                instance.ttn_graph.digraph.edges[edge]["external_index_setting"]
                for edge in root_in_edges
            ]
        )

        # Update the pivot(s) at the center tensor
        center_tensor_flattened = quantics_function.evaluate_index_setting(
            root_index_setting
        )
        center_tensor_shape = [
            instance.ttn_graph.digraph.edges[edge]["dimension"]
            for edge in root_in_edges
        ]

        instance.tensors[instance.ttn_graph.digraph.nodes[root_node]["id"]] = Tensor(
            center_tensor_flattened.reshape(center_tensor_shape),
            [instance.ttn_graph.digraph.edges[edge]["id"] for edge in root_in_edges],
            id=instance.ttn_graph.digraph.nodes[root_node]["id"],
        )

        return instance

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        # Extract physical index prefixes
        physical_nodes = [
            node
            for node in self.ttn_graph.digraph.nodes
            if self.ttn_graph.digraph.nodes[node]["node_type"] == "physical_index"
        ]

        physical_ids = [
            self.ttn_graph.digraph.nodes[node]["id"] for node in physical_nodes
        ]

        # Get unique prefixes from physical index IDs (part before '_')
        prefixes = set()
        for pid in physical_ids:
            if "_" in pid:
                prefixes.add(pid.split("_")[0])

        num_prefixes = len(prefixes)

        # Find maximum bond dimension
        max_bond_dim = self.get_max_bond_dimension()

        # Current gauge
        current_gauge = getattr(self, "gauge", "unknown (this should not happen...)")

        # TODO: draw the TTN graph with inbuilt function and convert to b64 HTML to include in the HTML output

        return f"""
        <div style="border: 1px solid #ccc; padding: 10px; margin: 5px;">
            <strong>TreeTensorNetwork</strong><br>
            <strong>Tensors:</strong> {len(self.tensors)}<br>
            <strong>Physical indices:</strong> {self.num_qubits()}<br>
            <strong>Variables:</strong> {num_prefixes} ({", ".join(list(prefixes))})<br>
            <strong>Max bond dimension:</strong> {max_bond_dim}<br>
            <strong>Current gauge:</strong> {current_gauge}
        </div>
        """

    def num_qubits(self) -> int:
        """
        Get the number of qubits in the TTN.

        Calculated by counting the number of physical indices in the graph.
        """
        return len(
            [
                node
                for node in self.ttn_graph.digraph.nodes
                if self.ttn_graph.digraph.nodes[node].get("node_type", "")
                == "physical_index"
            ]
        )

    def full_tensor(self) -> Tensor:
        """
        Construct and then contract the full tensor network of the TreeTensorNetwork.

        After the contraction only physical indices are left dangling.
        Note this procedure is exponentially expensive in number of qubits.

        Returns
        -------
        Tensor
            The contracted full tensor.

        """
        if self.num_qubits() > 16:
            msg = (
                "Full tensor contraction is exponentially expensive in system size."
                "Consider using another method (e.g. efficient contraction of tree)"
            )
            raise ValueError(msg)

        # Loop over the tensor metadata and numerical values and create a list of quimb tensors
        quimb_tensor_list = []
        for id, tensor in self.tensors.items():
            quimb_tensor = qtn.Tensor(
                tensor.tensor_ndarray, inds=tensor.indices, tags={id}
            )
            quimb_tensor_list.append(quimb_tensor)

        result = qtn.TensorNetwork(quimb_tensor_list).contract()

        # TODO: make this a Tensor object instead.
        return Tensor(result.data, result.inds, id="contracted_TTN")

    def svd_push(
        self,
        node_index: int,
        cutoff_rtol: float | None = None,
        max_bond_dim: int | None = None,
    ) -> None:
        """
        Perform an SVD push gauge transformation on the given node's tensor.

        Transforms the given node into an isometry by splitting it with (thin/
        economical) SVD: tensor -> U.S.V. Then replaces the tensor at the given node
        with U, and pushes the S.V. part up to its parent node in the tree.

        Truncation can be included via `cutoff_rtol` and `max_bond_dim`.

        Parameters
        ----------
        node_index : int
            The index of the node in the TTN graph to perform the SVD push on.
        cutoff_rtol : float, optional
            Relative tolerance for truncating singular values during SVD.
        max_bond_dim : int, optional
            Maximum allowed bond dimension after SVD truncation.

        """
        # Get incoming and outgoing edges of the node (to partition the tensor via SVD)
        in_edges = list(self.ttn_graph.digraph.in_edges(node_index))
        out_edges = list(self.ttn_graph.digraph.out_edges(node_index))

        in_edge_ids = [self.ttn_graph.digraph.edges[edge]["id"] for edge in in_edges]
        out_edge_ids = [self.ttn_graph.digraph.edges[edge]["id"] for edge in out_edges]

        assert len(out_edge_ids) == 1, (
            "There should be exactly one out edge for a non-central node in a tree structure."
        )
        assert len(in_edge_ids) > 0, (
            "Nodes must have at least one input edge (except for leaf nodes which are physical indices, not tensors)"
        )

        # Pick current tensor (will be split, replaced by the U isometry and the rest pushed up)
        current_tensor_id = self.ttn_graph.digraph.nodes[node_index]["id"]
        current_tensor = self.tensors[current_tensor_id]

        # SVD split the current tensor, then contract the S and V tensors (which we'll push up the tree)
        u_tensor, s_tensor, v_tensor, _ = current_tensor.svd(
            in_edge_ids,
            out_edge_ids,
            cutoff_rtol=cutoff_rtol,
            max_bond_dim=max_bond_dim,
        )
        sv_tensor = s_tensor @ v_tensor

        # Identify the next tensor up the tree with which we'll contract the SV tensor
        parent_node = list(self.ttn_graph.digraph.out_edges(node_index))[0][1]
        parent_tensor_id = self.ttn_graph.digraph.nodes[parent_node]["id"]
        parent_tensor = self.tensors[parent_tensor_id]

        # Contract the SV tensor with the parent tensor
        contracted_upper_tensor = sv_tensor @ parent_tensor
        common_indices = set(current_tensor.indices).intersection(parent_tensor.indices)
        assert len(common_indices) == 1, (
            "There should be exactly one common index between the tensor and its parent"
        )

        # Store the id of the bond that we're replacing, to restore its name later
        bond_id = list(common_indices)[0]

        # Identify the 'svd_left_' index that replaces the bond
        svd_left_indices = [
            idx for idx in u_tensor.indices if idx.startswith("svd_left_")
        ]
        assert len(svd_left_indices) == 1, (
            "There should be exactly one svd_left_ index in the SVD tensor"
        )

        svd_left_index = svd_left_indices[0]
        assert svd_left_index in contracted_upper_tensor.indices, (
            "The svd_left_ index should be present in the contracted tensor after SVD (replaces the bond)"
        )

        # Relabel the svd_left_ index to the original bond id
        index_map = {svd_left_index: bond_id}
        new_child_tensor = u_tensor.relabel_indices(index_map)

        # Rename the child tensor to original name and relabel indices
        new_child_tensor.rename_self(current_tensor_id)
        new_parent_tensor = contracted_upper_tensor.relabel_indices(index_map)

        new_parent_tensor.rename_self(parent_tensor_id)

        # Update the tensors in the TTN data structure
        self.tensors[current_tensor_id] = new_child_tensor
        self.tensors[parent_tensor_id] = new_parent_tensor

        # Update the graph metadata with new dimension and node type (isometry)
        new_dimension = new_child_tensor.tensor_ndarray.shape[
            new_child_tensor.indices.index(bond_id)
        ]
        self.ttn_graph.digraph.edges[(node_index, parent_node)]["dimension"] = (
            new_dimension
        )
        self.ttn_graph.digraph.nodes[node_index]["node_type"] = "isometry"

    def interpolative_push(self, node_index: int, relative_error: float) -> None:
        """
        Perform an interpolative push gauge transformation on the given node's tensor.

        Performs an interpolative decomposition on the tensor, pushing the skeleton
        (function values) upwards and replacing the tensor with the interpolation
        tensor. Also tracks the mapping of external index settings to keep
        interpolative gauge.

        Parameters
        ----------
        node_index : int
            The index of the node in the TTN graph to perform the interpolative push on.
        relative_error : float
            Relative error threshold for the interpolative decomposition

        """
        if not (0 < relative_error < 1):
            msg = "Relative error must be between 0 and 1."
            raise ValueError(msg)

        # Get incoming and outgoing edges of the node (to partition the tensor via ID)
        in_edges = list(self.ttn_graph.digraph.in_edges(node_index))
        out_edges = list(self.ttn_graph.digraph.out_edges(node_index))

        in_edge_ids = [self.ttn_graph.digraph.edges[edge]["id"] for edge in in_edges]
        out_edge_ids = [self.ttn_graph.digraph.edges[edge]["id"] for edge in out_edges]

        assert len(out_edge_ids) == 1, (
            "There should be exactly one out edge for a non-central node in a tree structure."
        )
        assert len(in_edge_ids) > 0, (
            "Nodes must have at least one input edge (except for leaf nodes which are physical indices, not tensors)"
        )

        # Pick current tensor (will be split, replaced by the U isometry and the rest pushed up)
        current_tensor_id = self.ttn_graph.digraph.nodes[node_index]["id"]
        current_tensor = self.tensors[current_tensor_id]

        # split the tensor via interpolative decomposition
        # N.B.: for SVD push, the right-hand side is pushed up, but for ID, the left-hand side is pushed up
        skeleton_tensor, interpolation_tensor, interpolated_column_indices = (
            current_tensor.interpolative_decomposition(
                out_edge_ids, in_edge_ids, relative_error=relative_error
            )
        )

        # Identify the next tensor up the tree with which we'll contract the skeleton tensor
        parent_node = list(self.ttn_graph.digraph.out_edges(node_index))[0][1]
        parent_tensor_id = self.ttn_graph.digraph.nodes[parent_node]["id"]
        parent_tensor = self.tensors[parent_tensor_id]

        # Contract the skeleton tensor with the parent tensor
        contracted_upper_tensor = skeleton_tensor @ parent_tensor

        # Store the id of the bond that we're replacing, then replace its name in the new tensors
        common_indices = set(current_tensor.indices).intersection(parent_tensor.indices)
        assert len(common_indices) == 1, (
            "There should be exactly one common index between the tensor and its parent"
        )

        bond_id = list(common_indices)[0]
        interpdecomp_id = interpolation_tensor.indices[0]
        assert interpdecomp_id.startswith("interpdecomp_"), (
            "Something has gone wrong with index ordering in the interpolative decomposition"
        )

        index_mapping = {interpdecomp_id: bond_id}

        # Use this map to create the new tensors with relabelled indices; rename these new tensors to match their original names
        new_child_tensor = interpolation_tensor.relabel_indices(index_mapping)
        new_parent_tensor = contracted_upper_tensor.relabel_indices(index_mapping)

        new_child_tensor.rename_self(current_tensor_id)
        new_parent_tensor.rename_self(parent_tensor_id)

        # Update the tensors in the TTN data structure
        self.tensors[current_tensor_id] = new_child_tensor
        self.tensors[parent_tensor_id] = new_parent_tensor

        # Update the graph metadata with new dimension and node type (interpolator)
        new_dimension = new_child_tensor.tensor_ndarray.shape[
            new_child_tensor.indices.index(bond_id)
        ]
        self.ttn_graph.digraph.edges[(node_index, parent_node)]["dimension"] = (
            new_dimension
        )
        self.ttn_graph.digraph.nodes[node_index]["node_type"] = "interpolator"

        # Now we equip the new edge with the appropriate external index setting
        # First get the tensor product of possible indices from below on the tree
        child_indices = [
            self.ttn_graph.digraph.edges[edge]["external_index_setting"]
            for edge in in_edges
        ]
        product_setting = ExternalIndexSetting.tensor_parent_index_settings(
            child_indices
        )
        new_setting = ExternalIndexSetting.from_subset(
            product_setting, interpolated_column_indices
        )

        # Set the new external index setting on the outgoing edge
        self.ttn_graph.digraph.edges[(node_index, parent_node)][
            "external_index_setting"
        ] = new_setting

        # Update the dimension of the edge in the graph
        new_dimension = new_child_tensor.tensor_ndarray.shape[
            new_child_tensor.indices.index(bond_id)
        ]
        self.ttn_graph.digraph.edges[(node_index, parent_node)]["dimension"] = (
            new_dimension
        )

    # TODO: Need to add functionality to this that can skip nodes already processed (e.g. when moving interpolative gauge center).
    # TODO: This could check if the node is already an isometry or interpolator (for orthogonality/interpolative gauge respectively), and if so, skip it.
    def move_gauge_centre(
        self,
        new_centre_node_id: int,
        gauge_type: str = "orthogonality",
        gauge_parameters: dict[str, float] | None = None,
    ) -> None:
        """
        Move the gauge center of the tree tensor network to a new node.

        Parameters
        ----------
        new_centre_node_id : int
            The node to use as the new gauge center.
        gauge_type : {"orthogonality", "interpolative"}
            The type of the new gauge center.
        gauge_parameters : dict[str, float] | None
            Optional additional arguments to pass to gauge transformation. For valid
            arguments see `svd_push` and `interpolative_push` documentation.

        https://www.tensors.net/tutorial-3

        """
        # Check inputs
        assert gauge_type in ["orthogonality", "interpolative"], (
            "Unsupported gauge type."
        )
        # Move the center in the internal graph structure (N.B.: this doesn't change the gauge yet, just the directions in the DiGraph)
        self.ttn_graph.move_tree_center(new_centre_node_id, center_type=gauge_type)

        if gauge_type == "interpolative":
            # Equip the physical index edges with external index settings that can be tensor'ed up the tree
            self.ttn_graph.setup_leaf_external_settings()
        else:
            # Clear all external index settings (these are meaningless outside interpolative gauge)
            self.ttn_graph.clear_external_settings()

        # Now work up the tree hierarchy (using topo sort) and push tensors up
        tensor_subgraph = self.ttn_graph.digraph.subgraph(
            [
                node
                for node in self.ttn_graph.digraph.nodes
                if self.ttn_graph.digraph.nodes[node].get("node_type", "")
                != "physical_index"
            ]
        )  # subgraph without physical indices (tensors only)

        processing_order = list(nx.topological_sort(tensor_subgraph))

        for current_node in processing_order:
            outgoing_edges = list(self.ttn_graph.digraph.out_edges(current_node))
            # Now process the current node with a 'push'.
            # -> For orthogonality centre setting, use SVD to turn into an isometry. (If this is too slow, it could be done faster with QR (though no compression)).
            # -> For interpolative gauge, use interpolative decomposition to turn into an interpolator tensor and push function values (skeletons) up.
            if len(outgoing_edges) > 0:
                # SVD push: decompose tensor with SVD, keep the isometry, contract the rest up into parent tensor
                if gauge_type == "orthogonality":
                    # Get the truncation parameters if they were supplied
                    cutoff_rtol = (
                        gauge_parameters.get("cutoff_rtol", None)
                        if gauge_parameters
                        else None
                    )
                    max_bond_dim = (
                        gauge_parameters.get("max_bond_dim", None)
                        if gauge_parameters
                        else None
                    )
                    # Apply SVD push from the current node up to its parent
                    self.svd_push(
                        current_node, cutoff_rtol=cutoff_rtol, max_bond_dim=max_bond_dim
                    )
                elif gauge_type == "interpolative":
                    relative_error = (
                        gauge_parameters.get("relative_error", None)
                        if gauge_parameters
                        else None
                    )
                    if relative_error is None:
                        raise ValueError(
                            "A relative error for the interpolative decomposition must be specified for interpolative gauge."
                        )

                    # Apply interpolative push from the current node up to its parent
                    self.interpolative_push(current_node, relative_error)

            # Center node
            else:
                self.ttn_graph.digraph.nodes[current_node]["node_type"] = (
                    "center_" + gauge_type
                )
        self.gauge = gauge_type  # Update the gauge type in the instance

    def sanity_check(self) -> None:
        """Validate the tree tensor network's properties."""
        self.ttn_graph.sanity_check_graph_structure()
        # TODO Check that self.tensors has ids matching the tensors
        # TODO Check tensor bond structure, e.g. dimensions match, each bond_ occurs exactly twice, isometries really are isometries
        # All edges should have bond dimension that (a) exists and (b) is <= the maximum possible (max_edge_dimension)
        # for edge in self.digraph.edges:
        # Check that the edge has a dimension set
        # if 'dimension' not in self.digraph.edges[edge]:
        #    raise ValueError(f"Edge {edge} does not have a 'dimension' attribute.")
        # Check that the dimension does not exceed the maximum possible for this edge
        # if self.digraph.edges[edge]['dimension'] > self.max_edge_dimension(edge_id):
        #    raise ValueError(f"Edge {edge} with id '{edge_id}' has dimension above the theoretical maximum.")
        # for edge in self.digraph.edges:
        # TODO All outgoing edges should have bond dimension not exceeding the product of incoming edge dimensions
        # TODO Check that the physical edges have index setting data.
        raise NotImplementedError("sanity check various structural things")

    def evaluate_index_setting(
        self,
        index_setting: ExternalIndexSetting,
    ) -> NDArray[np.float64]:
        """
        Evaluate the TTN with a given index setting.

        Convenience function that contracts the TTN with the specified index setting.

        TODO: Currently this is probably inefficient, using a for loop over the settings
        and quimb auto-contract. It's probably good enough because it's not used in any
        inner loops, but TODO optimize this if it proves an obstacle.

        Parameters
        ----------
        index_setting : ExternalIndexSetting
            The index setting to evaluate the TTN with.

        Returns
        -------
        NDArray[np.float64]
            The result of the evaluation.

        """
        # Check that the index setting is actually compatible with the TTN
        physical_indices = [
            self.ttn_graph.digraph.nodes[node]["id"]
            for node in self.ttn_graph.digraph.nodes
            if self.ttn_graph.digraph.degree[node] == 1
        ]

        setting_indices = index_setting.mapping.keys()
        assert set(physical_indices) == set(setting_indices), (
            "The index setting must exactly match physical indices of the TTN."
        )

        # Empty result array, TODO probably handle real/complex better
        result = np.empty(index_setting.setting_count, dtype=complex)

        # Loop over the index setting and evaluate the TTN for each setting
        for setting_index in range(index_setting.setting_count):
            cap_tensors = []
            for index_id, values in index_setting.mapping.items():
                cap_vector = np.zeros(2, dtype=float)
                cap_vector[values[setting_index]] = 1
                cap_tensors.append(Tensor(cap_vector, [index_id], id="cap_" + index_id))

            # Loop over the tensor metadata and numerical values and create a list of quimb tensors
            quimb_tensor_list = []
            for tensor in list(self.tensors.values()) + cap_tensors:
                quimb_tensor = qtn.Tensor(
                    tensor.tensor_ndarray, inds=tensor.indices, tags={tensor.id}
                )
                quimb_tensor_list.append(quimb_tensor)

            # quimb autocontract, TODO order might be suboptimal / this approach is inefficient
            result[setting_index] = qtn.TensorNetwork(quimb_tensor_list).contract()

        return result

    def get_pivots(
        self,
        quantics_function: QuanticsFunction,
    ) -> tuple[
        ExternalIndexSetting, dict[str, float], NDArray[np.float64], NDArray[np.float64]
    ]:
        """
        Get the index settings and respective function values given a QuanticsFunction.

        The TreeTensorNetwork must be in the interpolative gauge.

        Parameters
        ----------
        quantics_function : QuanticsFunction
            The function to evaluate at the pivots.

        Returns
        -------
        root_index_setting: ExternalIndexSetting
            The index setting at the center node, i.e. the pivot indices, implied by the
            interpolative gauge metadata.
        variable_values: dict[str, float]
            The corresponding variable values for these indices, i.e. the pivot variable
            values.
        center_tensor_flattened: NDArray[np.float64]
            The values at the center tensor of the TTN, flattened into a 1D array.
            Should approximately match the function values at the root_index_setting
            to machine precision if the TTN is in correct interpolative gauge.
        function_values: NDArray[np.float64]
            The values of the quantics_function at the root_index_setting, evaluated
            directly from the function. Should approximately match the function values
            at the root_index_setting to machine precision if the TTN is in correct
            interpolative gauge.

        """
        if not isinstance(quantics_function, QuanticsFunction):
            msg = "quantics_function must be an instance of QuanticsFunction"
            raise TypeError(msg)

        # Check that the TTN is in interpolative gauge
        if self.gauge != "interpolative":
            msg = "This function can only be used in interpolative gauge."
            raise ValueError(msg)

        # Get the external index setting for the root node's incoming edges
        center_node = self.ttn_graph.gauge_center()
        root_in_edges = list(self.ttn_graph.digraph.in_edges(center_node))
        root_index_setting = ExternalIndexSetting.tensor_parent_index_settings(
            [
                self.ttn_graph.digraph.edges[edge]["external_index_setting"]
                for edge in root_in_edges
            ]
        )

        # Get the center tensor
        center_tensor_flattened = self.tensors[
            self.ttn_graph.digraph.nodes[center_node]["id"]
        ].tensor_ndarray.flatten()

        # Get the function values at this index setting
        # (should be the same if the TTN is in interpolative gauge with correct pivots)
        function_values = quantics_function.evaluate_index_setting(root_index_setting)
        variable_values = quantics_function.index_setting_to_variable_setting(
            root_index_setting
        )

        return (
            root_index_setting,
            variable_values,
            center_tensor_flattened,
            function_values,
        )

    def get_max_bond_dimension(self) -> int:
        """Get the maximum bond dimension in the TTN based on the edges of the graph."""
        max_bond_dim = 0

        for edge in self.ttn_graph.digraph.edges:
            if "dimension" in self.ttn_graph.digraph.edges[
                edge
            ] and self.ttn_graph.digraph.edges[edge]["id"].startswith("bond_"):
                dim = self.ttn_graph.digraph.edges[edge]["dimension"]
                max_bond_dim = max(max_bond_dim, dim)
        return max_bond_dim

    def num_amplitudes(self) -> int:
        """Get the number of amplitudes across all tensors."""
        size = 0
        for node in self.ttn_graph.digraph.nodes:
            if (
                self.ttn_graph.digraph.nodes[node].get("node_type", None)
                != "physical_index"
            ):
                tensor_id = self.ttn_graph.digraph.nodes[node]["id"]
                elements = self.tensors[tensor_id].tensor_ndarray.size
                size += elements
        return size

    def memsize(self) -> int:
        """Get the memory size of the TTN in bytes, based on the tensor dimensions."""
        return (
            self.num_amplitudes()
            * list(self.tensors.values())[0].tensor_ndarray.dtype.itemsize
        )

    def save_to_file(self, filename: str) -> None:
        """
        Save the TTN to a file.

        Parameters
        ----------
        filename : str
            File to save the data to. Must be a .mat file.

        """
        if not isinstance(filename, str):
            msg = "Filename must be a string."
            raise TypeError(msg)

        if not filename.endswith(".mat"):
            msg = "Currently only .mat file format is supported."
            raise ValueError(msg)

        tensor_metadata_list = []
        numerical_tensors = {}

        for node in self.ttn_graph.digraph:
            if self.ttn_graph.digraph.nodes[node]["node_type"] != "physical_index":
                tensor_metadata = {}
                node_id = self.ttn_graph.digraph.nodes[node]["id"]
                node_indices = self.tensors[node_id].indices
                tensor_metadata["id"] = node_id
                tensor_metadata["indices"] = node_indices
                tensor_metadata["dimension"] = self.tensors[
                    node_id
                ].tensor_ndarray.shape
                directions = []

                incoming_edges = [
                    self.ttn_graph.digraph.edges[edge]["id"]
                    for edge in self.ttn_graph.digraph.in_edges(node)
                ]

                outgoing_edges = [
                    self.ttn_graph.digraph.edges[edge]["id"]
                    for edge in self.ttn_graph.digraph.out_edges(node)
                ]

                for index in node_indices:
                    if index in incoming_edges:
                        directions.append("child")
                    elif index in outgoing_edges:
                        directions.append("parent")
                    else:
                        msg = f"Index {index} not found in incoming or outgoing edges."
                        raise ValueError(msg)

                tensor_metadata["directions"] = directions
                numerical_tensors[node_id] = self.tensors[node_id].tensor_ndarray
                tensor_metadata_list.append(tensor_metadata)

        savemat(
            filename,
            {
                "tensor_metadata_list": tensor_metadata_list,
                "numerical_tensors": numerical_tensors,
            },
            long_field_names=True,
        )

    # TODO add an initializer that takes a list of tensors and a graph, checks compatability and consistency, and sets the gauge to generic
    # TODO add sanity checks (take from notebook) checking tensor dimension consistency between graph and stored tensors.
    # TODO add unit tests with pytest.
