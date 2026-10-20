from typing import Self

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from numpy.typing import NDArray

import treetensorcomp.helper.graphs as graph_helper
from treetensorcomp.quantics.index_settings import ExternalIndexSetting, IndexID


class TTNGraph:
    """Graph representation of a tree tensor network."""

    digraph: nx.DiGraph
    index_ids: list[IndexID]
    graph_prog: str

    @classmethod
    def from_binary_tree(
        cls,
        branching_ratio: int = 2,
        depth: int = 2,
        external_index_ids: list[IndexID] | None = None,
        variable: str = "q",
    ) -> Self:
        """
        Initialize a TTNGraph for a binary tree of given branching ratio and depth.

        Parameters
        ----------
        branching_ratio : int, optional
            The number of children each node has (default: 2).
        depth : int
            The depth of the binary tree (default: 2).
        external_index_ids : list[IndexID], optional
            Identifiers for the physical qubit indices.
        variable : str, optional
            An identifier for the physical qubit indices if external_index_ids is not
            provided (default: "q")

        Returns
        -------
        TTNGraph
            Tree Tensor Network Graph representing the binary tree.

        """
        instance = cls()

        # Set default qubit labels or check input format correctness if supplied
        if external_index_ids is None:
            if not isinstance(variable, str):
                msg = "variable must be a string."
                raise TypeError(msg)

            external_index_ids = [
                IndexID(variable, i + 1) for i in range(branching_ratio**depth)
            ]

        else:
            if len(external_index_ids) != branching_ratio**depth:
                msg = "Length of external_index_ids must match the number of qubits."
                raise ValueError(msg)

            if not all(isinstance(label, IndexID) for label in external_index_ids):
                msg = "All elements of qubit_labels must be IndexID instances."
                raise TypeError(msg)

        # Generate the undirected graph representing the binary tree
        undirected_graph = nx.balanced_tree(r=branching_ratio, h=depth)

        # Get rid of the degree-2 central node (and link up dangling edges)
        # This is to avoid the rank bottleneck that stops pivots from propagating
        if branching_ratio == 2:
            connected_nodes = list(undirected_graph.neighbors(0))
            assert len(connected_nodes) == 2, (
                "The center node should have exactly two neighbors in a binary tree."
            )

            undirected_graph.add_edge(connected_nodes[0], connected_nodes[1])
            undirected_graph.remove_node(0)

        leaf_nodes = [
            node
            for node in undirected_graph.nodes
            if undirected_graph.degree[node] == 1
        ]

        assert len(leaf_nodes) == len(external_index_ids), (
            "The number of qubits specified must match the number of leaf nodes in the tree."
        )

        for idx, node in enumerate(leaf_nodes):
            if undirected_graph.degree[node] == 1:
                undirected_graph.nodes[node]["external_index"] = external_index_ids[idx]

        instance.digraph = graph_helper.polish_graph(undirected_graph)
        instance.graph_prog = "dot"  # Default graph program for visualization

        return instance

    @classmethod
    def matrix_product_network(
        cls,
        # FIX(Vasco): should this maybe be list[str]?
        variables: list[int] | None = None,
        precisions: int | list[int] | None = None,
        ordering: str | None = None,
        index_ids: list[IndexID] | None = None,
    ) -> Self:
        """
        Initialize a TTNGraph for a matrix product state (MPS).

        Can construct the graph directly from a set of external tensor indices,
        otherwise first constructs the external tensor indices from the provided
        information.

        Parameters
        ----------
        variables : list[int], optional
            Identifiers for the variables that the external tensor indices represent.
            Cannot be set if `index_ids` is set.
        precisions : int | list[int], optional
            Precision in number of bits to be used for the representation of each
            variable. Cannot be set if `index_ids` is set.
        ordering : {"interleaved", "sequential"}, optional
            Order in which to construct the external tensor indices. Cannot be set if
            `index_ids` is set.
        index_ids : list[IndexID], optional
            A set of external tensor indices making up the MPS.

        Returns
        -------
        TTNGraph
            Tree Tensor Network Graph representing the matrix product state.

        """
        instance = cls()

        # Check index_ids or create them if not provided
        if index_ids is not None:
            assert variables is None, (
                "IndexIDs have been manually specified, so variables cannot be set. (set variables=None)"
            )
            assert precisions is None, (
                "IndexIDs have been manually specified, so precisions cannot be set. (set precisions=None)"
            )
            assert ordering is None, (
                "IndexIDs have been manually specified, so ordering cannot be set. (set ordering=None)"
            )

        else:
            # Check the arguments are good
            if isinstance(precisions, int):
                assert precisions > 1, "precision must be an integer > 1"
                precisions = [precisions for _ in variables]
            else:
                assert isinstance(precisions, list) and all(
                    isinstance(p, int) and p > 1 for p in precisions
                ), "precisions must be a list of integers > 1, or a single integer > 1"

            # interleaved order
            if ordering == "interleaved":
                index_ids = []
                max_precision = max(precisions)
                for bit in range(1, max_precision + 1):
                    for var, prec in zip(variables, precisions):
                        if bit <= prec:
                            index_ids.append(IndexID(var, bit))

            # sequential order
            elif ordering == "sequential":
                index_ids = []
                for var, prec in zip(variables, precisions):
                    for bit in range(1, prec + 1):
                        index_ids.append(IndexID(var, bit))

            else:
                msg = f"Unknown ordering: '{ordering}'. Please use 'interleaved' or 'sequential'."
                raise ValueError(msg)

        # Create the graph left-to-right for an MPN/MPS and label the physical nodes with the external indices.
        undirected_graph = nx.Graph()
        for idx_tensor_node, index_id in enumerate(index_ids):
            idx_physical_node = idx_tensor_node + len(index_ids)
            undirected_graph.add_node(idx_tensor_node)
            undirected_graph.add_node(idx_physical_node)
            undirected_graph.add_edge(idx_tensor_node, idx_physical_node)
            undirected_graph.nodes[idx_physical_node]["external_index"] = index_id

            if idx_tensor_node != 0:
                undirected_graph.add_edge(idx_tensor_node, idx_tensor_node - 1)

        # Polish the graph to have correct labellings, digraph flow etc
        digraph = graph_helper.polish_graph(undirected_graph)
        instance.digraph = digraph
        instance.index_ids = index_ids
        instance.graph_prog = "neato"

        return instance

    @classmethod
    def comb(
        cls,
        variables: list[str],
        precisions: int | list[int],
    ) -> Self:
        """
        Initialize a TTNGraph for a comb.

        Parameters
        ----------
        variables : list[str]
            Identifiers for the variables that the external tensor indices represent.
        precisions : int | list[int], optional
            Precision in number of bits to be used for the representation of all/ each
            variable.

        Returns
        -------
        TTNGraph
            Tree Tensor Network Graph representing the comb.

        """
        instance = cls()
        # Check arguments
        assert isinstance(variables, list) and all(
            isinstance(v, str) for v in variables
        ), "variables must be a list of strings"

        assert len(set(variables)) == len(variables), (
            "variables must be a list of unique strings"
        )

        if isinstance(precisions, int):
            assert precisions > 1, "precision must be an integer > 1"
            precisions = [precisions for _ in variables]
        else:
            assert isinstance(precisions, list) and all(
                isinstance(p, int) and p > 1 for p in precisions
            ), "precisions must be a list of integers > 1, or a single integer > 1"

        undirected_graph = nx.Graph()
        root_nodes = []
        node_counter = 0

        # Loop over the variables, and create an MPS-like chain for each of them
        for variable, precision in zip(variables, precisions):
            for idx_bit in range(precision):
                idx_tensor_node = node_counter
                idx_physical_node = node_counter + 1
                undirected_graph.add_node(idx_tensor_node)
                undirected_graph.add_node(idx_physical_node)
                undirected_graph.add_edge(idx_tensor_node, idx_physical_node)
                undirected_graph.nodes[idx_physical_node]["external_index"] = IndexID(
                    variable, idx_bit + 1
                )

                if idx_bit == 0:
                    # prepare to link the chains together later
                    root_nodes.append(idx_tensor_node)
                else:
                    # link to the previous tensor node in the chain
                    undirected_graph.add_edge(idx_tensor_node, node_counter - 2)

                node_counter += 2

        # Now link the chains together
        for idx in range(len(variables) - 1):
            undirected_graph.add_edge(root_nodes[idx], root_nodes[idx + 1])

        # Polish off for correct labellings, digraph flow etc
        digraph = graph_helper.polish_graph(undirected_graph)
        instance.digraph = digraph
        instance.index_ids = [
            IndexID(variable, idx_bit + 1)
            for variable, precision in zip(variables, precisions)
            for idx_bit in range(precision)
        ]
        instance.graph_prog = "twopi"  # Default graph program for visualization
        return instance

    @classmethod
    def from_linkage_matrix(
        cls,
        linkage_matrix: NDArray[np.float64],
        external_indices: list[IndexID],
    ) -> Self:
        """
        Create a TTNGraph instance from a linkage matrix.

        Assumes the linkage matrix matches the format from SciPy's hierarchical
        clustering routines.

        Parameters
        ----------
        linkage_matrix : NDArray[np.float64]
            The linkage matrix in the standard SciPy format.
        external_indices : list[IndexID]
            External tensor indices at the tree leaves. Must be ordered in the same
            order as the linkage matrix.

        Returns
        -------
        TTNGraph
            Tree Tensor Network Graph matching the given linkage matrix.

        """
        instance = cls()
        undirected_graph = nx.Graph()
        node_counter = 0

        # Generate leaf nodes labelled with external indices in order
        for index_id in external_indices:
            undirected_graph.add_node(node_counter)
            undirected_graph.nodes[node_counter]["external_index"] = index_id
            node_counter += 1

        # Loop through the linkage matrix to join linked nodes under a new parent
        for row in linkage_matrix:
            left_node = int(row[0])
            right_node = int(row[1])
            undirected_graph.add_node(node_counter)
            undirected_graph.add_edge(left_node, node_counter)
            undirected_graph.add_edge(right_node, node_counter)
            node_counter += 1

        # Polish off for correct labellings, digraph flow etc
        digraph = graph_helper.polish_graph(undirected_graph)
        instance.digraph = digraph
        instance.index_ids = external_indices
        instance.graph_prog = "dot"  # Default graph program for visualization

        return instance

    @classmethod
    def from_unpolished_graph(
        cls,
        undirected_graph: nx.Graph,
        external_indices: list[IndexID],
        graph_prog: str = "dot",
    ) -> Self:
        """
        Create a TTNGraph instance from a generic undirected graph.

        Parameters
        ----------
        undirected_graph : nx.Graph
            Undirected graph instance to represent. Must represent a tree.
        external_indices : list[IndexID]
            External tensor indices at the tree leaves.
        graph_prog : str, optional
            Graphing programme to use for visualization (default: "dot")

        Returns
        -------
        TTNGraph
            Tree Tensor Network Graph constructed from the given graph.


        """
        instance = cls()
        assert isinstance(undirected_graph, nx.Graph) and not isinstance(
            undirected_graph, nx.DiGraph
        ), "undirected_graph must be an undirected NetworkX Graph (not a DiGraph)."

        assert isinstance(external_indices, list) and all(
            isinstance(idx, IndexID) for idx in external_indices
        ), "external_indices must be a list of IndexID instances."

        assert nx.is_tree(undirected_graph), "undirected_graph must be a tree."

        # Assert that only degree-1 (leaf) nodes have an 'external_index' defined
        for node in undirected_graph.nodes:
            has_external_index = "external_index" in undirected_graph.nodes[node]
            is_leaf = undirected_graph.degree[node] == 1
            if has_external_index and not is_leaf:
                raise AssertionError(
                    f"Node {node} has an 'external_index' but is not a leaf (degree={undirected_graph.degree[node]})."
                )
            if not has_external_index and is_leaf:
                raise AssertionError(
                    f"Leaf node {node} (degree=1) does not have an 'external_index' defined."
                )

        leaf_indices = [
            undirected_graph.nodes[node]["external_index"]
            for node in undirected_graph.nodes
            if undirected_graph.degree[node] == 1
        ]

        assert set([str(idx_id) for idx_id in leaf_indices]) == set(
            [str(idx_id) for idx_id in external_indices]
        ), (
            "The external indices provided do not match the leaf nodes of the undirected graph."
        )

        digraph = graph_helper.polish_graph(undirected_graph)

        instance.digraph = digraph
        instance.index_ids = external_indices
        instance.graph_prog = graph_prog
        return instance

    def move_tree_center(self, new_center: int, center_type: str) -> None:
        """
        Move the center of the tree graph to a new node.

        Parameters
        ----------
        new_center : int
            The node to use as the new center of the tree.
        center_type : {"generic", "orthogonality", "interpolation"}
            The type of the new center.

        """
        # Check the graph is correctly structured before moving the center
        self.sanity_check_graph_structure()

        # Validate the new center node
        if new_center not in self.digraph.nodes:
            msg = f"Node {new_center} is not in the graph."
            raise ValueError(msg)

        # Clear the current center label
        for node in self.digraph.nodes:
            if self.digraph.nodes[node].get("node_type", "").startswith("center_"):
                del self.digraph.nodes[node]["node_type"]

        # Validate the center_type
        if not isinstance(center_type, str):
            msg = "center_type must be a string."
            raise TypeError(msg)

        approved_center_types = [
            "generic",
            "orthogonality",
            "interpolative",
        ]  # TODO: make 'interpolation' an option when implementing TCI.

        if center_type not in approved_center_types:
            msg = f"center_type must be one of {approved_center_types}."
            raise ValueError(msg)

        # Update the new center node
        self.digraph.nodes[new_center]["node_type"] = "center_" + center_type

        # Create an undirected copy of the graph for BFS
        undirected = self.digraph.to_undirected()

        # Create a new directed graph with all edges pointing toward the new center
        new_digraph = nx.DiGraph()
        new_digraph.add_nodes_from(self.digraph.nodes(data=True))

        # Use BFS to traverse the tree from the new center
        for u, v in nx.bfs_edges(undirected, new_center):
            # u is closer to the new center, v is farther
            # We want edges pointing toward the center, so add (v, u)
            # Copy any edge attributes from the original graph
            edge_data = {}
            if (v, u) in self.digraph.edges:
                edge_data = self.digraph.edges[v, u].copy()
            elif (u, v) in self.digraph.edges:
                edge_data = self.digraph.edges[u, v].copy()

            # Add the edge pointing toward the center
            new_digraph.add_edge(v, u, **edge_data)

        # If the tree has an orthogonality center, label all the other tensors as isometries
        if center_type == "orthogonality":
            for node in new_digraph.nodes:
                existing_node_type = new_digraph.nodes[node].get("node_type", "")
                if (
                    existing_node_type != "physical_index"
                    and not existing_node_type.startswith("center_")
                ):
                    new_digraph.nodes[node]["node_type"] = "isometry"

        # Replace the old digraph with the new one
        self.digraph = new_digraph

        # Check we didn't break anything
        self.sanity_check_graph_structure()

    def sanity_check_graph_structure(self) -> None:
        """Validate the structure of the graph."""
        # Check that the nodes and edges have been tagged with unique ids
        node_ids = set()
        for node in self.digraph.nodes:
            if "id" not in self.digraph.nodes[node]:
                msg = f"Node {node} does not have an 'id' attribute."
                raise ValueError(msg)
            node_id = self.digraph.nodes[node]["id"]
            if node_id in node_ids:
                msg = f"Duplicate node id found: {node_id}"
                raise ValueError(msg)
            node_ids.add(node_id)

        edge_ids = set()
        for edge in self.digraph.edges:
            if "id" not in self.digraph.edges[edge]:
                msg = f"Edge {edge} does not have an 'id' attribute."
                raise ValueError(msg)
            edge_id = self.digraph.edges[edge]["id"]
            if edge_id in edge_ids:
                msg = f"Duplicate edge id found: {edge_id}"
                raise ValueError(msg)
            edge_ids.add(edge_id)

        # The internal graph should be tree-like
        if not nx.is_tree(self.digraph):
            msg = "The DiGraph is not tree-like and doesn't correspond to a TTN."
            raise ValueError(msg)

        # Every node should have at most one parent.
        for node in self.digraph.nodes:
            if self.digraph.out_degree(node) > 1:
                msg = f"Node {node} has more than one parent."
                raise ValueError(msg)

        # There should be at most one center node, and it should have no parents.
        center_nodes = [
            node
            for node in self.digraph.nodes
            if self.digraph.nodes[node].get("node_type", "").startswith("center_")
        ]

        if len(center_nodes) > 1:
            msg = "There are multiple center nodes in the graph."
            raise ValueError(msg)

        elif len(center_nodes) == 1:
            center_node = center_nodes[0]
            if self.digraph.out_degree(center_node) != 0:
                msg = f"Center node {center_node} has a parent, which is not allowed."
                raise ValueError(msg)

        else:
            msg = "The graph doesn't have a designated center node."
            raise ValueError(msg)

        # If there is a center_ node that isn't center_generic, all other nodes should match its type (e.g. isometries, interpolative tensors, etc)
        center_node_type = self.digraph.nodes[center_node]["node_type"]
        assert center_node_type in [
            "center_generic",
            "center_orthogonality",
            "center_interpolative",
        ]
        if center_node_type == "center_orthogonality":
            # If the center is an orthogonality center, all other nodes should be isometries
            for node in self.digraph.nodes:
                if node != center_node and self.digraph.nodes[node].get(
                    "node_type", ""
                ) not in ["isometry", "physical_index"]:
                    msg = f"Node {node} should be tagged as an isometry, isn't."
                    raise ValueError(msg)

        elif center_node_type == "center_interpolative":
            # TODO: Check that these have the structure of interpolative tensors (permutation-ish).
            pass

        # Following the directed edges from anywhere must lead to the center node.
        for node in self.digraph.nodes:
            if node != center_node:
                # Check if there is a path from this node to the center node
                if not nx.has_path(self.digraph, node, center_node):
                    msg = f"No path flowing from node {node} to center node {center_node}."
                    raise ValueError(msg)

    def max_edge_dimension(self, edge_id: str) -> int:
        """
        Get the maximum dimension of the edge with the given id.

        Calculated based on the maximum SVD rank across this bond.

        Parameters
        ----------
        edge_id : str
            Identifier of the edge.

        Returns
        -------
        int
            The maximum dimension of the edge.

        """
        # Find all edges with the given id
        matching_edges = [
            edge
            for edge in self.digraph.edges
            if self.digraph.edges[edge].get("id") == edge_id
        ]
        # Ensure there is exactly one matching edge
        if len(matching_edges) != 1:
            msg = f"Expected exactly one edge with id '{edge_id}', but found {len(matching_edges)}."
            raise ValueError(msg)

        # Now work out the maximum dimension by partitioning the graph at this edge and counting physical indices
        # Get the two nodes of the edge
        u, v = matching_edges[0]

        # Create a temporary, undirected copy of the graph
        temp_graph = self.digraph.to_undirected()

        # Remove the edge in the temporary graph
        temp_graph.remove_edge(u, v)

        # Get the connected components (two subgraphs)
        components = list(nx.connected_components(temp_graph))

        # Count "physical_index" nodes in each component
        count_1 = sum(
            1
            for node in components[0]
            if self.digraph.nodes[node].get("node_type") == "physical_index"
        )

        count_2 = sum(
            1
            for node in components[1]
            if self.digraph.nodes[node].get("node_type") == "physical_index"
        )

        # Compute 2^minimum
        min_count = min(count_1, count_2)
        max_bond_dim = 2**min_count
        return max_bond_dim

    # Method to draw the tree
    def draw(
        self,
        show_edge_ids: bool = False,
        show_bond_dimensions: bool = True,
        figsize: tuple[float, float] = (8, 8),
    ) -> None:
        """
        Draw the tree graph.

        Requires graphviz, see README for further details.

        Parameters
        ----------
        show_edge_ids : bool, optional
            Whether to show edge identifiers in the final plot (default: False).
        show_bond_dimensions : bool, optional
            Whether to show bond dimensions in the final plot (default: True).
        figsize : tuple[float, float], optional
            The dimension of the final plot (default: [8, 8]).

        """
        # Define a dictionary mapping node_type to plot attributes
        node_type_attributes = {
            "center_": {"color": "red", "size": 800},
            "physical_index": {"color": "green", "size": 200},
            "default": {
                "color": "C0",
                "size": 500,
            },  # Default attributes for unspecified node types
        }

        # Extract node colors and sizes based on node_type (can start with the keys above, center and physical indices are special)
        node_colors = [
            next(
                (
                    attributes["color"]
                    for key, attributes in node_type_attributes.items()
                    if self.digraph.nodes[node].get("node_type", "").startswith(key)
                ),
                node_type_attributes["default"]["color"],
            )
            for node in self.digraph.nodes
        ]
        node_sizes = [
            next(
                (
                    attributes["size"]
                    for key, attributes in node_type_attributes.items()
                    if self.digraph.nodes[node].get("node_type", "").startswith(key)
                ),
                node_type_attributes["default"]["size"],
            )
            for node in self.digraph.nodes
        ]

        # Extract edge labels (dimension) for each edge
        edge_labels = {}
        for u, v in self.digraph.edges:
            edge_id = self.digraph.edges[u, v]["id"] if show_edge_ids else ""
            edge_dimension = self.digraph.edges[u, v].get("dimension", None)
            label = (
                edge_id.split("_")[1][0:3] + "..."
                if edge_id.startswith("bond_")
                else ""
            )
            if edge_dimension is not None and show_bond_dimensions:
                label = f"{label}({edge_dimension})"
            edge_labels[(u, v)] = label

        # Create a dictionary of node labels
        node_labels = {
            node: self.digraph.nodes[node]["id"]
            if not self.digraph.nodes[node].get("id", "").startswith("tensor_")
            else self.digraph.nodes[node]["id"].split("_")[1][:3] + "..."
            for node in self.digraph.nodes
        }

        # Draw the graph using the extracted attributes
        try:
            pos = graphviz_layout(self.digraph, prog=self.graph_prog)
        except ModuleNotFoundError:
            msg = (
                "pydot and graphviz are required for graphviz_layout. Please install "
                "them to use the draw method. pydot is available via Python package"
                "managers (e.g. pip), and graphviz must be separately installed:"
                "https://graphviz.org/download/"
            )
            raise ImportError(msg)

        # Create a new figure with specified size
        plt.figure(figsize=figsize)
        nx.draw(
            self.digraph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
        )
        nx.draw_networkx_labels(self.digraph, pos, labels=node_labels, font_size=7)
        nx.draw_networkx_edge_labels(
            self.digraph, pos, edge_labels=edge_labels, font_size=7
        )  # Draw edge labels
        plt.show()

    def find_node(self, id: str, partial: bool = False) -> int:
        """
        Find a node in the graph exactly matching a given ID or alternatively its start.

        Parameters
        ----------
        id : str
            Identifier for the node.
        partial : bool, optional
            Whether to mark any node with the same start as a match (default: False).

        Returns
        -------
        int
            The node found.

        """
        if not isinstance(id, str):
            msg = "id must be a string."
            raise TypeError(msg)

        if not id.startswith("tensor_"):
            id = "tensor_" + id

        if partial:
            matching_nodes = [
                node
                for node in self.digraph.nodes
                if self.digraph.nodes[node].get("id", "").startswith(id)
            ]
            if matching_nodes:
                if len(matching_nodes) == 1:
                    return matching_nodes[0]
                else:
                    msg = f"Multiple nodes found with id starting with '{id}': {matching_nodes}."
                    raise ValueError(msg)
            else:
                msg = f"No node found with id starting with '{id}'."
                raise ValueError(msg)
        else:
            for node in self.digraph.nodes:
                if self.digraph.nodes[node].get("id") == id:
                    return node

            msg = f"No node found with id '{id}'."
            raise ValueError(msg)

    def clear_external_settings(self) -> None:
        """
        Clear all 'external_index_setting' attributes from the edges of the graph.

        To be used when the graph is being moved out of interpolative gauge (e.g.
        setting orthogonality center).
        """
        for u, v in self.digraph.edges:
            if "external_index_setting" in self.digraph.edges[u, v]:
                del self.digraph.edges[u, v]["external_index_setting"]

    def setup_leaf_external_settings(self) -> None:
        """
        Give 'external_index_setting' attributes to the edges of leaf nodes.

        Required so that their tensor products can be passed up the tree when placing in
        interpolative gauge.
        """
        leaves = [node for node in self.digraph.nodes if self.digraph.degree[node] == 1]

        for leaf_node in leaves:
            out_edge = list(self.digraph.out_edges(leaf_node))[0]

            # Find the external index of the edge and create a corresponding setting
            index_id = self.digraph.nodes[out_edge[0]]["external_index"]
            index_setting = ExternalIndexSetting.from_physical_index(
                index_id, np.array([0, 1], dtype=int)
            )

            # Equip the edge with this setting
            self.digraph.edges[out_edge]["external_index_setting"] = index_setting

    def find_tree_center(self) -> int:
        """
        Find the center of the tree graph, i.e. the node with the minimum eccentricity.

        (Wraps the helper function).
        """
        return graph_helper.find_tree_center(self.digraph)

    def gauge_center(self) -> int:
        """
        Get the node index of the gauge center of the tree.

        Note the tree center is the current gauge center, not necessarily the one with
        minimum eccentricity.
        """
        center_nodes = [
            node
            for node in self.digraph.nodes
            if self.digraph.nodes[node].get("node_type", "").startswith("center_")
        ]

        if len(center_nodes) != 1:
            msg = "There should be exactly one center node in the graph."
            raise ValueError(msg)

        return center_nodes[0]
