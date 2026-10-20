import uuid

import networkx as nx


def find_tree_center(graph: nx.Graph | nx.DiGraph) -> int:
    """
    Find the center of a tree graph by minimizing eccentricity.

    Parameters
    ----------
    graph : nx.Graph | nx.DiGraph
        The tree graph to find the center of.

    Returns
    -------
    int
        The node index of the tree center.

    """
    if isinstance(graph, nx.DiGraph):
        undirected = graph.to_undirected()
    elif isinstance(graph, nx.Graph):
        undirected = graph
    else:
        msg = "graph must be a networkx Graph or DiGraph"
        raise TypeError(msg)

    # Calculate the eccentricity of each node (maximum distance to any other node)
    eccentricity = nx.eccentricity(undirected)

    # Find the node(s) with minimum eccentricity
    min_ecc = min(eccentricity.values())
    centers = [node for node, ecc in eccentricity.items() if ecc == min_ecc]

    # this should never happen...
    assert len(centers) <= 2, "A tree should have at most two centers"

    # If there's multiple centers, pick the first one arbitrarily
    return centers[0]


def to_digraph(graph: nx.Graph, tree_center: int) -> nx.DiGraph:
    """
    Construct a directed graph (digraph) based off an undirected graph.

    The resulting directed graph will flow towards the given tree center.

    Parameters
    ----------
    graph : nx.Graph
        The original, undirected graph to base the directed graph off of.
    tree_center : int
        The node index of the tree center used to construct the directed graph.

    Returns
    -------
    nx.DiGraph
        A directed graph flowing towards the given tree center node.

    """
    # TODO(Vasco): add check that isn't digraph?
    if not isinstance(graph, nx.Graph):
        msg = "graph must be a networkx Graph"
        raise TypeError(msg)

    if not nx.is_tree(graph):
        msg = "graph must be a tree"
        raise ValueError(msg)

    digraph = nx.DiGraph()

    for parent, child in nx.bfs_edges(graph, tree_center):
        digraph.add_edge(child, parent)

    for node in graph.nodes:
        if graph.nodes[node].get("external_index") is not None:
            digraph.nodes[node]["external_index"] = graph.nodes[node]["external_index"]

    return digraph


def label_graph(digraph: nx.DiGraph, tree_center: int) -> nx.DiGraph:
    """
    Label the physical indices, center, bond IDs, and tensor IDs of a digraph.

    Note: expects leaf nodes to have already been labelled with "external_index".

    Parameters
    ----------
    digraph : nx.DiGraph
        The directed graph to add labels to.
    tree_center : int
        The node index of the tree center of the directed graph.

    Returns
    -------
    nx.DiGraph
        The labelled directed graph.

    """
    # Label the center of the tree
    digraph.nodes[tree_center]["node_type"] = "center_generic"

    # Label the physical qubit indices
    physical_counter = 0
    for node in digraph.nodes:
        if digraph.degree[node] == 1:  # Leaf nodes have degree 1
            index_id = digraph.nodes[node].get("external_index", None)
            assert index_id is not None, "Leaf nodes must have an 'external_index' set."

            digraph.nodes[node]["node_type"] = "physical_index"
            outgoing_edge = list(digraph.out_edges(node))[0]
            digraph.edges[outgoing_edge]["id"] = str(index_id)
            digraph.nodes[node]["id"] = str(index_id)

            # assumed to be a qubit
            digraph.edges[outgoing_edge]["dimension"] = 2
            physical_counter += 1

    # First check the physical indices are labelled correctly and that ids aren't already conflictingly set
    for node in digraph.nodes:
        if digraph.degree[node] == 1:  # Leaf nodes have degree 1 in a tree
            assert digraph.nodes[node].get("node_type") == "physical_index", (
                "Leaf nodes must have a 'node_type' equal to 'physical_index'."
            )
            edge_id = digraph.edges[list(digraph.out_edges(node))[0]].get("id")
            assert edge_id is not None, "Outgoing edge must have an 'id'."
            assert not edge_id.startswith("bond_"), (
                "Outgoing edge 'id' must be a physical index (i.e. doens't start with 'bond_')."
            )
        else:
            assert "id" not in digraph.nodes[node], "Node already has an 'id' set!"
            for _, target in digraph.out_edges(node):
                assert "id" not in digraph.edges[(node, target)], (
                    "Edge already has an 'id' set!"
                )

    # Label the tensor nodes with unique ids
    for node in digraph.nodes:
        # Leaf nodes correspond to physical indices, not tensors
        if digraph.degree[node] != 1:
            label = "tensor_" + str(uuid.uuid4()).replace("-", "")
            digraph.nodes[node]["id"] = label

    # Label the edges (internal bonds) with unique ids
    for edge in digraph.edges:
        if "id" not in digraph.edges[edge]:
            label = "bond_" + str(uuid.uuid4()).replace("-", "")
            digraph.edges[edge]["id"] = label

    return digraph


def polish_graph(graph: nx.Graph) -> nx.DiGraph:
    """
    Canonicalise a generic tree graph.

    Identifies the tree center, labels the graph and converts to a digraph.

    Parameters
    ----------
    graph : nx.Graph
        The initial unlabelled undirected graph.

    Returns
    -------
    nx.DiGraph
        The equivalent directed graph with tree center identified and labels added.

    """
    if not isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
        msg = "graph must be an (undirected) networkx Graph."
        raise TypeError(msg)

    if not nx.is_tree(graph):
        msg = "graph must be a tree"
        raise ValueError(msg)

    # Find the central node
    tree_center = find_tree_center(graph)

    # Set up a directed graph flowing to the center of the tree
    directed_graph = to_digraph(graph, tree_center)

    # Label the graph (external indices, bond IDs, tensor IDs, center node)
    return label_graph(directed_graph, tree_center)


def build_binary_tree(n: int, root_node: int = 0) -> tuple[nx.Graph, int]:
    """
    Build a binary tree with exactly n nodes using a level-order approach.

    Parameters
    ----------
    n : int
        Number of nodes of the binary tree.
    root_node : int
        Index of the node to use for the binary tree root.

    Returns
    -------
    graph : nx.Graph
        The graph representation of the binary tree.
    # TODO(Vasco): check what this node index represent
    node : int


    """
    if not isinstance(n, int) or n <= 0:
        msg = "n must be a positive integer."
        raise ValueError(msg)

    if not isinstance(root_node, int) or n < 0:
        msg = "root_node must be a nonnegative integer."
        raise ValueError(msg)

    graph = nx.Graph()
    graph.add_node(root_node)

    node_counter = root_node + 1
    current_parent = root_node
    children_added = 0

    while node_counter - root_node < n:
        # Add left child if possible
        if node_counter - root_node < n:
            graph.add_node(node_counter)
            graph.add_edge(current_parent, node_counter)
            node_counter += 1
            children_added += 1

        # Add right child if possible
        if node_counter - root_node < n:
            graph.add_node(node_counter)
            graph.add_edge(current_parent, node_counter)
            node_counter += 1
            children_added += 1

        # Move to next parent node
        if children_added >= 2:
            current_parent += 1
            children_added = 0

    return graph, node_counter + 1
