import uuid

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering

from treetensorcomp.quantics.functions import QuanticsFunction
from treetensorcomp.quantics.index_settings import ExternalIndexSetting, IndexID
from treetensorcomp.structural_optimization import entanglement_metrics
from treetensorcomp.tensor_networks.tensors import Tensor
from treetensorcomp.tensor_networks.tree_graphs import TTNGraph
from treetensorcomp.tensor_networks.tree_tensor_network import TreeTensorNetwork

_2qubit_basis_settings = ((0, 0), (0, 1), (1, 0), (1, 1))


def _subdivide_cluster(
    graph: nx.Graph,
    index_ids: list[IndexID],
    affinity_matrix: NDArray[np.float64],
    branching_factor: int,
    parent_node_idx: str | None = None,
) -> None:
    """
    Recursively subdivide cluster of indices into sub-clusters.

    Base case: when the cluster has only one index, add a leaf node to the graph with
    this index as an attribute.

    Recursive case: partition the indices into sub-clusters using spectral clustering
    and then recursively call this function on each sub-cluster to further subdivide it
    until the base case is reached.

    See also `divisive_ttn_graph()`.

    Parameters
    ----------
    graph : nx.Graph
        The original graph to subdivide.
    index_ids : list[IndexID]
        Indices contained in the graph.
    affinity_matrix : NDArray[np.float64]
        Square matrix quantifying the pairwise correlation/ entanglement metrics
        between each pair of qubits. Must be dimensionally consistent with the
        number of external indices.
    branching_factor : int, optional
        The number of subcluster to create at each level during the recursive
        clustering procedure (default: 2).
    parent_node_idx : str, optional
        Optional node index to set as the parent node of the tree.

    """
    # Check inputs
    assert isinstance(affinity_matrix, np.ndarray), (
        "affinity_matrix must be a numpy ndarray."
    )
    assert isinstance(index_ids, list) and all(
        isinstance(idx, IndexID) for idx in index_ids
    ), "index_ids must be a list of IndexID instances."
    assert len(index_ids) == affinity_matrix.shape[0], (
        "Index IDs and affinity matrix must match in size."
    )
    assert affinity_matrix.shape[0] == affinity_matrix.shape[1], (
        "Affinity matrix must be square."
    )

    # Number of clusters must not exceed the number of indices
    n_clusters = min(branching_factor, len(index_ids))

    # Add to the graph
    current_node_idx = str(uuid.uuid4())
    graph.add_node(current_node_idx)
    if parent_node_idx is not None:
        graph.add_edge(parent_node_idx, current_node_idx)

    # Terminal case: if the cluster is size 1, then we've reached a leaf node (physical index)
    if len(index_ids) == 1:
        graph.nodes[current_node_idx]["external_index"] = index_ids[0]

    # Otherwise, recursively subdivide
    else:
        # Trivial case: if the number of clusters equals the number of indices to divide, each index is its own cluster
        if n_clusters == len(index_ids):
            clustering_labels = np.arange(len(index_ids))

        # Else, we should try using spectral clustering to partition the indices
        else:
            # Try spectral clustering. It might fail if the affinity matrix is poorly conditioned.
            try:
                clustering_labels = (
                    SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
                    .fit(affinity_matrix)
                    .labels_
                )

            # sklearn will throw an exception if the affinity matrix is poorly conditioned, so let's catch this
            except ValueError as e:
                print(
                    f"Near-singular affinity matrix encountered. Fallback to arbitary partition due to error: {e}"
                )
                print("This is nothing to worry about!")
                midpoint = len(index_ids) // 2
                clustering_labels = np.concatenate(
                    [np.zeros(midpoint), np.ones(len(index_ids) - midpoint)]
                )

        # Create sub-clusters based on the clustering labels
        clusters = []
        masks = []
        affinity_matrices = []
        for idx_cluster in range(n_clusters):
            clusters.append(
                [
                    index_id
                    for index_id, label in zip(index_ids, clustering_labels)
                    if label == idx_cluster
                ]
            )
            mask = clustering_labels == idx_cluster
            masks.append(mask)
            affinity_matrices.append(affinity_matrix[mask][:, mask])

        # Recursively subdivide these new sub-clusters
        for idx_cluster, (sub_cluster, mask, sub_affinity_matrix) in enumerate(
            zip(clusters, masks, affinity_matrices)
        ):
            _subdivide_cluster(
                graph,
                sub_cluster,
                sub_affinity_matrix,
                branching_factor,
                parent_node_idx=current_node_idx,
            )


class Clusterer:
    """TTN structure optimizer based on entanglement/ affinity metrics."""

    quantics_function: QuanticsFunction
    external_indices: list[IndexID]
    precisions: list[int]
    monte_carlo_samples: int | None
    ttn: TreeTensorNetwork | None
    projector_density_matrices: dict[tuple, Tensor] | None
    reduced_density_matrices: dict[tuple, Tensor] | None

    def __init__(
        self,
        quantics_function: QuanticsFunction,
        precisions: int | list[int],
        monte_carlo_samples: int | None = None,
        tree_tensor_network: TreeTensorNetwork | None = None,
    ) -> None:
        """
        Initialize the Clusterer using a given QuanticsFunction.

        At least one of `monte_carlo_samples` or `tree_tensor_network` must be set.
        If both are set the TTN will be used and the Monte Carlo setting will be
        ignored.

        Parameters
        ----------
        quantics_function : QuanticsFunction
            The function to encode in the tree tensor network.
        precisions : int | list[int]
            Precision in number of bits to be used for the representation of each
            variable.
        monte_carlo_samples : int, optional
            Number of samples to use for Monte Carlo calculation of affinities.
        tree_tensor_network : TreeTensorNetwork, optional
            A TreeTensorNetwork representation of the function, used to calculate
            affinities.

        """
        assert isinstance(quantics_function, QuanticsFunction), (
            "quantics_function must be a QuanticsFunction instance"
        )
        if isinstance(precisions, int):
            precisions = [precisions for _ in quantics_function.variable_names]
        else:
            assert isinstance(precisions, list) and all(
                isinstance(p, int) and p > 0 for p in precisions
            ), "precisions must be an integer or a list of positive integers"
            assert len(precisions) == len(quantics_function.variable_names), (
                "precisions must match the number of variable names in the quantics function"
            )
        if monte_carlo_samples is not None:
            assert isinstance(monte_carlo_samples, int) and monte_carlo_samples > 0, (
                "monte_carlo_samples must be a positive integer"
            )
        # Generate list of all external indices up to supplied maximum precisions
        self.external_indices = [
            IndexID(var_name, bit)
            for var_name, prec in zip(quantics_function.variable_names, precisions)
            for bit in range(1, prec + 1)
        ]

        self.precisions = precisions
        self.quantics_function = quantics_function
        self.monte_carlo_samples = monte_carlo_samples
        self.ttn = tree_tensor_network

        if self.ttn is not None and self.ttn.gauge != "orthogonality":
            # place in isometric gauge
            center_node = self.ttn.ttn_graph.find_tree_center()
            self.ttn.move_gauge_centre(center_node, gauge_type="orthogonality")
        self.projector_density_matrices = None
        self.reduced_density_matrices = None  # uninitializedcaches for density matrices

    def build_projectors(self) -> None:
        """
        Build and cache the marginalized state for each pair of qubits.

        The marginalized state is a rank-1 projector onto the state of these two qubits,
        obtained by coherent averaging.
        """
        print(
            "Warning: this may take a moderate amount of time to run, while the projector density matrices are built."
        )
        self.projector_density_matrices = {}

        # Loop over unique pairs of external indices
        for a in range(len(self.external_indices)):
            for b in range(a + 1, len(self.external_indices)):
                idx_a = self.external_indices[a]
                idx_b = self.external_indices[b]

                projector_tensor = self._build_projector(idx_a, idx_b)

                # Store the RDM in the cache, both (a,b) and (b,a)
                self.projector_density_matrices[(a, b)] = projector_tensor
                self.projector_density_matrices[(b, a)] = projector_tensor

    def build_rdms(self) -> None:
        """Build and cache reduced density matrices for each pair of qubits."""
        print(
            "Warning: this may take a moderate amount of time to run, while the reduced density matrices are built."
        )
        self.reduced_density_matrices = {}

        # Loop over unique pairs of external indices
        for a in range(len(self.external_indices)):
            for b in range(a + 1, len(self.external_indices)):
                idx_a = self.external_indices[a]
                idx_b = self.external_indices[b]

                rdm_tensor = self._build_rdm(idx_a, idx_b)

                # Store the RDM in the cache, both (a,b) and (b,a)
                self.reduced_density_matrices[(a, b)] = rdm_tensor
                self.reduced_density_matrices[(b, a)] = rdm_tensor

    def recall_projector(self, a: int, b: int) -> Tensor:
        """
        Recall a projector density matrix for index pairing (a, b) from the cache.

        Builds and caches the projectors if this hasn't already been done.

        Parameters
        ----------
        a, b : int
            Qubit pair to get the projector for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            The two qubit projector density matrix.

        """
        assert isinstance(a, int) and isinstance(b, int), (
            "a and b must be integers representing indices in the external_indices list"
        )
        assert 0 <= a < len(self.external_indices) and 0 <= b < len(
            self.external_indices
        ), "a and b must be valid indices in the external_indices list"

        if self.projector_density_matrices is None:
            self.build_projectors()

        return self.projector_density_matrices[(a, b)]

    def recall_rdm(self, a: int, b: int) -> Tensor:
        """
        Recall a reduced density matrix for index pairing (a, b) from the cache.

        Builds and caches the density matrices if this hasn't already been done.

        Parameters
        ----------
        a, b : int
            Qubit pair to get the reduced density matrix for. Must correspond to indices
            in the Clusterer's set of external indices.

        Returns
        -------
        Tensor
            The two qubit reduced density matrix.

        """
        assert isinstance(a, int) and isinstance(b, int), (
            "a and b must be integers representing indices in the external_indices list"
        )
        assert 0 <= a < len(self.external_indices) and 0 <= b < len(
            self.external_indices
        ), "a and b must be valid indices in the external_indices list"

        if self.reduced_density_matrices is None:
            self.build_rdms()

        return self.reduced_density_matrices[(a, b)]

    def _build_projector(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Construct a single projector density matrix.

        Calls the tensor contraction or Monte Carlo method according to whether `ttn`
        or `monte_carlo_samples` was set.

        See also `build_projectors()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the projector for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit projector density matrix.

        """
        assert isinstance(idx_a, IndexID) and isinstance(idx_b, IndexID), (
            "idx_a and idx_b must be IndexID instances"
        )
        assert idx_a != idx_b, "idx_a and idx_b must be different indices"
        assert idx_a in self.external_indices and idx_b in self.external_indices, (
            "idx_a and idx_b must be in the external indices of the quantics function"
        )

        if self.ttn is not None:
            return self._projector_ttn(idx_a, idx_b)

        if self.monte_carlo_samples is not None:
            return self._projector_monte_carlo(idx_a, idx_b)

        # If we reach here then something has gone wrong
        msg = "Clusterer must be initialized with either a tree tensor network or a number of Monte Carlo samples for density matrix calculation"
        raise ValueError(msg)

    def _build_rdm(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Construct a single reduced density matrix.

        Calls the tensor contraction or Monte Carlo method according to whether `ttn`
        or `monte_carlo_samples` was set.

        See also `build_rdms()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the density matrix for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit reduced density matrix.

        """
        assert isinstance(idx_a, IndexID) and isinstance(idx_b, IndexID), (
            "idx_a and idx_b must be IndexID instances"
        )
        assert idx_a != idx_b, "idx_a and idx_b must be different indices"
        assert idx_a in self.external_indices and idx_b in self.external_indices, (
            "idx_a and idx_b must be in the external indices of the quantics function"
        )

        if self.ttn is not None:
            return self._rdm_ttn(idx_a, idx_b)

        if self.monte_carlo_samples is not None:
            return self._rdm_monte_carlo(idx_a, idx_b)

        # If we reach here then something has gone wrong
        msg = "Clusterer must be initialized with either a tree tensor network or a number of Monte Carlo samples for density matrix calculation"
        raise ValueError(msg)

    def _projector_monte_carlo(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Calculate a projector density matrix by coherent averaging.

        See also `_build_projector()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the projector for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit projector density matrix.

        """
        randomized_index_settings = ExternalIndexSetting.random_index_setting(
            self.quantics_function.variable_names,
            self.precisions,
            count=self.monte_carlo_samples,
        )

        # Use these Monte Carlo sample points to generate the 4x4 two-qubit density matrix
        density_matrix = np.empty((4, 4), dtype=complex)

        # Populate the density matrix (only need to do one half of off-diagonal due to Hermiticity)
        for row_idx in range(4):
            for col_idx in range(row_idx + 1):
                # Evaluate the ket half
                randomized_index_settings.overwrite_indices(
                    (idx_a, idx_b), _2qubit_basis_settings[row_idx]
                )

                ket_factor = np.mean(
                    self.quantics_function.evaluate_index_setting(
                        randomized_index_settings
                    )
                )

                # Evaluate the bra half
                randomized_index_settings.overwrite_indices(
                    (idx_a, idx_b), _2qubit_basis_settings[col_idx]
                )

                bra_factor = np.mean(
                    self.quantics_function.evaluate_index_setting(
                        randomized_index_settings
                    )
                )

                # Populate the density matrix
                density_matrix[row_idx, col_idx] = ket_factor * bra_factor.conjugate()
                density_matrix[col_idx, row_idx] = density_matrix[
                    row_idx, col_idx
                ].conjugate()

        # Normalize the density matrix (it won't be normalized due to unnormalized distribution and Monte Carlo noise)
        density_matrix = density_matrix / np.trace(density_matrix)

        return Tensor(
            np.reshape(density_matrix, (2, 2, 2, 2)),
            (str(idx_a), str(idx_b), str(idx_a) + "*", str(idx_b) + "*"),
        )

    def _rdm_monte_carlo(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Calculate a reduced density matrix by coherent averaging.

        See also `_build_rdm()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the density matrix for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit reduced density matrix.

        """
        randomized_index_settings = ExternalIndexSetting.random_index_setting(
            self.quantics_function.variable_names,
            self.precisions,
            count=self.monte_carlo_samples,
        )

        # Use these Monte Carlo sample points to generate the 4x4 two-qubit density matrix
        density_matrix = np.empty((4, 4), dtype=complex)

        # Populate the density matrix (only need to do one half of off-diagonal due to Hermiticity)
        for row_idx in range(4):
            for col_idx in range(row_idx + 1):
                # Evaluate entry (upper-diagonal)
                randomized_index_settings.overwrite_indices(
                    (idx_a, idx_b), _2qubit_basis_settings[row_idx]
                )

                ket_samples = self.quantics_function.evaluate_index_setting(
                    randomized_index_settings
                )

                randomized_index_settings.overwrite_indices(
                    (idx_a, idx_b), _2qubit_basis_settings[col_idx]
                )

                bra_samples = np.conjugate(
                    self.quantics_function.evaluate_index_setting(
                        randomized_index_settings
                    )
                )

                entry = np.mean(ket_samples * bra_samples)

                # Populate the density matrix
                density_matrix[row_idx, col_idx] = entry
                density_matrix[col_idx, row_idx] = entry.conjugate()

        # Normalize the density matrix (it won't be normalized due to unnormalized distribution and Monte Carlo noise)
        density_matrix = density_matrix / np.trace(density_matrix)

        return Tensor(
            np.reshape(density_matrix, (2, 2, 2, 2)),
            (str(idx_a), str(idx_b), str(idx_a) + "*", str(idx_b) + "*"),
        )

    def _projector_ttn(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Calculate a projector density matrix by contracting the TreeTensorNetwork.

        See also `_build_projector()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the projector for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit projector density matrix.

        """
        # Pick out the non-open indices
        averaged_indices = [
            idx for idx in self.external_indices if idx != idx_a and idx != idx_b
        ]

        # List of tensors to contract: "cap" |+> tensor for averaging over other indices, plus the TTN tensors
        cap_tensors = [
            Tensor(
                np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=float),
                [str(idx)],
                id="plus_" + str(idx),
            )
            for idx in averaged_indices
        ]

        tensor_list = cap_tensors + list(self.ttn.tensors.values())

        # Autocontract with quimb
        quimb_tensor_list = []
        for tensor in tensor_list:
            quimb_tensor = qtn.Tensor(
                tensor.tensor_ndarray, inds=tensor.indices, tags={tensor.id}
            )
            quimb_tensor_list.append(quimb_tensor)

        result = qtn.TensorNetwork(quimb_tensor_list).contract()

        # Build the rank-1 projector from the averaged state
        averaged_state = Tensor(result.data, result.inds, id="rdm")
        projector = averaged_state.adjoint() @ averaged_state

        # Renormalize the projector
        density_matrix, shape_dict = projector.matricize(
            [str(idx_a) + "*", str(idx_b) + "*"], [str(idx_a), str(idx_b)]
        )

        assert np.max(density_matrix - np.conj(density_matrix).T) < 1e-4, (
            "Density matrix must be approximately Hermitian, but encountered error"
        )

        # Make the density matrix explicitly symmetric by copying upper triangle to lower triangle
        i_upper = np.triu_indices_from(density_matrix, k=1)
        density_matrix[i_upper[1], i_upper[0]] = np.conj(density_matrix[i_upper])

        # Renormalize the density matrix
        density_matrix = density_matrix / np.trace(density_matrix)

        projector_adjusted = Tensor(
            density_matrix.reshape(2, 2, 2, 2),
            [str(idx_a), str(idx_b), str(idx_a) + "*", str(idx_b) + "*"],
            id="projector_adjusted",
        )

        return projector_adjusted

    def _rdm_ttn(self, idx_a: IndexID, idx_b: IndexID) -> Tensor:
        """
        Calculate a reduced density matrix by contracting the TreeTensorNetwork.

        See also `_build_rdm()`.

        Parameters
        ----------
        idx_a, idx_b : IndexID
            Indices to build the density matrix for. Must correspond to indices in the
            Clusterer's set of external indices.

        Returns
        -------
        Tensor
            A new two qubit reduced density matrix.

        """
        if self.ttn.gauge != "orthogonality":
            # place in isometric gauge
            center_node = self.ttn.ttn_graph.find_tree_center()
            self.ttn.move_gauge_centre(
                center_node,
                gauge_type="orthogonality",
                gauge_parameters={"cutoff_rtol": 1e-10},
            )

        tensor_list = []

        contraction_graph = self.ttn.ttn_graph.digraph
        leaf_nodes = [
            node
            for node in contraction_graph.nodes
            if contraction_graph.in_degree(node) == 0
        ]

        lightcone_tensors = []
        lightcone_indices = []
        for idx_leaf in [idx_a, idx_b]:
            # Find the lightcone path from the leaf index to the center tensor
            start_node = [
                node
                for node in leaf_nodes
                if contraction_graph.nodes[node]["id"] == str(idx_leaf)
            ][0]
            end_node = [
                node
                for node in contraction_graph.nodes
                if contraction_graph.nodes[node]["node_type"] == "center_orthogonality"
            ][0]

            path = nx.shortest_path(
                contraction_graph, source=start_node, target=end_node
            )

            # Record the tensors and internal indices traversed along this lightcone
            lightcone_tensors += [
                contraction_graph.nodes[node]["id"] for node in path[1:]
            ]
            lightcone_indices += [
                contraction_graph.edges[(path[i], path[i + 1])]["id"]
                for i in range(1, len(path) - 1)
            ]

        # Get rid of duplicates in the tensors and indices (in case the light cones overlap)
        lightcone_tensors = list(dict.fromkeys(lightcone_tensors))
        lightcone_indices = list(dict.fromkeys(lightcone_indices))

        tensor_list = []
        # Now build a set of tensors to feed to the contractor
        # N.B.: We want the internal indices traversed along the lightcone to not be summed over, and the rest to be contracted between the two TTNs
        for tensor_id in lightcone_tensors:
            # Append the tensor to the list (ket copy)
            ket_tensor = self.ttn.tensors[tensor_id]
            tensor_list.append(ket_tensor)
            # Conjugate internal indices along the lightcone path so that they're not summed over by the contractor
            bra_indices = [
                idx
                if (
                    idx not in lightcone_indices
                    and idx != str(idx_a)
                    and idx != str(idx_b)
                )
                else idx + "*"
                for idx in ket_tensor.indices
            ]
            # Append the bra tensor to the list
            tensor_list.append(
                Tensor(
                    np.conjugate(ket_tensor.tensor_ndarray),
                    indices=bra_indices,
                    id=ket_tensor.id + "_adjoint",
                )
            )

        quimb_tensor_list = []
        for tensor in tensor_list:
            quimb_tensor = qtn.Tensor(
                tensor.tensor_ndarray, inds=tensor.indices, tags={tensor.id}
            )
            quimb_tensor_list.append(quimb_tensor)

        # quimb autocontract
        result = qtn.TensorNetwork(quimb_tensor_list).contract()
        rdm = Tensor(result.data, result.inds, id="rdm")
        density_matrix, shape_dict = rdm.matricize(
            [str(idx_a) + "*", str(idx_b) + "*"], [str(idx_a), str(idx_b)]
        )

        assert np.max(density_matrix - np.conj(density_matrix).T) < 1e-4, (
            "Density matrix must be approximately Hermitian"
        )

        # Make the density matrix explicitly symmetric by copying upper triangle to lower triangle
        i_upper = np.triu_indices_from(density_matrix, k=1)
        density_matrix[i_upper[1], i_upper[0]] = np.conj(density_matrix[i_upper])

        # Renormalize the density matrix
        density_matrix = density_matrix / np.trace(density_matrix)
        rdm_adjusted = Tensor(
            density_matrix.reshape(2, 2, 2, 2),
            [str(idx_a), str(idx_b), str(idx_a) + "*", str(idx_b) + "*"],
            id="rdm_adjusted",
        )
        return rdm_adjusted

    def projector_entropy_matrix(self) -> NDArray[np.float64]:
        """Construct the boolean Fourier entropy matrix."""
        projector_entropy_matrix = np.zeros(
            (len(self.external_indices), len(self.external_indices)), dtype=float
        )

        # Loop over unique pairs of external indices
        for a in range(len(self.external_indices)):
            for b in range(a + 1, len(self.external_indices)):
                idx_a = self.external_indices[a]
                idx_b = self.external_indices[b]

                projector_tensor = self.recall_projector(a, b)

                projector_entropy_matrix[a, b] = (
                    entanglement_metrics.mutual_information(
                        projector_tensor, [idx_a, idx_b]
                    )
                )

                projector_entropy_matrix[b, a] = projector_entropy_matrix[a, b]

        # Divide by 2 since I = S(A) + S(B) - S(AB) and S(AB) = 0
        # and S(A) = S(B) = entanglement entropy for pure states
        return projector_entropy_matrix / 2

    def mutual_information_matrix(self) -> NDArray[np.float64]:
        """Construct the mutual information matrix."""
        mutual_information_matrix = np.zeros(
            (len(self.external_indices), len(self.external_indices)), dtype=float
        )
        # Loop over unique pairs of external indices
        for a in range(len(self.external_indices)):
            for b in range(a + 1, len(self.external_indices)):
                idx_a = self.external_indices[a]
                idx_b = self.external_indices[b]

                rdm_tensor = self.recall_rdm(a, b)
                mutual_information_matrix[a, b] = (
                    entanglement_metrics.mutual_information(rdm_tensor, [idx_a, idx_b])
                )

                # Symmetric distance matrix
                mutual_information_matrix[b, a] = mutual_information_matrix[a, b]

        return mutual_information_matrix

    def entanglement_formation_matrix(self) -> NDArray[np.float64]:
        """Construct the entanglement of formation matrix."""
        entanglement_formation_matrix = np.zeros(
            (len(self.external_indices), len(self.external_indices)), dtype=float
        )
        # Loop over unique pairs of external indices
        for a in range(len(self.external_indices)):
            for b in range(a + 1, len(self.external_indices)):
                idx_a = self.external_indices[a]
                idx_b = self.external_indices[b]

                rdm_tensor = self.recall_rdm(a, b)

                entanglement_formation_matrix[a, b] = (
                    entanglement_metrics.entanglement_formation(
                        rdm_tensor, [idx_a, idx_b]
                    )
                )

                # Symmetric distance matrix
                entanglement_formation_matrix[b, a] = entanglement_formation_matrix[
                    a, b
                ]

        return entanglement_formation_matrix

    def divisive_ttn_graph(
        self,
        affinity_matrix: NDArray[np.float64],
        branching_factor: int = 2,
    ) -> TTNGraph:
        """
        Build a TTNGraph minimizing the distance between strongly entangled pairs.

        This minimization is performed by recursively dividing clusters of indices into
        sub-clusters based on the spectral clustering of the affinity matrix, yielding
        a tensor network with reduced bond dimension.

        Parameters
        ----------
        affinity_matrix : NDArray[np.float64]
            Square matrix quantifying the pairwise correlation/ entanglement metrics
            between each pair of qubits. Must be dimensionally consistent with the
            number of external indices of the Clusterer.
        branching_factor : int, optional
            The number of subcluster to create at each level during the recursive
            clustering procedure (default: 2).

        Returns
        -------
        TTNGraph
            The tree tensor network graph resulting from clustering.

        """
        # Check inputs and set up
        assert isinstance(branching_factor, int) and branching_factor > 0, (
            "branching_factor must be a positive integer"
        )
        assert isinstance(affinity_matrix, np.ndarray), (
            "affinity_matrix must be a numpy ndarray"
        )

        assert affinity_matrix.shape == (
            len(self.external_indices),
            len(self.external_indices),
        ), "affinity_matrix wrong shape"

        graph = nx.Graph()

        # Start the recursive subdivision of the clusters
        _subdivide_cluster(
            graph,
            self.external_indices,
            affinity_matrix,
            branching_factor,
            parent_node_idx=None,
        )

        # If the branching factor is 2, there will be a degree-2 node causing rank
        # bottleneck for TCI. Remove it and connect its neighbours.
        if branching_factor == 2:
            degree_2_nodes = [node for node, degree in graph.degree() if degree == 2]
            assert len(degree_2_nodes) == 1, "Only one node should have degree 2."

            node_to_remove = degree_2_nodes[0]
            neighbours = list(graph.neighbors(node_to_remove))
            graph.add_edge(neighbours[0], neighbours[1])
            graph.remove_node(node_to_remove)

        # Relabel the graph indices with integers
        graph = nx.convert_node_labels_to_integers(graph)

        return TTNGraph.from_unpolished_graph(graph, self.external_indices)

    def plot_affinity_matrix(
        self,
        affinity_matrix: NDArray[np.float64],
        label: str = "affinity",
    ) -> None:
        """
        Prettily-plot a given affinity matrix.

        Parameters
        ----------
        affinity_matrix : NDArray[np.float64]
            The affinity matrix data to plot.
        label : str, optional
            An optional label to use in the plot.

        """
        plt.matshow(np.log(affinity_matrix))
        plt.colorbar(label="log " + label)

        tick_positions = []
        tick_labels = []
        pos_counter = 0

        for idx_var, variable_name in enumerate(self.quantics_function.variable_names):
            precision = self.precisions[idx_var]
            tick_positions.append(pos_counter + precision / 2)
            tick_labels.append(variable_name)
            pos_counter += precision

        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)
