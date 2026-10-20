from collections.abc import Iterator

import networkx as nx
from tqdm import tqdm

from treetensorcomp.quantics.functions import QuanticsFunction
from treetensorcomp.quantics.index_settings import ExternalIndexSetting
from treetensorcomp.tensor_networks.tensors import Tensor
from treetensorcomp.tensor_networks.tree_tensor_network import TreeTensorNetwork


class TCISweepPlan:
    """Sweep structure for Tensor Cross Interpolation of a TreeTensorNetwork."""

    node_traversal_path: list[tuple]
    ttn_name: str


    # TODO: There is a bug here that causes a (root,root) step to appear in the path if the last non-root step is a child of the root, I think?
    def __init__(self, ttn: TreeTensorNetwork, sweep_root_node: int) -> None:
        """
        Plan the TCI sweep for a tree tensor network starting from a given root node.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            The TreeTensorNetwork to design the TCI sweep for.
        sweep_root_node : int
            The node to start each sweep from.

        """
        # Get node indices visited in order, excluding physical indices (does not include path traversed between each node)
        undirected_graph = ttn.ttn_graph.digraph.to_undirected()
        node_traversal_order = [
            node
            for node in nx.dfs_preorder_nodes(undirected_graph, source=sweep_root_node)
            if ttn.ttn_graph.digraph.nodes[node]["node_type"] != "physical_index"
        ]

        # Make sure to include a path back to the root node to reset ready for another sweep
        node_traversal_order.append(sweep_root_node)

        node_traversal_path = []

        # Include paths between nodes when jumping (sub-)branches
        for node_idx in range(len(node_traversal_order) - 1):
            # Get the path between the current node and the next node in the traversal order
            node = node_traversal_order[node_idx]
            next_node = node_traversal_order[node_idx + 1]
            path_at_this_step = nx.shortest_path(
                undirected_graph, source=node, target=next_node
            )

            # Create (a,b) sub-step pairs when moving a -> b
            sub_steps = [
                (path_at_this_step[substep_idx], path_at_this_step[substep_idx + 1])
                for substep_idx in range(len(path_at_this_step) - 1)
            ]
            node_traversal_path.extend(sub_steps)

        # The above traversal path might have "turn-backs" e.g. (a->b) followed by (b->a)
        # This would just repeat the same TCI update step, so we can remove the first element (as it is the second that gives us the correct gauge to continue the sweep)
        keep_step = [
            not (
                node_traversal_path[i][0] == node_traversal_path[i + 1][1]
                and node_traversal_path[i][1] == node_traversal_path[i + 1][0]
            )
            for i in range(len(node_traversal_path) - 1)
        ]

        # Mask the list with keep_step
        node_traversal_path = [
            node_traversal_path[i]
            for i in range(len(node_traversal_path) - 1)
            if keep_step[i]
        ]

        # Because of the indexing of the sub-step paths, we need to add the last step manually
        last_node = node_traversal_path[-1][1]
        if last_node != sweep_root_node:
            node_traversal_path.append((last_node, sweep_root_node))

        self.node_traversal_path = node_traversal_path
        self.ttn_name = ttn.name

    def __iter__(self) -> Iterator[tuple]:
        """Return an iterator over the sweep plan's node traversal path."""
        return iter(self.node_traversal_path)

    def __len__(self) -> int:
        """Return the number of nodes visited in the plan's node traversal path."""
        return len(self.node_traversal_path)


class Interpolator:
    """TODO docstring."""

    ttn : TreeTensorNetwork
    quantics_function: QuanticsFunction
    error_tol : float
    sweep_plan : TCISweepPlan
    sweep_count : int

    def __init__(
        self,
        ttn: TreeTensorNetwork,
        quantics_function: QuanticsFunction,
        error_tol: float = 1e-12,
    ) -> None:
        """
        Construct an approximate interpolator for a function using TCI.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            The tree tensor network to evaluate the Tensor Cross Interpolation over.
        quantics_function : QuanticsFunction
            The function for the interpolated tensor to approximate.
        error_tol : float
            The precision of the interpolated function.

        """
        assert isinstance(ttn, TreeTensorNetwork), "ttn must be a TreeTensorNetwork"
        self.ttn = ttn

        assert isinstance(quantics_function, QuanticsFunction), (
            "quantics_function must be a QuanticsFunction instance"
        )
        self.quantics_function = quantics_function

        # TODO: probably do some sort of checks that quantics_function has the right variables etc w.r.t. the ttn's indices
        self.error_tol = error_tol

        sweep_root_node = self.ttn.ttn_graph.find_tree_center()
        self.sweep_plan = TCISweepPlan(self.ttn, sweep_root_node)

        # Place TTN in interpolative gauge
        self.ttn.move_gauge_centre(
            sweep_root_node,
            gauge_type="interpolative",
            gauge_parameters={"relative_error": self.error_tol},
        )

        self.sweep_count = 0

    def tensor_pairs(self, compact: bool = True) -> list[str] | list[tuple[str, str]]:
        """
        Get the node pairs updated at each step of the sweep plan.

        Parameters
        ----------
        compact : bool, optional
            Whether to return a compact representation containing only the first three
            characters of each tensor ID in the output strings (default: True).

        Returns
        -------
        list[str] | list[tuple[str, str]]
            Compact or complete list of node pairs.

        """
        if compact:
            return [
                f"{id_1[7:10]}... -> {id_2[7:10]}..."
                for id_1, id_2 in self.tensor_pairs(compact=False)
            ]

        return [
            (
                self.ttn.ttn_graph.digraph.nodes[back_node]["id"],
                self.ttn.ttn_graph.digraph.nodes[forward_node]["id"],
            )
            for back_node, forward_node in self.sweep_plan
        ]

    def _tci_update(self, step_number: int) -> None:
        """
        Perform a given step (2-site optimization) within a TCI sweep.

        Parameters
        ----------
        step_number : int
            The index of the step in the TCI sweep plan to be evaluated.

        """
        # Get the current step in the sweep plan
        back_node, forward_node = self.sweep_plan.node_traversal_path[step_number]

        # Collect and check the edges of the nodes we're working on
        incoming_edges_back = list(self.ttn.ttn_graph.digraph.in_edges(back_node))
        incoming_edges_forward = list(self.ttn.ttn_graph.digraph.in_edges(forward_node))
        outgoing_edges = list(self.ttn.ttn_graph.digraph.out_edges(back_node)) + list(
            self.ttn.ttn_graph.digraph.out_edges(forward_node)
        )
        assert len(outgoing_edges) == 1, (
            "The only outgoing edge between the two nodes should be the bond we're working on, i.e. the one that connects them."
        )

        # The edge corresponding to the bond we're working on is in one of the lists (direction depending on whether we just skipped a turnback). Get rid of it
        for edge_list in (incoming_edges_back, incoming_edges_forward):
            edge_list[:] = [
                e
                for e in edge_list
                if e != (back_node, forward_node) and e != (forward_node, back_node)
            ]

        # Verify no duplicate edges (this shouldn't ever happen, but just in case)
        incoming_edges_total = incoming_edges_back + incoming_edges_forward
        assert len(incoming_edges_total) == len(set(incoming_edges_total)), (
            "incoming_edges contains duplicate elements"
        )

        # Collect the index IDs and tensor dimensions (sans the bond we're working on) of the back and forward tensors
        back_tensor_indices = [
            self.ttn.ttn_graph.digraph.edges[edge]["id"] for edge in incoming_edges_back
        ]

        forward_tensor_indices = [
            self.ttn.ttn_graph.digraph.edges[edge]["id"]
            for edge in incoming_edges_forward
        ]

        back_tensor_shape = [
            self.ttn.ttn_graph.digraph.edges[edge]["dimension"]
            for edge in incoming_edges_back
        ]

        forward_tensor_shape = [
            self.ttn.ttn_graph.digraph.edges[edge]["dimension"]
            for edge in incoming_edges_forward
        ]

        # Create a tensor'ed index setting for the 2-site \Pi tensor
        index_settings = [
            self.ttn.ttn_graph.digraph.edges[edge]["external_index_setting"]
            for edge in incoming_edges_total
        ]
        total_index_setting = ExternalIndexSetting.tensor_parent_index_settings(
            index_settings
        )

        # Evaluate the 2-site \Pi tensor with the current index setting
        pi_tensor_flattened = self.quantics_function.evaluate_index_setting(
            total_index_setting, check_var_ids=False
        )

        # Reshape it back to the correct shape
        pi_tensor = Tensor(
            pi_tensor_flattened.reshape(*back_tensor_shape, *forward_tensor_shape),
            back_tensor_indices + forward_tensor_indices,
            id="Pi",
        )

        # Split the tensor with the interpolative decomposition. The skeleton tensor goes forward (row indices).
        skeleton_tensor, interpolation_tensor, interpolated_column_indices = (
            pi_tensor.interpolative_decomposition(
                forward_tensor_indices,
                back_tensor_indices,
                relative_error=self.error_tol,
            )
        )

        # Rename the tensors to match those they're replacing
        back_tensor_name = self.ttn.ttn_graph.digraph.nodes[back_node]["id"]
        forward_tensor_name = self.ttn.ttn_graph.digraph.nodes[forward_node]["id"]
        skeleton_tensor.rename_self(forward_tensor_name)
        interpolation_tensor.rename_self(back_tensor_name)

        # Rename the bond ID to match the original bond (though bond dimension might change below)
        interpdecomp_bond_id = interpolation_tensor.indices[0]
        assert interpdecomp_bond_id.startswith("interpdecomp_"), (
            "This index should originate from the interpolative decomposition"
        )

        if self.ttn.ttn_graph.digraph.has_edge(forward_node, back_node):
            bond_edge = (forward_node, back_node)
        else:
            bond_edge = (back_node, forward_node)

        index_relabel_map = {
            interpdecomp_bond_id: self.ttn.ttn_graph.digraph.edges[bond_edge]["id"]
        }
        interpolation_tensor = interpolation_tensor.relabel_indices(index_relabel_map)
        skeleton_tensor = skeleton_tensor.relabel_indices(index_relabel_map)

        # Orient the digraph edge such that it's moving from back node to forward node (may or may not be a change depending on if we just skipped a turnback)
        if not self.ttn.ttn_graph.digraph.has_edge(back_node, forward_node):
            assert self.ttn.ttn_graph.digraph.has_edge(forward_node, back_node), (
                "The edge between the two nodes should exist in the graph."
            )
            edge_data = self.ttn.ttn_graph.digraph.edges[
                forward_node, back_node
            ].copy()  # readout the edge data to give to new edge

            self.ttn.ttn_graph.digraph.remove_edge(
                forward_node, back_node
            )  # remove the edge in the opposite direction

            self.ttn.ttn_graph.digraph.add_edge(
                back_node, forward_node, **edge_data
            )  # add the edge in the correct direction

        # Now get the new index setting along the active bond...
        back_index_settings = [
            self.ttn.ttn_graph.digraph.edges[edge]["external_index_setting"]
            for edge in incoming_edges_back
        ]
        back_index_setting = ExternalIndexSetting.tensor_parent_index_settings(
            back_index_settings
        )

        # ... and filter this index setting by the interpolated column indices
        filtered_index_setting = back_index_setting.from_subset(
            back_index_setting, interpolated_column_indices
        )

        # Update the graph with the new index setting at this edge, and update the new bond dimension
        self.ttn.ttn_graph.digraph.edges[back_node, forward_node][
            "external_index_setting"
        ] = filtered_index_setting

        self.ttn.ttn_graph.digraph.edges[back_node, forward_node]["dimension"] = len(
            interpolated_column_indices
        )

        # Update the TTN's tensor list with the new tensors
        self.ttn.tensors[back_tensor_name] = interpolation_tensor
        self.ttn.tensors[forward_tensor_name] = skeleton_tensor

        # Set center and interpolator labels correctly
        assert {
            self.ttn.ttn_graph.digraph.nodes[back_node]["node_type"],
            self.ttn.ttn_graph.digraph.nodes[forward_node]["node_type"],
        } == {"center_interpolative", "interpolator"}, (
            "To maintain gauge, we should be working on an interpolative center and an interpolator tensor."
        )

        self.ttn.ttn_graph.digraph.nodes[back_node]["node_type"] = "interpolator"
        self.ttn.ttn_graph.digraph.nodes[forward_node]["node_type"] = (
            "center_interpolative"  # set the new center at the forward node before we proceed
        )

    def _tci_sweep(self) -> None:
        """Perform a single TCI sweep."""
        for idx_step in range(len(self.sweep_plan)):
            self._tci_update(idx_step)
        self.sweep_count += 1

    def tensor_cross_interpolation(
        self,
        n_sweeps: int = 1,
        monitor: bool = False,
    ) -> None:
        """
        Perform the TCI sweep a set number of times.

        TODO: lots of interface design needed here.

        Parameters
        ----------
        n_sweeps : int, optional
            The number of full sweeps to perform (default: 1).
        monitor : bool, optional
            TODO (default: False)

        """
        for _ in range(n_sweeps):
            self._tci_sweep()

    def tracked_tci(self, n_sweeps: int = 1) -> None:
        """
        Perform the TCI sweep a set number of times, tracking the progress with tqdm.

        Parameters
        ----------
        n_sweeps : int, optional
            The number of full sweeps to perform (default: 1).

        """
        with tqdm(total=n_sweeps, desc="Sweeps", unit="sweep") as sweep_bar:
            for sweep_num in range(1, n_sweeps + 1):  # Start sweep numbering from 1
                with tqdm(
                    total=len(self.sweep_plan),
                    desc=f"TCI Steps (sweep {sweep_num})",
                    unit="step",
                ) as step_bar:

                    for idx_step in range(len(self.sweep_plan)):
                        self._tci_update(idx_step)
                        step_bar.update(1)  # Update the progress bar for each step
                        step_bar.set_postfix(
                            {
                                "Bond dimension": self.ttn.get_max_bond_dimension(),
                                "Size (MB)": self.ttn.memsize() / (1024 * 1024),
                            }
                        )

                sweep_bar.update(1)  # Update the progress bar for each sweep
