import uuid
from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import interpolative

from treetensorcomp.quantics.index_settings import ExternalIndexSetting


class Tensor:
    """A multi-dimensional numerical array implementing standard tensor operations."""

    tensor_ndarray: NDArray
    indices: tuple[str, ...]
    id: str

    def __init__(
        self,
        tensor_ndarray: NDArray[np.float64],
        indices: list[str],
        id: str | None = None,
    ) -> None:
        """
        Construct a Tensor object.

        Parameters
        ----------
        tensor_ndarray : NDArray[np.float64]
            Numerical entries for each index in the tensor.
        indices : list[str]
            String identifiers for the indices in the tensor.
        id : str, optional
            A string identifier for the overall tensor.

        """
        assert len(indices) == len(tensor_ndarray.shape), (
            "Indices must match tensor dimensions"
        )
        assert len(set(indices)) == len(indices), "Indices must be unique"
        assert isinstance(tensor_ndarray, np.ndarray), "Tensor must be a numpy ndarray"
        assert isinstance(indices, (list, tuple)), "Indices must be a list or tuple"
        assert all(isinstance(idx, str) for idx in indices), (
            "All indices must be strings"
        )
        self.tensor_ndarray = tensor_ndarray
        self.indices = tuple(indices)
        if id is None:
            self.id = "tensor_" + str(uuid.uuid4())
        else:
            assert isinstance(id, str), "id must be a string"
            self.id = id

    def __str__(self) -> str:
        """Get string representation."""
        return f"Tensor(id='{self.id}', shape={self.tensor_ndarray.shape}, indices={self.indices})"

    def __repr__(self) -> str:
        """Get inline notebook representation."""
        return str(self)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return f"""
        <div style="border: 1px solid #ccc; padding: 10px; margin: 5px;">
            <strong>Tensor: {self.id}</strong><br>
            <strong>Shape:</strong> {self.tensor_ndarray.shape}<br>
            <strong>Indices:</strong> {self.indices}<br>
            <strong>dtype:</strong> {self.tensor_ndarray.dtype}<br>
            <strong>Size:</strong> {self.tensor_ndarray.size} elements
        </div>
        """

    def __matmul__(self, other: Self) -> Self:
        """
        Contract two tensors along their common indices (Einstein summation).

        Overloads the '@' operator.

        Parameters
        ----------
        other : Tensor
            The other Tensor to contract with.

        Returns
        -------
        Tensor
            A new Tensor representing the result of the contraction.

        """
        if not isinstance(other, Tensor):
            msg = "Can only contract with another Tensor object."
            raise TypeError(msg)

        # Find common indices to contract over (Einstein summation convention)
        common_indices = [
            idx_str for idx_str in self.indices if idx_str in other.indices
        ]
        # Create a mapping of indices to positions in the tensors
        index_map_self = {idx_str: i for i, idx_str in enumerate(self.indices)}
        index_map_other = {idx_str: i for i, idx_str in enumerate(other.indices)}
        # Determine axes to contract over in both tensors, then contract with tensordot
        axes_self = [index_map_self[idx_str] for idx_str in common_indices]
        axes_other = [index_map_other[idx_str] for idx_str in common_indices]

        # Check that the dimensions of the contracted axes match
        for self_axis, other_axis, idx_str in zip(
            axes_self, axes_other, common_indices
        ):
            assert (
                self.tensor_ndarray.shape[self_axis]
                == other.tensor_ndarray.shape[other_axis]
            ), (
                f"Dimension mismatch for index '{idx_str}': {self.tensor_ndarray.shape[self_axis]} vs {other.tensor_ndarray.shape[other_axis]}"
            )
        contracted_tensor = np.tensordot(
            self.tensor_ndarray, other.tensor_ndarray, axes=(axes_self, axes_other)
        )
        # Build the output indices: all indices from self not contracted, then all from other not contracted
        result_indices = [idx for idx in self.indices if idx not in common_indices] + [
            idx for idx in other.indices if idx not in common_indices
        ]

        return Tensor(contracted_tensor, result_indices, self.id + "@" + other.id)

    # TODO: add + and - overloads for elementwise addition and subtraction of tensors with the same indices

    def rename_self(self, new_id: str) -> None:
        """Rename the tensor by changing its id."""
        if not isinstance(new_id, str):
            msg = "New id must be a string."
            raise TypeError(msg)

        self.id = new_id

    def relabel_indices(self, new_indices_map: dict[str, str]) -> Self:
        """
        Create a new tensor with its indices relabelled.

        Parameters
        ----------
        new_indices_map : dict[str, str]
            A mapping from the current indices to their new labels after renaming.

        Returns
        -------
        Tensor
            A new Tensor using the updated indices.

        """
        assert isinstance(new_indices_map, dict), "new_indices_map must be a dictionary"
        assert all(old_idx in self.indices for old_idx in new_indices_map.keys()), (
            "All keys in new_indices_map must be present in tensor indices"
        )

        new_indices = [new_indices_map.get(idx, idx) for idx in self.indices]
        return Tensor(self.tensor_ndarray, new_indices, self.id)

    def adjoint(self) -> Self:
        """
        Get the Hermitian adjoint (conjugate transpose) of the tensor.

        Conjugation is performed numerically whereas the transposition is handled via
        the starring of indices.

        Returns
        -------
        Tensor
            A new Tensor corresponding to the Hermitian adjoint.

        """
        # Star each index to indicate transposition
        starred_indices = [idx + "*" for idx in self.indices]

        return Tensor(
            np.conjugate(self.tensor_ndarray),
            starred_indices,
            self.id + "_adjoint",
        )

    def matricize(
        self,
        row_indices: list[str],
        col_indices: list[str],
    ) -> tuple[NDArray[np.float64], dict[str, int]]:
        """
        Matricize the tensor along specified row and column indices.

        Parameters
        ----------
        row_indices : list[str]
            Subset of `self.indices` to be grouped as the "row" indices.
        col_indices : list[str]
            Subset of `self.indices` to be grouped as the "column" indices.

        Returns
        -------
        NDArray[np.float64]
            The matricized tensor as a 2D numpy array.

        dict[str, int]]
            A mapping for each index ID to the dimension of the full tensor.

        """
        assert set(row_indices + col_indices) == set(self.indices), (
            "Row and column indices must cover all tensor indices"
        )
        assert len(row_indices) > 0 and len(col_indices) > 0, (
            "Must partition along at least one row and one column index"
        )

        # Maps each index id to the dimension of the full tensor
        shape_dict = {
            idx: self.tensor_ndarray.shape[i] for i, idx in enumerate(self.indices)
        }

        # Determine positions of row and column indices (might be entwined in general)
        row_positions = [self.indices.index(idx) for idx in row_indices]
        col_positions = [self.indices.index(idx) for idx in col_indices]

        # Determine dimensions for matricized tensor
        row_dim = np.prod([shape_dict[idx] for idx in row_indices])
        col_dim = np.prod([shape_dict[idx] for idx in col_indices])

        # Permute tensor to group row and column indices together, then matricize
        permuted_tensor = np.transpose(
            self.tensor_ndarray, row_positions + col_positions
        )
        matrix = permuted_tensor.reshape(row_dim, col_dim)

        return matrix, shape_dict

    def svd(
        self,
        row_indices: list[str],
        col_indices: list[str],
        cutoff_rtol: float | None = None,
        max_bond_dim: int | None = None,
    ) -> tuple[Self, Self, Self, float]:
        """
        Perform singular value decomposition (SVD) on the tensor.

        Matricizes the tensor along specified row and column indices with optional
        truncation based on a relative cutoff and/or maximum bond dimension.

        Parameters
        ----------
        row_indices : list[str]
             Subset of `self.indices` to be grouped as the "row" indices.
        col_indices : list[str]
            Subset of `self.indices` to be grouped as the "column" indices.
        cutoff_rtol : float, optional
            Relative cutoff for singular values (default: no cutoff).
        max_bond_dim : int | None
            Maximum number of singular values (bond dimension) to keep (default: no
            cutoff). If both `cutoff_rtol` and `max_bond_dim` are specified, the most
            aggressive truncation is applied.

        Returns
        -------
        u_tensor : Tensor
            Left isometric tensor reshaped from the left singular vectors, with new bond
            index appended.
        s_tensor : Tensor
            Diagonal matrix of retained singular values, with new bond indices.
        v_tensor : Tensor
            Right isometric tensor reshaped from the right singular vectors, with new
            bond index prepended.
        truncation_error : float
            Truncation error due to discarding low singular values (0.0 if no
            truncation).

        """
        # Matricize the tensor
        matrix, shape_dict = self.matricize(row_indices, col_indices)

        # Split the nodes with SVD. Note that by using full_matrices=False we get the
        # 'thin' or ecomnical SVD, so we have isometries instead of unitaries
        U_matrix, full_S, V_matrix = np.linalg.svd(matrix, full_matrices=False)

        # Copy the full singular values into a new list that can be pruned as we go
        trimmed_S = full_S.copy()

        # Truncation error to report, zero unless overwritten later
        truncation_error = 0.0

        if cutoff_rtol is not None or max_bond_dim is not None:
            keep_mask_cutoff = np.ones_like(full_S, dtype=bool)

            # Masks to keep singular values based on cutoff and max bond dimension
            keep_mask_max_bond_dim = np.ones_like(full_S, dtype=bool)

            # First, if a cutoff is imposed, truncate singular values below it
            if cutoff_rtol is not None:
                # Relative cutoff (compared to largest singular value)
                keep_mask_cutoff = full_S >= (cutoff_rtol * full_S[0])

            # Now if a maximum bond dimension is imposed, sort the singular values and keep only that many
            if max_bond_dim is not None and len(full_S) > max_bond_dim:
                # Since SVD returns singular values in descending order, we just keep the first max_bond_dim values
                keep_mask_max_bond_dim[max_bond_dim:] = False

            # Combine both masks to determine which values to keep
            keep_mask = keep_mask_cutoff & keep_mask_max_bond_dim

            truncation_error = np.sum(full_S[~keep_mask] ** 2) / np.sum(full_S**2)

            # Truncate matrices
            trimmed_S = full_S[keep_mask]
            U_matrix = U_matrix[:, keep_mask]
            V_matrix = V_matrix[keep_mask, :]

        # Now we reshape U and V back to tensors with labelled indices
        left_idx_name = "svd_left_" + str(uuid.uuid4())
        right_idx_name = "svd_right_" + str(uuid.uuid4())

        u_shape = tuple(shape_dict[idx] for idx in row_indices) + (len(trimmed_S),)
        v_shape = (len(trimmed_S),) + tuple(shape_dict[idx] for idx in col_indices)

        u_tensor = U_matrix.reshape(u_shape)
        v_tensor = V_matrix.reshape(v_shape)
        s_tensor = np.diag(trimmed_S)

        u_indices = row_indices + [left_idx_name]
        v_indices = [right_idx_name] + col_indices
        s_indices = [left_idx_name, right_idx_name]

        return (
            Tensor(u_tensor, u_indices, self.id + "_u"),
            Tensor(s_tensor, s_indices, self.id + "_s"),
            Tensor(v_tensor, v_indices, self.id + "_v"),
            truncation_error,
        )

    # TODO: scipy 1.15 broke this, since (1xn) matrices now cause an error message. for now use a lower version.
    # TODO: consider using prrLU from SciPostPhys.18.3.104 instead of scipy's randomized methods.
    def interpolative_decomposition(
        self,
        row_indices: list[str],
        col_indices: list[str],
        relative_error: float = 1e-12,
    ) -> tuple[Self, Self, NDArray[np.float64]]:
        """
        Perform an interpolative decomposition (ID) of the tensor.

        Matricizes the tensor along the specified row and column indices and then
        factorizes the matricized tensor A into A ≈ BP, where B is a skeleton matrix
        formed from selected columns of A, and P is an interpolation matrix.

        In this implementation, the left (row) indices are used for the skeleton, and
        the right (column) indices are used for the interpolation. Thus, the row index
        is pushed up the tree when setting interpolative gauge, as opposed to SVD for
        setting orthogonality centres, where the column index is pushed up.

            - The method returns both the skeleton and interpolation tensors, as well as the indices of the selected
              columns.

        Parameters
        ----------
        row_indices : list[str]
            The tensor indices to be grouped as rows in the matricization.
        col_indices : list[str]
            The tensor indices to be grouped as columns in the matricization.
        relative_error : float, optional
            The relative error tolerance for the decomposition (default: 1e-12).

        Returns
        -------
        skeleton_tensor : Tensor
            The skeleton tensor with indices `row_indices + [new_idx]`.
        interpolation_tensor : Tensor
            The interpolation tensor with indices `[new_idx] + col_indices`.
        interpolated_column_indices : NDArray[np.float64]
            Indices of the selected columns in the original matrix.

        """
        # Matricize the tensor
        matrix, shape_dict = self.matricize(row_indices, col_indices)
        decomposition_rank, interpolated_column_indices, interpolation_coefficients = (
            interpolative.interp_decomp(matrix, relative_error)
        )

        # A = BP, for skeleton B and interpolation matrix P. see https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html
        skeleton_matrix = interpolative.reconstruct_skel_matrix(
            matrix, decomposition_rank, interpolated_column_indices
        )
        interpolation_matrix = interpolative.reconstruct_interp_matrix(
            interpolated_column_indices, interpolation_coefficients
        )

        # Now we reshape the skeleton and interpolation matrices back to tensors with labelled indices
        new_idx_name = "interpdecomp_" + str(uuid.uuid4())
        skeleton_shape = tuple(shape_dict[idx] for idx in row_indices) + (
            decomposition_rank,
        )
        interpolation_shape = (decomposition_rank,) + tuple(
            shape_dict[idx] for idx in col_indices
        )

        skeleton_tensor = skeleton_matrix.reshape(skeleton_shape)
        interpolation_tensor = interpolation_matrix.reshape(interpolation_shape)

        skeleton_indices = row_indices + [new_idx_name]
        interpolation_indices = [new_idx_name] + col_indices

        return (
            Tensor(skeleton_tensor, skeleton_indices, self.id + "_skeleton"),
            Tensor(
                interpolation_tensor, interpolation_indices, self.id + "_interpolation"
            ),
            interpolated_column_indices[:decomposition_rank],
        )

    def partial_trace(self, trace_indices: Sequence[str]) -> Self:
        """
        Compute the partial trace of the tensor over specified indices.

        Parameters
        ----------
        trace_indices : Sequence[str]
            Indices to trace out.

        Returns
        -------
        Tensor
            A new Tensor object with the traced indices removed.

        """
        # Check indices valid
        assert set(trace_indices).issubset(set(self.indices)), (
            "Trace indices must be a subset of tensor indices"
        )

        working_tensor = self
        for idx in trace_indices:
            # Check the index has a conjugate counterpart
            conj_idx = idx + "*"
            assert conj_idx in self.indices, (
                f"Index '{idx}' must have a conjugate counterpart '{conj_idx}' in the tensor indices"
            )

            # Assert that the dimensions match for the index and its conjugate
            idx_pos = self.indices.index(idx)
            conj_pos = self.indices.index(conj_idx)
            idx_dim = self.tensor_ndarray.shape[idx_pos]
            assert idx_dim == self.tensor_ndarray.shape[conj_pos], (
                f"Dimension mismatch for index '{idx}' and its conjugate '{idx}*': {self.tensor_ndarray.shape[idx_pos]} vs {self.tensor_ndarray.shape[conj_pos]}"
            )

            # Contract the tensor with an identity that links/sums the index and its conjugate
            identity_tensor = Tensor(np.eye(idx_dim), [idx, conj_idx])
            working_tensor = working_tensor @ identity_tensor

        working_tensor.rename_self(self.id + f"_tr({trace_indices})")

        # If we've contracted down to a scalar, return it
        if len(working_tensor.indices) == 0:
            return working_tensor.tensor_ndarray.item()

        # Else return the new Tensor
        return working_tensor

    def evaluate_index_setting(
        self,
        index_setting: ExternalIndexSetting,
    ) -> NDArray[np.float64]:
        """
        Evaluate the tensor for a given external index setting.

        Parameters
        ----------
        index_setting : ExternalIndexSetting
            The mapping from index names to arrays of integer indices and a
            setting_count attribute indicating the number of index settings.

        Returns
        -------
        NDArray[np.float64]
            The evaluated tensor, indexed according to the provided index setting.

        """
        # Check inputs
        if not isinstance(index_setting, ExternalIndexSetting):
            msg = "index_setting must be an ExternalIndexSetting instance"
            raise TypeError(msg)

        assert set(self.indices) == set(index_setting.mapping.keys())

        # Construct indices for the tensor_ndarray
        tensor_indices = np.empty(
            (self.tensor_ndarray.ndim, index_setting.setting_count), dtype=int
        )

        for i, idx_str in enumerate(self.indices):
            tensor_indices[i, :] = index_setting.mapping[idx_str]

        # Now we can use the indices to index into the tensor_ndarray
        evaluated_tensor = self.tensor_ndarray[
            tuple(tensor_indices[i, :] for i in range(self.tensor_ndarray.ndim))
        ]

        return evaluated_tensor
