from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray


class IndexID:
    """
    Identifier for an external tensor index representing a physical degree of freedom.

    Specifically we identify each external tensor index with one qubit in the
    computational basis, i.e. in a |0> or |1> state, represented with a single bit.
    Thus each IndexID can be equivalently thought of as a single bit in a binary string.

    We assume bit indexing starting from 1 throughout.
    """

    variable_name: str
    bit_position: int

    def __init__(self, variable_name: str, bit_position: int) -> None:
        """
        Construct an IndexID representing a single external tensor index.

        Parameters
        ----------
        variable_name : str
            A text identifier for the tensor index.
        bit_position : int
            Bit index used to represent this tensor index. Indexed starting from 1.

        """
        assert isinstance(variable_name, str), "variable_name must be a string"
        assert "_" not in variable_name, "variable_name must not contain underscores"
        assert not (variable_name.startswith(("bond", "tensor"))), (
            "variable_name must not start with 'bond' or 'tensor'"
        )
        assert isinstance(bit_position, int) and bit_position > 0, (
            "bit_position must be a positive integer"
        )

        self.variable_name = variable_name
        self.bit_position = bit_position

    def __str__(self) -> str:
        """Get string representation."""
        return f"{self.variable_name}_{self.bit_position}"

    def __repr__(self) -> str:
        """Get inline notebook representation."""
        return f"IndexID({self.__str__()})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another object."""
        if not isinstance(other, IndexID):
            return False

        return (
            self.variable_name == other.variable_name
            and self.bit_position == other.bit_position
        )


class ExternalIndexSetting:
    """A set of index settings for one of more external tensor indices."""

    mapping: dict[str, NDArray]
    index_ids: list[IndexID]
    setting_count: int

    @classmethod
    def from_subset(
        cls,
        index_setting: Self,
        included_indices: NDArray[np.int_],
    ) -> Self:
        """
        Construct a new instance including a subset of the tensor indices of another.

        Parameters
        ----------
        index_setting : ExternalIndexSetting
            The ExternalIndexSetting to extract the settings from.

        included_indices : NDArray[np.int_]
            The subset of external tensor indices to include in the new instance.

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.

        """
        instance = cls()

        if not isinstance(index_setting, ExternalIndexSetting):
            msg = "index_setting must be an ExternalIndexSetting instance"
            raise TypeError(msg)

        assert (
            isinstance(included_indices, np.ndarray) and included_indices.ndim == 1
        ), "included_indices must be a 1D numpy ndarray"

        assert all(isinstance(i, (int, np.integer)) for i in included_indices), (
            "All elements of included_indices must be integers"
        )

        assert all(
            (i >= 0) and (i < index_setting.setting_count) for i in included_indices
        ), "included_indices must be valid indices within the range of setting_count"

        # Create a new instance
        instance.mapping = {
            key: value[included_indices] for key, value in index_setting.mapping.items()
        }
        instance.setting_count = len(included_indices)
        instance.index_ids = index_setting.index_ids
        return instance

    @classmethod
    def from_physical_index(
        cls,
        index_id: IndexID,
        qubit_settings: NDArray[np.int_],
    ) -> Self:
        """
        Construct a new instance including a single external tensor index.

        Parameters
        ----------
        index_id : IndexID
            The tensor index to include.

        qubit_settings : NDArray[np.int_]
            An array of bit states for the tensor index.

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.

        """
        instance = cls()
        assert isinstance(index_id, IndexID), (
            "index_id must be a treetensorcomp.quantics.index_settings.IndexID instance"
        )

        assert (
            isinstance(qubit_settings, np.ndarray)
            and (qubit_settings.dtype == int)
            and (qubit_settings.ndim == 1)
        ), "qubit_settings must be a 1D numpy ndarray of integers"

        assert np.all((qubit_settings == 0) | (qubit_settings == 1)), (
            "All elements of qubit_settings must be 0 or 1"
        )

        instance.mapping = {str(index_id): qubit_settings}
        instance.setting_count = qubit_settings.size
        instance.index_ids = [index_id]

        return instance

    @classmethod
    def tensor_parent_index_settings(
        cls,
        parent_index_settings: Sequence[Self],
    ) -> Self:
        """
        Construct a new instance from the tensor product of multiple instances.

        Parameters
        ----------
        parent_index_settings : Sequence[ExternalIndexSetting]
            The set of ExternalIndexSettings to combine.

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.

        """
        instance = cls()
        dimensions = _process_parent_settings(parent_index_settings)

        # Total dimension (tensor product)
        total_combinations = np.prod(dimensions)

        # Create a new mapping (currently empty values, we'll populate below) whose keys
        # are the same as the parent settings
        instance.mapping = {
            key: np.empty(total_combinations, dtype=int)
            for setting in parent_index_settings
            for key in setting.mapping
        }

        # Populate the mapping with the tensor-producted settings
        for global_idx in range(total_combinations):
            # Create a tensor multi-index using the same logic for numpy reshape
            multi_idx = np.unravel_index(global_idx, dimensions)

            # For each key in the mapping, find which parent setting it belongs to and get the appropriate value
            for key in instance.mapping.keys():
                # Find which parent setting contains this key
                for parent_idx, parent_setting in enumerate(parent_index_settings):
                    if key in parent_setting.mapping:
                        # Get the local index for this parent setting
                        local_idx = multi_idx[parent_idx]

                        # Set the value in our combined mapping
                        instance.mapping[key][global_idx] = parent_setting.mapping[key][
                            local_idx
                        ]

                        break

        # populate other attributes
        instance.setting_count = total_combinations
        instance.index_ids = [
            index_id
            for setting in parent_index_settings
            for index_id in setting.index_ids
        ]

        return instance

    @classmethod
    def union_parent_index_settings(cls, parent_index_settings: Sequence[Self]) -> Self:
        """
        Construct a new instance from the union of multiple instances.

        Parameters
        ----------
        parent_index_settings : Sequence[ExternalIndexSetting]
            The set of ExternalIndexSettings to combine.

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.

        """
        instance = cls()
        dimensions = _process_parent_settings(parent_index_settings)
        assert all(dim == dimensions[0] for dim in dimensions), (
            "All parent_index_settings must have the same setting_count"
        )

        instance.setting_count = dimensions[0]
        instance.index_ids = [
            index_id
            for setting in parent_index_settings
            for index_id in setting.index_ids
        ]

        # Combine all .mapping dictionaries from parent_index_settings
        instance.mapping = {}
        for setting in parent_index_settings:
            instance.mapping.update(setting.mapping)

        return instance

    @classmethod
    def random_index_setting(
        cls,
        variable_names: Sequence[str],
        max_precisions: Sequence[int],
        count: int = 100,
    ) -> Self:
        """
        Construct a new instance with random settings for multiple different variables.

        Parameters
        ----------
        variable_names : Sequence[str]
            Identifier for each of the variables.
        max_precisions : Sequence[int]
            The maximum number of indices (bits) to use for each variable setting. Must
            have the same size as variable_names.
        count : int, optional
            The number of different random settings to generate (default: 100).

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.


        """
        # Setup and initial input checks
        instance = cls()
        assert isinstance(count, int) and count > 0, "count must be a positive integer"
        assert isinstance(max_precisions, (list, tuple)) and len(max_precisions) == len(
            variable_names
        ), (
            "max_precisions must be a list or tuple with the same length as variable_names"
        )
        assert all(isinstance(p, int) and p > 0 for p in max_precisions), (
            "All max_precisions must be positive integers"
        )

        # Populate some attributes
        instance.index_ids = [
            IndexID(variable_name, idx)
            for variable_name, max_precision in zip(variable_names, max_precisions)
            for idx in range(1, max_precision + 1)
        ]
        instance.setting_count = count

        # Populate the mapping with random bits in the right sahpe
        instance.mapping = {}
        for index_id in instance.index_ids:
            # Generate random bits for each index ID
            bits = np.random.randint(0, 2, size=count, dtype=int)
            instance.mapping[str(index_id)] = bits

        return instance

    @classmethod
    def from_variable_values(
        cls,
        variable_name: str,
        precision: int,
        values: NDArray[np.float64],
    ) -> Self:
        """
        Construct a new instance with settings calculated from a set of real values.

        Parameters
        ----------
        variable_name : str
            The identifier for the variable.
        precision : int
            The maximum number of indices (bits) to use for each variable setting.
        values : NDArray[np.float64]
            Array of values which will be cast into binary format to define each
            setting. Must be in [0, 1] range.

        Returns
        -------
        ExternalIndexSetting
            The new ExternalIndexSetting instance.

        """
        instance = cls()
        # Validate inputs
        assert isinstance(variable_name, str), "variable_name must be a string"
        assert isinstance(precision, int) and precision > 0, (
            "precision must be a positive integer"
        )
        assert isinstance(values, np.ndarray), "values must be a numpy ndarray"
        assert values.dtype == float
        assert values.ndim == 1, "values must be a 1D numpy ndarray"
        assert np.all((values >= 0) & (values <= 1)), (
            "All elements of values must be between 0 and 1 inclusive"
        )

        instance.setting_count = len(values)
        instance.index_ids = [
            IndexID(variable_name, idx + 1) for idx in range(precision)
        ]
        # empty mapping to be populated
        instance.mapping = {
            str(index_id): np.empty(instance.setting_count, dtype=int)
            for index_id in instance.index_ids
        }

        for idx_value, value in enumerate(values):
            for idx_bit in range(precision):
                value *= 2
                bit = min(int(value), 1)
                instance.mapping[str(instance.index_ids[idx_bit])][idx_value] = bit
                value -= bit

        return instance

    # TODO probably a * overload that does two from parent_index_settings

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        # Prepare a preview of the first 8 values for each variable
        preview_size = 12
        preview_lines = []
        for var in self.variable_names():
            # Collect all index_ids for this variable, sorted by bit_position
            ids = sorted(
                [idx for idx in self.index_ids if idx.variable_name == var],
                key=lambda x: x.bit_position,
            )

            for idx in ids:
                values = self.mapping[str(idx)]
                # For each bit, show the first 8 values, add ... if more
                values_str = ",".join(str(v) for v in values[:preview_size])
                if len(values) > preview_size:
                    values_str += ",..."
                preview_lines.append(f"{str(idx)} → [{values_str}]")
        preview_html = "<br>".join(preview_lines)

        return f"""
        <div style="border: 1px solid #ccc; padding: 10px; margin: 5px;">
            <strong>ExternalIndexSetting</strong><br>
            <strong>Variables:</strong> {", ".join(self.variable_names())}<br>
            <strong>Setting count:</strong> {self.setting_count}<br>
            <strong>Preview:</strong><br>
            <code style="white-space: pre-line">{preview_html}</code>
        </div>
        """

    def __str__(self) -> str:
        """Get string representation."""
        indices_str = ", ".join(str(idx) for idx in self.index_ids)
        return f"ExternalIndexSetting([{indices_str}], {self.setting_count} settings)"

    def __repr__(self) -> str:
        """Get inline notebook representation."""
        return self.__str__()

    def __getitem__(self, key: IndexID | str) -> NDArray:
        """Get the value for the given key, which can be an IndexID or a string."""
        if isinstance(key, IndexID):
            key = str(key)
        return self.mapping[key]

    def variable_names(self) -> list[str]:
        """Return a (sorted) list of unique variable names from the indexIDs."""
        return sorted({index_id.variable_name for index_id in self.index_ids})

    def contains_index_id(self, index_id: IndexID | str) -> bool:
        """Check if the mapping contains an entry for the given IndexID."""
        return str(index_id) in self.mapping

    def max_bit_position(self, variable_name: str) -> int | None:
        """
        Get the largest bit_position for the given variable_name among the index_ids.

        Returns None if the variable_name is not present.
        """
        positions = [
            idx.bit_position
            for idx in self.index_ids
            if idx.variable_name == variable_name
        ]
        return max(positions) if positions else None

    def has_consecutive_bits(self, variable_name: str) -> bool:
        """Check if the setting for a given variable_name contains all bits up to max_bit_position."""
        max_bit = self.max_bit_position(variable_name)

        if max_bit is None:
            return False

        for bit in range(1, max_bit + 1):
            index_id = IndexID(variable_name, bit)
            if not self.contains_index_id(index_id):
                return False

        return True

    def overwrite_indices(
        self,
        indices: Sequence[IndexID],
        index_values: Sequence[int],
    ) -> None:
        """
        Overwrite the values for the specified indices with the given index_values.

        Parameters
        ----------
        indices : Sequence[IndexID]
            Set of IndexIDs to overwrite.
        index_values : Sequence[int]
            Binary values for each of the indices to be set to.

        """
        assert isinstance(indices, (list, tuple)) and len(indices) == len(
            index_values
        ), (
            "fixed_indices must be a list or tuple with the same length as fixed_index_values"
        )
        assert all(isinstance(idx, IndexID) for idx in indices), (
            "All fixed_indices must be IndexID instances"
        )
        assert all(
            isinstance(value, int) and value in (0, 1) for value in index_values
        ), "All fixed_index_values must be integers 0 or 1"

        # Populate the map with overwritten fixed value
        for index_id, value in zip(indices, index_values):
            self.mapping[str(index_id)] = np.full(self.setting_count, value, dtype=int)


# TODO: this should be replaced with a more descriptive name
# and probably broken down into more specific functions
def _process_parent_settings(
    parent_index_settings: Sequence[ExternalIndexSetting],
) -> list[int]:
    """
    Get the number of settings for each ExternalIndexSetting in a set.

    Also validates that the identifiers for all settings are distinct.
    """
    assert isinstance(parent_index_settings, (tuple, list)), (
        "parent_index_settings must be a tuple or list"
    )
    assert all(
        isinstance(item, ExternalIndexSetting) for item in parent_index_settings
    ), "All items in parent_index_settings must be ExternalIndexSettings"

    # Check that all keys in the parent index settings mappings are distinct
    all_keys = []
    for setting in parent_index_settings:
        all_keys.extend(setting.mapping.keys())

    assert len(all_keys) == len(set(all_keys)), (
        "Keys in parent_index_settings mappings must be distinct"
    )

    dimensions = [setting.setting_count for setting in parent_index_settings]
    return dimensions
