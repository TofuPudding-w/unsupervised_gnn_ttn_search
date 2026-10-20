import pytest
import numpy as np
from treetensorcomp.tensor_networks.tensors import Tensor

def test_tensor_contraction_indices():
    """Test that tensor contraction correctly handles indices."""
    # Create two simple tensors
    tensor_a = np.random.rand(2, 3, 4)  # 3D tensor
    tensor_b = np.random.rand(4, 5, 6)  # 3D tensor
    
    # Create Tensor objects with named indices
    a = Tensor(tensor_a, ["i", "j", "k"], id="tensor_a")
    b = Tensor(tensor_b, ["k", "l", "m"], id="tensor_b")
    
    # Contract along the common index "k"
    result = a @ b
    
    # Check that the result has the expected indices
    assert result.indices == ("i", "j", "l", "m")
    
    # Check that the result has the expected shape
    assert result.tensor_ndarray.shape == (2, 3, 5, 6)
    
    # Check the id is a combination of the input ids
    assert result.id == "tensor_a@tensor_b"
    
    # Test with multiple common indices
    tensor_c = np.random.rand(2, 3, 4, 5)
    tensor_d = np.random.rand(4, 3, 6, 7)
    
    c = Tensor(tensor_c, ["a", "b", "c", "d"], id="tensor_c")
    d = Tensor(tensor_d, ["c", "b", "e", "f"], id="tensor_d")
    
    # Contract along common indices "b" and "c"
    result2 = c @ d
    
    # Check indices - order should be non-contracted indices from first tensor
    # followed by non-contracted indices from second tensor
    assert result2.indices == ("a", "d", "e", "f")
    
    # Check shape
    assert result2.tensor_ndarray.shape == (2, 5, 6, 7)

def test_tensor_contraction_fails_with_mismatched_dimensions():
    """Test that tensor contraction raises an error when common indices have mismatched dimensions."""
    a = Tensor(np.random.rand(2, 3), ["i", "j"], id="tensor_a")
    b = Tensor(np.random.rand(4, 5), ["j", "k"], id="tensor_b")  # j dimension is 4 here but 3 in tensor a
    
    # Should raise an assertion error due to dimension mismatch
    with pytest.raises(AssertionError) as excinfo:
        result = a @ b

def test_isometry_identity():
    """Test numerical correctness of contraction by contracting an isometry with its adjoint.
    Also tests contraction along two indices."""
    import numpy as np
    from treetensorcomp.helper.linalg import generate_random_isometry
    from treetensorcomp.tensor_networks.tensors import Tensor
    
    # Generate a random isometry
    n = 12  # Larger dimension
    m = 3   # Smaller dimension
    isometry = generate_random_isometry(n, m)
    
    # The isometry shape is (n, m) = (12, 3)
    assert isometry.shape == (n, m)
    # Split the largest dimension (n=12) into two dimensions (e.g., 3x4)
    n1, n2 = 3, 4
    # Reshape the isometry into a 3D tensor
    tensor_array = isometry.reshape(n1, n2, m)
    # Create a Tensor object with named indices
    tensor = Tensor(tensor_array, ["i", "j", "k"], id="reshaped_isometry")
    tensor_adjoint = Tensor(np.conj(tensor_array), ["i", "j", "l"], id="adjoint_isometry")
    should_be_identity = (tensor @ tensor_adjoint).tensor_ndarray
    # Check that the result is close to the identity tensor
    assert should_be_identity.shape == (m, m)
    assert np.allclose(should_be_identity, np.eye(m)), "Contraction did not yield identity matrix as expected."

def test_tensor_svd():
    """Test the SVD decomposition of tensors and verify reconstruction."""
    import numpy as np
    from treetensorcomp.tensor_networks.tensors import Tensor
    
    # Create a random tensor with shape (2, 3, 4)
    tensor_array = np.random.rand(2, 3, 4)
    tensor = Tensor(tensor_array, ["i", "j", "k"], id="test_tensor")
    
    # Perform SVD with row_indices=["i", "j"] and col_indices=["k"]
    u_tensor, s_tensor, v_tensor, truncation_error = tensor.svd(
        row_indices=["i", "j"], 
        col_indices=["k"]
    )
    
    # Verify the shapes and indices
    assert u_tensor.tensor_ndarray.shape == (2, 3, min(2*3, 4))
    assert s_tensor.tensor_ndarray.shape == (min(2*3, 4), min(2*3, 4))
    assert v_tensor.tensor_ndarray.shape == (min(2*3, 4), 4)
    
    # Check that the truncation error is zero when no truncation is applied
    assert truncation_error == 0.0
    
    # Check that indices are as expected
    assert len(u_tensor.indices) == 3
    assert u_tensor.indices[0] == "i"
    assert u_tensor.indices[1] == "j"
    assert s_tensor.indices[0] == u_tensor.indices[2]  # Bond indices match
    assert s_tensor.indices[1] == v_tensor.indices[0]  # Bond indices match
    assert v_tensor.indices[1] == "k"
    
    # Contract the decomposed tensors to recover the original tensor
    reconstructed_tensor = u_tensor @ s_tensor @ v_tensor
    
    # Verify the reconstruction is accurate
    assert reconstructed_tensor.indices == ("i", "j", "k")
    assert np.allclose(reconstructed_tensor.tensor_ndarray, tensor_array)
    
    # Test with truncation
    u_trunc, s_trunc, v_trunc, trunc_error = tensor.svd(
        row_indices=["i", "j"],
        col_indices=["k"],
        max_bond_dim=2  # Only keep 2 singular values
    )
    
    # Verify truncation happened
    assert u_trunc.tensor_ndarray.shape == (2, 3, 2)
    assert s_trunc.tensor_ndarray.shape == (2, 2)
    assert v_trunc.tensor_ndarray.shape == (2, 4)
    assert trunc_error > 0.0  # Truncation error should be non-zero
    
    # Test SVD with different row/column splits
    u2_tensor, s2_tensor, v2_tensor, trunc_error2 = tensor.svd(
        row_indices=["i"],
        col_indices=["j", "k"]
    )
    
    # Verify shapes for this split
    assert u2_tensor.tensor_ndarray.shape == (2, min(2, 3*4))
    assert v2_tensor.tensor_ndarray.shape == (min(2, 3*4), 3, 4)
    
    # Contract to verify
    us2_tensor = u2_tensor @ s2_tensor
    reconstructed2_tensor = us2_tensor @ v2_tensor
    assert reconstructed2_tensor.indices == ("i", "j", "k")
    assert np.allclose(reconstructed2_tensor.tensor_ndarray, tensor_array)
    
    # Test with cutoff_rtol
    tensor_with_decay = Tensor(np.diag([1.0, 0.1, 0.01, 0.001]).reshape(2, 2, 4), ["a", "b", "c"])
    u3, s3, v3, err3 = tensor_with_decay.svd(
        row_indices=["a", "b"],
        col_indices=["c"],
        cutoff_rtol=0.05  # Should keep only the first two singular values
    )
    
    assert s3.tensor_ndarray.shape == (2, 2)  # Only 2 singular values kept
    assert err3 > 0.0
    
    # Verify the ids are properly set
    assert u_tensor.id == "test_tensor_u"
    assert s_tensor.id == "test_tensor_s"
    assert v_tensor.id == "test_tensor_v"

def test_partial_trace():
    """Test partial trace against a simple seperable case."""
    # Create a separable state: |0⟩⟨0| \otimes |+⟩⟨+|
    rho_0 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
    rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
    
    # Tensor product
    rho = np.kron(rho_0, rho_plus)
    rho = rho.reshape(2, 2, 2, 2)  # Reshape to rank-4 tensor
    
    # Create Tensor with proper indices
    tensor_rho = Tensor(rho, ["a", "b", "a*", "b*"], id="density_matrix")
    
    # Partial trace over system a
    rho_b = tensor_rho.partial_trace(["a"])
    
    # Result should be rho_plus
    assert np.allclose(rho_b.tensor_ndarray, rho_plus)
    
    # Partial trace over system b
    rho_a = tensor_rho.partial_trace(["b"])
    
    # Result should be rho_0
    assert np.allclose(rho_a.tensor_ndarray, rho_0)
    
    # Full trace should be 1
    assert np.isclose(tensor_rho.partial_trace(["a", "b"]), 1.0)

def test_interpolative_decomposition():
    """Basic test on shapes and indices of the interpolative decomposition."""

    # Create a random tensor with shape (2, 3, 4)
    tensor_array = np.random.rand(2, 3, 4)
    tensor = Tensor(tensor_array, ["i", "j", "k"], id="test_tensor")

    # Perform interpolative decomposition with row_indices=["i", "j"] and col_indices=["k"]
    skeleton_tensor, interpolation_tensor, interpolated_column_indices = tensor.interpolative_decomposition(row_indices=["i", "j"], col_indices=["k"], relative_error=1e-10)

    # Verify the shapes - actual rank determined by algorithm, check compatibility
    # The decomposition rank should be <= min(row_dim, col_dim)
    decomposition_rank = interpolation_tensor.tensor_ndarray.shape[0]
    assert decomposition_rank <= min(2*3, 4)

    assert skeleton_tensor.tensor_ndarray.shape == (2, 3, decomposition_rank)
    assert interpolation_tensor.tensor_ndarray.shape == (decomposition_rank, 4)

    # Check that indices are as expected
    assert len(skeleton_tensor.indices) == 3
    assert skeleton_tensor.indices[0] == "i"
    assert skeleton_tensor.indices[1] == "j"
    # Check the bond index connects the two tensors
    bond_idx = skeleton_tensor.indices[2]
    assert interpolation_tensor.indices[0] == bond_idx
    assert interpolation_tensor.indices[1] == "k"

    # Contract the decomposed tensors to recover the original tensor
    reconstructed_tensor = skeleton_tensor @ interpolation_tensor

    # Verify the reconstruction is accurate
    assert reconstructed_tensor.indices == ("i", "j", "k")
    assert np.allclose(reconstructed_tensor.tensor_ndarray, tensor_array, rtol=1e-8, atol=1e-8)

def test_ID_rank_revealing():
    """ Check if the interpolative decomposition can rediscover low-rank structure and pull the right columns in the right order. """
    rank = 2

    # Build a low-rank tensor by product of three tensors
    mat_A = np.random.rand(5, rank); mat_B = np.random.rand(rank, rank); mat_C = np.random.rand(rank, 4)
    tensor_A = Tensor(mat_A, ['i', 'left'], id='A'); tensor_B = Tensor(mat_B, ['left', 'right'], id='B'); tensor_C = Tensor(mat_C, ['right', 'j'], id='C')
    random_tensor = tensor_A @ tensor_B @ tensor_C

    # Do an ID
    relative_error = 1e-8
    skeleton_tensor, interpolation_tensor, interpolated_column_indices = random_tensor.interpolative_decomposition(['i'], ['j'], relative_error=relative_error)

    # Check that the ID has rediscovered the low rank structure
    assert skeleton_tensor.tensor_ndarray.shape[1] == rank
    assert interpolation_tensor.tensor_ndarray.shape[0] == rank

    # Check that the skeleton tensor has the correct columns pulled from the original tensor
    assert np.allclose(random_tensor.matricize(["i"], ["j"])[0][:, interpolated_column_indices], skeleton_tensor.matricize(["i"], [skeleton_tensor.indices[1]])[0])