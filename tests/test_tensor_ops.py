import pytest

from tynitorch import Tensor, DeviceType, DType
from tynitorch import cuda


class TestCreation:
    """Test cases for tensor construction."""

    def test_creation(self):
        tensor = Tensor([[1, 2], [3, 4]], device="cpu", dtype=DType.INT64)
        assert tensor.shape == (2, 2)
        assert tensor.device.type == DeviceType.CPU
        assert tensor.dtype == DType.INT64
        assert str(tensor) == "[\n  [1, 2],\n  [3, 4]\n]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_creation_cuda(self):
        tensor = Tensor([[1, 2], [3, 4]], device="cuda", dtype=DType.FLOAT32)
        assert tensor.device.type == DeviceType.CUDA
        assert tensor.dtype == DType.FLOAT32
        assert tensor.shape == (2, 2)
        assert str(tensor) == "[\n  [1.0, 2.0],\n  [3.0, 4.0]\n]"


class TestTranspose:
    """Test cases for Tensor.transpose()"""

    def test_transpose_2d_basic(self):
        """Test basic 2D transpose."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert t_t.shape == (3, 2)
        assert t_t.is_contiguous() == False  # Transposed view is not contiguous
        assert str(t_t) == "[\n  [1.0, 4.0],\n  [2.0, 5.0],\n  [3.0, 6.0]\n]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_2d_basic_cuda(self):
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert t_t.shape == (3, 2)
        assert t_t.is_contiguous() == False  # Transposed view is not contiguous
        assert str(t_t) == "[\n  [1.0, 4.0],\n  [2.0, 5.0],\n  [3.0, 6.0]\n]"

    def test_transpose_round_trip_2d(self):
        """Test that transpose(a,b) then transpose(b,a) returns original layout."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_t_t = t_t.transpose(0, 1)
        
        assert t_t_t.shape == t.shape
        assert str(t_t_t) == str(t)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_round_trip_2d_cuda(self):
        """Test that transpose(a,b) then transpose(b,a) returns original layout."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_t_t = t_t.transpose(0, 1)
        
        assert t_t_t.shape == t.shape
        assert str(t_t_t) == str(t)

    def test_transpose_3d(self):
        """Test transpose on 3D tensor."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cpu", dtype=DType.FLOAT32)
        # Original shape: (2, 2, 2)
        t_t = t.transpose(0, 2)
        
        assert t_t.shape == (2, 2, 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_3d_cuda(self):
        """Test transpose on 3D tensor."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cuda", dtype=DType.FLOAT32)
        # Original shape: (2, 2, 2)
        t_t = t.transpose(0, 2)
        
        assert t_t.shape == (2, 2, 2)

    def test_transpose_3d_content(self):
        """Transposing 3D tensor reorders elements as expected."""
        t = Tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            device="cpu",
            dtype=DType.FLOAT32,
        )
        t_t = t.transpose(0, 2)

        assert t_t.shape == (3, 2, 2)
        assert str(t_t) == "[\n  [\n    [1.0, 7.0],\n    [4.0, 10.0]\n  ],\n  [\n    [2.0, 8.0],\n    [5.0, 11.0]\n  ],\n  [\n    [3.0, 9.0],\n    [6.0, 12.0]\n  ]\n]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_3d_content_cuda(self):
        """Transposing 3D tensor reorders elements as expected."""
        t = Tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            device="cuda",
            dtype=DType.FLOAT32,
        )
        t_t = t.transpose(0, 2)

        assert t_t.shape == (3, 2, 2)
        assert str(t_t) == "[\n  [\n    [1.0, 7.0],\n    [4.0, 10.0]\n  ],\n  [\n    [2.0, 8.0],\n    [5.0, 11.0]\n  ],\n  [\n    [3.0, 9.0],\n    [6.0, 12.0]\n  ]\n]"

    def test_transpose_same_dim(self):
        """Test transposing same dimensions should be identity."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 0)
        
        assert t_t.shape == t.shape
        assert str(t_t) == str(t)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_same_dim_cuda(self):
        """Test transposing same dimensions should be identity."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 0)
        
        assert t_t.shape == t.shape
        assert str(t_t) == str(t)

    def test_transpose_invalid_dim(self):
        """Test transpose with out-of-bounds dimensions."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        
        with pytest.raises(IndexError):
            t.transpose(0, 2)
        
        with pytest.raises(IndexError):
            t.transpose(-1, 0)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_invalid_dim_cuda(self):
        """Test transpose with out-of-bounds dimensions."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        
        with pytest.raises(IndexError):
            t.transpose(0, 2)
        
        with pytest.raises(IndexError):
            t.transpose(-1, 0)

    def test_transpose_shares_storage(self):
        """Test that transpose creates a view sharing the same storage."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        # Both should reference the same storage
        assert t.storage is t_t.storage

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_shares_storage_cuda(self):
        """Test that transpose creates a view sharing the same storage."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        # Both should reference the same storage
        assert t.storage is t_t.storage

    def test_multiple_transposes_3d_content(self):
        """Chained transposes on 3D tensor keep data consistent."""
        t = Tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            device="cpu",
            dtype=DType.FLOAT32,
        )

        t_chain = t.transpose(0, 1).transpose(1, 2)

        assert t_chain.shape == (2, 3, 2)
        assert str(t_chain) == "[\n  [\n    [1.0, 7.0],\n    [2.0, 8.0],\n    [3.0, 9.0]\n  ],\n  [\n    [4.0, 10.0],\n    [5.0, 11.0],\n    [6.0, 12.0]\n  ]\n]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_multiple_transposes_3d_content_cuda(self):
        """Chained transposes on 3D tensor keep data consistent."""
        t = Tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            device="cuda",
            dtype=DType.FLOAT32,
        )

        t_chain = t.transpose(0, 1).transpose(1, 2)

        assert t_chain.shape == (2, 3, 2)
        assert str(t_chain) == "[\n  [\n    [1.0, 7.0],\n    [2.0, 8.0],\n    [3.0, 9.0]\n  ],\n  [\n    [4.0, 10.0],\n    [5.0, 11.0],\n    [6.0, 12.0]\n  ]\n]"


class TestView:
    """Test cases for Tensor.view()"""

    def test_view_flatten(self):
        """Test flattening a 2D tensor to 1D."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu", dtype=DType.FLOAT32)
        t_flat = t.view((6,))
        
        assert t_flat.shape == (6,)
        assert str(t_flat) == "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_flatten_cuda(self):
        """Test flattening a 2D tensor to 1D."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=DType.FLOAT32)
        t_flat = t.view((6,))
        
        assert t_flat.shape == (6,)
        assert str(t_flat) == "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"

    def test_view_reshape_2d(self):
        """Test reshaping a 1D tensor to 2D."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cpu", dtype=DType.FLOAT32)
        t_2d = t.view((2, 3))
        
        assert t_2d.shape == (2, 3)
        assert str(t_2d) == "[\n  [1.0, 2.0, 3.0],\n  [4.0, 5.0, 6.0]\n]"

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_reshape_2d_cuda(self):
        """Test reshaping a 1D tensor to 2D."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda", dtype=DType.FLOAT32)
        t_2d = t.view((2, 3))
        
        assert t_2d.shape == (2, 3)
        assert str(t_2d) == "[\n  [1.0, 2.0, 3.0],\n  [4.0, 5.0, 6.0]\n]"

    def test_view_reshape_3d(self):
        """Test reshaping to 3D."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cpu", dtype=DType.FLOAT32)
        t_3d = t.view((2, 2, 2))
        
        assert t_3d.shape == (2, 2, 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_reshape_3d_cuda(self):
        """Test reshaping to 3D."""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=DType.FLOAT32)
        t_3d = t.view((2, 2, 2))
        
        assert t_3d.shape == (2, 2, 2)

    def test_view_round_trip(self):
        """Test view(a) then view(b) returns original if b matches original shape."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu", dtype=DType.FLOAT32)
        t_flat = t.view((6,))
        t_back = t_flat.view((2, 3))
        
        assert t_back.shape == t.shape
        assert str(t_back) == str(t)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_round_trip_cuda(self):
        """Test view(a) then view(b) returns original if b matches original shape."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=DType.FLOAT32)
        t_flat = t.view((6,))
        t_back = t_flat.view((2, 3))
        
        assert t_back.shape == t.shape
        assert str(t_back) == str(t)

    def test_view_element_count_mismatch(self):
        """Test that view fails when element counts don't match."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        
        with pytest.raises(ValueError, match="Cannot view tensor"):
            t.view((3,))

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_element_count_mismatch_cuda(self):
        """Test that view fails when element counts don't match."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        
        with pytest.raises(ValueError, match="Cannot view tensor"):
            t.view((3,))

    def test_view_requires_contiguous(self):
        """Test that view requires the tensor to be contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)  # Creates non-contiguous view
        
        with pytest.raises(ValueError, match="Can only view a contiguous tensor"):
            t_t.view((4,))

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_requires_contiguous_cuda(self):
        """Test that view requires the tensor to be contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)  # Creates non-contiguous view
        
        with pytest.raises(ValueError, match="Can only view a contiguous tensor"):
            t_t.view((4,))

    def test_view_shares_storage(self):
        """Test that view creates a view sharing the same storage."""
        t = Tensor([1.0, 2.0, 3.0, 4.0], device="cpu", dtype=DType.FLOAT32)
        t_2d = t.view((2, 2))
        
        # Both should reference the same storage
        assert t.storage is t_2d.storage

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_shares_storage_cuda(self):
        """Test that view creates a view sharing the same storage."""
        t = Tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=DType.FLOAT32)
        t_2d = t.view((2, 2))
        
        # Both should reference the same storage
        assert t.storage is t_2d.storage

    def test_view_with_single_dimension(self):
        """Test view with single-element dimensions."""
        t = Tensor([[[1.0, 2.0]]], device="cpu", dtype=DType.FLOAT32)
        t_flat = t.view((2,))
        
        assert t_flat.shape == (2,)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_view_with_single_dimension_cuda(self):
        """Test view with single-element dimensions."""
        t = Tensor([[[1.0, 2.0]]], device="cuda", dtype=DType.FLOAT32)
        t_flat = t.view((2,))
        
        assert t_flat.shape == (2,)


class TestContiguous:
    """Test cases for Tensor.contiguous()"""

    def test_contiguous_already_contiguous(self):
        """Test that contiguous returns self if already contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_cont = t.contiguous()
        
        # Should return the same object
        assert t_cont is t

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_contiguous_already_contiguous_cuda(self):
        """Test that contiguous returns self if already contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_cont = t.contiguous()
        
        # Should return the same object
        assert t_cont is t

    def test_contiguous_non_contiguous_view(self):
        """Test that contiguous copies data from non-contiguous view."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)  # Non-contiguous
        t_cont = t_t.contiguous()
        
        # Should be a different object
        assert t_cont is not t_t
        assert t_cont.is_contiguous()
        assert str(t_cont) == str(t_t)  # Data should be the same
        # But storage should be different
        assert t_cont.storage is not t.storage

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_contiguous_non_contiguous_view_cuda(self):
        """Test that contiguous copies data from non-contiguous view."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)  # Non-contiguous
        t_cont = t_t.contiguous()
        
        # Should be a different object
        assert t_cont is not t_t
        assert t_cont.is_contiguous()
        assert str(t_cont) == str(t_t)  # Data should be the same
        # But storage should be different
        assert t_cont.storage is not t.storage

    def test_contiguous_preserves_data(self):
        """Test that contiguous preserves all data values."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 2)  # Non-contiguous
        t_cont = t_t.contiguous()
        
        assert str(t_cont) == str(t_t)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_contiguous_preserves_data_cuda(self):
        """Test that contiguous preserves all data values."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 2)  # Non-contiguous
        t_cont = t_t.contiguous()
        
        assert str(t_cont) == str(t_t)

    def test_contiguous_round_trip(self):
        """Test that contiguous(contiguous(t)) == contiguous(t)."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_cont1 = t_t.contiguous()
        t_cont2 = t_cont1.contiguous()
        
        # Second contiguous should return self
        assert t_cont2 is t_cont1

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_contiguous_round_trip_cuda(self):
        """Test that contiguous(contiguous(t)) == contiguous(t)."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_cont1 = t_t.contiguous()
        t_cont2 = t_cont1.contiguous()
        
        # Second contiguous should return self
        assert t_cont2 is t_cont1

    def test_contiguous_after_transpose(self):
        """Test making a transposed tensor contiguous."""
        original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t = Tensor(original, device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert not t_t.is_contiguous()
        
        t_cont = t_t.contiguous()
        assert t_cont.is_contiguous()
        assert t_cont.shape == (3, 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_contiguous_after_transpose_cuda(self):
        """Test making a transposed tensor contiguous."""
        original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t = Tensor(original, device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert not t_t.is_contiguous()
        
        t_cont = t_t.contiguous()
        assert t_cont.is_contiguous()
        assert t_cont.shape == (3, 2)


class TestIsContiguous:
    """Test cases for Tensor.is_contiguous()"""

    def test_is_contiguous_newly_created(self):
        """Test that newly created tensors are contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_newly_created_cuda(self):
        """Test that newly created tensors are contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    def test_is_contiguous_1d(self):
        """Test 1D tensor is always contiguous."""
        t = Tensor([1.0, 2.0, 3.0], device="cpu", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_1d_cuda(self):
        """Test 1D tensor is always contiguous."""
        t = Tensor([1.0, 2.0, 3.0], device="cuda", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    def test_is_contiguous_empty(self):
        """Test empty tensor is contiguous."""
        t = Tensor([], device="cpu", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_empty_cuda(self):
        """Test empty tensor is contiguous."""
        t = Tensor([], device="cuda", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    def test_is_contiguous_scalar(self):
        """Test scalar (0D) tensor is contiguous."""
        t = Tensor(5.0, device="cpu", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_scalar_cuda(self):
        """Test scalar (0D) tensor is contiguous."""
        t = Tensor(5.0, device="cuda", dtype=DType.FLOAT32)
        assert t.is_contiguous()

    def test_is_contiguous_after_transpose(self):
        """Test that transposed tensor is not contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert not t_t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_after_transpose_cuda(self):
        """Test that transposed tensor is not contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        assert not t_t.is_contiguous()

    def test_is_contiguous_after_contiguous(self):
        """Test that contiguous() makes non-contiguous tensor contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_cont = t_t.contiguous()
        
        assert t_cont.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_after_contiguous_cuda(self):
        """Test that contiguous() makes non-contiguous tensor contiguous."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        t_cont = t_t.contiguous()
        
        assert t_cont.is_contiguous()

    def test_is_contiguous_3d(self):
        """Test contiguity check on 3D tensor."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cpu", dtype=DType.FLOAT32)
        assert t.is_contiguous()
        
        t_t = t.transpose(0, 2)
        assert not t_t.is_contiguous()

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_is_contiguous_3d_cuda(self):
        """Test contiguity check on 3D tensor."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cuda", dtype=DType.FLOAT32)
        assert t.is_contiguous()
        
        t_t = t.transpose(0, 2)
        assert not t_t.is_contiguous()


class TestIntegration:
    """Integration tests combining multiple operations"""

    def test_transpose_view_contiguous_chain(self):
        """Test chaining transpose -> view (should fail) -> contiguous -> view (should work)."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        # View should fail on non-contiguous
        with pytest.raises(ValueError):
            t_t.view((4,))
        
        # After contiguous, view should work
        t_cont = t_t.contiguous()
        t_flat = t_cont.view((4,))
        assert t_flat.shape == (4,)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_transpose_view_contiguous_chain_cuda(self):
        """Test chaining transpose -> view (should fail) -> contiguous -> view (should work)."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=DType.FLOAT32)
        t_t = t.transpose(0, 1)
        
        # View should fail on non-contiguous
        with pytest.raises(ValueError):
            t_t.view((4,))
        
        # After contiguous, view should work
        t_cont = t_t.contiguous()
        t_flat = t_cont.view((4,))
        assert t_flat.shape == (4,)

    def test_multiple_transposes(self):
        """Test multiple sequential transposes."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cpu", dtype=DType.FLOAT32)
        
        # (2, 2, 2) -> transpose(0,1) -> (2, 2, 2)
        t = t.transpose(0, 1)
        # (2, 2, 2) -> transpose(1,2) -> (2, 2, 2)
        t = t.transpose(1, 2)
        # (2, 2, 2) -> transpose(0,2) -> (2, 2, 2)
        t = t.transpose(0, 2)
        
        assert t.shape == (2, 2, 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="Cuda not available")
    def test_multiple_transposes_cuda(self):
        """Test multiple sequential transposes."""
        t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device="cuda", dtype=DType.FLOAT32)
        
        # (2, 2, 2) -> transpose(0,1) -> (2, 2, 2)
        t = t.transpose(0, 1)
        # (2, 2, 2) -> transpose(1,2) -> (2, 2, 2)
        t = t.transpose(1, 2)
        # (2, 2, 2) -> transpose(0,2) -> (2, 2, 2)
        t = t.transpose(0, 2)
        
        assert t.shape == (2, 2, 2)
