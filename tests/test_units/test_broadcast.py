"""Unit tests for broadcasting functions."""

import pytest

from shapeonnx.infer_shape import (
    _broadcast_shapes,
    _compute_broadcasted_shape,
)


class TestBroadcastShapes:
    """Tests for broadcast_shapes function."""

    def test_broadcast_scalar_scalar(self):
        """Test broadcasting scalar with scalar."""
        result = _broadcast_shapes([], [])
        assert result == []

    def test_broadcast_scalar_vector(self):
        """Test broadcasting scalar with vector."""
        result = _broadcast_shapes([], [3, 4])
        assert result == [3, 4]

    def test_broadcast_vector_scalar(self):
        """Test broadcasting vector with scalar."""
        result = _broadcast_shapes([3, 4], [])
        assert result == [3, 4]

    def test_broadcast_same_shape(self):
        """Test broadcasting shapes that are the same."""
        result = _broadcast_shapes([3, 4], [3, 4])
        assert result == [3, 4]

    def test_broadcast_compatible_shapes(self):
        """Test broadcasting compatible shapes."""
        result = _broadcast_shapes([1, 4], [3, 4])
        assert result == [3, 4]

    def test_broadcast_compatible_shapes_reversed(self):
        """Test broadcasting compatible shapes reversed."""
        result = _broadcast_shapes([3, 1], [3, 4])
        assert result == [3, 4]

    def test_broadcast_different_ranks_compatible(self):
        """Test broadcasting shapes with different ranks."""
        result = _broadcast_shapes([4], [3, 4])
        assert result == [3, 4]

    def test_broadcast_multiple_dimensions(self):
        """Test broadcasting multiple dimensions."""
        result = _broadcast_shapes([1, 3, 1], [2, 1, 4])
        assert result == [2, 3, 4]

    def test_broadcast_with_zero_dimension(self):
        """Test broadcasting with 0 dimension."""
        result = _broadcast_shapes([0], [3, 4])
        assert result == [0]

    def test_broadcast_both_with_zero(self):
        """Test broadcasting when both have 0."""
        result = _broadcast_shapes([0, 4], [0, 4])
        assert result == [0, 4]


class TestComputeBroadcastedShape:
    """Tests for compute_broadcasted_shape function."""

    def test_compute_same_shapes(self):
        """Test computing same shapes."""
        result = _compute_broadcasted_shape([3, 4], [3, 4])
        assert result == [3, 4]

    def test_compute_one_dimension_one(self):
        """Test computing when one dimension is 1."""
        result = _compute_broadcasted_shape([1, 4], [3, 4])
        assert result == [3, 4]

    def test_compute_multiple_ones(self):
        """Test computing with multiple 1 dimensions."""
        result = _compute_broadcasted_shape([1, 1, 4], [3, 4, 4])
        assert result == [3, 4, 4]

    def test_compute_all_ones(self):
        """Test computing when all dimensions are 1."""
        result = _compute_broadcasted_shape([1, 1], [3, 4])
        assert result == [3, 4]

    def test_compute_complex_broadcasting(self):
        """Test complex broadcasting scenario."""
        result = _compute_broadcasted_shape([1, 3, 1, 4], [2, 1, 5, 4])
        assert result == [2, 3, 5, 4]

    def test_compute_incompatible_shapes(self):
        """Test computing incompatible shapes."""
        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _compute_broadcasted_shape([2, 3], [4, 5])


class TestBroadcastingEdgeCases:
    """Tests for edge cases in broadcasting."""

    def test_broadcast_all_ones(self):
        """Test broadcasting all 1 dimensions."""
        result = _broadcast_shapes([1, 1, 1], [3, 4, 5])
        assert result == [3, 4, 5]

    def test_broadcast_all_ones_reversed(self):
        """Test broadcasting all 1 dimensions reversed."""
        result = _broadcast_shapes([3, 4, 5], [1, 1, 1])
        assert result == [3, 4, 5]

    def test_broadcast_single_dimension(self):
        """Test broadcasting single dimension."""
        result = _broadcast_shapes([1], [4])
        assert result == [4]

    def test_broadcast_large_dimensions(self):
        """Test broadcasting large dimensions."""
        result = _broadcast_shapes([1024, 768], [1024, 768])
        assert result == [1024, 768]
