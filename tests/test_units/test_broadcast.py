"""Unit tests for broadcasting functions."""

import pytest

from shapeonnx.infer_shape import _broadcast_shapes, _compute_broadcasted_shape

_BROADCAST_CASES = [
    pytest.param([], [], [], id="scalar_scalar"),
    pytest.param([], [3, 4], [3, 4], id="scalar_vector"),
    pytest.param([3, 4], [], [3, 4], id="vector_scalar"),
    pytest.param([3, 4], [3, 4], [3, 4], id="same_shape"),
    pytest.param([1, 4], [3, 4], [3, 4], id="compatible"),
    pytest.param([3, 1], [3, 4], [3, 4], id="compatible_reversed"),
    pytest.param([4], [3, 4], [3, 4], id="different_ranks"),
    pytest.param([1, 3, 1], [2, 1, 4], [2, 3, 4], id="multiple_dims"),
    pytest.param([0], [3, 4], [0], id="with_zero_dim"),
    pytest.param([0, 4], [0, 4], [0, 4], id="both_with_zero"),
    pytest.param([1, 1, 1], [3, 4, 5], [3, 4, 5], id="all_ones"),
    pytest.param([3, 4, 5], [1, 1, 1], [3, 4, 5], id="all_ones_reversed"),
    pytest.param([1], [4], [4], id="single_dim"),
    pytest.param([1024, 768], [1024, 768], [1024, 768], id="large_dims"),
]

_COMPUTE_CASES = [
    pytest.param([3, 4], [3, 4], [3, 4], id="same_shapes"),
    pytest.param([1, 4], [3, 4], [3, 4], id="one_dimension_one"),
    pytest.param([1, 1, 4], [3, 4, 4], [3, 4, 4], id="multiple_ones"),
    pytest.param([1, 1], [3, 4], [3, 4], id="all_ones"),
    pytest.param([1, 3, 1, 4], [2, 1, 5, 4], [2, 3, 5, 4], id="complex"),
]


class TestBroadcastShapes:
    """Tests for _broadcast_shapes function."""

    @pytest.mark.parametrize(("shape1", "shape2", "expected"), _BROADCAST_CASES)
    def test_broadcast_returns_expected_shape(
        self, shape1: list[int], shape2: list[int], expected: list[int]
    ) -> None:
        """Verify _broadcast_shapes returns the expected merged shape.

        :param shape1: first input shape
        :param shape2: second input shape
        :param expected: expected broadcasted output shape
        """
        assert _broadcast_shapes(shape1, shape2) == expected


class TestComputeBroadcastedShape:
    """Tests for _compute_broadcasted_shape function."""

    @pytest.mark.parametrize(("shape1", "shape2", "expected"), _COMPUTE_CASES)
    def test_compute_returns_expected_shape(
        self, shape1: list[int], shape2: list[int], expected: list[int]
    ) -> None:
        """Verify _compute_broadcasted_shape returns the expected merged shape.

        :param shape1: first input shape
        :param shape2: second input shape
        :param expected: expected broadcasted output shape
        """
        assert _compute_broadcasted_shape(shape1, shape2) == expected

    def test_compute_incompatible_shapes_raises(self) -> None:
        """Verify _compute_broadcasted_shape rejects incompatible shapes."""
        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _compute_broadcasted_shape([2, 3], [4, 5])
