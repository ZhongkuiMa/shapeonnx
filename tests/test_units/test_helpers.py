"""Unit tests for shape inference helper functions."""

import onnx
import pytest

from shapeonnx.infer_shape import (
    _align_shapes,
    _get_data_shape,
    _get_explicit_shape,
    _get_shape,
    _preconvert_integer_initializers,
    _right_align_shapes,
)


class TestAlignShapes:
    """Tests for align_shapes function."""

    def test_align_both_empty(self):
        """Test aligning two empty shapes."""
        result = _align_shapes([], [])
        assert result == []

    def test_align_matching_dimensions(self):
        """Test aligning shapes with matching dimensions."""
        result = _align_shapes([3, 4], [3, 4])
        assert result == [3, 4]

    def test_align_base_matches_target(self):
        """Test aligning when base dimensions match target dimensions."""
        result = _align_shapes([3, 4], [3, 4, 5])
        assert result == [3, 4, 1]

    def test_align_single_base_dimension(self):
        """Test aligning single base dimension."""
        result = _align_shapes([3], [3, 4, 5])
        assert result == [3, 1, 1]

    def test_align_no_matching_dimensions(self):
        """Test aligning when no dimensions match."""
        result = _align_shapes([2], [3, 4, 5])
        assert result == [1, 1, 1]


class TestRightAlignShapes:
    """Tests for right_align_shapes function."""

    def test_right_align_same_rank(self):
        """Test right alignment with same rank."""
        result = _right_align_shapes([3, 4], [3, 4])
        assert result == ([3, 4], [3, 4])

    def test_right_align_different_rank(self):
        """Test right alignment with different ranks."""
        result = _right_align_shapes([4, 5], [3, 4, 5])
        assert result == ([1, 4, 5], [3, 4, 5])

    def test_right_align_scalar(self):
        """Test right alignment with scalar (empty list)."""
        result = _right_align_shapes([], [3, 4])
        assert result == ([1, 1], [3, 4])

    def test_right_align_both_empty(self):
        """Test right alignment with both empty."""
        result = _right_align_shapes([], [])
        assert result == ([], [])


class TestGetShape:
    """Tests for get_shape function."""

    def test_get_shape_from_data(self):
        """Test getting shape from data_shapes."""
        result = _get_shape("x", {"x": [3, 4]}, {})
        assert result == ([3, 4], False)

    def test_get_shape_from_explicit(self):
        """Test getting shape from explicit_shapes."""
        result = _get_shape("x", {}, {"x": [2, 5]})
        assert result == ([2, 5], True)

    def test_get_shape_not_found(self):
        """Test getting shape that doesn't exist."""
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _get_shape("x", {}, {})

    def test_get_shape_prefer_data(self):
        """Test that data_shapes takes priority over explicit_shapes."""
        result = _get_shape("x", {"x": [3, 4]}, {"x": [2, 5]})
        assert result == ([3, 4], False)


class TestGetDataShape:
    """Tests for get_data_shape function."""

    def test_get_data_shape_list(self):
        """Test getting list shape from data_shapes."""
        result = _get_data_shape("x", {"x": [3, 4]})
        assert result == [3, 4]

    def test_get_data_shape_scalar(self):
        """Test getting scalar shape from data_shapes."""
        result = _get_data_shape("x", {"x": 5})
        assert result == 5

    def test_get_data_shape_not_found(self):
        """Test getting shape that doesn't exist."""
        result = _get_data_shape("x", {})
        assert result is None


class TestGetExplicitShape:
    """Tests for get_explicit_shape function."""

    def test_get_explicit_shape_list(self):
        """Test getting list shape from explicit_shapes."""
        result = _get_explicit_shape("x", {"x": [3, 4]})
        assert result == [3, 4]

    def test_get_explicit_shape_scalar(self):
        """Test getting scalar shape from explicit_shapes."""
        result = _get_explicit_shape("x", {"x": 5})
        assert result == 5

    def test_get_explicit_shape_not_found(self):
        """Test getting shape that doesn't exist."""
        result = _get_explicit_shape("x", {})
        assert result is None


class TestPreconvertIntegerInitializers:
    """Tests for preconvert_integer_initializers function."""

    def test_preconvert_empty_initializers(self):
        """Test preconverting empty initializers dict."""
        result = _preconvert_integer_initializers({})
        assert result == {}

    def test_preconvert_float_initializer(self):
        """Test that float initializers are not converted."""
        import numpy as np

        weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weights, name="weight")
        result = _preconvert_integer_initializers({"weight": weight_tensor})
        assert "weight" not in result

    def test_preconvert_int_initializer(self):
        """Test preconverting integer initializers."""
        import numpy as np

        shape_array = np.array([1, 256], dtype=np.int64)
        shape_tensor = onnx.numpy_helper.from_array(shape_array, name="shape")
        result = _preconvert_integer_initializers({"shape": shape_tensor})
        assert "shape" in result
        assert result["shape"] == [1, 256]
