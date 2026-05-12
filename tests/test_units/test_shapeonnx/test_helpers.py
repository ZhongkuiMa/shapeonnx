"""Unit tests for shape inference helper functions."""

__docformat__ = "restructuredtext"

import numpy as np
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

    @pytest.mark.parametrize(
        ("base", "target", "expected"),
        [
            pytest.param([], [], [], id="both_empty"),
            pytest.param([3, 4], [3, 4], [3, 4], id="matching_dimensions"),
            pytest.param([3, 4], [3, 4, 5], [3, 4, 1], id="base_shorter"),
            pytest.param([3], [3, 4, 5], [3, 1, 1], id="single_base"),
            pytest.param([2], [3, 4, 5], [1, 1, 1], id="no_matching"),
        ],
    )
    def test_align(self, base, target, expected):
        """Test shape alignment with various base/target combinations."""
        result = _align_shapes(base, target)
        assert result == expected


class TestRightAlignShapes:
    """Tests for right_align_shapes function."""

    @pytest.mark.parametrize(
        ("base", "target", "expected"),
        [
            pytest.param([3, 4], [3, 4], ([3, 4], [3, 4]), id="same_rank"),
            pytest.param([4, 5], [3, 4, 5], ([1, 4, 5], [3, 4, 5]), id="different_rank"),
            pytest.param([], [3, 4], ([1, 1], [3, 4]), id="scalar_base"),
            pytest.param([], [], ([], []), id="both_empty"),
        ],
    )
    def test_right_align(self, base, target, expected):
        """Test right alignment with various base/target shape combinations."""
        result = _right_align_shapes(base, target)
        assert result == expected


class TestGetShape:
    """Tests for get_shape function."""

    def test_get_shape_from_data(self):
        """Test getting shape from data_shapes."""
        result = _get_shape("x", {"x": [3, 4]}, {})
        assert result == ([3, 4], False)

    def test_get_shape_from_explicit(self):
        """Test getting shape from explicit_shapes."""
        result = _get_shape("x", {}, {"x": [2, 5]})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == [2, 5]
        assert result[1] is True

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
        """Test getting shape that doesn't exist returns None."""
        result = _get_data_shape("x", {})
        assert result is None
        assert not isinstance(result, (list, int))


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
        """Test getting shape that doesn't exist returns None."""
        result = _get_explicit_shape("x", {})
        assert result is None
        assert not isinstance(result, (list, int))


class TestPreconvertIntegerInitializers:
    """Tests for preconvert_integer_initializers function."""

    def test_preconvert_empty_initializers(self):
        """Test preconverting empty initializers dict."""
        result = _preconvert_integer_initializers({})
        assert result == {}

    def test_preconvert_float_initializer(self):
        """Test that float initializers are not converted."""
        weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weights, name="weight")
        result = _preconvert_integer_initializers({"weight": weight_tensor})
        assert "weight" not in result

    def test_preconvert_int_initializer(self):
        """Test preconverting integer initializers."""
        shape_array = np.array([1, 256], dtype=np.int64)
        shape_tensor = onnx.numpy_helper.from_array(shape_array, name="shape")
        result = _preconvert_integer_initializers({"shape": shape_tensor})
        assert "shape" in result
        assert result["shape"] == [1, 256]
