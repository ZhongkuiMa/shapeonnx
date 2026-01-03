"""Unit tests for Reduce operations (ReduceMean, ReduceSum, etc.) shape inference.

This module provides test coverage for reduce operations that compute aggregate
values across tensor axes.

Test organization:
- TestReduceMeanBasic: Basic ReduceMean with keepdims
- TestReduceSumBasic: Basic ReduceSum with keepdims
- TestReduceMultiAxis: Reducing across multiple axes
- TestReduceAllAxes: Reducing across all axes (scalar output)
- TestReduceErrors: Error handling
"""

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_argmax_shape,
    _infer_reduce_shape,
)


def _make_axes_initializer(axes: list[int], name: str = "axes") -> onnx.TensorProto:
    """Create an axes initializer tensor."""
    array = np.array(axes, dtype=np.int64)
    return onnx.numpy_helper.from_array(array, name=name)


class TestReduceMeanBasic:
    """Test basic ReduceMean operation."""

    def test_reducemean_single_axis_keepdims(self):
        """Test ReduceMean on single axis with keepdims=True."""
        axes_tensor = _make_axes_initializer([1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 1, 4]

    def test_reducemean_single_axis_no_keepdims(self):
        """Test ReduceMean on single axis without keepdims."""
        axes_tensor = _make_axes_initializer([1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 4]

    def test_reducemean_first_axis(self):
        """Test ReduceMean reducing first axis."""
        axes_tensor = _make_axes_initializer([0])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 3, 224, 224]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [1, 3, 224, 224]

    def test_reducemean_last_axis(self):
        """Test ReduceMean reducing last axis."""
        axes_tensor = _make_axes_initializer([3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4, 5]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 3, 4]

    def test_reducemean_negative_axis(self):
        """Test ReduceMean with negative axis."""
        axes_tensor = _make_axes_initializer([-1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        # axis=-1 is same as axis=2 for 3D tensor
        assert results[0][0] == [2, 3, 1]


class TestReduceSumBasic:
    """Test basic ReduceSum operation."""

    def test_reducesum_single_axis_keepdims(self):
        """Test ReduceSum on single axis with keepdims=True."""
        axes_tensor = _make_axes_initializer([1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 1, 4]

    def test_reducesum_single_axis_no_keepdims(self):
        """Test ReduceSum on single axis without keepdims."""
        axes_tensor = _make_axes_initializer([1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 4]

    def test_reducesum_batch_dimension(self):
        """Test ReduceSum preserves other dimensions."""
        axes_tensor = _make_axes_initializer([1])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [32, 256, 224, 224]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [32, 224, 224]


class TestReduceMultiAxis:
    """Test reducing across multiple axes."""

    def test_reducemean_multiple_axes_keepdims(self):
        """Test ReduceMean on multiple axes with keepdims."""
        axes_tensor = _make_axes_initializer([1, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4, 5]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 1, 4, 1]

    def test_reducemean_multiple_axes_no_keepdims(self):
        """Test ReduceMean on multiple axes without keepdims."""
        axes_tensor = _make_axes_initializer([1, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4, 5]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [2, 4]

    def test_reducesum_spatial_axes(self):
        """Test ReduceSum on spatial axes (common in pooling)."""
        axes_tensor = _make_axes_initializer([2, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 64, 28, 28]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [1, 64, 1, 1]


class TestReduceAllAxes:
    """Test reducing across all axes (scalar output)."""

    def test_reducemean_all_axes_keepdims(self):
        """Test ReduceMean across all axes with keepdims."""
        axes_tensor = _make_axes_initializer([0, 1, 2])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [1, 1, 1]

    def test_reducemean_all_axes_no_keepdims(self):
        """Test ReduceMean across all axes without keepdims (empty shape)."""
        axes_tensor = _make_axes_initializer([0, 1, 2])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        # Reducing all axes without keepdims produces empty shape []
        assert results[0][0] == []

    def test_reducesum_all_axes_scalar(self):
        """Test ReduceSum to empty shape (reduces all axes without keepdims)."""
        axes_tensor = _make_axes_initializer([0, 1, 2, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 3, 224, 224]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        # Reducing all axes without keepdims produces empty shape []
        assert results[0][0] == []


class TestReduceErrors:
    """Test error handling for Reduce operations."""

    def test_reducemean_missing_input_shape(self):
        """Test error when input shape is missing."""
        axes_tensor = _make_axes_initializer([0])
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["missing_input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        with pytest.raises(RuntimeError):
            _infer_reduce_shape(node, ctx)

    def test_reducesum_missing_input_shape(self):
        """Test error when input shape is missing for ReduceSum."""
        axes_tensor = _make_axes_initializer([0])
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["missing_input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        with pytest.raises(RuntimeError):
            _infer_reduce_shape(node, ctx)


class TestReduceIntegration:
    """Integration tests for Reduce operations."""

    def test_reducemean_channel_pooling(self):
        """Test ReduceMean for channel pooling in CNNs."""
        axes_tensor = _make_axes_initializer([2, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [32, 128, 7, 7]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=1,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [32, 128, 1, 1]

    def test_reducesum_global_pool(self):
        """Test ReduceSum for global pooling."""
        axes_tensor = _make_axes_initializer([0, 2, 3])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 512, 14, 14]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=0,
        )

        results = _infer_reduce_shape(node, ctx)

        assert results[0][0] == [512]


class TestArgMaxOperation:
    """Test ArgMax operation shape inference."""

    def test_argmax_axis_0_keepdims(self):
        """Test ArgMax on axis 0 with keepdims=True."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [5, 4, 3]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            axis=0,
            keepdims=1,
        )

        results = _infer_argmax_shape(node, ctx)

        assert results[0][0] == [1, 4, 3]

    def test_argmax_axis_1_no_keepdims(self):
        """Test ArgMax on axis 1 without keepdims."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            axis=1,
            keepdims=0,
        )

        results = _infer_argmax_shape(node, ctx)

        assert results[0][0] == [2, 4]

    def test_argmax_last_axis(self):
        """Test ArgMax on last axis."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [3, 4, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            axis=2,
            keepdims=1,
        )

        results = _infer_argmax_shape(node, ctx)

        assert results[0][0] == [3, 4, 1]

    def test_argmax_scalar_input_with_keepdims(self):
        """Test ArgMax with scalar input returns shape."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # scalar
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            axis=0,
            keepdims=1,
        )

        results = _infer_argmax_shape(node, ctx)

        # Scalar input returns shape
        assert results[0][0] == 5


class TestReduceOperations:
    """Test Reduce operations (ReduceMean, ReduceSum)."""

    def test_reducemean_scalar_input(self):
        """Test ReduceMean with scalar input."""
        import numpy as np

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean", inputs=["input", "axes"], outputs=["output"], keepdims=1
        )

        result = _infer_reduce_shape(node, ctx)
        # Scalar returns scalar
        assert result[0][0] == 5

    def test_reducemean_zero_dimension(self):
        """Test ReduceMean with zero dimension input."""
        import numpy as np

        axes_array = np.array([1], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceMean", inputs=["input", "axes"], outputs=["output"], keepdims=1
        )

        result = _infer_reduce_shape(node, ctx)
        # Zero dimension preserved
        assert result[0][0] == [0]

    def test_reducesum_multiple_axes(self):
        """Test ReduceSum with multiple axes."""
        import numpy as np

        axes_array = np.array([0, 2], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["input", "axes"], outputs=["output"], keepdims=0
        )

        result = _infer_reduce_shape(node, ctx)
        # Axes 0 and 2 reduced without keepdims: [3]
        assert result[0][0] == [3]

    def test_reducesum_missing_input_error(self):
        """Test ReduceSum raises error when input is missing."""
        import numpy as np

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["input", "axes"], outputs=["output"], keepdims=1
        )

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_reduce_shape(node, ctx)
