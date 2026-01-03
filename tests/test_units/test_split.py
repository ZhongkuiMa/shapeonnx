"""Unit tests for Split operation shape inference.

This module provides comprehensive test coverage for the Split operation,
which is used to split a tensor along a specified axis into multiple outputs.

Test organization:
- TestSplitBasic: Basic split operations along different axes
- TestSplitEdgeCases: Edge cases like zero dimensions
- TestSplitErrors: Error handling
"""

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_split_shape


def _make_split_initializer(split_sizes: list[int], name: str = "split") -> onnx.TensorProto:
    """Create a split sizes initializer tensor."""
    array = np.array(split_sizes, dtype=np.int64)
    return onnx.numpy_helper.from_array(array, name=name)


class TestSplitBasic:
    """Test basic Split operation with various configurations."""

    def test_split_axis0_equal_sizes(self):
        """Test split along axis 0 with equal sized chunks."""
        split_initializer = _make_split_initializer([5, 5])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 20]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [5, 20]
        assert results[1][0] == [5, 20]

    def test_split_axis0_unequal_sizes(self):
        """Test split along axis 0 with unequal sized chunks."""
        split_initializer = _make_split_initializer([3, 7])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 20]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [3, 20]
        assert results[1][0] == [7, 20]

    def test_split_axis1_basic(self):
        """Test split along axis 1."""
        split_initializer = _make_split_initializer([4, 4, 4])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 12]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2", "output3"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 3
        assert results[0][0] == [4, 4]
        assert results[1][0] == [4, 4]
        assert results[2][0] == [4, 4]

    def test_split_axis_negative(self):
        """Test split with negative axis."""
        split_initializer = _make_split_initializer([4, 4])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 8]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # axis=-1 is same as axis=1 for 2D tensor
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=-1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [4, 4]
        assert results[1][0] == [4, 4]

    def test_split_3d_tensor(self):
        """Test split on 3D tensor."""
        split_initializer = _make_split_initializer([1, 2])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [2, 1, 4]
        assert results[1][0] == [2, 2, 4]

    def test_split_4d_tensor(self):
        """Test split on 4D tensor (batch images)."""
        split_initializer = _make_split_initializer([1, 2])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 32, 32]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split along channel axis (axis=1)
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [1, 1, 32, 32]
        assert results[1][0] == [1, 2, 32, 32]


class TestSplitEdgeCases:
    """Test Split operation edge cases."""

    def test_split_single_output(self):
        """Test split producing single output (no actual split)."""
        split_initializer = _make_split_initializer([10])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 20]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 1
        assert results[0][0] == [10, 20]

    def test_split_with_batch_dimension(self):
        """Test split preserving batch dimension."""
        split_initializer = _make_split_initializer([8, 8])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 16, 32, 32]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split channels (axis=1) from 16 to 8+8
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output1", "output2"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [1, 8, 32, 32]
        assert results[1][0] == [1, 8, 32, 32]

    def test_split_many_outputs(self):
        """Test split producing many outputs."""
        split_initializer = _make_split_initializer([4, 4, 4, 4, 4])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [20, 10]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split into 5 parts along axis 0
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["o1", "o2", "o3", "o4", "o5"],
            axis=0,
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 5
        for result in results:
            assert result[0] == [4, 10]

    def test_split_1d_tensor(self):
        """Test split on 1D tensor."""
        split_initializer = _make_split_initializer([30, 30, 40])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [100]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2", "o3"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 3
        assert results[0][0] == [30]
        assert results[1][0] == [30]
        assert results[2][0] == [40]

    def test_split_dimension_one(self):
        """Test split where one dimension is 1."""
        split_initializer = _make_split_initializer([4, 6])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 10]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split along axis 1 (dimension with size 10)
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert len(results) == 2
        assert results[0][0] == [1, 4]
        assert results[1][0] == [1, 6]


class TestSplitErrors:
    """Test Split operation error handling."""

    def test_split_missing_input_shape(self):
        """Test error when input shape is missing."""
        split_initializer = _make_split_initializer([5, 5])
        ctx = ShapeInferenceContext(
            data_shapes={},  # No input shape
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["missing_input", "split"], outputs=["o1", "o2"], axis=0
        )

        with pytest.raises(RuntimeError):
            _infer_split_shape(node, ctx)

    def test_split_missing_initializer(self):
        """Test error when split initializer is missing."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 20]},
            explicit_shapes={},
            initializers={},  # No split initializer
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        with pytest.raises(RuntimeError, match="must be an initializer"):
            _infer_split_shape(node, ctx)


class TestSplitExplicitShapes:
    """Test Split with explicit shape handling."""

    def test_split_explicit_shapes_preserved(self):
        """Test that explicit shapes are not affected by split."""
        split_initializer = _make_split_initializer([5, 5])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 20]},
            explicit_shapes={},  # No explicit shapes for input
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        # Explicit shape should be None for data shape operations
        assert results[0][1] is None
        assert results[1][1] is None


class TestSplitIntegration:
    """Integration tests for Split with other operations."""

    def test_split_preserves_other_dimensions(self):
        """Test split preserves dimensions not being split."""
        split_initializer = _make_split_initializer([6, 4])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 10, 3, 4]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split along axis 1, other dimensions preserved
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=1
        )

        results = _infer_split_shape(node, ctx)

        assert results[0][0] == [2, 6, 3, 4]
        assert results[1][0] == [2, 4, 3, 4]

    def test_split_axis_zero_preserves_spatial(self):
        """Test split along axis 0 preserves spatial dimensions."""
        split_initializer = _make_split_initializer([2, 2])
        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 3, 224, 224]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # Split batch dimension
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        assert results[0][0] == [2, 3, 224, 224]
        assert results[1][0] == [2, 3, 224, 224]

    def test_split_zero_dimension(self):
        """Test Split with zero dimension input."""
        import onnx

        split_initializer = onnx.numpy_helper.from_array(
            np.array([2, 2], dtype=np.int64), name="split"
        )

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        results = _infer_split_shape(node, ctx)

        # Zero dimension propagates to all outputs
        assert results[0][0] == [0]
        assert results[1][0] == [0]

    def test_split_scalar_input_error(self):
        """Test Split raises error for scalar input."""
        import onnx

        split_initializer = onnx.numpy_helper.from_array(
            np.array([2, 2], dtype=np.int64), name="split"
        )

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar input
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        with pytest.raises(RuntimeError, match="Split input shape cannot be scalar"):
            _infer_split_shape(node, ctx)

    def test_split_missing_input_error(self):
        """Test Split raises error when input is missing."""
        import onnx

        split_initializer = onnx.numpy_helper.from_array(
            np.array([2, 2], dtype=np.int64), name="split"
        )

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0
        )

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_split_shape(node, ctx)

    def test_split_num_outputs_not_supported_error(self):
        """Test Split with num_outputs raises NotImplementedError."""
        import onnx

        split_initializer = onnx.numpy_helper.from_array(
            np.array([2, 2], dtype=np.int64), name="split"
        )

        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 5]},
            explicit_shapes={},
            initializers={"split": split_initializer},
            verbose=False,
        )

        # num_outputs attribute instead of split tensor
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["o1", "o2"], axis=0, num_outputs=2
        )

        with pytest.raises(NotImplementedError, match="Split with num_outputs"):
            _infer_split_shape(node, ctx)
