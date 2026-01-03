"""Unit tests for Where (conditional) operator shape inference.

This module provides comprehensive test coverage for the Where operation,
which selects elements from two inputs based on a condition tensor.

Test organization:
- TestWhereBasic: Basic Where operations with various shape configurations
- TestWhereBroadcasting: Where with broadcasting between inputs
- TestWhereExplicitShapes: Where with explicit shape value computation
- TestWhereErrors: Error handling
"""

import onnx
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_where_shape


class TestWhereBasic:
    """Test basic Where operation with various configurations."""

    def test_where_same_shape(self):
        """Test Where with all inputs having the same shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [2, 3, 4],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert len(results) == 1
        assert results[0][0] == [2, 3, 4]

    def test_where_x_broadcast(self):
        """Test Where where x is broadcasted - returns x shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [1],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Where returns x shape if x is non-zero shape
        assert results[0][0] == [1]

    def test_where_y_broadcast(self):
        """Test Where where y is broadcasted."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [2, 3, 4],
                "y": [3, 1],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert results[0][0] == [2, 3, 4]

    def test_where_both_broadcast(self):
        """Test Where where both x and y are broadcasted - returns x shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [1, 1],
                "y": [4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Where returns x shape (first non-zero shape encountered)
        assert results[0][0] == [1, 1]

    def test_where_scalar_condition(self):
        """Test Where with scalar condition."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": 1,
                "x": [2, 3, 4],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert results[0][0] == [2, 3, 4]

    def test_where_different_ranks(self):
        """Test Where with different rank inputs - returns x shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [1, 4, 1],
                "x": [3],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Where returns x shape if non-zero
        assert results[0][0] == [3]

    def test_where_batch_dimension(self):
        """Test Where with batch dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [4, 3, 224, 224],
                "x": [4, 3, 224, 224],
                "y": [4, 3, 224, 224],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert results[0][0] == [4, 3, 224, 224]


class TestWhereExplicitShapes:
    """Test Where with explicit shape handling."""

    def test_where_explicit_shapes_preserved(self):
        """Test that explicit shapes are not used in Where (data shapes only)."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [2, 3, 4],
                "y": [2, 3, 4],
            },
            explicit_shapes={},  # Empty
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Result should have no explicit shape
        assert results[0][1] is None
        assert results[0][0] == [2, 3, 4]

    def test_where_1d_explicit_shape(self):
        """Test Where with 1D explicit shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [5],
                "x": [5],
                "y": [5],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert results[0][0] == [5]
        assert results[0][1] is None


class TestWhereErrors:
    """Test Where operation error handling."""

    def test_where_missing_condition_shape(self):
        """Test Where when condition shape is missing (not required)."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "x": [2, 3, 4],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Where", inputs=["missing_condition", "x", "y"], outputs=["output"]
        )

        # Where doesn't validate condition shape, only uses x and y for output
        results = _infer_where_shape(node, ctx)
        assert results[0][0] == [2, 3, 4]

    def test_where_missing_x_shape(self):
        """Test error when x shape is missing."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "y": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Where", inputs=["condition", "missing_x", "y"], outputs=["output"]
        )

        with pytest.raises(RuntimeError):
            _infer_where_shape(node, ctx)

    def test_where_missing_y_shape(self):
        """Test error when y shape is missing."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3, 4],
                "x": [2, 3, 4],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Where", inputs=["condition", "x", "missing_y"], outputs=["output"]
        )

        with pytest.raises(RuntimeError):
            _infer_where_shape(node, ctx)


class TestWhereIntegration:
    """Integration tests for Where operation."""

    def test_where_broadcast_all_axes(self):
        """Test Where with broadcasting - returns x shape."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [1, 1, 4, 1],
                "x": [2, 1, 1, 5],
                "y": [1, 3, 4, 5],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Where returns x shape if non-zero
        assert results[0][0] == [2, 1, 1, 5]

    def test_where_preserve_batch_size(self):
        """Test Where preserves batch size dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [32, 10],
                "x": [32, 10],
                "y": [32, 1],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        assert results[0][0] == [32, 10]

    def test_where_x_scalar_returns_zero(self):
        """Test Where with scalar x value returns [0]."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [3, 4],  # List
                "x": 5,  # Scalar
                "y": [3, 4],  # List
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Mismatched scalar and list in x returns [0]
        assert results[0][0] == [0]

    def test_where_all_zero_dimension(self):
        """Test Where when all inputs have zero dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [0],
                "x": [0],
                "y": [0],
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Zero dimension propagates
        assert results[0][0] == [0]

    def test_where_explicit_shape_computation(self):
        """Test Where with explicit shape value computation."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [2, 3],
                "x": [2, 3],
                "y": [2, 3],
            },
            explicit_shapes={
                "condition": [1, 0],  # condition[0]=1, condition[1]=0
                "x": [10, 20],  # x values
                "y": [30, 40],  # y values
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # Where selects: condition[0]=1 -> x[0]=10, condition[1]=0 -> y[1]=40
        # explicit result: [10, 40]
        assert results[0][1] == [10, 40]

    def test_where_explicit_shape_all_condition_true(self):
        """Test Where with all true condition in explicit shapes."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [3],
                "x": [3],
                "y": [3],
            },
            explicit_shapes={
                "condition": [1, 1, 1],  # All true
                "x": [5, 6, 7],
                "y": [10, 11, 12],
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # All true -> all from x: [5, 6, 7]
        assert results[0][1] == [5, 6, 7]

    def test_where_explicit_shape_all_condition_false(self):
        """Test Where with all false condition in explicit shapes."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [3],
                "x": [3],
                "y": [3],
            },
            explicit_shapes={
                "condition": [0, 0, 0],  # All false
                "x": [5, 6, 7],
                "y": [10, 11, 12],
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])

        results = _infer_where_shape(node, ctx)

        # All false -> all from y: [10, 11, 12]
        assert results[0][1] == [10, 11, 12]
