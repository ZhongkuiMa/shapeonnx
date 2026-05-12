"""Unit tests for Where (conditional) operator shape inference.

This module provides comprehensive test coverage for the Where operation,
which selects elements from two inputs based on a condition tensor.

Test organization:
- TestWhereBasic: Basic Where operations with various shape configurations
- TestWhereBroadcasting: Where with broadcasting between inputs
- TestWhereExplicitShapes: Where with explicit shape value computation
- TestWhereErrors: Error handling
"""

__docformat__ = "restructuredtext"

import onnx
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_where_shape


class TestWhereBasic:
    """Test basic Where operation with various configurations."""

    @pytest.mark.parametrize(
        ("condition_shape", "x_shape", "y_shape", "expected"),
        [
            pytest.param([2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], id="same_shape"),
            pytest.param([2, 3, 4], [1], [2, 3, 4], [1], id="x_broadcast"),
            pytest.param([2, 3, 4], [2, 3, 4], [3, 1], [2, 3, 4], id="y_broadcast"),
            pytest.param([2, 3, 4], [1, 1], [4], [1, 1], id="both_broadcast"),
            pytest.param(1, [2, 3, 4], [2, 3, 4], [2, 3, 4], id="scalar_condition"),
            pytest.param([1, 4, 1], [3], [2, 3, 4], [3], id="different_ranks"),
            pytest.param(
                [4, 3, 224, 224],
                [4, 3, 224, 224],
                [4, 3, 224, 224],
                [4, 3, 224, 224],
                id="batch_dimension",
            ),
        ],
    )
    def test_where_shape_output(self, condition_shape, x_shape, y_shape, expected):
        """Test Where output shape for various input configurations."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": condition_shape,
                "x": x_shape,
                "y": y_shape,
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])
        results = _infer_where_shape(node, ctx)

        assert len(results) >= 1
        assert results[0][0] == expected


class TestWhereExplicitShapes:
    """Test Where with explicit shape handling."""

    @pytest.mark.parametrize(
        ("condition_shape", "x_shape", "y_shape"),
        [
            pytest.param([2, 3, 4], [2, 3, 4], [2, 3, 4], id="explicit_shapes_preserved"),
            pytest.param([5], [5], [5], id="explicit_shapes_1d"),
        ],
    )
    def test_where_explicit_shapes_not_used(self, condition_shape, x_shape, y_shape):
        """Test that explicit shapes are not used in Where (data shapes only)."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": condition_shape,
                "x": x_shape,
                "y": y_shape,
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])
        results = _infer_where_shape(node, ctx)

        # Result should have no explicit shape
        assert results[0][1] is None
        assert results[0][0] == condition_shape


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

    @pytest.mark.parametrize(
        "node_inputs",
        [
            pytest.param(["condition", "missing_x", "y"], id="missing_x"),
            pytest.param(["condition", "x", "missing_y"], id="missing_y"),
        ],
    )
    def test_where_missing_input_raises(self, node_inputs):
        """Test error when an input shape is missing."""
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

        node = onnx.helper.make_node("Where", inputs=node_inputs, outputs=["output"])

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_where_shape(node, ctx)


class TestWhereIntegration:
    """Integration tests for Where operation."""

    @pytest.mark.parametrize(
        ("condition_shape", "x_shape", "y_shape", "expected"),
        [
            pytest.param(
                [1, 1, 4, 1], [2, 1, 1, 5], [1, 3, 4, 5], [2, 1, 1, 5], id="broadcast_all_axes"
            ),
            pytest.param([32, 10], [32, 10], [32, 1], [32, 10], id="preserve_batch_size"),
            pytest.param([3, 4], 5, [3, 4], [0], id="x_scalar_returns_zero"),
            pytest.param([0], [0], [0], [0], id="all_zero_dimension"),
        ],
    )
    def test_where_various_inputs(self, condition_shape, x_shape, y_shape, expected):
        """Test Where output shape for various input configurations."""
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": condition_shape,
                "x": x_shape,
                "y": y_shape,
            },
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])
        results = _infer_where_shape(node, ctx)

        assert results[0][0] == expected

    @pytest.mark.parametrize(
        ("condition_explicit", "x_explicit", "y_explicit", "expected"),
        [
            pytest.param([1, 0], [10, 20], [30, 40], [10, 40], id="explicit_shape_computation"),
            pytest.param([1, 1, 1], [5, 6, 7], [10, 11, 12], [5, 6, 7], id="all_condition_true"),
            pytest.param(
                [0, 0, 0], [5, 6, 7], [10, 11, 12], [10, 11, 12], id="all_condition_false"
            ),
        ],
    )
    def test_where_explicit_shape_computation(
        self, condition_explicit, x_explicit, y_explicit, expected
    ):
        """Test Where with explicit shape value computation."""
        n = len(expected)
        ctx = ShapeInferenceContext(
            data_shapes={
                "condition": [n],
                "x": [n],
                "y": [n],
            },
            explicit_shapes={
                "condition": condition_explicit,
                "x": x_explicit,
                "y": y_explicit,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["output"])
        results = _infer_where_shape(node, ctx)

        assert results[0][1] == expected
