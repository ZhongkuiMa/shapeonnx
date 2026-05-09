"""Unit tests for Reduce operations (ReduceMean, ReduceSum, ArgMax) shape inference.

This module tests reduce operations that aggregate values across tensor axes.

Test organization:
- TestReduceShape: parametrized golden-path coverage for ReduceMean/ReduceSum
- TestReduceErrors: error handling for missing input shapes
- TestReduceEdgeCases: scalar and zero-dimension inputs
- TestArgMaxOperation: ArgMax shape inference
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
    """Create an axes initializer tensor.

    :param axes: list of axis indices (may be negative)
    :param name: initializer name
    :return: ONNX TensorProto holding ``axes`` as int64
    """
    array = np.array(axes, dtype=np.int64)
    return onnx.numpy_helper.from_array(array, name=name)


def _make_reduce_ctx(input_shape: list[int] | int, axes: list[int]) -> ShapeInferenceContext:
    """Build a ShapeInferenceContext for a single reduce-op test case.

    :param input_shape: shape (or scalar) attached to ``input``
    :param axes: axes initializer values
    :return: populated ShapeInferenceContext
    """
    return ShapeInferenceContext(
        data_shapes={"input": input_shape},
        explicit_shapes={},
        initializers={"axes": _make_axes_initializer(axes)},
        verbose=False,
    )


_REDUCE_CASES = [
    # ReduceMean: single-axis variants
    pytest.param("ReduceMean", [2, 3, 4], [1], 1, [2, 1, 4], id="mean_axis1_keepdims"),
    pytest.param("ReduceMean", [2, 3, 4], [1], 0, [2, 4], id="mean_axis1_no_keepdims"),
    pytest.param("ReduceMean", [4, 3, 224, 224], [0], 1, [1, 3, 224, 224], id="mean_first_axis"),
    pytest.param("ReduceMean", [2, 3, 4, 5], [3], 0, [2, 3, 4], id="mean_last_axis"),
    pytest.param("ReduceMean", [2, 3, 4], [-1], 1, [2, 3, 1], id="mean_negative_axis"),
    # ReduceSum: single-axis variants
    pytest.param("ReduceSum", [2, 3, 4], [1], 1, [2, 1, 4], id="sum_axis1_keepdims"),
    pytest.param("ReduceSum", [2, 3, 4], [1], 0, [2, 4], id="sum_axis1_no_keepdims"),
    pytest.param("ReduceSum", [32, 256, 224, 224], [1], 0, [32, 224, 224], id="sum_drop_channel"),
    # Multi-axis variants
    pytest.param(
        "ReduceMean",
        [2, 3, 4, 5],
        [1, 3],
        1,
        [2, 1, 4, 1],
        id="mean_multi_axis_keepdims",
    ),
    pytest.param("ReduceMean", [2, 3, 4, 5], [1, 3], 0, [2, 4], id="mean_multi_axis_no_keepdims"),
    pytest.param("ReduceSum", [1, 64, 28, 28], [2, 3], 1, [1, 64, 1, 1], id="sum_spatial_axes"),
    # All-axes variants
    pytest.param("ReduceMean", [2, 3, 4], [0, 1, 2], 1, [1, 1, 1], id="mean_all_keepdims"),
    pytest.param("ReduceMean", [2, 3, 4], [0, 1, 2], 0, [], id="mean_all_no_keepdims"),
    pytest.param("ReduceSum", [4, 3, 224, 224], [0, 1, 2, 3], 0, [], id="sum_all_no_keepdims"),
    # Integration scenarios
    pytest.param(
        "ReduceMean",
        [32, 128, 7, 7],
        [2, 3],
        1,
        [32, 128, 1, 1],
        id="mean_channel_pooling",
    ),
    pytest.param("ReduceSum", [1, 512, 14, 14], [0, 2, 3], 0, [512], id="sum_global_pool"),
    pytest.param("ReduceSum", [2, 3, 4], [0, 2], 0, [3], id="sum_drop_outer_inner"),
]


class TestReduceShape:
    """Parametrized tests for ReduceMean / ReduceSum shape inference."""

    @pytest.mark.parametrize(
        ("op_type", "input_shape", "axes", "keepdims", "expected"), _REDUCE_CASES
    )
    def test_reduce_returns_expected_shape(
        self,
        op_type: str,
        input_shape: list[int],
        axes: list[int],
        keepdims: int,
        expected: list[int],
    ) -> None:
        """Verify _infer_reduce_shape returns the expected output shape.

        :param op_type: ONNX op type, "ReduceMean" or "ReduceSum"
        :param input_shape: shape attached to "input"
        :param axes: axes initializer values
        :param keepdims: 0 or 1, ONNX keepdims attribute
        :param expected: expected output shape
        """
        ctx = _make_reduce_ctx(input_shape, axes)
        node = onnx.helper.make_node(
            op_type, inputs=["input", "axes"], outputs=["output"], keepdims=keepdims
        )

        results = _infer_reduce_shape(node, ctx)

        assert len(results) >= 1
        assert results[0][0] == expected


class TestReduceErrors:
    """Error handling for Reduce operations."""

    @pytest.mark.parametrize("op_type", ["ReduceMean", "ReduceSum"])
    def test_missing_input_shape_raises(self, op_type: str) -> None:
        """Verify _infer_reduce_shape rejects missing input shapes.

        :param op_type: ONNX op type to test
        """
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={"axes": _make_axes_initializer([0])},
            verbose=False,
        )
        node = onnx.helper.make_node(
            op_type, inputs=["missing_input", "axes"], outputs=["output"], keepdims=1
        )

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_reduce_shape(node, ctx)


class TestReduceEdgeCases:
    """Edge-case inputs for Reduce operations."""

    def test_reducemean_scalar_input_returns_scalar(self) -> None:
        """Verify scalar input flows through ReduceMean unchanged."""
        ctx = _make_reduce_ctx(5, [0])
        node = onnx.helper.make_node(
            "ReduceMean", inputs=["input", "axes"], outputs=["output"], keepdims=1
        )

        result = _infer_reduce_shape(node, ctx)

        assert result[0][0] == 5

    def test_reducemean_zero_dim_preserved(self) -> None:
        """Verify zero-dimension axes are preserved through ReduceMean."""
        ctx = _make_reduce_ctx([0], [1])
        node = onnx.helper.make_node(
            "ReduceMean", inputs=["input", "axes"], outputs=["output"], keepdims=1
        )

        result = _infer_reduce_shape(node, ctx)

        assert result[0][0] == [0]


_ARGMAX_CASES = [
    pytest.param([5, 4, 3], 0, 1, [1, 4, 3], id="axis0_keepdims"),
    pytest.param([2, 3, 4], 1, 0, [2, 4], id="axis1_no_keepdims"),
    pytest.param([3, 4, 5], 2, 1, [3, 4, 1], id="last_axis"),
]


class TestArgMaxOperation:
    """Tests for ArgMax shape inference."""

    @pytest.mark.parametrize(("input_shape", "axis", "keepdims", "expected"), _ARGMAX_CASES)
    def test_argmax_returns_expected_shape(
        self,
        input_shape: list[int],
        axis: int,
        keepdims: int,
        expected: list[int],
    ) -> None:
        """Verify _infer_argmax_shape returns the expected reduced shape.

        :param input_shape: shape attached to "input"
        :param axis: ArgMax axis attribute
        :param keepdims: 0 or 1
        :param expected: expected output shape
        """
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            axis=axis,
            keepdims=keepdims,
        )

        results = _infer_argmax_shape(node, ctx)

        assert results[0][0] == expected

    def test_argmax_scalar_input_returns_scalar(self) -> None:
        """Verify ArgMax leaves scalar inputs unchanged."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "ArgMax", inputs=["input"], outputs=["output"], axis=0, keepdims=1
        )

        results = _infer_argmax_shape(node, ctx)

        assert results[0][0] == 5
