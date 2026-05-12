"""Targeted tests for coverage gaps in _infer_binary_op_shape and _infer_concat_shape.

These cases exercise rarely-hit branches: empty-list inputs, scalar value
computation, mixed scalar/list explicit shapes, and Concat early-return paths
with zero dimensions.
"""

__docformat__ = "restructuredtext"

import onnx
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_binary_op_shape,
    _infer_concat_shape,
)


def _make_binary_ctx(
    data_shapes: dict[str, int | list[int]],
    explicit_shapes: dict[str, int | list[int]] | None = None,
) -> ShapeInferenceContext:
    """Build a minimal context for binary-op tests.

    :param data_shapes: shapes attached to "a" / "b".

    :param explicit_shapes: optional explicit shapes.

    :return: populated ShapeInferenceContext
    """
    return ShapeInferenceContext(
        data_shapes=data_shapes,
        explicit_shapes=explicit_shapes or {},
        initializers={},
        verbose=False,
    )


# (op_type, data_shapes, explicit_shapes, result_idx, expected)
# result_idx selects which slot of result[0] to inspect — 0 for shape, 1 for scalar value.
_BINARY_OP_CASES = [
    # Both empty lists in data_shapes -> result is []
    pytest.param("Add", {"a": [], "b": []}, None, 0, [], id="add_both_empty_lists"),
    pytest.param("Sub", {"a": [], "b": []}, None, 0, [], id="sub_both_empty_lists"),
    pytest.param("Mul", {"a": [], "b": []}, None, 0, [], id="mul_both_empty_lists"),
    # Both scalars explicit -> compute the scalar value
    pytest.param("Add", {}, {"a": 2, "b": 3}, 1, 5, id="add_scalars_compute"),
    pytest.param("Sub", {}, {"a": 5, "b": 2}, 1, 3, id="sub_scalars_compute"),
    pytest.param("Div", {}, {"a": 6, "b": 2}, 1, 3, id="div_scalars_compute"),
    # One explicit scalar + empty data list
    pytest.param("Add", {"a": [], "b": []}, {"a": 5}, 0, [], id="add_explicit_scalar_one_side"),
    pytest.param("Mul", {"a": [], "b": []}, {"a": 2, "b": []}, 0, [], id="mul_scalar_and_empty"),
    # Scalar + list explicit shapes -> broadcast against data shape [3]
    pytest.param(
        "Add",
        {"a": [3], "b": [3]},
        {"a": 5, "b": [2, 3]},
        0,
        [3],
        id="add_scalar_then_list",
    ),
    pytest.param(
        "Sub",
        {"a": [3], "b": [3]},
        {"a": [2, 3], "b": 5},
        0,
        [3],
        id="sub_list_then_scalar",
    ),
    # Empty-list explicit shapes fall back to data shape [3]
    pytest.param(
        "Add",
        {"a": [3], "b": [3]},
        {"a": [], "b": 5},
        0,
        [3],
        id="add_empty_explicit_and_scalar",
    ),
    pytest.param(
        "Mul",
        {"a": [3], "b": [3]},
        {"a": [], "b": []},
        0,
        [3],
        id="mul_both_empty_explicit",
    ),
]


class TestBinaryOpCoverageGaps:
    """Branch coverage for _infer_binary_op_shape."""

    @pytest.mark.parametrize(
        ("op_type", "data_shapes", "explicit_shapes", "result_idx", "expected"),
        _BINARY_OP_CASES,
    )
    def test_binary_op_returns_expected(
        self,
        op_type: str,
        data_shapes: dict[str, int | list[int]],
        explicit_shapes: dict[str, int | list[int]] | None,
        result_idx: int,
        expected: int | list[int],
    ) -> None:
        """Verify _infer_binary_op_shape covers empty-list, scalar-compute, and mixed cases.

        :param op_type: ONNX op name.

        :param data_shapes: shapes attached to "a" / "b".

        :param explicit_shapes: optional explicit shapes.

        :param result_idx: 0 for shape slot, 1 for scalar value slot.

        :param expected: expected value at ``result[0][result_idx]``.

        """
        ctx = _make_binary_ctx(data_shapes, explicit_shapes)
        node = onnx.helper.make_node(op_type, inputs=["a", "b"], outputs=["output"])

        result = _infer_binary_op_shape(node, ctx)

        assert len(result) >= 1
        assert result[0][result_idx] == expected


_CONCAT_ZERO_DIM_CASES = [
    pytest.param(
        {"a": [2, 3], "b": [2, 4]},
        {"a": [0]},
        1,
        [0],
        id="explicit_zero_early_return",
    ),
    pytest.param(
        {"a": [0], "b": [2, 4]},
        {},
        0,
        [0],
        id="data_zero_early_return",
    ),
]


class TestConcatZeroDimensionEarlyReturn:
    """Concat early-return paths when an input shape is [0]."""

    @pytest.mark.parametrize(
        ("data_shapes", "explicit_shapes", "axis", "expected"), _CONCAT_ZERO_DIM_CASES
    )
    def test_concat_zero_dim_returns_early(
        self,
        data_shapes: dict[str, int | list[int]],
        explicit_shapes: dict[str, int | list[int]],
        axis: int,
        expected: list[int],
    ) -> None:
        """Verify _infer_concat_shape short-circuits to [0] when any input is [0].

        :param data_shapes: shapes attached to "a" / "b".

        :param explicit_shapes: optional explicit shapes.

        :param axis: Concat axis attribute.

        :param expected: expected result shape.

        """
        ctx = ShapeInferenceContext(
            data_shapes=data_shapes,
            explicit_shapes=explicit_shapes,
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=["a", "b"], outputs=["output"], axis=axis)

        result = _infer_concat_shape(node, ctx)

        assert len(result) >= 1
        assert result[0][0] == expected
