"""Direct unit tests for shapeonnx.infer_shape helpers.

Most ``_infer_*_shape`` functions are exercised indirectly via the
operation-specific test files (test_binary_ops, test_concat, test_reduce, etc.).
This module fills the gap by directly exercising the pure helpers and the
``ShapeInferenceContext`` dataclass that supports them: shape lookup,
broadcasting alignment, scalar arithmetic, initializer conversion, and the
``infer_onnx_shape`` public-API entrypoint on a minimal synthetic graph.
"""

import dataclasses

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _align_shapes,
    _compute_binary_op_value,
    _extract_initializer_shapes,
    _get_data_shape,
    _get_explicit_shape,
    _get_shape,
    _preconvert_integer_initializers,
    _right_align_shapes,
    _store_data_shape,
    _store_explicit_shape,
    extract_io_shapes,
    infer_onnx_shape,
)


def _make_int_initializer(
    values: list[int], name: str, dtype: int = TensorProto.INT64
) -> onnx.TensorProto:
    """Build a 1-D integer TensorProto initializer.

    :param values: integer values to store
    :param name: initializer name
    :param dtype: ONNX TensorProto dtype (default INT64)
    :return: TensorProto holding ``values``
    """
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)
    array = np.array(values, dtype=np_dtype)
    return onnx.numpy_helper.from_array(array, name=name)


class TestShapeInferenceContext:
    """Tests for the ShapeInferenceContext dataclass."""

    def test_context_is_frozen(self) -> None:
        """Verify the dataclass is frozen and rejects mutation."""
        ctx = ShapeInferenceContext(
            data_shapes={"x": [1, 2]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.verbose = True  # type: ignore[misc]

    def test_context_stores_dicts_by_reference(self) -> None:
        """Verify the dict fields keep their identity (frozen != deep-copy)."""
        data: dict[str, int | list[int]] = {"x": [3, 4]}
        ctx = ShapeInferenceContext(
            data_shapes=data,
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        assert ctx.data_shapes is data
        assert ctx.verbose is False

    def test_context_default_verbose(self) -> None:
        """Verify ``verbose`` defaults to False when not supplied."""
        ctx = ShapeInferenceContext(data_shapes={}, explicit_shapes={}, initializers={})
        assert ctx.verbose is False


class TestExtractIoShapes:
    """Tests for extract_io_shapes."""

    def test_extract_io_shapes_returns_named_dict(self) -> None:
        """Verify the helper maps node names to their reformatted shapes."""
        info = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])

        result = extract_io_shapes([info], has_batch_dim=True)

        assert "input" in result
        assert isinstance(result["input"], list)
        assert result["input"][-1] == 224

    def test_extract_io_shapes_empty_input(self) -> None:
        """Verify empty input list returns an empty mapping."""
        assert extract_io_shapes([], has_batch_dim=False) == {}


class TestExtractInitializerShapes:
    """Tests for _extract_initializer_shapes."""

    def test_extract_returns_dim_lists(self) -> None:
        """Verify each initializer is mapped to its dim list."""
        init = _make_int_initializer([1, 2, 3, 4], "weights")
        result = _extract_initializer_shapes({"weights": init})
        assert result == {"weights": [4]}

    def test_extract_empty_dict(self) -> None:
        """Verify empty initializer dict round-trips to empty mapping."""
        assert _extract_initializer_shapes({}) == {}


class TestGetShape:
    """Tests for _get_data_shape, _get_explicit_shape, and _get_shape."""

    def test_get_data_shape_returns_value(self) -> None:
        """Verify _get_data_shape forwards to dict lookup."""
        assert _get_data_shape("x", {"x": [1, 2]}) == [1, 2]

    def test_get_data_shape_missing_returns_none(self) -> None:
        """Verify _get_data_shape returns None for unknown keys."""
        assert _get_data_shape("missing", {}) is None

    def test_get_explicit_shape_returns_value(self) -> None:
        """Verify _get_explicit_shape forwards to dict lookup."""
        assert _get_explicit_shape("k", {"k": 7}) == 7

    def test_get_shape_prefers_data_over_explicit(self) -> None:
        """Verify _get_shape returns the data shape with is_explicit=False."""
        shape, is_explicit = _get_shape("x", {"x": [3]}, {"x": 99})
        assert shape == [3]
        assert is_explicit is False

    def test_get_shape_falls_back_to_explicit(self) -> None:
        """Verify _get_shape returns explicit shape with is_explicit=True."""
        shape, is_explicit = _get_shape("k", {}, {"k": 5})
        assert shape == 5
        assert is_explicit is True

    def test_get_shape_missing_raises(self) -> None:
        """Verify _get_shape raises RuntimeError when no source has the key."""
        with pytest.raises(RuntimeError, match="Cannot get shape of missing"):
            _get_shape("missing", {}, {})


class TestStoreShape:
    """Tests for _store_data_shape and _store_explicit_shape."""

    def test_store_data_shape_writes_in_place(self) -> None:
        """Verify _store_data_shape mutates the supplied dict."""
        store: dict[str, list[int]] = {}
        _store_data_shape([1, 2], store, "out")
        assert store == {"out": [1, 2]}

    def test_store_explicit_shape_writes_in_place(self) -> None:
        """Verify _store_explicit_shape mutates the supplied dict."""
        store: dict[str, int | list[int]] = {}
        _store_explicit_shape(42, store, "k")
        _store_explicit_shape([1, 1], store, "shape")
        assert store == {"k": 42, "shape": [1, 1]}


class TestPreconvertIntegerInitializers:
    """Tests for _preconvert_integer_initializers."""

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            (TensorProto.INT8, [1, 2, 3]),
            (TensorProto.INT16, [10, 20, 30]),
            (TensorProto.INT32, [100, 200, 300]),
            (TensorProto.INT64, [1000, 2000, 3000]),
            (TensorProto.UINT8, [1, 2, 3]),
        ],
    )
    def test_integer_dtypes_are_converted(self, dtype: int, values: list[int]) -> None:
        """Verify integer-typed initializers are converted to Python lists.

        :param dtype: ONNX integer dtype
        :param values: integer payload
        """
        init = _make_int_initializer(values, "k", dtype=dtype)

        result = _preconvert_integer_initializers({"k": init})

        assert result["k"] == values

    def test_float_initializers_are_skipped(self) -> None:
        """Verify float initializers are left out of the conversion result."""
        array = np.array([1.0, 2.0], dtype=np.float32)
        init = onnx.numpy_helper.from_array(array, name="weights")

        result = _preconvert_integer_initializers({"weights": init})

        assert result == {}

    def test_mixed_inputs(self) -> None:
        """Verify only integer initializers appear in the output mapping."""
        int_init = _make_int_initializer([7], "axes")
        array = np.array([1.0], dtype=np.float32)
        float_init = onnx.numpy_helper.from_array(array, name="w")

        result = _preconvert_integer_initializers({"axes": int_init, "w": float_init})

        assert result == {"axes": [7]}


_RIGHT_ALIGN_CASES = [
    pytest.param([3, 4], [3, 4], ([3, 4], [3, 4]), id="same_rank"),
    pytest.param([4], [3, 4], ([1, 4], [3, 4]), id="pad_left_shape1"),
    pytest.param([3, 4], [4], ([3, 4], [1, 4]), id="pad_left_shape2"),
    pytest.param([], [2, 3], ([1, 1], [2, 3]), id="empty_padded_to_target"),
]


class TestRightAlignShapes:
    """Tests for _right_align_shapes."""

    @pytest.mark.parametrize(("shape1", "shape2", "expected"), _RIGHT_ALIGN_CASES)
    def test_right_align_pads_with_ones(
        self,
        shape1: list[int],
        shape2: list[int],
        expected: tuple[list[int], list[int]],
    ) -> None:
        """Verify _right_align_shapes left-pads the shorter shape with 1s.

        :param shape1: first input shape
        :param shape2: second input shape
        :param expected: expected pair of right-aligned shapes
        """
        assert _right_align_shapes(shape1, shape2) == expected


_ALIGN_CASES = [
    pytest.param([3, 4], [3, 4], [3, 4], id="same_shape"),
    pytest.param([3, 4], [4], [1, 4], id="target_shorter"),
    pytest.param([2, 3, 4], [3], [1, 3, 1], id="middle_match"),
]


class TestAlignShapes:
    """Tests for _align_shapes."""

    @pytest.mark.parametrize(("base", "target", "expected"), _ALIGN_CASES)
    def test_align_returns_expected(
        self, base: list[int], target: list[int], expected: list[int]
    ) -> None:
        """Verify _align_shapes maps target dims into base structure.

        :param base: base shape providing structure
        :param target: target shape to align
        :param expected: expected aligned shape
        """
        assert _align_shapes(base, target) == expected


_BINARY_VALUE_CASES = [
    pytest.param("Add", 2, 3, 5, int, id="add_int"),
    pytest.param("Sub", 5, 2, 3, int, id="sub_int"),
    pytest.param("Mul", 4, 3, 12, int, id="mul_int"),
    pytest.param("Div", 6, 2, 3, int, id="div_int_clean"),
    pytest.param("Add", 2.0, 3, 5.0, float, id="add_mixed_promotes_to_float"),
    pytest.param("Mul", 2.5, 4.0, 10.0, float, id="mul_float"),
]


class TestComputeBinaryOpValue:
    """Tests for _compute_binary_op_value scalar arithmetic."""

    @pytest.mark.parametrize(
        ("op_type", "value1", "value2", "expected", "expected_type"), _BINARY_VALUE_CASES
    )
    def test_binary_op_value(
        self,
        op_type: str,
        value1: int | float,
        value2: int | float,
        expected: int | float,
        expected_type: type,
    ) -> None:
        """Verify _compute_binary_op_value returns expected value and type.

        :param op_type: ONNX op name
        :param value1: first operand
        :param value2: second operand
        :param expected: expected numeric result
        :param expected_type: expected Python type of the result
        """
        result = _compute_binary_op_value(op_type, value1, value2)

        assert result == expected
        assert isinstance(result, expected_type)

    def test_unsupported_op_raises(self) -> None:
        """Verify an unknown op type triggers a RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot calculate Pow"):
            _compute_binary_op_value("Pow", 2, 3)


class TestInferOnnxShapePublicApi:
    """Smoke tests for the public infer_onnx_shape entrypoint on a tiny graph."""

    @staticmethod
    def _build_relu_graph() -> tuple[
        list[onnx.ValueInfoProto],
        list[onnx.ValueInfoProto],
        list[onnx.NodeProto],
        dict[str, onnx.TensorProto],
    ]:
        """Build a 1-node Relu(input) -> output graph and return its parts."""
        relu = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
        input_info = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
        output_info = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
        return [input_info], [output_info], [relu], {}

    def test_infer_onnx_shape_returns_node_shapes(self) -> None:
        """Verify infer_onnx_shape returns shapes for input and Relu output."""
        inputs, outputs, nodes, inits = self._build_relu_graph()

        shapes = infer_onnx_shape(inputs, outputs, nodes, inits, has_batch_dim=True)

        assert shapes["output"] == [1, 4, 8, 8]
        assert shapes["input"] == [1, 4, 8, 8]
