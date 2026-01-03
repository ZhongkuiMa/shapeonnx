"""Targeted tests for coverage gaps to reach 95%+ coverage."""

import onnx

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_binary_op_shape,
    _infer_concat_shape,
)


class TestBinaryOpsEmptyLists:
    """Test binary operations with empty list inputs."""

    def test_add_both_empty_lists(self):
        """Test Add when both inputs are empty lists."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both empty lists: result is []
        assert result[0][0] == []

    def test_sub_both_empty_lists(self):
        """Test Sub when both inputs are empty lists."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sub", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both empty lists: result is []
        assert result[0][0] == []

    def test_mul_both_empty_lists(self):
        """Test Mul when both inputs are empty lists."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both empty lists: result is []
        assert result[0][0] == []


class TestBinaryOpsScalarCompute:
    """Test binary operations with scalar value computation."""

    def test_add_scalars_compute(self):
        """Test Add computing scalar values (2+3=5)."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"a": 2, "b": 3},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both scalars explicit: compute 2+3=5
        assert result[0][1] == 5

    def test_sub_scalars_compute(self):
        """Test Sub computing scalar values (5-2=3)."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"a": 5, "b": 2},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sub", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both scalars explicit: compute 5-2=3
        assert result[0][1] == 3

    def test_div_scalars_compute(self):
        """Test Div computing scalar values (6/2=3)."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"a": 6, "b": 2},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Div", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both scalars explicit: compute 6/2=3
        assert result[0][1] == 3


class TestBinaryOpsPartialExplicit:
    """Test binary operations with one explicit, one missing."""

    def test_add_one_explicit_one_empty_list(self):
        """Test Add where one input has explicit scalar, other has empty list."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={"a": 5},  # a is explicit scalar
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # One is scalar explicit, other is empty list: returns []
        assert result[0][0] == []

    def test_mul_scalar_and_empty_list(self):
        """Test Mul with scalar explicit and empty list."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={"a": 2, "b": []},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Mixed: returns []
        assert result[0][0] == []


class TestConcatExplicitZeroDim:
    """Test Concat with explicit [0] dimension inputs."""

    def test_concat_explicit_zero_dimension(self):
        """Test Concat when explicit shape is [0]."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [2, 4]},
            explicit_shapes={"a": [0]},  # Explicit [0]
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=["a", "b"], outputs=["output"], axis=1)
        result = _infer_concat_shape(node, ctx)
        # Explicit [0] causes early return: [0]
        assert result[0][0] == [0]

    def test_concat_data_zero_dimension_early_return(self):
        """Test Concat with data shape [0] returns early."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [2, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=["a", "b"], outputs=["output"], axis=0)
        result = _infer_concat_shape(node, ctx)
        # Data shape [0] causes early return: [0]
        assert result[0][0] == [0]


class TestBinaryOpsInvalidScalarCombos:
    """Test binary operations with scalar and list combinations."""

    def test_add_one_scalar_one_list_broadcast(self):
        """Test Add with one scalar and one list explicit."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": 5, "b": [2, 3]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # One is scalar, one is list: broadcasts to list shape [2, 3]
        assert result[0][0] == [3]

    def test_sub_one_list_one_scalar_broadcast(self):
        """Test Sub with one list and one scalar explicit."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": [2, 3], "b": 5},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sub", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # One is list, one is scalar: broadcasts
        assert result[0][0] == [3]


class TestBinaryOpsEmptyListExplicit:
    """Test binary operations with empty list explicit shapes."""

    def test_add_empty_list_and_scalar(self):
        """Test Add with empty list and scalar explicit."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": [], "b": 5},  # a is empty list
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Empty list and scalar: uses data shape
        assert result[0][0] == [3]

    def test_mul_both_empty_explicit_shapes(self):
        """Test Mul with both having empty list explicit shapes."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": [], "b": []},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Both empty lists: both inputs are list types, broadcasts
        assert result[0][0] == [3]
