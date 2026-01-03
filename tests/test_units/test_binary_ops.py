"""Unit tests for binary operations shape inference."""

import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape


class TestAddOperation:
    """Test Add operation shape inference with comprehensive test cases."""

    @pytest.mark.parametrize(
        ("shape1", "shape2", "expected"),
        [
            pytest.param([], [], [], id="scalar_plus_scalar"),
            pytest.param([1], [1], [1], id="1d_same_shape"),
            pytest.param([3, 4], [3, 4], [3, 4], id="2d_same_shape"),
            pytest.param([1, 3, 4], [1, 3, 4], [1, 3, 4], id="3d_same_shape"),
            pytest.param([1, 4], [3, 4], [3, 4], id="2d_broadcast_first_dim"),
            pytest.param([3, 1], [3, 4], [3, 4], id="2d_broadcast_second_dim"),
            pytest.param([1, 1], [3, 4], [3, 4], id="2d_broadcast_both_dims"),
            pytest.param([1, 3, 1, 4], [2, 1, 5, 4], [2, 3, 5, 4], id="4d_complex_broadcast"),
            pytest.param([], [3, 4], [3, 4], id="scalar_broadcast_with_vector"),
        ],
    )
    def test_add_shape_broadcast(self, shape1, shape2, expected):
        """Test Add operation with broadcasting rules."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected

    @pytest.mark.parametrize(
        ("shape1", "shape2"),
        [
            pytest.param([2, 3], [4, 5], id="incompatible_2d"),
            pytest.param([2, 3, 4], [2, 3, 5], id="incompatible_rightmost_dim"),
        ],
    )
    def test_add_incompatible_shapes(self, shape1, shape2):
        """Test Add operation with incompatible shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _infer_binary_op_shape(node, ctx)


class TestSubOperation:
    """Test Sub (subtract) operation shape inference."""

    @pytest.mark.parametrize(
        ("shape1", "shape2", "expected"),
        [
            pytest.param([3, 4], [3, 4], [3, 4], id="same_shape"),
            pytest.param([1, 4], [3, 4], [3, 4], id="broadcast_first_dim"),
            pytest.param([3, 1], [3, 4], [3, 4], id="broadcast_second_dim"),
        ],
    )
    def test_sub_shape(self, shape1, shape2, expected):
        """Test Sub operation shape inference."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sub", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected


class TestMulOperation:
    """Test Mul (multiply) operation shape inference."""

    @pytest.mark.parametrize(
        ("shape1", "shape2", "expected"),
        [
            pytest.param([], [], [], id="scalar_times_scalar"),
            pytest.param([3, 4], [3, 4], [3, 4], id="2d_same_shape"),
            pytest.param([1, 4], [3, 4], [3, 4], id="2d_broadcast"),
        ],
    )
    def test_mul_shape(self, shape1, shape2, expected):
        """Test Mul operation shape inference."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected


class TestDivOperation:
    """Test Div (divide) operation shape inference."""

    @pytest.mark.parametrize(
        ("shape1", "shape2", "expected"),
        [
            pytest.param([3, 4], [3, 4], [3, 4], id="same_shape"),
            pytest.param([1, 4], [3, 4], [3, 4], id="broadcast_first_dim"),
        ],
    )
    def test_div_shape(self, shape1, shape2, expected):
        """Test Div operation shape inference."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Div", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected


class TestBinaryOperationsEdgeCases:
    """Test edge cases for binary operations."""

    @pytest.mark.parametrize(
        ("operation", "shape1", "shape2", "expected"),
        [
            pytest.param("Add", [0], [1], [0], id="zero_dimension_add"),
            pytest.param("Sub", [1, 0], [1, 0], [1, 0], id="zero_dimension_sub"),
            pytest.param("Mul", [1], [100, 1], [100, 1], id="large_broadcast_mul"),
            pytest.param("Div", [1024, 768], [1024, 768], [1024, 768], id="large_dims_div"),
        ],
    )
    def test_binary_op_edge_cases(self, operation, shape1, shape2, expected):
        """Test binary operations with edge cases."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(operation, inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected


class TestComparisonOperators:
    """Test comparison operators (Equal, Greater, Less, etc.)."""

    @pytest.mark.parametrize(
        ("operation", "shape1", "shape2", "expected"),
        [
            pytest.param("Equal", [3, 4], [3, 4], [3, 4], id="equal_same_shape"),
            pytest.param("Equal", [1, 4], [3, 4], [3, 4], id="equal_broadcast"),
            pytest.param("Greater", [2, 3, 4], [2, 3, 4], [2, 3, 4], id="greater_3d"),
            pytest.param("Greater", [1, 3], [2, 3], [2, 3], id="greater_broadcast"),
            pytest.param("Less", [4, 5], [4, 5], [4, 5], id="less_same_shape"),
            pytest.param("Less", [1, 5], [4, 5], [4, 5], id="less_broadcast"),
            pytest.param("GreaterOrEqual", [3], [3], [3], id="gte_1d"),
            pytest.param("LessOrEqual", [2, 3], [2, 3], [2, 3], id="lte_2d"),
        ],
    )
    def test_comparison_operators(self, operation, shape1, shape2, expected):
        """Test comparison operators shape inference."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(operation, inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == expected

    def test_equal_with_scalars(self):
        """Test Equal operation with scalar inputs."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Equal", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == []

    def test_greater_with_broadcast(self):
        """Test Greater operation with complex broadcasting."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [1, 3, 1, 4], "b": [2, 1, 5, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Greater", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        assert result[0][0] == [2, 3, 5, 4]

    def test_less_with_scalars_explicit(self):
        """Test Less with scalar explicit shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": []},
            explicit_shapes={"a": 5, "b": 3},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Less", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Less returns broadcasted shape
        assert result[0][0] == []

    def test_greaterorequal_1d_explicit(self):
        """Test GreaterOrEqual with 1D explicit shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": [2, 4, 1], "b": [2, 4, 1]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("GreaterOrEqual", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Returns broadcasted shape
        assert result[0][0] == [3]

    def test_equal_explicit_mixed_types(self):
        """Test Equal where one explicit is None."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3]},
            explicit_shapes={"a": [2]},  # a has explicit, b doesn't
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Equal", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # When val2 is None, returns [0]
        assert result[0][0] == [3]


class TestBinaryOpExplicitShapes:
    """Test binary operations with explicit shape handling."""

    def test_add_explicit_shapes_not_used(self):
        """Test that explicit shapes are not used in Add (data shapes only)."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3, 4], "b": [3, 4]},
            explicit_shapes={},  # Empty
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)

        # Result should have no explicit shape
        assert result[0][1] is None
        assert result[0][0] == [3, 4]

    def test_mul_with_mixed_explicit_shapes(self):
        """Test Mul when one input has explicit shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [2, 3]},
            explicit_shapes={"a": [2, 3]},  # a has explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)

        # Output shape should come from explicit_shape of first input
        assert result[0][0] == [2, 3]

    def test_sub_explicit_shape_output(self):
        """Test Sub with both inputs having explicit shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3, 4], "b": [3, 4]},
            explicit_shapes={"a": [3, 4], "b": [3, 4]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sub", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)

        # Result should have explicit shape output
        assert result[0][0] == [3, 4]

    def test_unsupported_binary_op_error(self):
        """Test unsupported binary operation raises error."""
        from shapeonnx.infer_shape import _compute_binary_op_value

        # Test unsupported operator in _compute_binary_op_value
        with pytest.raises(RuntimeError, match="Cannot calculate"):
            _compute_binary_op_value("UnsupportedOp", 5, 3)
