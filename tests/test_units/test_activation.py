"""Unit tests for activation functions shape inference."""

import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_nochange_op_shape


class TestActivationFunctions:
    """Test activation functions (no-change operations)."""

    @pytest.mark.parametrize(
        ("op_type", "shape"),
        [
            pytest.param("Relu", [], id="relu_scalar"),
            pytest.param("Relu", [3, 4], id="relu_2d"),
            pytest.param("Relu", [1, 3, 224, 224], id="relu_4d"),
            pytest.param("LeakyRelu", [3, 4], id="leaky_relu_2d"),
            pytest.param("Sigmoid", [3, 4], id="sigmoid_2d"),
            pytest.param("Tanh", [3, 4], id="tanh_2d"),
            pytest.param("Cos", [3, 4], id="cos_2d"),
            pytest.param("Sin", [3, 4], id="sin_2d"),
            pytest.param("Sign", [3, 4], id="sign_2d"),
            pytest.param("Clip", [3, 4], id="clip_2d"),
        ],
    )
    def test_activation_shape_preserved(self, op_type, shape):
        """Test that activation functions preserve input shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(op_type, inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        assert result[0][0] == shape


class TestActivationEdgeCases:
    """Test activation functions with edge cases."""

    @pytest.mark.parametrize(
        ("op_type", "shape"),
        [
            pytest.param("Relu", [0], id="relu_zero_dimension"),
            pytest.param("Sigmoid", [1, 0], id="sigmoid_zero_in_2d"),
            pytest.param("Tanh", [1024, 768], id="tanh_large_shape"),
            pytest.param("Cos", [1], id="cos_single_element"),
        ],
    )
    def test_activation_edge_cases(self, op_type, shape):
        """Test activation functions with edge cases."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(op_type, inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        assert result[0][0] == shape


class TestActivationDifferentRanks:
    """Test activation functions with different tensor ranks."""

    @pytest.mark.parametrize(
        ("op_type", "rank"),
        [
            pytest.param("Relu", 1, id="relu_1d"),
            pytest.param("Relu", 2, id="relu_2d"),
            pytest.param("Relu", 3, id="relu_3d"),
            pytest.param("Relu", 4, id="relu_4d"),
            pytest.param("Relu", 5, id="relu_5d"),
            pytest.param("Sigmoid", 2, id="sigmoid_2d"),
            pytest.param("Sigmoid", 4, id="sigmoid_4d"),
        ],
    )
    def test_activation_different_ranks(self, op_type, rank):
        """Test activation functions preserve shapes of different ranks."""
        import onnx

        # Create shape with rank=rank (e.g., [2,3,4] for rank 3)
        shape = [2] * rank
        ctx = ShapeInferenceContext(
            data_shapes={"input": shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(op_type, inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        assert result[0][0] == shape

    def test_relu_with_explicit_shape(self):
        """Test Relu with explicit shape input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [2, 3, 4]},  # Explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        # Explicit shape preserved
        assert result[0][1] == [2, 3, 4]

    def test_activation_scalar_input(self):
        """Test activation function with scalar input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Sigmoid", inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        # Scalar preserved
        assert result[0][0] == 5

    def test_activation_zero_dimension(self):
        """Test activation function with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Tanh", inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        # Zero dimension preserved
        assert result[0][0] == [0]
