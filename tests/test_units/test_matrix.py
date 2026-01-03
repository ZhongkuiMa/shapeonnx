"""Unit tests for matrix operation shape inference."""

import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_gather_shape,
    _infer_gemm_shape,
    _infer_matmul_shape,
)


class TestMatMulOperation:
    """Test MatMul operation shape inference."""

    @pytest.mark.parametrize(
        ("shape1", "shape2", "expected"),
        [
            pytest.param([3, 4], [4, 5], [3, 5], id="matmul_2d_basic"),
            pytest.param([2, 3, 4], [4, 5], [2, 3, 5], id="matmul_batched_left"),
            pytest.param([2, 3, 4, 5], [2, 3, 5, 6], [2, 3, 4, 6], id="matmul_4d_batched"),
            pytest.param([5], [5, 3], [3], id="matmul_1d_left"),
        ],
    )
    def test_matmul_different_shapes(self, shape1, shape2, expected):
        """Test MatMul with different input shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        result = _infer_matmul_shape(node, ctx)
        assert result[0][0] == expected

    def test_matmul_with_zero_dimension(self):
        """Test MatMul with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [5, 3]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        result = _infer_matmul_shape(node, ctx)
        assert result[0][0] == [0]


class TestGemmOperation:
    """Test Gemm operation shape inference."""

    @pytest.mark.parametrize(
        ("shape_a", "shape_b", "trans_a", "trans_b", "expected"),
        [
            pytest.param([3, 4], [4, 5], 0, 0, [3, 5], id="gemm_no_transpose"),
            pytest.param([4, 3], [4, 5], 1, 0, [3, 5], id="gemm_transpose_a"),
            pytest.param([3, 4], [5, 4], 0, 1, [3, 5], id="gemm_transpose_b"),
            pytest.param([4, 3], [5, 4], 1, 1, [3, 5], id="gemm_transpose_both"),
            pytest.param([2, 3, 4], [4, 5], 0, 0, [2, 3, 5], id="gemm_batched_a"),
        ],
    )
    def test_gemm_different_transposes(self, shape_a, shape_b, trans_a, trans_b, expected):
        """Test Gemm with different transpose settings."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": shape_a, "b": shape_b, "c": [1]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["output"],
            transA=trans_a,
            transB=trans_b,
        )
        result = _infer_gemm_shape(node, ctx)
        assert result[0][0] == expected

    def test_gemm_with_zero_dimension(self):
        """Test Gemm with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [5, 3], "c": [1]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["output"],
            transA=0,
            transB=0,
        )
        result = _infer_gemm_shape(node, ctx)
        assert result[0][0] == [0]

    def test_gemm_scalar_input_error(self):
        """Test Gemm raises error for scalar input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": 5, "b": [4, 5], "c": [1]},  # Scalar a
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["output"],
            transA=0,
            transB=0,
        )
        with pytest.raises(RuntimeError, match="Cannot perform Gemm with shapes"):
            _infer_gemm_shape(node, ctx)

    def test_gemm_1d_shape(self):
        """Test Gemm with 1D input shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3], "b": [3, 5], "c": [1]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["output"],
            transA=0,
            transB=0,
        )
        result = _infer_gemm_shape(node, ctx)
        assert result[0][0] == [5]


class TestMatMulErrors:
    """Test MatMul error handling."""

    def test_matmul_scalar_input_error(self):
        """Test MatMul raises assertion for scalar input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": 5, "b": [4, 5]},  # Scalar a
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        with pytest.raises(AssertionError):
            _infer_matmul_shape(node, ctx)

    def test_matmul_missing_input_error(self):
        """Test MatMul raises error when input is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3, 4]},  # Missing b
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_matmul_shape(node, ctx)


class TestGatherVariants:
    """Test Gather operation variants."""

    def test_gather_2d_input_axis_1(self):
        """Test Gather on 2D input with axis=1."""
        import numpy as np
        import onnx

        indices_array = np.array([0, 2], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [4, 5]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=1
        )
        result = _infer_gather_shape(node, ctx)
        assert result[0][0] == [4, 2]

    def test_gather_3d_indices(self):
        """Test Gather with 3D indices shape."""
        import numpy as np
        import onnx

        indices_array = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 0]]], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [5, 10]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=1
        )
        result = _infer_gather_shape(node, ctx)
        # Gather uses len(indices) which is 2 (first dimension of 3D array)
        assert result[0][0] == [5, 2]

    def test_gather_missing_input_error(self):
        """Test Gather raises error when input is missing."""
        import numpy as np
        import onnx

        indices_array = np.array([0, 1], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=0
        )
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_gather_shape(node, ctx)
