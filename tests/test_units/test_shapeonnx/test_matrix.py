"""Unit tests for matrix operation shape inference."""

__docformat__ = "restructuredtext"

import numpy as np
import onnx
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
            pytest.param([3, 4], [4], [3], id="matmul_1d_right"),
            pytest.param([4], [4], [], id="matmul_dot_product"),
            pytest.param([3, 4], [2, 4, 5], [2, 3, 5], id="matmul_batched_right"),
            pytest.param([1, 3, 4], [2, 4, 5], [2, 3, 5], id="matmul_broadcast_batch"),
            pytest.param(
                [2, 1, 3, 4],
                [1, 5, 4, 6],
                [2, 5, 3, 6],
                id="matmul_multiaxis_batch_broadcast",
            ),
        ],
    )
    def test_matmul_different_shapes(self, shape1, shape2, expected):
        """Test MatMul with different input shapes."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": shape1, "b": shape2},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        result = _infer_matmul_shape(node, ctx)
        assert len(result) >= 1
        assert result[0][0] == expected

    def test_matmul_with_zero_dimension(self):
        """Test MatMul with zero dimension."""
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
        ],
    )
    def test_gemm_different_transposes(self, shape_a, shape_b, trans_a, trans_b, expected):
        """Test Gemm with different transpose settings."""
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
        assert len(result) >= 1
        assert result[0][0] == expected

    def test_gemm_with_zero_dimension(self):
        """Test Gemm with zero dimension."""
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
        with pytest.raises(RuntimeError, match="requires rank-2"):
            _infer_gemm_shape(node, ctx)

    @pytest.mark.parametrize(
        ("shape_a", "shape_b"),
        [
            pytest.param([3], [3, 5], id="rank1_a"),
            pytest.param([2, 3, 4], [4, 5], id="rank3_a"),
            pytest.param([3, 4], [2, 4, 5], id="rank3_b"),
        ],
    )
    def test_gemm_rejects_non_matrix_inputs(self, shape_a, shape_b):
        """Gemm is matrix-only; batched/rank-1 products belong to MatMul."""
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
            transA=0,
            transB=0,
        )
        with pytest.raises(RuntimeError, match="requires rank-2"):
            _infer_gemm_shape(node, ctx)

    def test_gemm_rejects_mismatched_inner_dimensions(self):
        """The post-transpose contraction dimensions must agree."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [4, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Gemm", inputs=["a", "b"], outputs=["output"])

        with pytest.raises(RuntimeError, match="inner dimensions"):
            _infer_gemm_shape(node, ctx)

    @pytest.mark.parametrize(
        "bias_shape",
        [[], [1], [5], [1, 1], [1, 5], [2, 5]],
        ids=["scalar", "singleton", "vector", "matrix_scalar", "row", "full"],
    )
    def test_gemm_accepts_unidirectionally_broadcastable_bias(self, bias_shape):
        """C may broadcast only into the computed (M,N) output shape."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [3, 5], "c": bias_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["output"])

        assert _infer_gemm_shape(node, ctx) == [([2, 5], None)]

    @pytest.mark.parametrize("bias_shape", [[2], [3, 5], [1, 2, 5]])
    def test_gemm_rejects_non_broadcastable_bias(self, bias_shape):
        """C dimensions must be one or equal to their target dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [3, 5], "c": bias_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Gemm", inputs=["a", "b", "c"], outputs=["output"])

        with pytest.raises(RuntimeError, match="bias must broadcast"):
            _infer_gemm_shape(node, ctx)


class TestMatMulErrors:
    """Test MatMul error handling."""

    def test_matmul_scalar_input_error(self):
        """MatMul rejects scalar inputs."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": 5, "b": [4, 5]},  # Scalar a
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])
        with pytest.raises(ValueError, match="tensor inputs"):
            _infer_matmul_shape(node, ctx)

    @pytest.mark.parametrize(
        ("shape_a", "shape_b", "match"),
        [
            pytest.param([2, 3], [4, 5], "contraction dimensions", id="inner_mismatch"),
            pytest.param(
                [2, 3, 4],
                [5, 4, 6],
                "batch dimensions",
                id="batch_broadcast_mismatch",
            ),
            pytest.param([], [1], "rank >= 1", id="scalar_shape"),
        ],
    )
    def test_matmul_rejects_invalid_geometry(self, shape_a, shape_b, match):
        ctx = ShapeInferenceContext(
            data_shapes={"a": shape_a, "b": shape_b},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"])

        with pytest.raises(ValueError, match=match):
            _infer_matmul_shape(node, ctx)

    def test_matmul_missing_input_error(self):
        """Test MatMul raises error when input is missing."""
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
