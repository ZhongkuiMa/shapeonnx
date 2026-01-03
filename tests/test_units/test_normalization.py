"""Unit tests for normalization operation shape inference."""

import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_batch_norm_shape


class TestBatchNormalizationOperation:
    """Test BatchNormalization operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "expected"),
        [
            pytest.param([1, 3, 224, 224], [1, 3, 224, 224], id="batchnorm_2d_preserves_shape"),
            pytest.param([2, 64, 28, 28], [2, 64, 28, 28], id="batchnorm_batch_2"),
            pytest.param(
                [1, 128, 14, 14, 14], [1, 128, 14, 14, 14], id="batchnorm_3d_preserves_shape"
            ),
            pytest.param([4, 256, 7, 7], [4, 256, 7, 7], id="batchnorm_larger_batch"),
        ],
    )
    def test_batchnorm_preserves_shape(self, input_shape, expected):
        """Test BatchNormalization preserves input shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
        )
        result = _infer_batch_norm_shape(node, ctx)
        assert result[0][0] == expected

    def test_batchnorm_with_zero_dimension(self):
        """Test BatchNormalization with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
        )
        result = _infer_batch_norm_shape(node, ctx)
        assert result[0][0] == [0]

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param([1, 16, 32, 32], id="batchnorm_various_channel_counts_1"),
            pytest.param([1, 32, 16, 16], id="batchnorm_various_channel_counts_2"),
            pytest.param([1, 64, 8, 8], id="batchnorm_various_channel_counts_3"),
        ],
    )
    def test_batchnorm_various_shapes(self, input_shape):
        """Test BatchNormalization with various input shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
        )
        result = _infer_batch_norm_shape(node, ctx)
        assert result[0][0] == input_shape

    def test_batchnorm_missing_input_error(self):
        """Test BatchNormalization raises error when input is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
        )
        with pytest.raises(RuntimeError, match="Cannot get shape of"):
            _infer_batch_norm_shape(node, ctx)
