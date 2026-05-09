"""Unit tests for Conv and ConvTranspose operation shape inference.

This module provides comprehensive test coverage for Conv and ConvTranspose
operators used in neural network models.

Test organization:
- TestConvBasic: Basic Conv operations with different configurations
- TestConvWithPadding: Conv with various padding modes
- TestConvWithStrides: Conv with different stride configurations
- TestConvTransposeBasic: ConvTranspose 2D operations
- TestConvTransposeWithOutputPadding: ConvTranspose with output_padding
- TestConvErrors: Error handling
"""

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_convtranspose_shape,
    _infer_pool_shape,
)


def _make_weight_tensor(shape: tuple[int, ...], name: str = "weight") -> onnx.TensorProto:
    """Create a test weight tensor for Conv-like operators."""
    rng = np.random.default_rng()
    array = rng.standard_normal(shape).astype(np.float32)
    return onnx.numpy_helper.from_array(array, name=name)


def _extract_tensor_shape(tensor: onnx.TensorProto) -> list[int]:
    """Extract shape from ONNX TensorProto."""
    return list(tensor.dims)


class TestConvBasic:
    """Test basic Conv operation with various configurations."""

    @pytest.mark.parametrize(
        ("weight_shape", "input_shape", "kernel_shape", "pads", "strides", "dilations", "expected"),
        [
            pytest.param(
                (16, 3, 3, 3),
                [1, 3, 28, 28],
                [3, 3],
                [0, 0, 0, 0],
                [1, 1],
                [1, 1],
                [1, 16, 26, 26],
                id="basic",
            ),
            pytest.param(
                (32, 16, 3, 3),
                [1, 16, 28, 28],
                [3, 3],
                [1, 1, 1, 1],
                [1, 1],
                [1, 1],
                [1, 32, 28, 28],
                id="with_padding",
            ),
            pytest.param(
                (64, 32, 3, 3),
                [1, 32, 56, 56],
                [3, 3],
                [1, 1, 1, 1],
                [2, 2],
                [1, 1],
                [1, 64, 28, 28],
                id="with_strides",
            ),
            pytest.param(
                (16, 3, 3, 3),
                [1, 3, 28, 28],
                [3, 3],
                [2, 2, 2, 2],
                [1, 1],
                [2, 2],
                [1, 16, 28, 28],
                id="with_dilation",
            ),
            pytest.param(
                (128, 64, 1, 1),
                [4, 64, 28, 28],
                [1, 1],
                [0, 0, 0, 0],
                [1, 1],
                [1, 1],
                [4, 128, 28, 28],
                id="batch_size_preserved",
            ),
        ],
    )
    def test_conv_2d_configurations(
        self, weight_shape, input_shape, kernel_shape, pads, strides, dilations, expected
    ):
        """Test 2D Conv with various configurations (padding, stride, dilation, batch)."""
        weight_tensor = _make_weight_tensor(weight_shape)
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape, "weight": _extract_tensor_shape(weight_tensor)},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dilations=dilations,
        )
        results = _infer_pool_shape(node, ctx)
        assert len(results) >= 1
        assert results[0][0] == expected

    @pytest.mark.parametrize(
        ("weight_shape", "input_shape", "kernel_shape", "pads", "strides", "dilations", "expected"),
        [
            pytest.param(
                (32, 16, 3), [1, 16, 100], [3], [0, 0], [1], [1], [1, 32, 98], id="conv_1d"
            ),
            pytest.param(
                (32, 3, 3, 3, 3),
                [1, 3, 16, 16, 16],
                [3, 3, 3],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 32, 14, 14, 14],
                id="conv_3d",
            ),
        ],
    )
    def test_conv_nd_configurations(
        self, weight_shape, input_shape, kernel_shape, pads, strides, dilations, expected
    ):
        """Test 1D and 3D Conv operations."""
        weight_tensor = _make_weight_tensor(weight_shape)
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape, "weight": _extract_tensor_shape(weight_tensor)},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dilations=dilations,
        )
        results = _infer_pool_shape(node, ctx)
        assert len(results) >= 1
        assert results[0][0] == expected


class TestConvWithAsymmetricPadding:
    """Test Conv with asymmetric padding."""

    @pytest.mark.parametrize(
        ("weight_shape", "input_shape", "kernel_shape", "pads", "expected"),
        [
            pytest.param(
                (16, 3, 3, 3),
                [1, 3, 28, 28],
                [3, 3],
                [0, 2, 0, 2],
                [1, 16, 26, 30],
                id="asymmetric",
            ),
            pytest.param(
                (16, 3, 1, 1),
                [1, 3, 10, 10],
                [1, 1],
                [2, 2, 2, 2],
                [1, 16, 14, 14],
                id="large_padding",
            ),
        ],
    )
    def test_conv_padding_configurations(
        self, weight_shape, input_shape, kernel_shape, pads, expected
    ):
        """Test Conv with asymmetric and large padding."""
        weight_tensor = _make_weight_tensor(weight_shape)
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape, "weight": _extract_tensor_shape(weight_tensor)},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=kernel_shape,
            pads=pads,
            strides=[1, 1],
            dilations=[1, 1],
        )
        results = _infer_pool_shape(node, ctx)
        assert results[0][0] == expected


class TestConvTransposeBasic:
    """Test basic ConvTranspose 2D operation."""

    @pytest.mark.parametrize(
        ("input_shape", "weight_shape", "kernel_shape", "pads", "strides", "expected"),
        [
            pytest.param(
                [1, 16, 28, 28],
                (16, 3, 3, 3),
                [3, 3],
                [1, 1, 1, 1],
                [1, 1],
                [1, 3, 28, 28],
                id="basic",
            ),
            pytest.param(
                [1, 16, 14, 14],
                (16, 3, 3, 3),
                [3, 3],
                [1, 1, 1, 1],
                [2, 2],
                [1, 3, 27, 27],
                id="with_strides",
            ),
            pytest.param(
                [1, 16, 28, 28],
                (16, 3, 3, 3),
                [3, 3],
                [0, 0, 0, 0],
                [1, 1],
                [1, 3, 30, 30],
                id="no_padding",
            ),
            pytest.param(
                [4, 128, 14, 14],
                (128, 64, 3, 3),
                [3, 3],
                [1, 1, 1, 1],
                [2, 2],
                [4, 64, 27, 27],
                id="batch_size_preserved",
            ),
        ],
    )
    def test_convtranspose_2d_configurations(
        self, input_shape, weight_shape, kernel_shape, pads, strides, expected
    ):
        """Test 2D ConvTranspose with various configurations."""
        weight_tensor = _make_weight_tensor(weight_shape)
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dilations=[1, 1],
            output_padding=[0, 0],
        )
        results = _infer_convtranspose_shape(node, ctx)
        assert len(results) >= 1
        assert results[0][0] == expected


class TestConvTransposeWithOutputPadding:
    """Test ConvTranspose with output_padding attribute."""

    @pytest.mark.parametrize(
        ("output_padding", "expected"),
        [
            pytest.param([1, 0], [1, 3, 28, 27], id="horizontal"),
            pytest.param([1, 1], [1, 3, 28, 28], id="both_axes"),
        ],
    )
    def test_convtranspose_output_padding(self, output_padding, expected):
        """Test ConvTranspose with output_padding on various axes."""
        weight_tensor = _make_weight_tensor((16, 3, 3, 3))
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 16, 14, 14]},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
            dilations=[1, 1],
            output_padding=output_padding,
        )
        results = _infer_convtranspose_shape(node, ctx)
        assert len(results) >= 1
        assert results[0][0] == expected


class TestConvErrors:
    """Test error handling for Conv operations."""

    def test_conv_missing_input_shape(self):
        """Test error when input shape is missing."""
        weight_tensor = _make_weight_tensor((16, 3, 3, 3))
        ctx = ShapeInferenceContext(
            data_shapes={},  # No input shape
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Conv",
            inputs=["missing_input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
        )

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_pool_shape(node, ctx)

    def test_conv_inconsistent_dimensions(self):
        """Test error with inconsistent kernel/pad/stride dimensions."""
        weight_tensor = _make_weight_tensor((16, 3, 3, 3))
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 28, 28]},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1],  # Wrong dimension (should be 4 for 2D)
            strides=[1, 1],
            dilations=[1, 1],
        )

        with pytest.raises(ValueError, match=r"Inconsistent dimensions"):
            _infer_pool_shape(node, ctx)

    def test_convtranspose_missing_weight(self):
        """Test ConvTranspose error when weight initializer missing."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 16, 28, 28]},
            explicit_shapes={},
            initializers={},  # No weight initializer
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            output_padding=[0, 0],
        )

        with pytest.raises(KeyError, match="weight"):
            _infer_convtranspose_shape(node, ctx)

    def test_convtranspose_unsupported_3d(self):
        """Test ConvTranspose error with unsupported 3D (only 2D supported)."""
        weight_tensor = _make_weight_tensor((3, 16, 3, 3, 3))
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 16, 14, 14, 14]},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3, 3],
            pads=[1, 1, 1, 1, 1, 1],
            strides=[1, 1, 1],
            dilations=[1, 1, 1],
            output_padding=[0, 0, 0],
        )

        with pytest.raises(NotImplementedError, match="ConvTranspose"):
            _infer_convtranspose_shape(node, ctx)


class TestConvExplicitShapes:
    """Test Conv with explicit shape handling."""

    def test_conv_explicit_shapes_not_used(self):
        """Test that explicit shapes are not used in Conv (data shapes only)."""
        weight_tensor = _make_weight_tensor((16, 3, 3, 3))
        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 28, 28], "weight": [16, 3, 3, 3]},
            explicit_shapes={},  # Empty
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
        )

        results = _infer_pool_shape(node, ctx)

        # Result should have no explicit shape
        assert results[0][1] is None
        assert results[0][0] == [1, 16, 28, 28]


class TestConvIntegration:
    """Integration tests for Conv operations."""

    def test_conv_typical_resnet_block(self):
        """Test Conv configuration typical in ResNet blocks."""
        weight1 = _make_weight_tensor((64, 64, 3, 3))
        weight2 = _make_weight_tensor((64, 64, 3, 3))

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 64, 56, 56], "w1": [64, 64, 3, 3], "w2": [64, 64, 3, 3]},
            explicit_shapes={},
            initializers={"w1": weight1, "w2": weight2},
            verbose=False,
        )

        # First conv in block
        node1 = onnx.helper.make_node(
            "Conv",
            inputs=["input", "w1"],
            outputs=["hidden"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
        )

        # Second conv in block
        node2 = onnx.helper.make_node(
            "Conv",
            inputs=["hidden", "w2"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
        )

        results1 = _infer_pool_shape(node1, ctx)
        assert results1[0][0] == [1, 64, 56, 56]

        # Update context for second conv
        ctx.data_shapes["hidden"] = results1[0][0]
        results2 = _infer_pool_shape(node2, ctx)
        assert results2[0][0] == [1, 64, 56, 56]

    def test_convtranspose_decoder_upsampling(self):
        """Test ConvTranspose in typical decoder/upsampling scenario."""
        weight = _make_weight_tensor((256, 128, 3, 3))

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 256, 14, 14], "weight": _extract_tensor_shape(weight)},
            explicit_shapes={},
            initializers={"weight": weight},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
            dilations=[1, 1],
            output_padding=[0, 0],
        )

        results = _infer_convtranspose_shape(node, ctx)
        # output = (14-1)*2 - 2 + 1*2 + 0 + 1 = 26 + 1 = 27
        assert results[0][0] == [1, 128, 27, 27]

    def test_convtranspose_zero_dimension(self):
        """Test ConvTranspose with zero dimension input."""
        rng = np.random.default_rng()
        weight = rng.standard_normal((3, 64, 3, 3), dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weight, "weight")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )

        result = _infer_convtranspose_shape(node, ctx)
        # Zero dimension propagates
        assert result[0][0] == [0]

    def test_convtranspose_missing_input_error(self):
        """Test ConvTranspose raises error when input is missing."""
        rng = np.random.default_rng()
        weight = rng.standard_normal((3, 64, 3, 3), dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weight, "weight")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_convtranspose_shape(node, ctx)

    def test_convtranspose_scalar_input_error(self):
        """Test ConvTranspose raises error for scalar input."""
        rng = np.random.default_rng()
        weight = rng.standard_normal((3, 64, 3, 3), dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weight, "weight")

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar input
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )

        with pytest.raises(RuntimeError, match="ConvTranspose input shape cannot be scalar"):
            _infer_convtranspose_shape(node, ctx)
