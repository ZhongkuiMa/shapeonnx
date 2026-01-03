"""Unit tests for pooling operation shape inference."""

import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_pool_shape


class TestMaxPoolOperation:
    """Test MaxPool operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "kernel", "strides", "pads", "expected"),
        [
            pytest.param(
                [1, 3, 4, 4], [2, 2], [1, 1], [0, 0, 0, 0], [1, 3, 3, 3], id="maxpool_basic_2d"
            ),
            pytest.param(
                [1, 3, 8, 8], [2, 2], [2, 2], [0, 0, 0, 0], [1, 3, 4, 4], id="maxpool_stride_2"
            ),
            pytest.param(
                [1, 3, 5, 5], [3, 3], [1, 1], [1, 1, 1, 1], [1, 3, 5, 5], id="maxpool_with_padding"
            ),
            pytest.param([1, 3, 7], [2], [1], [0, 0], [1, 3, 6], id="maxpool_1d"),
            pytest.param(
                [1, 3, 4, 4, 4],
                [2, 2, 2],
                [1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 3, 3, 3, 3],
                id="maxpool_3d",
            ),
        ],
    )
    def test_maxpool_different_shapes(self, input_shape, kernel, strides, pads, expected):
        """Test MaxPool with different kernel, stride, and padding."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=kernel,
            strides=strides,
            pads=pads,
        )
        result = _infer_pool_shape(node, ctx)
        assert result[0][0] == expected

    def test_maxpool_with_zero_dimension(self):
        """Test MaxPool with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[2],
            strides=[1],
            pads=[0, 0],
        )
        result = _infer_pool_shape(node, ctx)
        assert result[0][0] == [0]


class TestAveragePoolOperation:
    """Test AveragePool operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "kernel", "strides", "pads", "expected"),
        [
            pytest.param(
                [1, 3, 4, 4], [2, 2], [1, 1], [0, 0, 0, 0], [1, 3, 3, 3], id="avgpool_basic_2d"
            ),
            pytest.param(
                [1, 3, 8, 8], [2, 2], [2, 2], [0, 0, 0, 0], [1, 3, 4, 4], id="avgpool_stride_2"
            ),
            pytest.param(
                [1, 3, 5, 5], [3, 3], [1, 1], [1, 1, 1, 1], [1, 3, 5, 5], id="avgpool_with_padding"
            ),
        ],
    )
    def test_avgpool_different_shapes(self, input_shape, kernel, strides, pads, expected):
        """Test AveragePool with different kernel, stride, and padding."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=kernel,
            strides=strides,
            pads=pads,
        )
        result = _infer_pool_shape(node, ctx)
        assert result[0][0] == expected

    def test_maxpool_with_ceil_mode(self):
        """Test MaxPool with ceil_mode=True."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 5, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            ceil_mode=1,
        )
        result = _infer_pool_shape(node, ctx)
        # With ceil_mode: (5 + 0 + 0 - 2) / 2 + 1 = 2.5 -> ceil = 3
        assert result[0][0] == [1, 3, 3, 3]

    def test_avgpool_with_dilations(self):
        """Test AveragePool with dilations."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 7, 7]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[2, 2],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[2, 2],
        )
        result = _infer_pool_shape(node, ctx)
        # With dilation=2: (7 + 0 + 0 - 2*(2-1) - 1) / 1 + 1 = 5
        assert result[0][0] == [1, 3, 5, 5]

    def test_maxpool_ceil_mode(self):
        """Test MaxPool with ceil_mode=1."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 7, 7]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            ceil_mode=1,
        )
        result = _infer_pool_shape(node, ctx)
        # With ceil_mode: ceil((7 + 0 + 0 - 3 - 1) / 2 + 1) = ceil(2.5) = 3
        assert result[0][0] == [1, 3, 3, 3]

    def test_averagepool_with_pads(self):
        """Test AveragePool with padding."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 5, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        )
        result = _infer_pool_shape(node, ctx)
        # (5 + 1 + 1 - 2 - 1) / 2 + 1 = 3
        assert result[0][0] == [1, 3, 3, 3]

    def test_maxpool_ceil_mode_with_dilation(self):
        """Test MaxPool with ceil_mode and dilation."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 10, 10]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[2, 2],
            ceil_mode=1,
        )
        result = _infer_pool_shape(node, ctx)
        # With dilation=2 and ceil_mode: ceil((10 + 0 + 0 - 2*(3-1) - 1) / 1 + 1) = ceil(6) = 6
        assert result[0][0] == [1, 3, 6, 6]
