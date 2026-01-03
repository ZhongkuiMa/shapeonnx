"""Unit tests for tensor manipulation operation shape inference."""

import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_expand_shape,
    _infer_flatten_shape,
    _infer_reshape_shape,
    _infer_squeeze_shape,
    _infer_transpose_shape,
    _infer_unsqueeze_shape,
)


class TestReshapeOperation:
    """Test Reshape operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "target_shape", "expected"),
        [
            pytest.param([3, 4], [12], [12], id="reshape_2d_to_1d"),
            pytest.param([3, 4], [2, 6], [2, 6], id="reshape_2d_to_2d"),
            pytest.param([2, 3, 4], [6, 4], [6, 4], id="reshape_3d_to_2d"),
            pytest.param([12], [3, 4], [3, 4], id="reshape_1d_to_2d"),
            pytest.param([12], [2, 3, 2], [2, 3, 2], id="reshape_1d_to_3d"),
            pytest.param([3, 4], [3, -1], [3, 4], id="reshape_with_minus_one"),
            pytest.param([24], [-1, 4], [6, 4], id="reshape_infer_first_dim"),
            pytest.param([2, 3, 4], [-1], [24], id="reshape_flatten_with_minus_one"),
        ],
    )
    def test_reshape_different_shapes(self, input_shape, target_shape, expected):
        """Test Reshape with different target shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={"target": target_shape},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Reshape", inputs=["input", "target"], outputs=["output"])
        result = _infer_reshape_shape(node, ctx)
        assert result[0][0] == expected

    def test_reshape_with_zero_dimension(self):
        """Test Reshape with zero dimension and -1 target."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={"target": [-1, 4]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Reshape", inputs=["input", "target"], outputs=["output"])
        result = _infer_reshape_shape(node, ctx)
        assert result[0][0] == [0]

    def test_reshape_scalar_input(self):
        """Test Reshape with scalar input returns [0]."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar
            explicit_shapes={"target": [2, 3]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Reshape", inputs=["input", "target"], outputs=["output"])
        result = _infer_reshape_shape(node, ctx)
        # Scalar input returns [0]
        assert result[0][0] == [0]

    def test_reshape_missing_input_error(self):
        """Test Reshape raises error when input is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={"target": [2, 6]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Reshape", inputs=["input", "target"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_reshape_shape(node, ctx)


class TestFlattenOperation:
    """Test Flatten operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "axis", "expected"),
        [
            pytest.param([2, 3, 4], 0, [24], id="flatten_axis_0"),
            pytest.param([2, 3, 4], 1, [2, 12], id="flatten_axis_1"),
            pytest.param([2, 3, 4], 2, [2, 3, 4], id="flatten_axis_2"),
            pytest.param([2, 3, 4], 3, [2, 3, 4, 1], id="flatten_axis_3"),
            pytest.param([3, 4], 1, [3, 4], id="flatten_2d_axis_1"),
            pytest.param([12], 0, [12], id="flatten_1d_axis_0"),
        ],
    )
    def test_flatten_different_axes(self, input_shape, axis, expected):
        """Test Flatten with different axis values."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=axis)
        result = _infer_flatten_shape(node, ctx)
        assert result[0][0] == expected

    def test_flatten_with_zero_dimension(self):
        """Test Flatten with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=0)
        result = _infer_flatten_shape(node, ctx)
        assert result[0][0] == [0]


class TestTransposeOperation:
    """Test Transpose operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "perm", "expected"),
        [
            pytest.param([2, 3], [1, 0], [3, 2], id="transpose_2d_swap"),
            pytest.param([2, 3, 4], [2, 0, 1], [4, 2, 3], id="transpose_3d_rotate"),
            pytest.param([2, 3, 4], [0, 2, 1], [2, 4, 3], id="transpose_3d_swap_last"),
            pytest.param([1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 4], id="transpose_4d_identity"),
            pytest.param([1, 2, 3, 4], [3, 2, 1, 0], [4, 3, 2, 1], id="transpose_4d_reverse"),
        ],
    )
    def test_transpose_different_permutations(self, input_shape, perm, expected):
        """Test Transpose with different permutations."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=perm)
        result = _infer_transpose_shape(node, ctx)
        assert result[0][0] == expected

    def test_transpose_with_zero_dimension(self):
        """Test Transpose with zero dimension (1d shape becomes 2d)."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=[1, 0])
        result = _infer_transpose_shape(node, ctx)
        # 1D shape becomes [0, 1] (perm is ignored for 1D inputs)
        assert result[0][0] == [0, 1]


class TestSqueezeOperation:
    """Test Squeeze operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "axes", "expected"),
        [
            pytest.param([1, 3, 4], [0], [3, 4], id="squeeze_axis_0"),
            pytest.param([3, 1, 4], [1], [3, 4], id="squeeze_axis_1"),
            pytest.param([1, 1, 4], [0, 1], [4], id="squeeze_multiple_axes"),
            pytest.param([1, 3, 1], [], [1, 3], id="squeeze_remove_trailing_ones"),
            pytest.param([1], [0], [], id="squeeze_scalar_to_empty"),
        ],
    )
    def test_squeeze_different_axes(self, input_shape, axes, expected):
        """Test Squeeze with different axis specifications."""
        import onnx

        if axes:
            # Create axes tensor initializer
            import numpy as np

            axes_array = np.array(axes, dtype=np.int64)
            axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

            ctx = ShapeInferenceContext(
                data_shapes={"input": input_shape},
                explicit_shapes={"axes": axes},
                initializers={"axes": axes_tensor},
                verbose=False,
            )
            node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        else:
            ctx = ShapeInferenceContext(
                data_shapes={"input": input_shape},
                explicit_shapes={},
                initializers={},
                verbose=False,
            )
            node = onnx.helper.make_node("Squeeze", inputs=["input"], outputs=["output"])

        result = _infer_squeeze_shape(node, ctx)
        assert result[0][0] == expected

    def test_squeeze_with_zero_dimension(self):
        """Test Squeeze with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input"], outputs=["output"])
        result = _infer_squeeze_shape(node, ctx)
        assert result[0][0] == [0]

    def test_squeeze_single_element_input(self):
        """Test Squeeze with single element input."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 4]},
            explicit_shapes={"axes": [0]},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_squeeze_shape(node, ctx)
        assert result[0][0] == [3, 4]

    def test_squeeze_missing_input_error(self):
        """Test Squeeze raises error when input is missing."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={"axes": [0]},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_squeeze_shape(node, ctx)

    def test_squeeze_scalar_input_error(self):
        """Test Squeeze raises error when input is scalar."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar input
            explicit_shapes={"axes": [0]},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Input shape must be a list"):
            _infer_squeeze_shape(node, ctx)

    def test_squeeze_no_axes_auto_detect(self):
        """Test Squeeze without axes auto-detects 1 dimensions."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 1, 4, 1]},  # Multiple 1 dimensions
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input"], outputs=["output"])
        result = _infer_squeeze_shape(node, ctx)
        # Removes all 1s then prepends first element
        assert result[0][0] == [1, 3, 4]

    def test_squeeze_axes_is_scalar_int(self):
        """Test Squeeze when axes is a single scalar int."""
        import numpy as np
        import onnx

        axes_array = np.array(1, dtype=np.int64)  # Scalar int, not array
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 1, 4]},
            explicit_shapes={"axes": 1},  # Scalar int instead of list
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_squeeze_shape(node, ctx)
        # Should squeeze axis 1
        assert result[0][0] == [2, 4]

    def test_squeeze_non_unit_axis_error(self):
        """Test Squeeze raises error when trying to squeeze non-unit axis."""
        import numpy as np
        import onnx

        axes_array = np.array([1], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},  # Axis 1 has size 3, not 1
            explicit_shapes={"axes": [1]},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Squeeze", inputs=["input", "axes"], outputs=["output"])
        with pytest.raises(ValueError, match="Cannot squeeze axis"):
            _infer_squeeze_shape(node, ctx)


class TestUnsqueezeOperation:
    """Test Unsqueeze operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "axes", "expected"),
        [
            pytest.param([3, 4], [0], [1, 3, 4], id="unsqueeze_axis_0"),
            pytest.param([3, 4], [1], [3, 1, 4], id="unsqueeze_axis_1"),
            pytest.param([3, 4], [2], [3, 4, 1], id="unsqueeze_axis_2"),
            pytest.param([3, 4], [0, 2], [1, 3, 4, 1], id="unsqueeze_multiple_axes"),
            pytest.param([3], [0], [1, 3], id="unsqueeze_1d_axis_0"),
            pytest.param([3], [1], [3, 1], id="unsqueeze_1d_axis_1"),
        ],
    )
    def test_unsqueeze_different_axes(self, input_shape, axes, expected):
        """Test Unsqueeze with different axis specifications."""
        import numpy as np
        import onnx

        axes_array = np.array(axes, dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_unsqueeze_shape(node, ctx)
        assert result[0][0] == expected

    def test_unsqueeze_with_zero_dimension(self):
        """Test Unsqueeze with zero dimension."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_unsqueeze_shape(node, ctx)
        assert result[0][0] == [0]

    def test_unsqueeze_negative_axis(self):
        """Test Unsqueeze with negative axis."""
        import numpy as np
        import onnx

        axes_array = np.array([-1], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [3, 4]},
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_unsqueeze_shape(node, ctx)
        # Negative axis -1 adds dimension at end
        assert result[0][0] == [3, 4, 1]

    def test_unsqueeze_missing_input_error(self):
        """Test Unsqueeze raises error when input is missing."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_unsqueeze_shape(node, ctx)

    def test_unsqueeze_scalar_input_explicit(self):
        """Test Unsqueeze with scalar input from explicit shape."""
        import numpy as np
        import onnx

        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": 5},  # Scalar explicit shape
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_unsqueeze_shape(node, ctx)
        # Unsqueeze of scalar [5] with axis 0 gives [5]
        assert result[0][1] == [5]

    def test_unsqueeze_scalar_input_invalid_axes_error(self):
        """Test Unsqueeze with scalar input and invalid axes."""
        import numpy as np
        import onnx

        axes_array = np.array([1], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": 5},  # Scalar explicit shape
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        with pytest.raises(ValueError, match="Invalid axes"):
            _infer_unsqueeze_shape(node, ctx)


class TestExpandOperation:
    """Test Expand operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "target_shape", "expected"),
        [
            pytest.param([1, 3, 4], [2, 3, 4], [2, 3, 4], id="expand_first_dim"),
            pytest.param([3, 1, 4], [3, 2, 4], [3, 2, 4], id="expand_middle_dim"),
            pytest.param([1], [3, 4], [3, 4], id="expand_scalar_to_2d"),
            pytest.param([3, 4], [1, 3, 4], [1, 3, 4], id="expand_add_leading_dim"),
            pytest.param([1, 4], [3, 4], [3, 4], id="expand_broadcast_compatible"),
        ],
    )
    def test_expand_different_shapes(self, input_shape, target_shape, expected):
        """Test Expand with different target shapes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={"shape": target_shape},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Expand", inputs=["input", "shape"], outputs=["output"])
        result = _infer_expand_shape(node, ctx)
        assert result[0][0] == expected

    def test_expand_with_zero_dimension(self):
        """Test Expand with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1]},
            explicit_shapes={"shape": [0]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Expand", inputs=["input", "shape"], outputs=["output"])
        result = _infer_expand_shape(node, ctx)
        assert result[0][0] == [0]

    def test_expand_missing_target_shape_error(self):
        """Test Expand raises error when target shape is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3, 4]},
            explicit_shapes={},  # Missing shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Expand", inputs=["input", "shape"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_expand_shape(node, ctx)

    def test_expand_with_explicit_input_shape(self):
        """Test Expand with explicit input shape returns explicit output."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},  # No data shape
            explicit_shapes={"input": [1, 3, 4], "shape": [2, 3, 4]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Expand", inputs=["input", "shape"], outputs=["output"])
        result = _infer_expand_shape(node, ctx)
        # When input is explicit, output should be explicit
        assert result[0][1] == [2, 3, 4]


class TestTransposeEdgeCases:
    """Test Transpose edge cases and error conditions."""

    def test_transpose_with_explicit_shape(self):
        """Test Transpose with explicit shape input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [2, 3]},  # Explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=[1, 0])
        result = _infer_transpose_shape(node, ctx)
        # Explicit shape transposed: [2, 3] with perm [1, 0] = [3, 2]
        assert result[0][1] == [3, 2]


class TestUnsqueezeEdgeCases:
    """Test Unsqueeze edge cases and error conditions."""

    def test_unsqueeze_with_explicit_shape(self):
        """Test Unsqueeze with explicit shape input."""
        import numpy as np
        import onnx

        axes_array = np.array([1], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [3, 4]},  # Explicit shape
            initializers={"axes": axes_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Unsqueeze", inputs=["input", "axes"], outputs=["output"])
        result = _infer_unsqueeze_shape(node, ctx)
        # Unsqueeze on axis 1: [3, 4] -> [3, 1, 4]
        assert result[0][1] == [3, 1, 4]
