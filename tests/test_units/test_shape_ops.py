"""Unit tests for shape operation shape inference."""

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_constant_of_shape_shape,
    _infer_pad_shape,
    _infer_range_shape,
    _infer_resize_shape,
    _infer_shape_op_shape,
)


class TestShapeOperation:
    """Test Shape operator."""

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            pytest.param([2, 3, 4], [2, 3, 4], id="shape_3d"),
            pytest.param([1, 224, 224], [1, 224, 224], id="shape_image"),
            pytest.param([10], [10], id="shape_1d"),
            pytest.param([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], id="shape_5d"),
        ],
    )
    def test_shape_operator(self, input_shape, expected_shape):
        """Test Shape operator returns input shape as explicit shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        # Shape operator returns explicit shape
        assert result[0][1] == expected_shape

    def test_shape_with_zero_dimension(self):
        """Test Shape with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        assert result[0][1] == [0]

    def test_shape_explicit_input(self):
        """Test Shape with explicit shape input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [2, 3, 4]},  # Explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        # Shape of explicit shape returns the shape itself
        assert result[0][1] == [2, 3, 4]

    def test_shape_scalar_explicit(self):
        """Test Shape with scalar explicit shape input."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": 5},  # Scalar explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        # Shape of scalar returns empty list
        assert result[0][1] == []

    def test_shape_zero_dimension_explicit(self):
        """Test Shape with zero dimension explicit shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [0]},  # Zero dimension
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        # Shape of [0] returns [0]
        assert result[0][1] == [0]

    def test_shape_data_shape_not_list_error(self):
        """Test Shape raises error when data shape is not a list."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar, not a list
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        with pytest.raises(RuntimeError, match="Expected list shape"):
            _infer_shape_op_shape(node, ctx)


class TestPadOperation:
    """Test Pad operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "pads", "expected"),
        [
            pytest.param([3, 4], [1, 1, 1, 1], [5, 6], id="pad_all_sides"),
            pytest.param([4], [2, 1], [7], id="pad_1d"),
        ],
    )
    def test_pad_different_pads(self, input_shape, pads, expected):
        """Test Pad with different padding values."""
        import onnx

        pads_array = np.array(pads, dtype=np.int64)
        pads_tensor = onnx.numpy_helper.from_array(pads_array, name="pads")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"pads": pads_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        result = _infer_pad_shape(node, ctx)
        assert result[0][0] == expected

    def test_pad_with_zero_dimension(self):
        """Test Pad with zero dimension."""
        import onnx

        pads_array = np.array([0, 0, 0, 0], dtype=np.int64)
        pads_tensor = onnx.numpy_helper.from_array(pads_array, name="pads")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"pads": pads_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        result = _infer_pad_shape(node, ctx)
        assert result[0][0] == [0]

    def test_pad_3d_input(self):
        """Test Pad with 3D input."""
        import onnx

        pads_array = np.array([1, 1, 2, 2, 1, 1], dtype=np.int64)
        pads_tensor = onnx.numpy_helper.from_array(pads_array, name="pads")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"pads": pads_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        result = _infer_pad_shape(node, ctx)
        # Pads are [start_d0, start_d1, start_d2, end_d0, end_d1, end_d2]
        # combined_pads = [1+1, 1+1, 2+2] = [2, 2, 4]? No, formula is pads[i] + pads[i+dim]
        # So [1+2, 1+1, 2+1] = [3, 2, 3]
        # Result: [2+3, 3+2, 4+3] = [5, 5, 7]
        assert result[0][0] == [5, 5, 7]

    def test_pad_with_explicit_shape(self):
        """Test Pad with explicit shape input."""
        import onnx

        pads_array = np.array([1, 1, 1, 1], dtype=np.int64)
        pads_tensor = onnx.numpy_helper.from_array(pads_array, name="pads")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [3, 4]},  # Explicit shape
            initializers={"pads": pads_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        result = _infer_pad_shape(node, ctx)
        # With explicit shape: [3, 4] + pads [1, 1, 1, 1] = [5, 6]
        assert result[0][1] == [5, 6]

    def test_pad_missing_pads_error(self):
        """Test Pad raises error when pads initializer is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={},  # Missing pads
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        with pytest.raises(KeyError):
            _infer_pad_shape(node, ctx)


class TestResizeOperation:
    """Test Resize operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "scales", "expected"),
        [
            pytest.param([1, 3, 4, 4], [1.0, 1.0, 2.0, 2.0], [1, 3, 8, 8], id="resize_scale_2x"),
            pytest.param([2, 3, 8, 8], [1.0, 1.0, 0.5, 0.5], [2, 3, 4, 4], id="resize_scale_0.5x"),
            pytest.param([1, 4, 3], [1.0, 2.0, 3.0], [1, 8, 9], id="resize_1d_asymmetric"),
        ],
    )
    def test_resize_different_scales(self, input_shape, scales, expected):
        """Test Resize with different scale factors."""
        import onnx

        scales_array = np.array(scales, dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        result = _infer_resize_shape(node, ctx)
        assert result[0][0] == expected

    def test_resize_with_zero_dimension(self):
        """Test Resize with zero dimension."""
        import onnx

        scales_array = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        result = _infer_resize_shape(node, ctx)
        assert result[0][0] == [0]

    @pytest.mark.parametrize(
        ("input_shape", "scales", "nearest_mode", "expected"),
        [
            pytest.param(
                [1, 3, 5, 5],
                [1.0, 1.0, 1.5, 1.5],
                "ceil",
                [1, 3, 8, 8],
                id="resize_ceil",
            ),
            pytest.param(
                [1, 3, 5, 5],
                [1.0, 1.0, 1.5, 1.5],
                "round_prefer_floor",
                [1, 3, 7, 7],
                id="resize_round_prefer_floor",
            ),
            pytest.param(
                [1, 3, 5, 5],
                [1.0, 1.0, 1.5, 1.5],
                "round_prefer_ceil",
                [1, 3, 8, 8],
                id="resize_round_prefer_ceil",
            ),
        ],
    )
    def test_resize_different_nearest_modes(self, input_shape, scales, nearest_mode, expected):
        """Test Resize with different nearest_mode values."""
        import onnx

        scales_array = np.array(scales, dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode=nearest_mode,
        )
        result = _infer_resize_shape(node, ctx)
        assert result[0][0] == expected

    def test_resize_missing_input_shape_error(self):
        """Test Resize raises error when input shape is missing."""
        import onnx

        scales_array = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_resize_shape(node, ctx)

    def test_resize_scalar_input_error(self):
        """Test Resize raises error for scalar input."""
        import onnx

        scales_array = np.array([1.0, 1.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar input
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        with pytest.raises(RuntimeError, match="Resize input shape cannot be scalar"):
            _infer_resize_shape(node, ctx)

    def test_resize_empty_scales_error(self):
        """Test Resize raises error for empty scales."""
        import onnx

        scales_array = np.array([], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        with pytest.raises(ValueError, match="Resize with empty scales"):
            _infer_resize_shape(node, ctx)

    def test_resize_unsupported_mode_error(self):
        """Test Resize raises error for unsupported mode."""
        import onnx

        scales_array = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="linear",  # Unsupported mode
            nearest_mode="floor",
        )
        with pytest.raises(NotImplementedError, match="Resize mode=linear"):
            _infer_resize_shape(node, ctx)

    def test_resize_unsupported_align_mode_error(self):
        """Test Resize raises error for unsupported align_mode."""
        import onnx

        scales_array = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="tf_crop_and_resize",  # Unsupported
            mode="nearest",
            nearest_mode="floor",
        )
        with pytest.raises(NotImplementedError, match="Resize align_mode="):
            _infer_resize_shape(node, ctx)

    def test_resize_unsupported_nearest_mode_error(self):
        """Test Resize raises error for unsupported nearest_mode."""
        import onnx

        scales_array = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="invalid_mode",  # Unsupported nearest_mode
        )
        with pytest.raises(NotImplementedError, match="Resize nearest_mode="):
            _infer_resize_shape(node, ctx)


class TestRangeOperation:
    """Test Range operator shape inference."""

    def _make_range_initializer(self, value: int, name: str) -> onnx.TensorProto:
        """Create a range initializer tensor with a single value."""
        array = np.array([value], dtype=np.int64)
        return onnx.numpy_helper.from_array(array, name=name)

    def test_range_positive_delta(self):
        """Test Range with positive delta."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 0,
                "limit": 10,
                "delta": 1,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)

        # Range [0, 10) with step 1 produces 10 elements
        assert result[0][0] == [10]

    def test_range_with_step_2(self):
        """Test Range with step 2."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 0,
                "limit": 10,
                "delta": 2,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)

        # Range [0, 10) with step 2 produces 5 elements
        assert result[0][0] == [5]

    def test_range_negative_step(self):
        """Test Range with negative step."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 10,
                "limit": 0,
                "delta": -1,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)

        # Range [10, 0) with step -1 produces 10 elements
        assert result[0][0] == [10]

    def test_range_1_to_5(self):
        """Test Range from 1 to 5."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 1,
                "limit": 5,
                "delta": 1,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)

        # Range [1, 5) with step 1 produces 4 elements
        assert result[0][0] == [4]

    def test_range_empty(self):
        """Test Range that produces empty range."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 5,
                "limit": 5,
                "delta": 1,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)

        # Range [5, 5) produces 0 elements
        assert result[0][0] == [0]

    def test_range_zero_delta_error(self):
        """Test Range with zero delta raises error."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": 0,
                "limit": 10,
                "delta": 0,  # Zero delta
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        with pytest.raises(ValueError, match="Range step delta cannot be 0"):
            _infer_range_shape(node, ctx)

    def test_range_non_integer_inputs(self):
        """Test Range with non-integer inputs returns [0]."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={
                "start": [1],  # List, not integer
                "limit": 10,
                "delta": 1,
            },
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )

        result = _infer_range_shape(node, ctx)
        # Non-integer inputs return [0]
        assert result[0][0] == [0]


class TestConstantOfShapeOperation:
    """Test ConstantOfShape operator shape inference."""

    def test_constantofshape_1d(self):
        """Test ConstantOfShape with 1D shape."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape": [10]},
            initializers={},
            verbose=False,
        )

        # Create value tensor for ConstantOfShape
        value_tensor = onnx.numpy_helper.from_array(np.array([5], dtype=np.int64), name="value")

        node = onnx.helper.make_node(
            "ConstantOfShape",
            inputs=["shape"],
            outputs=["output"],
            value=value_tensor,
        )

        result = _infer_constant_of_shape_shape(node, ctx)

        # ConstantOfShape with shape [10] should produce [10]
        assert result[0][0] == [10]

    def test_constantofshape_2d(self):
        """Test ConstantOfShape with 2D shape."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape": [3, 4]},
            initializers={},
            verbose=False,
        )

        value_tensor = onnx.numpy_helper.from_array(np.array([7], dtype=np.int64), name="value")

        node = onnx.helper.make_node(
            "ConstantOfShape",
            inputs=["shape"],
            outputs=["output"],
            value=value_tensor,
        )

        result = _infer_constant_of_shape_shape(node, ctx)

        # ConstantOfShape with shape [3, 4] should produce [3, 4]
        assert result[0][0] == [3, 4]

    def test_constantofshape_with_explicit_value(self):
        """Test ConstantOfShape with explicit constant value."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape": [2, 3]},
            initializers={},
            verbose=False,
        )

        # Integer value
        value_tensor = onnx.numpy_helper.from_array(np.array([42], dtype=np.int64), name="value")

        node = onnx.helper.make_node(
            "ConstantOfShape",
            inputs=["shape"],
            outputs=["output"],
            value=value_tensor,
        )

        result = _infer_constant_of_shape_shape(node, ctx)

        # Data shape should be [2, 3]
        assert result[0][0] == [2, 3]
        # Explicit shape should contain the filled values
        assert result[0][1] == [[42, 42, 42], [42, 42, 42]]

    def test_constantofshape_zero_dimension(self):
        """Test ConstantOfShape with zero dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape": [0]},
            initializers={},
            verbose=False,
        )

        value_tensor = onnx.numpy_helper.from_array(np.array([5], dtype=np.int64), name="value")

        node = onnx.helper.make_node(
            "ConstantOfShape",
            inputs=["shape"],
            outputs=["output"],
            value=value_tensor,
        )

        result = _infer_constant_of_shape_shape(node, ctx)

        # ConstantOfShape with shape [0] should produce [0]
        assert result[0][0] == [0]
        # No explicit shape for zero dimension
        assert result[0][1] is None

    def test_constantofshape_missing_shape_error(self):
        """Test error when shape input is missing."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},  # shape not provided
            initializers={},
            verbose=False,
        )

        value_tensor = onnx.numpy_helper.from_array(np.array([5], dtype=np.int64), name="value")

        node = onnx.helper.make_node(
            "ConstantOfShape",
            inputs=["missing_shape"],
            outputs=["output"],
            value=value_tensor,
        )

        with pytest.raises(RuntimeError, match=r"Cannot get explicit shape"):
            _infer_constant_of_shape_shape(node, ctx)
