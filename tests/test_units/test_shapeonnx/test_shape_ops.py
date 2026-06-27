"""Unit tests for shape operation shape inference."""

__docformat__ = "restructuredtext"

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
            pytest.param([0], [0], id="shape_zero_dim"),
        ],
    )
    def test_shape_operator(self, input_shape, expected_shape):
        """Test Shape operator returns input shape as explicit shape."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        assert len(result) >= 1
        assert result[0][1] == expected_shape

    @pytest.mark.parametrize(
        ("explicit_input", "expected_explicit"),
        [
            pytest.param([2, 3, 4], [2, 3, 4], id="list_3d"),
            pytest.param(5, [], id="scalar"),
            pytest.param([0], [0], id="zero_dimension"),
        ],
    )
    def test_shape_with_explicit_input(self, explicit_input, expected_explicit):
        """Test Shape with various explicit shape inputs."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": explicit_input},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        result = _infer_shape_op_shape(node, ctx)
        assert result[0][1] == expected_explicit

    def test_shape_data_shape_not_list_error(self):
        """Test Shape raises error when data shape is not a list."""
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
        assert len(result) >= 1
        assert result[0][0] == expected

    @pytest.mark.parametrize(
        ("input_shape", "pads", "expected"),
        [
            pytest.param([0], [0, 0, 0, 0], [0], id="zero_dim"),
            pytest.param([2, 3, 4], [1, 1, 2, 2, 1, 1], [5, 5, 7], id="3d"),
        ],
    )
    def test_pad_edge_cases(self, input_shape, pads, expected):
        """Test Pad with edge case inputs."""
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

    def test_pad_with_explicit_shape(self):
        """Test Pad with explicit shape input."""
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
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={},  # Missing pads
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"])
        with pytest.raises(KeyError, match="pads"):
            _infer_pad_shape(node, ctx)

    def test_pad_attr_pads_opset_lt_11(self):
        """Pad with pads from node attribute (opset < 11, single input)."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [3, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Pad", inputs=["input"], outputs=["output"], pads=[0, 0, 1, 2])
        result = _infer_pad_shape(node, ctx)
        assert result[0][0] == [4, 6]


class TestResizeOperation:
    """Test Resize operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "scales", "expected"),
        [
            pytest.param([1, 3, 4, 4], [1.0, 1.0, 2.0, 2.0], [1, 3, 8, 8], id="resize_scale_2x"),
            pytest.param([2, 3, 8, 8], [1.0, 1.0, 0.5, 0.5], [2, 3, 4, 4], id="resize_scale_0.5x"),
            pytest.param([1, 4, 3], [1.0, 2.0, 3.0], [1, 8, 9], id="resize_1d_asymmetric"),
            pytest.param([0], [1.0, 1.0, 2.0], [0], id="resize_zero_dim"),
        ],
    )
    def test_resize_different_scales(self, input_shape, scales, expected):
        """Test Resize with different scale factors."""
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
        assert len(result) >= 1
        assert result[0][0] == expected

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
        assert len(result) >= 1
        assert result[0][0] == expected

    @pytest.mark.parametrize(
        ("data_shapes", "scales", "node_kwargs", "exc_type", "match_pattern"),
        [
            pytest.param(
                {},
                [1.0, 1.0, 2.0],
                {
                    "coordinate_transformation_mode": "asymmetric",
                },
                RuntimeError,
                "Cannot get shape",
                id="missing_input_shape",
            ),
            pytest.param(
                {"input": 5},
                [1.0, 1.0],
                {
                    "coordinate_transformation_mode": "asymmetric",
                },
                RuntimeError,
                "Resize input shape cannot be scalar",
                id="scalar_input",
            ),
            pytest.param(
                {"input": [2, 3, 4]},
                [],
                {
                    "coordinate_transformation_mode": "asymmetric",
                },
                ValueError,
                "Resize with empty scales",
                id="empty_scales",
            ),
            pytest.param(
                {"input": [2, 3, 4]},
                [1.0, 1.0, 2.0],
                {
                    "coordinate_transformation_mode": "asymmetric",
                    "mode": "linear",
                },
                NotImplementedError,
                "Resize mode=linear",
                id="unsupported_mode",
            ),
            pytest.param(
                {"input": [2, 3, 4]},
                [1.0, 1.0, 2.0],
                {
                    "coordinate_transformation_mode": "tf_crop_and_resize",
                },
                NotImplementedError,
                "Resize align_mode=",
                id="unsupported_align_mode",
            ),
            pytest.param(
                {"input": [2, 3, 4]},
                [1.0, 1.0, 2.0],
                {
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "invalid_nearest_mode",
                },
                NotImplementedError,
                "Resize nearest_mode=",
                id="unsupported_nearest_mode",
            ),
        ],
    )
    def test_resize_raises_for_invalid_inputs(
        self, data_shapes, scales, node_kwargs, exc_type, match_pattern
    ):
        """Test Resize raises the expected error for invalid inputs/attrs."""
        scales_array = np.array(scales, dtype=np.float32)
        scales_tensor = onnx.numpy_helper.from_array(scales_array, name="scales")

        ctx = ShapeInferenceContext(
            data_shapes=data_shapes,
            explicit_shapes={},
            initializers={"scales": scales_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            **node_kwargs,
        )
        with pytest.raises(exc_type, match=match_pattern):
            _infer_resize_shape(node, ctx)


class TestRangeOperation:
    """Test Range operator shape inference."""

    def _make_range_initializer(self, value: int, name: str) -> onnx.TensorProto:
        """Create a range initializer tensor with a single value."""
        array = np.array([value], dtype=np.int64)
        return onnx.numpy_helper.from_array(array, name=name)

    @pytest.mark.parametrize(
        ("start", "limit", "delta", "expected"),
        [
            pytest.param(0, 10, 1, [10], id="positive_delta"),
            pytest.param(0, 10, 2, [5], id="step_2"),
            pytest.param(10, 0, -1, [10], id="negative_step"),
            pytest.param(1, 5, 1, [4], id="1_to_5"),
            pytest.param(5, 5, 1, [0], id="empty"),
        ],
    )
    def test_range(self, start, limit, delta, expected):
        """Test Range shape inference with various start/limit/delta values."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"start": start, "limit": limit, "delta": delta},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )
        result = _infer_range_shape(node, ctx)
        assert result[0][0] == expected

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

    @pytest.mark.parametrize(
        ("shape", "expected_shape"),
        [
            pytest.param([10], [10], id="1d"),
            pytest.param([3, 4], [3, 4], id="2d"),
            pytest.param([0], [0], id="zero_dimension"),
        ],
    )
    def test_constantofshape_produces_expected_shape(self, shape, expected_shape):
        """Test ConstantOfShape produces correct output shape."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape": shape},
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

        # ConstantOfShape with given shape should produce it
        assert result[0][0] == expected_shape
        if expected_shape == [0]:
            assert result[0][1] is None

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
