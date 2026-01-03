"""Unit tests for concat operation shape inference."""

import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_concat_shape


class TestConcatOperation:
    """Test Concat operation shape inference."""

    @pytest.mark.parametrize(
        ("shapes", "axis", "expected"),
        [
            pytest.param([[3, 4], [3, 4]], 0, [6, 4], id="concat_axis_0"),
            pytest.param([[3, 4], [3, 5]], 1, [3, 9], id="concat_axis_1"),
            pytest.param([[2, 3, 4], [2, 3, 5]], 2, [2, 3, 9], id="concat_axis_2_3d"),
            pytest.param([[5], [3]], 0, [8], id="concat_1d_vectors"),
            pytest.param([[2, 3, 4], [2, 3, 4], [2, 3, 4]], 0, [6, 3, 4], id="concat_three_inputs"),
        ],
    )
    def test_concat_different_axes(self, shapes, axis, expected):
        """Test Concat along different axes."""
        import onnx

        input_names = [f"input{i}" for i in range(len(shapes))]
        data_shapes = dict(zip(input_names, shapes, strict=True))

        ctx = ShapeInferenceContext(
            data_shapes=data_shapes,
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=input_names, outputs=["output"], axis=axis)
        result = _infer_concat_shape(node, ctx)
        assert result[0][0] == expected

    def test_concat_with_zero_dimension(self):
        """Test Concat with zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [0], "input1": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=0
        )
        result = _infer_concat_shape(node, ctx)
        assert result[0][0] == [0]

    def test_concat_2d_shapes(self):
        """Test Concat with 2D shapes on different axes."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [2, 3], "input1": [2, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=1
        )
        result = _infer_concat_shape(node, ctx)
        assert result[0][0] == [2, 7]

    def test_concat_4d_inputs(self):
        """Test Concat with 4D inputs."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [2, 3, 4, 5], "input1": [2, 3, 4, 6]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=3
        )
        result = _infer_concat_shape(node, ctx)
        assert result[0][0] == [2, 3, 4, 11]

    def test_concat_negative_axis(self):
        """Test Concat with negative axis."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [2, 3, 4], "input1": [2, 3, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=-1
        )
        result = _infer_concat_shape(node, ctx)
        # Negative axis -1 is same as axis=2 for 3D
        assert result[0][0] == [2, 3, 9]

    def test_concat_missing_input_error(self):
        """Test Concat raises error when input is missing."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [2, 3]},  # Missing input1
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=1
        )
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_concat_shape(node, ctx)

    def test_concat_scalar_input_error(self):
        """Test Concat raises error when input is scalar."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"input0": 5, "input1": [3, 5]},  # Scalar input
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=1
        )
        with pytest.raises(RuntimeError, match="Cannot concatenate scalar"):
            _infer_concat_shape(node, ctx)

    def test_concat_explicit_scalar_error(self):
        """Test Concat raises error when explicit shape is scalar."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input0": 5, "input1": [3, 5]},  # Scalar explicit shape
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=1
        )
        with pytest.raises(RuntimeError, match="Cannot concatenate scalar"):
            _infer_concat_shape(node, ctx)

    def test_concat_zero_all_inputs(self):
        """Test Concat when all inputs have zero dimension."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=["a", "b"], outputs=["output"], axis=0)
        result = _infer_concat_shape(node, ctx)
        # Zero dimension: returns [0]
        assert result[0][0] == [0]
