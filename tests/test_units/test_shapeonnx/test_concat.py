"""Unit tests for concat operation shape inference."""

__docformat__ = "restructuredtext"

import onnx
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
        assert len(result) >= 1
        assert result[0][0] == expected

    @pytest.mark.parametrize(
        ("input0_shape", "input1_shape", "axis", "expected"),
        [
            pytest.param([0], [0], 0, [0], id="zero_dimension"),
            pytest.param([2, 3], [2, 4], 1, [2, 7], id="2d_axis1"),
            pytest.param([2, 3, 4, 5], [2, 3, 4, 6], 3, [2, 3, 4, 11], id="4d_axis3"),
        ],
    )
    def test_concat_two_inputs(self, input0_shape, input1_shape, axis, expected):
        """Test Concat with two inputs across various shapes/axes."""
        ctx = ShapeInferenceContext(
            data_shapes={"input0": input0_shape, "input1": input1_shape},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=axis
        )
        result = _infer_concat_shape(node, ctx)
        assert len(result) >= 1
        assert result[0][0] == expected

    def test_concat_negative_axis(self):
        """Test Concat with negative axis."""
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
        assert len(result) >= 1
        assert result[0][0] == [2, 3, 9]

    @pytest.mark.parametrize(
        ("data_shapes", "explicit_shapes", "match_pattern"),
        [
            pytest.param(
                {"input0": [2, 3]},
                {},
                "Cannot get shape",
                id="missing_input",
            ),
            pytest.param(
                {"input0": 5, "input1": [3, 5]},
                {},
                "Cannot concatenate scalar",
                id="scalar_data_input",
            ),
            pytest.param(
                {},
                {"input0": 5, "input1": [3, 5]},
                "Cannot concatenate scalar",
                id="scalar_explicit_input",
            ),
        ],
    )
    def test_concat_raises_for_invalid_inputs(self, data_shapes, explicit_shapes, match_pattern):
        """Test Concat raises RuntimeError for invalid inputs."""
        ctx = ShapeInferenceContext(
            data_shapes=data_shapes,
            explicit_shapes=explicit_shapes,
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Concat", inputs=["input0", "input1"], outputs=["output"], axis=1
        )
        with pytest.raises(RuntimeError, match=match_pattern):
            _infer_concat_shape(node, ctx)

    def test_concat_zero_all_inputs(self):
        """Test Concat when all inputs have zero dimension."""
        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Concat", inputs=["a", "b"], outputs=["output"], axis=0)
        result = _infer_concat_shape(node, ctx)
        # Zero dimension: returns [0]
        assert len(result) >= 1
        assert result[0][0] == [0]
