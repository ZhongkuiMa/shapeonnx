"""Unit tests for slicing operation shape inference."""

import numpy as np
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_gather_shape,
    _infer_slice_shape,
)


class TestSliceOperation:
    """Test Slice operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "starts", "ends", "axes", "steps", "expected"),
        [
            pytest.param([3, 4, 5], [0], [2], [0], [1], [2, 4, 5], id="slice_first_dim"),
            pytest.param([3, 4, 5], [1], [3], [1], [1], [3, 2, 5], id="slice_middle_dim"),
            pytest.param([3, 4, 5], [0], [4], [2], [1], [3, 4, 4], id="slice_last_dim"),
            pytest.param([8, 8], [0], [8], [0], [2], [4, 8], id="slice_with_step_2"),
            pytest.param([10], [1], [8], [0], [1], [7], id="slice_1d_range"),
        ],
    )
    def test_slice_different_ranges(self, input_shape, starts, ends, axes, steps, expected):
        """Test Slice with different start/end/axis/step values."""
        import onnx

        starts_array = np.array(starts, dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array(ends, dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        axes_array = np.array(axes, dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")
        steps_array = np.array(steps, dtype=np.int64)
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "axes": axes_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice",
            inputs=["input", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        )
        result = _infer_slice_shape(node, ctx)
        assert result[0][0] == expected

    def test_slice_with_zero_dimension(self):
        """Test Slice with zero dimension."""
        import onnx

        starts_array = np.array([0], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([1], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={"starts": starts_tensor, "ends": ends_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice", inputs=["input", "starts", "ends"], outputs=["output"]
        )
        result = _infer_slice_shape(node, ctx)
        assert result[0][0] == [0]

    @pytest.mark.parametrize(
        ("input_shape", "starts", "ends", "expected"),
        [
            pytest.param([5, 4], [1], [5], [4, 4], id="slice_missing_axes"),
            pytest.param([8, 6], [0], [8], [8, 6], id="slice_only_first_axis"),
            pytest.param([10, 10], [-5], [8], [3, 10], id="slice_negative_start"),
        ],
    )
    def test_slice_without_axes_or_steps(self, input_shape, starts, ends, expected):
        """Test Slice with missing axes or steps parameters."""
        import onnx

        starts_array = np.array(starts, dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array(ends, dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"starts": starts_tensor, "ends": ends_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice", inputs=["input", "starts", "ends"], outputs=["output"]
        )
        result = _infer_slice_shape(node, ctx)
        assert result[0][0] == expected

    def test_slice_large_step(self):
        """Test Slice with large step value."""
        import onnx

        starts_array = np.array([0], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([10], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")
        steps_array = np.array([5], dtype=np.int64)
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [10, 4]},
            explicit_shapes={},
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "axes": axes_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice",
            inputs=["input", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        )
        result = _infer_slice_shape(node, ctx)
        # Slice [0:10:5] produces 2 elements: [0, 5]
        assert result[0][0] == [2, 4]


class TestGatherOperation:
    """Test Gather operation shape inference."""

    @pytest.mark.parametrize(
        ("input_shape", "indices", "axis", "expected"),
        [
            pytest.param([3, 4, 5], [0, 2], 0, [2, 4, 5], id="gather_axis_0"),
            pytest.param([3, 4, 5], [1, 3], 1, [3, 2, 5], id="gather_axis_1"),
            pytest.param([3, 4, 5], [2, 4, 0], 2, [3, 4, 3], id="gather_axis_2"),
            pytest.param([10], [0, 5, 9], 0, [3], id="gather_1d_array"),
        ],
    )
    def test_gather_different_axes_and_indices(self, input_shape, indices, axis, expected):
        """Test Gather with different axes and indices."""
        import onnx

        indices_array = np.array(indices, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": input_shape},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=axis
        )
        result = _infer_gather_shape(node, ctx)
        assert result[0][0] == expected

    def test_gather_single_index(self):
        """Test Gather with single scalar index."""
        import onnx

        indices_array = np.array(2, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [5, 4, 3]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        assert result[0][0] == [4, 3]

    def test_gather_with_broadcast_indices(self):
        """Test Gather with scalar index."""
        import onnx

        indices_array = np.array(1, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [5, 4, 3]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        # Gathering with scalar index removes the axis dimension
        assert result[0][0] == [4, 3]

    def test_gather_from_explicit_shape(self):
        """Test Gather from explicit shape tensor (output from Shape op)."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_tensor": [2, 3, 4]},  # Explicit shape from Shape op
            initializers={},
            verbose=False,
        )
        # Gather indices from shape tensor using single index
        indices_array = np.array([1], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")
        ctx.initializers["indices"] = indices_tensor

        node = onnx.helper.make_node(
            "Gather", inputs=["shape_tensor", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        # Should return explicit shape of gathered value
        assert result[0][1] == [3]

    def test_gather_invalid_axis_on_explicit_shape_error(self):
        """Test Gather raises error for invalid axis on explicit shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_tensor": [2, 3, 4]},  # 1D explicit shape
            initializers={},
            verbose=False,
        )
        indices_array = np.array([0], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")
        ctx.initializers["indices"] = indices_tensor

        node = onnx.helper.make_node(
            "Gather", inputs=["shape_tensor", "indices"], outputs=["output"], axis=1
        )  # Invalid axis 1 for 1D shape
        with pytest.raises(ValueError, match="Invalid axis"):
            _infer_gather_shape(node, ctx)

    def test_gather_explicit_shape_non_list_error(self):
        """Test Gather raises error for non-list explicit shape."""
        import onnx

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_scalar": 5},  # Scalar explicit shape (invalid)
            initializers={},
            verbose=False,
        )
        indices_array = np.array([0], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")
        ctx.initializers["indices"] = indices_tensor

        node = onnx.helper.make_node(
            "Gather", inputs=["shape_scalar", "indices"], outputs=["output"], axis=0
        )
        with pytest.raises(RuntimeError, match="Cannot gather from non-list"):
            _infer_gather_shape(node, ctx)


class TestSliceExplicitShapes:
    """Test Slice operation with explicit shapes."""

    def test_slice_explicit_shape_from_shape_tensor(self):
        """Test Slice on explicit shape tensor (output from Shape op)."""
        import onnx

        starts_array = np.array([0], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([2], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")
        steps_array = np.array([1], dtype=np.int64)
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_tensor": [2, 3, 4]},  # Explicit shape from Shape op
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "axes": axes_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice",
            inputs=["shape_tensor", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        )
        result = _infer_slice_shape(node, ctx)
        # Slicing [2, 3, 4][0:2] on axis 0 gives [2, 3, 4]
        assert result[0][1] == [2, 3]

    def test_slice_explicit_shape_non_axis_0_error(self):
        """Test Slice on explicit shape with non-axis-0 raises error."""
        import onnx

        starts_array = np.array([1], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([2], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        axes_array = np.array([1], dtype=np.int64)  # Non-zero axis
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")
        steps_array = np.array([1], dtype=np.int64)
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_tensor": [2, 3, 4]},
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "axes": axes_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice",
            inputs=["shape_tensor", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        )
        with pytest.raises(ValueError, match="Invalid axes"):
            _infer_slice_shape(node, ctx)

    def test_slice_missing_input_error(self):
        """Test Slice raises error when input is missing."""
        import onnx

        starts_array = np.array([0], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([2], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        axes_array = np.array([0], dtype=np.int64)
        axes_tensor = onnx.numpy_helper.from_array(axes_array, name="axes")
        steps_array = np.array([1], dtype=np.int64)
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "axes": axes_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice",
            inputs=["input", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        )
        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_slice_shape(node, ctx)
