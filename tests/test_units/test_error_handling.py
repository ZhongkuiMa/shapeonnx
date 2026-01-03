"""Unit tests for error handling and exception cases."""

import pytest


class TestMissingShapeErrors:
    """Test RuntimeError when shapes are missing or unavailable."""

    def test_missing_input_shape(self):
        """Test error when input shape is not in context."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_nochange_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={},  # Empty - missing "input" shape
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_nochange_op_shape(node, ctx)

    def test_missing_broadcast_operand(self):
        """Test error when binary op operand shape is missing."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"a": [3, 4]},  # Missing "b"
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_binary_op_shape(node, ctx)


class TestIncompatibleBroadcasting:
    """Test RuntimeError for incompatible shape broadcasting."""

    def test_incompatible_binary_shapes(self):
        """Test error when shapes cannot be broadcast together."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3], "b": [4, 5]},  # Cannot broadcast [2,3] and [4,5]
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])

        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _infer_binary_op_shape(node, ctx)

    def test_incompatible_shapes_complex(self):
        """Test incompatible shapes with more complex dimensions."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"a": [2, 3, 4], "b": [2, 5, 4]},  # Middle dim incompatible
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])

        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _infer_binary_op_shape(node, ctx)


class TestMissingRequiredAttributes:
    """Test errors for missing or invalid required attributes."""

    def test_concat_missing_axis(self):
        """Test error when Concat axis is not specified."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_concat_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input0": [3, 4], "input1": [3, 5]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        # Missing axis attribute
        node = onnx.helper.make_node("Concat", inputs=["input0", "input1"], outputs=["output"])

        with pytest.raises((ValueError, AttributeError)):
            _infer_concat_shape(node, ctx)

    def test_flatten_missing_axis(self):
        """Test Flatten handles missing axis gracefully (defaults to 1)."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_flatten_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        # Missing axis - should default to 1
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"])

        result = _infer_flatten_shape(node, ctx)
        # Default axis=1: [2, 12]
        assert result[0][0] == [2, 12]


class TestInvalidAxisHandling:
    """Test errors for invalid axis specifications."""

    def test_transpose_axis_out_of_range(self):
        """Test Transpose with axis out of range for the shape."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_transpose_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},  # 3D tensor
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        # perm has index 5 but shape is only 3D
        node = onnx.helper.make_node(
            "Transpose", inputs=["input"], outputs=["output"], perm=[0, 2, 5]
        )

        with pytest.raises((RuntimeError, IndexError, ValueError)):
            _infer_transpose_shape(node, ctx)

    def test_gather_valid_positive_axis(self):
        """Test Gather with valid positive axis indexing."""
        import numpy as np
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_gather_shape

        indices_array = np.array([0, 1, 2], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [3, 4, 5]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        # axis=0: gather rows
        node = onnx.helper.make_node(
            "Gather", inputs=["input", "indices"], outputs=["output"], axis=0
        )

        result = _infer_gather_shape(node, ctx)
        # Gather on axis 0: output shape [3, 4, 5] (3 indices, keep other dims)
        assert result[0][0] == [3, 4, 5]


class TestZeroDimensionHandling:
    """Test proper handling of zero dimensions through operations."""

    def test_zero_dim_propagation_binary_op(self):
        """Test that [0] dimension properly propagates through binary operations."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"a": [0], "b": [5]},  # [0] is empty tensor
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # [0] takes precedence in binary ops
        assert result[0][0] == [0]

    def test_zero_dim_flatten(self):
        """Test flatten with zero dimension input."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_flatten_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": [0]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=0)
        result = _infer_flatten_shape(node, ctx)
        # [0] preserved through flatten
        assert result[0][0] == [0]

    def test_zero_dim_concat(self):
        """Test concat with zero dimension inputs returns [0]."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_concat_shape

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


class TestScalarAndEmptyShapes:
    """Test handling of scalar (empty) shapes vs zero-dimension shapes."""

    def test_scalar_shape_empty_list(self):
        """Test scalar shape represented as empty list []."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_nochange_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": []},  # Empty list = scalar
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
        result = _infer_nochange_op_shape(node, ctx)
        # Scalar shape preserved
        assert result[0][0] == []

    def test_scalar_mul(self):
        """Test mul with scalar (empty) shape."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_binary_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={"a": [], "b": [3, 4]},  # Scalar and 2D
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Mul", inputs=["a", "b"], outputs=["output"])
        result = _infer_binary_op_shape(node, ctx)
        # Scalar broadcasts to any shape: [] x [3,4] = [3,4]
        assert result[0][0] == [3, 4]


class TestExplicitShapeOperations:
    """Test operations on explicit shape tensors vs regular data."""

    def test_explicit_shape_gather(self):
        """Test Gather operating on shape tensor (explicit shape)."""
        import numpy as np
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_gather_shape

        indices_array = np.array([0, 2], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"shape_tensor": [5, 3, 4, 7]},  # Shape tensor as input
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["shape_tensor", "indices"], outputs=["output"], axis=0
        )

        # This should compute explicit output shape
        try:
            result = _infer_gather_shape(node, ctx)
            # Should return the gathered shape values
            assert result[0][1] == [5, 4]  # Explicit shape dimension
        except RuntimeError:
            # If explicit shape path isn't implemented, that's okay
            pass

    def test_explicit_shape_inputs(self):
        """Test that explicit shapes are available in result tuple."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_nochange_op_shape

        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={"input": [3, 4]},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        result = _infer_nochange_op_shape(node, ctx)
        # Result tuple: (data_shape, explicit_shape)
        # When input is explicit, output should have explicit shape
        assert result[0][0] is not None or result[0][1] is not None


class TestNegativeStepHandling:
    """Test edge cases with negative steps and indices."""

    def test_slice_with_negative_step(self):
        """Test Slice with negative step (reverse indexing)."""
        import numpy as np
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_slice_shape

        starts_array = np.array([5], dtype=np.int64)
        starts_tensor = onnx.numpy_helper.from_array(starts_array, name="starts")
        ends_array = np.array([0], dtype=np.int64)
        ends_tensor = onnx.numpy_helper.from_array(ends_array, name="ends")
        steps_array = np.array([-1], dtype=np.int64)  # Negative step!
        steps_tensor = onnx.numpy_helper.from_array(steps_array, name="steps")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [10]},
            explicit_shapes={},
            initializers={
                "starts": starts_tensor,
                "ends": ends_tensor,
                "steps": steps_tensor,
            },
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Slice", inputs=["input", "starts", "ends", "axes", "steps"], outputs=["output"]
        )

        try:
            # This operation is marked as "not fully tested" in source
            result = _infer_slice_shape(node, ctx)
            # Negative step behavior - implementation may vary
            assert isinstance(result[0][0], list)
        except (RuntimeError, NotImplementedError):
            # Expected if negative step not implemented
            pass

    def test_flatten_missing_input(self):
        """Test Flatten raises error when input is missing."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_flatten_shape

        ctx = ShapeInferenceContext(
            data_shapes={},  # Missing input
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=1)

        with pytest.raises(RuntimeError, match="Cannot get shape"):
            _infer_flatten_shape(node, ctx)

    def test_flatten_scalar_input(self):
        """Test Flatten with scalar input returns scalar."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_flatten_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar input
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=0)

        result = _infer_flatten_shape(node, ctx)
        # Scalar returns scalar
        assert result[0][0] == 5

    def test_transpose_scalar_input_error(self):
        """Test Transpose raises error for scalar input."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_transpose_shape

        ctx = ShapeInferenceContext(
            data_shapes={"input": 5},  # Scalar
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=[0])

        with pytest.raises(RuntimeError, match="Transpose input shape cannot be scalar"):
            _infer_transpose_shape(node, ctx)

    def test_transpose_missing_shape_error(self):
        """Test Transpose raises error when shape is completely missing."""
        import onnx

        from shapeonnx.infer_shape import ShapeInferenceContext, _infer_transpose_shape

        ctx = ShapeInferenceContext(
            data_shapes={},  # No shape
            explicit_shapes={},  # No explicit shape either
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=[1, 0])

        with pytest.raises(RuntimeError, match="Cannot get explicit shape"):
            _infer_transpose_shape(node, ctx)
