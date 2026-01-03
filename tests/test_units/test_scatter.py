"""Unit tests for scatter operation shape inference."""

import numpy as np
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext, _infer_gather_shape


class TestScatterIndexing:
    """Test scatter-like indexing operations through Gather."""

    @pytest.mark.parametrize(
        ("data_shape", "indices", "axis", "expected"),
        [
            pytest.param([5, 4, 3], [0, 2, 4], 0, [3, 4, 3], id="scatter_axis_0_3_indices"),
            pytest.param([5, 4, 3], [1, 3], 1, [5, 2, 3], id="scatter_axis_1_2_indices"),
            pytest.param([5, 4, 3], [0, 1, 2], 2, [5, 4, 3], id="scatter_axis_2_3_indices"),
        ],
    )
    def test_gather_as_scatter_indexing(self, data_shape, indices, axis, expected):
        """Test Gather operation mimicking scatter indexing behavior."""
        import onnx

        indices_array = np.array(indices, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": data_shape},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=axis
        )
        result = _infer_gather_shape(node, ctx)
        assert result[0][0] == expected

    def test_scatter_multiple_indices(self):
        """Test scatter-like gather with multiple batch indices."""
        import onnx

        # 2D data with multiple index selections
        data_shape = [10, 8]
        indices = [1, 3, 5, 7, 9]
        indices_array = np.array(indices, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": data_shape},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        # Output shape: [5, 8] (5 selected rows, 8 columns)
        assert result[0][0] == [5, 8]

    def test_scatter_single_index(self):
        """Test scatter gather with single index."""
        import onnx

        indices_array = np.array(3, dtype=np.int64)  # Scalar index
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": [10, 5, 4]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        # Scalar index returns shape without that dimension
        assert result[0][0] == [5, 4]

    def test_scatter_multidimensional_indices(self):
        """Test gather with 2D indices array."""
        import onnx

        # 2D indices - gather treats it as a flat list by default
        indices = [[0, 1], [2, 3]]
        indices_array = np.array(indices, dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": [10, 8]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=0
        )
        result = _infer_gather_shape(node, ctx)
        # Output shape: [2, 8] (2 selected rows with 8 columns each)
        # The indices shape [2, 2] is flattened in counting
        assert result[0][0] == [2, 8]

    def test_scatter_negative_axis(self):
        """Test gather with negative axis indexing."""
        import onnx

        indices_array = np.array([0, 1, 3], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": [5, 4, 3]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        # axis=-1 is equivalent to axis=2
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=-1
        )
        result = _infer_gather_shape(node, ctx)
        # Output: [5, 4, 3] with last dim replaced by indices size
        assert result[0][0] == [5, 4, 3]

    def test_gather_with_zero_dimension(self):
        """Test gather preserving zero dimensions."""
        import onnx

        indices_array = np.array([0, 1], dtype=np.int64)
        indices_tensor = onnx.numpy_helper.from_array(indices_array, name="indices")

        ctx = ShapeInferenceContext(
            data_shapes={"data": [0, 5, 3]},
            explicit_shapes={},
            initializers={"indices": indices_tensor},
            verbose=False,
        )
        node = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=1
        )
        result = _infer_gather_shape(node, ctx)
        # Zero dimension propagates: [0, 2, 3] not [2, 5, 3]
        assert result[0][0] == [0, 2, 3]
