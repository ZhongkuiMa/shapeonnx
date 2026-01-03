"""Unit tests for main shape inference API.

This module provides comprehensive test coverage for shapeonnx.infer_shape main
entry points:
- infer_onnx_shape(): Main API for shape inference
- _infer_all_node_shapes(): Helper for inferring all node shapes
- _process_node_outputs(): Helper for processing and storing node outputs

Test organization:
- TestInferOnnxShape: Complete end-to-end shape inference
- TestInferAllNodeShapes: Graph traversal and inference pipeline
- TestProcessNodeOutputs: Output storage and context updates
- TestMainAPIErrors: Error handling and edge cases
"""

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import (
    ShapeInferenceContext,
    _infer_all_node_shapes,
    _process_node_outputs,
    infer_onnx_shape,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _make_weight_tensor(shape: tuple[int, ...], name: str = "weight") -> onnx.TensorProto:
    """Create a test weight tensor for Conv-like operators."""
    rng = np.random.default_rng()
    array = rng.standard_normal(shape).astype(np.float32)
    return onnx.numpy_helper.from_array(array, name=name)


def _make_int_initializer(value: int | list[int], name: str) -> onnx.TensorProto:
    """Create an integer initializer tensor."""
    if isinstance(value, int):
        array = np.array([value], dtype=np.int64)
    else:
        array = np.array(value, dtype=np.int64)
    return onnx.numpy_helper.from_array(array, name=name)


# ============================================================================
# TestInferOnnxShape - Complete End-to-End Shape Inference
# ============================================================================


class TestInferOnnxShape:
    """Test complete shape inference on full ONNX models."""

    def test_infer_onnx_shape_simple_relu(self):
        """Test shape inference on simple model with ReLU operation."""
        input_info = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # ReLU op passes input through unchanged
        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
        )

        assert shapes["input"] == [1, 3, 224, 224]
        assert shapes["output"] == [1, 3, 224, 224]

    def test_infer_onnx_shape_relu_activation(self):
        """Test shape inference on model with ReLU activation."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
        )

        assert shapes["output"] == [1, 10]

    def test_infer_onnx_shape_sequential_ops(self):
        """Test shape inference on sequential operations."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Chain: Input -> Relu -> Cast -> Output
        node1 = onnx.helper.make_node("Relu", inputs=["input"], outputs=["relu_out"])
        node2 = onnx.helper.make_node(
            "Cast", inputs=["relu_out"], outputs=["output"], to=onnx.TensorProto.FLOAT
        )

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node1, node2],
            initializers={},
        )

        assert shapes["input"] == [1, 10]
        assert shapes["relu_out"] == [1, 10]
        assert shapes["output"] == [1, 10]

    def test_infer_onnx_shape_flatten_operation(self):
        """Test shape inference on Flatten operation."""
        input_info = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Flatten with axis=1
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=1)

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
        )

        assert shapes["output"] == [1, 3 * 224 * 224]

    def test_infer_onnx_shape_binary_add_operation(self):
        """Test shape inference on binary Add operation."""
        input1_info = onnx.helper.make_tensor_value_info(
            "input1", onnx.TensorProto.FLOAT, [1, 3, 4]
        )
        input2_info = onnx.helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [3, 4])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("Add", inputs=["input1", "input2"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input1_info, input2_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
        )

        # Broadcasting: [1, 3, 4] + [3, 4] -> [1, 3, 4]
        assert shapes["output"] == [1, 3, 4]

    def test_infer_onnx_shape_with_initializers(self):
        """Test shape inference with initializer parameters."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Weight tensor for Add operation
        weight_array = np.array([1.0] * 10, dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weight_array, "weight")

        node = onnx.helper.make_node("Add", inputs=["input", "weight"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={"weight": weight_tensor},
        )

        assert shapes["output"] == [1, 10]
        assert shapes["weight"] == [10]

    def test_infer_onnx_shape_multiple_outputs(self):
        """Test shape inference on model with multiple outputs."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])

        output1_info = onnx.helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, None)
        output2_info = onnx.helper.make_tensor_value_info("output2", onnx.TensorProto.FLOAT, None)

        # Two separate operations
        node1 = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output1"])
        node2 = onnx.helper.make_node(
            "Cast", inputs=["input"], outputs=["output2"], to=onnx.TensorProto.FLOAT
        )

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output1_info, output2_info],
            nodes=[node1, node2],
            initializers={},
        )

        assert shapes["output1"] == [1, 4]
        assert shapes["output2"] == [1, 4]

    def test_infer_onnx_shape_with_batch_dim_false(self):
        """Test shape inference with has_batch_dim=False."""
        input_info = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [3, 224, 224]
        )
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
            has_batch_dim=False,
        )

        assert shapes["output"] == [3, 224, 224]

    def test_infer_onnx_shape_verbose_mode(self, capsys):
        """Test shape inference with verbose output."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=[node],
            initializers={},
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Input shapes" in captured.out
        assert "output" in captured.out
        assert shapes["output"] == [1, 3]

    def test_infer_onnx_shape_constant_node_error(self):
        """Test error when Constant node is passed to inference."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Constant nodes should be converted to initializers
        node = onnx.helper.make_node("Constant", inputs=[], outputs=["const_out"])

        with pytest.raises(
            RuntimeError,
            match="Constant nodes must be converted to initializers",
        ):
            infer_onnx_shape(
                input_nodes=[input_info],
                output_nodes=[output_info],
                nodes=[node],
                initializers={},
            )

    def test_infer_onnx_shape_unsupported_operator_error(self):
        """Test error when unsupported operator is encountered."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Create a node with unsupported op type
        node = onnx.helper.make_node("NonExistentOp", inputs=["input"], outputs=["output"])

        with pytest.raises(RuntimeError, match="not supported"):
            infer_onnx_shape(
                input_nodes=[input_info],
                output_nodes=[output_info],
                nodes=[node],
                initializers={},
            )


# ============================================================================
# TestInferAllNodeShapes - Graph Traversal and Inference Pipeline
# ============================================================================


class TestInferAllNodeShapes:
    """Test _infer_all_node_shapes helper function."""

    def test_infer_all_node_shapes_single_node(self):
        """Test inference on graph with single node."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])

        _infer_all_node_shapes([node], ctx)

        assert ctx.data_shapes["output"] == [2, 3]

    def test_infer_all_node_shapes_sequential_graph(self):
        """Test inference on sequential computation graph."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 10]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        nodes = [
            onnx.helper.make_node("Relu", inputs=["input"], outputs=["relu_out"]),
            onnx.helper.make_node(
                "Cast", inputs=["relu_out"], outputs=["output"], to=onnx.TensorProto.FLOAT
            ),
        ]

        _infer_all_node_shapes(nodes, ctx)

        assert ctx.data_shapes["relu_out"] == [2, 10]
        assert ctx.data_shapes["output"] == [2, 10]

    def test_infer_all_node_shapes_branching_graph(self):
        """Test inference on branching computation graph."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [3, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        # Create two parallel branches from input
        nodes = [
            onnx.helper.make_node("Relu", inputs=["input"], outputs=["branch1"]),
            onnx.helper.make_node(
                "Cast", inputs=["input"], outputs=["branch2"], to=onnx.TensorProto.FLOAT
            ),
        ]

        _infer_all_node_shapes(nodes, ctx)

        assert ctx.data_shapes["branch1"] == [3, 4]
        assert ctx.data_shapes["branch2"] == [3, 4]

    def test_infer_all_node_shapes_with_initializers(self):
        """Test inference using initializer parameters."""
        weight_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weight_array, "weight")

        ctx = ShapeInferenceContext(
            data_shapes={"input": [1, 3], "weight": [3]},
            explicit_shapes={},
            initializers={"weight": weight_tensor},
            verbose=False,
        )

        node = onnx.helper.make_node("Add", inputs=["input", "weight"], outputs=["output"])

        _infer_all_node_shapes([node], ctx)

        assert ctx.data_shapes["output"] == [1, 3]

    def test_infer_all_node_shapes_constant_node_error(self):
        """Test error handling for Constant nodes."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Constant", inputs=[], outputs=["const_out"])

        with pytest.raises(RuntimeError, match="Constant nodes"):
            _infer_all_node_shapes([node], ctx)

    def test_infer_all_node_shapes_inference_error_propagation(self):
        """Test error handling when shape inference fails for a node."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        # Create an Add node without second input (will cause inference to fail)
        node = onnx.helper.make_node("Add", inputs=["input"], outputs=["output"])

        with pytest.raises(RuntimeError, match="Failed to infer shape"):
            _infer_all_node_shapes([node], ctx)


# ============================================================================
# TestProcessNodeOutputs - Output Storage and Context Updates
# ============================================================================


class TestProcessNodeOutputs:
    """Test _process_node_outputs helper function."""

    def test_process_node_outputs_single_output_data_shape(self):
        """Test storing single output with data shape."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
        results = [([2, 3], None)]

        _process_node_outputs(node, results, ctx)

        assert ctx.data_shapes["output"] == [2, 3]
        assert "output" not in ctx.explicit_shapes

    def test_process_node_outputs_single_output_explicit_shape(self):
        """Test storing single output with explicit shape."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["shape_out"])
        results = [(None, [2, 3])]

        _process_node_outputs(node, results, ctx)

        assert ctx.explicit_shapes["shape_out"] == [2, 3]
        assert ctx.data_shapes["shape_out"] == [2, 3]

    def test_process_node_outputs_multiple_outputs(self):
        """Test storing multiple outputs from single node."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        # Split operation produces multiple outputs
        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output1", "output2"],
            axis=0,
        )
        results = [([2, 3], None), ([2, 3], None)]

        _process_node_outputs(node, results, ctx)

        assert ctx.data_shapes["output1"] == [2, 3]
        assert ctx.data_shapes["output2"] == [2, 3]

    def test_process_node_outputs_both_shapes(self):
        """Test output with both data and explicit shapes."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node(
            "Range", inputs=["start", "limit", "delta"], outputs=["output"]
        )
        # Range can produce both data shape and explicit values
        results = [([10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]

        _process_node_outputs(node, results, ctx)

        assert ctx.data_shapes["output"] == [10]
        assert ctx.explicit_shapes["output"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_process_node_outputs_explicit_fallback_to_data_shape(self):
        """Test that explicit shape becomes data shape when data shape is None."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])
        # Only explicit shape provided
        results = [(None, [1, 3, 224, 224])]

        _process_node_outputs(node, results, ctx)

        assert ctx.explicit_shapes["output"] == [1, 3, 224, 224]
        assert ctx.data_shapes["output"] == [1, 3, 224, 224]

    def test_process_node_outputs_verbose_mode(self, capsys):
        """Test verbose output when storing node results."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=True,
        )

        node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
        results = [([2, 3], None)]

        _process_node_outputs(node, results, ctx)

        captured = capsys.readouterr()
        assert "Relu" in captured.out
        assert "output" in captured.out
        assert "[2, 3]" in captured.out


# ============================================================================
# TestMainAPIErrors - Error Handling and Edge Cases
# ============================================================================


class TestMainAPIErrors:
    """Test error handling in main API functions."""

    def test_error_invalid_broadcast_in_add(self):
        """Test error when incompatible shapes cannot be broadcast."""
        input1_info = onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [2, 3])
        input2_info = onnx.helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [4, 5])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        node = onnx.helper.make_node("Add", inputs=["input1", "input2"], outputs=["output"])

        with pytest.raises(RuntimeError, match=r"(Failed to infer shape|broadcast)"):
            infer_onnx_shape(
                input_nodes=[input1_info, input2_info],
                output_nodes=[output_info],
                nodes=[node],
                initializers={},
            )

    def test_error_missing_input_shape(self):
        """Test error when required input shape is missing."""
        ctx = ShapeInferenceContext(
            data_shapes={"input1": [2, 3]},  # Missing "input2"
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        node = onnx.helper.make_node("Add", inputs=["input1", "input2"], outputs=["output"])

        with pytest.raises(RuntimeError):
            _infer_all_node_shapes([node], ctx)

    def test_error_node_has_multiple_outputs_mismatch(self):
        """Test error when results count doesn't match node outputs."""
        ctx = ShapeInferenceContext(
            data_shapes={},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )

        # Node declares 2 outputs but we provide 1 result
        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output1", "output2"],
        )

        with pytest.raises((ValueError, AssertionError)):
            _process_node_outputs(node, [([2, 3], None)], ctx)

    def test_error_node_with_missing_implementation(self):
        """Test error when operator inference function is not implemented."""
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Use an operator that doesn't have inference implemented
        node = onnx.helper.make_node("UnimplementedOp", inputs=["input"], outputs=["output"])

        with pytest.raises(RuntimeError, match=r"(not supported|NotImplementedError)"):
            infer_onnx_shape(
                input_nodes=[input_info],
                output_nodes=[output_info],
                nodes=[node],
                initializers={},
            )


# ============================================================================
# TestMainAPIIntegration - Integration Tests with Real Operators
# ============================================================================


class TestMainAPIIntegration:
    """Integration tests with realistic ONNX model scenarios."""

    def test_integration_cnn_inference_pipeline(self):
        """Test inference on simplified CNN-like pipeline."""
        # Input -> Conv -> ReLU -> Flatten -> Output
        input_info = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 28, 28]
        )
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

        # Create Conv weight (3 input channels, 16 output channels, 3x3 kernel)
        weight_tensor = _make_weight_tensor((16, 3, 3, 3), "weight")

        nodes = [
            onnx.helper.make_node(
                "Conv",
                inputs=["input", "weight"],
                outputs=["conv_out"],
                kernel_shape=[3, 3],
            ),
            onnx.helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"]),
            onnx.helper.make_node("Flatten", inputs=["relu_out"], outputs=["output"], axis=1),
        ]

        shapes = infer_onnx_shape(
            input_nodes=[input_info],
            output_nodes=[output_info],
            nodes=nodes,
            initializers={"weight": weight_tensor},
        )

        assert shapes["input"] == [1, 3, 28, 28]
        assert "conv_out" in shapes
        assert "relu_out" in shapes
        assert shapes["output"][0] == 1  # Batch size preserved
