"""Unit tests for shapeonnx utility functions."""

import numpy as np
import onnx
import pytest

from shapeonnx.utils import (
    _reformat_io_shape,
    convert_constant_to_initializer,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)


class TestReformatIOShape:
    """Tests for reformat_io_shape function."""

    def test_reformat_with_batch_dim(self):
        """Test reformatting shape with batch dimension."""
        tensor_info = onnx.helper.make_tensor_value_info(
            "test", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        result = _reformat_io_shape(tensor_info, has_batch_dim=True)
        assert result == [1, 3, 224, 224]

    def test_reformat_without_batch_dim(self):
        """Test reformatting shape without batch dimension check."""
        tensor_info = onnx.helper.make_tensor_value_info(
            "test", onnx.TensorProto.FLOAT, [3, 224, 224]
        )
        result = _reformat_io_shape(tensor_info, has_batch_dim=False)
        assert result == [3, 224, 224]

    def test_reformat_scalar_output(self):
        """Test reformatting scalar output."""
        tensor_info = onnx.helper.make_tensor_value_info("test", onnx.TensorProto.FLOAT, [])
        result = _reformat_io_shape(tensor_info, has_batch_dim=True)
        assert result == []

    def test_reformat_invalid_batch_dim(self):
        """Test reformatting with invalid batch dimension."""
        tensor_info = onnx.helper.make_tensor_value_info("test", onnx.TensorProto.FLOAT, [1])
        with pytest.raises(ValueError, match="Expected batch dimension"):
            _reformat_io_shape(tensor_info, has_batch_dim=True)

    def test_reformat_normalize_batch_dim(self):
        """Test normalizing batch dimension to 1."""
        tensor_info = onnx.helper.make_tensor_value_info(
            "test", onnx.TensorProto.FLOAT, [32, 3, 224, 224]
        )
        result = _reformat_io_shape(tensor_info, has_batch_dim=True)
        assert result[0] == 1
        assert result[1:] == [3, 224, 224]


class TestGetInitializers:
    """Tests for get_initializers function."""

    def test_get_initializers_empty(self):
        """Test getting initializers from model with none."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10]
        )
        graph = onnx.helper.make_graph([], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        result = get_initializers(model)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_initializers_with_weights(self):
        """Test getting initializers from model with weights."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10]
        )

        # Create weights
        rng = np.random.default_rng(seed=42)
        weights = rng.standard_normal((10, 3)).astype(np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weights, name="weight")

        bias = rng.standard_normal(10).astype(np.float32)
        bias_tensor = onnx.numpy_helper.from_array(bias, name="bias")

        graph = onnx.helper.make_graph(
            [], "test_graph", [input_tensor], [output_tensor], [weight_tensor, bias_tensor]
        )
        model = onnx.helper.make_model(graph)

        result = get_initializers(model)
        assert "weight" in result
        assert "bias" in result
        assert len(result) == 2

    def test_get_initializers_names(self):
        """Test that initializer names are correctly mapped."""
        input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])
        output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])

        weights = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weights, name="my_weights")

        graph = onnx.helper.make_graph(
            [], "test_graph", [input_tensor], [output_tensor], [weight_tensor]
        )
        model = onnx.helper.make_model(graph)

        result = get_initializers(model)
        assert "my_weights" in result


class TestGetInputNodes:
    """Tests for get_input_nodes function."""

    def test_get_input_nodes_single_input(self):
        """Test getting input nodes from model with single input."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10]
        )
        graph = onnx.helper.make_graph([], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        result = get_input_nodes(model, {}, has_batch_dim=True)
        assert len(result) == 1
        assert result[0].name == "input"

    def test_get_input_nodes_exclude_initializers(self):
        """Test that initializers are excluded from input nodes."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10]
        )

        rng = np.random.default_rng(seed=42)
        weights = rng.standard_normal((10, 3)).astype(np.float32)
        weight_tensor = onnx.numpy_helper.from_array(weights, name="weight")

        # Create model with input that is also an initializer
        model_input = onnx.helper.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, [10, 3])
        graph = onnx.helper.make_graph(
            [], "test_graph", [input_tensor, model_input], [output_tensor], [weight_tensor]
        )
        model = onnx.helper.make_model(graph)

        result = get_input_nodes(model, {"weight": weight_tensor}, has_batch_dim=True)
        # Only "input" should be in the result, not "weight"
        assert len(result) == 1
        assert result[0].name == "input"


class TestGetOutputNodes:
    """Tests for get_output_nodes function."""

    def test_get_output_nodes_single_output(self):
        """Test getting output nodes from model with single output."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 10]
        )
        graph = onnx.helper.make_graph([], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        result = get_output_nodes(model, has_batch_dim=True)
        assert len(result) == 1
        assert result[0].name == "output"

    def test_get_output_nodes_multiple_outputs(self):
        """Test getting output nodes from model with multiple outputs."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor1 = onnx.helper.make_tensor_value_info(
            "output1", onnx.TensorProto.FLOAT, [1, 10]
        )
        output_tensor2 = onnx.helper.make_tensor_value_info(
            "output2", onnx.TensorProto.FLOAT, [1, 5]
        )
        graph = onnx.helper.make_graph(
            [], "test_graph", [input_tensor], [output_tensor1, output_tensor2]
        )
        model = onnx.helper.make_model(graph)

        result = get_output_nodes(model, has_batch_dim=True)
        assert len(result) == 2
        output_names = {node.name for node in result}
        assert "output1" in output_names
        assert "output2" in output_names


class TestConvertConstantToInitializer:
    """Tests for convert_constant_to_initializer function."""

    def test_convert_no_constant_nodes(self):
        """Test converting nodes with no Constant nodes."""
        node1 = onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"])
        nodes = [node1]
        initializers: dict[str, onnx.TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)
        assert len(result) == 1
        assert result[0].op_type == "Relu"
        assert len(initializers) == 0

    def test_convert_with_constant_nodes(self):
        """Test converting nodes with Constant nodes."""
        # Create a Constant node
        constant_value = onnx.helper.make_tensor(
            "const_val", onnx.TensorProto.FLOAT, [2, 3], [1.0] * 6
        )
        const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output"],
            value=constant_value,
        )

        relu_node = onnx.helper.make_node("Relu", inputs=["const_output"], outputs=["y"])
        nodes = [const_node, relu_node]
        initializers: dict[str, onnx.TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)
        assert len(result) == 1
        assert result[0].op_type == "Relu"
        assert "const_output" in initializers

    def test_convert_multiple_constant_nodes(self):
        """Test converting multiple Constant nodes."""
        const1 = onnx.helper.make_tensor("c1", onnx.TensorProto.FLOAT, [2], [1.0, 2.0])
        const_node1 = onnx.helper.make_node("Constant", inputs=[], outputs=["out1"], value=const1)

        const2 = onnx.helper.make_tensor("c2", onnx.TensorProto.FLOAT, [2], [3.0, 4.0])
        const_node2 = onnx.helper.make_node("Constant", inputs=[], outputs=["out2"], value=const2)

        add_node = onnx.helper.make_node("Add", inputs=["out1", "out2"], outputs=["result"])
        nodes = [const_node1, const_node2, add_node]
        initializers: dict[str, onnx.TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)
        assert len(result) == 1
        assert result[0].op_type == "Add"
        assert "out1" in initializers
        assert "out2" in initializers
