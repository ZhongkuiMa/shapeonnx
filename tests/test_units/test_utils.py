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

    @pytest.mark.parametrize(
        ("tensor_shape", "has_batch_dim", "expected"),
        [
            pytest.param([1, 3, 224, 224], True, [1, 3, 224, 224], id="with_batch_dim"),
            pytest.param([3, 224, 224], False, [3, 224, 224], id="without_batch_dim"),
            pytest.param([], True, [], id="scalar_output"),
            pytest.param([32, 3, 224, 224], True, [1, 3, 224, 224], id="normalize_batch_dim"),
        ],
    )
    def test_reformat_shape(self, tensor_shape, has_batch_dim, expected):
        """Test reformatting shape with various configurations."""
        tensor_info = onnx.helper.make_tensor_value_info(
            "test", onnx.TensorProto.FLOAT, tensor_shape
        )
        result = _reformat_io_shape(tensor_info, has_batch_dim=has_batch_dim)
        assert result == expected

    def test_reformat_invalid_batch_dim(self):
        """Test reformatting with invalid batch dimension."""
        tensor_info = onnx.helper.make_tensor_value_info("test", onnx.TensorProto.FLOAT, [1])
        with pytest.raises(ValueError, match="Expected batch dimension"):
            _reformat_io_shape(tensor_info, has_batch_dim=True)


class TestGetInitializers:
    """Tests for get_initializers function."""

    @pytest.mark.parametrize(
        ("initializer_specs", "expected_names", "expected_count"),
        [
            pytest.param([], set(), 0, id="empty"),
            pytest.param(
                [("weight", (10, 3)), ("bias", (10,))],
                {"weight", "bias"},
                2,
                id="with_weights",
            ),
            pytest.param([("my_weights", (1, 3))], {"my_weights"}, 1, id="names"),
        ],
    )
    def test_get_initializers(self, initializer_specs, expected_names, expected_count):
        """Test getting initializers from model with various configurations."""
        input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])
        output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
        rng = np.random.default_rng(seed=42)
        init_tensors = []
        for name, shape in initializer_specs:
            data = rng.standard_normal(shape).astype(np.float32)
            init_tensors.append(onnx.numpy_helper.from_array(data, name=name))
        graph = onnx.helper.make_graph(
            [], "test_graph", [input_tensor], [output_tensor], init_tensors
        )
        model = onnx.helper.make_model(graph)

        result = get_initializers(model)
        assert set(result.keys()) == expected_names
        assert len(result) == expected_count


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

    @pytest.mark.parametrize(
        ("output_specs", "expected_names"),
        [
            pytest.param([("output", [1, 10])], {"output"}, id="single_output"),
            pytest.param(
                [("output1", [1, 10]), ("output2", [1, 5])],
                {"output1", "output2"},
                id="multiple_outputs",
            ),
        ],
    )
    def test_get_output_nodes(self, output_specs, expected_names):
        """Test getting output nodes with various output configurations."""
        input_tensor = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensors = [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
            for name, shape in output_specs
        ]
        graph = onnx.helper.make_graph([], "test_graph", [input_tensor], output_tensors)
        model = onnx.helper.make_model(graph)

        result = get_output_nodes(model, has_batch_dim=True)
        assert len(result) == len(output_specs)
        output_names = {node.name for node in result}
        assert output_names == expected_names


class TestConvertConstantToInitializer:
    """Tests for convert_constant_to_initializer function."""

    @pytest.mark.parametrize(
        ("const_specs", "op_type", "op_inputs", "op_outputs", "expected_init_keys"),
        [
            pytest.param([], "Relu", ["x"], ["y"], set(), id="no_constant_nodes"),
            pytest.param(
                [("const_output", [1.0] * 6, [2, 3])],
                "Relu",
                ["const_output"],
                ["y"],
                {"const_output"},
                id="single_constant",
            ),
            pytest.param(
                [("out1", [1.0, 2.0], [2]), ("out2", [3.0, 4.0], [2])],
                "Add",
                ["out1", "out2"],
                ["result"],
                {"out1", "out2"},
                id="multiple_constant_nodes",
            ),
        ],
    )
    def test_convert_constant_to_initializer(
        self, const_specs, op_type, op_inputs, op_outputs, expected_init_keys
    ):
        """Test converting Constant nodes to initializers."""
        nodes = []
        for name, values, shape in const_specs:
            tensor = onnx.helper.make_tensor(
                name.replace("out", "c"), onnx.TensorProto.FLOAT, shape, values
            )
            nodes.append(onnx.helper.make_node("Constant", inputs=[], outputs=[name], value=tensor))
        nodes.append(onnx.helper.make_node(op_type, inputs=op_inputs, outputs=op_outputs))

        initializers: dict[str, onnx.TensorProto] = {}
        result = convert_constant_to_initializer(nodes, initializers)

        assert len(result) == 1
        assert result[0].op_type == op_type
        assert set(initializers.keys()) == expected_init_keys
