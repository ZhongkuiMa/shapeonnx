__docformat__ = "restructuredtext"
__all__ = [
    "reformat_io_shape",
    "get_input_nodes",
    "get_output_nodes",
    "get_initializers",
    "convert_constant_to_initializer",
]

import onnx
from onnx import ModelProto, ValueInfoProto, TensorProto, NodeProto


def reformat_io_shape(node: ValueInfoProto, has_batch_dim: bool = True) -> list[int]:
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]
    if has_batch_dim:
        if len(shape) < 2:
            raise ValueError(
                f"There should have been a batch dimension. "
                f"Node {node.name} has invalid shape {shape}."
            )
        if shape[0] != 1:
            shape[0] = 1

    return shape


def get_input_nodes(
    model: ModelProto, initializers: dict[str, TensorProto], has_batch_dim: bool = True
) -> list[ValueInfoProto]:
    # Exclude initializers from inputs because sometimes the initializers are also
    # included in the inputs
    nodes = []
    for input_i in model.graph.input:
        if input_i.name not in initializers:
            shape = reformat_io_shape(input_i, has_batch_dim)
            node = onnx.helper.make_tensor_value_info(
                name=input_i.name,
                elem_type=input_i.type.tensor_type.elem_type,
                shape=shape,
            )
            nodes.append(node)

    return nodes


def get_output_nodes(
    model: ModelProto, has_batch_dim: bool = True
) -> list[ValueInfoProto]:
    nodes = []
    for output_i in model.graph.output:
        shape = reformat_io_shape(output_i, has_batch_dim)
        node = onnx.helper.make_tensor_value_info(
            name=output_i.name,
            elem_type=output_i.type.tensor_type.elem_type,
            shape=shape,
        )
        nodes.append(node)

    return nodes


def get_initializers(model: ModelProto) -> dict[str, TensorProto]:
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    return initializers


def convert_constant_to_initializer(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            np_array = onnx.numpy_helper.to_array(node.attribute[0].t)
            initializer = onnx.numpy_helper.from_array(np_array, node.output[0])
            initializers[node.output[0]] = initializer

            count += 1
            continue

        new_nodes.append(node)

    return new_nodes
