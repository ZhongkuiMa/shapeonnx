"""ONNX model utilities for shape inference."""

__docformat__ = "restructuredtext"
__all__ = [
    "convert_constant_to_initializer",
    "get_initializers",
    "get_input_nodes",
    "get_output_nodes",
]

import warnings

import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto


def _reformat_io_shape(node: ValueInfoProto, has_batch_dim: bool = True) -> list[int]:
    """
    Extract and reformat shape from ONNX value info node.

    :param node: ONNX value info node.

    :param has_batch_dim: Whether to normalize batch dimension.

    :return: Shape as list of integers
    """
    shape = []
    for d in node.type.tensor_type.shape.dim:
        if d.dim_param:
            warnings.warn(
                f"Dynamic dimension dim_param={d.dim_param!r} in node {node.name}; "
                f"using -1 as placeholder",
                stacklevel=2,
            )
            shape.append(-1)
        elif d.dim_value == 0:
            warnings.warn(
                f"Zero-valued dimension in node {node.name}; "
                f"using -1 as placeholder for dynamic dim",
                stacklevel=2,
            )
            shape.append(-1)
        else:
            shape.append(d.dim_value)
    if has_batch_dim:
        # Allow scalar [] and rank-1 [N] outputs - they carry no separate feature
        # axis to normalize (e.g. outputs reduced via Squeeze / keepdims=0
        # reductions down to the surviving batch dim); accept them as-is.
        if len(shape) <= 1:
            return shape
        if shape[0] != 1:
            shape[0] = 1

    return shape


def get_input_nodes(
    model: ModelProto, initializers: dict[str, TensorProto], has_batch_dim: bool = True
) -> list[ValueInfoProto]:
    """
    Get model input nodes excluding initializers.

    :param model: ONNX model.

    :param initializers: Model initializers dictionary.

    :param has_batch_dim: Whether to normalize batch dimension.

    :return: List of input value info nodes
    """
    nodes = []
    for input_i in model.graph.input:
        if input_i.name not in initializers:
            shape = _reformat_io_shape(input_i, has_batch_dim)
            node = onnx.helper.make_tensor_value_info(
                name=input_i.name,
                elem_type=input_i.type.tensor_type.elem_type,
                shape=shape,
            )
            nodes.append(node)

    return nodes


def get_output_nodes(model: ModelProto, has_batch_dim: bool = True) -> list[ValueInfoProto]:
    """
    Get model output nodes.

    :param model: ONNX model.

    :param has_batch_dim: Whether to normalize batch dimension.

    :return: List of output value info nodes
    """
    nodes = []
    for output_i in model.graph.output:
        shape = _reformat_io_shape(output_i, has_batch_dim)
        node = onnx.helper.make_tensor_value_info(
            name=output_i.name,
            elem_type=output_i.type.tensor_type.elem_type,
            shape=shape,
        )
        nodes.append(node)

    return nodes


def get_initializers(model: ModelProto) -> dict[str, TensorProto]:
    """
    Extract initializers from ONNX model.

    :param model: ONNX model.

    :return: Dictionary mapping initializer names to TensorProto
    """
    return {initializer.name: initializer for initializer in model.graph.initializer}


def convert_constant_to_initializer(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    """
    Convert Constant nodes to initializers.

    :param nodes: List of ONNX nodes.

    :param initializers: Initializers dictionary to update.

    :return: List of nodes with Constant nodes removed
    """
    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            if value_attr is None:
                raise ValueError(f"Constant node {node.name} has no 'value' attribute")
            np_array = onnx.numpy_helper.to_array(value_attr.t)
            initializer = onnx.numpy_helper.from_array(np_array, node.output[0])
            initializers[node.output[0]] = initializer
            continue

        new_nodes.append(node)

    return new_nodes
