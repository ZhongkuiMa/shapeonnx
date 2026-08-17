"""ONNX shape inference engine."""

__docformat__ = "restructuredtext"
__all__ = ["extract_io_shapes", "infer_onnx_shape"]

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto

from shapeonnx.onnx_attrs import _get_onnx_attrs
from shapeonnx.utils import _reformat_io_shape

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapeInferenceContext:
    """
    Immutable context for shape inference.

    :param data_shapes: Maps tensor names to their inferred shapes.

    :param explicit_shapes: Maps tensor names to constant shape values.

    :param initializers: ONNX model initializers.

    :param verbose: Whether to print debug information.

    """

    data_shapes: dict[str, int | list[int]]
    explicit_shapes: dict[str, int | list[int]]
    initializers: dict[str, TensorProto]
    verbose: bool = False


def extract_io_shapes(nodes: list[ValueInfoProto], has_batch_dim: bool) -> dict[str, list[int]]:
    """
    Extract shapes from model input/output nodes.

    :param nodes: List of ONNX value info nodes.

    :param has_batch_dim: Whether nodes have a batch dimension.

    :return: Dictionary mapping node names to shapes
    """
    return {node.name: _reformat_io_shape(node, has_batch_dim) for node in nodes}


def _extract_initializer_shapes(
    initializers: dict[str, TensorProto],
) -> dict[str, list[int]]:
    """
    Extract shapes from model initializers.

    :param initializers: Dictionary of ONNX initializers.

    :return: Dictionary mapping initializer names to shapes
    """
    return {name: list(map(int, init.dims)) for name, init in initializers.items()}


def _get_data_shape(name: str, shapes: dict[str, int | list[int]]) -> int | list[int] | None:
    """
    Retrieve data shape by name.

    :param name: Tensor name.

    :param shapes: Shape dictionary.

    :return: Shape if found, None otherwise
    """
    return shapes.get(name)


def _extract_integer_initializers(
    initializers: dict[str, TensorProto],
) -> dict[str, int | list[int]]:
    """
    Pre-convert integer-type initializers to Python int/list for shape operations.

    :param initializers: ONNX initializers.

    :return: Dictionary mapping initializer names to converted values
    """
    converted = {}
    integer_types = (
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    )
    for name, initializer in initializers.items():
        if initializer.data_type in integer_types:
            converted[name] = onnx.numpy_helper.to_array(initializer).tolist()
    return converted


def _get_explicit_shape(
    name: str,
    explicit_shapes: dict[str, int | list[int]],
) -> int | list[int] | None:
    """
    Retrieve explicit constant shape value.

    :param name: Tensor name.

    :param explicit_shapes: Explicit shape dictionary.

    :return: Constant value if found, None otherwise
    """
    return explicit_shapes.get(name)


def _get_shape(
    name: str,
    shapes: dict[str, int | list[int]],
    explicit_shapes: dict[str, int | list[int]],
) -> tuple[int | list[int] | None, bool]:
    """
    Retrieve shape from any available source.

    :param name: Tensor name.

    :param shapes: Data shape dictionary.

    :param explicit_shapes: Explicit shape dictionary.

    :return: Tuple of (shape, is_explicit)
    """
    if (shape := shapes.get(name)) is not None:
        return shape, False
    if (explicit_shape := explicit_shapes.get(name)) is not None:
        return explicit_shape, True
    raise RuntimeError(f"Cannot get shape of {name}")


def _store_data_shape(shape: list[int], shapes: dict[str, list[int]], name: str) -> None:
    """
    Store inferred data shape.

    :param shape: Inferred shape.

    :param shapes: Shape dictionary to update.

    :param name: Tensor name.

    """
    shapes[name] = shape


def _store_explicit_shape(
    shape: int | list[int], explicit_shapes: dict[str, int | list[int]], name: str
) -> None:
    """
    Store constant shape value.

    :param shape: Constant shape value.

    :param explicit_shapes: Explicit shape dictionary to update.

    :param name: Tensor name.

    """
    explicit_shapes[name] = shape


def _align_shapes(base: list[int], target: list[int]) -> list[int]:
    """
    Align target shape to base shape structure.

    :param base: Base shape.

    :param target: Target shape to align.

    :return: Aligned shape
    """
    aligned = [1] * max(len(base), len(target))
    j = 0
    for i in range(len(base)):
        if j < len(target) and base[i] == target[j]:
            aligned[i] = target[j]
            j += 1
            if j >= len(target):
                break
    return aligned


def _right_align_shapes(shape1: list[int], shape2: list[int]) -> tuple[list[int], list[int]]:
    """
    Right-align two shapes by padding with 1s.

    :param shape1: First shape.

    :param shape2: Second shape.

    :return: Tuple of right-aligned shapes
    """
    max_len = max(len(shape1), len(shape2))
    aligned1 = [1] * (max_len - len(shape1)) + shape1
    aligned2 = [1] * (max_len - len(shape2)) + shape2
    return aligned1, aligned2


def _compute_broadcasted_shape(shape1: list[int], shape2: list[int]) -> list[int]:
    """
    Compute broadcasted shape from two aligned shapes.

    :param shape1: First shape.

    :param shape2: Second shape.

    :return: Broadcasted shape
    """
    result = []
    for s1, s2 in zip(shape1, shape2, strict=False):
        if s1 != s2 and s1 != 1 and s2 != 1:
            raise RuntimeError(f"Cannot broadcast {shape1} and {shape2}")
        result.append(max(s1, s2))
    return result


def _broadcast_shapes(shape1: list[int], shape2: list[int]) -> list[int]:
    """
    Broadcast two shapes using numpy broadcasting rules.

    :param shape1: First shape.

    :param shape2: Second shape.

    :return: Broadcasted shape
    """
    if [0] in (shape1, shape2):
        return [0]
    if not shape1:
        return shape2
    if not shape2:
        return shape1

    aligned1, aligned2 = _right_align_shapes(shape1, shape2)
    return _compute_broadcasted_shape(aligned1, aligned2)


def _topological_sort(nodes: list[NodeProto]) -> list[NodeProto]:
    """
    Topologically sort ONNX nodes using Kahn's algorithm.

    :param nodes: Unsorted list of ONNX nodes.
    :return: Nodes in topological order.
    :raises RuntimeError: If the graph contains a cycle.
    """
    output_to_idx: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for output in node.output:
            output_to_idx[output] = i

    children: list[list[int]] = [[] for _ in nodes]
    in_degree = [0] * len(nodes)

    for i, node in enumerate(nodes):
        for inp in node.input:
            if inp in output_to_idx:
                parent = output_to_idx[inp]
                children[parent].append(i)
                in_degree[i] += 1

    queue = [i for i, deg in enumerate(in_degree) if deg == 0]
    result: list[NodeProto] = []
    while queue:
        idx = queue.pop(0)
        result.append(nodes[idx])
        for child in children[idx]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(nodes):
        raise RuntimeError("Graph contains a cycle; cannot topologically sort nodes")
    return result


def _infer_identity_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for operators that preserve input shape.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def _compute_binary_op_value(op_type: str, value1: int | float, value2: int | float) -> int | float:
    """
    Compute binary operation on scalar values.

    :param op_type: Operation type.

    :param value1: First operand.

    :param value2: Second operand.

    :return: Operation result
    """
    operations = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
    }
    if op_type not in operations:
        raise RuntimeError(f"Cannot calculate {op_type} with values {value1} and {value2}")
    result = operations[op_type](value1, value2)
    return int(result) if isinstance(value1, int) and isinstance(value2, int) else result


def _compute_explicit_binary_shape(
    op_type: str, e_shape1: int | list[int], e_shape2: int | list[int]
) -> list[int]:
    """
    Compute explicit shape for binary operations.

    :param op_type: Operation type.

    :param e_shape1: First explicit shape.

    :param e_shape2: Second explicit shape.

    :return: Computed explicit shape
    """
    if op_type == "Mul":
        if isinstance(e_shape1, int) and isinstance(e_shape2, list):
            return [e_shape1 * s for s in e_shape2]
        if isinstance(e_shape2, int) and isinstance(e_shape1, list):
            return [e_shape2 * s for s in e_shape1]
        raise NotImplementedError(f"Cannot calculate explicit shape of {e_shape1} and {e_shape2}")
    # Equal and other comparison ops output boolean tensors (0 or 1)
    # The explicit shape cannot be computed from input explicit shapes
    # because the output values are not derived from input values
    # Instead, return None to indicate no explicit shape available
    raise NotImplementedError(
        f"Cannot calculate explicit shape of {op_type} with {e_shape1} and {e_shape2}"
    )


def _compute_equal_explicit_shape(
    input1: str, input2: str, explicit_shapes: dict[str, int | list[int]]
) -> list[int] | None:
    """
    Compute explicit shape for Equal operator.

    :param input1: First input name.

    :param input2: Second input name.

    :param explicit_shapes: Dictionary of explicit shapes.

    :return: Explicit shape for Equal result, or None
    """
    e_shape1 = _get_explicit_shape(input1, explicit_shapes)
    e_shape2 = _get_explicit_shape(input2, explicit_shapes)
    if (
        e_shape1 is not None
        and isinstance(e_shape1, list)
        and e_shape2 is not None
        and isinstance(e_shape2, list)
        and len(e_shape1) == len(e_shape2)
    ):
        # Equal compares element-wise: output is 1 where equal, 0 where different
        return [1 if e_shape1[i] == e_shape2[i] else 0 for i in range(len(e_shape1))]
    return None


def _compute_binary_explicit_shape(
    op_type: str, input1: str, input2: str, explicit_shapes: dict[str, int | list[int]]
) -> int | list[int] | None:
    """
    Compute explicit shape for binary operators.

    :param op_type: Operator type.

    :param input1: First input name.

    :param input2: Second input name.

    :param explicit_shapes: Dictionary of explicit shapes.

    :return: Explicit shape result, or None
    """
    e_shape1 = _get_explicit_shape(input1, explicit_shapes)
    if e_shape1 is not None and isinstance(e_shape1, (int, list)):
        e_shape2 = _get_explicit_shape(input2, explicit_shapes)
        if e_shape2 is not None and isinstance(e_shape2, (int, list)):
            try:
                return _compute_explicit_binary_shape(op_type, e_shape1, e_shape2)
            except NotImplementedError:
                return None
    return None


def _infer_binary_op_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for binary operators.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e1 = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, is_e2 = _get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    is_explicit = is_e1 or is_e2

    is_list1 = isinstance(shape1, list)
    is_list2 = isinstance(shape2, list)
    is_int1 = isinstance(shape1, int)
    is_int2 = isinstance(shape2, int)

    # For Equal and other comparison ops, [0] should not be treated specially
    # as the output shape is determined by broadcasting, not by the values
    skip_zero_check = node.op_type in ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"]

    shape: int | list[int]
    if not skip_zero_check and (shape1 == [0] or shape2 == [0]):
        shape = [0]
    elif is_list1 and is_list2 and not shape1 and not shape2:
        shape = []
    elif (is_int1 or (is_list1 and not shape1)) and (is_int2 or (is_list2 and not shape2)):
        val1 = shape1 if is_int1 else _get_explicit_shape(node.input[0], ctx.explicit_shapes)
        val2 = shape2 if is_int2 else _get_explicit_shape(node.input[1], ctx.explicit_shapes)
        if val1 is None or val2 is None or isinstance(val1, list) or isinstance(val2, list):
            shape = [0]
        else:
            shape = int(_compute_binary_op_value(node.op_type, val1, val2))
    elif is_list1 and is_list2:
        assert isinstance(shape1, list)
        assert isinstance(shape2, list)
        shape = _broadcast_shapes(shape1, shape2)
    else:
        raise RuntimeError(f"Cannot calculate {node.op_type} with shape {shape1} and {shape2}")

    if is_explicit:
        return [(None, shape)]

    # Special handling for Equal operator
    if node.op_type == "Equal":
        equal_explicit_shape = _compute_equal_explicit_shape(
            node.input[0], node.input[1], ctx.explicit_shapes
        )
        return [(shape, equal_explicit_shape)]

    # For other operations, try to compute explicit shape
    binary_explicit_shape = _compute_binary_explicit_shape(
        node.op_type, node.input[0], node.input[1], ctx.explicit_shapes
    )
    return [(shape, binary_explicit_shape)]


def _infer_argmax_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ArgMax operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    axis, keepdims = attrs["axis"], attrs["keepdims"]

    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")

    if isinstance(shape, int):
        return [(shape, None)]

    if shape != [0]:
        shape_copy = list(shape)
        shape_copy[axis] = 1
        if not keepdims:
            shape_copy.pop(axis)
        shape = shape_copy

    return [(shape, None)]


def _infer_batch_norm_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """Infer shape for BatchNormalization operator (preserves input shape)."""
    return _infer_identity_shape(node, ctx)


def _collect_concat_input_shapes(
    input_names: Sequence[str], ctx: ShapeInferenceContext
) -> tuple[list[list[int]], bool, bool] | tuple[list[int], None]:
    """
    Collect shapes from all concat inputs.

    :param input_names: List of input tensor names.

    :param ctx: Shape inference context.

    :return: Either (shape_list, all_explicit, any_explicit) or ([0], None) for early return
    """
    shape_list = []
    all_explicit = True
    any_explicit = False

    for name in input_names:
        # Try explicit shape first
        if name in ctx.explicit_shapes:
            shape_i = _get_explicit_shape(name, ctx.explicit_shapes)
            if shape_i is not None:
                any_explicit = True
                if not isinstance(shape_i, list):
                    raise RuntimeError(f"Cannot concatenate scalar shape from {name}")
                shape_list.append(shape_i)
                if shape_i == [0]:
                    return ([0], None)
                continue

        # Fallback to data shape
        all_explicit = False
        shape_i, _ = _get_shape(name, ctx.data_shapes, ctx.explicit_shapes)
        if shape_i is None:
            raise RuntimeError(f"Cannot infer shape for Concat input {name}")
        if not isinstance(shape_i, list):
            raise RuntimeError(f"Cannot concatenate scalar shape from {name}")
        if shape_i == [0]:
            return ([0], None)
        shape_list.append(shape_i)

    return (shape_list, all_explicit, any_explicit)


def _infer_concat_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Concat operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    axis = attrs["axis"]

    result = _collect_concat_input_shapes(node.input, ctx)
    if result[1] is None:
        # Early return with [0]
        return [(result[0], None)]

    shape_list, all_explicit, any_explicit = result

    if all_explicit:
        shape = np.concatenate(shape_list, axis=axis).tolist()
        return [(None, shape)]

    rank = len(shape_list[0])
    if rank == 0:
        raise RuntimeError("Cannot concatenate scalar inputs")
    if any(len(input_shape) != rank for input_shape in shape_list[1:]):
        raise RuntimeError(
            f"Concat inputs must have the same rank, got {[len(s) for s in shape_list]}"
        )
    if axis < -rank or axis >= rank:
        raise RuntimeError(f"Concat axis {axis} is out of range for input rank {rank}")
    axis %= rank

    shape = shape_list[0].copy()
    for input_index, other_shape in enumerate(shape_list[1:], start=1):
        for dim, (expected, actual) in enumerate(zip(shape, other_shape, strict=True)):
            if dim != axis and actual != expected:
                raise RuntimeError(
                    f"Concat input {input_index} shape {other_shape} has dimension "
                    f"{dim}={actual}, expected {expected}"
                )
        shape[axis] += other_shape[axis]

    # Only mark as explicit if we had any explicit inputs and result is concrete
    if any_explicit:
        is_concrete = all(isinstance(d, int) and d >= 0 for d in shape)
        if is_concrete:
            return [(None, shape)]

    return [(shape, None)]


def _infer_constant_of_shape_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ConstantOfShape operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}")

    if shape != [0]:
        value = _get_onnx_attrs(node, ctx.initializers)["value"]
        if np.issubdtype(value.dtype, np.integer):
            constant = np.full(shape, value, dtype=value.dtype).tolist()
            return [(shape, constant)]

    return [(shape, None)]


def _compute_convtranspose_output_hw(
    input_shape: list[int],
    weight_shape: list[int],
    kernel_shape: list[int],
    dilations: list[int],
    output_padding: list[int],
    pads: list[int],
    strides: list[int],
) -> list[int]:
    """
    Compute output height/width for ConvTranspose.

    :param input_shape: Input tensor shape.

    :param weight_shape: Weight tensor shape.

    :param kernel_shape: Kernel dimensions.

    :param dilations: Dilation factors.

    :param output_padding: Output padding.

    :param pads: Input padding.

    :param strides: Stride values.

    :return: Output height/width
    """
    dim = len(kernel_shape)
    temp1 = [pads[i] + pads[i + dim] for i in range(dim)]
    temp2 = [dilations[i] * (kernel_shape[i] - 1) for i in range(dim)]
    output_hw = [
        math.ceil(
            (input_shape[i + 2] - 1) * strides[i] - temp1[i] + temp2[i] + output_padding[i] + 1
        )
        for i in range(dim)
    ]
    return output_hw


def _infer_convtranspose_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ConvTranspose operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    output_padding = attrs["output_padding"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    group = attrs["group"]

    if not (
        len(kernel_shape) == len(dilations) == len(output_padding) == len(strides) == 2
        and len(pads) == 4
    ):
        raise NotImplementedError(
            f"ConvTranspose with kernel_shape={kernel_shape}, dilations={dilations}, "
            f"output_padding={output_padding}, pads={pads}, strides={strides} is not supported"
        )
    if (
        any(value <= 0 for value in (*kernel_shape, *dilations, *strides))
        or any(value < 0 for value in (*output_padding, *pads))
        or any(
            extra >= max(step, dilation)
            for extra, step, dilation in zip(output_padding, strides, dilations, strict=True)
        )
    ):
        raise ValueError("ConvTranspose attributes are outside the executable 2D domain")

    input_shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")
    if input_shape == [0]:
        return [([0], None)]

    # ConvTranspose requires list shape
    if isinstance(input_shape, int):
        raise RuntimeError(f"ConvTranspose input shape cannot be scalar: {input_shape}")

    weight_shape = list(ctx.initializers[node.input[1]].dims)
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise NotImplementedError(
            f"ConvTranspose supports NCHW 2D tensors only, got {input_shape} and {weight_shape}"
        )
    if input_shape[1] != weight_shape[0] or weight_shape[0] % group:
        raise ValueError(
            f"ConvTranspose input/group geometry is incoherent: input={input_shape}, "
            f"weight={weight_shape}, group={group}"
        )
    out_channels = weight_shape[1] * group
    if len(node.input) > 2 and node.input[2]:
        bias_shape = tuple(ctx.initializers[node.input[2]].dims)
        if bias_shape != (out_channels,):
            raise ValueError(
                f"ConvTranspose bias shape {bias_shape} != output channels {(out_channels,)}"
            )

    if attrs["output_shape"] is not None:
        output_hw = list(attrs["output_shape"])
        if len(output_hw) != 2 or any(size <= 0 for size in output_hw):
            raise ValueError(
                f"ConvTranspose output_shape must contain two positive integers, got {output_hw}"
            )
    else:
        output_hw = _compute_convtranspose_output_hw(
            input_shape,
            weight_shape,
            kernel_shape,
            dilations,
            output_padding,
            pads,
            strides,
        )
    shape = [input_shape[0], out_channels, *output_hw]
    return [(shape, None)]


def _infer_expand_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Expand operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e1 = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2 = _get_explicit_shape(node.input[1], ctx.explicit_shapes)
    if shape2 is None:
        shape2 = _get_data_shape(node.input[1], ctx.data_shapes)
        if shape2 is None:
            raise RuntimeError(f"Cannot get shape of {node.input[1]}")

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        raise RuntimeError(f"Cannot expand with shapes {shape1} and {shape2}")

    shape = _broadcast_shapes(shape1, shape2)

    if is_e1:
        return [(None, shape)]
    return [(shape, None)]


def _infer_flatten_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Flatten operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")

    axis = _get_onnx_attrs(node, ctx.initializers)["axis"]
    dims = [] if isinstance(shape, int) else shape
    rank = len(dims)
    if not isinstance(axis, int) or axis < -rank or axis > rank:
        raise ValueError(f"Flatten axis {axis!r} is outside [-{rank}, {rank}]")
    canonical_axis = axis + rank if axis < 0 else axis

    def _product(extents: list[int]) -> int:
        return -1 if any(extent < 0 for extent in extents) else math.prod(extents)

    output_shape = [_product(dims[:canonical_axis]), _product(dims[canonical_axis:])]
    return [(output_shape, None)]


def _infer_gather_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Gather operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    axis = _get_onnx_attrs(node, ctx.initializers)["axis"]
    indices = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
    is_int_indices = isinstance(indices, int)

    # Check for explicit shape first (for shape tensors like from Shape op)
    e_shape = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is not None:
        # Gathering from a shape tensor (explicit shape)
        if e_shape != [0]:
            if axis != 0:
                raise ValueError(f"Invalid axis {axis} for gather from explicit shape")
            if not isinstance(e_shape, list):
                raise RuntimeError(f"Cannot gather from non-list explicit shape {e_shape}")
            if is_int_indices:
                e_shape = e_shape[indices]
            else:
                e_shape = [
                    e_shape[i] for i in indices if isinstance(indices, list) and i < len(e_shape)
                ]
        return [(None, e_shape)]

    # Fallback to data shape (for regular data tensors)
    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is not None:
        # Handle scalar shapes
        if isinstance(shape, int):
            return [(shape, None)]

        # ONNX Gather allows a negative axis (counts from the end); normalize
        # before comparing against positional indices, else a negative axis
        # matches nothing and the gathered dim is neither replaced nor removed.
        norm_axis = axis % len(shape) if shape else axis
        if shape != [0]:
            shape = [
                len(indices) if i == norm_axis and not is_int_indices else shape[i]
                for i in range(len(shape))
                if not (i == norm_axis and is_int_indices)
            ]
        return [(shape, None)]

    raise RuntimeError(f"Cannot get shape of {node.input[0]}")


def _infer_gemm_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Gemm operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    trans_a, trans_b = attrs["transA"], attrs["transB"]

    shape1, _ = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = _get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)

    if [0] in (shape1, shape2):
        return [([0], None)]

    if (
        not isinstance(shape1, list)
        or not isinstance(shape2, list)
        or len(shape1) != 2
        or len(shape2) != 2
    ):
        raise RuntimeError(
            f"Gemm requires rank-2 A and B tensors, got shapes {shape1} and {shape2}"
        )

    shape1 = shape1.copy()
    shape2 = shape2.copy()
    if trans_a:
        shape1[-2], shape1[-1] = shape1[-1], shape1[-2]
    if trans_b:
        shape2[-2], shape2[-1] = shape2[-1], shape2[-2]

    if shape1[1] != shape2[0]:
        raise RuntimeError(
            f"Gemm inner dimensions must match after transpose, got {shape1} and {shape2}"
        )

    output_shape = [shape1[0], shape2[1]]
    if len(node.input) > 2 and node.input[2]:
        bias_shape, _ = _get_shape(node.input[2], ctx.data_shapes, ctx.explicit_shapes)
        if bias_shape != [0]:
            if isinstance(bias_shape, int):
                bias_shape = []
            if not isinstance(bias_shape, list) or len(bias_shape) > 2:
                raise RuntimeError(f"Gemm bias must broadcast to {output_shape}, got {bias_shape}")
            padded_bias = [1] * (2 - len(bias_shape)) + bias_shape
            if any(b not in (1, out) for b, out in zip(padded_bias, output_shape, strict=True)):
                raise RuntimeError(f"Gemm bias must broadcast to {output_shape}, got {bias_shape}")

    return [(output_shape, None)]


def _broadcast_matmul_batch_shape(left: list[int], right: list[int]) -> list[int]:
    """Broadcast two MatMul batch prefixes using ONNX multidirectional rules."""
    out: list[int] = []
    for ldim, rdim in zip(reversed(left), reversed(right), strict=False):
        if ldim == rdim or rdim == 1:
            out.append(ldim)
        elif ldim == 1:
            out.append(rdim)
        else:
            raise ValueError(f"MatMul batch dimensions cannot broadcast: {left} and {right}")
    longer = left if len(left) > len(right) else right
    out.extend(reversed(longer[: abs(len(left) - len(right))]))
    return list(reversed(out))


def _matmul_output_shape(left: list[int], right: list[int]) -> list[int]:
    """Return exact ONNX MatMul output geometry for two non-scalar shapes."""
    if not left or not right:
        raise ValueError(f"MatMul requires rank >= 1 inputs, got {left} and {right}")

    left_was_vector = len(left) == 1
    right_was_vector = len(right) == 1
    promoted_left = [1, *left] if left_was_vector else left
    promoted_right = [*right, 1] if right_was_vector else right

    if promoted_left[-1] != promoted_right[-2]:
        raise ValueError(f"MatMul contraction dimensions must match, got {left} and {right}")

    batch = _broadcast_matmul_batch_shape(promoted_left[:-2], promoted_right[:-2])
    out = [*batch, promoted_left[-2], promoted_right[-1]]
    if left_was_vector:
        out.pop(len(batch))
    if right_was_vector:
        out.pop()
    return out


def _infer_matmul_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for MatMul operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, _ = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = _get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    if [0] in (shape1, shape2):
        return [([0], None)]

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        raise ValueError(f"MatMul requires tensor inputs, got {shape1} and {shape2}")

    return [(_matmul_output_shape(shape1, shape2), None)]


def _compute_pool_output_hw(
    input_shape: list[int],
    kernel_shape: list[int],
    dilations: list[int],
    pads: list[int],
    strides: list[int],
    ceil_mode: bool,
) -> list[int]:
    """
    Compute output height/width for pooling operations.

    :param input_shape: Input tensor shape.

    :param kernel_shape: Kernel dimensions.

    :param dilations: Dilation factors.

    :param pads: Padding values.

    :param strides: Stride values.

    :param ceil_mode: Whether to use ceiling for output size.

    :return: Output height/width
    """
    dim = len(kernel_shape)
    output_hw = []
    for i in range(dim):
        temp1 = pads[i] + pads[i + dim]
        temp2 = dilations[i] * (kernel_shape[i] - 1)
        size = (input_shape[i + 2] + temp1 - temp2 - 1) / strides[i] + 1
        output_hw.append(math.ceil(size) if ceil_mode else math.floor(size))
    return output_hw


def _infer_pool_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for pooling operators.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    ceil_mode = attrs.get("ceil_mode", False)

    dim = len(kernel_shape)
    if not (len(dilations) == dim and len(pads) == dim * 2 and len(strides) == dim):
        raise ValueError(
            f"Inconsistent dimensions: kernel={kernel_shape}, dilations={dilations}, "
            f"pads={pads}, strides={strides}"
        )

    input_shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")
    if input_shape == [0]:
        return [([0], None)]

    # MaxPool requires list shape
    if isinstance(input_shape, int):
        raise RuntimeError(f"MaxPool input shape cannot be scalar: {input_shape}")

    if len(node.input) > 1:
        weight_shape, _ = _get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
        if not isinstance(weight_shape, list):
            raise RuntimeError(f"Weight shape must be a list, got {weight_shape}")
        output_channel = weight_shape[0]
    else:
        output_channel = input_shape[1]

    output_hw = _compute_pool_output_hw(
        input_shape, kernel_shape, dilations, pads, strides, ceil_mode
    )
    shape = [input_shape[0], output_channel, *output_hw]
    return [(shape, None)]


def _infer_pad_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Pad operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    input_shape, is_explicit = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    if input_shape == [0]:
        if is_explicit:
            return [(None, [0])]
        return [([0], None)]

    if not isinstance(input_shape, list):
        raise RuntimeError(f"Input shape must be a list, got {input_shape}")

    # pads source: input[1] (opset >= 11) or the 'pads' attribute (opset < 11).
    if len(node.input) == 1:
        pads_attr = _get_onnx_attrs(node, ctx.initializers)["pads"]
        if pads_attr is None:
            raise RuntimeError(f"Pad node {node.name} has no pads input or attribute")
        pads = list(pads_attr)
    else:
        pads = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
        if len(node.input) == 4:
            axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[3]]).tolist()
            raise NotImplementedError(f"Pad with axes={axes} is not supported")

    dim = len(pads) // 2
    combined_pads = [pads[i] + pads[i + dim] for i in range(dim)]
    shape = [s + p for s, p in zip(input_shape, combined_pads, strict=False)]

    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def _infer_range_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Range operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    start = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    limit = _get_explicit_shape(node.input[1], ctx.explicit_shapes)
    delta = _get_explicit_shape(node.input[2], ctx.explicit_shapes)

    if not (isinstance(start, int) and isinstance(limit, int) and isinstance(delta, int)):
        return [([0], None)]

    if delta > 0:
        length = max(0, (limit - start + delta - 1) // delta)
    elif delta < 0:
        length = max(0, (start - limit - delta - 1) // (-delta))
    else:
        raise ValueError("Range step delta cannot be 0")

    return [([length], None)]


def _infer_reduce_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for reduction operators.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    keepdims = attrs["keepdims"]

    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")

    # Axes source: input[1] (opset >= 13) or the 'axes' attribute (opset < 13);
    # absent in both means reduce over all axes.
    if len(node.input) > 1 and node.input[1] in ctx.initializers:
        axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
    elif attrs["axes"] is not None:
        axes = list(attrs["axes"])
    elif isinstance(shape, list):
        axes = list(range(len(shape)))
    else:
        axes = [0]

    # Handle scalar shapes
    if isinstance(shape, int):
        return [(shape, None)]

    # Copy to avoid mutating the original shape in data_shapes
    shape = shape.copy()
    if shape != [0]:
        if keepdims:
            for axis in axes:
                shape[axis] = 1
        else:
            for axis in sorted(axes, reverse=True):
                if axis < len(shape):
                    shape.pop(axis)

    return [(shape, None)]


def _infer_reshape_output_shape(ori_shape: list[int], new_shape: list[int]) -> list[int]:
    """
    Infer reshaped output shape without actual computation.

    :param ori_shape: Original shape.

    :param new_shape: Target shape with possible -1.

    :return: Inferred output shape
    """
    total = math.prod(ori_shape)
    inferred_idx = -1
    remaining = total
    result = new_shape.copy()

    for idx, dim in enumerate(new_shape):
        if dim == -1:
            inferred_idx = idx
        elif dim == 0:
            # ONNX Reshape: a 0 target dim copies the corresponding input dim
            # (allowzero=0, the default). Substitute the input dim and divide by it.
            copied = ori_shape[idx]
            result[idx] = copied
            remaining //= copied
        else:
            remaining //= dim

    if inferred_idx != -1:
        result[inferred_idx] = remaining

    return result


def _infer_reshape_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Reshape operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    data_shape, _ = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    target_shape = _get_explicit_shape(node.input[1], ctx.explicit_shapes)

    if not isinstance(data_shape, list) or not isinstance(target_shape, list):
        return [([0], None)]

    if target_shape == [0] or (data_shape == [0] and -1 in target_shape):
        return [([0], None)]

    shape = _infer_reshape_output_shape(data_shape, target_shape)
    return [(shape, None)]


def _infer_resize_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Resize operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    align_mode = attrs["coordinate_transformation_mode"]
    mode = attrs["mode"]
    nearest_mode = attrs.get("nearest_mode", "floor")

    if mode != "nearest":
        raise NotImplementedError(f"Resize mode={mode} is not supported")

    input_shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")
    if input_shape == [0]:
        return [([0], None)]

    # Resize requires list shape
    if isinstance(input_shape, int):
        raise RuntimeError(f"Resize input shape cannot be scalar: {input_shape}")

    if align_mode not in {"asymmetric", "half_pixel"}:
        raise NotImplementedError(f"Resize align_mode={align_mode} is not supported")
    if nearest_mode not in {"floor", "ceil", "round_prefer_floor", "round_prefer_ceil"}:
        raise NotImplementedError(f"Resize nearest_mode={nearest_mode} is not supported")

    scales_name = node.input[2] if len(node.input) > 2 else ""
    sizes_name = node.input[3] if len(node.input) > 3 else ""
    if bool(scales_name) == bool(sizes_name):
        raise ValueError("Resize requires exactly one of scales or sizes")

    if sizes_name:
        if sizes_name not in ctx.initializers:
            raise ValueError("Resize with dynamic sizes is not supported")
        sizes = onnx.numpy_helper.to_array(ctx.initializers[sizes_name]).tolist()
        if (
            not isinstance(sizes, list)
            or len(sizes) != len(input_shape)
            or any(not isinstance(size, int) or size <= 0 for size in sizes)
        ):
            raise ValueError(f"Resize sizes must be positive full-rank integers, got {sizes!r}")
        return [(sizes, None)]

    if scales_name not in ctx.initializers:
        raise ValueError("Resize with dynamic scales is not supported")
    scales = onnx.numpy_helper.to_array(ctx.initializers[scales_name]).tolist()
    if not scales:
        raise ValueError("Resize with empty scales is not supported")
    if len(scales) != len(input_shape):
        raise ValueError(f"Resize scales rank {len(scales)} != input rank {len(input_shape)}")

    # ONNX output extent is floor(input * scale); nearest_mode controls only
    # source-index selection and must never change tensor geometry.
    shape = [math.floor(dim * scale) for dim, scale in zip(input_shape, scales, strict=True)]
    return [(shape, None)]


def _infer_shape_op_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Shape operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)

    if not is_explicit:
        if not isinstance(shape, list):
            raise RuntimeError(f"Expected list shape, got {shape}")
        return [(None, shape)]

    if isinstance(shape, int):
        result_shape = []
    elif isinstance(shape, list) and shape == [0]:
        result_shape = [0]
    elif isinstance(shape, list):
        # Return the actual shape values, not [1, len(shape)]
        result_shape = shape
    else:
        raise RuntimeError(f"Unexpected explicit shape type {type(shape)}")

    return [(None, result_shape)]


def _infer_sliced_shape(
    shape: list[int],
    axes: list[int],
    starts: list[int],
    ends: list[int],
    steps: list[int],
) -> list[int]:
    """
    Infer shape after slicing operation.

    :param shape: Original shape.

    :param axes: Axes to slice.

    :param starts: Start indices.

    :param ends: End indices.

    :param steps: Step sizes.

    :return: Sliced shape
    """
    new_shape = list(shape)
    for axis, start, end, step in zip(axes, starts, ends, steps, strict=True):
        size = shape[axis]
        start = min(max(start + size if start < 0 else start, 0), size)
        end = min(max(end + size if end < 0 else end, 0), size)
        if step < 0:
            warnings.warn(f"Negative step ({step}) is not fully tested", stacklevel=2)
        new_shape[axis] = max(0, (end - start + (step - (1 if step > 0 else -1))) // step)
    return new_shape


def _infer_slice_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Slice operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    if len(node.input) == 1:
        # opset < 10: starts/ends/axes carried as node attributes (no steps).
        attrs = _get_onnx_attrs(node, ctx.initializers)
        if attrs["starts"] is None or attrs["ends"] is None:
            # No params at all: slice is a no-op, pass the input shape through.
            shape = _get_data_shape(node.input[0], ctx.data_shapes)
            e_shape = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
            if e_shape is not None:
                return [(None, e_shape)]
            if shape is not None:
                return [(shape, None)]
            return [([0], None)]
        starts = list(attrs["starts"])
        ends = list(attrs["ends"])
        axes = list(attrs["axes"]) if attrs["axes"] is not None else list(range(len(starts)))
        steps = [1] * len(axes)
    else:
        if any(name not in ctx.initializers for name in node.input[1:]):
            shape = [0]
            if node.input[0] in ctx.explicit_shapes:
                return [(None, shape)]
            return [(shape, None)]

        starts = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
        ends = onnx.numpy_helper.to_array(ctx.initializers[node.input[2]]).tolist()

        axes = (
            onnx.numpy_helper.to_array(ctx.initializers[node.input[3]]).tolist()
            if len(node.input) > 3
            else list(range(len(starts)))
        )
        steps = (
            onnx.numpy_helper.to_array(ctx.initializers[node.input[4]]).tolist()
            if len(node.input) > 4
            else [1] * len(axes)
        )

    # Check for explicit shape first (for shape tensors like from Shape op)
    e_shape = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is not None:
        if not isinstance(e_shape, list):
            raise RuntimeError(f"Expected list for explicit shape slice, got {e_shape}")

        if axes != [0]:
            raise ValueError(f"Invalid axes {axes} for explicit shape slice")

        e_shape = e_shape[starts[0] : ends[0] : steps[0]] if e_shape != [0] else [0]
        return [(None, e_shape)]

    # Fallback to data shape (for regular data tensors)
    shape_data = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape_data is not None:
        assert isinstance(shape_data, list)
        shape_result = (
            _infer_sliced_shape(shape_data, axes, starts, ends, steps) if shape_data != [0] else [0]
        )
        return [(shape_result, None)]

    raise RuntimeError(f"Cannot get shape of {node.input[0]}")


def _make_equal_split_sizes(axis_len: int, count: int) -> list[int]:
    """Compute ONNX equal sections with a possibly smaller final chunk."""
    if count <= 0:
        raise ValueError(f"Split output count must be positive, got {count}")
    chunk = (axis_len + count - 1) // count
    sizes = [chunk] * (count - 1) + [axis_len - chunk * (count - 1)]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Split axis length {axis_len} cannot produce {count} positive outputs")
    return sizes


def _resolve_split_sizes(
    node: NodeProto,
    ctx: ShapeInferenceContext,
    attrs: dict[str, object],
    axis_len: int,
) -> list[int]:
    """Resolve exact section sizes from Split attributes or initializer input."""
    input_name = node.input[1] if len(node.input) > 1 else ""
    split_attr = attrs["split"]
    num_outputs = attrs["num_outputs"]
    if sum(value is not None for value in (split_attr, num_outputs)) + bool(input_name) > 1:
        raise ValueError(
            "Split accepts exactly one of split input, split attribute, or num_outputs"
        )
    if input_name:
        if input_name not in ctx.initializers:
            raise RuntimeError(f"Split input[1]={input_name} must be an initializer")
        values = np.asarray(onnx.numpy_helper.to_array(ctx.initializers[input_name]))
        if values.ndim != 1 or values.dtype not in (np.int32, np.int64):
            raise ValueError("Split input must be a 1D int32/int64 tensor")
        sizes = [int(value) for value in values.tolist()]
    elif split_attr is not None:
        if not isinstance(split_attr, (list, tuple)) or any(
            not isinstance(size, int) for size in split_attr
        ):
            raise ValueError("Split attribute must contain integer sizes")
        sizes = [int(size) for size in split_attr]
    else:
        if num_outputs is not None and not isinstance(num_outputs, int):
            raise ValueError("Split num_outputs must be an integer")
        count = len(node.output) if num_outputs is None else num_outputs
        sizes = _make_equal_split_sizes(axis_len, count)
    if len(sizes) != len(node.output) or any(size <= 0 for size in sizes):
        raise ValueError("Split sizes must contain one positive entry per output")
    if sum(sizes) != axis_len:
        raise ValueError(f"Split sizes {sizes} do not sum to axis length {axis_len}")
    return sizes


def _infer_split_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Split operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}")

    if shape == [0]:
        return [([0], None) for _ in node.output]

    # Split requires list shape
    if isinstance(shape, int):
        raise RuntimeError(f"Split input shape cannot be scalar: {shape}")

    attrs = _get_onnx_attrs(node, ctx.initializers)
    axis = attrs["axis"]
    if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
        raise ValueError(f"Split axis {axis!r} is invalid for rank {len(shape)}")
    axis %= len(shape)
    split_sizes = _resolve_split_sizes(node, ctx, attrs, shape[axis])

    output_shapes: list[tuple[int | list[int] | None, int | list[int] | None]] = []
    for split_size in split_sizes:
        output_shape = [*shape[:axis], split_size, *shape[axis + 1 :]]
        output_shapes.append((output_shape, None))

    return output_shapes


def _infer_squeeze_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Squeeze operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    input_shape, _ = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    if input_shape == [0]:
        return [([0], None)]

    if not isinstance(input_shape, list):
        raise RuntimeError(f"Input shape must be a list, got {input_shape}")

    if len(node.input) > 1:
        axes = _get_explicit_shape(node.input[1], ctx.explicit_shapes)
        if axes is None:
            axes = [i for i in range(len(input_shape)) if input_shape[i] == 1]
        elif not isinstance(axes, list):
            axes = [axes] if isinstance(axes, int) else []

        shape = []
        for i in range(len(input_shape)):
            if i in axes:
                if input_shape[i] != 1:
                    raise ValueError(f"Cannot squeeze axis {i} with size {input_shape[i]}")
                continue
            shape.append(input_shape[i])
    else:
        shape = [input_shape[0]] + [s for s in input_shape[1:] if s != 1]

    return [(shape, None)]


def _infer_transpose_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Transpose operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = _get_onnx_attrs(node, ctx.initializers)
    perm = attrs["perm"]

    shape = _get_data_shape(node.input[0], ctx.data_shapes)
    if shape is not None:
        if isinstance(shape, int):
            raise RuntimeError(f"Transpose input shape cannot be scalar: {shape}")

        if len(shape) == 1 or shape == [0]:
            pass
        else:
            effective_perm = perm if perm is not None else tuple(reversed(range(len(shape))))
            shape = [shape[i] for i in effective_perm]
        return [(shape, None)]

    e_shape = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}")

    if not isinstance(e_shape, list):
        raise RuntimeError(f"Expected list for transpose, got {e_shape}")

    if len(e_shape) == 1 or e_shape == [0]:
        pass
    else:
        effective_perm = perm if perm is not None else tuple(reversed(range(len(e_shape))))
        e_shape = [e_shape[i] for i in effective_perm]
    return [(None, e_shape)]


def _infer_unsqueeze_output_shape(ori_shape: list[int], axes: list[int]) -> list[int]:
    """
    Infer output shape for unsqueeze operation.

    :param ori_shape: Original shape.

    :param axes: Axes to unsqueeze.

    :return: Unsqueezed shape
    """
    new_shape = list(ori_shape)
    for axis in sorted(axes, reverse=True):
        if axis < 0:
            axis += len(ori_shape) + 1
        new_shape.insert(axis, 1)
    return new_shape


def _infer_unsqueeze_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Unsqueeze operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = _get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    # axes source: input[1] (opset >= 13) or the 'axes' attribute (opset < 13).
    if len(node.input) == 1:
        axes_attr = _get_onnx_attrs(node, ctx.initializers)["axes"]
        if axes_attr is None:
            raise RuntimeError(f"Unsqueeze node {node.name} has no axes input or attribute")
        axes = list(axes_attr)
    else:
        axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()

    if isinstance(shape, int):
        if axes != [0]:
            raise ValueError(f"Invalid axes {axes} for scalar unsqueeze")
        result_shape = [shape]
        return [(None, result_shape)]

    if not isinstance(shape, list):
        raise RuntimeError(f"Expected list shape for unsqueeze, got {shape}")

    if shape != [0]:
        shape = _infer_unsqueeze_output_shape(shape, axes)

    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def _infer_where_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Where operator.

    :param node: ONNX node.

    :param ctx: Shape inference context.

    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e = _get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = _get_shape(node.input[2], ctx.data_shapes, ctx.explicit_shapes)

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        shape = [0]
    else:
        shape = shape1 if shape1 != [0] else (shape2 if shape2 != [0] else [0])

    if is_e:
        return [(None, shape)]

    condition = _get_explicit_shape(node.input[0], ctx.explicit_shapes)
    value1 = _get_explicit_shape(node.input[1], ctx.explicit_shapes)
    value2 = _get_explicit_shape(node.input[2], ctx.explicit_shapes)

    if isinstance(condition, list) and isinstance(value1, list) and isinstance(value2, list):
        new_shape = value1.copy()
        for i in range(len(condition)):
            if condition[i] == 0:
                new_shape[i] = value2[i]
        return [(shape, new_shape)]

    return [(shape, None)]


ShapeInferFunc = Callable[
    [NodeProto, ShapeInferenceContext],
    list[tuple[int | list[int] | None, int | list[int] | None]],
]
INFER_SHAPE_FUNC_MAPPING: dict[str, ShapeInferFunc] = {
    "Abs": _infer_identity_shape,
    "Add": _infer_binary_op_shape,
    "ArgMax": _infer_argmax_shape,
    "AveragePool": _infer_pool_shape,
    "BatchNormalization": _infer_batch_norm_shape,
    "Cast": _infer_identity_shape,
    "Clip": _infer_identity_shape,
    "Concat": _infer_concat_shape,
    "ConstantOfShape": _infer_constant_of_shape_shape,
    "Conv": _infer_pool_shape,
    "ConvTranspose": _infer_convtranspose_shape,
    "Cos": _infer_identity_shape,
    "Div": _infer_binary_op_shape,
    "Dropout": _infer_identity_shape,
    "Elu": _infer_identity_shape,
    "Equal": _infer_binary_op_shape,
    "Exp": _infer_identity_shape,
    "Expand": _infer_expand_shape,
    "Flatten": _infer_flatten_shape,
    "Floor": _infer_identity_shape,
    "Gather": _infer_gather_shape,
    "Gelu": _infer_identity_shape,
    "Gemm": _infer_gemm_shape,
    "GlobalAveragePool": _infer_identity_shape,
    "LeakyRelu": _infer_identity_shape,
    "Log": _infer_identity_shape,
    "MatMul": _infer_matmul_shape,
    "Max": _infer_identity_shape,
    "MaxPool": _infer_pool_shape,
    "Min": _infer_identity_shape,
    "Mul": _infer_binary_op_shape,
    "Neg": _infer_identity_shape,
    "Pad": _infer_pad_shape,
    "Pow": _infer_identity_shape,
    "Range": _infer_range_shape,
    "Reciprocal": _infer_identity_shape,
    "ReduceMean": _infer_reduce_shape,
    "ReduceSum": _infer_reduce_shape,
    "Relu": _infer_identity_shape,
    "Reshape": _infer_reshape_shape,
    "Resize": _infer_resize_shape,
    "Scatter": _infer_identity_shape,
    "ScatterElements": _infer_identity_shape,
    "ScatterND": _infer_identity_shape,
    "Shape": _infer_shape_op_shape,
    "Sigmoid": _infer_identity_shape,
    "Sign": _infer_identity_shape,
    "Sin": _infer_identity_shape,
    "Slice": _infer_slice_shape,
    "Split": _infer_split_shape,
    "Sqrt": _infer_identity_shape,
    "Softmax": _infer_identity_shape,
    "Squeeze": _infer_squeeze_shape,
    "Sub": _infer_binary_op_shape,
    "Tanh": _infer_identity_shape,
    "Transpose": _infer_transpose_shape,
    "Unsqueeze": _infer_unsqueeze_shape,
    "Where": _infer_where_shape,
}


def _print_shapes(title: str, shapes: dict[str, list[int]], verbose: bool) -> None:
    """
    Print shape information if verbose mode is enabled.

    :param title: Section title.

    :param shapes: Shape dictionary.

    :param verbose: Whether to print.

    """
    if not verbose:
        return
    _logger.debug(title)
    _logger.debug(f"{'Name':<20} Shape")
    for name, shape in shapes.items():
        _logger.debug(f"{name:<20} {shape}")


def _process_node_outputs(
    node: NodeProto,
    results: Sequence[tuple[int | list[int] | None, int | list[int] | None]],
    ctx: ShapeInferenceContext,
) -> None:
    """
    Process and store node output shapes.

    :param node: ONNX node.

    :param results: Inference results.

    :param ctx: Shape inference context.

    """
    for output_name, (data_shape, explicit_shape) in zip(node.output, results, strict=True):
        if data_shape is not None:
            ctx.data_shapes[output_name] = data_shape
            if ctx.verbose:
                _logger.debug(f"{node.op_type:<20} {output_name:<20} {data_shape}")

        if explicit_shape is not None:
            ctx.explicit_shapes[output_name] = explicit_shape
            if ctx.verbose:
                _logger.debug(f"{node.op_type:<20} {output_name:<20} {explicit_shape} (explicit)")
            # Only use explicit_shape as data_shape if data_shape was not set
            if data_shape is None:
                ctx.data_shapes[output_name] = explicit_shape


def _infer_all_node_shapes(nodes: list[NodeProto], ctx: ShapeInferenceContext) -> None:
    """
    Infer shapes for all nodes in the graph.

    Nodes are topologically sorted first to ensure inputs are available.

    :param nodes: List of ONNX nodes.

    :param ctx: Shape inference context.

    """
    nodes = _topological_sort(nodes)
    for node in nodes:
        if node.op_type == "Constant":
            raise RuntimeError(
                "Constant nodes must be converted to initializers before shape inference"
            )

        infer_func = INFER_SHAPE_FUNC_MAPPING.get(node.op_type)
        if infer_func is None:
            raise NotImplementedError(f"Operator {node.op_type} is not supported")

        try:
            results = infer_func(node, ctx)
        except Exception as e:
            raise RuntimeError(
                f"Failed to infer shape for node {node.name} ({node.op_type}): {e}"
            ) from e

        _process_node_outputs(node, results, ctx)


def infer_onnx_shape(
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    has_batch_dim: bool = True,
    verbose: bool = False,
) -> dict[str, int | list[int]]:
    """
    Infer shapes for all tensors in an ONNX model.

    :param input_nodes: Model input value infos.

    :param output_nodes: Model output value infos.

    :param nodes: Model computation nodes.

    :param initializers: Model initializers.

    :param has_batch_dim: Whether tensors have batch dimension.

    :param verbose: Whether to print debug information.

    :return: Dictionary mapping all tensor names to their inferred shapes
    """
    input_shapes = extract_io_shapes(input_nodes, has_batch_dim)
    output_shapes = extract_io_shapes(output_nodes, has_batch_dim)
    initializer_shapes = _extract_initializer_shapes(initializers)

    # Type annotation to allow both int and list shapes during inference
    data_shapes: dict[str, int | list[int]] = {
        **input_shapes,
        **output_shapes,
        **initializer_shapes,
    }
    explicit_shapes = _extract_integer_initializers(initializers)

    if verbose:
        _print_shapes("Input shapes", input_shapes, verbose=True)
        _print_shapes("Output shapes", output_shapes, verbose=True)
        _print_shapes("Initializer shapes", initializer_shapes, verbose=True)
        _logger.debug("Inferring node shapes")
        _logger.debug(f"{'Op Type':20} {'Name':20} Output Shape")

    ctx = ShapeInferenceContext(
        data_shapes=data_shapes,
        explicit_shapes=explicit_shapes,
        initializers=initializers,
        verbose=verbose,
    )

    _infer_all_node_shapes(nodes, ctx)

    return data_shapes
