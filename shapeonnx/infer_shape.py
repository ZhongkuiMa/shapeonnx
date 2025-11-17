__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape", "extract_io_shapes"]

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto

from .onnx_attrs import get_onnx_attrs
from .utils import reformat_io_shape


@dataclass(frozen=True)
class ShapeInferenceContext:
    """
    Immutable context for shape inference.

    :param data_shapes: Maps tensor names to their inferred shapes
    :param explicit_shapes: Maps tensor names to constant shape values
    :param initializers: ONNX model initializers
    :param verbose: Whether to print debug information
    """

    data_shapes: dict[str, list[int]]
    explicit_shapes: dict[str, int | list[int]]
    initializers: dict[str, TensorProto]
    verbose: bool = False


def extract_io_shapes(
    nodes: list[ValueInfoProto], has_batch_dim: bool
) -> dict[str, list[int]]:
    """
    Extract shapes from model input/output nodes.

    :param nodes: List of ONNX value info nodes
    :param has_batch_dim: Whether nodes have a batch dimension
    :return: Dictionary mapping node names to shapes
    """
    return {node.name: reformat_io_shape(node, has_batch_dim) for node in nodes}


def extract_initializer_shapes(
    initializers: dict[str, TensorProto],
) -> dict[str, list[int]]:
    """
    Extract shapes from model initializers.

    :param initializers: Dictionary of ONNX initializers
    :return: Dictionary mapping initializer names to shapes
    """
    return {name: list(map(int, init.dims)) for name, init in initializers.items()}


def get_data_shape(name: str, shapes: dict[str, list[int]]) -> list[int] | None:
    """
    Retrieve data shape by name.

    :param name: Tensor name
    :param shapes: Shape dictionary
    :return: Shape if found, None otherwise
    """
    return shapes.get(name)


def preconvert_integer_initializers(
    initializers: dict[str, TensorProto],
) -> dict[str, int | list[int]]:
    """
    Pre-convert integer-type initializers to Python int/list for shape operations.

    This optimization avoids repeated conversions during shape inference.

    :param initializers: ONNX initializers
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


def get_explicit_shape(
    name: str,
    explicit_shapes: dict[str, int | list[int]],
) -> int | list[int] | None:
    """
    Retrieve explicit constant shape value.

    :param name: Tensor name
    :param explicit_shapes: Explicit shape dictionary (includes pre-converted initializers)
    :return: Constant value if found, None otherwise
    """
    return explicit_shapes.get(name)


def get_shape(
    name: str,
    shapes: dict[str, list[int]],
    explicit_shapes: dict[str, int | list[int]],
) -> tuple[int | list[int] | None, bool]:
    """
    Retrieve shape from any available source.

    :param name: Tensor name
    :param shapes: Data shape dictionary
    :param explicit_shapes: Explicit shape dictionary (includes pre-converted initializers)
    :return: Tuple of (shape, is_explicit)
    :raises RuntimeError: If shape cannot be found
    """
    if (shape := shapes.get(name)) is not None:
        return shape, False
    if (explicit_shape := explicit_shapes.get(name)) is not None:
        return explicit_shape, True
    raise RuntimeError(f"Cannot get shape of {name}.")


def store_data_shape(shape: list[int], shapes: dict[str, list[int]], name: str) -> None:
    """
    Store inferred data shape.

    :param shape: Inferred shape
    :param shapes: Shape dictionary to update
    :param name: Tensor name
    """
    shapes[name] = shape


def store_explicit_shape(
    shape: int | list[int], explicit_shapes: dict[str, int | list[int]], name: str
) -> None:
    """
    Store constant shape value.

    :param shape: Constant shape value
    :param explicit_shapes: Explicit shape dictionary to update
    :param name: Tensor name
    """
    explicit_shapes[name] = shape


def align_shapes(base: list[int], target: list[int]) -> list[int]:
    """
    Align target shape to base shape structure.

    :param base: Base shape
    :param target: Target shape to align
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


def right_align_shapes(
    shape1: list[int], shape2: list[int]
) -> tuple[list[int], list[int]]:
    """
    Right-align two shapes by padding with 1s.

    :param shape1: First shape
    :param shape2: Second shape
    :return: Tuple of right-aligned shapes
    """
    max_len = max(len(shape1), len(shape2))
    aligned1 = [1] * (max_len - len(shape1)) + shape1
    aligned2 = [1] * (max_len - len(shape2)) + shape2
    return aligned1, aligned2


def compute_broadcasted_shape(shape1: list[int], shape2: list[int]) -> list[int]:
    """
    Compute broadcasted shape from two aligned shapes.

    :param shape1: First shape
    :param shape2: Second shape
    :return: Broadcasted shape
    :raises RuntimeError: If shapes are incompatible
    """
    result = []
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            raise RuntimeError(f"Cannot broadcast {shape1} and {shape2}.")
        result.append(max(s1, s2))
    return result


def broadcast_shapes(shape1: list[int], shape2: list[int]) -> list[int]:
    """
    Broadcast two shapes using numpy broadcasting rules.

    :param shape1: First shape
    :param shape2: Second shape
    :return: Broadcasted shape
    """
    if [0] in (shape1, shape2):
        return [0]
    if not shape1:
        return shape2
    if not shape2:
        return shape1

    # Align dimensions if one is subset of another
    if all(s2 in shape1 for s2 in shape2):
        shape2 = align_shapes(shape1, shape2)
    elif all(s1 in shape2 for s1 in shape1):
        shape1 = align_shapes(shape2, shape1)

    aligned1, aligned2 = right_align_shapes(shape1, shape2)
    return compute_broadcasted_shape(aligned1, aligned2)


def infer_nochange_op_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for operators that preserve input shape.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def compute_binary_op_value(
    op_type: str, value1: int | float, value2: int | float
) -> int | float:
    """
    Compute binary operation on scalar values.

    :param op_type: Operation type
    :param value1: First operand
    :param value2: Second operand
    :return: Operation result
    :raises RuntimeError: If operation is unsupported
    """
    operations = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
    }
    if op_type not in operations:
        raise RuntimeError(
            f"Cannot calculate {op_type} with values {value1} and {value2}."
        )
    result = operations[op_type](value1, value2)
    return (
        int(result) if isinstance(value1, int) and isinstance(value2, int) else result
    )


def compute_explicit_binary_shape(
    op_type: str, e_shape1: int | list[int], e_shape2: int | list[int]
) -> list[int]:
    """
    Compute explicit shape for binary operations.

    :param op_type: Operation type
    :param e_shape1: First explicit shape
    :param e_shape2: Second explicit shape
    :return: Computed explicit shape
    :raises NotImplementedError: If operation is not supported
    """
    if op_type == "Mul":
        if isinstance(e_shape1, int) and isinstance(e_shape2, list):
            return [e_shape1 * s for s in e_shape2]
        if isinstance(e_shape2, int) and isinstance(e_shape1, list):
            return [e_shape2 * s for s in e_shape1]
        raise NotImplementedError(
            f"Cannot calculate explicit shape of {e_shape1} and {e_shape2}."
        )
    if op_type == "Equal":
        if not isinstance(e_shape1, list) or not isinstance(e_shape2, list):
            raise NotImplementedError(
                f"Cannot calculate explicit Equal of {e_shape1} and {e_shape2}."
            )
        if len(e_shape1) != len(e_shape2):
            raise ValueError(f"Shape mismatch: {e_shape1} vs {e_shape2}")
        return [1 if e_shape1[i] == e_shape2[i] else 0 for i in range(len(e_shape1))]
    raise NotImplementedError(
        f"Cannot calculate explicit shape of {op_type} with {e_shape1} and {e_shape2}."
    )


def infer_binary_op_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for binary operators.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e1 = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, is_e2 = get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    is_explicit = is_e1 or is_e2

    # Cache type checks to avoid redundant isinstance calls
    is_list1 = isinstance(shape1, list)
    is_list2 = isinstance(shape2, list)
    is_int1 = isinstance(shape1, int)
    is_int2 = isinstance(shape2, int)

    # Handle dynamic shapes
    if shape1 == [0] or shape2 == [0]:
        shape = [0]
    # Both are scalar shapes
    elif is_list1 and is_list2 and not shape1 and not shape2:
        shape = []
    # Scalar arithmetic
    elif (is_int1 or (is_list1 and not shape1)) and (
        is_int2 or (is_list2 and not shape2)
    ):
        val1 = (
            shape1
            if is_int1
            else get_explicit_shape(node.input[0], ctx.explicit_shapes)
        )
        val2 = (
            shape2
            if is_int2
            else get_explicit_shape(node.input[1], ctx.explicit_shapes)
        )
        if (
            val1 is None
            or val2 is None
            or isinstance(val1, list)
            or isinstance(val2, list)
        ):
            shape = [0]
        else:
            shape = int(compute_binary_op_value(node.op_type, val1, val2))
    # Broadcast tensor shapes
    elif is_list1 and is_list2:
        shape = broadcast_shapes(shape1, shape2)
    else:
        raise RuntimeError(
            f"Cannot calculate {node.op_type} with shape {shape1} and {shape2}."
        )

    if is_explicit:
        return [(None, shape)]

    # Check for explicit shape calculation
    e_shape1 = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape1 is not None and isinstance(e_shape1, (int, list)):
        e_shape2 = get_explicit_shape(node.input[1], ctx.explicit_shapes)
        if e_shape2 is not None and isinstance(e_shape2, (int, list)):
            explicit_shape = compute_explicit_binary_shape(
                node.op_type, e_shape1, e_shape2
            )
            return [(shape, explicit_shape)]

    return [(shape, None)]


def infer_argmax_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ArgMax operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    axis, keepdims = attrs["axis"], attrs["keepdims"]

    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")

    if shape != [0]:
        shape[axis] = 1
        if not keepdims:
            shape.pop(axis)
        if all(s == 1 for s in shape):
            shape = []

    return [(shape, None)]


def infer_batch_norm_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for BatchNormalization operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")
    return [(shape, None)]


def infer_concat_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Concat operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    axis = attrs["axis"]

    is_explicit = any(name in ctx.explicit_shapes for name in node.input)

    shape_list = []
    for name in node.input:
        if is_explicit:
            shape_i = get_explicit_shape(name, ctx.explicit_shapes)
            if shape_i is None:
                raise RuntimeError(f"Cannot get explicit shape of {name}.")
        else:
            shape_i, _ = get_shape(name, ctx.data_shapes, ctx.explicit_shapes)
        if shape_i == [0]:
            return [([0], None)]
        shape_list.append(shape_i)

    if is_explicit:
        shape = np.concatenate(shape_list, axis=axis).tolist()
        return [(None, shape)]

    # Handle mixed batch dimensions without mutating input shapes
    max_ndim = max(len(s) for s in shape_list)
    normalized_shapes = []
    for s in shape_list:
        s_len = len(s)
        if s_len not in (max_ndim, max_ndim - 1):
            raise ValueError(f"Invalid shape {s}")
        normalized_shapes.append([1, *s] if s_len == max_ndim - 1 else s)

    # Sum along concat axis
    shape = normalized_shapes[0].copy()
    for other_shape in normalized_shapes[1:]:
        shape[axis] += other_shape[axis]

    return [(shape, None)]


def infer_constant_of_shape_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ConstantOfShape operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}.")

    if shape != [0]:
        # Check if value is constant integer, Maybe we create a shape constant
        value = get_onnx_attrs(node, ctx.initializers)["value"]
        if np.issubdtype(value.dtype, np.integer):
            constant = np.full(shape, value, dtype=value.dtype).tolist()
            return [(shape, constant)]

    return [(shape, None)]


def compute_convtranspose_output_hw(
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

    :param input_shape: Input tensor shape
    :param weight_shape: Weight tensor shape
    :param kernel_shape: Kernel dimensions
    :param dilations: Dilation factors
    :param output_padding: Output padding
    :param pads: Input padding
    :param strides: Stride values
    :return: Output height/width
    """
    dim = len(kernel_shape)
    temp1 = [pads[i] + pads[i + dim] for i in range(dim)]
    temp2 = [dilations[i] * (kernel_shape[i] - 1) for i in range(dim)]
    output_hw = [
        math.ceil(
            (input_shape[i + 2] - 1) * strides[i]
            - temp1[i]
            + temp2[i]
            + output_padding[i]
            + 1
        )
        for i in range(dim)
    ]
    return output_hw


def infer_convtranspose_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for ConvTranspose operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    output_padding = attrs["output_padding"]
    pads = attrs["pads"]
    strides = attrs["strides"]

    if not (
        len(kernel_shape) == len(dilations) == 2
        and len(pads) == 4
        and len(strides) == 2
    ):
        raise NotImplementedError(
            f"ConvTranspose with kernel_shape={kernel_shape}, dilations={dilations}, "
            f"pads={pads}, strides={strides} is not supported."
        )

    input_shape = get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")
    if input_shape == [0]:
        return [([0], None)]

    weight_shape = list(ctx.initializers[node.input[1]].dims)
    output_hw = compute_convtranspose_output_hw(
        input_shape,
        weight_shape,
        kernel_shape,
        dilations,
        output_padding,
        pads,
        strides,
    )
    shape = [input_shape[0], weight_shape[1], *output_hw]
    return [(shape, None)]


def infer_expand_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Expand operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e1 = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2 = get_explicit_shape(node.input[1], ctx.explicit_shapes)
    if shape2 is None:
        shape2 = get_data_shape(node.input[1], ctx.data_shapes)
        if shape2 is None:
            raise RuntimeError(f"Cannot get shape of {node.input[1]}.")

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        raise RuntimeError(f"Cannot expand with shapes {shape1} and {shape2}.")

    shape = broadcast_shapes(shape1, shape2)

    if is_e1:
        return [(None, shape)]
    return [(shape, None)]


def infer_flatten_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Flatten operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")

    axis = get_onnx_attrs(node, ctx.initializers)["axis"]
    if shape != [0]:
        total = math.prod(shape)
        prefix = math.prod(shape[:axis])
        shape = shape[:axis] + [total // prefix]

    return [(shape, None)]


def infer_gather_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Gather operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    axis = get_onnx_attrs(node, ctx.initializers)["axis"]
    indices = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
    is_int_indices = isinstance(indices, int)

    # Gather from variable
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is not None:
        if shape != [0]:
            shape = [
                len(indices) if i == axis and not is_int_indices else shape[i]
                for i in range(len(shape))
                if not (i == axis and is_int_indices)
            ]
        return [(shape, None)]

    # Gather from explicit shape
    e_shape = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}.")

    if e_shape != [0]:
        if axis != 0:
            raise ValueError(f"Invalid axis {axis} for gather from explicit shape.")
        if not isinstance(e_shape, list):
            raise RuntimeError(f"Cannot gather from non-list explicit shape {e_shape}.")
        if is_int_indices:
            e_shape = e_shape[indices]
        else:
            e_shape = [
                e_shape[i]
                for i in indices
                if isinstance(indices, list) and i < len(e_shape)
            ]

    return [(None, e_shape)]


def infer_gemm_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Gemm operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    trans_a, trans_b = attrs["transA"], attrs["transB"]

    shape1, _ = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)

    if [0] in (shape1, shape2):
        return [([0], None)]

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        raise RuntimeError(f"Cannot perform Gemm with shapes {shape1} and {shape2}.")

    shape1 = shape1.copy()
    shape2 = shape2.copy()
    if trans_a and len(shape1) >= 2:
        shape1[-2], shape1[-1] = shape1[-1], shape1[-2]
    if trans_b and len(shape2) >= 2:
        shape2[-2], shape2[-1] = shape2[-1], shape2[-2]

    shape = shape2 if not shape1 else shape1[:-1] + shape2[-1:]
    return [(shape, None)]


def infer_matmul_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for MatMul operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, _ = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    assert isinstance(shape1, list) and isinstance(shape2, list)

    if [0] in (shape1, shape2):
        return [([0], None)]

    if len(shape2) <= 2:  # *-Vector, *-Matrix MatMul
        shape = [*shape1[:-1], *shape2[1:]]
    elif len(shape1) == len(shape2) and len(shape1) > 2 and len(shape2) > 2:
        # Batch MatMul
        shape = [*shape1[:-1], shape2[-1]]
    else:
        raise ValueError(f"Invalid shapes {shape1} and {shape2} for MatMul.")

    return [(shape, None)]


def compute_pool_output_hw(
    input_shape: list[int],
    kernel_shape: list[int],
    dilations: list[int],
    pads: list[int],
    strides: list[int],
    ceil_mode: bool,
) -> list[int]:
    """
    Compute output height/width for pooling operations.

    :param input_shape: Input tensor shape
    :param kernel_shape: Kernel dimensions
    :param dilations: Dilation factors
    :param pads: Padding values
    :param strides: Stride values
    :param ceil_mode: Whether to use ceiling for output size
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


def infer_pool_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for pooling operators (Conv, MaxPool, AveragePool).

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
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

    input_shape = get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")
    if input_shape == [0]:
        return [([0], None)]

    # Output channel: from weight for Conv, same as input for pooling
    if len(node.input) > 1:
        weight_shape, _ = get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
        if not isinstance(weight_shape, list):
            raise RuntimeError(f"Weight shape must be a list, got {weight_shape}.")
        output_channel = weight_shape[0]
    else:
        output_channel = input_shape[1]

    output_hw = compute_pool_output_hw(
        input_shape, kernel_shape, dilations, pads, strides, ceil_mode
    )
    shape = [input_shape[0], output_channel, *output_hw]
    return [(shape, None)]


def infer_pad_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Pad operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    input_shape, is_explicit = get_shape(
        node.input[0], ctx.data_shapes, ctx.explicit_shapes
    )
    if input_shape == [0]:
        if is_explicit:
            return [(None, [0])]
        return [([0], None)]

    if not isinstance(input_shape, list):
        raise RuntimeError(f"Input shape must be a list, got {input_shape}.")

    pads = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
    if len(node.input) == 4:
        axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[3]]).tolist()
        raise NotImplementedError(f"Pad with axes={axes} is not supported.")

    dim = len(pads) // 2
    combined_pads = [pads[i] + pads[i + dim] for i in range(dim)]
    shape = [s + p for s, p in zip(input_shape, combined_pads)]

    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def infer_range_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Range operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    start = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    limit = get_explicit_shape(node.input[1], ctx.explicit_shapes)
    delta = get_explicit_shape(node.input[2], ctx.explicit_shapes)

    if not (
        isinstance(start, int) and isinstance(limit, int) and isinstance(delta, int)
    ):
        return [([0], None)]

    if delta > 0:
        length = max(0, (limit - start + delta - 1) // delta)
    elif delta < 0:
        length = max(0, (start - limit - delta - 1) // (-delta))
    else:
        raise ValueError("Range step delta cannot be 0.")

    return [([length], None)]


def infer_reduce_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for reduction operators (ReduceMean, ReduceSum).

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    keepdims = get_onnx_attrs(node, ctx.initializers)["keepdims"]
    axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()

    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")

    if shape != [0]:
        for axis in axes:
            shape[axis] = 1 if keepdims else 0
        shape = [x for x in shape if x != 0]

    return [(shape, None)]


def infer_reshape_output_shape(ori_shape: list[int], new_shape: list[int]) -> list[int]:
    """
    Infer reshaped output shape without actual computation.

    :param ori_shape: Original shape
    :param new_shape: Target shape with possible -1
    :return: Inferred output shape
    """
    total = math.prod(ori_shape)
    inferred_idx = -1
    remaining = total

    for idx, dim in enumerate(new_shape):
        if dim == -1:
            inferred_idx = idx
        else:
            remaining //= dim

    result = new_shape.copy()
    if inferred_idx != -1:
        result[inferred_idx] = remaining

    return result


def infer_reshape_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Reshape operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    data_shape, _ = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    target_shape = get_explicit_shape(node.input[1], ctx.explicit_shapes)

    if not isinstance(data_shape, list) or not isinstance(target_shape, list):
        return [([0], None)]

    if target_shape == [0] or (data_shape == [0] and -1 in target_shape):
        return [([0], None)]

    shape = infer_reshape_output_shape(data_shape, target_shape)
    return [(shape, None)]


def create_resize_rounding_op(nearest_mode: str):
    """
    Create rounding function for resize operation.

    :param nearest_mode: Nearest mode strategy
    :return: Rounding function
    :raises NotImplementedError: If mode is unsupported
    """
    if nearest_mode == "floor":
        return math.floor
    if nearest_mode == "ceil":
        return math.ceil
    if nearest_mode == "round_prefer_floor":
        return lambda x: int(x + 0.4999999)
    if nearest_mode == "round_prefer_ceil":
        return lambda x: int(x + 0.5000001)
    raise NotImplementedError(f"Resize nearest_mode={nearest_mode} is not supported.")


def infer_resize_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Resize operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    align_mode = attrs["coordinate_transformation_mode"]
    mode = attrs["mode"]
    nearest_mode = attrs.get("nearest_mode", "floor")

    if mode != "nearest":
        raise NotImplementedError(f"Resize mode={mode} is not supported.")

    input_shape = get_data_shape(node.input[0], ctx.data_shapes)
    if input_shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")
    if input_shape == [0]:
        return [([0], None)]

    op_round = create_resize_rounding_op(nearest_mode)

    scales = onnx.numpy_helper.to_array(ctx.initializers[node.input[2]]).tolist()
    if not scales:
        raise ValueError(f"Resize with empty scales is not supported.")

    if align_mode not in {"asymmetric", "half_pixel"}:
        raise NotImplementedError(f"Resize align_mode={align_mode} is not supported.")

    shape = [op_round(dim * scale) for dim, scale in zip(input_shape, scales)]
    return [(shape, None)]


def infer_shape_op_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Shape operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)

    if not is_explicit:
        if not isinstance(shape, list):
            raise RuntimeError(f"Expected list shape, got {shape}.")
        return [(None, shape)]

    if isinstance(shape, int):
        result_shape = []
    elif isinstance(shape, list) and shape == [0]:
        result_shape = [0]
    elif isinstance(shape, list):
        result_shape = [1, len(shape)]
    else:
        raise RuntimeError(f"Unexpected explicit shape type {type(shape)}.")

    return [(None, result_shape)]


def infer_sliced_shape(
    shape: list[int],
    axes: list[int],
    starts: list[int],
    ends: list[int],
    steps: list[int],
) -> list[int]:
    """
    Infer shape after slicing operation.

    :param shape: Original shape
    :param axes: Axes to slice
    :param starts: Start indices
    :param ends: End indices
    :param steps: Step sizes
    :return: Sliced shape
    """
    new_shape = list(shape)
    for axis, start, end, step in zip(axes, starts, ends, steps):
        size = shape[axis]
        start = min(max(start + size if start < 0 else start, 0), size)
        end = min(max(end + size if end < 0 else end, 0), size)
        if step < 0:
            warnings.warn(f"Negative step ({step}) is not fully tested.")
        new_shape[axis] = max(
            0, (end - start + (step - (1 if step > 0 else -1))) // step
        )
    return new_shape


def infer_slice_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Slice operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
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

    # Slice tensor
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is not None:
        shape = (
            infer_sliced_shape(shape, axes, starts, ends, steps)
            if shape != [0]
            else [0]
        )
        return [(shape, None)]

    # Slice explicit shape
    e_shape = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}.")

    if not isinstance(e_shape, list):
        raise RuntimeError(f"Expected list for explicit shape slice, got {e_shape}.")

    if axes != [0]:
        raise ValueError(f"Invalid axes {axes} for explicit shape slice.")

    e_shape = e_shape[starts[0] : ends[0] : steps[0]] if e_shape != [0] else [0]
    return [(None, e_shape)]


def infer_split_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Split operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is None:
        raise RuntimeError(f"Cannot get shape of {node.input[0]}.")

    if shape == [0]:
        return [([0], None) for _ in node.output]

    attrs = get_onnx_attrs(node, ctx.initializers)
    axis = attrs["axis"]
    if attrs["num_outputs"] is not None:
        raise NotImplementedError(
            f"Split with num_outputs={attrs['num_outputs']} is not supported."
        )
    if node.input[1] not in ctx.initializers:
        raise RuntimeError(f"Split input[1]={node.input[1]} must be an initializer.")

    split_sizes = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()
    if axis < 0:
        axis += len(shape)

    output_shapes = []
    for split_size, output_name in zip(split_sizes, node.output):
        output_shape = shape[:axis] + [split_size] + shape[axis + 1 :]
        output_shapes.append((output_shape, None))

    return output_shapes


def infer_squeeze_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Squeeze operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    input_shape, _ = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    if input_shape == [0]:
        return [([0], None)]

    if not isinstance(input_shape, list):
        raise RuntimeError(f"Input shape must be a list, got {input_shape}.")

    if len(node.input) > 1:
        axes = get_explicit_shape(node.input[1], ctx.explicit_shapes)
        if axes is None:
            axes = [i for i in range(len(input_shape)) if input_shape[i] == 1]
        elif not isinstance(axes, list):
            axes = [axes] if isinstance(axes, int) else []

        shape = []
        for i in range(len(input_shape)):
            if i in axes:
                if input_shape[i] != 1:
                    raise ValueError(
                        f"Cannot squeeze axis {i} with size {input_shape[i]}."
                    )
                continue
            shape.append(input_shape[i])
    else:
        # Squeeze all 1 dims but keep batch
        shape = [s for s in input_shape if s != 1]
        if shape:
            shape = [input_shape[0]] + shape

    return [(shape, None)]


def infer_transpose_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Transpose operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    attrs = get_onnx_attrs(node, ctx.initializers)
    perm = attrs["perm"]

    # Transpose tensor
    shape = get_data_shape(node.input[0], ctx.data_shapes)
    if shape is not None:
        if len(shape) == 1:
            shape = [shape[0], 1]
        else:
            shape = [shape[i] for i in perm] if shape != [0] else [0]
        return [(shape, None)]

    # Transpose explicit shape
    e_shape = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    if e_shape is None:
        raise RuntimeError(f"Cannot get explicit shape of {node.input[0]}.")

    if not isinstance(e_shape, list):
        raise RuntimeError(f"Expected list for transpose, got {e_shape}.")

    if len(e_shape) == 1:
        e_shape = [e_shape[0], 1]
    else:
        e_shape = [e_shape[i] for i in perm] if e_shape != [0] else [0]
    return [(None, e_shape)]


def infer_unsqueeze_output_shape(ori_shape: list[int], axes: list[int]) -> list[int]:
    """
    Infer output shape for unsqueeze operation.

    :param ori_shape: Original shape
    :param axes: Axes to unsqueeze
    :return: Unsqueezed shape
    """
    new_shape = list(ori_shape)
    for axis in sorted(axes, reverse=True):
        if axis < 0:
            axis += len(ori_shape) + 1
        new_shape.insert(axis, 1)
    return new_shape


def infer_unsqueeze_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Unsqueeze operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape, is_explicit = get_shape(node.input[0], ctx.data_shapes, ctx.explicit_shapes)
    axes = onnx.numpy_helper.to_array(ctx.initializers[node.input[1]]).tolist()

    # Handle scalar expansion
    if isinstance(shape, int):
        if axes != [0]:
            raise ValueError(f"Invalid axes {axes} for scalar unsqueeze.")
        result_shape = [shape]
        return [(None, result_shape)]

    if not isinstance(shape, list):
        raise RuntimeError(f"Expected list shape for unsqueeze, got {shape}.")

    if shape != [0]:
        shape = infer_unsqueeze_output_shape(shape, axes)

    if is_explicit:
        return [(None, shape)]
    return [(shape, None)]


def infer_where_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """
    Infer shape for Where operator.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    shape1, is_e = get_shape(node.input[1], ctx.data_shapes, ctx.explicit_shapes)
    shape2, _ = get_shape(node.input[2], ctx.data_shapes, ctx.explicit_shapes)

    if not isinstance(shape1, list) or not isinstance(shape2, list):
        shape = [0]
    else:
        shape = shape1 if shape1 != [0] else (shape2 if shape2 != [0] else [0])

    # If is_e, return explicit update
    if is_e:
        return [(None, shape)]

    # Not explicit - return data update, but also try explicit shape calculation
    # Try explicit shape calculation
    condition = get_explicit_shape(node.input[0], ctx.explicit_shapes)
    value1 = get_explicit_shape(node.input[1], ctx.explicit_shapes)
    value2 = get_explicit_shape(node.input[2], ctx.explicit_shapes)

    # Process the where operation if all inputs have explicit shapes
    if all(isinstance(v, list) for v in [condition, value1, value2]):
        new_shape = value1.copy()
        for i in range(len(condition)):
            if condition[i] == 0:
                new_shape[i] = value2[i]
        return [(shape, new_shape)]

    return [(shape, None)]


# Operator inference function mapping
# Return type: list of (data_shape, explicit_shape) tuples
# - Single-output ops: list with 1 element [(data_shape, explicit_shape)]
# - Multi-output ops: list with N elements, one per output
ShapeInferFunc = Callable[
    [NodeProto, ShapeInferenceContext],
    list[tuple[int | list[int] | None, int | list[int] | None]],
]
INFER_SHAPE_FUNC_MAPPING: dict[str, ShapeInferFunc] = {
    "Add": infer_binary_op_shape,
    "ArgMax": infer_argmax_shape,
    "AveragePool": infer_pool_shape,
    "BatchNormalization": infer_batch_norm_shape,
    "Cast": infer_nochange_op_shape,
    "Clip": infer_nochange_op_shape,
    "Concat": infer_concat_shape,
    "ConstantOfShape": infer_constant_of_shape_shape,
    "Conv": infer_pool_shape,
    "ConvTranspose": infer_convtranspose_shape,
    "Cos": infer_nochange_op_shape,
    "Div": infer_binary_op_shape,
    "Dropout": infer_nochange_op_shape,
    "Equal": infer_binary_op_shape,
    "Expand": infer_expand_shape,
    "Flatten": infer_flatten_shape,
    "Floor": infer_nochange_op_shape,
    "Gather": infer_gather_shape,
    "Gemm": infer_gemm_shape,
    "GlobalAveragePool": infer_nochange_op_shape,
    "LeakyRelu": infer_nochange_op_shape,
    "MatMul": infer_matmul_shape,
    "Max": infer_nochange_op_shape,
    "MaxPool": infer_pool_shape,
    "Min": infer_nochange_op_shape,
    "Mul": infer_binary_op_shape,
    "Neg": infer_nochange_op_shape,
    "Pad": infer_pad_shape,
    "Pow": infer_nochange_op_shape,
    "Range": infer_range_shape,
    "ReduceMean": infer_reduce_shape,
    "ReduceSum": infer_reduce_shape,
    "Relu": infer_nochange_op_shape,
    "Reshape": infer_reshape_shape,
    "Resize": infer_resize_shape,
    "Scatter": infer_nochange_op_shape,
    "ScatterElements": infer_nochange_op_shape,
    "ScatterND": infer_nochange_op_shape,
    "Shape": infer_shape_op_shape,
    "Sigmoid": infer_nochange_op_shape,
    "Sign": infer_nochange_op_shape,
    "Sin": infer_nochange_op_shape,
    "Slice": infer_slice_shape,
    "Split": infer_split_shape,
    "Softmax": infer_nochange_op_shape,
    "Squeeze": infer_squeeze_shape,
    "Sub": infer_binary_op_shape,
    "Tanh": infer_nochange_op_shape,
    "Transpose": infer_transpose_shape,
    "Unsqueeze": infer_unsqueeze_shape,
    "Where": infer_where_shape,
}


def print_shapes(title: str, shapes: dict[str, list[int]], verbose: bool) -> None:
    """
    Print shape information if verbose mode is enabled.

    :param title: Section title
    :param shapes: Shape dictionary
    :param verbose: Whether to print
    """
    if not verbose:
        return
    print(f"{title}...")
    print(f"{'Name':<20} Shape")
    for name, shape in shapes.items():
        print(f"{name:<20} {shape}")


def infer_onnx_shape(
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    has_batch_dim: bool = True,
    verbose: bool = False,
) -> dict[str, list[int]]:
    """
    Infer shapes for all tensors in an ONNX model.

    This function performs shape inference on ONNX models, tracking both data
    shapes (tensor dimensions) and explicit shapes (constant shape values used
    in shape calculations).

    :param input_nodes: Model input value infos
    :param output_nodes: Model output value infos
    :param nodes: Model computation nodes
    :param initializers: Model initializers (constants)
    :param has_batch_dim: Whether tensors have batch dimension
    :param verbose: Whether to print debug information
    :return: Dictionary mapping all tensor names to their inferred shapes
    :raises RuntimeError: If Constant nodes are found (should be converted to initializers)
    """
    # Extract initial shapes
    input_shapes = extract_io_shapes(input_nodes, has_batch_dim)
    output_shapes = extract_io_shapes(output_nodes, has_batch_dim)
    initializer_shapes = extract_initializer_shapes(initializers)

    data_shapes = {**input_shapes, **output_shapes, **initializer_shapes}

    # Pre-convert integer initializers for performance (avoids repeated conversions)
    explicit_shapes = preconvert_integer_initializers(initializers)

    if verbose:
        print_shapes("Input shapes", input_shapes, True)
        print_shapes("Output shapes", output_shapes, True)
        print_shapes("Initializer shapes", initializer_shapes, True)
        print("Inferring node shapes...")
        print(f"{'Op Type':20} {'Name':20} Output Shape")

    ctx = ShapeInferenceContext(
        data_shapes=data_shapes,
        explicit_shapes=explicit_shapes,
        initializers=initializers,
        verbose=verbose,
    )

    # Infer shapes for each node
    for node in nodes:
        if node.op_type == "Constant":
            raise RuntimeError(
                "Constant nodes must be converted to initializers before "
                "shape inference."
            )

        infer_func = INFER_SHAPE_FUNC_MAPPING.get(node.op_type)
        if infer_func is None:
            raise NotImplementedError(f"Operator {node.op_type} is not supported.")

        try:
            results = infer_func(node, ctx)
        except Exception as e:
            raise RuntimeError(
                f"Failed to infer shape for node {node.name} ({node.op_type}): {e}"
            ) from e

        # Assign shapes to outputs - zip results with output names
        for output_name, (data_shape, explicit_shape) in zip(node.output, results):
            if data_shape is not None:
                data_shapes[output_name] = data_shape
                if verbose:
                    print(f"{node.op_type:<20} {output_name:<20} {data_shape}")

            if explicit_shape is not None:
                explicit_shapes[output_name] = explicit_shape
                if verbose:
                    print(
                        f"{node.op_type:<20} {output_name:<20} {explicit_shape} "
                        f"(explicit)"
                    )

    data_shapes.update(explicit_shapes)
    return data_shapes
