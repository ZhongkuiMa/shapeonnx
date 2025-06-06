__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape"]

import math
import warnings
from math import ceil, floor

import numpy as np
import onnx
from onnx import ValueInfoProto, NodeProto, TensorProto

from slimonnx.slimonnx.onnx_attrs import get_onnx_attrs
from slimonnx.slimonnx.utils import reformat_io_shape

_VERBOSE = False


def _get_input_shape(input_nodes: list[ValueInfoProto], shapes: dict[str, list[int]]):
    for input_node in input_nodes:
        shape = reformat_io_shape(input_node)
        shapes[input_node.name] = shape
        if _VERBOSE:
            print(f"Input {input_node.name:<50} Shape={shape}")


def _get_output_shape(output_nodes: list[ValueInfoProto], shapes: dict[str, list[int]]):
    for output_node in output_nodes:
        shape = reformat_io_shape(output_node)
        shapes[output_node.name] = shape
        if _VERBOSE:
            print(f"Output {output_node.name:<50} Shape={shape}")


def _get_initializer_shape(
    initializers: dict[str, TensorProto], shapes: dict[str, list[int]]
):
    for initializer in initializers.values():
        shape = [int(x) for x in initializer.dims]
        shapes[initializer.name] = shape
        if _VERBOSE:
            print(f"Initializer {initializer.name:<50} Shape={shape}")


def _get_data_shape(
    name: str, shapes: dict[str, list[int]], silent: bool = True
) -> int | list[int] | None:
    shape = shapes.get(name)
    if shape is not None:
        # The data shape must be a list of int
        return shape.copy()
    if silent:
        return None
    raise RuntimeError(f"Cannot get data shape of {name}.")


def _get_explicit_shape(
    name: str,
    initializers: dict[str, TensorProto],
    explicit_shapes: dict[str, int | list[int]],
    silent: bool = True,
) -> int | list[int] | None:
    shape = initializers.get(name)
    if shape is not None:
        shape = onnx.numpy_helper.to_array(shape).tolist()
        return shape
    shape = explicit_shapes.get(name)
    if shape is not None:
        # The explicit shape either is a list of int or a single int
        if type(shape) is not int:
            shape = shape.copy()
        return shape
    if silent:
        return None
    raise RuntimeError(f"Cannot get explicit shape of {name}.")


def _get_shape(
    name: str,
    shapes: dict[str, list[int] | None],
    initializers: dict[str, TensorProto],
    explicit_shapes: dict[str, int | list[int]],
    silent: bool = True,
) -> tuple[int | list[int] | None, bool]:
    shape = shapes.get(name)
    if shape is not None:
        # The data shape must be a list of int
        return shape.copy(), False
    shape = initializers.get(name)
    if shape is not None:
        shape = onnx.numpy_helper.to_array(shape).tolist()
        return shape, True
    shape = explicit_shapes.get(name)
    if shape is not None:
        # The explicit shape either is a list of int or a single int
        if type(shape) is not int:
            shape = shape.copy()
        return shape, True
    if silent:
        return None, False
    raise RuntimeError(f"Cannot get shape of {name}.")


def _store_data_shape(
    shape: list[int], shapes: dict[str, list[int] | None], op_type: str, name: str
):
    shapes[name] = shape
    if _VERBOSE:
        print(f"Node {op_type:<20} {name:<40} shape={shape}")


def _store_explicit_shape(
    shape: list[int],
    explicit_shapes: dict[str, int | list[int]],
    op_type: str,
    name: str,
):
    explicit_shapes[name] = shape
    if _VERBOSE:
        print(f"Node {op_type:<20} {name:<40} value={shape}")


def _broadcast_shapes(shape1: list[int], shape2: list[int]) -> list[int]:
    if shape1 == [0] or shape2 == [0]:
        # This is a dynamic shape.
        return [0]

    if len(shape1) == 0:
        return shape2.copy()
    if len(shape2) == 0:
        return shape1.copy()

    # Align some dims if possible.
    if all(s2 in shape1 for s2 in shape2):
        new_shape2 = [1] * max(len(shape1), len(shape2))
        j = 0
        for i in range(len(shape1)):
            if shape1[i] == shape2[j]:
                new_shape2[i] = shape2[j]
                j += 1
                if j >= len(shape2):
                    break
        shape2 = new_shape2
    elif all(s1 in shape2 for s1 in shape1):
        new_shape1 = [1] * max(len(shape1), len(shape2))
        j = 0
        for i in range(len(shape2)):
            if shape2[i] == shape1[j]:
                new_shape1[i] = shape1[j]
                j += 1
                if j >= len(shape1):
                    break
        shape1 = new_shape1

    # The following code is about the expand operator in ONNX.
    # When one of the shape has some 1s, we need to expand the dimension to the real
    # value. For example, [411].expand([1, 1]) => [1, 411]
    # Apply right-alignment and fill the missing dims with 1s.
    shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    shape2 = [1] * (len(shape1) - len(shape2)) + shape2

    new_shape = []
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            raise RuntimeError(f"Cannot broadcast {shape1} and {shape2}.")
        s = max(s1, s2)
        new_shape.append(s)

    return new_shape


def _infer_shape_of_nochange_op(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int]],
    explicit_shapes: dict[str, int | list[int]],
):
    shape, is_e = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    if is_e:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_binary_op(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape1, is_e1 = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    shape2, is_e2 = _get_shape(
        node.input[1], shapes, initializers, explicit_shapes, False
    )
    is_explicit = is_e1 or is_e2
    # If one of two is explicit, this binary operation is to calculate a new
    # explicit shape.

    if shape1 == [0] or shape2 == [0]:
        # This is a dynamic shape, and we use [0] to represent the shape.
        # Because there may need to be a broadcast operation.
        shape = [0]
        if is_explicit:
            _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        else:
            _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if (type(shape1) is int or len(shape1) == 0) and (
        type(shape2) is int or len(shape2) == 0
    ):
        # This is a scalar, single item, try to find the static value.
        if type(shape1) is list:
            shape1 = _get_explicit_shape(node.input[0], initializers, explicit_shapes)
        if type(shape2) is list:
            shape2 = _get_explicit_shape(node.input[1], initializers, explicit_shapes)

        if shape1 is None or shape2 is None:
            # This is a dynamic shape.
            shape = [0]
        else:
            # Operate the actual values
            if node.op_type == "Add":
                shape = shape1 + shape2
            elif node.op_type == "Mul":
                shape = shape1 * shape2
            elif node.op_type == "Sub":
                shape = shape1 - shape2
            elif node.op_type == "Div":
                shape = shape1 / shape2
            else:
                raise RuntimeError(
                    f"Cannot calculate {node.op_type} with shape {shape1} and {shape2}."
                )
            shape = int(shape)

    elif len(shape1) > 0 or len(shape2) > 0:
        # Use a broadcast mechanism to calculate the output shape。
        shape = _broadcast_shapes(shape1, shape2)

    else:
        raise RuntimeError(
            f"Cannot calculate {node.op_type} with shape {shape1} and {shape2}."
        )

    if is_explicit:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])

        # There maybe an explicit shape calculation in the same time.
        def _need_operate_shapes(name: str) -> list[int] | None:
            e_shape = _get_explicit_shape(name, initializers, explicit_shapes, True)
            if e_shape is None or not (
                isinstance(e_shape, int)
                or (isinstance(e_shape, list) and isinstance(e_shape[0], int))
            ):
                return None
            return e_shape

        e_shape1 = _need_operate_shapes(node.input[0])
        if e_shape1 is None:
            return
        e_shape2 = _need_operate_shapes(node.input[1])
        if e_shape2 is None:
            return

        if node.op_type == "Mul":
            if isinstance(e_shape1, int):
                assert isinstance(e_shape2, list) and isinstance(
                    e_shape2[0], int
                ), f"Invalid shape {e_shape2}"
                shape = [e_shape1 * s for s in e_shape2]
            elif isinstance(e_shape2, int):
                assert isinstance(e_shape1, list) and isinstance(
                    e_shape1[0], int
                ), f"Invalid shape {e_shape1}"
                shape = [e_shape2 * s for s in e_shape1]
            else:
                raise NotImplementedError(
                    f"Cannot calculate exlicit shape of {e_shape1} and {e_shape2}."
                )
        elif node.op_type == "Equal":
            assert len(e_shape1) == len(e_shape2)
            shape = [
                1 if e_shape1[i] == e_shape2[i] else 0 for i in range(len(e_shape1))
            ]
        else:
            raise NotImplementedError(
                f"Cannot calculate exlicit shape of {node.op_type} "
                f"with {e_shape1} and {e_shape2}."
            )
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_argmax(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    axis = attrs["axis"]
    keepdims = attrs["keepdims"]

    shape = _get_data_shape(node.input[0], shapes, False)
    if shape != [0]:
        shape[axis] = 1
        if not keepdims:
            shape.pop(axis)
        if all(s == 1 for s in shape):
            # This is a scalar, single item
            shape = []

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_batch_norm(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape = _get_data_shape(node.input[0], shapes, False)
    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_concat(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    axis = attrs["axis"]

    """
    There are two cases:
    1. Concatenate several shapes, e.g., shape1=[1, 48], shape2=[-1], then [1, 48, -1]
    2. Concatenate several tensor values.
    """
    is_explicit = any(name in explicit_shapes for name in node.input)

    if is_explicit:
        # We will calculate the explicit shapes with initializers.
        # This is to concatenate several 1d lists of int by order
        # and the axis must be 0.
        shape_list = []
        for name in node.input:
            shape_i = _get_explicit_shape(name, initializers, explicit_shapes, False)
            if shape_i == [0]:
                shape = [0]
                _store_data_shape(shape, shapes, node.op_type, node.output[0])
                return
            shape_list.append(shape_i)
        shape = np.concatenate(shape_list, axis=axis).tolist()
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        return

    # We will calculate the shapes of tensors
    # This is to calculate the size sum of the specified axis.
    shape_list = []
    for name in node.input:
        shape_i, _ = _get_shape(name, shapes, initializers, explicit_shapes, False)
        if shape_i == [0]:
            shape = [0]
            _store_data_shape(shape, shapes, node.op_type, node.output[0])
            return
        shape_list.append(shape_i)

    # Sometimes, it concats initializers (without batch dim) with variables (with
    # batch dim), so we need to add the batch dim.
    max_ndim = max(len(s) for s in shape_list)
    for s in shape_list:
        assert len(s) == max_ndim or len(s) == max_ndim - 1, f"Invalid shape {s}"
        if len(s) == max_ndim - 1:
            # Add the batch dim
            s.insert(0, 1)

    # Calculate the output shape
    shape = shape_list[0]
    for i in range(1, len(shape_list)):
        shape[axis] += shape_list[i][axis]

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_constant_of_shape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape = _get_explicit_shape(node.input[0], initializers, explicit_shapes, False)
    _store_data_shape(shape, shapes, node.op_type, node.output[0])

    attrs = get_onnx_attrs(node, initializers)
    value = attrs["value"]
    if np.issubdtype(value.dtype, np.integer):
        # This may be an explicit shape
        constant = np.full(shape, value, dtype=value.dtype).tolist()
        _store_explicit_shape(constant, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_convtranspose(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    output_padding = attrs["output_padding"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    if (
        not len(kernel_shape) == 2
        and len(dilations) == 2
        and len(pads) == 4
        and len(strides) == 2
    ):
        raise NotImplementedError(
            f"We have not supported ConvTranspose node with "
            f"kernel_shape={kernel_shape}, dilations={dilations}, "
            f"pads={pads}, strides={strides}."
        )

    input_shape = _get_data_shape(node.input[0], shapes)
    if input_shape == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    weight_shape = list(initializers[node.input[1]].dims)
    # Calculate the output size
    temp1 = [pads[0] + pads[1], pads[2] + pads[3]]
    temp2 = [dilations[0] * (kernel_shape[0] - 1), dilations[1] * (kernel_shape[1] - 1)]
    output_hw = [0, 0]
    ceil_mode = True
    for i in range(2):
        output_hw[i] = (
            (input_shape[i + 2] - 1) * strides[i]
            - temp1[i]
            + temp2[i]
            + output_padding[i]
            + 1
        )
        output_hw[i] = ceil(output_hw[i]) if ceil_mode else floor(output_hw[i])
    shape = [input_shape[0], weight_shape[1]]
    for i in range(2):
        shape.append(output_hw[i])

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_expand(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape1, is_e1 = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    shape2 = _get_explicit_shape(node.input[1], initializers, explicit_shapes, True)
    if shape2 is None:
        shape2 = _get_data_shape(node.input[1], shapes, False)

    shape = _broadcast_shapes(shape1, shape2)

    if is_e1:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_flatten(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    axis = get_onnx_attrs(node, initializers)["axis"]

    shape = _get_data_shape(node.input[0], shapes, False)
    if shape != [0]:
        shape = shape[:axis] + [int(math.prod(shape) / math.prod(shape[:axis]))]

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_gather(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    axis = get_onnx_attrs(node, initializers)["axis"]
    indices = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()
    is_int_indices = type(indices) is int

    # When the input is a variable.
    shape = _get_data_shape(node.input[0], shapes)
    if shape is not None:
        if shape != [0]:
            new_shape = []
            for i in range(len(shape)):
                if i == axis:
                    if is_int_indices:
                        # If the indices is a single int, we need to remove the axis
                        continue
                    else:
                        # If the indices is a list, we need to keep the axis
                        new_shape.append(len(indices))
                else:
                    new_shape.append(shape[i])
            shape = new_shape
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    # When the input itself is a shape.
    e_shape = _get_explicit_shape(node.input[0], initializers, explicit_shapes, False)
    if e_shape != [0]:
        assert axis == 0, f"Invalid axis {axis}"
        new_e_shape = []
        for i in range(len(e_shape)):
            if is_int_indices:
                if i == indices:
                    new_e_shape = e_shape[i]
                    break
            else:
                if i in indices:
                    new_e_shape.append(e_shape[i])
        e_shape = new_e_shape
        # Here we do not consider the scalar shape because gather may extract some
        # useful value to serve for the next node. So we cannot use [] to represent the
        # shape of the output. In another word, for explicit shape, we do not need to
        # consider the scalar shape, and we treat it as a value.

    _store_explicit_shape(e_shape, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_gemm(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    transA = attrs["transA"]
    transB = attrs["transB"]
    shape1, _ = _get_shape(node.input[0], shapes, initializers, explicit_shapes, False)
    shape2, _ = _get_shape(node.input[1], shapes, initializers, explicit_shapes, False)

    if shape1 == [0] or shape2 == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if transA:
        shape1 = shape1[:-2] + [shape1[-1], shape1[-2]]
    if transB:
        shape2 = shape2[:-2] + [shape2[-1], shape2[-2]]

    if len(shape1) == 0:
        # The first input is a scalar
        assert shape2[0] == 1
        shape = shape2
    else:
        shape = shape1[:-1] + shape2[-1:]

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_matmul(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape1, _ = _get_shape(node.input[0], shapes, initializers, explicit_shapes, False)
    shape2, _ = _get_shape(node.input[1], shapes, initializers, explicit_shapes, False)

    if shape1 == [0] or shape2 == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if len(shape1) == 2 and len(shape2) == 1:
        shape = [shape1[0]]
    elif len(shape1) == 1 and len(shape2) == 2:
        shape = [shape2[-1]]
    elif len(shape1) == 1 and len(shape2) == 1:
        shape = []
    elif len(shape1) >= len(shape2):
        shape = [*shape1[:-1], shape2[-1]]
    else:
        raise RuntimeError(
            f"Cannot matmul {node.op_type} with shape {shape1} and {shape2}."
        )

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_pool(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    ceil_mode = attrs.get("ceil_mode", False)
    dim = len(kernel_shape)
    assert len(dilations) == dim
    assert len(pads) == dim * 2
    assert len(strides) == dim

    input_shape = _get_data_shape(node.input[0], shapes)

    if input_shape == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    # Get the output channel
    if len(node.input) > 1:
        # For Conv node, output channel is the first dim of weight.
        weight_shape, _ = _get_shape(
            node.input[1], shapes, initializers, explicit_shapes, False
        )
        output_channel = weight_shape[0]
    else:
        # For MaxPool or AvgPool node, output channel is the same as input.
        output_channel = input_shape[1]  # Remove batch dim

    # Calculate the output size
    dim = len(kernel_shape)
    output_hw = [0] * dim
    for i in range(dim):
        temp1 = pads[i] + pads[i + dim]
        temp2 = dilations[i] * (kernel_shape[i] - 1)
        output_hw[i] = (input_shape[i + 2] + temp1 - temp2 - 1) / strides[i] + 1
        output_hw[i] = ceil(output_hw[i]) if ceil_mode else floor(output_hw[i])
    shape = [input_shape[0], output_channel]  # (Batch, Channel)
    for i in range(dim):
        shape.append(output_hw[i])

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_pad(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    input_shape, is_explicit = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    if input_shape == [0]:
        shape = [0]
        if is_explicit:
            _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        else:
            _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    pads = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()
    if len(node.input) == 4:
        axes = onnx.numpy_helper.to_array(initializers[node.input[2]]).tolist()
        raise NotImplementedError(f"Pad with axes={axes} is not supported.")

    dim = int(len(pads) / 2)
    pads = pads[:dim] + pads[dim:]
    shape = [s + p for s, p in zip(input_shape, pads)]

    if is_explicit:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_range(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    start = _get_explicit_shape(node.input[0], initializers, explicit_shapes, False)
    limit = _get_explicit_shape(node.input[1], initializers, explicit_shapes, False)
    delta = _get_explicit_shape(node.input[2], initializers, explicit_shapes, False)

    if type(start) is not int or type(limit) is not int or type(delta) is not int:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if delta > 0:
        length = max(0, (limit - start + delta - 1) // delta)
    elif delta < 0:
        length = max(0, (start - limit - delta - 1) // (-delta))
    else:
        raise RuntimeError("The step delta of range is 0.")

    shape = [length]
    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_reduce(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    keepdims = get_onnx_attrs(node, initializers)["keepdims"]
    axes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()

    shape = _get_data_shape(node.input[0], shapes, False)
    if shape != [0]:
        for axis in axes:
            shape[axis] = 1 if keepdims else 0
        # Remove all 0 in shape and not keep dims
        shape = [x for x in shape if x != 0]

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_reshape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    data_shape, _ = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    shape = _get_explicit_shape(node.input[1], initializers, explicit_shapes)
    if shape is None or shape == [0] or (data_shape == [0] and -1 in shape):
        # This is a dynamic reshape
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    def _infer_reshape_shape(ori_shape: list[int], new_shape: list[int]) -> list[int]:
        """
        Infers the shape of an array after reshaping, without performing actual
        computation.

        :param ori_shape: Original shape of the array
        :param new_shape: New shape of the array
        :return: The reshaped array
        """
        inferred_shape = math.prod(ori_shape)
        inferred_idx = -1
        for idx, new_shape_i in enumerate(new_shape):
            if new_shape_i == -1:
                inferred_idx = idx
                continue
            inferred_shape = inferred_shape / new_shape_i
        if inferred_idx != -1:
            new_shape[inferred_idx] = int(inferred_shape)
        return new_shape

    shape = _infer_reshape_shape(data_shape, shape)
    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_resize(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    align_mode = attrs["coordinate_transformation_mode"]
    mode = attrs["mode"]
    nearest_mode = attrs.get("nearest_mode")

    if mode != "nearest":
        raise NotImplementedError(f"Resize with mode={mode} is not supported.")

    input_shape = _get_data_shape(node.input[0], shapes, False)
    if input_shape == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if nearest_mode == "floor":
        op_round = floor
    elif nearest_mode == "ceil":
        op_round = ceil
    elif nearest_mode == "round_prefer_floor":
        op_round = lambda x: int(x + 0.4999999)
    elif nearest_mode == "round_prefer_ceil":
        op_round = lambda x: int(x + 0.5000001)
    else:
        raise NotImplementedError(
            f"Resize with nearest_mode={nearest_mode} is not supported."
        )

    scales = onnx.numpy_helper.to_array(initializers[node.input[2]]).tolist()
    if len(scales) == 0:
        raise RuntimeError(f"Resize with scales={scales} is not supported.")

    if align_mode in {"asymmetric", "half_pixel"}:
        shape = [op_round(dim * scale) for dim, scale in zip(input_shape, scales)]
    else:
        raise NotImplementedError(
            f"Resize with align_mode={align_mode} is not supported."
        )

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_shape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape, is_explicit = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    if not is_explicit:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        return

    if type(shape) is int:
        shape = []
    else:
        if shape != [0]:  # The shape of a shape tuple is a 1d list of int.
            shape = [1, len(shape)]
        else:  # This is a dynamic shape
            shape = [0]

    _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_slice(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    if any(name not in initializers for name in node.input[1:]):
        # This is a dynamic slice
        shape = [0]
        if node.input[0] in shapes:
            _store_data_shape(shape, shapes, node.op_type, node.output[0])
        else:
            _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        return

    starts = initializers[node.input[1]]
    ends = initializers[node.input[2]]
    starts = onnx.numpy_helper.to_array(starts).tolist()
    ends = onnx.numpy_helper.to_array(ends).tolist()

    n_inputs = len(node.input)
    if n_inputs > 3:
        axes = initializers[node.input[3]]
        axes = onnx.numpy_helper.to_array(axes).tolist()
    else:
        axes = list(range(len(starts)))
    if n_inputs > 4:
        steps = initializers[node.input[4]]
        steps = onnx.numpy_helper.to_array(steps).tolist()
    else:
        steps = [1] * len(axes)

    # There are two cases:
    # (1) slice a shape, e.g., shape=[1, 2, 3, 4], shape[0:2:1]=[1, 2]
    # (2) slice a tensor value

    def _infer_sliced_shape(
        shape_: list[int],
        axes_: list[int],
        starts_: list[int],
        ends_: list[int],
        steps_: list[int],
    ) -> list[int]:
        new_shape = list(shape_)
        for axis, start, end, step in zip(axes_, starts_, ends_, steps_):
            size = shape_[axis]
            # Handle negative indices
            start = min(max(start + size if start < 0 else start, 0), size)
            end = min(max(end + size if end < 0 else end, 0), size)
            if step < 0:
                warnings.warn(f"Negative step ({step}) is not tested.")
            # Compute the new dimension size
            new_shape[axis] = max(
                0, (end - start + (step - (1 if step > 0 else -1))) // step
            )
        return new_shape

    # slice a tensor and we need its new shape
    shape = _get_data_shape(node.input[0], shapes)
    if shape is not None:
        shape = (
            _infer_sliced_shape(shape, axes, starts, ends, steps)
            if shape != [0]
            else [0]
        )
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    # slice a shape (extracted by shape node) and we need the sliced shape
    shape = _get_explicit_shape(node.input[0], initializers, explicit_shapes, False)
    # Commonly, the shape is a 1d list of int.
    # In such case, the node aims to extract a dimension on the specified axis.
    # e.g. I want to know the shape of the first dimension of the input tensor
    assert axes == [0], f"Invalid axis {axes}"
    shape = shape[starts[0] : ends[0] : steps[0]] if shape != [0] else [0]
    _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_split(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape = _get_data_shape(node.input[0], shapes, False)
    if shape == [0]:
        # This is a dynamic shape.
        for output_name in node.output:
            _store_data_shape([0], shapes, node.op_type, output_name)
        return

    attrs = get_onnx_attrs(node, initializers)
    axis = attrs["axis"]
    if attrs["num_outputs"] is not None:
        raise NotImplementedError(
            f"Split node with num_outputs={attrs['num_outputs']} has not be supported."
        )
    if node.input[1] not in initializers:
        raise RuntimeError(
            f"Split node with input[1]={node.input[1]} is not supported."
        )

    split_sizes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()
    shape = _get_data_shape(node.input[0], shapes)

    output_shapes = []
    if axis < 0:
        axis += len(shape)
    for split_size in split_sizes:
        output_shape = shape[:axis] + [split_size] + shape[axis + 1 :]
        output_shapes.append(output_shape)

    for output_name, output_shape in zip(node.output, output_shapes):
        _store_data_shape(output_shape, shapes, node.op_type, output_name)


def _infer_shape_of_squeeze(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    input_shape, _ = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes, False
    )
    if input_shape == [0]:
        # This is a dynamic shape.
        shape = [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    if len(node.input) > 1:
        axes = _get_explicit_shape(node.input[1], initializers, explicit_shapes)
        if axes is None:
            axes = [i for i in range(len(input_shape)) if input_shape[i] == 1]

        shape = []
        for i in range(len(input_shape)):
            if i in axes:
                assert (
                    input_shape[i] == 1
                ), f"Invalid axis {i} for squeeze {input_shape}"
                continue
            shape.append(input_shape[i])
    else:
        # Squeeze all 1 dims
        shape = [s for s in input_shape if s != 1]
        if len(shape) != 0:
            # Add batch dim
            shape = [input_shape[0]] + shape

    _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_transpose(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    attrs = get_onnx_attrs(node, initializers)
    perm = attrs["perm"]

    # There are two cases:
    # (1) Transpose an initializer
    # (2) Transpose a tensor value
    shape = _get_data_shape(node.input[0], shapes, False)
    if shape is not None:
        if len(shape) == 1:  # The input is a vector
            shape = [shape[0], 1]
        else:
            shape = [shape[i] for i in perm] if shape != [0] else [0]
        _store_data_shape(shape, shapes, node.op_type, node.output[0])
        return

    shape = _get_explicit_shape(node.input[0], initializers, explicit_shapes, False)
    if len(shape) == 1:  # The input is a vector
        shape = [shape[0], 1]
    else:
        shape = [shape[i] for i in perm] if shape != [0] else [0]
    _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])


def _infer_shape_of_unsqueeze(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape, is_explicit = _get_shape(
        node.input[0], shapes, initializers, explicit_shapes
    )

    axes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()

    # If the data_shape is a single int, it means that the input is a scalar, and we
    # want to expand it to a shape of [1]. This happens after a gather node.
    if type(shape) is int:
        assert axes == [0], f"Invalid axis {axes}"
        shape = [shape]
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
        return

    def _infer_unsqueeze_shape(ori_shape_: list[int], axes_: list[int]) -> list[int]:
        new_shape = list(ori_shape_)
        for axis in sorted(axes_, reverse=True):
            if axis < 0:
                axis += len(ori_shape_) + 1
            new_shape.insert(axis, 1)
        return new_shape

    if shape != [0]:
        shape = _infer_unsqueeze_shape(shape, axes)

    if is_explicit:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])


def _infer_shape_of_where(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, int | list[int]],
):
    shape1, is_e = _get_shape(
        node.input[1], shapes, initializers, explicit_shapes, False
    )
    shape2, is_e = _get_shape(
        node.input[2], shapes, initializers, explicit_shapes, False
    )
    if shape1 != [0]:
        shape = shape1
    elif shape2 != [0]:
        shape = shape2
    else:
        shape = [0]
    if is_e:
        _store_explicit_shape(shape, explicit_shapes, node.op_type, node.output[0])
    else:
        _store_data_shape(shape, shapes, node.op_type, node.output[0])

        # There maybe an explicit shape calculation at the same time
        condition = _get_explicit_shape(
            node.input[0], initializers, explicit_shapes, False
        )
        value1 = _get_explicit_shape(
            node.input[1], initializers, explicit_shapes, False
        )
        value2 = _get_explicit_shape(
            node.input[2], initializers, explicit_shapes, False
        )

        new_shape = value1.copy()
        for i in range(len(condition)):
            if condition[i] == 0:
                new_shape[i] = value2[i]
        _store_explicit_shape(new_shape, explicit_shapes, node.op_type, node.output[0])


INFER_SHAPE_FUNC_MAPPING = {
    "Add": _infer_shape_of_binary_op,
    "ArgMax": _infer_shape_of_argmax,
    "AveragePool": _infer_shape_of_pool,
    "BatchNormalization": _infer_shape_of_batch_norm,
    "Cast": _infer_shape_of_nochange_op,
    "Clip": _infer_shape_of_nochange_op,
    "Concat": _infer_shape_of_concat,
    "ConstantOfShape": _infer_shape_of_constant_of_shape,
    "Conv": _infer_shape_of_pool,
    "ConvTranspose": _infer_shape_of_convtranspose,
    "Cos": _infer_shape_of_nochange_op,
    "Div": _infer_shape_of_binary_op,
    "Dropout": _infer_shape_of_nochange_op,
    "Equal": _infer_shape_of_binary_op,
    "Expand": _infer_shape_of_expand,
    "Flatten": _infer_shape_of_flatten,
    "Floor": _infer_shape_of_nochange_op,
    "Gather": _infer_shape_of_gather,
    "Gemm": _infer_shape_of_gemm,
    "LeakyRelu": _infer_shape_of_nochange_op,
    "MatMul": _infer_shape_of_matmul,
    "Max": _infer_shape_of_nochange_op,
    "MaxPool": _infer_shape_of_pool,
    "Min": _infer_shape_of_nochange_op,
    "Mul": _infer_shape_of_binary_op,
    "Neg": _infer_shape_of_nochange_op,
    "Pad": _infer_shape_of_pad,
    "Pow": _infer_shape_of_nochange_op,
    "Range": _infer_shape_of_range,
    "ReduceMean": _infer_shape_of_reduce,
    "ReduceSum": _infer_shape_of_reduce,
    "Relu": _infer_shape_of_nochange_op,
    "Reshape": _infer_shape_of_reshape,
    "Resize": _infer_shape_of_resize,
    "Scatter": _infer_shape_of_nochange_op,
    "ScatterElements": _infer_shape_of_nochange_op,
    "ScatterND": _infer_shape_of_nochange_op,
    "Shape": _infer_shape_of_shape,
    "Sigmoid": _infer_shape_of_nochange_op,
    "Sin": _infer_shape_of_nochange_op,
    "Slice": _infer_shape_of_slice,
    "Split": _infer_shape_of_split,
    "Softmax": _infer_shape_of_nochange_op,
    "Squeeze": _infer_shape_of_squeeze,
    "Sub": _infer_shape_of_binary_op,
    "Tanh": _infer_shape_of_nochange_op,
    "Transpose": _infer_shape_of_transpose,
    "Unsqueeze": _infer_shape_of_unsqueeze,
    "Where": _infer_shape_of_where,
}


def infer_onnx_shape(
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    verbose: bool = False,
) -> dict[str, list[int]]:
    global _VERBOSE
    _VERBOSE = verbose
    """
    Two kinds of shapes in the model:
    1. Explicit shape: The shape is extracted by some nodes (e.g. Shape). The shape 
        itself is the data of the node.
    2. Shapes: The shape is inferred from the node and the node does not show 
       the shape. The shape is not the data of the node. 
    """
    data_shapes = {}
    explicit_shapes = {}

    if verbose:
        print("Inferring shape of input(s)...")
    _get_input_shape(input_nodes, data_shapes)

    if verbose:
        print("Inferring shape of output(s)...")
    _get_output_shape(output_nodes, data_shapes)

    if verbose:
        print("Inferring shape of initializer(s)...")
    _get_initializer_shape(initializers, data_shapes)

    if verbose:
        print("Inferring shape of node(s)...")
    for node in nodes:
        if verbose:
            print(node.name)
        op_type = node.op_type
        if op_type == "Constant":
            raise RuntimeError(
                "There are constant nodes in the model, and you should convert them to "
                "initializers first by argument constant_to_initializer=True."
            )
        _infer_shape = INFER_SHAPE_FUNC_MAPPING[op_type]
        _infer_shape(node, initializers, data_shapes, explicit_shapes)

    data_shapes.update(explicit_shapes)

    return data_shapes
