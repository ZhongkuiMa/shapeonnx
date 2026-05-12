"""Shared fixtures for shapeonnx unit tests."""

__docformat__ = "restructuredtext"

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext


@pytest.fixture
def empty_context() -> ShapeInferenceContext:
    """Create an empty ShapeInferenceContext.

    :return: ShapeInferenceContext with no shapes or initializers
    """
    return ShapeInferenceContext(
        data_shapes={},
        explicit_shapes={},
        initializers={},
        verbose=False,
    )


@pytest.fixture
def make_context() -> Callable[..., ShapeInferenceContext]:
    """Build ShapeInferenceContext instances with arbitrary shapes and initializers.

    Replaces the per-test inline ``ShapeInferenceContext(data_shapes=..., ...)``
    boilerplate. Defaults are empty dicts and ``verbose=False``.

    :return: callable returning a populated ShapeInferenceContext
    """

    def _make(
        data_shapes: dict[str, int | list[int]] | None = None,
        explicit_shapes: dict[str, int | list[int]] | None = None,
        initializers: dict[str, onnx.TensorProto] | None = None,
        verbose: bool = False,
    ) -> ShapeInferenceContext:
        return ShapeInferenceContext(
            data_shapes=data_shapes or {},
            explicit_shapes=explicit_shapes or {},
            initializers=initializers or {},
            verbose=verbose,
        )

    return _make


@pytest.fixture
def simple_node() -> Callable[..., onnx.NodeProto]:
    """Build ONNX nodes via ``onnx.helper.make_node``.

    :return: callable forwarding (op_type, inputs, outputs, **attrs) to make_node
    """

    def _make_node(
        op_type: str, inputs: list[str], outputs: list[str], **kwargs: Any
    ) -> onnx.NodeProto:
        return onnx.helper.make_node(op_type, inputs=inputs, outputs=outputs, **kwargs)

    return _make_node


@pytest.fixture
def make_weight_tensor() -> Callable[..., onnx.TensorProto]:
    """Build Conv/Gemm weight initializer tensors with random standard-normal values.

    Replaces the duplicated ``_make_weight_tensor`` helper currently inlined in
    test_conv.py, test_main_api.py, and test_onnx_attrs.py.

    :return: callable taking (shape, name="weight") and returning a float32 TensorProto
    """

    def _make(shape: tuple[int, ...], name: str = "weight") -> onnx.TensorProto:
        rng = np.random.default_rng()
        array = rng.standard_normal(shape).astype(np.float32)
        return onnx.numpy_helper.from_array(array, name=name)

    return _make


@pytest.fixture
def make_axes_initializer() -> Callable[..., onnx.TensorProto]:
    """Build axes initializer tensors used by reduce/split/etc.

    :return: callable taking (axes, name="axes") and returning an int64 TensorProto
    """

    def _make(axes: list[int], name: str = "axes") -> onnx.TensorProto:
        array = np.array(axes, dtype=np.int64)
        return onnx.numpy_helper.from_array(array, name=name)

    return _make
