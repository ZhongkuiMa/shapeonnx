"""Shared fixtures for unit tests."""

import onnx
import pytest

from shapeonnx.infer_shape import ShapeInferenceContext


@pytest.fixture
def empty_context():
    """Create empty shape inference context for testing."""
    return ShapeInferenceContext(
        data_shapes={},
        explicit_shapes={},
        initializers={},
        verbose=False,
    )


@pytest.fixture
def simple_node():
    """Create ONNX nodes."""

    def _make_node(op_type, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op_type, inputs=inputs, outputs=outputs, **kwargs)

    return _make_node
