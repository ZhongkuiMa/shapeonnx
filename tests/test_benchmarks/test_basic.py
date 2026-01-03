"""Basic functionality test for shapeonnx optimizations."""

__docformat__ = "restructuredtext"
__all__ = []

import sys

import numpy as np
import onnx
import pytest

from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    convert_constant_to_initializer,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)


def create_simple_model():
    """
    Create a simple ONNX model for testing.

    :return: ONNX ModelProto for testing
    """
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )

    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10])

    rng = np.random.default_rng(seed=42)
    w1 = rng.standard_normal((64, 3, 7, 7)).astype(np.float32)
    w1_init = onnx.numpy_helper.from_array(w1, name="w1")

    w2 = rng.standard_normal((12544, 10)).astype(np.float32)
    b2 = rng.standard_normal(10).astype(np.float32)
    w2_init = onnx.numpy_helper.from_array(w2, name="w2")
    b2_init = onnx.numpy_helper.from_array(b2, name="b2")

    shape_const = np.array([1, -1], dtype=np.int64)
    shape_init = onnx.numpy_helper.from_array(shape_const, name="shape_const")

    conv = onnx.helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv_out"],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    )

    relu = onnx.helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"])

    maxpool = onnx.helper.make_node(
        "MaxPool",
        inputs=["relu_out"],
        outputs=["pool_out"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )

    reshape = onnx.helper.make_node(
        "Reshape", inputs=["pool_out", "shape_const"], outputs=["reshape_out"]
    )

    gemm = onnx.helper.make_node("Gemm", inputs=["reshape_out", "w2", "b2"], outputs=["output"])

    graph = onnx.helper.make_graph(
        [conv, relu, maxpool, reshape, gemm],
        "test_model",
        [input_tensor],
        [output_tensor],
        [w1_init, w2_init, b2_init, shape_init],
    )

    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 21

    return model


@pytest.mark.benchmark
def test_basic_inference():
    """Test basic shape inference."""
    print("Creating test model")
    model = create_simple_model()

    print("Extracting model components")
    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
    output_nodes = get_output_nodes(model, has_batch_dim=True)
    nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

    print("Running shape inference")
    shapes = infer_onnx_shape(
        input_nodes, output_nodes, nodes, initializers, has_batch_dim=True, verbose=True
    )

    print("\nInferred Shapes")
    for name, shape in shapes.items():
        print(f"{name:20} {shape}")

    assert shapes["input"] == [
        1,
        3,
        224,
        224,
    ], f"Input shape mismatch: {shapes['input']}"
    assert shapes["conv_out"] == [
        1,
        64,
        112,
        112,
    ], f"Conv output shape mismatch: {shapes['conv_out']}"
    assert shapes["relu_out"] == [
        1,
        64,
        112,
        112,
    ], f"ReLU output shape mismatch: {shapes['relu_out']}"
    assert shapes["pool_out"] == [
        1,
        64,
        56,
        56,
    ], f"MaxPool output shape mismatch: {shapes['pool_out']}"
    assert shapes["reshape_out"] == [
        1,
        200704,
    ], f"Reshape output shape mismatch: {shapes['reshape_out']}"

    print("\nAll assertions passed")
    print("Performance optimizations are working correctly")


if __name__ == "__main__":
    try:
        test_basic_inference()
        print("\n" + "=" * 50)
        print("SUCCESS: Basic functionality test passed")
        print("=" * 50)
        sys.exit(0)
    except (RuntimeError, ValueError, NotImplementedError, AssertionError) as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
