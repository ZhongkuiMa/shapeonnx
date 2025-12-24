"""VNNComp Benchmark Test Runner for ShapeONNX."""

__docformat__ = "restructuredtext"
__all__ = []

import pytest
from shapeonnx.tests.utils import (
    find_benchmarks_folders,
    get_all_onnx_files,
    if_has_batch_dim,
    infer_shape,
    load_onnx_model,
)


def get_onnx_models():
    """
    Collect all ONNX models from vnncomp2024 benchmarks.

    :return: List of ONNX file paths
    """
    dir_name = "vnncomp2024_benchmarks"
    benchmark_dirs = find_benchmarks_folders(dir_name)
    return get_all_onnx_files(benchmark_dirs)


@pytest.mark.parametrize("onnx_path", get_onnx_models())
def test_shape_inference(onnx_path):
    """
    Test shape inference on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    """
    model = load_onnx_model(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    # This will raise an exception if shape inference fails
    data_shapes = infer_shape(model, has_batch_dim, verbose=False)

    # Verify we got some shapes back
    assert data_shapes is not None
    assert len(data_shapes) > 0
