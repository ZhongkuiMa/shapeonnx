"""VNNComp Benchmark Test Runner for ShapeONNX."""

__docformat__ = "restructuredtext"
__all__ = []

import pytest

from tests.test_benchmarks.utils import (
    find_benchmarks_folders,
    get_all_onnx_files,
    if_has_batch_dim,
    infer_shape,
    load_onnx_model,
)


def get_onnx_models():
    """
    Collect all ONNX models from vnncomp2024 benchmarks.

    :return: List of ONNX file paths or empty list if benchmarks not found
    """
    from pathlib import Path

    dir_name = "vnncomp2024_benchmarks"
    benchmarks_path = Path(__file__).parent / dir_name

    # Skip if benchmarks directory doesn't exist
    if not benchmarks_path.exists():
        return []

    try:
        benchmark_dirs = find_benchmarks_folders(dir_name)
        return get_all_onnx_files(benchmark_dirs)
    except (FileNotFoundError, OSError):
        return []


@pytest.mark.benchmark
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
