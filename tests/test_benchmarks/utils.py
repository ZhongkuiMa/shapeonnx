"""Utility functions for ShapeONNX testing."""

__docformat__ = "restructuredtext"
__all__ = [
    "check_shape_compatibility",
    "find_benchmarks_folders",
    "get_benchmark_name",
    "get_onnx_files_from_instances",
    "if_has_batch_dim",
    "infer_shape",
    "load_onnx_model",
]

import csv
from pathlib import Path

import onnx

BENCHMARK_WITHOUT_BATCH_DIM = [
    "cctsdb_yolo",
    "cgan",
    "pensieve_big_parallel.onnx",
    "pensieve_mid_parallel.onnx",
    "pensieve_small_parallel.onnx",
    "test_nano.onnx",
    "test_small.onnx",
    "test_tiny.onnx",
]


def find_benchmarks_folders(base_dir: str) -> list[str]:
    """
    Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    # Make path relative to this test file's location
    test_dir = Path(__file__).parent
    base_path = test_dir / base_dir

    # If base_path is a file (containing a path), read it and resolve
    if base_path.is_file():
        with base_path.open() as f:
            path_str = f.read().strip()
        base_path = (test_dir / path_str).resolve()

    return [str(entry) for entry in base_path.iterdir() if entry.is_dir()]


def get_onnx_files_from_instances(
    benchmark_dir: str, max_instances: int | None = None
) -> list[str]:
    """
    Get unique ONNX file paths from instances.csv in a benchmark directory.

    :param benchmark_dir: Benchmark directory path
    :param max_instances: Maximum unique instances to load (None = all)
    :return: List of absolute ONNX file paths (deduplicated)
    """
    instances_csv = Path(benchmark_dir) / "instances.csv"
    if not instances_csv.exists():
        return []

    onnx_files = []
    seen_onnx = set()

    with instances_csv.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                onnx_rel_path = row[0]
                if onnx_rel_path not in seen_onnx:
                    seen_onnx.add(onnx_rel_path)
                    onnx_abs_path = Path(benchmark_dir) / onnx_rel_path
                    if onnx_abs_path.exists():
                        onnx_files.append(str(onnx_abs_path))
                        if max_instances and len(onnx_files) >= max_instances:
                            break

    return onnx_files


def get_all_onnx_files(benchmark_dirs: list[str], max_per_benchmark: int | None = 20) -> list[str]:
    """
    Get all ONNX files from benchmark directories using instances.csv.

    :param benchmark_dirs: List of benchmark directory paths
    :param max_per_benchmark: Maximum ONNX files per benchmark
    :return: List of ONNX file paths
    """
    onnx_files = []
    for bdir in benchmark_dirs:
        onnx_files.extend(get_onnx_files_from_instances(bdir, max_per_benchmark))
    return onnx_files


def get_benchmark_name(onnx_path: str, benchmarks_dir: str = "benchmarks") -> str:
    """
    Extract benchmark name from ONNX file path.

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Benchmark name
    """
    path = Path(onnx_path)
    path_parts = path.parts

    try:
        bench_idx = path_parts.index(benchmarks_dir)
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    return path.parent.name


def load_onnx_model(onnx_path: str):
    """
    Load ONNX model and convert to version 21.

    :param onnx_path: Path to ONNX model file
    :return: ONNX ModelProto converted to version 21
    """
    model = onnx.load(onnx_path)
    return onnx.version_converter.convert_version(model, target_version=21)


def if_has_batch_dim(onnx_path: str) -> bool:
    """
    Determine if model has batch dimension.

    :param onnx_path: Path to ONNX model file
    :return: True if model has batch dimension, False otherwise
    """
    return all(bname not in onnx_path for bname in BENCHMARK_WITHOUT_BATCH_DIM)


def check_shape_compatibility(inferred_shape: int | list[int], expected_shape: list[int]) -> bool:
    """
    Check if inferred shape is compatible with expected shape.

    :param inferred_shape: Shape inferred by shape inference
    :param expected_shape: Expected shape from model metadata
    :return: True if shapes are compatible, False otherwise
    """
    if inferred_shape == expected_shape:
        return True
    if inferred_shape == [] and expected_shape == [1]:
        return True
    # [0] is used as a marker for empty/scalar shapes in some contexts
    if inferred_shape == [0] and expected_shape == []:
        return True
    return inferred_shape == [] and expected_shape == [0]


def infer_shape(
    model, has_batch_dim: bool = True, verbose: bool = False
) -> dict[str, int | list[int]]:
    """
    Run shape inference on model and validate against expected shapes.

    :param model: ONNX ModelProto
    :param has_batch_dim: Whether model has batch dimension
    :param verbose: Whether to print verbose output during inference
    :return: Dictionary mapping tensor names to inferred shapes
    """
    from shapeonnx import extract_io_shapes, infer_onnx_shape
    from shapeonnx.utils import (
        convert_constant_to_initializer,
        get_initializers,
        get_input_nodes,
        get_output_nodes,
    )

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)
    nodes = list(model.graph.node)
    nodes = convert_constant_to_initializer(nodes, initializers)

    data_shapes = infer_onnx_shape(
        input_nodes, output_nodes, nodes, initializers, has_batch_dim, verbose
    )

    expected_input_shapes = extract_io_shapes(input_nodes, has_batch_dim)
    expected_output_shapes = extract_io_shapes(output_nodes, has_batch_dim)

    for input_node in input_nodes:
        input_name = input_node.name
        shape = data_shapes[input_name]
        expected_shape = expected_input_shapes[input_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Input shape mismatch for '{input_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

    for output_node in output_nodes:
        output_name = output_node.name
        shape = data_shapes[output_name]
        expected_shape = expected_output_shapes[output_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Output shape mismatch for '{output_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

    return data_shapes
