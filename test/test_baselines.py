"""Baseline management for ShapeONNX regression testing.

This module provides functions to create and compare shape inference baselines.
Each ONNX model has its own JSON baseline file containing inferred shapes.

Usage::

    # Create/update baseline for one model
    update_baseline("path/to/model.onnx")

    # Compare one model against baseline
    compare_baseline("path/to/model.onnx")

    # Batch: Create baselines for all VNNComp benchmarks
    update_all_benchmarks()
"""

import json
import os
import time

from shapeonnx import infer_onnx_shape
from shapeonnx.shapeonnx.utils import (
    get_initializers,
    get_input_nodes,
    get_output_nodes,
    convert_constant_to_initializer,
)
from utils import (
    load_onnx_model,
    if_has_batch_dim,
    find_benchmarks_folders,
    find_all_onnx_files,
    get_benchmark_name,
)


def get_baseline_path(onnx_path: str, baselines_dir: str = "baselines") -> str:
    """Get baseline JSON path for an ONNX model with benchmark subdirectory.

    The baseline file is stored in a subdirectory matching the benchmark name.
    For example: benchmarks/acasxu_2023/model.onnx -> baselines/acasxu_2023/model.json

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Root directory to store baseline files
    :return: Path to baseline JSON file
    """
    # Extract benchmark name from path
    benchmark_name = get_benchmark_name(onnx_path)

    # Get model basename without .onnx extension
    basename = os.path.basename(onnx_path)
    if basename.endswith(".onnx"):
        basename = basename[:-5]

    # Create path with benchmark subdirectory
    return os.path.join(baselines_dir, benchmark_name, f"{basename}.json")


def load_baseline_shapes(baseline_path: str):
    """Load shapes from baseline JSON file.

    :param baseline_path: Path to baseline JSON file
    :return: Dictionary of shapes, or None if file not found
    """
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path, "r") as f:
        return json.load(f)


def save_baseline_shapes(shapes: dict, baseline_path: str):
    """Save shapes dict to baseline JSON file.

    :param shapes: Dictionary mapping tensor names to shapes
    :param baseline_path: Path to baseline JSON file
    """
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(shapes, f, indent=2)


def update_baseline(onnx_path: str, baselines_dir: str = "baselines"):
    """Create or update baseline for ONE ONNX model.

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Root directory to store baseline files
    :return: Dictionary of inferred shapes
    """
    has_batch_dim = if_has_batch_dim(onnx_path)
    model = load_onnx_model(onnx_path)

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)
    nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

    shapes = infer_onnx_shape(
        input_nodes,
        output_nodes,
        nodes,
        initializers,
        has_batch_dim=has_batch_dim,
        verbose=False,
    )

    baseline_path = get_baseline_path(onnx_path, baselines_dir)
    save_baseline_shapes(shapes, baseline_path)

    benchmark_name = get_benchmark_name(onnx_path)
    print(
        f"[{benchmark_name}] Saved {os.path.basename(onnx_path)} ({len(shapes)} shapes)"
    )

    return shapes


def compare_baseline(onnx_path: str, baselines_dir: str = "baselines") -> bool:
    """Compare ONE ONNX model's shapes against its baseline.

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Root directory containing baseline files
    :return: True if shapes match baseline, False otherwise
    """
    baseline_path = get_baseline_path(onnx_path, baselines_dir)
    baseline_shapes = load_baseline_shapes(baseline_path)

    if baseline_shapes is None:
        print(f"No baseline: {os.path.basename(onnx_path)}")
        return False

    has_batch_dim = if_has_batch_dim(onnx_path)
    model = load_onnx_model(onnx_path)

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)
    nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

    current_shapes = infer_onnx_shape(
        input_nodes,
        output_nodes,
        nodes,
        initializers,
        has_batch_dim=has_batch_dim,
        verbose=False,
    )

    if current_shapes == baseline_shapes:
        print(f"OK: {os.path.basename(onnx_path)}")
        return True

    print(f"MISMATCH: {os.path.basename(onnx_path)}")
    all_keys = set(current_shapes.keys()) | set(baseline_shapes.keys())
    for key in sorted(all_keys):
        if key not in current_shapes:
            print(f"  Missing: {key}")
        elif key not in baseline_shapes:
            print(f"  New: {key} = {current_shapes[key]}")
        elif current_shapes[key] != baseline_shapes[key]:
            print(f"  {key}: {baseline_shapes[key]} -> {current_shapes[key]}")

    return False


def update_all_benchmarks(
    benchmark_dir: str,
    baselines_dir: str,
    max_per_benchmark: int = 20,
):
    """Helper to create/update baselines for all VNNComp models.

    :param benchmark_dir: Root directory of benchmarks
    :param baselines_dir: Root directory to store baseline files
    :param max_per_benchmark: Maximum models per benchmark to process
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Creating baselines for {len(onnx_files)} models")

    success = 0
    failed = []
    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        print(f"[{i}/{len(onnx_files)}] ", end="")
        try:
            update_baseline(onnx_path, baselines_dir)
            success += 1
        except Exception as e:
            print(f"Error: {e}")
            failed.append(onnx_path)

    total_time = time.perf_counter() - start_time

    print(f"\nCompleted: {success}/{len(onnx_files)} success, {len(failed)} failed")
    if failed:
        print("Failed models:")
        for f in failed:
            print(f"  {os.path.basename(f)}")
    print(
        f"Total time: {total_time:.2f}s (avg {total_time/len(onnx_files):.2f}s/model)"
    )


def verify_all_benchmarks(
    benchmark_dir: str,
    baselines_dir: str,
    max_per_benchmark: int = 20,
):
    """Helper to verify all VNNComp models against baselines.

    :param benchmark_dir: Root directory of benchmarks
    :param baselines_dir: Root directory containing baseline files
    :param max_per_benchmark: Maximum models per benchmark to verify
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Verifying {len(onnx_files)} models")

    passed = 0
    failed = []
    missing = []
    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        print(f"[{i}/{len(onnx_files)}] ", end="")
        try:
            baseline_path = get_baseline_path(onnx_path, baselines_dir)
            if not os.path.exists(baseline_path):
                print(f"Skip {os.path.basename(onnx_path)} - no baseline")
                missing.append(onnx_path)
                continue

            if compare_baseline(onnx_path, baselines_dir):
                passed += 1
            else:
                failed.append(onnx_path)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(onnx_path)

    total_time = time.perf_counter() - start_time
    tested = len(onnx_files) - len(missing)

    print(f"\nTested: {tested}/{len(onnx_files)}")
    print(f"Passed: {passed}/{tested}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {os.path.basename(f)}")
    if missing:
        print(f"Missing baselines: {len(missing)}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    # Example 1: Update baseline for one model
    # model = "../../../vnncomp2024_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx"
    # update_baseline(model, baselines_dir="baselines")

    # Example 2: Verify one model against baseline
    # success = compare_baseline(model, baselines_dir="baselines")

    # Example 3: Create baselines for all VNNComp benchmarks
    # update_all_benchmarks(
    #     benchmark_dir="benchmarks",
    #     baselines_dir="baselines",
    #     max_per_benchmark=20,
    # )

    # Example 4: Verify all benchmarks against baselines
    verify_all_benchmarks(
        benchmark_dir="benchmarks",
        baselines_dir="baselines",
        max_per_benchmark=20,
    )
