"""VNNComp Benchmark Test Runner for ShapeONNX."""

__docformat__ = "restructuredtext"
__all__ = []

import os
import time

from utils import (
    find_benchmarks_folders,
    get_all_onnx_files,
    if_has_batch_dim,
    infer_shape,
    load_onnx_model,
)

if __name__ == "__main__":
    dir_name = "benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collected {len(benchmark_dirs)} benchmark directories")
    onnx_paths = get_all_onnx_files(benchmark_dirs)
    print(f"Collected {len(onnx_paths)} ONNX models")

    failed_onnx_paths = []
    verbose = False

    success_count = 0
    total_count = 0
    for i, onnx_path in enumerate(onnx_paths):
        print(f"[{i}/{len(onnx_paths)}] ", end="")
        time_start = time.perf_counter()
        model = load_onnx_model(onnx_path)
        has_batch_dim = if_has_batch_dim(onnx_path)

        try:
            success = False
            infer_shape(model, has_batch_dim, verbose)
            success = True
            success_count += 1
        except Exception as e:
            failed_onnx_paths.append(onnx_path)
            raise e

        print(
            f"{'Success' if success else 'Failure'} "
            f"({time.perf_counter() - time_start:.2f}s) "
            f"for {os.path.basename(onnx_path)}"
        )
        total_count += 1

    if total_count > success_count:
        print(
            f"{len(failed_onnx_paths)} failed models:\n" + "\n".join(failed_onnx_paths)
        )

    print(f"\nSuccessfully inferred shapes for {success_count}/{total_count} models")
