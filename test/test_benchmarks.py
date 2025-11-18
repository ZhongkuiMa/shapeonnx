"""VNNComp Benchmark Test Runner for ShapeONNX.

Runs shape inference on all VNNComp benchmark models and validates results.
"""

import os
import time

from utils import (
    find_benchmarks_folders,
    find_all_onnx_files,
    load_onnx_model,
    if_has_batch_dim,
    infer_shape,
)

if __name__ == "__main__":
    dir_name = "benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collect {len(benchmark_dirs)} benchmark directories.")
    onnx_paths = find_all_onnx_files(benchmark_dirs)
    print(f"Collect {len(onnx_paths)} ONNX models.")

    # Uncomment to test specific models
    # onnx_paths = [
    #     "..\\..\\..\\vnncomp2024_benchmarks\\benchmarks\\cctsdb_yolo_2023\\onnx\\patch-1.onnx"
    # ]

    failed_onnx_paths = []
    # verbose = True
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
            # print(f"Failed to infer shapes: {e}")

        print(
            f"{'Success!' if success else 'Failure.'} "
            f"({time.perf_counter() - time_start:.2f}s) "
            f"for {os.path.basename(onnx_path)}"
        )
        total_count += 1

    if total_count > success_count:
        print(
            f"{len(failed_onnx_paths)} failed models:\n" + "\n".join(failed_onnx_paths)
        )

    print(f"\nSuccessfully inferred shapes for {success_count}/{total_count} models.")
