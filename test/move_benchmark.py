"""Script to collect ONNX models from VNNComp 2024 benchmarks."""

__docformat__ = "restructuredtext"
__all__ = []

import os

from utils import find_all_onnx_files, find_benchmarks_folders, find_onnx_folders

if __name__ == "__main__":
    dir_name = "../../../vnncomp2024_benchmarks/benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collected {len(benchmark_dirs)} benchmark directories")
    onnx_dirs = find_onnx_folders(benchmark_dirs)
    print(f"Collected {len(onnx_dirs)} ONNX directories")

    dir_name = "benchmarks/"
    os.makedirs(dir_name, exist_ok=True)
    for bdir in benchmark_dirs:
        benchmark_name = os.path.basename(bdir)
        os.makedirs(os.path.join(dir_name, benchmark_name), exist_ok=True)

    max_onnx_per_benchmark = 20
    i = 0
    for onnx_dir in onnx_dirs:
        benchmark_name = os.path.basename(os.path.dirname(onnx_dir))
        target_dir = os.path.join(dir_name, benchmark_name)

        onnx_files = find_all_onnx_files([onnx_dir], num_limit=max_onnx_per_benchmark)
        for onnx_path in onnx_files:
            target_path = os.path.join(target_dir, os.path.basename(onnx_path))
            with open(onnx_path, "rb") as src_file:
                with open(target_path, "wb") as dst_file:
                    dst_file.write(src_file.read())
            i += 1

    print(f"Copied {i} ONNX models to {dir_name}")
