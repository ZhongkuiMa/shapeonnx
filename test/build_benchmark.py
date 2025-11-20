"""Script to collect ONNX models from VNNComp 2024 benchmarks with instances.csv."""

__docformat__ = "restructuredtext"
__all__ = []

import csv
import os
import shutil
from pathlib import Path


def copy_benchmark_with_instances(
    source_benchmark_dir: str,
    target_base_dir: str,
    max_instances: int | None = None,
) -> int:
    """
    Copy benchmark folder with instances.csv and referenced ONNX files.

    :param source_benchmark_dir: Source benchmark directory path
    :param target_base_dir: Target base directory
    :param max_instances: Maximum instances to copy (None = all)
    :return: Number of ONNX files copied
    """
    benchmark_name = os.path.basename(source_benchmark_dir)
    target_dir = Path(target_base_dir) / benchmark_name
    target_dir.mkdir(parents=True, exist_ok=True)

    instances_csv = Path(source_benchmark_dir) / "instances.csv"
    if not instances_csv.exists():
        return 0

    # Read instances and collect unique ONNX files
    onnx_files = set()
    rows_to_copy = []
    seen_onnx = set()

    with open(instances_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                onnx_path = row[0]
                if onnx_path not in seen_onnx:
                    seen_onnx.add(onnx_path)
                    onnx_files.add(onnx_path)
                    rows_to_copy.append(row)
                    if max_instances and len(rows_to_copy) >= max_instances:
                        break

    # Copy instances.csv with deduplicated entries
    with open(target_dir / "instances.csv", "w", newline="") as dst_f:
        writer = csv.writer(dst_f)
        writer.writerows(rows_to_copy)

    # Copy ONNX files preserving structure
    copied_count = 0
    for onnx_rel_path in onnx_files:
        source_onnx = Path(source_benchmark_dir) / onnx_rel_path
        target_onnx = target_dir / onnx_rel_path

        if source_onnx.exists():
            target_onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_onnx, target_onnx)
            copied_count += 1

    return copied_count


def build_benchmarks(
    source_base_dir: str,
    target_base_dir: str = "benchmarks",
    max_instances_per_benchmark: int | None = 20,
) -> None:
    """
    Build benchmark collection from VNNComp benchmarks.

    :param source_base_dir: Source VNNComp benchmarks directory
    :param target_base_dir: Target directory for collected benchmarks
    :param max_instances_per_benchmark: Max instances per benchmark (None = all)
    """
    source_path = Path(source_base_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_base_dir}")

    Path(target_base_dir).mkdir(parents=True, exist_ok=True)

    total_benchmarks = 0
    total_onnx_files = 0

    for benchmark_dir in sorted(source_path.iterdir()):
        if not benchmark_dir.is_dir():
            continue

        copied = copy_benchmark_with_instances(
            str(benchmark_dir),
            target_base_dir,
            max_instances_per_benchmark,
        )

        if copied > 0:
            total_benchmarks += 1
            total_onnx_files += copied
            print(f"[{benchmark_dir.name}] Copied {copied} ONNX files")

    print(f"\nTotal: {total_benchmarks} benchmarks, {total_onnx_files} ONNX files")


if __name__ == "__main__":
    source_dir = "../../../vnncomp2024_benchmarks/benchmarks"
    target_dir = "benchmarks"
    max_per_benchmark = 20

    build_benchmarks(source_dir, target_dir, max_per_benchmark)
