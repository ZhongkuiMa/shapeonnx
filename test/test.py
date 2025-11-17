import os


def find_benchmarks_folders(base_dir):
    benchmark_dirs = []
    # Only consider first-level subdirectories
    for entry in os.listdir(base_dir):
        subdir = os.path.normpath(os.path.join(base_dir, entry))
        if os.path.isdir(subdir):
            benchmark_dirs.append(subdir)
    return benchmark_dirs


def find_onnx_folders(benchmark_dirs):
    onnx_dirs = []
    for bdir in benchmark_dirs:
        onnx_subdir = os.path.join(bdir, "onnx")
        if os.path.isdir(onnx_subdir):
            onnx_dirs.append(onnx_subdir)
    return onnx_dirs


def find_all_onnx_files(onnx_dirs, num_limit: int = 20):
    onnx_files = []
    for odir in onnx_dirs:
        for i, file in enumerate(os.listdir(odir)):
            if file.endswith(".onnx"):
                onnx_files.append(os.path.normpath(os.path.join(odir, file)))
            if i + 1 >= num_limit:
                break
    return onnx_files


def load_onnx_model(onnx_path: str):
    import onnx

    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=21)
    return model


benchmark_without_batch_dim = [
    "cctsdb_yolo",
    "cgan",
    "pensieve_big_parallel",
    "pensieve_mid_parallel",
    "pensieve_small_parallel",
]


def if_has_batch_dim(onnx_path: str):
    return all(bname not in onnx_path for bname in benchmark_without_batch_dim)


def check_shape_compatibility(
    inferred_shape,
    expected_shape,
) -> bool:
    if inferred_shape == expected_shape:
        return True
    if inferred_shape == [] and expected_shape == [1]:
        return True
    return False


def infer_shape(
    model,
    has_batch_dim: bool = True,
    verbose: bool = False,
):
    from shapeonnx import infer_onnx_shape, extract_io_shapes
    from shapeonnx.shapeonnx.utils import (
        get_initializers,
        get_input_nodes,
        get_output_nodes,
        convert_constant_to_initializer,
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

    for output_name in output_nodes:
        output_name = output_name.name
        shape = data_shapes[output_name]
        expected_shape = expected_output_shapes[output_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Output shape mismatch for '{output_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )


if __name__ == "__main__":
    dir_name = "../../../vnncomp2024_benchmarks/benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collect {len(benchmark_dirs)} benchmark directories.")
    onnx_dirs = find_onnx_folders(benchmark_dirs)
    print(f"Collect {len(onnx_dirs)} ONNX directories.")
    onnx_paths = find_all_onnx_files(onnx_dirs)
    print(f"Collect {len(onnx_paths)} ONNX models.")

    # onnx_paths = [
    #     "..\\..\\..\\vnncomp2024_benchmarks\\benchmarks\\cctsdb_yolo_2023\\onnx\\patch-1.onnx"
    #     #     "..\\..\\..\\vnncomp2024_benchmarks\\benchmarks\\ml4acopf_2023\\onnx\\118_ieee_ml4acopf.onnx"
    # ]

    import time

    failed_onnx_paths = []
    # verbose = True
    verbose = False

    success_count = 0
    total_count = 0
    for onnx_path in onnx_paths:
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

    print(f"Successfully inferred shapes for {success_count}/{total_count} models.")
