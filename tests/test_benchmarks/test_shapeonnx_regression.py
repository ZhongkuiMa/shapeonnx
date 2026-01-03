"""Regression testing for ShapeONNX against baselines and ONNX reference."""

__docformat__ = "restructuredtext"

import json
from pathlib import Path

import onnx
import onnx.shape_inference
import pytest

from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    convert_constant_to_initializer,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)
from tests.test_benchmarks.utils import (
    find_benchmarks_folders,
    get_all_onnx_files,
    if_has_batch_dim,
    load_onnx_model,
)


def get_baseline_path(onnx_path: str, baselines_dir: Path) -> Path:
    """
    Get baseline JSON path for an ONNX model.

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Root directory (Path object) to store baseline files
    :return: Path to baseline JSON file
    """
    path = Path(onnx_path)
    # Extract benchmark name from path (e.g., "test", "acasxu_2023", etc.)
    # Path structure: .../benchmarks/{benchmark_name}/onnx/{model}.onnx
    benchmark_name = path.parent.parent.name

    # Build path relative to baselines directory fixture
    baseline_path = baselines_dir / benchmark_name / path.name.replace(".onnx", ".json")
    return baseline_path


def load_baseline_shapes(baseline_path: str | Path) -> dict | None:
    """
    Load baseline data from JSON file.

    :param baseline_path: Path to baseline JSON file (str or Path)
    :return: Baseline data dictionary, or None if file not found
    """
    path = Path(baseline_path) if isinstance(baseline_path, str) else baseline_path
    if not path.exists():
        return None
    with path.open() as f:
        data: dict = json.load(f)

    # Convert old format to new format for compatibility
    if "shapeonnx_shapes" not in data:
        return {"shapeonnx_shapes": data, "onnx_shapes": None, "differences": None}

    return data


def _extract_shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int] | None:
    """
    Extract shape from ONNX ValueInfoProto.

    :param value_info: ONNX value info proto
    :return: List of dimension sizes, or None if no shape available
    """
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            # Dynamic dimension (could be dim_param or unknown)
            shape.append(-1)
    return shape


def get_onnx_reference_shapes(onnx_path: str) -> dict[str, list[int]]:
    """
    Get shapes using ONNX's built-in shape inference.

    :param onnx_path: Path to ONNX model file
    :return: Dictionary mapping tensor names to shapes
    """
    model = onnx.load(onnx_path)
    inferred_model = onnx.shape_inference.infer_shapes(model)

    shapes = {}

    # Extract shapes from value_info (intermediate tensors)
    for value_info in inferred_model.graph.value_info:
        shape = _extract_shape_from_value_info(value_info)
        if shape is not None:
            shapes[value_info.name] = shape

    # Extract shapes from outputs
    for output in inferred_model.graph.output:
        shape = _extract_shape_from_value_info(output)
        if shape is not None:
            shapes[output.name] = shape

    # Extract shapes from inputs
    for input_val in inferred_model.graph.input:
        shape = _extract_shape_from_value_info(input_val)
        if shape is not None:
            shapes[input_val.name] = shape

    return shapes


def _is_shape_tensor_match(shape_onnx: list[int], shape_shapeonnx: int | list[int]) -> bool:
    """
    Check if shapes match due to shape tensor metadata vs actual values.

    :param shape_onnx: ONNX inferred shape
    :param shape_shapeonnx: ShapeONNX inferred shape
    :return: True if this is a shape tensor metadata mismatch (acceptable)
    """
    # ONNX: [n] (metadata), shapeonnx: list with n elements (actual values)
    if isinstance(shape_shapeonnx, int):
        return False
    return len(shape_onnx) == 1 and shape_onnx[0] > 0 and len(shape_shapeonnx) == shape_onnx[0]


def _normalize_shapes(
    shape_onnx: list[int], shape_shapeonnx: int | list[int]
) -> tuple[list[int], list[int]]:
    """
    Normalize shapes by treating dynamic dimensions consistently.

    :param shape_onnx: ONNX inferred shape
    :param shape_shapeonnx: ShapeONNX inferred shape
    :return: Tuple of (normalized_onnx, normalized_shapeonnx)
    """
    # Handle scalar shapes - return as-is wrapped in lists
    if isinstance(shape_shapeonnx, int):
        return shape_onnx, [shape_shapeonnx]

    normalized_onnx = []
    normalized_shapeonnx = []

    for _i, (d_onnx, d_shapeonnx) in enumerate(zip(shape_onnx, shape_shapeonnx, strict=False)):
        # If one side has -1 (dynamic) and the other has 1, treat both as dynamic
        if (
            (d_onnx == -1 and d_shapeonnx == 1)
            or (d_onnx == 1 and d_shapeonnx == -1)
            or d_onnx <= 0
            or d_shapeonnx <= 0
        ):
            normalized_onnx.append(-1)
            normalized_shapeonnx.append(-1)
        else:
            normalized_onnx.append(d_onnx)
            normalized_shapeonnx.append(d_shapeonnx)

    return normalized_onnx, normalized_shapeonnx


def compare_with_onnx_reference(
    onnx_path: str, shapeonnx_shapes: dict[str, int | list[int]]
) -> dict:
    """
    Compare shapeonnx results with ONNX reference implementation.

    :param onnx_path: Path to ONNX model file
    :param shapeonnx_shapes: Shapes inferred by shapeonnx
    :return: Dictionary categorizing differences
    """
    try:
        onnx_shapes = get_onnx_reference_shapes(onnx_path)
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        return {
            "error": str(e),
            "shapeonnx_only": [],
            "onnx_only": [],
            "mismatches": [],
            "dynamic_diffs": [],
        }

    differences: dict[str, list] = {
        "shapeonnx_only": [],
        "onnx_only": [],
        "mismatches": [],
        "dynamic_diffs": [],
    }

    all_keys = set(shapeonnx_shapes.keys()) | set(onnx_shapes.keys())

    for key in all_keys:
        if key not in shapeonnx_shapes:
            differences["onnx_only"].append((key, onnx_shapes[key]))
        elif key not in onnx_shapes:
            differences["shapeonnx_only"].append((key, shapeonnx_shapes[key]))
        elif shapeonnx_shapes[key] != onnx_shapes[key]:
            shape_onnx = onnx_shapes[key] if not isinstance(onnx_shapes[key], int) else []
            shape_shapeonnx = (
                shapeonnx_shapes[key] if not isinstance(shapeonnx_shapes[key], int) else []
            )

            if _is_shape_tensor_match(shape_onnx, shape_shapeonnx):
                differences["dynamic_diffs"].append((key, shapeonnx_shapes[key], onnx_shapes[key]))
                continue

            normalized_onnx, normalized_shapeonnx = _normalize_shapes(shape_onnx, shape_shapeonnx)

            if normalized_onnx == normalized_shapeonnx:
                differences["dynamic_diffs"].append((key, shapeonnx_shapes[key], onnx_shapes[key]))
            else:
                differences["mismatches"].append((key, shapeonnx_shapes[key], onnx_shapes[key]))

    return differences


def infer_shapeonnx_shapes(onnx_path: str) -> dict[str, int | list[int]]:
    """
    Run shapeonnx shape inference on a model.

    :param onnx_path: Path to ONNX model file
    :return: Dictionary mapping tensor names to inferred shapes
    """
    has_batch_dim = if_has_batch_dim(onnx_path)
    model = load_onnx_model(onnx_path)

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)
    nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

    return infer_onnx_shape(
        input_nodes,
        output_nodes,
        nodes,
        initializers,
        has_batch_dim=has_batch_dim,
        verbose=False,
    )


# Pytest fixtures and test collection


def get_onnx_models():
    """Collect all ONNX models from vnncomp2024 benchmarks.

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


# Main regression tests (baselines_dir fixture is defined in conftest.py)


@pytest.mark.benchmark
@pytest.mark.parametrize("onnx_path", get_onnx_models())
def test_create_baseline(onnx_path, baselines_dir):
    """
    Create or update baseline for each ONNX model.

    This test creates baselines that include both shapeonnx and ONNX reference
    shapes for comparison.

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Directory to store baselines
    """
    # Run shapeonnx inference
    shapeonnx_shapes = infer_shapeonnx_shapes(onnx_path)

    # Get ONNX reference shapes
    onnx_shapes = get_onnx_reference_shapes(onnx_path)

    # Compare with ONNX reference
    differences = compare_with_onnx_reference(onnx_path, shapeonnx_shapes)

    # Save enhanced baseline with both results
    baseline_data = {
        "shapeonnx_shapes": shapeonnx_shapes,
        "onnx_shapes": onnx_shapes,
        "differences": {
            "shapeonnx_only": len(differences["shapeonnx_only"]),
            "onnx_only": len(differences["onnx_only"]),
            "mismatches": len(differences["mismatches"]),
            "dynamic_diffs": len(differences["dynamic_diffs"]),
        },
    }

    baseline_path = get_baseline_path(onnx_path, baselines_dir)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w") as f:
        json.dump(baseline_data, f, indent=2)

    # Verify baseline was created
    assert baseline_path.exists()
    assert shapeonnx_shapes is not None
    assert len(shapeonnx_shapes) > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("onnx_path", get_onnx_models())
def test_verify_baseline(onnx_path, baselines_dir):
    """
    Verify current shapeonnx inference matches stored baseline.

    :param onnx_path: Path to ONNX model file
    :param baselines_dir: Directory containing baselines
    """
    baseline_path = get_baseline_path(onnx_path, baselines_dir)
    baseline_data = load_baseline_shapes(baseline_path)

    # Skip if no baseline exists yet
    if baseline_data is None:
        pytest.skip(f"No baseline for {Path(onnx_path).name}")

    baseline_shapes = baseline_data["shapeonnx_shapes"]

    # Run current inference
    current_shapes = infer_shapeonnx_shapes(onnx_path)

    # Compare with baseline
    if current_shapes != baseline_shapes:
        # Get ONNX reference for debugging
        onnx_ref_shapes = get_onnx_reference_shapes(onnx_path)

        all_keys = set(current_shapes.keys()) | set(baseline_shapes.keys())
        mismatches = []
        for key in sorted(all_keys):
            if key not in current_shapes:
                mismatches.append(f"Missing: {key}")
            elif key not in baseline_shapes:
                onnx_match = (
                    " (ONNX agrees)"
                    if key in onnx_ref_shapes and onnx_ref_shapes[key] == current_shapes[key]
                    else ""
                )
                mismatches.append(f"New: {key} = {current_shapes[key]}{onnx_match}")
            elif current_shapes[key] != baseline_shapes[key]:
                onnx_info = ""
                if key in onnx_ref_shapes:
                    if onnx_ref_shapes[key] == current_shapes[key]:
                        onnx_info = " (current matches ONNX)"
                    elif onnx_ref_shapes[key] == baseline_shapes[key]:
                        onnx_info = " (baseline matches ONNX)"
                    else:
                        onnx_info = f" (ONNX: {onnx_ref_shapes[key]})"

                mismatches.append(
                    f"{key}: {baseline_shapes[key]} -> {current_shapes[key]}{onnx_info}"
                )

        pytest.fail("\n".join(mismatches))

    assert current_shapes == baseline_shapes


@pytest.mark.benchmark
@pytest.mark.parametrize("onnx_path", get_onnx_models())
def test_onnx_consistency(onnx_path):
    """
    Compare shapeonnx inference with ONNX reference implementation.

    This test verifies that shapeonnx produces consistent results with ONNX's
    built-in shape inference, allowing for acceptable differences like:
    - Tensors only shapeonnx infers (shape tensors, intermediate values)
    - Different dynamic dimension representations

    :param onnx_path: Path to ONNX model file
    """
    # Run shapeonnx inference
    shapeonnx_shapes = infer_shapeonnx_shapes(onnx_path)

    # Compare with ONNX reference
    differences = compare_with_onnx_reference(onnx_path, shapeonnx_shapes)

    # Check for errors
    if "error" in differences:
        pytest.skip(f"ONNX shape inference failed: {differences['error']}")

    # Report differences for informational purposes
    info_parts = []
    if differences["shapeonnx_only"]:
        info_parts.append(f"{len(differences['shapeonnx_only'])} tensors only in shapeonnx")
    if differences["onnx_only"]:
        info_parts.append(f"{len(differences['onnx_only'])} tensors only in ONNX")
    if differences["dynamic_diffs"]:
        info_parts.append(f"{len(differences['dynamic_diffs'])} dynamic dimension differences")

    # Actual mismatches are failures
    if differences["mismatches"]:
        mismatch_details = []
        for tensor_name, shape_shapeonnx, shape_onnx in differences["mismatches"]:
            mismatch_details.append(
                f"  {tensor_name}: shapeonnx={shape_shapeonnx}, onnx={shape_onnx}"
            )

        pytest.fail("Shape mismatches between shapeonnx and ONNX:\n" + "\n".join(mismatch_details))

    # Test passes - differences are acceptable
    if info_parts:
        print(f"\nInfo: {', '.join(info_parts)}")
