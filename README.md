# ShapeONNX

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/shapeonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/shapeonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/shapeonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/shapeonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![ONNX 1.16](https://img.shields.io/badge/ONNX-1.16-brightgreen.svg)](https://onnx.ai)
[![NumPy 1.26](https://img.shields.io/badge/NumPy-1.26-green.svg)](https://numpy.org/)
[![VNN-COMP 2024](https://img.shields.io/badge/VNN--COMP-2024-orange.svg)](https://sites.google.com/view/vnn2024)
[![Version](https://img.shields.io/badge/version-2026.1.0-blue.svg)](https://github.com/ZhongkuiMa/shapeonnx/releases)

Static shape inference for ONNX models where standard tools fail.

**Tested on 138 VNN-COMP 2024 models with 100% success rate.**
**Provides more accurate shape information than ONNX's built-in inference.**

## Overview

ONNX's built-in `onnx.shape_inference.infer_shapes` handles most models correctly, but fails in several critical scenarios:

- Models with inconsistent ONNX versions or opset mismatches
- Non-standard conversions from PyTorch or other frameworks
- Dynamic shape operations where shape computations depend on data
- Shape operator chains (`Shape → Gather → Slice → Concat → Reshape`)
- Models with custom shape manipulations (Vision Transformers, GANs)

**ShapeONNX goes beyond ONNX's capabilities** through advanced static shape computation:

- **Shape Tensor Tracking**: Propagates actual shape values (e.g., `[1, 48, 2, 2]`) where ONNX only tracks tensor metadata (e.g., `[4]`)
- **Static Resolution**: Resolves shapes ONNX marks as dynamic (-1) to concrete values when statically computable
- **Operator Chain Analysis**: Processes complex `Shape → Gather → Slice → Concat` patterns to static constants
- **Explicit Shape Propagation**: Distinguishes shape tensors from data tensors for accurate downstream inference
- **Verification-Ready**: Provides the precise static shapes required by neural network verification tools

## Key Advantages Over ONNX

ShapeONNX provides more accurate shape information than ONNX's built-in inference:

| Scenario | ONNX Result | ShapeONNX Result |
|----------|-------------|------------------|
| Shape operation output | `[4]` (tensor metadata) | `[1, 48, 2, 2]` (actual values) |
| Slice of shape tensor | `[2]` (tensor type) | `[1, 48]` (sliced values) |
| Concat of shapes | `[3]` (1D array) | `[1, 48, -1]` (reshape target) |
| ConstantOfShape | `[-1, -1, -1]` (dynamic) | `[1, 1, 48]` (concrete shape) |
| Batch dimensions | `-1` (dynamic) | `1` (static when determinable) |

**Why this matters**: Neural network verification tools need **exact static shapes** for layer-by-layer analysis. ShapeONNX resolves shapes ONNX marks as dynamic to concrete values when they're statically computable.

## Motivation

Neural network verification tools require precise static shapes for:

- Layer-by-layer bound propagation
- Memory allocation for symbolic execution
- Constraint generation for SMT solvers
- Model optimization and fusion (SlimONNX)

When ONNX shape inference fails or returns dynamic shapes, verification pipelines break. ShapeONNX fills this gap by providing robust static shape inference for the complex models encountered in verification research.

## Features

- **Superior Shape Inference**: More accurate than ONNX's built-in inference for shape tensors
- **Advanced Shape Tracking**: Propagates actual shape values through operator chains
- **Shape Operator Chains**: Resolves `Shape → Gather → Slice → Concat → Reshape` patterns
- **Dynamic to Static**: Converts shapes ONNX marks as dynamic to concrete static values
- **Explicit Shape Propagation**: Distinguishes shape tensors from data tensors
- **51 Operators**: Comprehensive coverage including Sign, Conv1d, and DivCst
- **Fast Performance**: Single-pass O(1) forward propagation
- **Pure Python**: No C/C++ dependencies, easy integration
- **Production Ready**: Tested on 138 VNN-COMP 2024 models with ONNX consistency validation

## Use Cases

ShapeONNX is essential for:

1. **Neural Network Verification**: Tools requiring static shapes (α,β-CROWN, ERAN, Marabou)
2. **Model Optimization**: Pre-optimization shape resolution (SlimONNX)
3. **Shape-Dependent Transformations**: Operations requiring known tensor dimensions
4. **Complex Model Analysis**: Understanding shape propagation in non-standard models

## Installation

### Requirements

- Python 3.11 or higher
- onnx 1.16.0
- numpy 1.26.4

**Important**: ShapeONNX is not available on PyPI. Local installation from source is required.

### Local Installation (Development & Usage)

ShapeONNX must be installed locally from the source repository:

```bash
# Clone the repository
git clone <repository-url>
cd shapeonnx

# Install dependencies
pip install onnx==1.16.0 numpy==1.26.4

# Install ShapeONNX in editable mode for development
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"  # Includes ruff, mypy, pytest, pytest-cov
```

### Version Compatibility

- **ONNX 1.16.0**: Tested opset range 17-21
- **NumPy 1.26.4**: Required for Python 3.11+ compatibility
- **Python 3.11+**: Required for modern type hint syntax (using `|` for unions)

Models should be converted to ONNX IR version 21 using `onnx.version_converter` for maximum compatibility.

## Quick Start

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    get_initializers,
    get_input_nodes,
    get_output_nodes,
    convert_constant_to_initializer,
)

# Load and prepare model
model = onnx.load("model.onnx")
model = onnx.version_converter.convert_version(model, target_version=21)

# Extract model components
initializers = get_initializers(model)
input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
output_nodes = get_output_nodes(model, has_batch_dim=True)

# Convert Constant nodes to initializers (required preprocessing)
nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

# Infer shapes
shapes = infer_onnx_shape(
    input_nodes,
    output_nodes,
    nodes,
    initializers,
    has_batch_dim=True,
    verbose=False,
)

# Access inferred shapes
for tensor_name, shape in shapes.items():
    print(f"{tensor_name}: {shape}")
```

## API Reference

### Core Functions

#### infer_onnx_shape()

Main shape inference function.

```python
def infer_onnx_shape(
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    has_batch_dim: bool = True,
    verbose: bool = False,
) -> dict[str, list[int]]
```

**Parameters**:
- `input_nodes` (list[ValueInfoProto]): Model input value infos
- `output_nodes` (list[ValueInfoProto]): Model output value infos
- `nodes` (list[NodeProto]): Model computation nodes (Constant nodes must be converted to initializers)
- `initializers` (dict[str, TensorProto]): Model initializers (weights and constants)
- `has_batch_dim` (bool): Whether model has batch dimension (default: True)
- `verbose` (bool): Print debug information during inference (default: False)

**Returns**: dict[str, list[int]] - Dictionary mapping tensor names to inferred shapes

**Note**: Constant nodes must be converted to initializers before calling this function using `convert_constant_to_initializer()`.

#### extract_io_shapes()

Extract shapes from model input/output nodes.

```python
def extract_io_shapes(
    nodes: list[ValueInfoProto],
    has_batch_dim: bool
) -> dict[str, list[int]]
```

**Parameters**:
- `nodes` (list[ValueInfoProto]): Input or output value infos
- `has_batch_dim` (bool): Whether tensors have batch dimension

**Returns**: dict[str, list[int]] - Dictionary mapping names to shapes

### Utility Functions

#### convert_constant_to_initializer()

Convert Constant nodes to initializers (required preprocessing step).

```python
def convert_constant_to_initializer(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto]
) -> list[NodeProto]
```

**Parameters**:
- `nodes` (list[NodeProto]): Model nodes
- `initializers` (dict[str, TensorProto]): Initializer dictionary (modified in-place)

**Returns**: list[NodeProto] - Nodes with Constant nodes removed

#### get_initializers()

Extract initializers from model.

```python
def get_initializers(model: ModelProto) -> dict[str, TensorProto]
```

#### get_input_nodes()

Extract input nodes with proper shape formatting.

```python
def get_input_nodes(
    model: ModelProto,
    initializers: dict[str, TensorProto],
    has_batch_dim: bool
) -> list[ValueInfoProto]
```

#### get_output_nodes()

Extract output nodes with proper shape formatting.

```python
def get_output_nodes(
    model: ModelProto,
    has_batch_dim: bool
) -> list[ValueInfoProto]
```

## Supported Operators

ShapeONNX supports 51 operators across 10 categories:

### Arithmetic Operations
Add, Sub, Mul, Div, DivCst, Pow, Neg

### Activation Functions
Relu, LeakyRelu, Sigmoid, Tanh, Clip, Sin, Cos, Sign

### Convolution and Pooling
Conv, Conv1d, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool

### Normalization
BatchNormalization

### Tensor Manipulation
Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand

### Slicing and Concatenation
Slice, Split, Gather, Concat

### Shape Operations
Shape, ConstantOfShape, Range

### Reduction Operations
ReduceMean, ReduceSum, ArgMax

### Comparison and Selection
Equal, Where, Max, Min

### Matrix Operations
MatMul, Gemm

### Other Operations
Cast, Dropout, Pad, Resize, Scatter, ScatterElements, ScatterND, Softmax, Floor

## Architecture

### Design Principles

- **Immutable Context**: Frozen dataclass for shape inference context
- **Pure Functions**: All shape inference functions are stateless with explicit inputs
- **Direct Dictionary Access**: Minimal abstraction for performance
- **Full Type Hints**: Complete type annotations using Python 3.11+ syntax

### Performance Characteristics

- **Single-Pass Forward Propagation**: O(1) complexity per operator
- **Pre-Converted Initializers**: Integer tensors converted once at initialization
- **Efficient Operator Dispatch**: Dictionary-based operator function mapping
- **Minimal Memory Allocations**: Shape lists reused where possible

**Benchmark**: 140 VNN-COMP 2024 models processed in approximately 6.5 seconds on Intel i5-12400F.

### Module Structure

```
shapeonnx/
├── __init__.py              # Public API exports
├── infer_shape.py           # Main shape inference engine and ShapeInferenceContext
├── onnx_attrs.py            # ONNX attribute extraction utilities
└── utils.py                 # Helper functions (get_initializers, input/output extraction, etc.)
```

## Examples

### Example 1: Basic Shape Inference

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    get_initializers,
    get_input_nodes,
    get_output_nodes,
    convert_constant_to_initializer,
)

# Load model
model = onnx.load("resnet18.onnx")

# Prepare components
initializers = get_initializers(model)
input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
output_nodes = get_output_nodes(model, has_batch_dim=True)
nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

# Infer shapes
shapes = infer_onnx_shape(
    input_nodes, output_nodes, nodes, initializers,
    has_batch_dim=True, verbose=True
)

# Print all tensor shapes
for name, shape in sorted(shapes.items()):
    print(f"{name}: {shape}")
```

### Example 2: Integration with SlimONNX

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    get_initializers,
    get_input_nodes,
    get_output_nodes,
    convert_constant_to_initializer,
)

# Load and prepare model
model = onnx.load("model.onnx")
model = onnx.version_converter.convert_version(model, target_version=21)

initializers = get_initializers(model)
input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
output_nodes = get_output_nodes(model, has_batch_dim=True)
nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

# Infer shapes for optimization
shapes = infer_onnx_shape(
    input_nodes, output_nodes, nodes, initializers,
    has_batch_dim=True
)

# Use shapes for optimization decisions
for node in nodes:
    for input_name in node.input:
        if input_name in shapes:
            input_shape = shapes[input_name]
            # Make optimization decisions based on shape
            if len(input_shape) == 2:
                # Can apply matrix-specific optimizations
                pass
```

### Example 3: Handling Shape Operator Chains

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    get_initializers,
    get_input_nodes,
    get_output_nodes,
    convert_constant_to_initializer,
)

# Model with Shape → Gather → Add → Reshape pattern
model = onnx.load("dynamic_reshape_model.onnx")

initializers = get_initializers(model)
input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
output_nodes = get_output_nodes(model, has_batch_dim=True)
nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

# ShapeONNX resolves shape chains to static values
shapes = infer_onnx_shape(
    input_nodes, output_nodes, nodes, initializers,
    has_batch_dim=True, verbose=True
)

# Dynamic reshape operations now have static target shapes
print("Resolved static shapes for all tensors")
```

## Testing and Validation

### VNN-COMP 2024 Benchmarks

ShapeONNX has been extensively tested on models from VNN-COMP 2024:

- **Total Models Tested**: 138 diverse neural networks
- **Success Rate**: 100% (all models successfully processed)
- **Model Types**: CNNs, ResNets, VGG, Vision Transformers, GANs, Graph Neural Networks
- **Opset Coverage**: Opset 17-21
- **ONNX Consistency**: Validated against ONNX reference with special handling for shape tensors

### Test Suite

The comprehensive pytest-based test suite includes:

1. **Shape Inference Tests** (`test_shapeonnx.py`): Validates shape inference on all 136 models
2. **Baseline Tests** (`test_shapeonnx_regression.py`):
   - Creates and verifies baselines for regression detection
   - Compares with ONNX reference implementation
   - Handles shape tensor differences (shapeonnx tracks values, ONNX tracks metadata)

### Run Tests

```bash
cd shapeonnx
pytest tests/test_shapeonnx.py -v                    # Run shape inference tests
pytest tests/test_shapeonnx_regression.py -v          # Run regression tests with ONNX comparison
```

**Expected output**: `414 passed` (138 models × 3 test types)

### Advanced Features Validated

The test suite demonstrates ShapeONNX's superior capabilities:

- **Shape Tensor Tracking**: Correctly infers `[1, 48, 2, 2]` for Shape operations where ONNX only knows `[4]`
- **Static Resolution**: Resolves `ConstantOfShape` outputs to `[1, 1, 48]` where ONNX shows `[-1, -1, -1]`
- **Operator Chains**: Processes `Shape → Slice → Concat → Reshape` to concrete target shapes
- **Dynamic to Static**: Converts batch dimensions and other dynamic shapes to concrete values when statically determinable

## Performance Benchmarks

**Hardware**: Intel i5-12400F (6 cores, 12 threads)

**Results**:
- 138 VNN-COMP 2024 models: ~9.6 seconds total (shape inference)
- Average per model: ~70 milliseconds
- Complex models (Vision Transformers, GANs): <200ms
- Simple models (ACAS Xu, TLL): <10ms
- Full test suite (590 tests): ~1.5 seconds

**Memory**: Typical peak memory usage under 500MB for largest models.

**Note**: ShapeONNX's comprehensive shape tensor tracking adds minimal overhead while providing significantly more accurate shape information than ONNX's built-in inference.

## Known Limitations

- Constant nodes must be converted to initializers before shape inference
- Asymmetric padding in Conv/Pool operations not supported
- Control flow operators (If, Loop, Scan) not supported
- Some operators have limited attribute support
- Assumes static input shapes (dynamic batch size handled via `has_batch_dim` flag)

## Related Projects

- **[SlimONNX](https://github.com/ZhongkuiMa/slimonnx)**: ONNX model optimization. Uses ShapeONNX for shape-dependent optimizations like constant folding and redundant operation removal.
- **[TorchVNNLIB](https://github.com/ZhongkuiMa/torchvnnlib)**: VNN-LIB to tensor converter for neural network verification.
- **[VNN-COMP](https://sites.google.com/view/vnn2024)**: International Verification of Neural Networks Competition.

## Contributing & Development

### Development Setup

1. **Clone and install in development mode**:
   ```bash
   git clone <repository-url>
   cd shapeonnx
   pip install -e ".[dev]"  # Installs with all dev dependencies
   ```

2. **Verify your setup**:
   ```bash
   pytest tests/test_units/ -q  # Should run 590 tests
   ruff check src/shapeonnx tests  # Should pass with 0 errors
   ruff format --check src/shapeonnx tests  # Should show all files formatted
   mypy src/shapeonnx  # Should show no issues
   ```

### Development Workflow

Before making any changes, create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Work on your changes and commit regularly:

```bash
git add .
git commit -m "Description of your changes"
```

### Quality Checks

Before submitting a pull request, ensure all checks pass:

```bash
# 1. Run all tests
pytest tests/test_units/ -q

# 2. Run with coverage
pytest tests/test_units/ --cov=shapeonnx --cov-report=term-missing

# 3. Check code style
ruff check src/shapeonnx tests

# 4. Check formatting
ruff format --check src/shapeonnx tests

# 5. Type checking
mypy src/shapeonnx
```

### Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**:
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your feature branch as the source
   - Provide a clear title and description

3. **Pull Request Requirements**:
   - All 590 unit tests must pass
   - Code coverage must be ≥85%
   - All code style checks (ruff) must pass
   - All type checks (mypy) must pass
   - No direct pushes to main branch (PRs required)

4. **After Approval**:
   - Maintainers will merge your PR
   - Your code will be automatically tested by GitHub Actions

### Adding New Operators

To add a new ONNX operator:

1. **Implement the shape inference function** in `src/shapeonnx/infer_shape.py`:

```python
def _infer_<operator>_shape(
    node: NodeProto,
    ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """Infer shape for <Operator> node.

    :param node: ONNX node
    :param ctx: Shape inference context with data_shapes, explicit_shapes, initializers
    :return: List of (data_shape, explicit_shape) tuples for each output
    """
    # Get input shapes
    input_name = node.input[0]
    data_shape = ctx.data_shapes.get(input_name)

    # Compute output shape
    output_shape = ...  # Your logic here

    return [(output_shape, None)]
```

2. **Register the function** in the `INFER_SHAPE_FUNC_MAPPING` dictionary:

```python
INFER_SHAPE_FUNC_MAPPING: dict[str, ShapeInferFunc] = {
    # ... existing entries ...
    "<Operator>": _infer_<operator>_shape,
}
```

3. **Add comprehensive test cases** in `tests/test_units/test_<feature>.py`:

```python
class TestOperator:
    def test_basic_case(self):
        """Test operator with standard inputs."""
        ctx = ShapeInferenceContext(
            data_shapes={"input": [2, 3, 4]},
            explicit_shapes={},
            initializers={},
            verbose=False,
        )
        node = onnx.helper.make_node("<Operator>", inputs=["input"], outputs=["output"])
        result = _infer_<operator>_shape(node, ctx)
        assert result[0][0] == [expected_shape]
```

4. **Verify all tests pass**:

```bash
pytest tests/test_units/ -q
```

### Code Style Guidelines

- Use **ruff** for formatting: `ruff format src/shapeonnx tests`
- Follow **PEP 257** docstring conventions
- Use **type hints** for all function signatures
- Keep functions focused and under 100 lines when possible
- Prefix private functions with underscore (`_function_name`)

## License

MIT License. See LICENSE file for details.
