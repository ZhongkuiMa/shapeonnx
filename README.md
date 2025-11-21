# ShapeONNX

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX 1.17](https://img.shields.io/badge/ONNX-1.17-brightgreen.svg)](https://onnx.ai)
[![NumPy 2.2](https://img.shields.io/badge/NumPy-2.2-green.svg)](https://numpy.org/)
[![VNN-COMP 2024](https://img.shields.io/badge/VNN--COMP-2024-orange.svg)](https://sites.google.com/view/vnn2024)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Static shape inference for ONNX models where standard tools fail.

**Tested on all models from VNN-COMP 2024 with 100% success rate.**

## Overview

ONNX's built-in `onnx.shape_inference.infer_shapes` handles most models correctly, but fails in several critical scenarios:

- Models with inconsistent ONNX versions or opset mismatches
- Non-standard conversions from PyTorch or other frameworks
- Dynamic shape operations where shape computations depend on data
- Shape operator chains (`Shape → Gather → Add → Reshape`)
- Models with custom shape manipulations

ShapeONNX solves these problems through static shape computation:

- Simulates shape calculations through a mini computation graph
- Propagates static values through shape operator chains
- Resolves intermediate shape tensors to compile-time constants
- Converts dynamic shape operations to static equivalents
- Provides reliable shape inference for neural network verification

## Motivation

Neural network verification tools require precise static shapes for:

- Layer-by-layer bound propagation
- Memory allocation for symbolic execution
- Constraint generation for SMT solvers
- Model optimization and fusion (SlimONNX)

When ONNX shape inference fails, verification pipelines break. ShapeONNX fills this gap by providing robust static shape inference for the complex models encountered in verification research.

## Features

- **Robust Shape Inference**: Handles models where onnx.shape_inference fails
- **Shape Operator Chains**: Resolves `Shape → Gather → Slice → Add` patterns
- **Dynamic to Static**: Converts runtime shape computations to compile-time constants
- **46 Operators**: Comprehensive coverage across 10 operator categories
- **Fast Performance**: Single-pass O(1) forward propagation
- **Pure Python**: No C/C++ dependencies, easy integration
- **Production Ready**: Tested on 140 VNN-COMP 2024 models

## Use Cases

ShapeONNX is essential for:

1. **Neural Network Verification**: Tools requiring static shapes (α,β-CROWN, ERAN, Marabou)
2. **Model Optimization**: Pre-optimization shape resolution (SlimONNX)
3. **Shape-Dependent Transformations**: Operations requiring known tensor dimensions
4. **Complex Model Analysis**: Understanding shape propagation in non-standard models

## Installation

### Requirements

- Python 3.10 or higher
- onnx 1.17.0
- numpy 2.2.4

**Important**: ONNX version compatibility matters. Use the specified versions to avoid opset incompatibilities.

### Setup

```bash
pip install onnx==1.17.0 numpy==2.2.4
```

### Version Compatibility

- **ONNX 1.17.0**: Tested opset range 17-21
- **NumPy 2.2.4**: Required for Python 3.10+ compatibility

Models should be converted to ONNX IR version 21 using `onnx.version_converter` for maximum compatibility.

## Quick Start

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.shapeonnx.utils import (
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

ShapeONNX supports 46 operators across 10 categories:

### Arithmetic Operations
Add, Sub, Mul, Div, Pow, Neg

### Activation Functions
Relu, LeakyRelu, Sigmoid, Tanh, Clip, Sin, Cos

### Convolution and Pooling
Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool

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
Cast, Dropout, Pad, Resize, Scatter, ScatterElements, ScatterND, Softmax, Floor, Sign

## Architecture

### Design Principles

- **Immutable Context**: Frozen dataclass for shape inference context
- **Pure Functions**: All shape inference functions are stateless with explicit inputs
- **Direct Dictionary Access**: Minimal abstraction for performance
- **Full Type Hints**: Complete type annotations using Python 3.10+ syntax

### Performance Characteristics

- **Single-Pass Forward Propagation**: O(1) complexity per operator
- **Pre-Converted Initializers**: Integer tensors converted once at initialization
- **Efficient Operator Dispatch**: Dictionary-based operator function mapping
- **Minimal Memory Allocations**: Shape lists reused where possible

**Benchmark**: 140 VNN-COMP 2024 models processed in approximately 6.5 seconds on Intel i5-12400F.

### Module Structure

```
shapeonnx/
├── __init__.py              # Public API
├── infer_shape.py           # Main shape inference
├── utils.py                 # Utility functions
├── context.py               # Shape inference context
└── operators/               # Operator-specific inference
    ├── arithmetic.py        # Add, Sub, Mul, Div, Pow, Neg
    ├── activation.py        # Relu, Sigmoid, Tanh, etc.
    ├── conv_pool.py         # Conv, Pool operations
    ├── normalization.py     # BatchNormalization
    ├── tensor_ops.py        # Reshape, Transpose, etc.
    ├── slicing.py           # Slice, Gather, Concat
    ├── shape_ops.py         # Shape, ConstantOfShape
    ├── reduction.py         # ReduceMean, ReduceSum
    ├── comparison.py        # Equal, Where, Max, Min
    └── matrix.py            # MatMul, Gemm
```

## Examples

### Example 1: Basic Shape Inference

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.shapeonnx.utils import (
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
from shapeonnx.shapeonnx.utils import (
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
from shapeonnx.shapeonnx.utils import (
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

- **Total Models Tested**: 140 diverse neural networks
- **Success Rate**: 100% (all models successfully processed)
- **Model Types**: CNNs, ResNets, VGG, GANs, Transformers, Graph Neural Networks
- **Opset Coverage**: Opset 17-21

### Run Tests

```bash
cd shapeonnx/test
python test_baseline.py
```

**Expected output**: `Tested: 140/140, Passed: 140/140`

### Baseline Testing

The test suite includes baseline comparison to detect regressions. Shapes for all 140 models are stored in `test/baselines/` and compared against current inference results.

## Performance Benchmarks

**Hardware**: Intel i5-12400F (6 cores, 12 threads)

**Results**:
- 100+ VNN-COMP models: ~6.5 seconds total
- Average per model: ~46 milliseconds
- Complex models (VIT, ResNet50): <200ms
- Simple models (ACAS Xu): <10ms

**Memory**: Typical peak memory usage under 500MB for largest models.

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

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Implement operator following existing patterns in `operators/` directory
4. Add tests and verify baseline tests pass
5. Run black formatter on all modified files
6. Submit a pull request

Direct pushes to main branch are restricted.

### Adding New Operators

To add a new operator:

```python
def infer_<operator>_shape(
    node: NodeProto,
    ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    """Infer shape for <Operator> node.

    :param node: ONNX node
    :param ctx: Shape inference context
    :return: List of (data_shape, explicit_shape) tuples
    """
    # Implementation
    return [(output_shape, None)]
```

Then register in `INFER_SHAPE_FUNC_MAPPING` dictionary and add test cases.

## License

MIT License. See LICENSE file for details.

