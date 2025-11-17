# ShapeONNX: Static Shape Inference for ONNX Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX 1.17](https://img.shields.io/badge/ONNX-1.17-brightgreen.svg)](https://onnx.ai)
[![NumPy 2.2](https://img.shields.io/badge/NumPy-2.2-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Static shape inference for complex ONNX models where standard tools fail.

## Why ShapeONNX?

ONNX's `onnx.shape_inference.infer_shapes` works for most models, but fails with inconsistent versions, non-standard conversions (PyTorch, custom manipulations), and dynamic shapes.

**ShapeONNX** solves this by:

- Simulating shape computations through a mini computation graph
- Propagating static values through shape operator chains (`Shape → Gather → Add`)
- Resolving intermediate shape tensors to constants when possible
- Converting dynamic operations to static equivalents

## Use Cases

Essential for:

1. **Shape operator chains** - `Shape`, `Gather`, `Slice`, `Add`, `Sub`, `Mul` operations
2. **Dynamic shape models** - Shape as model input
3. **Neural network verification** - Tools requiring precise static shapes (e.g., SlimONNX)
4. **Model optimization** - Static shape resolution before runtime

## Installation

```bash
pip install onnx==1.17.0 numpy==2.2.4
```

**Requirements:**
- Python 3.10+ (for built-in generics and type hints)
- ONNX 1.17.0 (IR version 21.0.0 baseline)

**Note:** Convert models to IR 21 using `onnx.version_converter` to ensure compatibility.

## Quick Start

```python
import onnx
from shapeonnx import infer_onnx_shape, extract_io_shapes
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
for name, shape in shapes.items():
    print(f"{name}: {shape}")
```

## API Reference

### Core Functions

#### `infer_onnx_shape`

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

Infer shapes for all tensors in an ONNX model.

**Parameters:**
- `input_nodes`: Model input value infos
- `output_nodes`: Model output value infos
- `nodes`: Model computation nodes (Constant nodes must be converted to initializers first)
- `initializers`: Model initializers (constants)
- `has_batch_dim`: Whether tensors have batch dimension
- `verbose`: Print debug information during inference

**Returns:** Dictionary mapping tensor names to inferred shapes

#### `extract_io_shapes`

```python
def extract_io_shapes(
    nodes: list[ValueInfoProto],
    has_batch_dim: bool
) -> dict[str, list[int]]
```

Extract shapes from model input/output nodes.

### Utility Functions

#### `convert_constant_to_initializer`

```python
def convert_constant_to_initializer(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto]
) -> list[NodeProto]
```

Convert Constant nodes to initializers (required preprocessing step).

#### `get_initializers`, `get_input_nodes`, `get_output_nodes`

Helper functions to extract model components with proper shape formatting.

## Architecture

**Design principles:**
- Immutable context (frozen dataclass)
- Pure functions, explicit inputs
- Direct dict access, minimal abstraction
- Full type hints

**Performance:**
- Single-pass O(1) forward propagation
- Pre-converted integer initializers
- Efficient operator dispatch via dict mapping
- Minimal allocations

**Benchmark:** 140 VNN-COMP 2024 models in ~6.5s (i5-12400F)

## Supported Operators

**46 operators across 10 categories:**

- **Arithmetic:** Add, Sub, Mul, Div, Pow, Neg
- **Activation:** Relu, LeakyRelu, Sigmoid, Tanh, Clip, Sin, Cos
- **Pooling:** Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool
- **Normalization:** BatchNormalization
- **Tensor:** Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand
- **Slicing:** Slice, Split, Gather, Concat
- **Shape:** Shape, ConstantOfShape, Range
- **Reduction:** ReduceMean, ReduceSum, ArgMax
- **Comparison:** Equal, Where, Max, Min
- **Matrix:** MatMul, Gemm
- **Other:** Cast, Dropout, Pad, Resize, Scatter, ScatterElements, ScatterND, Softmax, Floor, Sign

## Testing

```bash
cd shapeonnx/test
python test_baseline.py
# Expected: Tested: 140/140, Passed: 140/140
```

**Test suite:**
- 140 diverse VNN-COMP 2024 models (CNNs, ResNets, GANs, transformers)
- Baseline comparison for regression detection

## Contributing

**Process:**
1. Fork and create feature branch
2. Implement operator following existing patterns
3. Add tests and verify baselines pass
4. Submit PR (no direct pushes to main)

**Adding operators:**
```python
def infer_<op>_shape(
    node: NodeProto, ctx: ShapeInferenceContext
) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
    # Return [(data_shape, explicit_shape)]
```
Then register in `INFER_SHAPE_FUNC_MAPPING` and add tests.

## Limitations

- Constant nodes must be converted to initializers
- No asymmetric padding in Conv/Pool
- No control flow operators (If, Loop)
- Limited attribute support for some operators

## Related Projects

- **[SlimONNX](https://github.com/ZhongkuiMa/slimonnx)** - ONNX optimization using ShapeONNX

## License

MIT
