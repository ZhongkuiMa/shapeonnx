# ShapeONNX

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/shapeonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/shapeonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/shapeonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/shapeonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Infer static tensor shapes for ONNX models where `onnx.shape_inference` fails or returns dynamic dimensions.

ShapeONNX tracks actual shape values through operator chains (`Shape -> Gather -> Concat -> Reshape`) that standard tools cannot resolve. Neural network verification tools need exact static shapes for bound propagation; ShapeONNX resolves shapes ONNX marks as dynamic to concrete values when statically computable.

| Scenario | `onnx.shape_inference` | ShapeONNX |
|----------|------------------------|-----------|
| Shape op output | `[4]` (tensor metadata) | `[1, 48, 2, 2]` (actual values) |
| Slice of shape tensor | `[2]` (tensor type) | `[1, 48]` (sliced values) |
| Concat of shapes | `[3]` (1D array) | `[1, 48, -1]` (reshape target) |
| ConstantOfShape | `[-1, -1, -1]` (dynamic) | `[1, 1, 48]` (concrete shape) |
| Batch dimension | `-1` (dynamic) | `1` (static when determinable) |

## Installation

```bash
git clone https://github.com/ZhongkuiMa/shapeonnx.git
cd shapeonnx
pip install -e ".[dev]"
```

Requires Python 3.11+, onnx, numpy.

## Quick Start

```python
import onnx
from shapeonnx import infer_onnx_shape
from shapeonnx.utils import (
    convert_constant_to_initializer,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)

model = onnx.load("model.onnx")
model = onnx.version_converter.convert_version(model, target_version=21)

initializers = get_initializers(model)
input_nodes = get_input_nodes(model, initializers, has_batch_dim=True)
output_nodes = get_output_nodes(model, has_batch_dim=True)
nodes = convert_constant_to_initializer(list(model.graph.node), initializers)

shapes = infer_onnx_shape(
    input_nodes, output_nodes, nodes, initializers, has_batch_dim=True
)
for name, shape in shapes.items():
    print(f"{name}: {shape}")
# conv1: [1, 48, 28, 28]
# reshape1: [1, 48, -1]
```

`convert_constant_to_initializer` must be called before `infer_onnx_shape` -- it moves Constant nodes into the initializer dict and removes them from the node list.

## API

| Function | Module | Description |
|----------|--------|-------------|
| `infer_onnx_shape` | `shapeonnx` | Infer shapes for all tensors. Returns `dict[str, int \| list[int]]` |
| `extract_io_shapes` | `shapeonnx` | Extract shapes from input/output `ValueInfoProto` nodes |
| `get_initializers` | `shapeonnx.utils` | Extract initializers as `dict[str, TensorProto]` |
| `get_input_nodes` | `shapeonnx.utils` | Get input nodes excluding initializers |
| `get_output_nodes` | `shapeonnx.utils` | Get output nodes with normalized shapes |
| `convert_constant_to_initializer` | `shapeonnx.utils` | Move Constant nodes to initializer dict (modifies in-place) |

## Supported Operators (51)

| Category | Operators |
|----------|-----------|
| Arithmetic | Add, Sub, Mul, Div, DivCst, Pow, Neg |
| Activation | Relu, LeakyRelu, Sigmoid, Tanh, Clip, Sin, Cos, Sign |
| Conv/Pool | Conv, Conv1d, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool |
| Normalization | BatchNormalization |
| Shape manipulation | Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand |
| Slicing/Concat | Slice, Split, Gather, Concat |
| Shape ops | Shape, ConstantOfShape, Range |
| Reduction | ReduceMean, ReduceSum, ArgMax |
| Comparison | Equal, Where, Max, Min |
| Matrix | MatMul, Gemm |
| Other | Cast, Dropout, Pad, Resize, Scatter, ScatterElements, ScatterND, Softmax, Floor |

## Limitations

- Constant nodes must be converted to initializers before inference (see Quick Start)
- Asymmetric padding in Conv/Pool not supported
- Control flow operators (If, Loop, Scan) not supported
- Assumes static input shapes (dynamic batch handled via `has_batch_dim` flag)

## Project Structure

```
shapeonnx/
├── src/shapeonnx/
│   ├── __init__.py       # Exports: infer_onnx_shape, extract_io_shapes
│   ├── infer_shape.py    # Shape inference engine + ShapeInferenceContext
│   ├── onnx_attrs.py     # ONNX attribute extraction
│   └── utils.py          # Model loading helpers
└── tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License -- see [LICENSE](LICENSE).
