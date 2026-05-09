# ShapeONNX Architecture

Static shape inference for ONNX models -- resolves shapes that `onnx.shape_inference` leaves dynamic.

## Package Tree

```
shapeonnx/
├── src/shapeonnx/
│   ├── __init__.py       Public API (infer_onnx_shape, extract_io_shapes)
│   ├── infer_shape.py    Shape inference engine + all operator implementations
│   │                     (modify when: adding/fixing operators)
│   ├── onnx_attrs.py     ONNX attribute extraction from nodes
│   │                     (modify when: new attribute types needed)
│   └── utils.py          Model loading helpers (get_initializers, get_input_nodes, etc.)
│                         (modify when: changing model preprocessing)
└── tests/
    ├── test_units/       Per-operator unit tests
    └── test_benchmarks/  Full-model regression tests (manual)
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add ONNX operator | `src/shapeonnx/infer_shape.py` | Add to `INFER_SHAPE_FUNC_MAPPING`, add test in `tests/test_units/` | `utils.py`, `onnx_attrs.py` | Return `list[tuple[data_shape, explicit_shape]]` (enforced) | `NotImplementedError` at runtime |
| Fix operator shape logic | `src/shapeonnx/infer_shape.py` | Update/add test case | Changing `ShapeInferenceContext` fields | Must not break existing tests (enforced) | `pytest tests/test_units/` failure |
| Add new attribute extraction | `src/shapeonnx/onnx_attrs.py` | Use in `infer_shape.py` via `_get_onnx_attrs` | Duplicating extraction logic inline | Must handle missing attrs with defaults (observed) | `KeyError` at runtime |
| Change model preprocessing | `src/shapeonnx/utils.py` | Update `__init__.py` exports if new public function | `infer_shape.py` internals | Constant nodes must be converted before inference (enforced) | Shape inference fails on Constant nodes |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| `infer_shape.py` imports from `onnx_attrs.py` and `utils.py` only | (observed) | Circular import |
| `onnx_attrs.py` and `utils.py` are independent (no cross-imports) | (observed) | Circular import |
| Absolute imports only (`from shapeonnx.xxx`) | (enforced) ruff TID | `ruff check` failure |
| No external deps beyond onnx + numpy | (enforced) pyproject.toml | Install failure |

## Key Abstractions

| Abstraction | What it is | Modification note |
|-------------|-----------|------------------|
| `ShapeInferenceContext` | Frozen dataclass; carries `data_shapes`/`explicit_shapes` dicts mutated during traversal | Add fields here to pass new inference state |
| `INFER_SHAPE_FUNC_MAPPING` | Dispatch dict: op_type string → inference function | Register new operators here (alphabetical) |
| Return format | `list[tuple[data_shape, explicit_shape]]` per output | `explicit_shape` is non-None only for shape-producing ops (Shape, Gather on shape, Concat of shapes) |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Adding operator but not registering in `INFER_SHAPE_FUNC_MAPPING` | `NotImplementedError` at runtime | Add entry in alphabetical order |
| Forgetting to handle `explicit_shapes` for shape-producing ops (Shape, Gather on shape, Concat of shapes) | Downstream Reshape/ConstantOfShape fails | Return explicit values in second tuple element |
| Not calling `convert_constant_to_initializer` before inference | `RuntimeError: Constant nodes must be converted` | Always preprocess with `utils.convert_constant_to_initializer` |

## Conventions

- Operator functions: `_infer_<op>_shape(node, ctx) -> list[tuple[...]]`
- Shape-preserving ops reuse `_infer_nochange_op_shape`
- Binary broadcast ops reuse `_infer_binary_op_shape`
- Attributes extracted via `_get_onnx_attrs(node, default_attrs_dict)`

## Related Documents

- [README.md](README.md) -- API reference and supported operators
- [CONTRIBUTING.md](CONTRIBUTING.md) -- development workflow
- [../ARCHITECTURE.md](../ARCHITECTURE.md) -- parent project architecture
