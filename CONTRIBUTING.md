---
type: DESCRIPTION
note: "Descriptive. Mirrors current code; update to follow code changes. < functional code."
---

> **This file IS**: the shapeonnx-specific contributing workflow and conventions. **It is NOT**: a replacement for the root CONTRIBUTING.md (shared conventions live there).

# Contributing to ShapeONNX

Shared conventions (imports, formatting, docstrings) are in the root [CONTRIBUTING.md](../CONTRIBUTING.md).
This file covers shapeonnx-specific workflow only.

## Setup

```bash
cd shapeonnx
pip install -e ".[dev]"
pre-commit install
python -c "from shapeonnx import infer_onnx_shape; print('OK')"
```

## Checks

```bash
pre-commit run --all-files   # lint, format, type-check
pytest tests/test_units/ -q  # tests
```

## Adding a New ONNX Operator

Most contributions follow this pattern.

1. **Implement shape inference function** in `src/shapeonnx/infer_shape.py`:

   ```python
   def _infer_<operator>_shape(
       node: NodeProto,
       ctx: ShapeInferenceContext,
   ) -> list[tuple[int | list[int] | None, int | list[int] | None]]:
       """Infer shape for <Operator> node."""
       input_name = node.input[0]
       data_shape = ctx.data_shapes.get(input_name)
       output_shape = ...  # Your logic
       return [(output_shape, None)]
   ```

2. **Register** in `INFER_SHAPE_FUNC_MAPPING` (alphabetical order):

   ```python
   INFER_SHAPE_FUNC_MAPPING: dict[str, ShapeInferFunc] = {
       ...
       "<Operator>": _infer_<operator>_shape,
       ...
   }
   ```

3. **Add tests** in `tests/test_units/test_<operator>.py`:

   ```python
   class TestOperator:
       def test_basic_case(self):
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

4. **Verify**: `pytest tests/test_units/ -q`

## Constraints

| Rule | Details |
|------|---------|
| Absolute imports only | `from shapeonnx.xxx import yyy` (no relative) |
| `__docformat__` + `__all__` | Required in every module |
| Type hints | All function signatures |
| Frozen dataclass | `ShapeInferenceContext` is immutable |
| Constant nodes | Must be converted to initializers before inference |
| Return format | `list[tuple[data_shape, explicit_shape]]` per output |
