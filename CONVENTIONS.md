---
type: LAW
note: "Normative. Code obeys this; change only via deliberate revision. Greater authority than functional code."
---

> **Law**. Conventions/rules for shapeonnx. Code obeys this. Change via deliberate revision.

# Shapeonnx Conventions

This file defines style and documentation conventions for the shapeonnx package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering architecture or usage guidance | ☐ |
| 1.3 | **Format**: ReST plain text, no `:param:` or `:return:` tags at module level | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use ASCII equivalents | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Core inference module (`infer_shape.py`) | Summary + paragraph covering architecture | `"""ONNX shape inference engine..."""` |
| Attrs extraction (`onnx_attrs.py`) | One line | `"""Extract operator attributes from ONNX nodes."""` |
| Utility module (`utils.py`) | One line | `"""Utility functions for shape inference."""` |
| `__init__.py` | Summary of public API with listed functions | `"""ONNX shape inference... Public API: infer_onnx_shape, extract_io_shapes."""` |

---

## 2. Function/Class Docstrings

### 2.1 Structure

```python
def infer_onnx_shape(model: ModelProto) -> ModelProto:
    """
    Short imperative description of what the function computes.

    Extended description (optional) — the algorithm or approach.

    :param param1: Description of param1 (capitalized, ends with period).
    :param param2: Description of param2.

    :return: Description of return value (capitalized, ends with period).
    :raises ValueError: When and why this exception is raised.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: imperative mood, describes what the function computes, ends with period | ☐ |
| 2.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags | ☐ |
| 2.3 | `:param` descriptions: **capitalized, end with period**, describe semantics not types | ☐ |
| 2.4 | `:return` description: **capitalized, end with period**; include output type description | ☐ |
| 2.5 | Private shape inference helpers (`_infer_*_shape`) may use a single-line docstring without `:param:` tags | ☐ |
| 2.6 | Public API functions (`infer_onnx_shape`, `extract_io_shapes`) require full `:param:` documentation | ☐ |
| 2.7 | No docstring on `__init__` of a dataclass (the class docstring covers it) | ☐ |
| 2.8 | **No non-ASCII characters** in docstrings | ☐ |

### 2.3 Good examples

```python
def infer_onnx_shape(model: ModelProto) -> ModelProto:
    """
    Infer and annotate shapes for all nodes in an ONNX model.

    Traverses the graph in topological order, computing output shapes
    for each node based on input shapes and operator semantics.

    :param model: ONNX model to annotate with shape information.
    :return: The same model with shape information added to each node's output.
    :raises ValueError: If shape inference fails for any node.
    """
```

```python
def _infer_conv_shape(node: NodeProto, context: ShapeInferenceContext) -> tuple[int, ...]:
    """Infer output shape for Conv operator."""
```

---

## 3. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | Comment **why**, not what — the code already says what | ☐ |
| 3.2 | Only add comments when the reasoning is non-obvious (shape arithmetic rationale, ONNX spec edge cases) | ☐ |
| 3.3 | **No inline shape comments** on function signatures — shapes belong in `:param:`/`:return:` docstrings | ☐ |
| 3.4 | No commented-out code — delete it | ☐ |
| 3.5 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001) | ☐ |

---

## 4. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | **Classes**: PascalCase — `ShapeInferenceContext` | ☐ |
| 4.2 | **Functions**: snake_case — `infer_onnx_shape`, `extract_io_shapes`, `_get_attrs_maxpool` | ☐ |
| 4.3 | **Shape inference helpers**: `_infer_` prefix — `_infer_conv_shape`, `_infer_matmul_shape`, `_infer_relu_shape` | ☐ |
| 4.4 | **Attrs extractors**: `_get_attrs_` prefix — `_get_attrs_maxpool`, `_get_attrs_conv` | ☐ |
| 4.5 | **Private modules**: `_` prefix — (currently none, but would follow pattern) | ☐ |
| 4.6 | **Constants**: UPPER_CASE — `EXTRACT_ATTRS_MAP`, `_DEFAULT_SHAPE` | ☐ |
| 4.7 | **ONNX protobuf names**: `NodeProto`, `TensorProto`, `ModelProto`, `ValueInfoProto` — match ONNX API exactly | ☐ |

---

## 5. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **100-char line length** (enforced by ruff) | ☐ |
| 5.2 | **Double quotes** for strings and docstrings | ☐ |
| 5.3 | **Absolute imports only** — `from shapeonnx.onnx_attrs import _get_onnx_attrs` | ☐ |
| 5.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 5.5 | `__all__` in every source module, alphabetically sorted, listing all public names | ☐ |
| 5.6 | **Import order**: stdlib → third-party (`numpy`, `onnx`) → first-party (`shapeonnx.*`). Groups separated by blank lines. | ☐ |
| 5.7 | `import numpy as np`; `from onnx import NodeProto, TensorProto` (for type annotations) | ☐ |
| 5.8 | **McCabe complexity ≤ 10** (enforced by ruff C90) | ☐ |
| 5.9 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 5.10 | **No string annotations** when type is already imported — write `-> ModelProto` not `-> "ModelProto"` | ☐ |
| 5.11 | Use `math` module for shape arithmetic (`math.prod`, `math.floor`) — not numpy for scalar math | ☐ |

---

## 6. Shape Inference Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | Entry points: `infer_onnx_shape(model)` for full-graph inference; `extract_io_shapes(model)` for input/output shapes only | ☐ |
| 6.2 | Shape inference functions return `tuple[int, ...]` for concrete shapes | ☐ |
| 6.3 | Dynamic dimensions (unknown at inference time) are represented as `-1` or string dimension names | ☐ |
| 6.4 | Shape inference helpers (`_infer_*_shape`) are stateless — they take inputs and return outputs without side effects | ☐ |
| 6.5 | `ShapeInferenceContext` holds mutable shared state (value_info, initializers) threaded through inference calls | ☐ |
| 6.6 | Topological order is determined by `onnx.helper.topological_sort()` or manual traversal of `model.graph.node` | ☐ |
| 6.7 | Unknown ops log a warning via `warnings.warn()` and propagate shapes as-is (pass-through) | ☐ |

---

## 7. Attrs Extraction Map Pattern

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | `EXTRACT_ATTRS_MAP: dict[str, Callable]` maps op_type → extractor function | ☐ |
| 7.2 | Map keys are ONNX op_type strings (lowercase): `"conv"`, `"maxpool"`, `"batchnorm"` | ☐ |
| 7.3 | Each extractor function takes `(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]` | ☐ |
| 7.4 | Factory function `_get_attrs_simple(attr_names)` generates extractors for ops with straightforward attribute extraction | ☐ |
| 7.5 | Complex ops (Conv, Gemm, BatchNorm) get hand-written extractors that handle weight-derived attributes | ☐ |
| 7.6 | New ops are added by: (1) write extractor in `onnx_attrs.py`, (2) add to `EXTRACT_ATTRS_MAP`, (3) add shape inference in `infer_shape.py` | ☐ |

---

## 8. Dataclass Context Pattern

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | `ShapeInferenceContext` is `@dataclass(frozen=True)` — immutable context | ☐ |
| 8.2 | Every field has an explicit type annotation | ☐ |
| 8.3 | Class docstring describes what the context holds and how it's used during inference | ☐ |
| 8.4 | Passed as the second argument to all `_infer_*_shape` functions for consistency | ☐ |

---

## 10. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`) and suffixes (`_DIR`, `_NAME`) for clarity | ☐ |
| 10.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder `_constants.py` → package-level. Promote when a second consumer at broader scope appears | ☐ |
| 10.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 10.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults already named by the parameter | ☐ |
| 10.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 10.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable `list` or `set` | ☐ |

---

## 11. Test Style

### 11.1 Directory Layout

```
tests/
├── test_arch/                 # architecture/import enforcement
├── test_benchmarks/           # integration tests (opt-in)
│   └── baselines/test/
└── test_units/
    └── test_shapeonnx/
        ├── __init__.py
        ├── conftest.py
        └── test_<concern>.py
```

### 11.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | **Test file naming**: `test_<concern>.py` — `test_infer_shape.py`, `test_onnx_attrs.py` | ☐ |
| 11.2 | **Test class naming**: `Test<Behavior>` — `TestConvShapeInference`, `TestAttrsExtraction` | ☐ |
| 11.3 | `__init__.py` at leaf `test_shapeonnx/` level (collision avoidance) | ☐ |
| 11.4 | `conftest.py` for shared fixtures (test ONNX models, sample nodes) | ☐ |
| 11.5 | **No pytest markers** except `@pytest.mark.parametrize` | ☐ |
| 11.6 | Test ONNX models built with `onnx.helper.make_model()` in fixtures or helpers | ☐ |
| 11.7 | Shape assertions: compare tuples directly — `assert result_shape == (1, 64, 112, 112)` | ☐ |
| 11.8 | Benchmark tests go in `test_benchmarks/` and are excluded from default `pytest` runs | ☐ |
| 11.9 | Test module docstrings: 1-3 lines max summarizing what the file validates | ☐ |
| 11.10 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Benchmark tests are opt-in | ☐ |
| 11.11 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment | ☐ |
