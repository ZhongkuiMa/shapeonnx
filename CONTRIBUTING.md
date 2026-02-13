# Contributing to ShapeONNX

## Development Setup

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

## Development Workflow

Before making any changes, create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Work on your changes and commit regularly:

```bash
git add .
git commit -m "Description of your changes"
```

## Code Quality Standards

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

### Code Style Guidelines

- Use **ruff** for formatting: `ruff format src/shapeonnx tests`
- Follow **PEP 257** docstring conventions
- Use **type hints** for all function signatures
- Keep functions focused and under 100 lines when possible
- Prefix private functions with underscore (`_function_name`)

## Pull Request Guidelines

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

## Testing

### Run Tests

```bash
cd shapeonnx
pytest tests/test_shapeonnx.py -v                    # Run shape inference tests
pytest tests/test_shapeonnx_regression.py -v          # Run regression tests with ONNX comparison
```

**Expected output**: `414 passed` (138 models × 3 test types)

## Adding New Operators

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

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue with reproducible example
- **Feature requests**: Open an Issue with use case description
