# === INFERRED IMPORT CONTRACTS (review before committing) ===
#
# Package: shapeonnx
# Structure: flat package — all modules at the same level (no subpackage layers)
#   src/shapeonnx/__init__.py
#   src/shapeonnx/infer_shape.py
#   src/shapeonnx/onnx_attrs.py
#   src/shapeonnx/utils.py
#
# ALLOWED (observed in source):
#   __init__    -> infer_shape, onnx_attrs, utils  (re-exports)
#   infer_shape -> onnx_attrs, utils
#
# FORBIDDEN (no lower-layer importing higher-layer applies here — flat structure):
#   No inferred forbidden edges for a flat package.
#   ARC3 layer boundary tests are N/A: all modules are peers.
#   If new subpackages are introduced, revisit this file.
#
# [REVIEW] Approve the smoke test before treating ARC2 as resolved.
# ================================================================
"""Import architecture tests for shapeonnx."""

__docformat__ = "restructuredtext"

import importlib
from pathlib import Path

_SRC = Path(__file__).parent.parent.parent / "src" / "shapeonnx"


class TestImportSmoke:
    """Package imports without circular dependencies or broken __init__ chains."""

    def test_top_level_import(self):
        """Import shapeonnx top-level without error."""
        mod = importlib.import_module("shapeonnx")
        assert mod.__name__ == "shapeonnx"

    def test_submodule_imports_cleanly(self):
        """Every submodule imports without ImportError."""
        errors: list[str] = []
        for f in _SRC.rglob("*.py"):
            rel = f.relative_to(_SRC.parent).with_suffix("")
            mod = ".".join(rel.parts)
            try:
                importlib.import_module(mod)
            except ImportError as e:
                errors.append(f"{mod}: {e}")
        assert not errors, "Import errors:\n" + "\n".join(errors)
