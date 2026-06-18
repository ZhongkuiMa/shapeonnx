"""Core shape inference module.

.. seealso:: ``shapeonnx/CONVENTIONS.md`` for code conventions.
"""

__version__ = "2026.6.0"

__docformat__ = "restructuredtext"
__all__ = ["__version__", "extract_io_shapes", "infer_onnx_shape"]

from shapeonnx.infer_shape import (
    extract_io_shapes,
    infer_onnx_shape,
)
