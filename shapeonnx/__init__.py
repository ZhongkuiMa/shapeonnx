"""Core shape inference module."""

__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape", "extract_io_shapes"]

from .infer_shape import extract_io_shapes, infer_onnx_shape
