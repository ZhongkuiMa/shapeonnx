"""Core shape inference module."""

from .infer_shape import extract_io_shapes, infer_onnx_shape

__all__ = ["infer_onnx_shape", "extract_io_shapes"]
