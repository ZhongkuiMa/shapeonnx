"""ShapeONNX: Static shape inference for complex ONNX models."""

from .shapeonnx.infer_shape import extract_io_shapes, infer_onnx_shape

__all__ = ["infer_onnx_shape", "extract_io_shapes"]
