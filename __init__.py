"""ShapeONNX: Static shape inference for complex ONNX models."""

__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape", "extract_io_shapes"]

from .shapeonnx.infer_shape import extract_io_shapes, infer_onnx_shape
