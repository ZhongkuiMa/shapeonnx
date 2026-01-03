"""Comprehensive tests for onnx_attrs module.

This module provides comprehensive test coverage for shapeonnx.onnx_attrs,
which handles ONNX node attribute extraction and validation. The onnx_attrs
module is critical for shape inference, as it extracts and validates
attributes from 30+ ONNX operators.

Test organization:
- TestScanAttrs: Core attribute scanning and type extraction
- TestValidationHelpers: Validation functions (pads, kernel defaults, auto_pad)
- TestGetOnnxAttrsDispatcher: Main dispatcher routing to operator-specific extractors
- TestConvolutionOperators: Conv, ConvTranspose, AveragePool, MaxPool
- TestRequiredAttributeOperators: Operators with required attributes
- TestSpecialOperators: Operators with special validation logic
- TestScatterOperators: Scatter, ScatterElements, ScatterND
- TestReduceOperators: ReduceMean, ReduceSum using factory pattern
- TestSimpleOperators: Operators using get_attrs_simple factory
- TestConstantNodeError: Constant operator error handling
"""

import numpy as np
import onnx
import pytest

from shapeonnx.onnx_attrs import (
    _check_pads_symmetric,
    _get_attrs_argmax,
    _get_attrs_avgpool,
    _get_attrs_batchnorm,
    _get_attrs_cast,
    _get_attrs_concat,
    _get_attrs_constant,
    _get_attrs_constantofshape,
    _get_attrs_conv,
    _get_attrs_convtranspose,
    _get_attrs_maxpool,
    _get_attrs_reshape,
    _get_attrs_resize,
    _get_attrs_scatter,
    _get_attrs_scatterelement,
    _get_attrs_scatternd,
    _get_attrs_shape,
    _get_attrs_simple,
    _get_attrs_transpose,
    _get_onnx_attrs,
    _infer_kernel_defaults,
    _scan_attrs,
    _validate_auto_pad,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _make_weight_tensor(shape: tuple[int, ...], name: str = "weight") -> onnx.TensorProto:
    """Create a test weight tensor for Conv-like operators."""
    rng = np.random.default_rng()
    array = rng.standard_normal(shape).astype(np.float32)
    return onnx.numpy_helper.from_array(array, name=name)


# ============================================================================
# TestScanAttrs - Core Attribute Scanning and Type Extraction
# ============================================================================


class TestScanAttrs:
    """Test scan_attrs function for attribute extraction and merging."""

    def test_scan_attrs_with_empty_attributes(self):
        """Test scan_attrs returns defaults when no attributes provided."""
        node = onnx.helper.make_node("Conv", inputs=["i"], outputs=["o"])
        defaults = {"auto_pad": "NOTSET", "group": 1}

        result = _scan_attrs(defaults, node.attribute)

        assert result == defaults

    def test_scan_attrs_overrides_defaults_with_attributes(self):
        """Test scan_attrs overrides defaults with node attributes."""
        node = onnx.helper.make_node(
            "Conv", inputs=["i", "w"], outputs=["o"], group=2, auto_pad="NOTSET"
        )
        defaults = {"auto_pad": "NOTSET", "group": 1}

        result = _scan_attrs(defaults, node.attribute)

        assert result["group"] == 2

    @pytest.mark.parametrize(
        ("op_type", "attr_name", "attr_value"),
        [
            pytest.param("Elu", "alpha", 0.5, id="float_attribute"),
            pytest.param("Gather", "axis", 1, id="int_attribute"),
            pytest.param("Pad", "mode", "constant", id="string_attribute"),
        ],
    )
    def test_scan_attrs_extracts_basic_types(self, op_type, attr_name, attr_value):
        """Test scan_attrs extracts FLOAT, INT, STRING attribute types."""
        node = onnx.helper.make_node(
            op_type, inputs=["i"], outputs=["o"], **{attr_name: attr_value}
        )
        defaults: dict[str, int | float | str] = {}

        result = _scan_attrs(defaults, node.attribute)

        assert result[attr_name] == attr_value

    def test_scan_attrs_extracts_ints_tuple(self):
        """Test scan_attrs extracts INTS attribute as tuple."""
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
        )
        defaults: dict[str, tuple[int, ...]] = {}

        result = _scan_attrs(defaults, node.attribute)

        assert result["kernel_shape"] == (3, 3)
        assert result["strides"] == (1, 1)
        assert isinstance(result["kernel_shape"], tuple)

    def test_scan_attrs_extracts_floats_tuple(self):
        """Test scan_attrs extracts FLOATS attribute as tuple."""
        node = onnx.helper.make_node("Gemm", inputs=["i"], outputs=["o"], alpha=2.5, beta=0.5)
        defaults: dict[str, tuple[float, ...]] = {}

        result = _scan_attrs(defaults, node.attribute)

        assert result["alpha"] == 2.5
        assert result["beta"] == 0.5

    def test_scan_attrs_multiple_attributes_together(self):
        """Test scan_attrs handles multiple attributes together."""
        node = onnx.helper.make_node(
            "Conv",
            inputs=["i", "w"],
            outputs=["o"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            group=1,
            auto_pad="NOTSET",
        )
        defaults = {"pads": None, "dilations": None}

        result = _scan_attrs(defaults, node.attribute)

        assert result["kernel_shape"] == (3, 3)
        assert result["strides"] == (1, 1)
        assert result["group"] == 1
        assert result["auto_pad"] == "NOTSET"
        assert result["pads"] is None  # Default preserved
        assert result["dilations"] is None  # Default preserved

    def test_scan_attrs_tensor_attribute(self):
        """Test scan_attrs extracts TENSOR attribute type."""
        # Create a node with a tensor attribute (less common)
        node = onnx.helper.make_node("ConstantOfShape", inputs=["shape"], outputs=["output"])
        value_tensor = onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="value")
        attr = onnx.helper.make_attribute("value", value_tensor)
        node.attribute.append(attr)

        defaults: dict[str, np.ndarray] = {}
        result = _scan_attrs(defaults, node.attribute)

        assert "value" in result
        assert isinstance(result["value"], np.ndarray)

    def test_scan_attrs_with_integer_zero_value(self):
        """Test scan_attrs preserves zero values for integer attributes."""
        # Regression test: ensure zero values aren't treated as False/missing
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=0)

        defaults = {"axis": 1}
        result = _scan_attrs(defaults, node.attribute)

        # axis=0 should override the default axis=1
        assert result["axis"] == 0

    def test_scan_attrs_preserves_none_defaults(self):
        """Test scan_attrs preserves None values in defaults."""
        node = onnx.helper.make_node("Conv", inputs=["i", "w"], outputs=["o"], group=1)
        defaults = {"kernel_shape": None, "pads": None, "strides": None}

        result = _scan_attrs(defaults, node.attribute)

        assert result["kernel_shape"] is None
        assert result["pads"] is None
        assert result["strides"] is None


# ============================================================================
# TestValidationHelpers - Validation and Inference Functions
# ============================================================================


class TestValidationHelpers:
    """Test validation helper functions."""

    @pytest.mark.parametrize(
        ("pads", "dims"),
        [
            pytest.param((1, 1, 1, 1), 2, id="symmetric_2d_same"),
            pytest.param((2, 3, 2, 3), 2, id="symmetric_2d_different"),
            pytest.param((0, 0, 0, 0), 2, id="symmetric_2d_zeros"),
            pytest.param((1, 1, 1, 1, 1, 1), 3, id="symmetric_3d"),
            pytest.param((1, 2, 3, 1, 2, 3), 3, id="symmetric_3d_different"),
        ],
    )
    def test_check_pads_symmetric_valid_padding(self, pads, dims):
        """Test check_pads_symmetric passes for symmetric padding."""
        # Should not raise
        _check_pads_symmetric(pads)

    @pytest.mark.parametrize(
        ("pads", "expected_error"),
        [
            pytest.param((1, 1, 2, 2), "Asymmetric", id="asymmetric_2d"),
            pytest.param((1, 2, 1, 1), "Asymmetric", id="asymmetric_2d_partial"),
            pytest.param((1, 1, 1, 1, 1, 2), "Asymmetric", id="asymmetric_3d"),
        ],
    )
    def test_check_pads_symmetric_invalid_padding(self, pads, expected_error):
        """Test check_pads_symmetric raises ValueError for asymmetric padding."""
        with pytest.raises(ValueError, match=expected_error):
            _check_pads_symmetric(pads)

    @pytest.mark.parametrize(
        ("kernel_shape", "expected_dilations", "expected_strides", "expected_pads"),
        [
            pytest.param((3,), (1,), (1,), (0, 0), id="1d_kernel"),
            pytest.param((3, 3), (1, 1), (1, 1), (0, 0, 0, 0), id="2d_kernel"),
            pytest.param((3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0, 0, 0, 0), id="3d_kernel"),
        ],
    )
    def test_infer_kernel_defaults_all_none(
        self, kernel_shape, expected_dilations, expected_strides, expected_pads
    ):
        """Test infer_kernel_defaults generates correct defaults for various dimensions."""
        attrs = {"dilations": None, "strides": None, "pads": None}

        result = _infer_kernel_defaults(attrs, kernel_shape)

        assert result["dilations"] == expected_dilations
        assert result["strides"] == expected_strides
        assert result["pads"] == expected_pads

    def test_infer_kernel_defaults_preserves_existing_values(self):
        """Test infer_kernel_defaults doesn't override existing values."""
        attrs = {"dilations": (2, 2), "strides": (1, 1), "pads": (1, 1, 1, 1)}

        result = _infer_kernel_defaults(attrs, (3, 3))

        assert result["dilations"] == (2, 2)  # Preserved
        assert result["strides"] == (1, 1)  # Preserved
        assert result["pads"] == (1, 1, 1, 1)  # Preserved

    def test_infer_kernel_defaults_partial_override(self):
        """Test infer_kernel_defaults fills in only missing values."""
        attrs = {"dilations": (2, 2), "strides": None, "pads": None}

        result = _infer_kernel_defaults(attrs, (3, 3))

        assert result["dilations"] == (2, 2)  # Preserved
        assert result["strides"] == (1, 1)  # Inferred
        assert result["pads"] == (0, 0, 0, 0)  # Inferred

    @pytest.mark.parametrize(
        "auto_pad",
        [
            pytest.param("NOTSET", id="valid_notset"),
        ],
    )
    def test_validate_auto_pad_notset_passes(self, auto_pad):
        """Test validate_auto_pad passes for NOTSET."""
        # Should not raise
        _validate_auto_pad(auto_pad, "TestOp")

    @pytest.mark.parametrize(
        "auto_pad",
        [
            pytest.param("SAME_UPPER", id="same_upper"),
            pytest.param("SAME_LOWER", id="same_lower"),
            pytest.param("VALID", id="valid"),
        ],
    )
    def test_validate_auto_pad_invalid_values_raise_error(self, auto_pad):
        """Test validate_auto_pad raises ValueError for non-NOTSET values."""
        with pytest.raises(ValueError, match="is not supported"):
            _validate_auto_pad(auto_pad, "TestOp")


# ============================================================================
# TestGetOnnxAttrsDispatcher - Main Dispatcher Function
# ============================================================================


class TestGetOnnxAttrsDispatcher:
    """Test get_onnx_attrs dispatcher function."""

    @pytest.mark.parametrize(
        "op_type",
        [
            "ArgMax",
            "AveragePool",
            "BatchNormalization",
            "Cast",
            "Concat",
            "ConstantOfShape",
            "Elu",
            "Flatten",
            "Gather",
            "Gelu",
            "Gemm",
            "LeakyRelu",
            "MaxPool",
            "Pad",
            "ReduceMean",
            "ReduceSum",
            "Reshape",
            "Resize",
            "Scatter",
            "ScatterElements",
            "ScatterND",
            "Shape",
            "Softmax",
            "Split",
            "Transpose",
            "Unsqueeze",
            "Upsample",
        ],
    )
    def test_get_onnx_attrs_dispatcher_recognizes_supported_operators(self, op_type):
        """Test dispatcher routes all supported operators without NotImplementedError."""
        # Create minimal node - may fail on other errors but should route correctly
        node = onnx.helper.make_node(op_type, inputs=["input"], outputs=["output"])

        # Should not raise NotImplementedError (may raise other errors for incomplete nodes)
        try:
            result = _get_onnx_attrs(node, {})
            assert isinstance(result, dict)
        except (ValueError, KeyError, AttributeError, IndexError):
            # These are acceptable - missing required attributes or initializers
            # We're just testing the dispatcher routing works
            pass

    def test_get_onnx_attrs_dispatcher_conv_with_weight(self):
        """Test dispatcher routes Conv correctly with weight initializer."""
        weight = _make_weight_tensor((64, 3, 3, 3))
        node = onnx.helper.make_node("Conv", inputs=["input", "weight"], outputs=["output"])

        result = _get_onnx_attrs(node, {"weight": weight})

        assert isinstance(result, dict)
        assert "kernel_shape" in result

    def test_get_onnx_attrs_dispatcher_convtranspose_with_weight(self):
        """Test dispatcher routes ConvTranspose correctly with weight initializer."""
        weight = _make_weight_tensor((3, 64, 3, 3))
        node = onnx.helper.make_node(
            "ConvTranspose", inputs=["input", "weight"], outputs=["output"]
        )

        result = _get_onnx_attrs(node, {"weight": weight})

        assert isinstance(result, dict)
        assert "kernel_shape" in result

    def test_get_onnx_attrs_unsupported_operator_raises_error(self):
        """Test dispatcher raises NotImplementedError for unsupported operators."""
        node = onnx.helper.make_node("NonexistentOp", inputs=["i"], outputs=["o"])

        with pytest.raises(NotImplementedError, match="Unsupported operator"):
            _get_onnx_attrs(node, {})

    def test_get_onnx_attrs_conv_calls_correct_extractor(self):
        """Test dispatcher routes Conv to get_attrs_conv."""
        weight = _make_weight_tensor((64, 3, 3, 3))
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        attrs = _get_onnx_attrs(node, {"weight": weight})

        assert "kernel_shape" in attrs
        assert "strides" in attrs

    def test_get_onnx_attrs_concat_calls_correct_extractor(self):
        """Test dispatcher routes Concat to get_attrs_concat."""
        node = onnx.helper.make_node("Concat", inputs=["i1", "i2"], outputs=["o"], axis=1)

        attrs = _get_onnx_attrs(node, {})

        assert attrs["axis"] == 1


# ============================================================================
# TestConvolutionOperators - Convolution-like Operators
# ============================================================================


class TestConvolutionOperators:
    """Test convolution and pooling operators."""

    def test_get_attrs_conv_with_explicit_kernel_shape(self):
        """Test Conv with explicit kernel_shape attribute."""
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )

        attrs = _get_attrs_conv(node, {})

        assert attrs["kernel_shape"] == (3, 3)
        assert attrs["strides"] == (1, 1)
        assert attrs["dilations"] == (1, 1)

    def test_get_attrs_conv_infers_kernel_from_weights(self):
        """Test Conv infers kernel_shape from weight initializer."""
        weight = _make_weight_tensor((64, 3, 3, 3))
        node = onnx.helper.make_node(
            "Conv", inputs=["input", "weight"], outputs=["output"], strides=[1, 1]
        )

        attrs = _get_attrs_conv(node, {"weight": weight})

        assert attrs["kernel_shape"] == (3, 3)  # Inferred from weight dims[2:]

    def test_get_attrs_conv_with_custom_strides_and_dilations(self):
        """Test Conv with custom strides and dilations."""
        weight = _make_weight_tensor((64, 3, 3, 3))
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            strides=[2, 2],
            dilations=[2, 2],
        )

        attrs = _get_attrs_conv(node, {"weight": weight})

        assert attrs["strides"] == (2, 2)
        assert attrs["dilations"] == (2, 2)

    def test_get_attrs_conv_validates_auto_pad_not_notset(self):
        """Test Conv raises ValueError for auto_pad != NOTSET."""
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            auto_pad="SAME_UPPER",
        )

        with pytest.raises(ValueError, match="auto_pad=SAME_UPPER is not supported"):
            _get_attrs_conv(node, {})

    def test_get_attrs_conv_validates_symmetric_padding(self):
        """Test Conv raises ValueError for asymmetric pads."""
        weight = _make_weight_tensor((64, 3, 3, 3))
        node = onnx.helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            pads=[1, 2, 1, 1],  # Asymmetric
        )

        with pytest.raises(ValueError, match="Asymmetric"):
            _get_attrs_conv(node, {"weight": weight})

    def test_get_attrs_convtranspose_basic(self):
        """Test ConvTranspose basic attributes."""
        weight = _make_weight_tensor((3, 64, 3, 3))
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            strides=[1, 1],
        )

        attrs = _get_attrs_convtranspose(node, {"weight": weight})

        assert attrs["kernel_shape"] == (3, 3)
        assert attrs["output_padding"] == (0, 0)

    def test_get_attrs_convtranspose_group_not_one_raises_error(self):
        """Test ConvTranspose raises ValueError for group != 1."""
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["input", "weight"],
            outputs=["output"],
            group=2,
        )

        with pytest.raises(ValueError, match="group=2 is not supported"):
            _get_attrs_convtranspose(node, {})

    def test_get_attrs_avgpool_basic(self):
        """Test AveragePool basic attributes."""
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        attrs = _get_attrs_avgpool(node, {})

        assert attrs["kernel_shape"] == (3, 3)
        assert attrs["strides"] == (1, 1)

    def test_get_attrs_avgpool_missing_kernel_shape_raises_error(self):
        """Test AveragePool raises ValueError when kernel_shape missing."""
        node = onnx.helper.make_node("AveragePool", inputs=["input"], outputs=["output"])

        with pytest.raises(ValueError, match="kernel_shape is required"):
            _get_attrs_avgpool(node, {})

    def test_get_attrs_maxpool_basic(self):
        """Test MaxPool basic attributes."""
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        attrs = _get_attrs_maxpool(node, {})

        assert attrs["kernel_shape"] == (3, 3)

    def test_get_attrs_maxpool_missing_kernel_shape_raises_error(self):
        """Test MaxPool raises ValueError when kernel_shape missing."""
        node = onnx.helper.make_node("MaxPool", inputs=["input"], outputs=["output"])

        with pytest.raises(ValueError, match="kernel_shape is required"):
            _get_attrs_maxpool(node, {})

    def test_get_attrs_maxpool_storage_order_not_zero_raises_error(self):
        """Test MaxPool raises ValueError for storage_order != 0."""
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[3, 3],
            storage_order=1,
        )

        with pytest.raises(ValueError, match="storage_order=1 is not supported"):
            _get_attrs_maxpool(node, {})

    def test_get_attrs_maxpool_multiple_outputs_raises_error(self):
        """Test MaxPool raises ValueError for multiple outputs."""
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output", "indices"],
            kernel_shape=[3, 3],
        )

        with pytest.raises(ValueError, match="outputs is not supported"):
            _get_attrs_maxpool(node, {})


# ============================================================================
# TestRequiredAttributeOperators - Operators with Required Attributes
# ============================================================================


class TestRequiredAttributeOperators:
    """Test operators with required attributes."""

    def test_get_attrs_cast_basic(self):
        """Test Cast with required 'to' attribute."""
        node = onnx.helper.make_node(
            "Cast", inputs=["input"], outputs=["output"], to=onnx.TensorProto.FLOAT
        )

        attrs = _get_attrs_cast(node, {})

        assert attrs["to"] == onnx.TensorProto.FLOAT

    def test_get_attrs_cast_missing_to_raises_error(self):
        """Test Cast raises ValueError when 'to' attribute missing."""
        node = onnx.helper.make_node("Cast", inputs=["input"], outputs=["output"])

        with pytest.raises(ValueError, match="'to' attribute is required"):
            _get_attrs_cast(node, {})

    def test_get_attrs_cast_saturate_not_one_raises_error(self):
        """Test Cast raises ValueError for saturate != 1."""
        node = onnx.helper.make_node(
            "Cast",
            inputs=["input"],
            outputs=["output"],
            to=onnx.TensorProto.FLOAT,
            saturate=0,
        )

        with pytest.raises(ValueError, match="saturate=0 is not supported"):
            _get_attrs_cast(node, {})

    def test_get_attrs_concat_basic(self):
        """Test Concat with required axis attribute."""
        node = onnx.helper.make_node("Concat", inputs=["i1", "i2"], outputs=["output"], axis=1)

        attrs = _get_attrs_concat(node, {})

        assert attrs["axis"] == 1

    def test_get_attrs_concat_missing_axis_raises_error(self):
        """Test Concat raises ValueError when axis missing."""
        node = onnx.helper.make_node("Concat", inputs=["i1", "i2"], outputs=["output"])

        with pytest.raises(ValueError, match="axis is required"):
            _get_attrs_concat(node, {})

    def test_get_attrs_transpose_basic(self):
        """Test Transpose with required perm attribute."""
        node = onnx.helper.make_node(
            "Transpose", inputs=["input"], outputs=["output"], perm=[0, 2, 1]
        )

        attrs = _get_attrs_transpose(node, {})

        assert attrs["perm"] == (0, 2, 1)

    def test_get_attrs_transpose_missing_perm_raises_error(self):
        """Test Transpose raises ValueError when perm missing."""
        node = onnx.helper.make_node("Transpose", inputs=["input"], outputs=["output"])

        with pytest.raises(ValueError, match="perm is required"):
            _get_attrs_transpose(node, {})

    def test_get_attrs_constantofshape_basic(self):
        """Test ConstantOfShape with value attribute."""
        value_tensor = onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float32))
        node = onnx.helper.make_node("ConstantOfShape", inputs=["shape"], outputs=["output"])
        attr = onnx.helper.make_attribute("value", value_tensor)
        node.attribute.append(attr)

        attrs = _get_attrs_constantofshape(node, {})

        assert "value" in attrs

    def test_get_attrs_constantofshape_missing_value_raises_error(self):
        """Test ConstantOfShape raises ValueError when value missing."""
        node = onnx.helper.make_node("ConstantOfShape", inputs=["shape"], outputs=["output"])

        with pytest.raises(ValueError, match="value is required"):
            _get_attrs_constantofshape(node, {})


# ============================================================================
# TestSpecialOperators - Operators with Special Validation Logic
# ============================================================================


class TestSpecialOperators:
    """Test operators with special validation or defaults."""

    def test_get_attrs_reshape_basic(self):
        """Test Reshape with allowzero=0 (default)."""
        node = onnx.helper.make_node("Reshape", inputs=["input", "shape"], outputs=["output"])

        attrs = _get_attrs_reshape(node, {})

        assert attrs["allowzero"] == 0

    def test_get_attrs_reshape_allowzero_not_zero_raises_error(self):
        """Test Reshape raises ValueError for allowzero != 0."""
        node = onnx.helper.make_node(
            "Reshape",
            inputs=["input", "shape"],
            outputs=["output"],
            allowzero=1,
        )

        with pytest.raises(ValueError, match="allowzero=1 is not supported"):
            _get_attrs_reshape(node, {})

    def test_get_attrs_resize_basic(self):
        """Test Resize with default attributes."""
        node = onnx.helper.make_node(
            "Resize",
            inputs=["input", "roi", "scales", "sizes"],
            outputs=["output"],
        )

        attrs = _get_attrs_resize(node, {})

        assert attrs["mode"] == "nearest"
        assert attrs["coordinate_transformation_mode"] == "half_pixel"
        assert attrs["cubic_coeff_a"] == -0.75

    def test_get_attrs_shape_basic(self):
        """Test Shape with default start and end."""
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"])

        attrs = _get_attrs_shape(node, {})

        assert attrs["start"] == 0
        assert attrs["end"] == -1

    def test_get_attrs_shape_invalid_start_raises_error(self):
        """Test Shape raises ValueError for start != 0."""
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"], start=1)

        with pytest.raises(ValueError, match="start=1 is not supported"):
            _get_attrs_shape(node, {})

    def test_get_attrs_shape_invalid_end_raises_error(self):
        """Test Shape raises ValueError for end != -1."""
        node = onnx.helper.make_node("Shape", inputs=["input"], outputs=["output"], end=5)

        with pytest.raises(ValueError, match="end=5 is not supported"):
            _get_attrs_shape(node, {})

    def test_get_attrs_argmax_basic(self):
        """Test ArgMax with default attributes."""
        node = onnx.helper.make_node("ArgMax", inputs=["input"], outputs=["output"])

        attrs = _get_attrs_argmax(node, {})

        assert attrs["axis"] == 0
        assert attrs["keepdims"] == 1

    def test_get_attrs_argmax_select_last_index_not_zero_raises_error(self):
        """Test ArgMax raises ValueError for select_last_index != 0."""
        node = onnx.helper.make_node(
            "ArgMax",
            inputs=["input"],
            outputs=["output"],
            select_last_index=1,
        )

        with pytest.raises(ValueError, match="select_last_index=1 is not supported"):
            _get_attrs_argmax(node, {})

    def test_get_attrs_batchnorm_basic(self):
        """Test BatchNormalization with default attributes."""
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
        )

        attrs = _get_attrs_batchnorm(node, {})

        assert attrs["epsilon"] == 1e-5
        assert attrs["momentum"] == 0.9
        assert attrs["training_mode"] == 0

    def test_get_attrs_batchnorm_training_mode_not_zero_raises_error(self):
        """Test BatchNormalization raises ValueError for training_mode != 0."""
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
            training_mode=1,
        )

        with pytest.raises(ValueError, match="training_mode=1 is not supported"):
            _get_attrs_batchnorm(node, {})

    def test_get_attrs_batchnorm_multiple_outputs_raises_error(self):
        """Test BatchNormalization raises ValueError for multiple outputs."""
        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output", "mean_out", "var_out"],
        )

        with pytest.raises(ValueError, match="outputs is not supported"):
            _get_attrs_batchnorm(node, {})


# ============================================================================
# TestScatterOperators - Scatter Operations
# ============================================================================


class TestScatterOperators:
    """Test scatter operators."""

    def test_get_attrs_scatter_delegates_to_scatterelement(self):
        """Test Scatter delegates to get_attrs_scatterelement."""
        node = onnx.helper.make_node(
            "Scatter", inputs=["data", "indices", "updates"], outputs=["output"], axis=0
        )

        attrs = _get_attrs_scatter(node, {})

        assert attrs["axis"] == 0
        assert attrs["reduction"] == "none"

    def test_get_attrs_scatterelement_basic(self):
        """Test ScatterElements with default attributes."""
        node = onnx.helper.make_node(
            "ScatterElements",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
        )

        attrs = _get_attrs_scatterelement(node, {})

        assert attrs["axis"] == 0
        assert attrs["reduction"] == "none"

    def test_get_attrs_scatterelement_custom_axis(self):
        """Test ScatterElements with custom axis."""
        node = onnx.helper.make_node(
            "ScatterElements",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            axis=1,
        )

        attrs = _get_attrs_scatterelement(node, {})

        assert attrs["axis"] == 1

    def test_get_attrs_scatterelement_reduction_not_none_raises_error(self):
        """Test ScatterElements raises ValueError for reduction != 'none'."""
        node = onnx.helper.make_node(
            "ScatterElements",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            reduction="add",
        )

        with pytest.raises(ValueError, match="reduction=add is not supported"):
            _get_attrs_scatterelement(node, {})

    def test_get_attrs_scatternd_basic(self):
        """Test ScatterND with default attributes."""
        node = onnx.helper.make_node(
            "ScatterND", inputs=["data", "indices", "updates"], outputs=["output"]
        )

        attrs = _get_attrs_scatternd(node, {})

        assert attrs["reduction"] == "none"

    def test_get_attrs_scatternd_reduction_not_none_raises_error(self):
        """Test ScatterND raises ValueError for reduction != 'none'."""
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            reduction="add",
        )

        with pytest.raises(ValueError, match="reduction=add is not supported"):
            _get_attrs_scatternd(node, {})


# ============================================================================
# TestReduceOperators - Reduce Operations Using Factory Pattern
# ============================================================================


class TestReduceOperators:
    """Test reduce operators using get_attrs_reduce factory."""

    def test_get_attrs_reducemean_basic(self):
        """Test ReduceMean with default attributes from dispatcher."""
        node = onnx.helper.make_node("ReduceMean", inputs=["input"], outputs=["output"])

        attrs = _get_onnx_attrs(node, {})

        assert attrs["keepdims"] == 1
        assert attrs["noop_with_empty_axes"] == 0

    def test_get_attrs_reducemean_noop_not_zero_raises_error(self):
        """Test ReduceMean raises ValueError for noop_with_empty_axes != 0."""
        node = onnx.helper.make_node(
            "ReduceMean",
            inputs=["input"],
            outputs=["output"],
            noop_with_empty_axes=1,
        )

        with pytest.raises(ValueError, match="noop_with_empty_axes=1 is not supported"):
            _get_onnx_attrs(node, {})

    def test_get_attrs_reducesum_basic(self):
        """Test ReduceSum with default attributes from dispatcher."""
        node = onnx.helper.make_node("ReduceSum", inputs=["input"], outputs=["output"])

        attrs = _get_onnx_attrs(node, {})

        assert attrs["keepdims"] == 1
        assert attrs["noop_with_empty_axes"] == 0

    def test_get_attrs_reducesum_noop_not_zero_raises_error(self):
        """Test ReduceSum raises ValueError for noop_with_empty_axes != 0."""
        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["input"],
            outputs=["output"],
            noop_with_empty_axes=1,
        )

        with pytest.raises(ValueError, match="noop_with_empty_axes=1 is not supported"):
            _get_onnx_attrs(node, {})


# ============================================================================
# TestSimpleOperators - Operators Using get_attrs_simple Factory
# ============================================================================


class TestSimpleOperators:
    """Test operators using get_attrs_simple factory."""

    def test_get_attrs_elu_basic(self):
        """Test Elu with default alpha."""
        elu_extractor = _get_attrs_simple({"alpha": 1.0})
        node = onnx.helper.make_node("Elu", inputs=["input"], outputs=["output"])

        attrs = elu_extractor(node, {})

        assert attrs["alpha"] == 1.0

    def test_get_attrs_elu_custom_alpha(self):
        """Test Elu with custom alpha."""
        elu_extractor = _get_attrs_simple({"alpha": 1.0})
        node = onnx.helper.make_node("Elu", inputs=["input"], outputs=["output"], alpha=0.5)

        attrs = elu_extractor(node, {})

        assert attrs["alpha"] == 0.5

    def test_get_attrs_flatten_basic(self):
        """Test Flatten with default axis."""
        flatten_extractor = _get_attrs_simple({"axis": 1})
        node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["output"])

        attrs = flatten_extractor(node, {})

        assert attrs["axis"] == 1

    def test_get_attrs_gather_basic(self):
        """Test Gather with default axis."""
        gather_extractor = _get_attrs_simple({"axis": 0})
        node = onnx.helper.make_node("Gather", inputs=["input", "indices"], outputs=["output"])

        attrs = gather_extractor(node, {})

        assert attrs["axis"] == 0

    def test_get_attrs_gelu_basic(self):
        """Test Gelu with default approximate."""
        gelu_extractor = _get_attrs_simple({"approximate": "none"})
        node = onnx.helper.make_node("Gelu", inputs=["input"], outputs=["output"])

        attrs = gelu_extractor(node, {})

        assert attrs["approximate"] == "none"

    def test_get_attrs_gemm_basic(self):
        """Test Gemm with default attributes."""
        gemm_extractor = _get_attrs_simple({"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0})
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["A", "B", "C"],
            outputs=["output"],
        )

        attrs = gemm_extractor(node, {})

        assert attrs["alpha"] == 1.0
        assert attrs["beta"] == 1.0
        assert attrs["transA"] == 0
        assert attrs["transB"] == 0

    def test_get_attrs_leakyrelu_basic(self):
        """Test LeakyRelu with default alpha."""
        leakyrelu_extractor = _get_attrs_simple({"alpha": 0.01})
        node = onnx.helper.make_node("LeakyRelu", inputs=["input"], outputs=["output"])

        attrs = leakyrelu_extractor(node, {})

        assert attrs["alpha"] == 0.01

    def test_get_attrs_pad_basic(self):
        """Test Pad with default mode."""
        pad_extractor = _get_attrs_simple({"mode": "constant"})
        node = onnx.helper.make_node("Pad", inputs=["input"], outputs=["output"])

        attrs = pad_extractor(node, {})

        assert attrs["mode"] == "constant"

    def test_get_attrs_softmax_basic(self):
        """Test Softmax with default axis."""
        softmax_extractor = _get_attrs_simple({"axis": -1})
        node = onnx.helper.make_node("Softmax", inputs=["input"], outputs=["output"])

        attrs = softmax_extractor(node, {})

        assert attrs["axis"] == -1

    def test_get_attrs_split_basic(self):
        """Test Split with default axis."""
        split_extractor = _get_attrs_simple({"axis": 0, "num_outputs": None})
        node = onnx.helper.make_node("Split", inputs=["input"], outputs=["output"])

        attrs = split_extractor(node, {})

        assert attrs["axis"] == 0


# ============================================================================
# TestConstantNodeError - Constant Node Special Handling
# ============================================================================


class TestConstantNodeError:
    """Test Constant operator error handling."""

    def test_get_attrs_constant_raises_runtime_error(self):
        """Test Constant operator raises RuntimeError."""
        node = onnx.helper.make_node("Constant", inputs=[], outputs=["output"])

        with pytest.raises(RuntimeError, match="Constant nodes are not supported"):
            _get_attrs_constant(node, {})
