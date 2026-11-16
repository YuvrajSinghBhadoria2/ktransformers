"""Check benchmark FP8 inputs without importing the native KT extension."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from fp8_fixture import quantize_fp8_blockwise

register_cpu_ci(est_time=5, suite="default")


def _decode_e4m3fn(bits):
    """Independent scalar format reference; no torch float8 operations."""
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    if exponent == 0:
        value = math.ldexp(mantissa, -9)
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -value if bits & 0x80 else value


def _encode_e4m3fn_reference(value):
    """Choose the nearest finite code; an even low bit breaks halfway ties."""
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), 448.0)
    code = min(range(127), key=lambda bits: (abs(_decode_e4m3fn(bits) - magnitude), bits & 1))
    return sign | code


def test_original_integer_cast_regression():
    weight = np.zeros((128, 128), dtype=np.float32)
    weight[0, :4] = [448.0, 1.0, -1.0, -448.0]
    encoded, scales = quantize_fp8_blockwise(weight)

    np.testing.assert_array_equal(scales, [[1.0]])
    np.testing.assert_array_equal(encoded[0, :4], [0x7E, 0x38, 0xB8, 0xFE])
    # The previous numeric uint8 cast emitted 0x01 here: E4M3FN decodes it to 2^-9.
    assert _decode_e4m3fn(int(encoded[0, 1])) == 1.0


def test_every_finite_encoding_including_signed_zero_round_trips():
    codes = [bits for bits in range(256) if bits & 0x7F != 0x7F]
    weight = np.array([_decode_e4m3fn(bits) for bits in codes], dtype=np.float32).reshape(2, 127)
    encoded, scales = quantize_fp8_blockwise(weight)

    np.testing.assert_array_equal(scales, [[1.0]])
    np.testing.assert_array_equal(encoded, np.array(codes, dtype=np.uint8).reshape(weight.shape))


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize(
    "midpoint",
    [2.0**-10, 3.0 * 2.0**-10, 15.0 * 2.0**-10, 1.0625, 1.1875, 1.9375, 248.0, 432.0],
)
def test_rounding_at_and_around_subnormal_and_exponent_boundaries(midpoint, sign):
    midpoint = np.float32(midpoint)
    values = sign * np.array(
        [np.nextafter(midpoint, np.float32(-np.inf)), midpoint, np.nextafter(midpoint, np.float32(np.inf))],
        dtype=np.float32,
    )
    # A 448 anchor makes scale exactly one, isolating float8 encoding/rounding.
    weight = np.concatenate([np.array([448.0], dtype=np.float32), values]).reshape(1, -1)
    encoded, scales = quantize_fp8_blockwise(weight)

    np.testing.assert_array_equal(scales, [[1.0]])
    expected = [_encode_e4m3fn_reference(float(value)) for value in values]
    np.testing.assert_array_equal(encoded[0, 1:], expected)


def test_all_zero_block_preserves_zero_signs_and_uses_unit_scale():
    weight = np.array([[0.0, -0.0], [-0.0, 0.0]], dtype=np.float32)
    encoded, scales = quantize_fp8_blockwise(weight)

    np.testing.assert_array_equal(scales, [[1.0]])
    np.testing.assert_array_equal(encoded, [[0x00, 0x80], [0x80, 0x00]])


def test_block_scales_partial_blocks_and_noncontiguous_input():
    storage = np.zeros((6, 10), dtype=np.float32)
    weight = storage[::2, ::2]
    weight[:] = [[448.0, 1.0, 896.0, 2.0, 0.0], [-448.0, -1.0, -896.0, -2.0, -0.0], [224.0, 0.5, 1792.0, 4.0, 0.0]]
    original = weight.copy()
    encoded, scales = quantize_fp8_blockwise(weight, group_size=2)
    expected_scales = np.array([[1.0, 2.0, 1.0], [0.5, 4.0, 1.0]], dtype=np.float32)
    expected = np.array(
        [
            [_encode_e4m3fn_reference(float(weight[i, j] / expected_scales[i // 2, j // 2])) for j in range(5)]
            for i in range(3)
        ],
        dtype=np.uint8,
    )

    assert not weight.flags.c_contiguous
    assert encoded.dtype == np.uint8 and scales.dtype == np.float32
    assert encoded.flags.c_contiguous and scales.flags.c_contiguous
    np.testing.assert_array_equal(scales, expected_scales)
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_array_equal(weight, original)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
