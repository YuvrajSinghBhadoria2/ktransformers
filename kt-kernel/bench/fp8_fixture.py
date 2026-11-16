"""CPU-only FP8 weight fixture generation for the AVX2 benchmark."""

import numpy as np
import torch


def quantize_fp8_blockwise(weight, group_size=128):
    """Encode finite FP32 weights as E4M3FN bytes with per-block FP32 scales."""
    n, k = weight.shape
    n_blocks_n = (n + group_size - 1) // group_size
    n_blocks_k = (k + group_size - 1) // group_size
    quantized = np.zeros((n, k), dtype=np.uint8)
    scales = np.zeros((n_blocks_n, n_blocks_k), dtype=np.float32)

    for i in range(n_blocks_n):
        for j in range(n_blocks_k):
            row_start, col_start = i * group_size, j * group_size
            row_end = min(row_start + group_size, n)
            col_end = min(col_start + group_size, k)
            block = weight[row_start:row_end, col_start:col_end]
            max_val = np.max(np.abs(block))
            scale = max_val / 448.0 if max_val > 0 else 1.0
            scales[i, j] = scale

            # uint8 is the carrier for FP8 bits, not an integer quantization.
            # Float8 conversion rounds to nearest, ties to even, including
            # subnormals and exponent boundaries. Keep finite values saturated.
            normalized = torch.from_numpy(np.ascontiguousarray(block / scale))
            encoded = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            quantized[row_start:row_end, col_start:col_end] = encoded.view(torch.uint8).numpy()

    return quantized, scales
