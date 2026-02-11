# kernels/linear_fp32.py

import numpy as np
from core.tensor import Tensor
from core.dtypes import DType


def linear_fp32(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """
    FP32 Linear kernel.

    Computes:
        output = input @ weight.T + bias

    All tensors must be FP32.
    """

    # ---- DType validation ----
    Tensor.assert_dtype(input, DType.FP32)
    Tensor.assert_dtype(weight, DType.FP32)
    Tensor.assert_dtype(bias, DType.FP32)

    # ---- Shape validation ----
    if input.ndim != 2:
        raise ValueError("Input must be 2D (batch, in_features).")

    if weight.ndim != 2:
        raise ValueError("Weight must be 2D (out_features, in_features).")

    if bias.ndim != 1:
        raise ValueError("Bias must be 1D (out_features,).")

    batch_size, in_features = input.shape
    out_features, weight_in = weight.shape

    if in_features != weight_in:
        raise ValueError(
            f"Incompatible shapes: {input.shape} and {weight.shape}"
        )

    if bias.shape[0] != out_features:
        raise ValueError(
            f"Bias shape {bias.shape} does not match output features {out_features}"
        )

    # ---- Core computation ----
    # NumPy uses optimized BLAS under the hood
    output_data = input.data @ weight.data.T
    output_data += bias.data  # Broadcasting over batch dimension

    # ---- Wrap result in Tensor ----
    return Tensor.from_fp32(output_data)
