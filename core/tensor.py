"""
Tensor abstraction for the hardware-aware inference engine.

This is NOT a thin wrapper around NumPy.
This is a SYSTEMS object with rules and invariants.
"""

import numpy as np
from core.dtypes import DType


class Tensor:
    """
    Tensor represents data flowing through the inference engine.

    Design goals:
    - Explicit dtype
    - Explicit ownership of memory
    - Deterministic behavior
    - Future-proof for quantization
    """

    def __init__(
        self,
        data: np.ndarray,
        dtype: DType,
        scale: float = None,
        zero_point: int = None,
    ):
        """
        Initialize a Tensor.

        Parameters
        ----------
        data : np.ndarray
            Raw data buffer. Must match dtype.
        dtype : DType
            Logical data type of the tensor.
        scale : float, optional
            Quantization scale (used for INT8).
        zero_point : int, optional
            Quantization zero-point (used for INT8).
        """

        # ---- Basic validation ----
        if not isinstance(data, np.ndarray):
            raise TypeError("Tensor data must be a NumPy ndarray.")

        if not isinstance(dtype, DType):
            raise TypeError("dtype must be an instance of DType enum.")

        # ---- Enforce dtype consistency ----
        if dtype == DType.FP32 and data.dtype != np.float32:
            raise ValueError(
                "FP32 Tensor must use np.float32 data buffer."
            )

        if dtype == DType.INT8 and data.dtype != np.int8:
            raise ValueError(
                "INT8 Tensor must use np.int8 data buffer."
            )

        # ---- Quantization metadata rules ----
        if dtype == DType.INT8:
            if scale is None or zero_point is None:
                raise ValueError(
                    "INT8 Tensor requires scale and zero_point."
                )
        else:
            # FP32 tensors must NOT have quantization metadata
            if scale is not None or zero_point is not None:
                raise ValueError(
                    "FP32 Tensor cannot have scale or zero_point."
                )

        # ---- Assign fields ----
        self.data = data
        self.dtype = dtype
        self.shape = data.shape

        # Quantization metadata (None for FP32)
        self.scale = scale
        self.zero_point = zero_point

    def numpy(self) -> np.ndarray:
        """
        Return the underlying NumPy array.

        WARNING:
        This exposes raw memory.
        Use only for debugging or benchmarking.
        """
        return self.data

    def __repr__(self):
        """
        Human-readable representation.
        Useful for debugging and logging.
        """
        return (
            f"Tensor(shape={self.shape}, "
            f"dtype={self.dtype.name}, "
            f"scale={self.scale}, "
            f"zero_point={self.zero_point})"
        )
