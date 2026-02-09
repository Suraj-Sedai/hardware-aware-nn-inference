"""
Extended Tensor abstraction with validation, invariants,
and future-proof hooks for quantization-aware inference.
"""

import numpy as np
from core.dtypes import DType


class Tensor:
    """
    Tensor represents data flowing through the inference engine.

    Key guarantees:
    - Explicit dtype
    - Immutable precision semantics
    - Shape safety
    - Quantization metadata correctness
    """

    def __init__(
        self,
        data: np.ndarray,
        dtype: DType,
        scale: float = None,
        zero_point: int = None,
    ):
        # ---- Type validation ----
        if not isinstance(data, np.ndarray):
            raise TypeError("Tensor data must be a NumPy ndarray.")

        if not isinstance(dtype, DType):
            raise TypeError("dtype must be an instance of DType enum.")

        # ---- Enforce buffer dtype ----
        if dtype == DType.FP32 and data.dtype != np.float32:
            raise ValueError("FP32 Tensor requires np.float32 buffer.")

        if dtype == DType.INT8 and data.dtype != np.int8:
            raise ValueError("INT8 Tensor requires np.int8 buffer.")

        # ---- Quantization invariants ----
        if dtype == DType.INT8:
            if scale is None or zero_point is None:
                raise ValueError(
                    "INT8 Tensor must define scale and zero_point."
                )
        else:
            if scale is not None or zero_point is not None:
                raise ValueError(
                    "FP32 Tensor cannot have quantization metadata."
                )

        # ---- Assign immutable fields ----
        self._data = data
        self._dtype = dtype
        self._shape = data.shape
        self._scale = scale
        self._zero_point = zero_point

    # -----------------------------
    # Read-only properties
    # -----------------------------

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return self._data.size

    @property
    def scale(self):
        return self._scale

    @property
    def zero_point(self):
        return self._zero_point

    # -----------------------------
    # Factory constructors
    # -----------------------------

    @classmethod
    def from_fp32(cls, array: np.ndarray):
        """
        Create a FP32 Tensor safely.

        Forces:
        - float32 casting
        - correct dtype tagging
        """
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        return cls(data=array, dtype=DType.FP32)

    @classmethod
    def from_int8(cls, array: np.ndarray, scale: float, zero_point: int):
        """
        Create an INT8 Tensor safely.

        Forces:
        - int8 buffer
        - required quantization metadata
        """
        if array.dtype != np.int8:
            array = array.astype(np.int8)

        return cls(
            data=array,
            dtype=DType.INT8,
            scale=scale,
            zero_point=zero_point,
        )

    # -----------------------------
    # Validation helpers
    # -----------------------------

    @staticmethod
    def assert_same_shape(a: "Tensor", b: "Tensor"):
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch: {a.shape} vs {b.shape}"
            )

    @staticmethod
    def assert_dtype(tensor: "Tensor", expected: DType):
        if tensor.dtype != expected:
            raise ValueError(
                f"Expected dtype {expected}, got {tensor.dtype}"
            )

    def numpy(self) -> np.ndarray:
        """
        Explicit escape hatch to raw NumPy array.
        """
        return self._data

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, "
            f"dtype={self.dtype.name}, "
            f"scale={self.scale}, "
            f"zero_point={self.zero_point})"
        )
