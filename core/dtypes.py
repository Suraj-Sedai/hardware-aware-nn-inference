"""
Defines supported data types for the inference engine.

IMPORTANT DESIGN CHOICE:
- DType is explicit
- No implicit casting
- This file will grow when we add INT8
"""

from enum import Enum


class DType(Enum):
    """
    Enumeration of supported tensor data types.

    We use an Enum instead of strings to:
    - avoid typos
    - make comparisons explicit
    - keep the engine strict
    """

    FP32 = "float32"
    INT8 = "int8"   # Placeholder for future phases
