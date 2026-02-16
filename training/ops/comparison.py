"""Comparison ALU operations: SLT (set less than), SLTU (unsigned)."""

import numpy as np
from .base import AluOp


class SltOp(AluOp):
    """Set less than (signed comparison)."""

    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"slt{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        return 1

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Interpret as signed (two's complement)
        sign_bit = 1 << (self._word_size - 1)
        a_signed = np.where(a >= sign_bit, a.astype(np.int64) - (1 << self._word_size), a.astype(np.int64))
        b_signed = np.where(b >= sign_bit, b.astype(np.int64) - (1 << self._word_size), b.astype(np.int64))
        return (a_signed < b_signed).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit signed set-less-than"


class SltuOp(AluOp):
    """Set less than unsigned."""

    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"sltu{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        return 1

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a < b).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit unsigned set-less-than"
