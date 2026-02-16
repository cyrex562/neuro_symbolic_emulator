"""Shift ALU operations: SLL, SRL, SRA."""

import numpy as np
from .base import AluOp


class SllOp(AluOp):
    """Shift left logical."""

    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"sll{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        return self._word_size

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = (1 << self._word_size) - 1
        # Only lower log2(word_size) bits of b matter for shift amount
        shift_mask = self._word_size - 1
        shift = (b & shift_mask).astype(np.int64)
        return ((a.astype(np.int64) << shift) & mask)

    def description(self) -> str:
        return f"{self._word_size}-bit shift left logical"


class SrlOp(AluOp):
    """Shift right logical."""

    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"srl{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        return self._word_size

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        shift_mask = self._word_size - 1
        shift = (b & shift_mask).astype(np.int64)
        return (a.astype(np.int64) >> shift)

    def description(self) -> str:
        return f"{self._word_size}-bit shift right logical"


class SraOp(AluOp):
    """Shift right arithmetic (sign-extending)."""

    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"sra{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        return self._word_size

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = (1 << self._word_size) - 1
        sign_bit = 1 << (self._word_size - 1)
        shift_mask = self._word_size - 1
        shift = (b & shift_mask).astype(np.int64)

        # Convert to signed, arithmetic shift, convert back to unsigned
        a_signed = np.where(a >= sign_bit, a.astype(np.int64) - (1 << self._word_size), a.astype(np.int64))
        shifted = (a_signed >> shift)
        return (shifted & mask)

    def description(self) -> str:
        return f"{self._word_size}-bit shift right arithmetic"
