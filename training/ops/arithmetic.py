"""Arithmetic ALU operations: ADD, SUB."""

import numpy as np
from .base import AluOp


class AddOp(AluOp):
    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"add{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        # Result + carry bit
        return self._word_size + 1

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit unsigned addition with carry"


class SubOp(AluOp):
    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"sub{self._word_size}"

    @property
    def input_bits(self) -> int:
        return self._word_size * 2

    @property
    def output_bits(self) -> int:
        # Result + borrow bit
        return self._word_size + 1

    @property
    def word_size(self) -> int:
        return self._word_size

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # For subtraction, result is a - b. If a < b, borrow bit is set.
        # Store as (word_size)-bit result + 1 borrow bit.
        # We compute in wider int to capture borrow.
        mask = (1 << self._word_size) - 1
        raw = a.astype(np.int64) - b.astype(np.int64)
        result_bits = raw & mask
        borrow = (raw < 0).astype(np.int64)
        return result_bits | (borrow << self._word_size)

    def description(self) -> str:
        return f"{self._word_size}-bit unsigned subtraction with borrow"
