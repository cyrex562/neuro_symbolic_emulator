"""Bitwise ALU operations: AND, OR, XOR."""

import numpy as np
from .base import AluOp


class AndOp(AluOp):
    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"and{self._word_size}"

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
        return (a & b).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit bitwise AND"


class OrOp(AluOp):
    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"or{self._word_size}"

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
        return (a | b).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit bitwise OR"


class XorOp(AluOp):
    def __init__(self, word_size: int = 8):
        self._word_size = word_size

    @property
    def name(self) -> str:
        return f"xor{self._word_size}"

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
        return (a ^ b).astype(np.int64)

    def description(self) -> str:
        return f"{self._word_size}-bit bitwise XOR"
