"""Base classes for ALU operations and data generation."""

from abc import ABC, abstractmethod
import torch
import numpy as np


class AluOp(ABC):
    """Defines a single ALU operation: its semantics, I/O shape, and data generation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def input_bits(self) -> int:
        """Total input bits (e.g., 64 for two 32-bit operands)."""

    @property
    @abstractmethod
    def output_bits(self) -> int:
        """Total output bits."""

    @property
    def word_size(self) -> int:
        """Word size in bits. Derived from input_bits for binary ops."""
        return self.input_bits // 2

    @abstractmethod
    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the operation on integer arrays. a, b are arrays of unsigned ints.
        Returns array of unsigned int results."""

    @abstractmethod
    def description(self) -> str:
        """One-line description of this operation."""


class DataGenerator:
    """Generates training/validation data for an ALU operation."""

    def __init__(self, op: AluOp):
        self.op = op

    def int_to_bits(self, val: np.ndarray, n_bits: int) -> np.ndarray:
        """Convert integer array to bit representation. LSB first.
        val: (batch,) int array
        Returns: (batch, n_bits) float32 array of 0.0/1.0
        """
        bits = np.zeros((len(val), n_bits), dtype=np.float32)
        for i in range(n_bits):
            bits[:, i] = ((val >> i) & 1).astype(np.float32)
        return bits

    def bits_to_int(self, bits: np.ndarray, n_bits: int) -> np.ndarray:
        """Convert bit representation back to integers.
        bits: (batch, n_bits) float array (thresholded at 0.5)
        Returns: (batch,) int array
        """
        result = np.zeros(len(bits), dtype=np.int64)
        for i in range(n_bits):
            result += ((bits[:, i] > 0.5).astype(np.int64)) << i
        return result

    def generate_batch(self, batch_size: int, rng: np.random.Generator = None) -> tuple:
        """Generate a random batch of (inputs, targets) as torch tensors.
        Returns: (input_tensor, target_tensor) both float32
        """
        if rng is None:
            rng = np.random.default_rng()

        word = self.op.word_size
        max_val = (1 << word)

        a = rng.integers(0, max_val, size=batch_size, dtype=np.int64)
        b = rng.integers(0, max_val, size=batch_size, dtype=np.int64)

        result = self.op.compute(a, b)

        a_bits = self.int_to_bits(a, word)
        b_bits = self.int_to_bits(b, word)
        inputs = np.concatenate([a_bits, b_bits], axis=1)

        targets = self.int_to_bits(result, self.op.output_bits)

        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def generate_exhaustive(self) -> tuple:
        """Generate all possible input combinations. Only feasible for small word sizes.
        Returns: (input_tensor, target_tensor, a_ints, b_ints)
        """
        word = self.op.word_size
        max_val = 1 << word
        if max_val > 65536:
            raise ValueError(
                f"Exhaustive generation not feasible for {word}-bit "
                f"({max_val}^2 = {max_val**2} pairs)"
            )

        a_range = np.arange(max_val, dtype=np.int64)
        b_range = np.arange(max_val, dtype=np.int64)
        a_grid, b_grid = np.meshgrid(a_range, b_range)
        a = a_grid.flatten()
        b = b_grid.flatten()

        result = self.op.compute(a, b)

        a_bits = self.int_to_bits(a, word)
        b_bits = self.int_to_bits(b, word)
        inputs = np.concatenate([a_bits, b_bits], axis=1)
        targets = self.int_to_bits(result, self.op.output_bits)

        return torch.from_numpy(inputs), torch.from_numpy(targets), a, b

    def generate_edge_cases(self) -> tuple:
        """Generate structured edge cases: 0, 1, max, powers of 2, etc.
        Returns: (input_tensor, target_tensor, a_ints, b_ints)
        """
        word = self.op.word_size
        max_val = (1 << word) - 1

        # Important values to test
        special = [0, 1, 2, max_val, max_val - 1]
        # Powers of 2
        for i in range(word):
            special.append(1 << i)
        # Alternating bit patterns
        mask_a = 0
        mask_b = 0
        for i in range(word):
            if i % 2 == 0:
                mask_a |= (1 << i)
            else:
                mask_b |= (1 << i)
        special.extend([mask_a, mask_b])
        # Remove duplicates and clamp
        special = list(set(v & max_val for v in special))

        # All pairs of special values
        pairs_a = []
        pairs_b = []
        for va in special:
            for vb in special:
                pairs_a.append(va)
                pairs_b.append(vb)

        a = np.array(pairs_a, dtype=np.int64)
        b = np.array(pairs_b, dtype=np.int64)
        result = self.op.compute(a, b)

        a_bits = self.int_to_bits(a, word)
        b_bits = self.int_to_bits(b, word)
        inputs = np.concatenate([a_bits, b_bits], axis=1)
        targets = self.int_to_bits(result, self.op.output_bits)

        return torch.from_numpy(inputs), torch.from_numpy(targets), a, b
