"""Bit-slice architecture: chain of smaller NNs with carry propagation.

For a 32-bit adder, this chains four 8-bit adder slices, each receiving
the carry output of the previous slice as an extra input.
"""

import torch
import torch.nn as nn


class BitSlice(nn.Module):
    """A single bit-slice: processes `slice_bits` from each operand plus carry_in."""

    def __init__(self, slice_bits: int, carry_bits: int = 1,
                 hidden_sizes: list[int] = None):
        super().__init__()
        self.slice_bits = slice_bits
        self.carry_bits = carry_bits
        # Inputs: slice_bits (a) + slice_bits (b) + carry_bits (carry_in)
        in_size = slice_bits * 2 + carry_bits
        # Outputs: slice_bits (result) + carry_bits (carry_out)
        out_size = slice_bits + carry_bits

        if hidden_sizes is None:
            w = max(in_size * 4, 32)
            hidden_sizes = [w, w]

        layers = []
        prev = in_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_size))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, a_slice: torch.Tensor, b_slice: torch.Tensor,
                carry_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([a_slice, b_slice, carry_in], dim=1)
        out = self.net(x)
        result = out[:, :self.slice_bits]
        carry_out = out[:, self.slice_bits:]
        return result, carry_out


class BitSliceNetwork(nn.Module):
    """Chains multiple BitSlice modules for wide operations.

    For operations without carry semantics (AND, OR, XOR), each slice
    operates independently (carry is always zero).
    """

    def __init__(self, input_bits: int, output_bits: int,
                 hidden_sizes: list[int] = None, slice_bits: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.word_size = input_bits // 2
        self.slice_bits = slice_bits
        self.n_slices = (self.word_size + slice_bits - 1) // slice_bits

        # Carry bits: 1 for operations with carry, 0 for pure bitwise
        # We always include carry_bits=1; for bitwise ops it learns to ignore it
        self.carry_bits = 1

        self.slices = nn.ModuleList([
            BitSlice(slice_bits, self.carry_bits, hidden_sizes)
            for _ in range(self.n_slices)
        ])

        # If output_bits != word_size (e.g., adder has carry out), we handle
        # the final carry as part of the last slice's carry_out
        self._has_extra_carry = output_bits > self.word_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        word = self.word_size
        sb = self.slice_bits

        # Split input into a and b
        a_bits = x[:, :word]
        b_bits = x[:, word:word*2]

        carry = torch.zeros(batch, self.carry_bits, device=x.device)
        result_slices = []

        for i, slice_mod in enumerate(self.slices):
            start = i * sb
            end = min(start + sb, word)
            actual_bits = end - start

            a_slice = a_bits[:, start:end]
            b_slice = b_bits[:, start:end]

            # Pad if last slice is smaller
            if actual_bits < sb:
                pad = sb - actual_bits
                a_slice = torch.cat([a_slice, torch.zeros(batch, pad, device=x.device)], dim=1)
                b_slice = torch.cat([b_slice, torch.zeros(batch, pad, device=x.device)], dim=1)

            result, carry = slice_mod(a_slice, b_slice, carry)

            # Only take actual_bits from result (trim padding)
            result_slices.append(result[:, :actual_bits])

        out = torch.cat(result_slices, dim=1)

        if self._has_extra_carry:
            out = torch.cat([out, carry], dim=1)

        # Trim to output_bits
        return out[:, :self.output_bits]

    @staticmethod
    def arch_name() -> str:
        return "bitslice"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (f"BitSliceNetwork({self.word_size}-bit, "
                f"{self.n_slices}x{self.slice_bits}-bit slices, "
                f"params={self.param_count()})")
