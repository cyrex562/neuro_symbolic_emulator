"""Hierarchical architecture with residual connections for long-range dependencies.

Uses skip connections to help the network learn patterns like carry propagation
across all 32 bits. Inspired by ResNet — intermediate layers can "shortcut"
information from input to output.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A residual block: x + f(x) where f is a 2-layer MLP."""

    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class HierarchicalNetwork(nn.Module):
    """Network with residual blocks for learning long-range bit dependencies.

    Architecture:
      input -> project to hidden_width -> [N residual blocks] -> project to output -> sigmoid
    """

    def __init__(self, input_bits: int, output_bits: int,
                 hidden_sizes: list[int] = None, n_blocks: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits

        if hidden_sizes is not None:
            hidden_width = hidden_sizes[0]
        else:
            hidden_width = max(input_bits * 4, 128)

        self.input_proj = nn.Sequential(
            nn.Linear(input_bits, hidden_width),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_width) for _ in range(n_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_width, output_bits),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h)

    @staticmethod
    def arch_name() -> str:
        return "hierarchical"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        n_blocks = len(self.blocks)
        if n_blocks > 0:
            width = self.input_proj[0].out_features
        else:
            width = 0
        return (f"HierarchicalNetwork({self.input_bits} -> "
                f"{n_blocks} ResBlocks@{width} -> {self.output_bits}, "
                f"params={self.param_count()})")
