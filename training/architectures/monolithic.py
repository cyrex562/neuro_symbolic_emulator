"""Monolithic MLP architecture: a single wide/deep network per ALU operation."""

import torch
import torch.nn as nn


class MonolithicMLP(nn.Module):
    """Standard multi-layer perceptron for ALU operations.

    Architecture: input -> [hidden layers with ReLU] -> output (sigmoid).
    Configurable depth and width.
    """

    def __init__(self, input_bits: int, output_bits: int,
                 hidden_sizes: list[int] = None, dropout: float = 0.0):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits

        if hidden_sizes is None:
            # Default: two hidden layers, 4x input width
            w = max(input_bits * 4, 64)
            hidden_sizes = [w, w]

        layers = []
        prev_size = input_bits
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_bits))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def arch_name() -> str:
        return "monolithic"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        sizes = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                sizes.append(str(m.out_features))
        return f"MonolithicMLP({self.input_bits} -> [{', '.join(sizes[:-1])}] -> {self.output_bits})"
