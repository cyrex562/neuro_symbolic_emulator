"""Export trained PyTorch models to portable weight formats.

Supports JSON (compatible with the Rust ndarray loader) and raw binary.
"""

import json
import torch
import torch.nn as nn


def export_weights(model: nn.Module, arch_name: str, path: str):
    """Export model weights to JSON format.

    Format is a list of layers, each with:
      - name: layer name
      - weight: {dim: [...], data: [...]}
      - bias: {dim: [...], data: [...]}
    """
    layers = []
    state = model.state_dict()

    for name, param in state.items():
        t = param.detach().cpu()
        entry = {
            "name": name,
            "shape": list(t.shape),
            "data": t.flatten().tolist(),
        }
        layers.append(entry)

    doc = {
        "arch": arch_name,
        "params": sum(p.numel() for p in model.parameters()),
        "layers": layers,
    }

    with open(path, "w") as f:
        json.dump(doc, f)
