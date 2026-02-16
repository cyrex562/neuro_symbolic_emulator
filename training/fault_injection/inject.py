"""Fault injection framework for testing neural FU resilience.

Simulates:
  - Weight zeroing (stuck-at-zero faults, dead neurons)
  - Weight noise (analog drift, thermal noise)
  - Neuron dropout (complete component failure)
  - Weight bit flips (single-event upsets / radiation)

Usage:
    python training/fault_injection/inject.py --model training/results/add8_monolithic.pt --sweep
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops import ALL_OPS, DataGenerator
from architectures import ARCHITECTURES
from validation.validate import load_model, validate_statistical


def inject_weight_zeros(model, fraction: float, rng: np.random.Generator = None):
    """Zero out a fraction of all weights (simulates stuck-at-zero faults)."""
    if rng is None:
        rng = np.random.default_rng()

    corrupted = copy.deepcopy(model)
    with torch.no_grad():
        for param in corrupted.parameters():
            mask = torch.from_numpy(
                rng.random(param.shape).astype(np.float32)
            ) > fraction
            param.mul_(mask.float())
    return corrupted


def inject_weight_noise(model, sigma: float, rng: np.random.Generator = None):
    """Add Gaussian noise to all weights (simulates analog drift/thermal noise)."""
    if rng is None:
        rng = np.random.default_rng()

    corrupted = copy.deepcopy(model)
    with torch.no_grad():
        for param in corrupted.parameters():
            noise = torch.from_numpy(
                rng.normal(0, sigma, param.shape).astype(np.float32)
            )
            param.add_(noise)
    return corrupted


def inject_neuron_dropout(model, fraction: float, rng: np.random.Generator = None):
    """Zero out entire neurons (rows of weight matrices) — simulates component failure."""
    if rng is None:
        rng = np.random.default_rng()

    corrupted = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in corrupted.named_parameters():
            if "weight" in name and param.dim() == 2:
                n_neurons = param.shape[0]
                n_kill = int(n_neurons * fraction)
                if n_kill == 0:
                    continue
                kill_indices = rng.choice(n_neurons, size=n_kill, replace=False)
                param[kill_indices, :] = 0.0
                # Also zero corresponding bias if it exists
                bias_name = name.replace("weight", "bias")
                for bname, bparam in corrupted.named_parameters():
                    if bname == bias_name:
                        bparam[kill_indices] = 0.0
    return corrupted


def inject_bit_flips(model, fraction: float, rng: np.random.Generator = None):
    """Flip random bits in the float32 representation of weights (simulates SEUs)."""
    if rng is None:
        rng = np.random.default_rng()

    corrupted = copy.deepcopy(model)
    with torch.no_grad():
        for param in corrupted.parameters():
            flat = param.flatten()
            n_total_bits = len(flat) * 32
            n_flips = int(n_total_bits * fraction)
            if n_flips == 0:
                continue

            for _ in range(n_flips):
                idx = rng.integers(0, len(flat))
                bit_pos = rng.integers(0, 32)
                # Convert float to int bits, flip, convert back
                val = flat[idx].item()
                import struct
                int_bits = struct.unpack(">I", struct.pack(">f", val))[0]
                int_bits ^= (1 << bit_pos)
                new_val = struct.unpack(">f", struct.pack(">I", int_bits))[0]
                flat[idx] = new_val
    return corrupted


FAULT_TYPES = {
    "weight_zeros": inject_weight_zeros,
    "weight_noise": inject_weight_noise,
    "neuron_dropout": inject_neuron_dropout,
    "bit_flips": inject_bit_flips,
}


def sweep_fault(model, op, fault_type: str, levels: list[float],
                n_samples: int = 100_000, n_trials: int = 3) -> list[dict]:
    """Sweep a fault type across severity levels, measuring accuracy at each."""
    inject_fn = FAULT_TYPES[fault_type]
    results = []

    for level in levels:
        trial_accuracies = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 1000 + int(level * 10000))
            corrupted = inject_fn(model, level, rng=rng)
            corrupted.eval()

            gen = DataGenerator(op)
            test_rng = np.random.default_rng(42)
            inputs, targets = gen.generate_batch(n_samples, rng=test_rng)

            with torch.no_grad():
                outputs = corrupted(inputs)

            predicted = (outputs > 0.5).float()
            bit_correct = (predicted == targets).float()
            op_correct = bit_correct.all(dim=1).float().mean().item()
            trial_accuracies.append(op_correct)

        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies)

        results.append({
            "fault_type": fault_type,
            "level": level,
            "mean_op_accuracy": float(mean_acc),
            "std_op_accuracy": float(std_acc),
            "min_op_accuracy": float(min(trial_accuracies)),
            "max_op_accuracy": float(max(trial_accuracies)),
            "n_trials": n_trials,
        })

        print(f"    {fault_type} level={level:.4f}: "
              f"acc={mean_acc:.6f} +/- {std_acc:.6f}")

    return results


def run_fault_sweep(model_path: str, n_samples: int = 100_000,
                    n_trials: int = 3) -> dict:
    """Run complete fault injection sweep on a model."""
    model, op, arch_name, checkpoint = load_model(model_path)
    bits = checkpoint["bits"]

    print(f"\nFault injection sweep: {op.name} | arch={arch_name} | {bits}-bit")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    all_results = {
        "model_path": model_path,
        "op": op.name,
        "arch": arch_name,
        "bits": bits,
        "sweeps": {},
    }

    # Define sweep levels for each fault type
    sweep_configs = {
        "weight_zeros": [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "weight_noise": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0],
        "neuron_dropout": [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "bit_flips": [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    }

    for fault_type, levels in sweep_configs.items():
        print(f"\n  Sweeping {fault_type}:")
        results = sweep_fault(model, op, fault_type, levels,
                              n_samples=n_samples, n_trials=n_trials)
        all_results["sweeps"][fault_type] = results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Fault injection testing for neural ALU models")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--sweep", action="store_true", help="Run full fault sweep")
    parser.add_argument("--fault-type", type=str, choices=list(FAULT_TYPES.keys()),
                        help="Specific fault type to test")
    parser.add_argument("--level", type=float, default=0.1,
                        help="Fault severity level (0.0 to 1.0)")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Validation samples per trial")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per level")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    if args.sweep:
        results = run_fault_sweep(args.model, n_samples=args.samples,
                                  n_trials=args.trials)
    elif args.fault_type:
        model, op, arch_name, checkpoint = load_model(args.model)
        print(f"\nSingle fault test: {args.fault_type} @ level={args.level}")
        results = sweep_fault(model, op, args.fault_type, [args.level],
                              n_samples=args.samples, n_trials=args.trials)
    else:
        parser.print_help()
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
