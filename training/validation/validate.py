"""Validation framework for neural ALU functional units.

Supports:
  - Exhaustive testing (8-bit: all 65,536 input pairs)
  - Statistical testing (32-bit: configurable sample count with confidence intervals)
  - Edge case testing (boundary values, powers of 2, etc.)
  - Per-bit error analysis (which output bits are most error-prone)

Usage:
    python training/validation/validate.py --model training/results/add8_monolithic.pt
    python training/validation/validate.py --model training/results/add8_monolithic.pt --exhaustive
    python training/validation/validate.py --dir training/results/
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops import ALL_OPS, DataGenerator
from architectures import ARCHITECTURES


def load_model(model_path: str):
    """Load a saved model and return (model, metadata)."""
    checkpoint = torch.load(model_path, weights_only=False)
    op_name = checkpoint["op_name"]
    bits = checkpoint["bits"]
    arch_name = checkpoint["arch_name"]

    op_cls = ALL_OPS[op_name]
    op = op_cls(word_size=bits)

    config = checkpoint.get("config", {})
    arch_kwargs = {}
    if arch_name == "bitslice":
        arch_kwargs["slice_bits"] = config.get("slice_bits", min(8, bits))
    if arch_name == "hierarchical":
        arch_kwargs["n_blocks"] = config.get("n_blocks", 4)
    hidden = config.get("hidden_sizes", None)
    if hidden:
        arch_kwargs["hidden_sizes"] = hidden

    arch_cls = ARCHITECTURES[arch_name]
    model = arch_cls(
        input_bits=op.input_bits,
        output_bits=op.output_bits,
        **arch_kwargs,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, op, arch_name, checkpoint


def validate_exhaustive(model, op) -> dict:
    """Run exhaustive validation (all input pairs). Only for <= 16-bit."""
    gen = DataGenerator(op)
    inputs, targets, a_ints, b_ints = gen.generate_exhaustive()

    t0 = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    inference_time = time.time() - t0

    predicted = (outputs > 0.5).float()
    bit_correct = (predicted == targets).float()

    # Per-sample: all bits correct?
    op_correct = bit_correct.all(dim=1)
    op_accuracy = op_correct.float().mean().item()
    total_samples = len(a_ints)
    n_errors = total_samples - int(op_correct.sum().item())

    # Per-bit accuracy
    per_bit_acc = bit_correct.mean(dim=0).tolist()

    # Collect error cases
    error_mask = ~op_correct
    error_indices = error_mask.nonzero(as_tuple=True)[0].tolist()
    error_examples = []
    for idx in error_indices[:20]:  # First 20 errors
        a_val = int(a_ints[idx])
        b_val = int(b_ints[idx])
        expected = targets[idx].tolist()
        got = predicted[idx].tolist()
        raw = outputs[idx].tolist()
        error_examples.append({
            "a": a_val, "b": b_val,
            "expected_bits": expected, "got_bits": got,
            "raw_output": [round(v, 4) for v in raw],
        })

    return {
        "type": "exhaustive",
        "total_samples": total_samples,
        "errors": n_errors,
        "op_accuracy": op_accuracy,
        "bit_accuracy": bit_correct.mean().item(),
        "per_bit_accuracy": per_bit_acc,
        "inference_time": inference_time,
        "error_examples": error_examples,
    }


def validate_statistical(model, op, n_samples: int = 1_000_000, seed: int = 12345) -> dict:
    """Run statistical validation with confidence intervals."""
    gen = DataGenerator(op)
    rng = np.random.default_rng(seed)

    batch_size = min(n_samples, 50000)
    n_batches = (n_samples + batch_size - 1) // batch_size

    total_op_correct = 0
    total_bit_correct = 0
    total_bits = 0
    total_samples = 0
    per_bit_correct = np.zeros(op.output_bits, dtype=np.int64)

    t0 = time.time()
    model.eval()

    for batch_idx in range(n_batches):
        actual_batch = min(batch_size, n_samples - total_samples)
        inputs, targets = gen.generate_batch(actual_batch, rng=rng)

        with torch.no_grad():
            outputs = model(inputs)

        predicted = (outputs > 0.5).float()
        bit_correct = (predicted == targets).float()

        op_correct = bit_correct.all(dim=1).sum().item()
        total_op_correct += op_correct
        total_bit_correct += bit_correct.sum().item()
        total_bits += bit_correct.numel()
        total_samples += actual_batch

        per_bit_correct += bit_correct.sum(dim=0).numpy().astype(np.int64)

    inference_time = time.time() - t0

    op_accuracy = total_op_correct / total_samples
    bit_accuracy = total_bit_correct / total_bits

    # Wilson score confidence interval for op_accuracy
    z = 1.96  # 95% CI
    n = total_samples
    p_hat = op_accuracy
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)

    per_bit_acc = (per_bit_correct / total_samples).tolist()

    return {
        "type": "statistical",
        "total_samples": total_samples,
        "errors": total_samples - int(total_op_correct),
        "op_accuracy": op_accuracy,
        "bit_accuracy": bit_accuracy,
        "per_bit_accuracy": per_bit_acc,
        "confidence_interval_95": [ci_lower, ci_upper],
        "inference_time": inference_time,
    }


def validate_edge_cases(model, op) -> dict:
    """Run validation on structured edge cases."""
    gen = DataGenerator(op)
    inputs, targets, a_ints, b_ints = gen.generate_edge_cases()

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    predicted = (outputs > 0.5).float()
    bit_correct = (predicted == targets).float()
    op_correct = bit_correct.all(dim=1)
    op_accuracy = op_correct.float().mean().item()

    error_mask = ~op_correct
    error_indices = error_mask.nonzero(as_tuple=True)[0].tolist()
    error_examples = []
    for idx in error_indices[:20]:
        a_val = int(a_ints[idx])
        b_val = int(b_ints[idx])
        expected = targets[idx].tolist()
        got = predicted[idx].tolist()
        raw = outputs[idx].tolist()
        error_examples.append({
            "a": a_val, "b": b_val,
            "expected_bits": expected, "got_bits": got,
            "raw_output": [round(v, 4) for v in raw],
        })

    return {
        "type": "edge_cases",
        "total_samples": len(a_ints),
        "errors": len(error_indices),
        "op_accuracy": op_accuracy,
        "bit_accuracy": bit_correct.mean().item(),
        "error_examples": error_examples,
    }


def validate_model(model_path: str, exhaustive: bool = False,
                    n_samples: int = 1_000_000) -> dict:
    """Run full validation suite on a saved model."""
    model, op, arch_name, checkpoint = load_model(model_path)
    bits = checkpoint["bits"]

    print(f"\nValidating: {op.name} | arch={arch_name} | {bits}-bit")
    print(f"  Model: {checkpoint.get('model_describe', 'unknown')}")

    results = {
        "model_path": model_path,
        "op": op.name,
        "arch": arch_name,
        "bits": bits,
    }

    # Edge cases always
    print("  Running edge case validation...")
    edge_result = validate_edge_cases(model, op)
    results["edge_cases"] = edge_result
    print(f"    Edge cases: {edge_result['op_accuracy']:.6f} "
          f"({edge_result['errors']}/{edge_result['total_samples']} errors)")

    # Exhaustive if requested and feasible
    if exhaustive and bits <= 16:
        print(f"  Running exhaustive validation ({(1<<bits)**2:,} pairs)...")
        exh_result = validate_exhaustive(model, op)
        results["exhaustive"] = exh_result
        print(f"    Exhaustive: {exh_result['op_accuracy']:.6f} "
              f"({exh_result['errors']}/{exh_result['total_samples']} errors)")
    elif exhaustive:
        print(f"  Skipping exhaustive (infeasible for {bits}-bit)")

    # Statistical always
    print(f"  Running statistical validation ({n_samples:,} samples)...")
    stat_result = validate_statistical(model, op, n_samples=n_samples)
    results["statistical"] = stat_result
    ci = stat_result["confidence_interval_95"]
    print(f"    Statistical: {stat_result['op_accuracy']:.6f} "
          f"({stat_result['errors']}/{stat_result['total_samples']} errors) "
          f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # Per-bit accuracy
    per_bit = stat_result["per_bit_accuracy"]
    worst_bit = min(range(len(per_bit)), key=lambda i: per_bit[i])
    print(f"    Worst bit: bit[{worst_bit}] = {per_bit[worst_bit]:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate neural ALU models")
    parser.add_argument("--model", type=str, help="Path to .pt model file")
    parser.add_argument("--dir", type=str, help="Directory of .pt files to validate")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Run exhaustive validation (8-bit only)")
    parser.add_argument("--samples", type=int, default=1_000_000,
                        help="Number of samples for statistical validation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    all_results = {}

    if args.model:
        r = validate_model(args.model, exhaustive=args.exhaustive,
                           n_samples=args.samples)
        key = f"{r['op']}_{r['arch']}"
        all_results[key] = r

    elif args.dir:
        for fname in sorted(os.listdir(args.dir)):
            if fname.endswith(".pt"):
                path = os.path.join(args.dir, fname)
                r = validate_model(path, exhaustive=args.exhaustive,
                                   n_samples=args.samples)
                key = f"{r['op']}_{r['arch']}"
                all_results[key] = r
    else:
        parser.print_help()
        return

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'OpAcc':>10} {'BitAcc':>10} {'Errors':>10} {'Samples':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for key, r in sorted(all_results.items()):
        stat = r.get("statistical", {})
        print(f"{key:<35} {stat.get('op_accuracy', 0):>10.6f} "
              f"{stat.get('bit_accuracy', 0):>10.6f} "
              f"{stat.get('errors', '?'):>10} {stat.get('total_samples', '?'):>10}")

    if args.output:
        # Convert for JSON serialization
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
