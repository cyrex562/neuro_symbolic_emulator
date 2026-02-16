"""Main training script for neural ALU functional units.

Usage:
    python training/train.py --op add --bits 8 --arch monolithic
    python training/train.py --op add --bits 8 --arch bitslice --slice-bits 4
    python training/train.py --op add --bits 8 --arch hierarchical
    python training/train.py --op all --bits 8 --arch all
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

from ops import ALL_OPS, DataGenerator
from architectures import ARCHITECTURES
from export.weights import export_weights


def create_model(arch_name: str, op, **kwargs):
    """Create a model for the given architecture and operation."""
    arch_cls = ARCHITECTURES[arch_name]
    return arch_cls(
        input_bits=op.input_bits,
        output_bits=op.output_bits,
        **kwargs,
    )


def train_model(model, op, config: dict) -> dict:
    """Train a model for a given ALU operation.

    Returns a dict of training metrics.
    """
    gen = DataGenerator(op)
    epochs = config.get("epochs", 200)
    batch_size = config.get("batch_size", 1024)
    lr = config.get("lr", 0.001)
    batches_per_epoch = config.get("batches_per_epoch", 100)
    val_samples = config.get("val_samples", 10000)
    patience = config.get("patience", 30)
    lr_schedule = config.get("lr_schedule", True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6
        )

    rng = np.random.default_rng(42)

    # Pre-generate validation set
    val_inputs, val_targets = gen.generate_batch(val_samples, rng=np.random.default_rng(999))

    best_accuracy = 0.0
    best_state = None
    no_improve = 0
    history = []

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(batches_per_epoch):
            inputs, targets = gen.generate_batch(batch_size, rng=rng)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches_per_epoch

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_inputs)
            val_loss = criterion(val_out, val_targets).item()

            # Bit-level accuracy: threshold at 0.5 and compare
            predicted = (val_out > 0.5).float()
            bit_correct = (predicted == val_targets).float()
            bit_accuracy = bit_correct.mean().item()

            # Operation-level accuracy: ALL bits must be correct for a given input
            op_correct = bit_correct.all(dim=1).float()
            op_accuracy = op_correct.mean().item()

        if lr_schedule:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "bit_accuracy": bit_accuracy,
            "op_accuracy": op_accuracy,
            "lr": current_lr,
        })

        if op_accuracy > best_accuracy:
            best_accuracy = op_accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d} | loss={avg_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | bit_acc={bit_accuracy:.6f} | "
                  f"op_acc={op_accuracy:.6f} | lr={current_lr:.2e} | "
                  f"{elapsed:.1f}s")

        # Early stopping
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

        # Perfect accuracy — stop
        if op_accuracy >= 1.0:
            print(f"  Perfect accuracy reached at epoch {epoch+1}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - t0
    return {
        "best_op_accuracy": best_accuracy,
        "best_bit_accuracy": max(h["bit_accuracy"] for h in history),
        "final_epoch": len(history),
        "total_time": total_time,
        "history": history,
        "params": model.param_count(),
    }


def run_training(op_name: str, bits: int, arch_name: str, config: dict,
                 output_dir: str) -> dict:
    """Train a single op+arch combination."""
    # Instantiate operation
    op_cls = ALL_OPS[op_name]
    op = op_cls(word_size=bits)

    print(f"\n{'='*70}")
    print(f"Training: {op.name} | arch={arch_name} | {op.description()}")
    print(f"  Input bits: {op.input_bits}, Output bits: {op.output_bits}")

    # Build model
    arch_kwargs = {}
    if arch_name == "bitslice":
        arch_kwargs["slice_bits"] = config.get("slice_bits", min(8, bits))
    if arch_name == "hierarchical":
        arch_kwargs["n_blocks"] = config.get("n_blocks", 4)

    hidden = config.get("hidden_sizes", None)
    if hidden:
        arch_kwargs["hidden_sizes"] = hidden

    model = create_model(arch_name, op, **arch_kwargs)
    print(f"  Architecture: {model.describe()}")
    print(f"  Parameters: {model.param_count():,}")

    metrics = train_model(model, op, config)
    print(f"  Result: op_accuracy={metrics['best_op_accuracy']:.6f}, "
          f"bit_accuracy={metrics['best_bit_accuracy']:.6f}, "
          f"time={metrics['total_time']:.1f}s")

    # Save model and weights
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{op.name}_{arch_name}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "op_name": op_name,
        "bits": bits,
        "arch_name": arch_name,
        "metrics": {k: v for k, v in metrics.items() if k != "history"},
        "config": config,
        "model_describe": model.describe(),
    }, model_path)

    # Export weights to portable JSON format
    json_path = os.path.join(output_dir, f"{op.name}_{arch_name}_weights.json")
    export_weights(model, arch_name, json_path)

    # Save training history
    hist_path = os.path.join(output_dir, f"{op.name}_{arch_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(metrics["history"], f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train neural ALU functional units")
    parser.add_argument("--op", type=str, default="add",
                        choices=list(ALL_OPS.keys()) + ["all"],
                        help="Operation to train (or 'all')")
    parser.add_argument("--bits", type=int, default=8,
                        help="Word size in bits")
    parser.add_argument("--arch", type=str, default="monolithic",
                        choices=list(ARCHITECTURES.keys()) + ["all"],
                        help="Architecture (or 'all')")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--hidden", type=int, nargs="+", default=None,
                        help="Hidden layer sizes (e.g., --hidden 128 128)")
    parser.add_argument("--slice-bits", type=int, default=None,
                        help="Slice size for bitslice architecture")
    parser.add_argument("--n-blocks", type=int, default=4,
                        help="Number of residual blocks for hierarchical architecture")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "batches_per_epoch": args.batches_per_epoch,
        "patience": args.patience,
        "hidden_sizes": args.hidden,
        "slice_bits": args.slice_bits or min(8, args.bits),
        "n_blocks": args.n_blocks,
    }

    ops = list(ALL_OPS.keys()) if args.op == "all" else [args.op]
    archs = list(ARCHITECTURES.keys()) if args.arch == "all" else [args.arch]

    results = {}
    for op_name in ops:
        for arch_name in archs:
            key = f"{op_name}_{args.bits}bit_{arch_name}"
            metrics = run_training(op_name, args.bits, arch_name, config,
                                   args.output_dir)
            results[key] = {
                "op_accuracy": metrics["best_op_accuracy"],
                "bit_accuracy": metrics["best_bit_accuracy"],
                "params": metrics["params"],
                "time": metrics["total_time"],
                "epochs": metrics["final_epoch"],
            }

    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<40} {'OpAcc':>8} {'BitAcc':>8} {'Params':>10} {'Time':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for key, r in sorted(results.items()):
        print(f"{key:<40} {r['op_accuracy']:>8.4f} {r['bit_accuracy']:>8.4f} "
              f"{r['params']:>10,} {r['time']:>7.1f}s")

    # Save summary
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
