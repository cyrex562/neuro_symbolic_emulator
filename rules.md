# Neuro-Symbolic Emulator - Phase 2 Rules (NTE)

## 1. Architectural Constraints

### 1.1 Block-Level Granularity
*   **The Golden Rule**: Logic must be performed by **Neural Functional Units (FUs)** via matrix multiplication.
*   **No Discrete Gates**: Do not build adders from NAND gates anymore. Train an "Adder Network".
*   **Voter Logic**: Even consensus logic should ideally be neural (weighted average or trained voter), though simple statistical averaging is acceptable for "Recalibration" logic.

### 1.2 Vectorized Data
*   **Internal Representation**: Data must remain as floating-point vectors (e.g., `[0.0, 0.9, 0.1, ...]`) within the system.
*   **Symbolic Boundary**: Only convert to strict Integers at the very Input (User -> CPU) and very Output (CPU -> User).
*   **Drift Tracking**: You must log precision loss. If a value drifts (e.g., `0.8` instead of `1.0`), it is a feature, not a bug, until it breaks consensus.

### 1.3 TTA Compliance
*   **Instruction Set**: The ONLY architectural instruction is `MOVE(SRC, DEST)`.
*   **Execution**: Computation occurs *only* as a side-effect of moving data to a "Trigger" port of an FU.

## 2. Implementation Standards

### 2.1 Dependencies
*   Continue using `ndarray` for vector/matrix ops.
*   `serde` for saving FU weights (`fu_weights.json`).

### 2.2 Resiliency
*   The system must support injecting noise into weights (`fu.perturb(amount)`) to demonstrate robustness.
