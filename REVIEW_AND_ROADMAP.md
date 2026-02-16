# Neuro-Symbolic Logic Emulator: Codebase Review & Roadmap

**Date:** 2026-02-16
**Status:** Architectural review complete. Ready for Phase 1 clean-room implementation.

---

## 1. Project Vision

The NSLE is a research project exploring whether **neural networks can completely replace the traditional digital logic substrate of a CPU**. Not just the ALU — the entire computational pipeline: fetch, decode, execute, memory access, control flow, and eventually the software layer above it.

The end deliverable is a **QEMU-bootable virtual machine** running a neural CPU that boots into a minimal firmware environment. Performance and reliability measurements from this system will characterize the "art of the possible" for neural computation as a hardware replacement.

### Long-term aspirations (beyond initial scope)
- Compile conventional software into neural network representations
- Run the neural CPU on real hardware (GPU, neural accelerator, FPGA)
- Explore Transport Triggered Architecture as an alternative machine model

---

## 2. Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **ISA** | RISC-V rv32i | Inherits mature toolchain (GCC, LLVM, binutils), existing firmware (OpenSBI), path to QEMU integration. Option to explore custom ISA/TTA later. |
| **Word size** | 32-bit | Minimum for RISC-V rv32i. 64-bit aspirational but deferred — doubles NN input size. |
| **NN abstraction level** | ALU-operation primary, composable | Each ALU op (add, sub, and, or, xor, slt, shifts) is one NN. Some higher-level ops (fetch, decode) may also be single NNs. System should support composition at multiple levels. |
| **Neural scope** | Everything eventually neural | Not just ALU — control, routing, decode, register management. Conventional logic only as scaffolding during development. |
| **Redundancy strategy** | Test both approaches | Internal redundancy (wide/deep NNs) vs. external TMR (multiple NNs + voter). Fault injection campaigns will determine the best tradeoff. |
| **Accuracy methodology** | Train to perfection on large samples + TMR safety net | Statistical validation with progressive refinement of confidence intervals. TMR catches residual errors. |
| **Toolchain** | RISC-V primary, WASM VM as parallel exploration | RISC-V gives immediate toolchain. WASM provides a simpler compilation target for experimentation. |
| **Languages** | Rust (emulator, QEMU integration) + Python (NN training) | Python/PyTorch for rapid training iteration. Rust for the runtime that loads and executes trained NNs. |
| **Codebase approach** | Clean-room rewrite | Current code preserved as reference. New architecture built from scratch with 32-bit rv32i as the target. |

---

## 3. Review of Existing Codebase (~1,900 lines Rust)

### 3.1 What exists

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `fu.rs` | 353 | **Partially reusable** | BaseFU MLP structure, activation functions, and training loop are solid foundations. Stateful FUs (PC, LoadStore, Stack, UART) use conventional logic — need neural reimplementation. |
| `bus.rs` | 213 | **Needs rewrite** | Built around TTA MoveOps and 8-bit assumptions. Memory map concept is good but implementation is incomplete (FU read returns zeros, no output routing). |
| `system.rs` | 72 | **Needs rewrite** | Thin orchestrator. PC is a plain usize, not wired to ProgramCounterFU. No instruction fetch/decode. |
| `register.rs` | 99 | **Partially reusable** | Neural register concept good. Needs widening to 32-bit. Cleanup/quantization logic is useful. |
| `gui.rs` | 276 | **Defer** | egui dashboard. Useful later but not needed for Phase 1. |
| `loader.rs` | 176 | **Needs rewrite** | JSON manifest loader. Good pattern but format will change for rv32i system. |
| `voter.rs` | 35 | **Reusable concept** | Vote function works but is never integrated. Needs to become part of the FU pipeline. |
| `manage_fus.rs` | 356 | **Replace** | CLI training tool. Will be superseded by Python training pipeline. |
| `train_fu.rs` | 112 | **Replace** | Same — Python training pipeline replaces this. |
| `legacy/` | 180 | **Archive** | Gate-level and circuit-level code. Historical reference only. |

### 3.2 Structural issues in current code

1. **FU output is a dead end.** When a MOVE writes to an FU, `forward()` runs and the result is cached in `fu_io_cache`, but it's never written back to any register or memory. The bus `read_mem` for FU addresses returns `Array1::zeros(8)`.

2. **Multi-operand gap.** Binary ops need two inputs (e.g., 16 inputs for 8-bit add), but the TTA MOVE instruction only delivers one value at a time. There's no accumulation buffer or multi-port input mechanism.

3. **Stateful FUs bypass neural computation.** ProgramCounterFU, LoadStoreFU, StackPointerFU, and UartFU all use conventional Rust logic (bitwise ops, if statements, wrapping_sub) rather than neural forward passes.

4. **No instruction fetch/decode.** `system.rs` reads from a pre-parsed `Vec<MoveOp>` and increments a Rust `usize`. There's no simulated instruction memory, no fetch from RAM, no decode stage.

5. **LoadStoreFU is incomplete.** Extensive comments in the code acknowledge the design confusion around LOAD vs STORE and how data flows through the TTA model.

6. **Fixed 8-bit word size.** Everything assumes 8-bit vectors. Widening to 32-bit touches every component.

### 3.3 What to preserve

- **BaseFU architecture** — The MLP structure with configurable layers, activations, serialization, and `train_step` is a good starting point. Will need to be generalized for deeper networks and larger input/output sizes.
- **Activation functions** — ReLU, Sigmoid, Tanh, Identity with derivatives. Well-tested.
- **Property-based testing pattern** — proptest for register roundtrip and activation bounds. This approach should be expanded massively.
- **egui GUI shell** — Will be useful in later phases for visualization/debugging.
- **Voter concept** — The threshold-based disagreement detection is the right primitive.
- **Neural register concept** — Floating-point vector storage with cleanup/quantization.

---

## 4. Architecture: Neural RISC-V Emulator

### 4.1 High-level design

```
┌─────────────────────────────────────────────────────────────┐
│                        QEMU Machine                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Neural CPU (Rust runtime)                 │  │
│  │                                                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐ │  │
│  │  │  Fetch   │→│ Decode  │→│ Execute  │→│Writeback│ │  │
│  │  │  (NN)    │  │  (NN)   │  │ (NN ALU) │  │ (NN)   │ │  │
│  │  └─────────┘  └─────────┘  └──────────┘  └────────┘ │  │
│  │       ↕             ↕            ↕            ↕       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │           Neural Register File (32×32-bit)      │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │       ↕                                               │  │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │  │
│  │  │   RAM    │  │   UART     │  │  Other MMIO      │  │  │
│  │  │(conventional)│(conventional)│  │  (conventional)   │  │  │
│  │  └──────────┘  └────────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** RAM, storage, and I/O devices are conventional (as they would be on real neural hardware — an FPGA or neural chip would use standard DRAM). The CPU pipeline stages are neural. Over time, more components become neural as feasibility is proven.

### 4.2 Neural Functional Units needed (rv32i)

**ALU operations (one NN each):**

| Operation | Inputs | Outputs | Notes |
|-----------|--------|---------|-------|
| ADD/SUB | 64 (2×32) + 1 (sub flag) | 32 + 1 (overflow) | Could be one NN with mode bit |
| AND | 64 | 32 | |
| OR | 64 | 32 | |
| XOR | 64 | 32 | |
| SLT/SLTU | 64 | 1 | Set less than (signed/unsigned) |
| SLL | 64 | 32 | Shift left logical |
| SRL | 64 | 32 | Shift right logical |
| SRA | 64 | 32 | Shift right arithmetic |

**Control operations (future neural, initially conventional scaffolding):**

| Operation | Description | Neural complexity |
|-----------|-------------|-------------------|
| Instruction decode | 32-bit instruction → opcode, operands, immediate | Medium NN |
| Branch comparison | Compare two 32-bit values, output branch decision | Small NN (similar to SLT) |
| Immediate generation | Extract and sign-extend immediates from instruction | Medium NN |
| PC calculation | Current PC + offset or absolute address | Reuse ADD NN |

### 4.3 NN architecture approaches to prototype

1. **Monolithic MLP** — Single wide network per op. For 32-bit add: 65 inputs → hidden layers → 33 outputs. May need 512-2048 hidden neurons to learn carry propagation.

2. **Bit-slice composition** — Chain of 8-bit NNs with carry propagation. Four 8-bit adder NNs (proven tractable at ~16 inputs), with inter-slice carry. Carry logic can itself be neural or conventional.

3. **Hierarchical/Skip-connection** — Network with architectural features (residual connections, attention-like mechanisms) that help learn long-range bit dependencies like carry chains.

All three will be prototyped and compared on accuracy, fault tolerance, and inference speed.

### 4.4 Redundancy architectures to test

**Approach A: Internal redundancy**
- Overprovision hidden layers (e.g., 4× more neurons than minimum needed)
- Redundant internal pathways emerge from training
- Test by killing N% of neurons/weights and measuring accuracy degradation

**Approach B: External TMR (Triple Modular Redundancy)**
- Three independent NNs trained for the same operation
- Voter compares outputs, selects majority
- Test by corrupting one NN and verifying voter catches it

**Approach C: Hybrid**
- Moderately overprovisioned NNs + 2-of-3 voting
- May offer the best performance/reliability tradeoff

Measurement methodology:
- Systematic fault injection: zero out N% of weights, flip random weights, add noise
- Sweep N from 0% to 50% in increments
- Measure accuracy at each level across full statistical test suite
- Compare A vs B vs C at equivalent parameter counts (fair comparison)

---

## 5. Phased Roadmap

### Phase 1: Neural Network Methodology (Current)

**Goal:** Prove that NNs can reliably implement every rv32i ALU operation at 32-bit width.

**Deliverables:**
- [ ] Python training pipeline (PyTorch) for arbitrary ALU ops
  - Configurable architecture (layers, widths, activations)
  - Configurable bit width (8, 16, 32)
  - Training data generation for each op
  - Weight export to portable format (JSON, ONNX, or custom)
- [ ] Validation framework
  - Exhaustive testing for 8-bit ops (65K pairs)
  - Statistical testing for 32-bit ops (100M+ samples + structured edge cases)
  - Accuracy reporting with confidence intervals
  - Regression tracking (accuracy must not decrease between code changes)
- [ ] Prototype all three NN architectures (monolithic, bit-slice, hierarchical)
  - 8-bit first (fast iteration), then scale to 32-bit
  - Compare accuracy, training time, inference speed, parameter count
- [ ] Redundancy experiments
  - Internal redundancy: vary width/depth, measure fault tolerance curves
  - External TMR: train 3 instances, implement voter, measure fault tolerance
  - Hybrid: compare at equivalent parameter budgets
- [ ] Fault injection framework
  - Weight zeroing (simulates stuck-at-zero faults)
  - Weight noise injection (simulates analog noise/drift)
  - Neuron dropout at inference time (simulates component failure)
  - Systematic sweep of fault percentages with accuracy measurement

### Phase 2: Emulator Core

**Goal:** Build an instruction-accurate rv32i CPU emulator where all ALU operations use trained NNs.

**Deliverables:**
- [ ] Rust runtime that loads trained NN weights and performs inference (ndarray or burn)
- [ ] rv32i instruction fetch, decode, execute pipeline
  - Decode: initially conventional, later neural
  - Execute: all ALU ops are neural from day one
  - Memory: conventional (load/store from/to RAM array)
  - Control flow: conventional branch/jump resolution initially
- [ ] 32 general-purpose registers (x0-x31, x0 hardwired to 0)
- [ ] Memory subsystem (byte-addressable, little-endian, conventional)
- [ ] rv32i compliance test suite (use riscv-tests or similar)
- [ ] Performance benchmarking (instructions/second, comparison to conventional emulator)
- [ ] Integration of TMR/voting for critical operations
- [ ] Progressive replacement of conventional scaffolding with neural implementations

### Phase 3: QEMU Integration + VM Exploration

**Goal:** Boot the neural CPU as a QEMU machine. Optionally explore WASM-like VM.

**Deliverables:**
- [ ] QEMU machine definition using existing rv32i frontend
  - Neural CPU as the backend implementing rv32i semantics
  - Standard QEMU memory/device model
  - UART serial console
- [ ] Boot OpenSBI or minimal firmware to serial output
- [ ] **Optional parallel track:** WASM-like VM with neural execution
  - Simpler instruction set (stack machine)
  - Potentially easier to achieve high NN accuracy
  - Test writing and compiling programs for it
- [ ] Research: path to hardware execution (GPU inference, neural accelerator, FPGA)

### Phase 4: Firmware & Software

**Goal:** Run real programs on the neural CPU.

**Deliverables:**
- [ ] Minimal firmware (bare-metal "Hello World" via UART)
- [ ] Memory test routine (proves load/store and arithmetic work in concert)
- [ ] Program loader (load ELF binary from simulated storage, jump to entry point)
- [ ] Compile and run a small C program (e.g., Fibonacci, sorting) using RISC-V GCC
- [ ] Demonstrate general-purpose computation capability

### Phase 5: Measurement & Analysis

**Goal:** Characterize the neural CPU's behavior for the research paper/report.

**Deliverables:**
- [ ] Accuracy measurements across all ALU ops under various fault conditions
- [ ] Performance comparison: neural CPU vs. conventional emulation
- [ ] Redundancy analysis: internal vs. TMR vs. hybrid, parameter efficiency
- [ ] Reliability over time: drift measurement during extended execution
- [ ] Resource analysis: NN parameter counts, memory usage, inference cost
- [ ] Environmental fault simulation: sustained noise, correlated failures, progressive degradation
- [ ] Identification of which components benefit most from neural implementation
- [ ] Recommendations for hardware implementation feasibility

---

## 6. Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 32-bit adder NN fails to reach sufficient accuracy | Blocks entire project | Bit-slice fallback (chain proven 8-bit NNs). Also explore custom loss functions that penalize bit errors differently. |
| Training time for 32-bit NNs is prohibitive | Slows iteration | GPU training via PyTorch. Start with 8-bit for rapid prototyping. Architecture search on small scale first. |
| QEMU integration is more complex than expected | Delays Phase 3 | Start with standalone Rust emulator that passes riscv-tests. QEMU integration can be deferred without blocking measurement. |
| NN inference too slow for practical emulation | Limits program complexity in Phase 4 | Batch inference, SIMD optimization, or GPU-accelerated inference in Rust (candle/burn). Also: measurement of "how slow" is itself a research result. |
| Neural control/decode proves much harder than neural ALU | Limits "everything neural" vision | Acceptable — measuring where the boundary lies IS the research contribution. Document which components are tractable and which aren't. |

---

## 7. Repository Structure (Proposed)

```
neuro_symbolic_emulator/
├── archive/                    # Previous codebase (preserved as reference)
│   ├── src/                    # Original Rust source
│   ├── assets/                 # Original trained weights
│   └── *.md                    # Original iteration docs
├── training/                   # Python training pipeline
│   ├── train.py                # Main training script
│   ├── ops/                    # Operation definitions (add, sub, and, ...)
│   ├── architectures/          # NN architecture definitions
│   ├── validation/             # Test suite and accuracy reporting
│   ├── fault_injection/        # Fault simulation framework
│   └── export/                 # Weight export utilities
├── emulator/                   # Rust rv32i emulator
│   ├── src/
│   │   ├── main.rs             # Entry point
│   │   ├── cpu/                # CPU pipeline stages
│   │   ├── nn/                 # NN inference runtime
│   │   ├── memory/             # RAM, memory bus
│   │   ├── devices/            # UART, etc.
│   │   └── gui/                # Optional debug dashboard
│   └── Cargo.toml
├── qemu/                       # QEMU machine integration
├── firmware/                   # Bare-metal firmware for the neural CPU
├── tests/                      # Integration tests
├── docs/                       # Design documents
│   ├── initial_plan.md
│   ├── cpu_architecture.md
│   ├── iteration_*.md
│   └── rules.md
├── REVIEW_AND_ROADMAP.md       # This document
└── README.md
```

---

## 8. Immediate Next Steps

1. **Archive the current codebase** into `archive/` directory
2. **Set up the Python training pipeline** (PyTorch project skeleton)
3. **Implement 8-bit adder training** as the first proof point — all three architectures
4. **Build the validation framework** with exhaustive 8-bit testing
5. **Scale to 32-bit** once 8-bit methodology is proven
6. **Begin redundancy experiments** once we have working 32-bit NNs

---

## 9. Open Questions for Future Exploration

- **TTA as alternative:** Once rv32i path is proven, explore whether a TTA instruction model maps better to neural execution (single instruction type = simpler decode NN).
- **Neural compiler:** Can programs be compiled into NN weight matrices rather than instruction sequences? This is the long-term "software as neural networks" vision.
- **Hardware mapping:** Which existing hardware platforms (GPU, TPU, neuromorphic chips like Intel Loihi, analog NN accelerators) could run this most efficiently?
- **Formal verification:** Can we prove correctness properties of the trained NNs, or are we limited to statistical testing?
- **Continuous learning:** Can FUs retrain online to recover from accumulated drift, rather than requiring offline recalibration?
