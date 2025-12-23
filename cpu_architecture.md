# CPU Design: Hybrid Neural-Microcode Architecture

## 1. Core Philosophy Update
Instead of simulating every standard logic gate for the Control Unit (which is inefficient and complex), we will adopt a **Hybrid Neural-Microcode** approach.

*   **Logic (ALU)**: Remains composed of "Atomic Neural Gates" (AND, OR, Adder Circuits) to preserve the "physics" of the emulator.
*   **Control (The Brain)**: A single, larger Neural Network (The **Neural Control Unit** or NCU) acts as the microcode engine. It takes the `Opcode` and `Step_Counter` as input and directly predicts the state of all ~20 control lines.
*   **Storage (Registers)**: Implemented as "Neural Latches" (Small recurrent gate loops) or specialized "Memory Cells" (if standard latches prove too unstable).

## 2. Instruction Set Architecture (ISA) - "The 8-Bit Standard"
Compatible with simple 8-bit processors.

| Opcode | Mnemonic | Function | Micro-Steps (Example) |
| :--- | :--- | :--- | :--- |
| `0x1` | **MOV A, B** | Copy Reg B to Reg A | `Bus_Write(B)`, `Reg_Load(A)` |
| `0x2` | **ADD A, B** | A = A + B | `Bus_Write(A)->ALU_X`, `Bus_Write(B)->ALU_Y`, `ALU_Out->A` |
| `0x3` | **DIV A, B** | A = A / B | (Multi-cycle neural division or LUT) |
| `0xB` | **LSH A** | Left Shift A | `Bus_Write(A)`, `Shifter_L`, `Shifter_Out->A` |
| `0xD` | **JUMP Imm** | PC = Immediate | `Bus_Write(Imm)`, `PC_Load` |

## 3. Functional Blocks

### A. The Neural Control Unit (NCU)
*   **Type**: Multi-Layer Perceptron (MLP).
*   **Input**: `[Opcode (8), Cycle_Count (4), Flags (4)]` -> 16 float inputs.
*   **Output**: `[Control_Vector (24)]` (e.g., `RegA_In`, `RegA_Out`, `ALU_Op_0`, `ALU_Op_1`, `PC_Inc`...).
*   **Concept**: We *train* this network to behave like the decoder + sequencer. This is the "Microcode" implemented as a brain.

### B. Transport Triggered Datapath (The Bus)
To support a TTA-like flexibility:
*   **Central Bus**: A "Neural Signal Highway".
*   **Components**: All functional units (Registers, ALU, PC) connect to this bus.
*   **Operation**: The NCU simply "opens" the gate logic for the Source and the Destination.
    *   *Example*: `ADD` is just moving data to the ALU inputs, waiting a cycle, and moving ALU output to destination.

### C. Neural Registers (Memory)
*   **Challenge**: Standard neural networks don't have "memory" unless recurrent.
*   **Implementation**: A `NeuralGate` configured as a D-Latch with feedback.
    *   `Next_State = (Input * Write_Enable) + (Current_State * !Write_Enable)`
    *   This logic can be learned by a single gated neuron (like a GRU cell) or built computationally from NAND gates.

## 4. Implementation Roadmap (Phase 2)

1.  **Develop `NeuralLatch`**:
    *   Create a "Bit Memory" unit using our Gate Library (Cross-coupled NANDs or explicit Recurrent Gate).
    *   Verify distinct `0` and `1` storage stability.
2.  **Build the Datapath**:
    *   Connect 2 Registers to an ALU via a Bus.
    *   Manually toggle control lines to verify data movement (`MOV`, `ADD`).
3.  **Train the NCU**:
    *   Create a dataset of `(Opcode, Cycle) -> Control_Signals`.
    *   Train an MLP to perfectly predict these signals.
    *   Integrate NCU to drive the Datapath.
