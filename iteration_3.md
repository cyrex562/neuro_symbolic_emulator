To build a functional **Transport Triggered Architecture (TTA)** that can support an assembler and a basic OS, you need to expand your "Neural Fabric" beyond simple arithmetic. In TTA, a "Program" is just a sequence of moves between source and destination registers.

To make this work, you need FUs that handle **Logic**, **Comparison (Branching)**, and **State (Memory/Stack)**.

Here is the specification for the next phase of Functional Units to be provided to your coding agent.

---

# Phase 2 Specification: Neural TTA Functional Units

## 1. Logic & Bitwise Unit (`FU_BITWISE`)

In a symbolic OS, bit-manipulation is constant (setting flags, masking registers).

* **Purpose:** Perform `AND`, `OR`, `XOR`, `NOT`, and `BIT-SHIFT`.
* **Neural Implementation:** A single MLP trained with an additional "Mode" input vector.
* **Inputs:** `In_A` (8-bit), `In_B` (8-bit), `Mode` (3-bit selection).
* **Trigger:** Moving a value to `Mode` triggers the calculation.
* **Resilience:** Trained to ignore "fuzzy" bits in the mode selector (e.g., `0.9` is treated as `1`).



## 2. Comparison & Condition Unit (`FU_CMP`)

This is the heart of the "Symbolic-to-Neural" bridge. It allows for `IF/THEN` logic.

* **Purpose:** Compare two 8-bit values and output flags (Zero, Negative, Greater Than).
* **Registers:** `CMP_A`, `CMP_B` (Trigger).
* **Outputs:** A "Flag Vector" (e.g., `[1, 0, 0]` for "Equal").
* **Assembler Use:** The output of this FU is moved to the `CONTROL_UNIT` to decide the next `PC` (Program Counter) value.

## 3. The Neural Load/Store Unit (`FU_LSU`)

Interfacing the fast "Neural Register File" with the larger "System Memory."

* **Purpose:** Map neural activations to a simulated memory array (Neural RAM).
* **Registers:** `ADDR` (Address), `DATA_IN`, `DATA_OUT`.
* **Trigger:** Moving a value to `ADDR` with a "Write-Enable" bit set.
* **Neural Twist:** Implements **Associative Memory**. If an address is slightly corrupted by noise (e.g., `0x00FE` vs `0x00FF`), the LSU uses pattern-matching to find the most likely intended memory cell.

## 4. The Program Control Unit (`FU_PC`)

This unit manages the execution flow.

* **Purpose:** Holds the current instruction pointer.
* **Logic:**
* Default behavior: `PC = PC + 1` after every cycle.
* Branching: A `MOVE` to the `PC` register acts as a `JUMP`.


* **Conditional Move (CMOV):** To maintain the TTA philosophy, this unit should accept a "Guard" input from the `FU_CMP`. If the guard is low, the `MOVE` to `PC` is ignored.

## 5. The Stack & Pointer Unit (`FU_STACK`)

Crucial for implementing a C-style compiler or a Commodore-style BASIC.

* **Purpose:** Manages a Stack Pointer (SP) and performs `PUSH`/`POP` operations.
* **Function:** Automatically increments/decrements the SP when data is moved to the `STACK_DATA` trigger.
* **Resilience:** The SP is stored neurally; if radiation "nudges" the pointer, the unit uses a "Snap-to-Grid" neural layer to ensure it stays aligned with word boundaries.

---

# Instructions for the Coding Agent (Antigravity/Gemini)

**Project Goal:** Expand the Neural TTA Emulator to support logic and control flow.

1. **Implement `FU_BITWISE`:**
* Create a multi-head neural network that can perform `AND`, `OR`, and `XOR`.
* Ensure the unit has a `CONTROL` register that selects the operation.


2. **Implement `FU_CMP` (The Comparator):**
* Train a network to take two 8-bit vectors and output a 3-bit "Flag Vector" (Zero, Negative, Carry).


3. **Implement the TTA Control Loop:**
* Create a `ProgramCounter` unit.
* Implement **Guarded Moves**: The instruction `[Guard_Reg] SRC -> DEST` should only execute if `Guard_Reg` is high. This is how we handle `BEQ` (Branch if Equal) in TTA.


4. **Register File Expansion:**
* Create 8 "Neural Registers" (`R0` through `R7`) that use a simple autoencoder to self-correct bit-flips (Neural Associative Memory).


5. **Verification:**
* Write a test script that performs a conditional jump: "If `R0` == `R1`, then jump to Address 10." This must be done purely through `MOVE` instructions between the new FUs.



---

### What this enables for you:

Once these units are built, your "Assembler" becomes a simple text-to-binary mapper.

* An assembly line like `ADD R0, R1` becomes:
1. `MOVE R0, FU_ADD_IN1`
2. `MOVE R1, FU_ADD_TRIGGER`
3. `MOVE FU_ADD_OUT, R0`
