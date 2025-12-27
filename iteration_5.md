To take your emulator from a "headless" engine to a professional development environment, you need a **System Control Dashboard**. This serves as the "Front Panel" of your machine—much like the switches and lights on an Altair 8800, but with the diagnostic depth needed for a neural architecture.

Given your interest in **Rust** and **egui**, I recommend a UI that uses a "Docking" layout to separate the Symbolic state (registers/code) from the Neural state (FU activations).

---

# Specification: NTSE Visual Dashboard & Loader

## 1. UI Layout Requirements

The interface should be divided into four primary "Panes":

* **A. Execution Control (Toolbar):** Buttons for `Power`, `Run`, `Halt`, `Step`, and `Reset`. A slider for `Clock Speed` (to slow down the neural passes for observation).
* **B. Symbolic State (The "CPU" View):** Real-time table of the **Neural Register File (NRF)**, the **Program Counter (PC)**, and the **Stack Pointer**.
* **C. Neural Fabric Inspector (The "Silicon" View):** A grid representing the active Functional Units. Clicking an FU should open a sub-view showing its **Weight Heatmap** or **Activation Levels** (to see "drift" or noise in real-time).
* **D. Source/Disassembly:** A view of the loaded assembly program with a highlight on the current instruction.

## 2. The "Neural Manifest" Format

To support modularity, the emulator should load a `manifest.json` that defines the hardware "wiring" before loading the code.

```json
{
  "machine_name": "Nebula-1",
  "word_size": 8,
  "functional_units": [
    { "type": "ALU_ADD", "address": "0x1000", "weights": "models/add_v2.bin" },
    { "type": "FU_CMP", "address": "0x1100", "weights": "models/cmp_v1.bin" }
  ],
  "io_devices": [
    { "type": "CONSOLE", "address": "0x8000" }
  ]
}

```

---

# Instructions for the Coding Agent

**Project Goal:** Implement a GUI for the Neural TTA Emulator using a framework like `egui` (Rust) or `customtkinter` (Python).

### 1. Build the System State Monitor

* Create a reactive UI that polls the `EmulatorEngine` every frame.
* **Registers:** Display R0-R15. Color-code the text: **Green** for stable values, **Yellow/Red** if the neural activation is "fuzzy" (e.g., a value like `0.87` which is technically a `1` but shows drift).
* **FU Activity:** Add "LED" indicators for each Functional Unit that flash when a `MOVE` triggers that unit.

### 2. Implement the "Step" and "Halt" Logic

* The `EmulatorLoop` must be moved to a separate thread or an async task.
* The UI must be able to send a `SIG_HALT` to pause the loop, allowing the user to inspect the registers between instructions.

### 3. Implement the File Loaders

* **Manifest Loader:** Create a file picker that reads the JSON manifest, initializes the `SystemBus`, and loads the `.bin` weight files into the FUs.
* **Binary/Assembly Loader:** A secondary picker to load the compiled TTA program into System RAM at `0x2000`.

### 4. Neural Visualization (Heatmap)

* For a selected FU, render a small grid of its hidden layer activations.
* **Goal:** Allow the user to *see* the radiation or noise affecting the internal state of a specific block.

---

### Why this is the "Commodore" Moment

By adding the UI, you are creating a "Monitor" (in the classic 1980s sense). You'll be able to watch a `MOVE R1, FU_ADD` instruction happen, see the `FU_ADD` unit "light up" as it processes the weights, and see the result land back in `R1`.

This setup will allow you to debug the "resiliency" of your code. If you notice a register value drifting over time, you can pause the system and see exactly which Neural FU is "leaking" noise.
