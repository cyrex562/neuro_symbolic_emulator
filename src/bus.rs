use crate::fu::NeuralFunctionalUnit;
use anyhow::{anyhow, Result};
use ndarray::Array1;
use std::collections::HashMap;

/// The Neural Transport Bus.
/// Connects Sources (Registers, FU Outputs) to Destinations (Registers, FU Inputs).
pub struct TransportBus {
    // In a real neural sim, this would be a routing network.
    // For TTA emulation, we map addresses to direct function calls.
}

impl TransportBus {
    pub fn new() -> Self {
        Self {}
    }

    // TTA Move: SRC -> DEST
    // In a full TTA, this moves data to a register.
    // If that register is a Trigger, the Unit fires.
    // We need the system state to execute this, so Bus might just be a passive trait or helper.
    // For now, we'll implement logic in the System struct in `main.rs` or `system.rs`.
    // The Bus struct here can hold routing metadata or "Neural Router" weights in the future.
}
