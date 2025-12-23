use crate::fu::{BaseFU, NeuralFunctionalUnit};
use crate::register::NeuralRegister;
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MoveOp {
    pub src: String,
    pub dest: String,
    pub guard: Option<String>, // Reg to check before execution
}

pub struct TransportBus {
    pub registers: HashMap<String, NeuralRegister>,
    pub units: HashMap<String, Box<dyn NeuralFunctionalUnit>>,
    // We map "Trigger" addresses to Unit IDs
    pub trigger_map: HashMap<String, String>,
    pub pc_name: String, // Name of the PC Unit
}

impl TransportBus {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            units: HashMap::new(),
            trigger_map: HashMap::new(),
            pc_name: "PC".to_string(),
        }
    }

    pub fn add_register(&mut self, name: &str, width: usize) {
        self.registers.insert(name.to_string(), NeuralRegister::new(width));
    }

    pub fn add_unit(&mut self, name: &str, unit: Box<dyn NeuralFunctionalUnit>) {
        self.units.insert(name.to_string(), unit);
    }

    /// Execute a single Move instruction
    pub fn execute(&mut self, op: &MoveOp) -> String {
        // 0. Check Guard
        if let Some(guard_reg_name) = &op.guard {
             if let Some(r) = self.registers.get(guard_reg_name) {
                 // Logic: If any bit > 0.5? Or Logic High?
                 // Prompt: "If guard is low, the MOVE is ignored."
                 // Let's assume Guard is a single bit or we check max value.
                 // Simple sum check for now? or LSB?
                 // Let's check if the first element > 0.5.
                 if r.state.get(0).unwrap_or(&0.0) <= &0.5 {
                     return "Skipped (Guard Low)".to_string();
                 }
             } else {
                 return format!("Error: Guard {} not found", guard_reg_name);
             }
        }

        // 1. Read Source
        let data = if let Some(reg) = self.registers.get(&op.src) {
            reg.read()
        } else {
            return format!("Error: Source {} not found", op.src);
        };

        // 2. Write Destination
        if let Some(reg) = self.registers.get_mut(&op.dest) {
            reg.write(&data);
        } else {
            return format!("Error: Dest {} not found", op.dest);
        }

        // 3. Check Trigger
        if op.dest.ends_with("_TRIGGER") {
            let unit_name = op.dest.trim_end_matches("_TRIGGER");
            return self.fire_unit(unit_name);
        }
        
        // Special Case: Writing to "PC" implies a jump (handled by FU_PC forward).
        // But if we write to PC *without* trigger suffix?
        // In our model, PC is a Unit. If we write to "PC_IN", does it trigger?
        // Let's assume we map "PC" register to "PC_IN".
        // And "PC_TRIGGER" fires the update.
        // For simplicity, if dest == "PC", we might want to trigger update implicitly?
        // No, stick to TTA: Write to Trigger to execute.
        
        "Moved".to_string()
    }

    fn fire_unit(&mut self, unit_name: &str) -> String {
        let mut input_vec: Vec<f32> = Vec::new();
        
        let in1_name = format!("{}_IN1", unit_name);
        let in2_name = format!("{}_IN2", unit_name);
        // Also support single input "UNIT_IN"
        let in_name = format!("{}_IN", unit_name);
        
        // Collect inputs
        if let Some(r) = self.registers.get(&in_name) {
             input_vec.extend(r.read().iter());
        }
        if let Some(r1) = self.registers.get(&in1_name) {
            input_vec.extend(r1.read().iter());
        }
        if let Some(r2) = self.registers.get(&in2_name) {
            input_vec.extend(r2.read().iter());
        }
        // Also support "MODE" input for Bitwise
        let mode_name = format!("{}_MODE", unit_name);
        if let Some(rm) = self.registers.get(&mode_name) {
             input_vec.extend(rm.read().iter());
        }
        
        if input_vec.is_empty() {
             // Maybe it's a unit with NO inputs? (Like a sensitive RNG?)
             // Or maybe inputs are the trigger register itself?
             // Allowed for PC if we treat trigger write as input.
        }
        
        let input_arr = Array1::from(input_vec);
        
        if let Some(unit) = self.units.get_mut(unit_name) {
            let output = unit.forward(&input_arr);
            
            let out_name = format!("{}_OUT", unit_name);
            if let Some(out_reg) = self.registers.get_mut(&out_name) {
                out_reg.write(&output);
                return format!("Unit {} Executed.", unit_name);
            }
             // For PC unit, output is current PC.
             if unit_name == "PC" {
                 // PC Update logic handles itself internally?
                 // See ProgramCounterFU forward().
                 return "PC Updated".to_string();
             }
        }
        
        format!("Unit {} not found", unit_name)
    }
    
    pub fn tick_all(&mut self) {
        for unit in self.units.values_mut() {
            unit.tick();
        }
    }
}
