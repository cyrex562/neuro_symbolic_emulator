use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context};
use ndarray::Array1;
use crate::system::SystemEmulator;
use crate::register::NeuralRegister;
use crate::bus::SystemBus;
use crate::fu::{BaseFU, UartFU};


#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub ram_size: usize,
    pub units: Vec<UnitConfig>,
    pub program_path: Option<String>,
    pub ram_init: Option<HashMap<String, Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
pub struct UnitConfig {
    pub name: String,
    pub address: u16,
    pub unit_type: String, // "comparator", "bitwise", "uart", "generic"
    pub weights_path: Option<String>,
}

// Helper to deserialize MoveOps
#[derive(Deserialize)]
struct ProgramFile {
    ops: Vec<crate::bus::MoveOp>,
}

pub fn load_manifest(path: &Path, console_sink: Option<std::sync::Arc<std::sync::Mutex<String>>>) -> Result<SystemEmulator> {
    let file = std::fs::File::open(path)?;
    let manifest: Manifest = serde_json::from_reader(file)?;

    let mut bus = SystemBus::new();

    // 1. Initialize RAM
    // Pre-populate RAM if ram_init is present
    if let Some(init_map) = &manifest.ram_init {
        for (addr_str, data_vec) in init_map {
             if let Ok(addr) = addr_str.parse::<u16>() {
                 bus.ram.insert(addr, Array1::from(data_vec.clone()));
             }
        }
    }

    // 2. Initialize Registers (Fixed R0-R15 for now)
    for i in 0..16 {
        bus.registers.insert(i, NeuralRegister::new(8));
    }

    // 3. Initialize Functional Units
    for unit_cfg in manifest.units {
        // The original code used a match statement to create the unit, then added it.
        // The new instruction implies an if-else if structure and direct addition.
        // We'll adapt the existing logic to this new structure.
        if unit_cfg.unit_type == "uart" {
            // Inject sink if available
            let fu = if let Some(sink) = &console_sink {
                UartFU::with_sink(sink.clone())
            } else {
                UartFU::new()
            };
            // UART is MMIO usually, but manifest treats as unit?
            // Manifest has address 32768 (0x8000), which is MMIO.
            // But valid Unit range is 0x1000..0x1FFF.
            // Bus adds to MMIO if addr >= 0x8000.
            if unit_cfg.address >= 0x8000 {
                bus.mmio.insert(unit_cfg.address, Box::new(fu));
            } else {
                bus.units.insert(unit_cfg.address, Box::new(fu));
            }
        } else if unit_cfg.unit_type == "comparator" {
            let fu = BaseFU::create_comparator();
            if let Some(w_path) = unit_cfg.weights_path {
                if Path::new(&w_path).exists() {
                   // fu.load_weights(&w_path)?;
                }
            }
            if unit_cfg.address >= 0x8000 {
                bus.mmio.insert(unit_cfg.address, Box::new(fu));
            } else {
                bus.units.insert(unit_cfg.address, Box::new(fu));
            }
        } else if unit_cfg.unit_type == "bitwise" {
            let fu = BaseFU::create_bitwise();
            if let Some(w_path) = unit_cfg.weights_path {
                if Path::new(&w_path).exists() {
                   // fu.load_weights(&w_path)?;
                }
            }
            if unit_cfg.address >= 0x8000 {
                bus.mmio.insert(unit_cfg.address, Box::new(fu));
            } else {
                bus.units.insert(unit_cfg.address, Box::new(fu));
            }
        } else {
            // Default generic or error
            let fu = BaseFU::create_random(8, 8, 8); // Dummy
            if unit_cfg.address >= 0x8000 {
                bus.mmio.insert(unit_cfg.address, Box::new(fu));
            } else {
                bus.units.insert(unit_cfg.address, Box::new(fu));
            }
        }
    }
    
    let mut emulator = SystemEmulator::new(bus);

    // 4. Load Program if specified
    if let Some(prog_path_str) = manifest.program_path {
        let prog_path = path.parent().unwrap_or(Path::new(".")).join(prog_path_str);
        if prog_path.exists() {
            let pfile = std::fs::File::open(prog_path)?;
            let ops: Vec<crate::bus::MoveOp> = serde_json::from_reader(pfile)?;
            emulator.load_program(ops);
        } else {
             eprintln!("Warning: Program file not found at {:?}", prog_path);
        }
    }

    Ok(emulator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_manifest_parsing() {
        let json_content = r#"
        {
            "ram_size": 1024,
            "units": [
                {
                    "name": "TestUART",
                    "address": 32768,
                    "unit_type": "uart",
                    "weights_path": null
                },
                 {
                    "name": "TestCmp",
                    "address": 4096,
                    "unit_type": "comparator",
                    "weights_path": null
                }
            ]
        }
        "#;
        
        // Create temp file
        let mut temp_file = std::env::temp_dir();
        temp_file.push("test_manifest.json");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(json_content.as_bytes()).unwrap();
        
        // Test Load
        let mut sys = load_manifest(&temp_file, None).expect("Failed to load manifest");
        
        // Verify Config
        // Check MMIO (UART at 0x8000 = 32768)
        assert!(sys.bus.mmio.contains_key(&32768));
        // Check FU (Cmp at 0x1000 = 4096)
        assert!(sys.bus.units.contains_key(&4096));
        // Check Registers
        assert!(sys.bus.registers.contains_key(&0));
        assert!(sys.bus.registers.contains_key(&15));
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
}
