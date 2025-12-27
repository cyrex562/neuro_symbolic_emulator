use crate::fu::{BaseFU, NeuralFunctionalUnit};
use crate::register::NeuralRegister;
use ndarray::Array1;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveOp {
    pub src: u16,  // Address
    pub dest: u16, // Address
    pub guard: Option<u16>, // Address of Guard Register
}

pub struct SystemBus {
    pub registers: HashMap<u16, NeuralRegister>, // 0x0000 - 0x0FFF (Mapped by ID)
    pub units: HashMap<u16, Box<dyn NeuralFunctionalUnit>>, // 0x1000 range. Mapped by Base Port Address?
    pub ram: HashMap<u16, Array1<f32>>, // 0x2000 - 0x7FFF
    pub mmio: HashMap<u16, Box<dyn NeuralFunctionalUnit>>, // 0x8000+
    
    // Phase 9: Inspection Cache (Addr -> (Last Input, Last Output))
    pub fu_io_cache: HashMap<u16, (Array1<f32>, Array1<f32>)>,
}

impl SystemBus {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            units: HashMap::new(),
            ram: HashMap::new(),
            mmio: HashMap::new(),
            fu_io_cache: HashMap::new(),
        }
    }

    pub fn add_register(&mut self, addr: u16, width: usize) {
        self.registers.insert(addr, NeuralRegister::new(width));
    }

    pub fn add_unit(&mut self, base_addr: u16, unit: Box<dyn NeuralFunctionalUnit>) {
        // We might map multiple ports for one unit (e.g. IN1, IN2, TRIGGER, OUT)
        // For simplicity, we store the unit pointer at the Base Address, 
        // and dispatch logic handles offsets (Base+0=IN1, Base+1=IN2...).
        // BUT `units` map stores generic unit. 
        // Let's store unit at `base_addr`.
        self.units.insert(base_addr, unit);
    }
    
    pub fn add_mmio(&mut self, addr: u16, device: Box<dyn NeuralFunctionalUnit>) {
        self.mmio.insert(addr, device);
    }

    /// The core System Dispatch
    pub fn execute(&mut self, op: &MoveOp) -> String {
        // 0. Check Guard
        if let Some(guard_addr) = op.guard {
            let guard_val = self.read_mem(guard_addr);
            // Check LSB or Sum > 0.5
            if guard_val.get(0).unwrap_or(&0.0) <= &0.5 {
                 return "Skipped (Guard Low)".to_string();
            }
        }

        // 1. Read Source
        let data = self.read_mem(op.src);

        // 2. Write Destination
        let dest_desc = self.write_mem(op.dest, &data);
        
        // Format Log: "Moved [0, 0, 0, 1...] to D"
        // Show as integer vector for compactness if values are near 0/1
        let vec_str: Vec<String> = data.iter().take(8).map(|&v| {
            if (v - 1.0).abs() < 0.1 { "1".to_string() }
            else if v.abs() < 0.1 { "0".to_string() }
            else { format!("{:.1}", v) }
        }).collect();
        let val_str = format!("[{}]", vec_str.join(", "));
        
        format!("Moved {} to {}", val_str, dest_desc)
    }

    fn read_mem(&mut self, addr: u16) -> Array1<f32> {
        if addr < 0x1000 {
            // NRF
            if let Some(reg) = self.registers.get(&addr) {
                return reg.read();
            }
        } else if addr < 0x2000 {
            // FU Read (Output ports)
            // Assuming (Addr & 0xFFF0) is Unit Base? 
            // Simplified: If key exists in `units`, query it?
            // Units usually provide output via `forward` return value or state.
            // If we want to READ from a unit (like Status), we need `read()` trait method?
            // For now, return Zeros mock.
        } else if addr < 0x8000 {
            // RAM
            if let Some(val) = self.ram.get(&addr) {
                return val.clone();
            }
        } else {
             // MMIO Read (e.g. Keyboard)
             if let Some(dev) = self.mmio.get_mut(&addr) {
                  // Hack: using forward as read? Or specific read?
                  // TTA usually reads from a "Output Register" of the Unit.
                  // Let's assume MMIO read returns mock.
             }
        }
        Array1::zeros(8) // Default
    }

    fn write_mem(&mut self, addr: u16, data: &Array1<f32>) -> String {
        if addr < 0x1000 {
            if let Some(reg) = self.registers.get_mut(&addr) {
                reg.write(data);
                return format!("R{}", addr);
            }
        } else if addr < 0x2000 {
            // FU Write (Inputs or Trigger)
            if let Some(unit) = self.units.get_mut(&addr) {
                let output = unit.forward(data); 
                self.fu_io_cache.insert(addr, (data.clone(), output));
                return format!("FU[0x{:X}]", addr);
            }
            
        } else if addr < 0x8000 {
            self.ram.insert(addr, data.clone());
            return format!("RAM[0x{:X}]", addr);
        } else {
            // MMIO
            if let Some(dev) = self.mmio.get_mut(&addr) {
                let output = dev.forward(data);
                self.fu_io_cache.insert(addr, (data.clone(), output));
                if addr == 0x8000 { return "UART".to_string(); }
                return format!("MMIO[0x{:X}]", addr);
            }
        }
        format!("Unknown[0x{:X}]", addr)
    }
    
    pub fn tick_all(&mut self) {
        for unit in self.units.values_mut() {
            unit.tick();
        }
        for dev in self.mmio.values_mut() {
            dev.tick();
        }
        // PC tick logic needs to happen here too if PC is a unit.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock FU for testing bus dispatch
    #[derive(Debug, Clone)]
    struct MockFU {
        pub last_in: Array1<f32>,
    }
    impl NeuralFunctionalUnit for MockFU {
        fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
            self.last_in = input.clone();
            Array1::from(vec![1.0, 2.0, 3.0]) // Mock output
        }
        fn perturb(&mut self, _a: f32) {}
    }

    #[test]
    fn test_bus_memory_map() {
        let mut bus = SystemBus::new();
        bus.add_register(0, 8);
        bus.add_unit(0x1000, Box::new(MockFU { last_in: Array1::zeros(0) }));
        bus.ram.insert(0x2000, Array1::from(vec![42.0]));
        
        // 1. Test Register Write/Read
        let data = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        bus.write_mem(0, &data);
        let read_back = bus.read_mem(0);
        assert_eq!(read_back, data);
        
        // 2. Test RAM Write/Read
        let ram_data = Array1::from(vec![99.9]);
        bus.write_mem(0x2000, &ram_data);
        let ram_read = bus.read_mem(0x2000);
        assert_eq!(ram_read, ram_data);
    }
    
    #[test]
    fn test_guard_logic() {
        let mut bus = SystemBus::new();
        bus.add_register(0, 1); // Source
        bus.add_register(1, 1); // Dest
        bus.add_register(2, 1); // Guard
        
        // Init Source = 1.0
        bus.write_mem(0, &Array1::from(vec![1.0]));
        // Init Dest = 0.0
        bus.write_mem(1, &Array1::from(vec![0.0]));
        
        // Case 1: Guard Low (0.0) -> No Move
        bus.write_mem(2, &Array1::from(vec![0.0]));
        let op = MoveOp { src: 0, dest: 1, guard: Some(2) };
        let res = bus.execute(&op);
        assert!(res.contains("Skipped"));
        assert_eq!(bus.read_mem(1)[0], 0.0);
        
        // Case 2: Guard High (1.0) -> Move
        bus.write_mem(2, &Array1::from(vec![1.0]));
        let res = bus.execute(&op);
        assert!(!res.contains("Skipped"));
        assert_eq!(bus.read_mem(1)[0], 1.0);
    }
}
