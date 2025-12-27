use crate::bus::{SystemBus, MoveOp};
use crate::fu::UartFU;

// System struct removed in favor of SystemEmulator


// Extended System struct to hold the ROM for iteration 4 transparency
pub struct SystemEmulator {
    pub bus: SystemBus,
    pub program: Vec<MoveOp>,
    pub pc: usize, // Index in program vector
    
    // Phase 6: Stats & Logs
    pub total_steps: usize,
    pub logs: Vec<String>,
    
    // Phase 7: Console Output
    pub console_sink: std::sync::Arc<std::sync::Mutex<String>>,
}

impl SystemEmulator {
    pub fn new(bus: SystemBus) -> Self {
        Self {
            bus,
            program: Vec::new(),
            pc: 0,
            total_steps: 0,
            logs: Vec::new(),
            console_sink: std::sync::Arc::new(std::sync::Mutex::new(String::new())),
        }
    }

    pub fn default() -> Self {
        let mut bus = SystemBus::new();
        // Setup Registers R0-R15
        for i in 0..16 {
            bus.add_register(i, 8);
        }
        // UART at 0x8000
        bus.add_mmio(0x8000, Box::new(UartFU::new()));
        
        Self::new(bus)
    }
    
    pub fn load_firmware(&mut self) {
        // Init default FUs if needed.
    }
    
    pub fn load_program(&mut self, prog: Vec<MoveOp>) {
        self.program = prog;
    }
    
    pub fn step(&mut self) -> bool {
        if self.pc >= self.program.len() {
             return false; // Halted
        }
        
        let op = &self.program[self.pc];
        let exec_log = self.bus.execute(op);
        
        // Log the result
        // TODO: Circular buffer optimization if logs get huge
        self.logs.push(format!("[Step {} | PC {}] {}", self.total_steps, self.pc, exec_log));
        
        // Clock Tick
        self.bus.tick_all();
        self.pc += 1; // Simple PC increment
        self.total_steps += 1;
        
        true
    }
}
