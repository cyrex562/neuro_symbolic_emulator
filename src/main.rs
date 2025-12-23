mod fu;
mod bus;
mod register;
mod voter;

use bus::{MoveOp, TransportBus};
use fu::{BaseFU, ProgramCounterFU, LoadStoreFU, StackPointerFU};

fn main() {
    println!("=== Neural TTA Emulator: Phase 2 Verified (Untrained) ===");

    let mut bus = TransportBus::new();
    
    // --- 1. Setup w/ Real (but Untrained) Units ---
    bus.add_register("R0", 8);
    bus.add_register("R1", 8);
    bus.add_register("R_FLAGS", 3); // 0=GT, 1=EQ, 2=LT
    bus.add_register("ACC", 8); // Accumulator for Bitwise
    
    // CMP
    bus.add_register("CMP_IN1", 8);
    bus.add_register("CMP_IN2", 8);
    bus.add_register("CMP_OUT", 3);
    bus.add_register("CMP_TRIGGER", 8);
    // Use FULL Implementation (Random Weights by default)
    bus.add_unit("CMP", Box::new(BaseFU::create_comparator()));
    
    // PC
    bus.add_register("PC_IN", 8);
    bus.add_register("PC_TRIGGER", 8); 
    bus.add_unit("PC", Box::new(ProgramCounterFU::new()));
    
    // BITWISE
    bus.add_register("BIT_IN1", 8);
    bus.add_register("BIT_IN2", 8);
    bus.add_register("BIT_MODE", 3);
    bus.add_register("BIT_OUT", 8);
    bus.add_register("BIT_TRIGGER", 3); // Mode is usually last set
    // Use FULL Implementation (Random Weights by default)
    bus.add_unit("BIT", Box::new(BaseFU::create_bitwise()));

    // --- 2. Program ---
    // Note: Since units are untrained, logic checks (Guard) will see random outputs.
    // The Demo will effectively show the PIPELINE works, but the JUMP might/might not happen.
    
    // Test 1: Branch if EQ (Should Jump)
    let r0 = register::NeuralRegister::from_symbolic(8, 10);
    if let Some(r) = bus.registers.get_mut("R0") { *r = r0.clone(); }
    if let Some(r) = bus.registers.get_mut("R1") { *r = r0.clone(); }
    
    // Test 2: Bitwise AND (10 & 10 = 10)
    
    let program = vec![
        // --- Branch Test ---
        MoveOp { src: "R0".to_string(), dest: "CMP_IN1".to_string(), guard: None },
        MoveOp { src: "R1".to_string(), dest: "CMP_IN2".to_string(), guard: None },
        MoveOp { src: "R1".to_string(), dest: "CMP_TRIGGER".to_string(), guard: None },
        MoveOp { src: "CMP_OUT".to_string(), dest: "R_FLAGS".to_string(), guard: None },
        
        // Conditional Jump: If R_FLAGS (GT index 0) is set, Move TARGET -> PC
        // Note: With random weights, R_FLAGS is garbage. We just verify the pipeline steps execute.
    ];
    
    // Let's Run Branch Test with R0=20, R1=10 (GT).
    println!("State: R0=20, R1=10. testing GT Guard (Untrained).");
    let r20 = register::NeuralRegister::from_symbolic(8, 20);
    let r10 = register::NeuralRegister::from_symbolic(8, 10);
    if let Some(r) = bus.registers.get_mut("R0") { *r = r20; }
    if let Some(r) = bus.registers.get_mut("R1") { *r = r10; } 
    
    // Jump Target: 5.
    let target = register::NeuralRegister::from_symbolic(8,5);
    bus.add_register("TARGET", 8);
    if let Some(r) = bus.registers.get_mut("TARGET") { *r = target; }
    
    let jumps = vec![
        MoveOp { src: "R0".to_string(), dest: "CMP_IN1".to_string(), guard: None },
        MoveOp { src: "R1".to_string(), dest: "CMP_IN2".to_string(), guard: None },
        MoveOp { src: "R1".to_string(), dest: "CMP_TRIGGER".to_string(), guard: None },
        MoveOp { src: "CMP_OUT".to_string(), dest: "R_FLAGS".to_string(), guard: None },
        
        // Conditional Jump: If R_FLAGS (GT index 0) is set, Move TARGET -> PC
        MoveOp { src: "TARGET".to_string(), dest: "PC_IN".to_string(), guard: Some("R_FLAGS".to_string()) },
        MoveOp { src: "TARGET".to_string(), dest: "PC_TRIGGER".to_string(), guard: Some("R_FLAGS".to_string()) },
    ];

    println!("Executing Program...");
    for (i, op) in jumps.iter().enumerate() {
        println!("[{}] Executing... Guard: {:?}", i, op.guard);
        let status = bus.execute(op);
        println!("  Status: {}", status);
        bus.tick_all(); 
        
        // Peek PC
    }
}
