use ndarray::Array1;
use neuro_symbolic_emulator::circuit::NeuralCircuit;
use neuro_symbolic_emulator::gate::NeuralGate;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    // Load gate library
    let mut file = File::open("gate_weights.json")?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let gate_library: HashMap<String, NeuralGate> = serde_json::from_str(&json)?;

    println!("Loaded gate library: {:?}", gate_library.keys());

    test_half_adder(&gate_library)?;
    test_full_adder(&gate_library)?;

    Ok(())
}

fn test_half_adder(library: &HashMap<String, NeuralGate>) -> anyhow::Result<()> {
    println!("\n=== Testing Half Adder ===");
    // Inputs: A, B
    // Sum = XOR(A, B)
    // Carry = AND(A, B)
    
    let mut circuit = NeuralCircuit::new(2);
    
    let xor_gate = library["XOR"].clone();
    let and_gate = library["AND"].clone();
    
    let g_sum = circuit.add_gate(xor_gate);
    let g_carry = circuit.add_gate(and_gate);
    
    // Connect Inputs -> XOR
    circuit.connect(None, 0, g_sum, 0); // Input 0 -> XOR In 0
    circuit.connect(None, 1, g_sum, 1); // Input 1 -> XOR In 1
    
    // Connect Inputs -> AND
    circuit.connect(None, 0, g_carry, 0);
    circuit.connect(None, 1, g_carry, 1);
    
    // Outputs
    circuit.set_output(g_sum, 0);   // Out 0: Sum
    circuit.set_output(g_carry, 0); // Out 1: Carry
    
    // Truth Table
    let test_cases = vec![
        (vec![0.0, 0.0], (0.0, 0.0)), // S=0, C=0
        (vec![0.0, 1.0], (1.0, 0.0)), // S=1, C=0
        (vec![1.0, 0.0], (1.0, 0.0)), // S=1, C=0
        (vec![1.0, 1.0], (0.0, 1.0)), // S=0, C=1
    ];
    
    for (input, (target_sum, target_carry)) in test_cases {
        let input_arr = Array1::from(input.clone());
        let outputs = circuit.forward(&input_arr)?;
        
        // Threshold outputs
        let s = if outputs[0] > 0.5 { 1.0 } else { 0.0 };
        let c = if outputs[1] > 0.5 { 1.0 } else { 0.0 };
        
        println!("Input: {:?} -> Sum: {:.2}({}), Carry: {:.2}({})", input, outputs[0], s, outputs[1], c);
        
        if s != target_sum || c != target_carry {
            anyhow::bail!("Half Adder Failed on input {:?}", input);
        }
    }
    
    println!("Half Adder Passed!");
    Ok(())
}

fn test_full_adder(library: &HashMap<String, NeuralGate>) -> anyhow::Result<()> {
    println!("\n=== Testing Full Adder ===");
    // Inputs: A, B, Cin (Size 3)
    // Circuit:
    // HA1 = HalfAdder(A, B) -> s1, c1
    // HA2 = HalfAdder(s1, Cin) -> Sum, c2
    // CarryOut = OR(c1, c2)
    
    let mut circuit = NeuralCircuit::new(3);
    
    let xor = library["XOR"].clone();
    let and = library["AND"].clone();
    let or = library["OR"].clone();
    
    // HA1 Gates
    let ha1_xor = circuit.add_gate(xor.clone());
    let ha1_and = circuit.add_gate(and.clone());
    
    // Connect A (In 0) and B (In 1) to HA1
    circuit.connect(None, 0, ha1_xor, 0);
    circuit.connect(None, 1, ha1_xor, 1);
    circuit.connect(None, 0, ha1_and, 0);
    circuit.connect(None, 1, ha1_and, 1);
    
    // HA2 Gates (Inputs: HA1 Sum, Cin (In 2))
    let ha2_xor = circuit.add_gate(xor.clone());
    let ha2_and = circuit.add_gate(and.clone());
    
    // Connect HA1 Sum (ha1_xor out 0) -> HA2 XOR & AND input 0
    circuit.connect(Some(ha1_xor), 0, ha2_xor, 0);
    circuit.connect(Some(ha1_xor), 0, ha2_and, 0);
    
    // Connect Cin (In 2) -> HA2 XOR & AND input 1
    circuit.connect(None, 2, ha2_xor, 1);
    circuit.connect(None, 2, ha2_and, 1);
    
    // Carry Output Gate: OR(ha1_carry, ha2_carry)
    let or_gate = circuit.add_gate(or.clone());
    circuit.connect(Some(ha1_and), 0, or_gate, 0);
    circuit.connect(Some(ha2_and), 0, or_gate, 1);
    
    // Outputs
    circuit.set_output(ha2_xor, 0); // Sum
    circuit.set_output(or_gate, 0); // CarryOut
    
    let test_cases = vec![
        // A, B, Cin -> Sum, Cout
        (vec![0.0, 0.0, 0.0], (0.0, 0.0)),
        (vec![0.0, 0.0, 1.0], (1.0, 0.0)),
        (vec![0.0, 1.0, 0.0], (1.0, 0.0)),
        (vec![0.0, 1.0, 1.0], (0.0, 1.0)),
        (vec![1.0, 0.0, 0.0], (1.0, 0.0)),
        (vec![1.0, 0.0, 1.0], (0.0, 1.0)),
        (vec![1.0, 1.0, 0.0], (0.0, 1.0)),
        (vec![1.0, 1.0, 1.0], (1.0, 1.0)),
    ];
    
    for (input, (target_sum, target_cout)) in test_cases {
        let input_arr = Array1::from(input.clone());
        let outputs = circuit.forward(&input_arr)?;
        
        let s = if outputs[0] > 0.5 { 1.0 } else { 0.0 };
        let c = if outputs[1] > 0.5 { 1.0 } else { 0.0 };
        
        println!("Input: {:?} -> Sum: {:.2}, Cout: {:.2}", input, outputs[0], outputs[1]);
        
        if s != target_sum || c != target_cout {
            anyhow::bail!("Full Adder Failed on input {:?}", input);
        }
    }
    
    println!("Full Adder Passed!");
    Ok(())
}
