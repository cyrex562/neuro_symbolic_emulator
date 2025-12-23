use ndarray::Array1;
use neuro_symbolic_emulator::circuit::NeuralCircuit;
use neuro_symbolic_emulator::gate::NeuralGate;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use rand::Rng;

fn main() -> anyhow::Result<()> {
    // Load gate library
    let mut file = File::open("gate_weights.json")?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let gate_library: HashMap<String, NeuralGate> = serde_json::from_str(&json)?;

    println!("Loaded gate library.");

    test_ripple_carry_adder(&gate_library)?;

    Ok(())
}

fn test_ripple_carry_adder(library: &HashMap<String, NeuralGate>) -> anyhow::Result<()> {
    println!("\n=== Testing 4-bit Ripple Carry Adder ===");
    // Inputs: A (4 bits), B (4 bits), Cin (1 bit, usually 0)
    // Total Inputs: 9
    // Circuit Graph:
    // FA0(A0, B0, Cin) -> S0, C0
    // FA1(A1, B1, C0)  -> S1, C1
    // FA2(A2, B2, C1)  -> S2, C2
    // FA3(A3, B3, C2)  -> S3, Cout
    
    // Input Mapping:
    // 0: A0, 1: B0
    // 2: A1, 3: B1
    // 4: A2, 5: B2
    // 6: A3, 7: B3
    // 8: Cin
    
    let mut circuit = NeuralCircuit::new(9);
    
    let xor = library["XOR"].clone();
    let and = library["AND"].clone();
    let or = library["OR"].clone();

    // Helper to add a Full Adder
    // Returns (Sum_Node_Id, Cout_Node_Id)
    // We need to know where to connect inputs. 
    // This helper is tricky because we need to connect to existing nodes or inputs.
    // Instead, let's just build it iteratively.
    
    // We need to track the Carry Output from previous stage.
    // Initial Carry In is Input 8.
    let mut carry_src: (Option<usize>, usize) = (None, 8); // (NodeID, OutputIdx)

    let mut sum_bit_nodes = Vec::new();

    for bit in 0..4 {
        // A_idx = bit * 2
        // B_idx = bit * 2 + 1
        let a_idx = bit * 2;
        let b_idx = bit * 2 + 1;

        // --- Full Adder Logic ---
        // HA1
        let ha1_xor = circuit.add_gate(xor.clone());
        let ha1_and = circuit.add_gate(and.clone());
        
        // Connect A, B to HA1
        circuit.connect(None, a_idx, ha1_xor, 0);
        circuit.connect(None, b_idx, ha1_xor, 1);
        circuit.connect(None, a_idx, ha1_and, 0);
        circuit.connect(None, b_idx, ha1_and, 1);
        
        // HA2
        let ha2_xor = circuit.add_gate(xor.clone());
        let ha2_and = circuit.add_gate(and.clone());
        
        // Connect HA1 Sum (ha1_xor out 0) matches to HA2
        circuit.connect(Some(ha1_xor), 0, ha2_xor, 0);
        circuit.connect(Some(ha1_xor), 0, ha2_and, 0);
        
        // Connect Carry In (carry_src) matches to HA2
        circuit.connect(carry_src.0, carry_src.1, ha2_xor, 1);
        circuit.connect(carry_src.0, carry_src.1, ha2_and, 1);
        
        // Carry Output Gate: OR(ha1_carry, ha2_carry)
        let or_gate = circuit.add_gate(or.clone());
        circuit.connect(Some(ha1_and), 0, or_gate, 0);
        circuit.connect(Some(ha2_and), 0, or_gate, 1);
        
        // Store Sum Node (ha2_xor)
        sum_bit_nodes.push(ha2_xor);
        
        // Update Carry Source for next stage
        carry_src = (Some(or_gate), 0);
    }
    
    // Set Outputs: S0, S1, S2, S3, Cout
    for node in sum_bit_nodes {
        circuit.set_output(node, 0);
    }
    circuit.set_output(carry_src.0.unwrap(), 0); // Final Cout
    
    // Verify with random inputs
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let a: u8 = rng.gen_range(0..16); // 4 bits
        let b: u8 = rng.gen_range(0..16);
        let cin = 0; // Test with 0 carry in
        
        let expected_sum = a as u16 + b as u16 + cin as u16;
        // Expected outputs (S0..S3, Cout)
        
        // Prepare input vector
        let mut inputs = Vec::new();
        // 0: A0, 1: B0, ...
        for bit in 0..4 {
            inputs.push(if (a >> bit) & 1 == 1 { 1.0 } else { 0.0 });
            inputs.push(if (b >> bit) & 1 == 1 { 1.0 } else { 0.0 });
        }
        inputs.push(0.0); // Cin
        
        let input_arr = Array1::from(inputs.clone());
        let outputs = circuit.forward(&input_arr)?;
        
        // Decode Output
        let mut result_sum: u16 = 0;
        for bit in 0..4 {
            let val = if outputs[bit] > 0.5 { 1 } else { 0 };
            result_sum |= (val as u16) << bit;
        }
        let cout = if outputs[4] > 0.5 { 1 } else { 0 };
        result_sum |= (cout as u16) << 4; // Cout is effectively bit 4
        
        println!("{} + {} = {} (Expected {})", a, b, result_sum, expected_sum);
        
        if result_sum != expected_sum {
             anyhow::bail!("Ripple Adder Failed: {} + {} = {}, expected {}", a, b, result_sum, expected_sum);
        }
    }
    
    println!("Ripple Carry Adder Passed!");
    Ok(())
}
