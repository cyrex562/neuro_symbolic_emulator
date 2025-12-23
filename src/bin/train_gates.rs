use ndarray::{Array1, Array2};
use rand::Rng;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use neuro_symbolic_emulator::gate::{Activation, NeuralGate};

fn main() -> anyhow::Result<()> {
    let gates_to_train = vec![
        ("AND", vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ]),
        ("OR", vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 1.0),
        ]),
        ("XOR", vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ]),
        ("NAND", vec![
            (vec![0.0, 0.0], 1.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ]),
        // NOT is 1 input -> 1 output. We can model it as 2 inputs where the second is ignored or just handle variable input size.
        // For simplicity in the library, let's treat NOT as having 2 inputs where the second is 0, or trained to ignore separate input?
        // Better: Train NOT as 1 input.
    ];
    
    // Train NOT separately or handle variable inputs
    let not_data = vec![
        (vec![0.0], 1.0),
        (vec![1.0], 0.0),
    ];

    let mut trained_gates = HashMap::new();

    // Train 2-input gates
    for (name, data) in gates_to_train {
        println!("Training {}...", name);
        let gate = train_gate(2, data);
        trained_gates.insert(name.to_string(), gate);
    }

    // Train NOT
    println!("Training NOT...");
    let gate_not = train_gate(1, not_data);
    trained_gates.insert("NOT".to_string(), gate_not);

    // Save to file
    let json = serde_json::to_string_pretty(&trained_gates)?;
    let mut file = File::create("gate_weights.json")?;
    file.write_all(json.as_bytes())?;
    
    println!("Saved gate weights to gate_weights.json");
    Ok(())
}

fn train_gate(input_size: usize, data: Vec<(Vec<f32>, f32)>) -> NeuralGate {
    let hidden_size = 4;
    let mut rng = rand::thread_rng();

    // Initialize random gate
    let mut best_gate = NeuralGate::new(
        random_array(hidden_size, input_size),
        random_array_1d(hidden_size),
        random_array(1, hidden_size),
        random_array_1d(1),
        Activation::ReLU,
        Activation::Sigmoid,
    );
    
    let mut best_loss = evaluate(&best_gate, &data);

    // Simple Hill Climbing / Evolution
    // 10,000 iterations
    for _ in 0..20000 {
        if best_loss < 0.001 {
            break;
        }

        // Mutate
        let mut candidate = best_gate.clone();
        mutate(&mut candidate, 0.5); // high mutation rate

        let loss = evaluate(&candidate, &data);
        if loss < best_loss {
            best_gate = candidate;
            best_loss = loss;
        }
    }
    
    // Verify
    println!("  Final validations:");
    for (inputs, target) in &data {
        let input_arr = Array1::from(inputs.clone());
        let out = best_gate.forward(&input_arr)[0];
        let bit = if out > 0.5 { 1.0 } else { 0.0 };
        println!("    {:?} -> {:.4} (bit {}) (target {})", inputs, out, bit, target);
        assert_eq!(bit, *target, "Failed to learn gate!");
    }

    best_gate
}

fn evaluate(gate: &NeuralGate, data: &Vec<(Vec<f32>, f32)>) -> f32 {
    let mut mse = 0.0;
    for (inputs, target) in data {
        let input_arr = Array1::from(inputs.clone());
        let out = gate.forward(&input_arr)[0];
        mse += (out - target).powi(2);
    }
    mse
}

fn mutate(gate: &mut NeuralGate, scale: f32) {
    let mut rng = rand::thread_rng();
    // Mutate all small amount
    for v in gate.w1.iter_mut() {
         *v += rng.gen_range(-scale..scale);
    }
    for v in gate.b1.iter_mut() {
         *v += rng.gen_range(-scale..scale);
    }
    for v in gate.w2.iter_mut() {
         *v += rng.gen_range(-scale..scale);
    }
    for v in gate.b2.iter_mut() {
         *v += rng.gen_range(-scale..scale);
    }
}

fn random_array(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0))
}

fn random_array_1d(len: usize) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(len, |_| rng.gen_range(-1.0..1.0))
}
