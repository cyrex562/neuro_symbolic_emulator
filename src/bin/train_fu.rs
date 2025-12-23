use ndarray::{Array1, Array2};
use rand::Rng;
use serde_json;
use neuro_symbolic_emulator::fu::{BaseFU, Activation, NeuralFunctionalUnit};
use std::fs::File;
use std::io::Write;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    train_adder()
}

fn train_adder() -> anyhow::Result<()> {
    println!("Training 8-bit Adder FU...");
    
    // Config
    let input_size = 16; // 8 bits A + 8 bits B
    let output_size = 9; // 8 bits Sum + 1 bit Carry
    let hidden_size = 32; // Need enough capacity
    
    // Dataset: Generate random pairs
    // We can't train on all 65536 + combinations.
    // Train on random batches.
    
    let mut rng = rand::thread_rng();
    
    // Initialize
    let mut best_fu = BaseFU::new(
        random_array(hidden_size, input_size),
        random_array_1d(hidden_size),
        random_array(output_size, hidden_size),
        random_array_1d(output_size),
        Activation::ReLU,
        Activation::Sigmoid,
    );
    
    let iterations = 20_000;
    let batch_size = 50;
    let lr = 0.01;
    
    // Test Initial Loss
    let val_set = generate_batch(100, &mut rng);
    let init_loss = evaluate(&best_fu, &val_set);
    println!("Initial Loss: {:.4}", init_loss);
    
    for i in 0..iterations {
        // Generate a random sample
        // For SGD, we can just do online learning or mini-batch.
        // Let's do mini-batch.
        let batch = generate_batch(batch_size, &mut rng);
        
        for (input, target) in batch {
             best_fu.train_step(&input, &target, lr);
        }
        
        if i % 1000 == 0 {
             let cur_loss = evaluate(&best_fu, &val_set);
             println!("Iter {}: Val Loss = {:.4}", i, cur_loss);
             if cur_loss < 0.005 {
                 println!("Converged early!");
                 break;
             }
        }
    }
    
    // Final Validation on hardcoded set
    let final_loss = evaluate(&best_fu, &val_set);
    println!("Final Validation Loss: {:.4}", final_loss);
    
    // Validate accuracy (bit errors)
    let mut bit_errors = 0;
    let total_bits = val_set.len() * 9;
    for (input, target) in &val_set {
         let pred = best_fu.forward(input);
         for j in 0..9 {
             let bit_p = if pred[j] > 0.5 { 1.0 } else { 0.0 };
             let bit_t = if target[j] > 0.5 { 1.0 } else { 0.0 };
             if bit_p != bit_t { bit_errors += 1; }
         }
    }
    println!("Bit Accuracy: {:.2}% ({}/{} errors)", 100.0 * (1.0 - bit_errors as f32 / total_bits as f32), bit_errors, total_bits);
    
    // Save
    let mut library = HashMap::new();
    library.insert("FU_ADD".to_string(), best_fu);
    
    let json = serde_json::to_string_pretty(&library)?;
    let mut file = File::create("fu_weights.json")?;
    file.write_all(json.as_bytes())?;
    
    println!("Saved to fu_weights.json");
    Ok(())
}

fn generate_batch<R: Rng>(size: usize, rng: &mut R) -> Vec<(Array1<f32>, Array1<f32>)> {
    let mut batch = Vec::with_capacity(size);
    for _ in 0..size {
        let a_int: u8 = rng.gen();
        let b_int: u8 = rng.gen();
        let sum_int = (a_int as u16) + (b_int as u16);
        
        // Input Vector
        let mut input = Array1::zeros(16);
        for i in 0..8 {
            if (a_int >> i) & 1 == 1 { input[i] = 1.0; }
            else { input[i] = 0.0; } // Explicit 0.0
            
            if (b_int >> i) & 1 == 1 { input[i+8] = 1.0; }
            else { input[i+8] = 0.0; }
        }
        
        // Output Vector (Sum 0-7, Carry 8)
        let mut output = Array1::zeros(9);
        for i in 0..8 {
             if (sum_int >> i) & 1 == 1 { output[i] = 1.0; }
             else { output[i] = 0.0; }
        }
        if (sum_int >> 8) & 1 == 1 { output[8] = 1.0; } // Carry Out
        else { output[8] = 0.0; }
        
        batch.push((input, output));
    }
    batch
}

fn evaluate(fu: &BaseFU, batch: &Vec<(Array1<f32>, Array1<f32>)>) -> f32 {
    let mut error = 0.0;
    for (input, target) in batch {
        let preds = fu.forward(input);
        // MSE
        for i in 0..preds.len() {
            error += (preds[i] - target[i]).powi(2);
        }
    }
    error / batch.len() as f32
}

// Removed mutate function


fn random_array(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.5..0.5))
}

fn random_array_1d(len: usize) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(len, |_| rng.gen_range(-0.5..0.5))
}
