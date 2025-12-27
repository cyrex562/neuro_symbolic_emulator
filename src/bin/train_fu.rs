use ndarray::{Array1, Array2};
use rand::Rng;
use serde_json;
use neuro_symbolic_emulator::fu::{BaseFU, Activation, NeuralFunctionalUnit};
use std::fs::File;
use std::io::Write;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // In a real CLI, we'd use clap. For now, train all or uncomment.
    // train_adder()?; 
    train_comparator()?;
    // train_bitwise()?;
    Ok(())
}

fn train_comparator() -> anyhow::Result<()> {
    println!("Training 8-bit Comparator FU...");
    
    let input_size = 16; // 8 A + 8 B
    let output_size = 3; // GT, EQ, LT
    let hidden_size = 32;
    
    let mut rng = rand::thread_rng();
    
    let mut best_fu = BaseFU::new(
        random_array(hidden_size, input_size),
        random_array_1d(hidden_size),
        random_array(output_size, hidden_size),
        random_array_1d(output_size),
        Activation::ReLU,
        Activation::Sigmoid,
    );
    
    // Config
    let iterations = 10_000;
    let batch_size = 50;
    let lr = 0.05;
    
    let val_set = generate_cmp_batch(100, &mut rng);
    
    for i in 0..iterations {
        let batch = generate_cmp_batch(batch_size, &mut rng);
        for (input, target) in batch {
             best_fu.train_step(&input, &target, lr);
        }
        
        if i % 1000 == 0 {
             let cur_loss = evaluate(&mut best_fu, &val_set);
             println!("Iter {}: Val Loss = {:.4}", i, cur_loss);
        }
    }

    save_fu("FU_CMP", best_fu)?;
    Ok(())
}

fn generate_cmp_batch<R: Rng>(size: usize, rng: &mut R) -> Vec<(Array1<f32>, Array1<f32>)> {
    let mut batch = Vec::with_capacity(size);
    for _ in 0..size {
        let a_int: u8 = rng.gen();
        let b_int: u8 = rng.gen();
        
        let mut input = Array1::zeros(16);
        for i in 0..8 {
            if (a_int >> i) & 1 == 1 { input[i] = 1.0; }
            if (b_int >> i) & 1 == 1 { input[i+8] = 1.0; }
        }
        
        let mut output = Array1::zeros(3);
        if a_int > b_int { output[0] = 1.0; }
        else if a_int == b_int { output[1] = 1.0; }
        else { output[2] = 1.0; }
        
        batch.push((input, output));
    }
    batch
}

// Helper for saving
fn save_fu(name: &str, fu: BaseFU) -> anyhow::Result<()> {
    let mut library = HashMap::new();
    library.insert(name.to_string(), fu);
    // In reality we should append or merge, but overwrite for now is okay for single unit training
    let json = serde_json::to_string_pretty(&library)?;
    let filename = format!("{}_weights.json", name.to_lowercase());
    let mut file = File::create(filename)?;
    file.write_all(json.as_bytes())?;
    println!("Saved to {}_weights.json", name.to_lowercase());
    Ok(())
}

fn evaluate(fu: &mut BaseFU, batch: &Vec<(Array1<f32>, Array1<f32>)>) -> f32 {
    let mut error = 0.0;
    for (input, target) in batch {
        let preds = fu.forward(input);
        for i in 0..preds.len() {
             error += (preds[i] - target[i]).powi(2);
        }
    }
    error / batch.len() as f32
}

fn random_array(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.5..0.5))
}

fn random_array_1d(len: usize) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(len, |_| rng.gen_range(-0.5..0.5))
}
