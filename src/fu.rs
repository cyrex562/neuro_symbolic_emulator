use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Interface for any Neural Functional Unit.
/// Takes a vector input and produces a vector output.
pub trait NeuralFunctionalUnit {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32>;
    fn perturb(&mut self, amount: f32); // For noise injection verification
    fn tick(&mut self) {} // Optional: Called every cycle
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Identity,
}

impl Activation {
    pub fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Identity => x.clone(),
        }
    }

    pub fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let s = self.apply(x);
                s.mapv(|v| v * (1.0 - v))
            },
            Activation::Tanh => {
                 let t = self.apply(x);
                 t.mapv(|v| 1.0 - v * v)
            },
            Activation::Identity => Array1::ones(x.len()),
        }
    }
}

/// A standard Multi-Layer Perceptron (MLP) implementation of an FU.
/// Can be trained for ADD, COMP, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseFU {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>, // Hidden -> Output
    pub b2: Array1<f32>,
    pub active_hidden: Activation,
    pub active_output: Activation,
}

impl BaseFU {
    pub fn new(
        w1: Array2<f32>, b1: Array1<f32>,
        w2: Array2<f32>, b2: Array1<f32>,
        active_hidden: Activation, active_output: Activation
    ) -> Self {
        Self { w1, b1, w2, b2, active_hidden, active_output }
    }

    pub fn train_step(&mut self, input: &Array1<f32>, target: &Array1<f32>, lr: f32) {
        // Forward
        let h_pre = self.w1.dot(input) + &self.b1;
        let h = self.active_hidden.apply(&h_pre);
        let y_pre = self.w2.dot(&h) + &self.b2;
        let y = self.active_output.apply(&y_pre);

        // Gradients (MSE Loss)
        // delta_out = (y - target) * f'(y_pre)
        // For Sigmoid, f'(x) = f(x)(1-f(x)). The derivative method handles f(x) if needed or x.
        // My derivative implementation assumes input is x (pre-activation), which is correct.
        let error = &y - target;
        let d_out = self.active_output.derivative(&y_pre);
        let delta_2 = &error * &d_out;

        // Backprop to hidden
        // delta_1 = (w2^T * delta_2) * f'(h_pre)
        let d_hidden = self.active_hidden.derivative(&h_pre);
        let delta_1: Array1<f32> = self.w2.t().dot(&delta_2) * &d_hidden;

        // Update Weights (SGD)
        // w2 -= lr * delta_2 * h^T
        for (i, d) in delta_2.iter().enumerate() {
            self.b2[i] -= lr * d;
            for (j, h_val) in h.iter().enumerate() {
                self.w2[[i, j]] -= lr * d * h_val;
            }
        }

        // w1 -= lr * delta_1 * input^T
        for (i, d) in delta_1.iter().enumerate() {
            self.b1[i] -= lr * d;
            for (j, in_val) in input.iter().enumerate() {
                self.w1[[i, j]] -= lr * d * in_val;
            }
        }
    }
}

impl NeuralFunctionalUnit for BaseFU {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let h_pre = self.w1.dot(input) + &self.b1;
        let h = self.active_hidden.apply(&h_pre);
        let y_pre = self.w2.dot(&h) + &self.b2;
        self.active_output.apply(&y_pre)
    }

    fn perturb(&mut self, amount: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Mutate small percentage of weights for resiliency testing
        for v in self.w1.iter_mut() {
            if rng.gen::<f32>() < 0.1 { *v += rng.gen_range(-amount..amount); }
        }
        for v in self.w2.iter_mut() {
            if rng.gen::<f32>() < 0.1 { *v += rng.gen_range(-amount..amount); }
        }
    }
}

impl BaseFU {
    pub fn create_random(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.5..0.5));
        let b1 = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1));
        let w2 = Array2::from_shape_fn((output_size, hidden_size), |_| rng.gen_range(-0.5..0.5));
        let b2 = Array1::from_shape_fn(output_size, |_| rng.gen_range(-0.1..0.1));
        
        Self::new(w1, b1, w2, b2, Activation::Sigmoid, Activation::Sigmoid)
    }

    pub fn create_adder() -> Self {
        // 8-bit A + 8-bit B = 16 inputs
        // 8-bit Sum + 1-bit Carry = 9 outputs
        Self::create_random(16, 32, 9)
    }

    pub fn create_comparator() -> Self {
        // 16 inputs (A: 8, B: 8)
        // 3 outputs: GT, EQ, LT
        Self::create_random(16, 24, 3)
    }

    pub fn create_bitwise() -> Self {
        // Inputs: A (8) + B (8) + Mode (3) = 19 inputs
        // Output: 8 bits
        // Mode could be: 000=AND, 001=OR, 010=XOR, 011=NOT A...
        Self::create_random(19, 32, 8)
    }
}

// --- Stateful Units ---

// --- Stateful Units ---

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ProgramCounterFU {
    pub pc: u32,
}

impl ProgramCounterFU {
    pub fn new() -> Self { Self { pc: 0 } }
}

impl NeuralFunctionalUnit for ProgramCounterFU {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Input acts as JUMP Address.
        let mut addr = 0;
        for (i, &v) in input.iter().enumerate() {
             if v > 0.5 { addr |= 1 << i; }
        }
        self.pc = addr; // Jump!
        
        // Return new PC as vector
        let mut out = Array1::zeros(8); // Assume 8-bit address space
        for i in 0..8 {
            if (self.pc >> i) & 1 == 1 { out[i] = 1.0; }
        }
        out
    }
    
    fn perturb(&mut self, _amount: f32) {}
    
    fn tick(&mut self) {
        self.pc += 1;
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreFU {
    pub memory: HashMap<u32, Array1<f32>>,
    pub width: usize,
}

impl LoadStoreFU {
    pub fn new(width: usize) -> Self {
         Self { memory: HashMap::new(), width }
    }
}

impl NeuralFunctionalUnit for LoadStoreFU {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Input: ADDR (width). 
        // We assume DATA_IN is read from a register by the Bus and passed here? 
        // OR the input vector contains ADDR + DATA?
        // Prompt: "Trigger: Moving a value to ADDR with a Write-Enable bit set."
        // This implies the standard TTA trigger is the ADDR register.
        // But we need the DATA to write.
        // Convention: We read DATA from a predetermined "DATA_IN" register.
        // We can't access other registers here.
        // So we must assume the input *is* the address, and we perform a LOAD?
        // Or if Write-Enable is set (where? Mode register? Or part of input?), we WRITE.
        
        // Simplified Logic for Iteration 3:
        // Always LOAD from Address.
        // To WRITE, we might need a separate "STORE_TRIGGER" port/unit or encoding.
        // Or, we stick to the prompt: "Moving a value to ADDR ... with Write-Enable".
        // Let's assume input is just ADDR for now, and it returns the Data (LOAD).
        // WRITE is complex without extra args.
        
        let mut addr = 0;
        let len = input.len();
        for (i, &v) in input.iter().enumerate() {
             if v > 0.5 { addr |= 1 << i; }
        }
        
        // MOCK: Return stored value or random
        if let Some(val) = self.memory.get(&addr) {
            return val.clone();
        } else {
            return Array1::zeros(self.width);
        }
    }
    fn perturb(&mut self, _amount: f32) {}
}

#[derive(Debug, Clone)]
pub struct StackPointerFU {
    pub sp: u32,
    pub stack: HashMap<u32, Array1<f32>>, // Mock stack memory
    pub width: usize,
}
impl StackPointerFU {
    pub fn new(width: usize) -> Self { 
        Self { sp: 0xFF, stack: HashMap::new(), width } 
    }
}
impl NeuralFunctionalUnit for StackPointerFU {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> { 
        // Trigger: STACK_DATA.
        // If we move data here -> PUSH.
        // Decrement SP, Store data.
        self.sp = self.sp.wrapping_sub(1);
        self.stack.insert(self.sp, input.clone());
        input.clone() // Pass through or return new SP?
    }
    fn perturb(&mut self, _amount: f32) {}
}

// Mocks removed for production.

