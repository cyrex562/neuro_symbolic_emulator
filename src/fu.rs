use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Interface for any Neural Functional Unit.
/// Takes a vector input and produces a vector output.
pub trait NeuralFunctionalUnit {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32>;
    fn perturb(&mut self, amount: f32); // For noise injection verification
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Identity,
}

impl Activation {
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
        let delta_1 = self.w2.t().dot(&delta_2) * &d_hidden;

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
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
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
