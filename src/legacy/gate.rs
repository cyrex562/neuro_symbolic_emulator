use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Step, // Threshold at 0.5 (or custom)
    Identity,
}

impl Activation {
    pub fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Step => x.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 }),
            Activation::Identity => x.clone(),
        }
    }
}

/// The Atomic Neural Gate (ANG).
/// Structure: Input -> Hidden Layer -> Output Node.
/// Can model non-linear functions (XOR).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGate {
    pub w1: Array2<f32>, // Input -> Hidden
    pub b1: Array1<f32>, // Hidden Bias
    pub w2: Array2<f32>, // Hidden -> Output
    pub b2: Array1<f32>, // Output Bias
    pub activation_hidden: Activation,
    pub activation_output: Activation,
}

impl NeuralGate {
    pub fn new(
        w1: Array2<f32>,
        b1: Array1<f32>,
        w2: Array2<f32>,
        b2: Array1<f32>,
        activation_hidden: Activation,
        activation_output: Activation,
    ) -> Self {
        Self {
            w1,
            b1,
            w2,
            b2,
            activation_hidden,
            activation_output,
        }
    }

    /// Forward pass:
    /// h = ActivationHidden(W1 . x + b1)
    /// y = ActivationOutput(W2 . h + b2)
    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let h_pre = self.w1.dot(inputs) + &self.b1;
        let h = self.activation_hidden.apply(&h_pre);
        
        let y_pre = self.w2.dot(&h) + &self.b2;
        self.activation_output.apply(&y_pre)
    }
}
