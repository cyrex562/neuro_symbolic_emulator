use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct NeuralRegister {
    pub state: Array1<f32>,
    pub width: usize,
}

impl NeuralRegister {
    pub fn new(width: usize) -> Self {
        Self {
            state: Array1::zeros(width),
            width,
        }
    }

    pub fn write(&mut self, value: &Array1<f32>) {
        if value.len() == self.width {
            self.state = value.clone();
            // TODO: Apply cleanup/autoencoder here
        } else {
            // Log error or panic in debug?
            eprintln!("Warning: Register write size mismatch");
        }
    }

    pub fn read(&self) -> Array1<f32> {
        self.state.clone()
    }
    
    // Convert to symbolic integer (0-255) for I/O
    pub fn to_symbolic(&self) -> u32 {
        let mut result = 0;
        for (i, &v) in self.state.iter().enumerate() {
            if v > 0.5 {
                 result |= 1 << i;
            }
        }
        result
    }
    
    // Load from symbolic integer
    pub fn from_symbolic(width: usize, val: u32) -> Self {
        let mut reg = Self::new(width);
        let mut vec = Array1::zeros(width);
        for i in 0..width {
            // Mapping: 0.0 for 0, 1.0 for 1
            if (val >> i) & 1 == 1 {
                vec[i] = 1.0;
            } else {
                vec[i] = 0.0;
            }
        }
        reg.write(&vec);
        reg
    }

    /// "Cleans" the noisy neural state back to binary 0.0/1.0
    /// This simulates the Autoencoder/Hopfield cleanup step.
    pub fn cleanup(&mut self) {
        self.state.mapv_inplace(|v| if v > 0.5 { 1.0 } else { 0.0 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_symbolic_roundtrip(val in 0u32..256u32) {
            let width = 8;
            let reg = NeuralRegister::from_symbolic(width, val);
            let out = reg.to_symbolic();
            assert_eq!(val, out);
        }
        
        #[test]
        fn test_cleanup_property(fuzz in proptest::collection::vec(0.0f32..1.0f32, 8)) {
            let mut reg = NeuralRegister::new(8);
            reg.write(&Array1::from(fuzz.clone()));
            
            reg.cleanup();
            
            for (i, &v) in reg.state.iter().enumerate() {
                assert!(v == 0.0 || v == 1.0);
                
                // Assert it snapped to correct side
                let original = fuzz[i];
                if original > 0.5 {
                    assert_eq!(v, 1.0);
                } else {
                    assert_eq!(v, 0.0);
                }
            }
        }
    }
}
