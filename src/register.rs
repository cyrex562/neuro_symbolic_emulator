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
            if (val >> i) & 1 == 1 {
                vec[i] = 1.0;
            } else {
                vec[i] = 0.0;
            }
        }
        reg.write(&vec);
        reg
    }
}
