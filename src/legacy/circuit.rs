use crate::legacy::gate::NeuralGate;
use anyhow::{anyhow, Result};
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NeuralCircuit {
    pub gates: HashMap<usize, NeuralGate>,
    // Map: (dest_gate_id, input_index) -> (src_gate_id, output_index)
    // If src_gate_id is None, it means it comes from Circuit Input.
    pub connections: HashMap<(usize, usize), (Option<usize>, usize)>,
    pub input_size: usize,
    pub output_mapping: Vec<(usize, usize)>, // (gate_id, output_idx) for circuit outputs
    pub next_gate_id: usize,
}

impl NeuralCircuit {
    pub fn new(input_size: usize) -> Self {
        Self {
            gates: HashMap::new(),
            connections: HashMap::new(),
            input_size,
            output_mapping: Vec::new(),
            next_gate_id: 0,
        }
    }

    pub fn add_gate(&mut self, gate: NeuralGate) -> usize {
        let id = self.next_gate_id;
        self.gates.insert(id, gate);
        self.next_gate_id += 1;
        id
    }

    /// Connect a source signal to a destination gate's input.
    /// source_gate_id: None means Circuit Input.
    pub fn connect(
        &mut self,
        source_gate_id: Option<usize>,
        source_output_idx: usize,
        dest_gate_id: usize,
        dest_input_idx: usize,
    ) {
        self.connections.insert(
            (dest_gate_id, dest_input_idx),
            (source_gate_id, source_output_idx),
        );
    }

    pub fn set_output(&mut self, source_gate_id: usize, source_output_idx: usize) {
        self.output_mapping.push((source_gate_id, source_output_idx));
    }

    /// Topologically sort gates or just lazy eval if acyclic.
    /// For simplicity, we'll do a naive evaluation: compute all needed values recursively (memoized) or standard iterative if sorted.
    /// Given the small scale, a memoized recursive eval per frame is easy.
    pub fn forward(&self, circuit_inputs: &Array1<f32>) -> Result<Vec<f32>> {
        if circuit_inputs.len() != self.input_size {
            return Err(anyhow!("Input size mismatch"));
        }

        let mut gate_outputs: HashMap<usize, Array1<f32>> = HashMap::new();
        // Since it's a DAG, we can try to resolve dependencies.
        // However, a simple way is to define an evaluation order.
        // Or recursively "pull" data.
        
        // Let's implement a recursive "get_value" with memoization in `gate_outputs`.
        
        let mut results = Vec::new();
        for &(gate_id, out_idx) in &self.output_mapping {
             let val = self.resolve_gate_output(gate_id, out_idx, circuit_inputs, &mut gate_outputs)?;
             results.push(val);
        }
        
        Ok(results)
    }

    fn resolve_gate_output(
        &self,
        gate_id: usize,
        output_idx: usize,
        circuit_inputs: &Array1<f32>,
        memo: &mut HashMap<usize, Array1<f32>>,
    ) -> Result<f32> {
        if let Some(outputs) = memo.get(&gate_id) {
            return outputs.get(output_idx).copied().ok_or(anyhow!("Invalid output index for gate {}", gate_id));
        }

        let gate = self.gates.get(&gate_id).ok_or(anyhow!("Gate {} not found", gate_id))?;
        
        // Determine gate input size from w1 dims
        let n_inputs = gate.w1.shape()[1]; 
        let mut gate_input_vec = Array1::zeros(n_inputs);

        for i in 0..n_inputs {
            // Find connection
            if let Some(&(src_id_opt, src_out_idx)) = self.connections.get(&(gate_id, i)) {
                let val = match src_id_opt {
                    Some(src_id) => self.resolve_gate_output(src_id, src_out_idx, circuit_inputs, memo)?,
                    None => *circuit_inputs.get(src_out_idx).ok_or(anyhow!("Circuit input index out of bounds"))?,
                };
                gate_input_vec[i] = val;
            } else {
                // Default to 0.0 if not connected? or error?
                // For now, 0.0
                gate_input_vec[i] = 0.0;
            }
        }

        let output_vec = gate.forward(&gate_input_vec);
        memo.insert(gate_id, output_vec.clone());

        output_vec.get(output_idx).copied().ok_or(anyhow!("Output index out of bounds after compute"))
    }
}
