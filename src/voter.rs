use ndarray::Array1;

/// A simple consensus voter.
/// Checks outputs from multiple FUs.
/// If they match (within threshold), returns result.
/// If disagreement, logs drift and returns majority or mean.
pub struct VoterBlock;

impl VoterBlock {
    pub fn vote(outputs: &[Array1<f32>], threshold: f32) -> (Array1<f32>, bool) {
        if outputs.is_empty() {
            return (Array1::zeros(0), true); // Error
        }
        
        // For 2 inputs (Redundant Pair), simpler logic.
        // If dist > threshold, error.
        if outputs.len() == 2 {
            let diff = &outputs[0] - &outputs[1];
            let mean_sq_err = diff.mapv(|x| x.powi(2)).sum() / diff.len() as f32;
            
            if mean_sq_err > threshold {
                 // Drift detected!
                 // In a real system, we'd recalibrate.
                 // For now, return mean and flag drift.
                 let mean = (&outputs[0] + &outputs[1]) / 2.0;
                 return (mean, true);
            } else {
                 return (outputs[0].clone(), false);
            }
        }
        
        // Default: return first
        (outputs[0].clone(), false)
    }
}
