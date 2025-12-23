use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::Path;
use neuro_symbolic_emulator::fu::{BaseFU, NeuralFunctionalUnit, ProgramCounterFU};
use ndarray::{Array1, Array2};

#[derive(Parser)]
#[command(name = "manage_fus")]
#[command(about = "Manage Neural Functional Units", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a single FU
    Train {
        name: String,
        #[arg(value_enum)]
        type_: FUType,
    },
    /// Batch train FUs from a manifest
    BatchTrain {
        manifest: String,
    },
    /// Verify a trained FU
    Verify {
        name: String,
    },
    /// List all trained FUs
    List,
}

#[derive(clap::ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq)]
enum FUType {
    ADDER,
    CMP,
    BITWISE,
    PC,
}

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    units: Vec<UnitConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
struct UnitConfig {
    name: String,
    #[serde(rename = "type")]
    type_: FUType,
    description: Option<String>,
    conf: serde_json::Value,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let assets_dir = Path::new("assets/fus");
    std::fs::create_dir_all(assets_dir)?;

    match cli.command {
        Commands::Train { name, type_ } => {
            println!("Training {} ({:?})...", name, type_);
            train_fu(&name, type_, assets_dir)?;
        }
        Commands::BatchTrain { manifest } => {
            println!("Batch training from manifest: {}", manifest);
            let content = fs::read_to_string(manifest)?;
            let manifest: Manifest = serde_json::from_str(&content)?;
            
            for unit in manifest.units {
                println!("Processing {}...", unit.name);
                train_fu(&unit.name, unit.type_, assets_dir)?;
            }
        }
        Commands::Verify { name } => {
            println!("Verifying {}...", name);
            verify_fu(&name, assets_dir)?;
        }
        Commands::List => {
            println!("Listing trained FUs:");
            for entry in fs::read_dir(assets_dir)? {
                let entry = entry?;
                println!(" - {}", entry.file_name().to_string_lossy());
            }
        }
    }

    Ok(())
}

fn train_fu(name: &str, type_: FUType, out_dir: &Path) -> anyhow::Result<()> {
    // This is where we will delegate to specific training functions
    // For now, we stub it out or reuse existing logic from fu.rs if available
    
    // We need to support saving the resulting FU.
    // The BaseFU is serializable.
    
    let fu_file = out_dir.join(format!("{}.json", name));
    
    // Check if exists first for load
    if !fu_file.exists() && matches!(type_, FUType::PC) {
         println!("PC unit does not require training.");
         return Ok(());
    }

    match type_ {
        FUType::ADDER => {
            let fu = BaseFU::create_adder();
            let trained_fu = train_adder(fu);
            save_fu(&trained_fu, &fu_file)?;
        },
        FUType::CMP => {
            let fu = BaseFU::create_comparator();
            let trained_fu = train_comparator(fu);
            save_fu(&trained_fu, &fu_file)?;
        },
        FUType::BITWISE => {
             let fu = BaseFU::create_bitwise();
             let trained_fu = train_bitwise(fu);
             save_fu(&trained_fu, &fu_file)?;
        },
        FUType::PC => {
            println!("PC Unit is structural, no training needed.");
        }
    }
    
    Ok(())
}

fn save_fu<T: Serialize>(fu: &T, path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, fu)?;
    println!("Saved to {:?}", path);
    Ok(())
}

fn load_fu_base(path: &Path) -> anyhow::Result<BaseFU> {
    let file = File::open(path)?;
    let fu = serde_json::from_reader(file)?;
    Ok(fu)
}

fn verify_fu(name: &str, out_dir: &Path) -> anyhow::Result<()> {
    // Determine type from name or manifest? 
    // The CLI verify command only takes name. We might need to look up type from manifest or infer or try all.
    // For simplicity, let's just try to load as BaseFU and run a generic check or specific check based on name conventions.
    
    let path = out_dir.join(format!("{}.json", name));
    if !path.exists() {
        println!("FU {} not found at {:?}", name, path);
        return Ok(());
    }
    
    // We assume BaseFU for trained units
    let mut fu = load_fu_base(&path)?;
    
    // Choose generator based on name
    let generator: Option<Box<dyn Fn() -> (Array1<f32>, Array1<f32>)>> = if name.contains("adder") {
        Some(Box::new(|| {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let a = rng.gen::<u8>();
            let b = rng.gen::<u8>();
            let sum = (a as u16) + (b as u16);
            let mut input = u8_to_vec(a);
            input.extend(u8_to_vec(b));
            let mut target = u8_to_vec((sum & 0xFF) as u8);
            target.push(if sum > 0xFF { 1.0 } else { 0.0 });
            (Array1::from(input), Array1::from(target))
        }))
    } else if name.contains("compare") || name.contains("cmp") {
        Some(Box::new(|| {
             use rand::Rng;
            let mut rng = rand::thread_rng();
            let a = rng.gen::<u8>();
            let b = rng.gen::<u8>();
            let mut input = u8_to_vec(a);
            input.extend(u8_to_vec(b));
             let target = if a > b { vec![1.0, 0.0, 0.0] } 
                     else if a == b { vec![0.0, 1.0, 0.0] } 
                     else { vec![0.0, 0.0, 1.0] };
            (Array1::from(input), Array1::from(target))
        }))
    } else if name.contains("bitwise") {
        Some(Box::new(|| {
             use rand::Rng;
            let mut rng = rand::thread_rng();
            let a = rng.gen::<u8>();
            let b = rng.gen::<u8>();
            let mode = rng.gen_range(0..3); 
            let mut input = u8_to_vec(a);
            input.extend(u8_to_vec(b));
             match mode {
                0 => input.extend(vec![1.0, 0.0, 0.0]),
                1 => input.extend(vec![0.0, 1.0, 0.0]),
                _ => input.extend(vec![0.0, 0.0, 1.0]),
            }
            let res = match mode {
                0 => a & b,
                1 => a | b,
                _ => a ^ b,
            };
            (Array1::from(input), Array1::from(u8_to_vec(res)))
        }))
    } else {
        None
    };

    if let Some(gen) = generator {
        let mut errors = 0;
        let samples = 1000;
        for _ in 0..samples {
            let (input, target) = gen();
            let output = fu.forward(&input);
            let out_bits: Vec<u8> = output.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
            let target_bits: Vec<u8> = target.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
            if out_bits != target_bits {
                errors += 1;
            }
        }
        println!("Verification for {}: {} errors / {} samples", name, errors, samples);
    } else {
        println!("Unknown FU type for verification: {}", name);
    }

    Ok(())
}


// --- Training Logic ---

const LEARNING_RATE: f32 = 0.1;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 100;

fn train_loop(
    mut fu: BaseFU, 
    data_generator: impl Fn() -> (Array1<f32>, Array1<f32>),
    name: &str
) -> BaseFU {
    let mut rng = rand::thread_rng();
    
    for epoch in 0..EPOCHS {
        let mut total_error = 0.0;
        
        for _ in 0..BATCH_SIZE {
            let (input, target) = data_generator();
            fu.train_step(&input, &target, LEARNING_RATE);
            
            // Simple loss tracking
            let output = fu.forward(&input);
            for (i, t) in target.iter().enumerate() {
                total_error += (output[i] - t).powi(2);
            }
        }
        
        if epoch % 1000 == 0 {
            println!("  [{}] Epoch {}: Loss = {:.4}", name, epoch, total_error / BATCH_SIZE as f32);
        }
    }
    
    // Verify
    let mut errors = 0;
    println!("  [{}] Verifying...", name);
    for _ in 0..100 {
        let (input, target) = data_generator();
        let output = fu.forward(&input);
        
        // Threshold check
        let out_bits: Vec<u8> = output.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
        let target_bits: Vec<u8> = target.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
        
        if out_bits != target_bits {
            errors += 1;
        }
    }
    println!("  [{}] Verification Errors (Sample 100): {}", name, errors);
    
    fu
}

// --- Data Generators ---

fn u8_to_vec(val: u8) -> Vec<f32> {
    (0..8).map(|i| if (val >> i) & 1 == 1 { 1.0 } else { 0.0 }).collect()
}



fn train_adder(fu: BaseFU) -> BaseFU {
    use rand::Rng;
    train_loop(fu, || {
        let mut rng = rand::thread_rng();
        let a = rng.gen::<u8>();
        let b = rng.gen::<u8>();
        let sum = (a as u16) + (b as u16);
        
        let mut input = u8_to_vec(a);
        input.extend(u8_to_vec(b));
        
        let mut target = u8_to_vec((sum & 0xFF) as u8); // Sum low byte
        target.push(if sum > 0xFF { 1.0 } else { 0.0 }); // Carry bit
        
        (Array1::from(input), Array1::from(target))
    }, "Adder")
}

fn train_comparator(fu: BaseFU) -> BaseFU {
    use rand::Rng;
    train_loop(fu, || {
        let mut rng = rand::thread_rng();
        let a = rng.gen::<u8>();
        let b = rng.gen::<u8>();
        
        let mut input = u8_to_vec(a);
        input.extend(u8_to_vec(b));
        
        let target = if a > b { vec![1.0, 0.0, 0.0] } // GT
                     else if a == b { vec![0.0, 1.0, 0.0] } // EQ
                     else { vec![0.0, 0.0, 1.0] }; // LT
                     
        (Array1::from(input), Array1::from(target))
    }, "Comparator")
}

fn train_bitwise(fu: BaseFU) -> BaseFU {
    use rand::Rng;
    train_loop(fu, || {
        let mut rng = rand::thread_rng();
        let a = rng.gen::<u8>();
        let b = rng.gen::<u8>();
        let mode = rng.gen_range(0..3); // 0=AND, 1=OR, 2=XOR
        
        let mut input = u8_to_vec(a);
        input.extend(u8_to_vec(b));
        // Mode bits (3 bits)
        match mode {
            // let's use 1-hot for mode
            0 => input.extend(vec![1.0, 0.0, 0.0]),
            1 => input.extend(vec![0.0, 1.0, 0.0]),
            _ => input.extend(vec![0.0, 0.0, 1.0]),
        }
        
        let res = match mode {
            0 => a & b,
            1 => a | b,
            _ => a ^ b,
        };
        
        (Array1::from(input), Array1::from(u8_to_vec(res)))
    }, "Bitwise")
}

