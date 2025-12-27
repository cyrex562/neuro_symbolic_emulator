mod fu;
mod bus;
mod register;
mod voter;
mod system;
mod gui;
mod loader;

// use bus::{MoveOp, SystemBus}; // Removed references to CLI-only run
// use system::SystemEmulator;
// use register::NeuralRegister;

fn main() -> Result<(), eframe::Error> {
    // env_logger::init(); 

    let options = eframe::NativeOptions ::default();
    
    eframe::run_native(
        "NTSE System Control Dashboard",
        options,
        Box::new(|cc| Box::new(gui::NtseApp::new(cc))),
    )
}
