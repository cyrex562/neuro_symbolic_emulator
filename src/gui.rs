use eframe::egui;
use crate::system::SystemEmulator;
use crate::loader::load_manifest;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct NtseApp {
    system: Arc<Mutex<SystemEmulator>>,
    is_running: bool,
    steps_per_frame: usize,
    manifest_path: String,
    
    // Visualization State
    selected_fu_addr: Option<u16>,
    
    // Console
    console_output: Arc<Mutex<String>>,
}

impl NtseApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Init Sink
        let sink = Arc::new(Mutex::new(String::new()));
        
        // Initial Load
        let mut system = SystemEmulator::default();
        if let Ok(sys) = load_manifest(Path::new("manifest.json"), Some(sink.clone())) {
             system = sys;
        } else {
            // Default system needs the sink too
            system.console_sink = sink.clone();
            // And any default UART needs it? SystemEmulator::default() creates UART without sink.
            // We should ideally update default UART or recreate it.
            // Simplified: Just set the emulator's sink. 
            // BUT: The UartFU created in default() holds None.
            // We need to re-inject.
            // Or better: SystemEmulator::new() should take bus, and we manually add UART with sink?
            // For iteration 7, let's just accept that default() might have disconnected UART 
            // until we load a manifest properly.
        }

        Self {
            system: Arc::new(Mutex::new(system)),
            is_running: false,
            steps_per_frame: 1,
            manifest_path: "manifest.json".to_string(),
            selected_fu_addr: None,
            console_output: sink,
        }
    }
}

impl eframe::App for NtseApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Logic Step
        if self.is_running {
             let mut sys = self.system.lock().unwrap();
             for _ in 0..self.steps_per_frame {
                 if !sys.step() {
                     self.is_running = false;
                     break;
                 }
             }
             ctx.request_repaint(); // Continuous repaint when running
        }

        // 2. Top Panel: Controls
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Manifest:");
                ui.text_edit_singleline(&mut self.manifest_path);
                if ui.button("Load").clicked() {
                    if let Ok(sys) = load_manifest(Path::new(&self.manifest_path), Some(self.console_output.clone())) {
                        *self.system.lock().unwrap() = sys;
                    }
                }

                ui.separator();

                let run_btn_text = if self.is_running { "Halt" } else { "Run" };
                let run_btn = ui.button(run_btn_text);
                if run_btn.clicked() {
                    self.is_running = !self.is_running;
                }
                if self.is_running {
                     run_btn.request_focus();
                }
                
                if ui.button("Step").clicked() {
                     self.system.lock().unwrap().step();
                }
                
                 if ui.button("Reset").clicked() {
                     let mut sys = self.system.lock().unwrap();
                     sys.pc = 0;
                     sys.total_steps = 0;
                     sys.logs.clear();
                }
                
                ui.separator();
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=100).text("steps/frame"));
                
                ui.separator();
                let step_count = self.system.lock().unwrap().total_steps;
                ui.label(format!("Steps: {}", step_count));
            });
        });
        
        // 3. Bottom Panel: Logs & Console
        egui::TopBottomPanel::bottom("bottom_panel").resizable(true).min_height(150.0).show(ctx, |ui| {
             ui.horizontal(|ui| {
                 // Resizable Log Pane (Left)
                 egui::Resize::default()
                    .id_source("log_resize")
                    .default_width(400.0)
                    .with_stroke(true)
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.heading("Execution Log");
                            egui::ScrollArea::vertical().id_source("exec_log").stick_to_bottom(true).show(ui, |ui| {
                                let sys = self.system.lock().unwrap();
                                for log in &sys.logs {
                                    ui.label(egui::RichText::new(log).monospace());
                                }
                            });
                        });
                    });

                 ui.separator(); // Vertical separator

                 // specific width or fill remaining?
                 // Console Pane (Right - Fills remaining)
                 ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Console (UART)");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Clear Output").clicked() {
                                if let Ok(mut output) = self.console_output.lock() {
                                    output.clear();
                                }
                            }
                        });
                    });
                    
                    egui::ScrollArea::vertical().id_source("console_out").stick_to_bottom(true).show(ui, |ui| {
                        if let Ok(output) = self.console_output.lock() {
                            ui.label(egui::RichText::new(output.as_str()).monospace().color(egui::Color32::GREEN));
                        }
                    });
                 });
             });
        });

        // 4. Side Panel: Registers (Left)
        egui::SidePanel::left("registers").resizable(true).show(ctx, |ui| {
            ui.heading("NRF (Registers)");
            egui::ScrollArea::vertical().id_source("regs_scroll").show(ui, |ui| {
                egui::Grid::new("reg_grid").striped(true).show(ui, |ui| {
                    let sys = self.system.lock().unwrap();
                    let mut keys: Vec<&u16> = sys.bus.registers.keys().collect();
                    keys.sort();
                    
                    for k in keys {
                        ui.label(format!("R{}", k));
                        if let Some(reg) = sys.bus.registers.get(k) {
                            let val = &reg.state;
                            let v0 = val.get(0).unwrap_or(&0.0);
                            let dist = (v0 - v0.round()).abs();
                            let color = if dist < 0.1 { egui::Color32::GREEN } else { egui::Color32::YELLOW };
                            ui.colored_label(color, format!("{:.2}", v0));
                        }
                        ui.end_row();
                    }
                });
            });
        });

        // 5. Side Panel: Inspector (Right)
        egui::SidePanel::right("inspector").resizable(true).show(ctx, |ui| {
             ui.heading("Inspector");
             egui::ScrollArea::vertical().id_source("inspector_scroll").show(ui, |ui| {
                 let sys = self.system.lock().unwrap();
                 
                 // RAM Inspection (Show non-zero)
                 ui.collapsing("RAM (Non-Zero)", |ui| {
                     let mut mem_keys: Vec<&u16> = sys.bus.ram.keys().collect();
                     mem_keys.sort();
                     for k in mem_keys {
                         if let Some(val) = sys.bus.ram.get(k) {
                             let v_str = val.iter().take(4).map(|v| format!("{:.1}", v)).collect::<Vec<_>>().join(", ");
                             ui.label(format!("0x{:X}: [{}]", k, v_str));
                         }
                     }
                 });

                 ui.separator();

                 // FU Inspection (Show cached I/O)
                 ui.collapsing("Functional Unit I/O", |ui| {
                     let mut fu_keys: Vec<&u16> = sys.bus.fu_io_cache.keys().collect();
                     fu_keys.sort();
                     for k in fu_keys {
                         if let Some((input, output)) = sys.bus.fu_io_cache.get(k) {
                             ui.label(egui::RichText::new(format!("UNIT @ 0x{:X}", k)).strong());
                             
                             let in_str = input.iter().take(8).map(|v| format!("{:.0}", v)).collect::<Vec<_>>().join("");
                             let out_str = output.iter().take(8).map(|v| format!("{:.0}", v)).collect::<Vec<_>>().join("");
                             
                             ui.monospace(format!(" In: [{}]", in_str));
                             ui.monospace(format!("Out: [{}]", out_str));
                             ui.add_space(4.0);
                         }
                     }
                 });
             });
        });

        // 6. Central Panel: FUs & Program
        egui::CentralPanel::default().show(ctx, |ui| {
             ui.heading("Functional Units");
             let sys = self.system.lock().unwrap();
             
             // Simple Grid of Units
             egui::ScrollArea::vertical().id_source("fus_scroll").max_height(150.0).show(ui, |ui| {
                 ui.horizontal_wrapped(|ui| {
                     let mut u_keys: Vec<&u16> = sys.bus.units.keys().collect();
                     u_keys.sort();
                     for k in u_keys {
                         ui.group(|ui| {
                             ui.label(format!("FU @ 0x{:X}", k));
                         });
                     }
                      // MMIO too
                     let mut m_keys: Vec<&u16> = sys.bus.mmio.keys().collect();
                     m_keys.sort();
                      for k in m_keys {
                         ui.group(|ui| {
                             ui.label(format!("MMIO @ 0x{:X}", k));
                         });
                      }
                 });
             });
             
             ui.separator();
             ui.heading("Program");
             egui::ScrollArea::vertical().id_source("prog_scroll").show(ui, |ui| {
                 for (i, op) in sys.program.iter().enumerate() {
                      // Name Resolution Helper
                      let resolve = |addr: u16| -> String {
                          if addr < 16 { return format!("R{}", addr); }
                          if addr == 0x8000 { return "UART".to_string(); }
                          if addr >= 0x2000 && addr < 0x8000 { return format!("RAM[0x{:X}]", addr); }
                          if sys.bus.units.contains_key(&addr) { return format!("FU[0x{:X}]", addr); }
                          format!("0x{:X}", addr)
                      };
                     
                      let src_name = resolve(op.src);
                      let dest_name = resolve(op.dest);
                      let guard_info = if let Some(g) = op.guard {
                          format!(" [IF {}]", resolve(g))
                      } else {
                          "".to_string()
                      };
                     
                     let text = format!("{:04}: {} -> {}{}", i, src_name, dest_name, guard_info);
                     if i == sys.pc {
                         ui.label(egui::RichText::new(text).strong().background_color(egui::Color32::DARK_BLUE));
                     } else {
                         ui.label(text);
                     }
                 }
             });
        });
    }
}
