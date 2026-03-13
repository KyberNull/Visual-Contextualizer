// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod llama;

use crate::llama::inference::{ContextConfig, GenerationConfig, LlamaPipeline, LlamaRuntime};
use crate::llama::model::LlamaModel;
use std::fs;
use std::sync::Mutex;
use tauri::State;

// TODO: Make it dynamic later, maybe with a file picker in the frontend. For now we can hardcode it for testing.
const MODEL_PATH: &str = "/home/aasish/Visual Contextualizer/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf";

#[tauri::command]
fn get_img(data: Vec<u8>) -> Result<String, String> {
    let path = "dot.png";

    match fs::write(path, data) {
        Ok(_) => {
            println!("Successfully saved image");
            Ok(format!("Successfully saved images"))
        }
        Err(e) => {
            eprintln!("The error gotten is {}", e);
            Err(format!("The error gotten is {}", e))
        }
    }
}

struct AppState {
    runtime: LlamaRuntime, // owns llama ackend, lives for app lifetime
    pipeline: Mutex<LlamaPipeline>,
}

impl AppState {
    fn new() -> Result<Self, String> {
        let runtime = LlamaRuntime::init();

        println!("Loading model from {}", MODEL_PATH);

        let model = LlamaModel::load(MODEL_PATH)?; // Lazy loads because mmap() is on by default in llama.cpp
        let cfg = ContextConfig::default();
        let pipeline = LlamaPipeline::from_model(model, &cfg)?;

        println!("Model loaded successfully");

        Ok(Self {
            runtime,
            pipeline: Mutex::new(pipeline),
        })
    }
}

#[tauri::command]
fn generate_text(state: State<AppState>, prompt: String) -> Result<String, String> {
    let mut pipeline = state
        .pipeline
        .lock()
        .map_err(|_| "Failed to lock pipeline".to_string())?;

    let cfg = GenerationConfig::default();

    println!("Generating text for prompt: {}", prompt);

    pipeline.generate(&prompt, &cfg)
}

// Initialise App Stae which loads llama.cpp backend and model, and creates the pipeline.
// App State exists throughout the entire app lifetime (automatically manages memory).
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new().unwrap())// TODO: Handle errors properly later.
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            generate_text,
            get_img,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
