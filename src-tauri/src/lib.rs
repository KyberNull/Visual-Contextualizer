// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod llama;

use crate::llama::inference::{resolve_dependency_path, ContextConfig, GenerationConfig, LlamaPipeline, LlamaRuntime};
use crate::llama::model::LlamaModel;
use serde::Serialize;
use piper_rs::synth::PiperSpeechSynthesizer;
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
use std::fs;
use std::num::NonZeroU16;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::Emitter;
use tauri::Listener;
use tauri::{AppHandle, Manager, State};

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
    _runtime: LlamaRuntime, // owns llama ackend, lives for app lifetime
    pipeline: Mutex<LlamaPipeline>,

    synth: Arc<PiperSpeechSynthesizer>,
    audio_player: Arc<Mutex<Player>>,
    _mixer_sink: rodio::MixerDeviceSink,
}

impl AppState {
    fn new() -> Result<Self, String> {
        let _runtime = LlamaRuntime::init();

        let path =
            resolve_dependency_path(Path::new("Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))?;
        let path_str = path.to_str().ok_or("Model path is not valid UTF-8")?;

        println!("Loading model from {}", path.display());

        let model = LlamaModel::load(path_str)?; // Lazy loads because mmap() is on by default in llama.cpp
        let cfg = ContextConfig::default();
        let pipeline = LlamaPipeline::from_model(model, &cfg)?;

        println!("Model loaded successfully");

        let mixer_sink = DeviceSinkBuilder::open_default_sink()
            .map_err(|e| format!("Audio Device Error: {}", e))?;
        let player = Player::connect_new(mixer_sink.mixer());

        let config_path = resolve_dependency_path(Path::new("en_US-libritts_r-medium.onnx.json"))?;
        let piper_model = piper_rs::from_config_path(&config_path).map_err(|e| e.to_string())?;

        let synth = PiperSpeechSynthesizer::new(piper_model).map_err(|e| e.to_string())?;

        Ok(Self {
            _runtime,
            pipeline: Mutex::new(pipeline),
            synth: Arc::new(synth),
            audio_player: Arc::new(Mutex::new(player)),
            _mixer_sink: mixer_sink,
        })
    }
}

#[derive(Clone, Serialize)]
struct WordPayload {
    word: String,
}

pub fn setup_voice_engine(app: &AppHandle) {
    let handle = app.clone();
    let word_queue = Arc::new(Mutex::new(Vec::<String>::new()));

    app.listen("got_a_word", move |event| {
        let word: String = serde_json::from_str(event.payload()).unwrap_or_default();

        let mut queue = word_queue.lock().unwrap();
        queue.push(word.clone());

        let is_end = word.contains('.') || word.contains('?') || word.contains('!');

        if queue.len() >= 5 || is_end {
            let text_to_speak = queue.join(" ");
            queue.clear();

            if let Err(err) = handle.emit(
                "tts_word",
                WordPayload {
                    word: text_to_speak.clone(),
                },
            ) {
                eprintln!("Failed to emit tts_word event: {}", err);
                return;
            }

            let state = handle.state::<crate::AppState>();
            let synth = Arc::clone(&state.synth);
            let player_lock = Arc::clone(&state.audio_player);

            std::thread::spawn(move || {
                if let Ok(audio_result) = synth.synthesize_parallel(text_to_speak, None) {
                    let mut samples = Vec::new();
                    for result in audio_result {
                        if let Ok(buf) = result {
                            samples.append(&mut buf.into_vec());
                        }
                    }

                    let channels = NonZeroU16::new(1).unwrap();
                    let rate = NonZeroU32::new(22050).unwrap();
                    let source = SamplesBuffer::new(channels, rate, samples);

                    if let Ok(player) = player_lock.lock() {
                        player.append(source);
                    }
                }
            });
        }
    });
}

#[tauri::command]
async fn generate_text(
    state: State<'_, AppState>,
    prompt: String,
    image_bytes: Vec<u8>,
    app: AppHandle,
) -> Result<String, String> {
    let app_handle = app.clone();

    let res = tauri::async_runtime::spawn_blocking(move || {
        let app_handle2 = app_handle.clone();
        let state = app_handle.state::<AppState>();
        let mut pipeline = state
            .pipeline
            .lock()
            .map_err(|_| "Failed to lock pipeline".to_string())?;

        let cfg = GenerationConfig::default();

        let image_data = if image_bytes.is_empty() {
            None
        } else {
            Some(image_bytes)
        };

        //println!("Generating text for prompt: {}", prompt);

        pipeline.generate(&prompt, image_data, &cfg, app_handle2)
    })
    .await
    .map_err(|e| e.to_string())?;

    res
}

// Initialise App Stae which loads llama.cpp backend and model, and creates the pipeline.
// App State exists throughout the entire app lifetime (automatically manages memory).
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new().unwrap()) // TODO: Handle errors properly later.
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![generate_text, get_img,])
        .setup(|app| {
            // Pass the handle to our separate function
            setup_voice_engine(app.handle());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
