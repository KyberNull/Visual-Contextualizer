// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod llama;

use crate::llama::inference::{ContextConfig, GenerationConfig, LlamaPipeline, LlamaRuntime};
use crate::llama::model::LlamaModel;
use std::fs;
use std::env;
use std::num::NonZeroU32;
use std::sync::Mutex;
use tauri::{AppHandle, LogicalSize, Manager, Size, State};
use tauri:: Listener;
use piper_rs::synth::PiperSpeechSynthesizer;
use rodio::{DeviceSinkBuilder, Player, buffer::SamplesBuffer};
use std::sync::Arc;
use std::num::NonZeroU16;
use tauri::Emitter;
use::serde::Serialize;
use screenshots::Screen;
use screenshots::image::ImageFormat;
use std::io::Cursor;
use std::thread;
use std::time::Duration;

// こんにちは世界

#[tauri::command]
async fn get_full_screenshot_bytes() -> Result<Vec<u8>, String> {
    let screens = Screen::all().map_err(|e| e.to_string())?;

    let screen = screens
        .first()
        .ok_or_else(|| "No screens found".to_string())?;
    let image = screen.capture().map_err(|e| e.to_string())?;
    
    let mut buffer = Cursor::new(Vec::new());
    image.write_to(&mut buffer, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    
    Ok(buffer.into_inner())
}

#[tauri::command]
async fn capture_hidden_window_screenshot(window: tauri::Window) -> Result<Vec<u8>, String> {
    // 1. Hide the window to capture what's behind it
    window
        .minimize()
        .map_err(|e| format!("Failed to minimize window: {}", e))?;

    // Give the OS compositor time to hide the app window
    thread::sleep(Duration::from_millis(250));

    let capture_result = (|| -> Result<Vec<u8>, String> {
        let screens = Screen::all().map_err(|e| e.to_string())?;
        let screen = screens
            .first()
            .ok_or_else(|| "No screens found".to_string())?;

        let image = screen.capture().map_err(|e| e.to_string())?;
        let mut buffer = Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        Ok(buffer.into_inner())
    })();

    // 2. THE FIX: If capture worked, prepare for Snipping mode
    if capture_result.is_ok() {
        // Unminimize so it's active again
        let _ = window.unminimize();
        // Go Fullscreen so the canvas covers the whole monitor
        let _ = window.set_fullscreen(true);
        // Set Always on Top so the snip tool isn't covered by other windows
        let _ = window.set_always_on_top(true);
        let _ = window.set_focus();
    } else {
        // Fallback if capture failed
        let _ = window.unminimize();
        let _ = window.maximize();
    }

    capture_result
}

#[tauri::command]
fn reset_window_to_initial_size(window: tauri::Window) -> Result<(), String> {
    let _ = window.unminimize();
    let _ = window.unmaximize();
    let _ = window.set_fullscreen(false);
    let _ = window.set_always_on_top(false);

    window
        .set_size(Size::Logical(LogicalSize::new(400.0, 600.0)))
        .map_err(|e| format!("Failed to resize window: {}", e))?;

    let _ = window.center();
    let _ = window.set_focus();
    Ok(())
}


#[tauri::command]
fn set_fullscreen(window: tauri::Window, is_fullscreen: bool) -> Result<(), String> {
    window
        .set_fullscreen(is_fullscreen)
        .map_err(|e| format!("Failed to set fullscreen: {}", e))?;
    // In a snip tool, we usually want the window on top of everything
    if is_fullscreen {
        window
            .set_always_on_top(true)
            .map_err(|e| format!("Failed to set always-on-top: {}", e))?;
    } else {
        window
            .set_always_on_top(false)
            .map_err(|e| format!("Failed to clear always-on-top: {}", e))?;
    }
    Ok(())
}


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

    synth: Arc<PiperSpeechSynthesizer>,
    audio_player: Arc<Mutex<Player>>,
    _mixer_sink: rodio::MixerDeviceSink,
}

impl AppState {
    fn new() -> Result<Self, String> {
        let runtime = LlamaRuntime::init();

        //Finding model with the relative path
        let path = env::current_dir().unwrap().join("Qwen3.5-0.8B-GGUF").join("Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
        let path_str = path.to_str().ok_or("Error")?;

        println!("Loading model from {}", path.display());

        let model = LlamaModel::load(path_str)?; // Lazy loads because mmap() is on by default in llama.cpp
        let cfg = ContextConfig::default();
        let pipeline = LlamaPipeline::from_model(model, &cfg)?;

        println!("Model loaded successfully");
        

        let mixer_sink = DeviceSinkBuilder::open_default_sink()
            .map_err(|e| format!("Audio Device Error: {}", e))?;
        let player = Player::connect_new(mixer_sink.mixer());

        let config_path = env::current_dir().unwrap().join("en_US-libritts_r-medium.onnx.json");
        let piper_model = piper_rs::from_config_path(&config_path)
            .map_err(|e| e.to_string())?;
    
        let synth = PiperSpeechSynthesizer::new(piper_model)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            runtime,
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


pub fn setup_voice_engine(app : &AppHandle){

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

            if let Err(err) = handle.emit("tts_word", WordPayload { word: text_to_speak.clone() }) {
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
async fn generate_text(state: State<'_ , AppState>, prompt: String,image_bytes:Vec<u8>, app : AppHandle) -> Result<String, String> {

    let app_handle = app.clone();

    let res = tauri::async_runtime::spawn_blocking(move ||{
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
    
        }).await.map_err(|e| e.to_string())?;
    
    res 
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
            get_full_screenshot_bytes,
            capture_hidden_window_screenshot,
            reset_window_to_initial_size,
            set_fullscreen,
            get_img,
        ])
        .setup(|app| {
            // Pass the handle to our separate function
            setup_voice_engine(app.handle());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}