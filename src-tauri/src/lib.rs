// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::fs;
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![get_img])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
