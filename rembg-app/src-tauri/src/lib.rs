use tauri::Emitter;
use serde::Serialize;
use std::path::PathBuf;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
async fn remove_background(
	window: tauri::Window,
	request_id: u64,
	input_bytes: Vec<u8>,
	options: rembg_rs::core::RemoveOptions
) -> Result<rembg_rs::core::RemoveResult, String> {
	let win = window.clone();
	tauri::async_runtime::spawn_blocking(move || {
		rembg_rs::core::remove_background_bytes(&input_bytes, &options, |evt| {
			#[derive(Serialize, Clone)]
			#[serde(rename_all = "camelCase")]
			struct UiProgress {
				request_id: u64,
				#[serde(flatten)]
				evt: rembg_rs::core::ProgressEvent
			}
			let _ = win.emit("rembg:progress", UiProgress { request_id, evt });
		})
		.map_err(|e| e.to_string())
	})
	.await
	.map_err(|e| e.to_string())?
}

#[tauri::command]
fn write_file_bytes(path: String, bytes: Vec<u8>) -> Result<(), String> {
	let p = PathBuf::from(path);
	if let Some(parent) = p.parent() {
		std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
	}
	std::fs::write(&p, bytes).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![remove_background, write_file_bytes])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
