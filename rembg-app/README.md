# rembg-app

Tauri v2 desktop GUI for `rembg-rs` with a SvelteKit UI.

## Run (Dev)

From this folder:

1. `npm install`
2. `npm run tauri dev`

## What It Does

- Lets you drop/select an input image.
- Shows output PNG with a checkerboard background.
- Exposes model/device/options and re-runs on changes.
- Shows download progress while fetching runtime/model files.

## Backend

Tauri commands are implemented in `rembg-app/src-tauri/src/lib.rs` and call into the `rembg-rs` crate.
