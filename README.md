# rembg (Rust)

This repo contains a Rust reimplementation of the core "rembg" workflow (background removal) plus a desktop GUI.

- `rembg-rs/`: Rust library + CLI that runs ONNX models (U2Net/ISNet family) using ONNX Runtime loaded dynamically.
- `rembg-app/`: Tauri v2 desktop app (Rust backend) with a SvelteKit UI.

## How It Works (High Level)

1. Decode input image to RGB.
2. Resize/normalize to the model input size.
3. Run an ONNX segmentation model to produce a 1-channel mask.
4. Convert the mask to an alpha matte and compose an RGBA PNG.

## Downloads / Cache

To avoid shipping huge binaries, the app downloads on demand and caches locally:

- ONNX models: downloaded from `danielgatis/rembg` GitHub release assets.
- ONNX Runtime: downloaded as a platform wheel from PyPI and extracted to get the native runtime libraries.

On Windows the cache is typically under:
`%LOCALAPPDATA%\\rembg\\rembg-rs\\cache\\`

## GPU Support

- Windows: DirectML (DirectX 12) backend (`onnxruntime-directml`).
- CUDA backend exists for supported platforms, but switching between different ONNX Runtime DLLs inside one process is not supported.

To allow CPU <-> GPU toggling in the GUI on Windows without restarts, the code prefers loading the DirectML ONNX Runtime DLL even for CPU runs (when available/cached).

## Development

### CLI

Build:
`cargo build -p rembg-rs`

Run:
`cargo run -p rembg-rs -- <input.png> --model u2netp`

### GUI App

From `rembg-app/`:

1. `npm install`
2. `npm run tauri dev`

## Notes / Limitations

- "Inner background" removal depends on the model. Some models produce a mostly-solid silhouette mask. There is an optional "color key tolerance" post-process to punch holes for background-colored regions, but it is heuristic.
- This is not a fully static, single-file runtime: ONNX Runtime native libs and model files are downloaded/cached at first use.

