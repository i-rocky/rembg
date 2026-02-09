# rembg-rs

Rust background removal library + CLI. It runs ONNX segmentation models (U2Net/ISNet variants) via ONNX Runtime.

## CLI

Build:
`cargo build -p rembg-rs`

Basic usage:
`cargo run -p rembg-rs -- input.jpg --model u2netp`

Options:

- `--device cpu|gpu|auto`
- `--gpu-backend auto|directml|cuda`
- `--mask-threshold 0..255` (binarize mask; helps remove residual haze but can cause jagged edges)
- `--color-key-tolerance 0..255` (heuristic "punch-through" for background-colored pixels)
- `--bgcolor RRGGBB` (composite onto a solid color instead of transparency)
- `--only-mask` (write the grayscale mask)

## Models

Model `.onnx` files are downloaded from `danielgatis/rembg` GitHub release assets and cached.
Supported model ids are defined in `rembg-rs/src/model.rs`.

## ONNX Runtime

`rembg-rs` uses the `ort` crate with dynamic loading (`load-dynamic`). At first run it downloads an ONNX Runtime wheel
from PyPI for your platform, then extracts the native runtime libraries from the wheel's `capi/` folder.

This keeps the Rust binary smaller and avoids requiring a Python environment.

Cache root (Windows example):
`%LOCALAPPDATA%\\rembg\\rembg-rs\\cache\\`

