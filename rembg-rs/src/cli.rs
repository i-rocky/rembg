use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Device {
	/// Always use CPU inference.
	Cpu,
	/// Prefer GPU. Falls back to CPU if unavailable.
	Gpu,
	/// Default device selection (CPU unless you confirm GPU on Windows).
	Auto
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GpuBackend {
	/// Platform default (Windows: DirectML, Linux: CUDA).
	Auto,
	/// Windows only, uses DirectML (DirectX 12).
	Directml,
	/// NVIDIA CUDA execution provider (Windows x64, Linux x64/aarch64).
	Cuda
}

#[derive(Debug, Parser)]
#[command(name = "rembg-rs", version, about = "Background removal (rembg-like) as a single CLI binary")]
pub struct Args {
	/// Input image path.
	pub input: PathBuf,

	/// Output image path (defaults to `<input>.png` or `<input>_mask.png`).
	#[arg(short, long)]
	pub output: Option<PathBuf>,

	/// Model name (see `rembg-rs/src/model.rs` for the supported list).
	#[arg(short = 'm', long, default_value = "u2netp")]
	pub model: String,

	/// Device selection.
	#[arg(long, value_enum, default_value_t = Device::Auto)]
	pub device: Device,

	/// Which GPU backend to use (only relevant when `--device gpu` or when `--device auto` enables GPU).
	#[arg(long, value_enum, default_value_t = GpuBackend::Auto)]
	pub gpu_backend: GpuBackend,

	/// Output just the mask (grayscale PNG), not an RGBA cutout.
	#[arg(long)]
	pub only_mask: bool,

	/// Binarize the mask: alpha becomes 0 or 255 based on this threshold (0-255).
	/// Helps remove residual "inner background" caused by soft masks.
	#[arg(long, value_parser = clap::value_parser!(u8))]
	pub mask_threshold: Option<u8>,

	/// Force alpha=0 for pixels close to the estimated background color (sampled from corners).
	/// Format: 0-255, where higher removes more. Recommended start: 20-40. (0 disables)
	#[arg(long, value_parser = clap::value_parser!(u8))]
	pub color_key_tolerance: Option<u8>,

	/// Composite the foreground over a solid background instead of transparency.
	/// Format: RRGGBB or #RRGGBB.
	#[arg(long)]
	pub bgcolor: Option<String>,

	/// Assume "yes" for interactive prompts (e.g., downloading GPU backend).
	#[arg(short = 'y', long)]
	pub yes: bool
}
