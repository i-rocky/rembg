use std::io::Cursor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, ImageFormat};
use serde::{Deserialize, Serialize};

use crate::{compose, model, runtime, u2net};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
	Cpu,
	Gpu
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
	Auto,
	Directml,
	Cuda
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveOptions {
	pub model: String,
	pub device: Device,
	pub gpu_backend: GpuBackend,
	pub mask_threshold: Option<u8>,
	pub bgcolor: Option<String>,
	/// If set, uses a simple color-key to force alpha=0 for pixels close to the estimated background color.
	/// Useful for punching "inner background" holes when the model returns a solid silhouette.
	pub color_key_tolerance: Option<u8>,
	/// If false, backend returns an error instead of downloading runtime/model.
	pub allow_download: bool,
	/// If true, return mask bytes as well.
	pub include_mask: bool
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveResult {
	pub output_png: Vec<u8>,
	pub mask_png: Option<Vec<u8>>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
	pub stage: String,          // "runtime" | "model" | "infer" | "encode"
	pub url: Option<String>,    // for downloads
	pub downloaded: Option<u64>,
	pub total: Option<u64>,
	pub done: Option<bool>,
	pub message: Option<String>
}

pub fn remove_background_bytes(
	input_bytes: &[u8],
	opts: &RemoveOptions,
	mut on_progress: impl FnMut(ProgressEvent)
) -> Result<RemoveResult> {
	on_progress(ProgressEvent {
		stage: "decode".to_string(),
		url: None,
		downloaded: None,
		total: None,
		done: None,
		message: None
	});

	let img = image::load_from_memory(input_bytes).context("decode input image")?;
	let rgb = img.to_rgb8();

	let plan = runtime::plan_noninteractive(
		match opts.device {
			Device::Cpu => crate::cli::Device::Cpu,
			Device::Gpu => crate::cli::Device::Gpu
		},
		match opts.gpu_backend {
			GpuBackend::Auto => crate::cli::GpuBackend::Auto,
			GpuBackend::Directml => crate::cli::GpuBackend::Directml,
			GpuBackend::Cuda => crate::cli::GpuBackend::Cuda
		},
		opts.allow_download
	)?;

	on_progress(ProgressEvent {
		stage: "runtime".to_string(),
		url: None,
		downloaded: None,
		total: None,
		done: None,
		message: Some(format!("Ensure ONNX Runtime ({})", plan.runtime_package))
	});

	let rt = runtime::ensure_onnxruntime_noninteractive(&plan, |p| {
		on_progress(ProgressEvent {
			stage: "runtime".to_string(),
			url: Some(p.url.to_string()),
			downloaded: Some(p.progress.downloaded),
			total: p.progress.total,
			done: Some(p.progress.done),
			message: None
		});
	})?;
	runtime::init_ort(&rt)?;

	on_progress(ProgressEvent {
		stage: "model".to_string(),
		url: None,
		downloaded: None,
		total: None,
		done: None,
		message: Some(format!("Ensure model ({})", opts.model))
	});

	let model_install = model::ensure_model_noninteractive(&opts.model, opts.allow_download, |p| {
		on_progress(ProgressEvent {
			stage: "model".to_string(),
			url: Some(p.url.to_string()),
			downloaded: Some(p.progress.downloaded),
			total: p.progress.total,
			done: Some(p.progress.done),
			message: None
		});
	})?;

	on_progress(ProgressEvent {
		stage: "infer".to_string(),
		url: None,
		downloaded: None,
		total: None,
		done: None,
		message: None
	});

	let mask = u2net::predict_mask(&model_install.path, model_install.input_size, &rgb, plan.ep)
		.with_context(|| format!("run model: {}", model_install.path.display()))?;

	let out_img: DynamicImage = if let Some(bg) = opts.bgcolor.as_deref() {
		compose::composite_over_bg(&rgb, &mask, opts.mask_threshold, bg)?
	} else {
		compose::apply_alpha(&rgb, &mask, opts.mask_threshold, opts.color_key_tolerance)
	};

	on_progress(ProgressEvent {
		stage: "encode".to_string(),
		url: None,
		downloaded: None,
		total: None,
		done: None,
		message: None
	});

	let output_png = encode_png(&out_img)?;
	let mask_png = if opts.include_mask {
		Some(encode_mask_png(&mask, opts.mask_threshold)?)
	} else {
		None
	};

	Ok(RemoveResult { output_png, mask_png })
}

fn encode_png(img: &DynamicImage) -> Result<Vec<u8>> {
	let mut buf = Vec::new();
	let mut cur = Cursor::new(&mut buf);
	img.write_to(&mut cur, ImageFormat::Png).context("encode png")?;
	Ok(buf)
}

fn encode_mask_png(mask: &GrayImage, threshold: Option<u8>) -> Result<Vec<u8>> {
	let mut m = mask.clone();
	if let Some(t) = threshold {
		for p in m.pixels_mut() {
			p.0[0] = if p.0[0] >= t { 255 } else { 0 };
		}
	}
	let img = DynamicImage::ImageLuma8(m);
	encode_png(&img)
}
