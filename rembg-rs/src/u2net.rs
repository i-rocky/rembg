use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{GrayImage, Luma, RgbImage};
use image::imageops::FilterType;
use ndarray::Array4;
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;

pub fn predict_mask(
	model_path: &Path,
	input_size: u32,
	img: &RgbImage,
	preferred_ep: Option<crate::runtime::PreferredEp>
) -> Result<GrayImage> {
	let mut session = match preferred_ep {
		None => Session::builder()
			.context("create ORT session builder")?
			.commit_from_file(model_path)
			.with_context(|| format!("load onnx model: {}", model_path.display()))?,
		Some(crate::runtime::PreferredEp::DirectML) => {
			match Session::builder()
				.context("create ORT session builder")?
				.with_execution_providers([ep::DirectML::default().build()])
				.context("configure DirectML EP")?
				.commit_from_file(model_path)
			{
				Ok(s) => s,
				Err(e) => {
					eprintln!("DirectML init failed, falling back to CPU. This can happen if the DirectML provider cannot be loaded on this system: {e:#}");
					Session::builder()
						.context("create ORT session builder")?
						.commit_from_file(model_path)
						.with_context(|| format!("load onnx model (CPU fallback): {}", model_path.display()))?
				}
			}
		}
		Some(crate::runtime::PreferredEp::Cuda) => {
			match Session::builder()
				.context("create ORT session builder")?
				.with_execution_providers([ep::CUDA::default().build()])
				.context("configure CUDA EP")?
				.commit_from_file(model_path)
			{
				Ok(s) => s,
				Err(e) => {
					eprintln!("CUDA init failed, falling back to CPU. This often means the NVIDIA driver / CUDA libraries aren't available on this system: {e:#}");
					Session::builder()
						.context("create ORT session builder")?
						.commit_from_file(model_path)
						.with_context(|| format!("load onnx model (CPU fallback): {}", model_path.display()))?
				}
			}
		}
	};

	let resized = image::imageops::resize(img, input_size, input_size, FilterType::Lanczos3);

	let input = image_to_tensor_nchw(&resized)?;
	let outputs = session
		.run(ort::inputs![TensorRef::from_array_view(&input)?])
		.context("run inference")?;

	if outputs.len() == 0 {
		bail!("model produced no outputs");
	}

	let out0 = &outputs[0];
	let out = out0.try_extract_array::<f32>().context("extract output tensor")?;
	let shape = out.shape();
	if shape.len() != 4 {
		bail!("unexpected output rank: {} (expected 4)", shape.len());
	}
	let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
	if n != 1 {
		bail!("unexpected batch size: {n} (expected 1)");
	}
	if c != 1 {
		// Some exports can produce (1,H,W) or similar, but the common ones are (1,1,H,W).
		// Fail loud for now.
		bail!("unexpected output channels: {c} (expected 1)");
	}

	// Some exported models return probabilities in [0, 1], others return logits.
	// If we incorrectly apply sigmoid to an already-[0,1] map, everything shifts to ~[0.5, 0.73],
	// causing semi-transparent background and broken thresholding.
	let mut min_v = f32::INFINITY;
	let mut max_v = f32::NEG_INFINITY;
	for v in out.iter() {
		min_v = min_v.min(*v);
		max_v = max_v.max(*v);
	}
	let treat_as_prob = min_v >= -0.01 && max_v <= 1.01;

	let mut mask_small = GrayImage::new(w as u32, h as u32);
	for y in 0..h {
		for x in 0..w {
			let v = out[[0, 0, y, x]];
			let s = if treat_as_prob {
				v
			} else {
				// Most segmentation ONNX exports output logits; sigmoid gets us a stable [0,1] probability map.
				1.0 / (1.0 + (-v).exp())
			};
			let px = (s.clamp(0.0, 1.0) * 255.0).round() as u8;
			mask_small.put_pixel(x as u32, y as u32, Luma([px]));
		}
	}

	let mask = image::imageops::resize(&mask_small, img.width(), img.height(), FilterType::Lanczos3);
	Ok(mask)
}

fn image_to_tensor_nchw(img: &RgbImage) -> Result<Array4<f32>> {
	let (w, h) = (img.width() as usize, img.height() as usize);
	let mut t = Array4::<f32>::zeros((1, 3, h, w));

	// This normalization matches a common U2Net ONNX export convention.
	// If a specific model expects Imagenet mean/std, we can add a model-specific branch later.
	for y in 0..h {
		for x in 0..w {
			let p = img.get_pixel(x as u32, y as u32);
			let r = p[0] as f32 / 255.0;
			let g = p[1] as f32 / 255.0;
			let b = p[2] as f32 / 255.0;

			// Scale to [-1, 1]
			t[[0, 0, y, x]] = (r - 0.5) / 0.5;
			t[[0, 1, y, x]] = (g - 0.5) / 0.5;
			t[[0, 2, y, x]] = (b - 0.5) / 0.5;
		}
	}

	Ok(t)
}
