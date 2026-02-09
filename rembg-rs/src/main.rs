use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use rembg_rs::{cli, compose, model, runtime, u2net};

fn main() {
	// Keep stdout clean for piping; errors go to stderr via `anyhow`.
	if let Err(e) = run() {
		eprintln!("{e:#}");
		std::process::exit(1);
	}
}

fn run() -> Result<()> {
	let args = cli::Args::parse();

	let plan = runtime::resolve_plan(&args)?;
	let rt = runtime::ensure_onnxruntime(&plan)?;
	runtime::init_ort(&rt)?;

	let model = model::ensure_model(&args.model)?;

	let input_path = &args.input;
	let img = image::open(input_path).with_context(|| format!("open image: {}", input_path.display()))?;
	let img_rgb = img.to_rgb8();

	let mask = u2net::predict_mask(&model.path, model.input_size, &img_rgb, plan.ep)
		.with_context(|| format!("run model: {}", model.path.display()))?;

	let out_path: PathBuf = match args.output {
		Some(p) => p,
		None => {
			let stem = input_path
				.file_stem()
				.and_then(|s| s.to_str())
				.unwrap_or("out");
			let suffix = if args.only_mask { "_mask.png" } else { "_rembg.png" };
			input_path.with_file_name(format!("{stem}{suffix}"))
		}
	};

	if args.only_mask {
		let mask_out = if let Some(t) = args.mask_threshold {
			let mut m = mask.clone();
			for p in m.pixels_mut() {
				p.0[0] = if p.0[0] >= t { 255 } else { 0 };
			}
			m
		} else {
			mask.clone()
		};
		mask_out.save(&out_path)
			.with_context(|| format!("write mask: {}", out_path.display()))?;
		return Ok(());
	}

	let out = if let Some(bg) = args.bgcolor.as_deref() {
		compose::composite_over_bg(&img_rgb, &mask, args.mask_threshold, bg)?
	} else {
		compose::apply_alpha(&img_rgb, &mask, args.mask_threshold, args.color_key_tolerance)
	};

	out.save(&out_path)
		.with_context(|| format!("write image: {}", out_path.display()))?;
	Ok(())
}
