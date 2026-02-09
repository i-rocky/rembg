use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::download;

pub struct ModelInstall {
	pub path: PathBuf,
	pub input_size: u32
}

pub fn ensure_model(name: &str) -> Result<ModelInstall> {
	ensure_model_noninteractive(name, true, |_p| {}) // CLI behavior: always allow download, progress prints via download.rs
}

pub struct DownloadProgress<'a> {
	pub url: &'a str,
	pub progress: download::Progress
}

pub fn ensure_model_noninteractive(
	name: &str,
	allow_download: bool,
	mut on_progress: impl FnMut(DownloadProgress<'_>)
) -> Result<ModelInstall> {
	let m = model_spec(name)?;
	let base = cache_base_dir()?.join("models");
	let path = base.join(format!("{}.onnx", m.name));

	if !path.exists() {
		if !allow_download {
			bail!("download required: model {} ({})", m.name, m.url);
		}

		download::download_to_path_with_progress(
			m.url,
			&path,
			download::Digests {
				sha256_hex: None,
				md5_hex: None
			},
			|p| on_progress(DownloadProgress { url: m.url, progress: p })
		)
		.with_context(|| format!("download model {} from {}", m.name, m.url))?;
	}

	Ok(ModelInstall {
		path,
		input_size: m.input_size
	})
}

struct ModelSpec {
	name: &'static str,
	url: &'static str,
	input_size: u32
}

fn model_spec(name: &str) -> Result<ModelSpec> {
	// Model URLs from the upstream rembg release assets (v0.0.0).
	// Keep the list small for now; add more once the pipeline is solid.
	match name.trim().to_ascii_lowercase().as_str() {
		"u2netp" => Ok(ModelSpec {
			name: "u2netp",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
			input_size: 320
		}),
		"u2net" => Ok(ModelSpec {
			name: "u2net",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
			input_size: 320
		}),
		"u2net_human_seg" => Ok(ModelSpec {
			name: "u2net_human_seg",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
			input_size: 320
		}),
		"u2net_cloth_seg" => Ok(ModelSpec {
			name: "u2net_cloth_seg",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx",
			input_size: 320
		}),
		"silueta" => Ok(ModelSpec {
			name: "silueta",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
			input_size: 320
		}),
		// ISNet models tend to prefer larger input sizes; 1024 is common in rembg usage.
		// This will be slower but should improve detail and interior background separation.
		"isnet-general-use" => Ok(ModelSpec {
			name: "isnet-general-use",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
			input_size: 1024
		}),
		"isnet-anime" => Ok(ModelSpec {
			name: "isnet-anime",
			url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
			input_size: 1024
		}),
		other => bail!("unsupported model: {other} (supported: u2netp, u2net, u2net_human_seg, u2net_cloth_seg, silueta, isnet-general-use, isnet-anime)")
	}
}

fn cache_base_dir() -> Result<PathBuf> {
	let dirs = directories::ProjectDirs::from("rs", "rembg", "rembg-rs")
		.ok_or_else(|| anyhow::anyhow!("unable to resolve user cache directory"))?;
	Ok(dirs.cache_dir().to_path_buf())
}
