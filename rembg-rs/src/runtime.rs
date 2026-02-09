use std::{
	env,
	io::{self, Write},
	path::{Path, PathBuf}
};

use anyhow::{Context, Result, bail};
use std::sync::OnceLock;

use crate::{cli, download, pypi};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredEp {
	DirectML,
	Cuda
}

#[derive(Debug, Clone)]
pub struct Plan {
	pub runtime_package: &'static str,
	pub ep: Option<PreferredEp>,
	/// If true, we already prompted the user (or `-y` was passed) and can download without a second prompt.
	pub allow_download: bool
}

pub struct OnnxRuntimeInstall {
	pub main_lib: PathBuf
}

static ORT_MAIN_LIB: OnceLock<PathBuf> = OnceLock::new();

pub struct DownloadProgress<'a> {
	pub url: &'a str,
	pub progress: download::Progress
}

pub fn plan_noninteractive(device: cli::Device, gpu_backend: cli::GpuBackend, allow_download: bool) -> Result<Plan> {
	let os = env::consts::OS;
	let arch = env::consts::ARCH;

	if device == cli::Device::Cpu {
		// On Windows, prefer the DirectML runtime even for CPU runs when possible. This keeps the
		// process pinned to a single ONNX Runtime DLL so the GUI can toggle CPU <-> GPU (DirectML)
		// without requiring an app restart.
		//
		// If downloads are disallowed and DirectML isn't cached yet, fall back to the CPU runtime.
		if os == "windows" && gpu_backend != cli::GpuBackend::Cuda {
			if allow_download || has_any_cached_runtime("onnxruntime-directml")? {
				return Ok(Plan {
					runtime_package: "onnxruntime-directml",
					ep: None,
					allow_download
				});
			}
		}
		return Ok(Plan {
			runtime_package: "onnxruntime",
			ep: None,
			allow_download
		});
	}

	let backend = match gpu_backend {
		cli::GpuBackend::Auto => match os {
			"windows" => {
				// Prefer cached backend to avoid downloads when possible.
				if has_any_cached_runtime("onnxruntime-directml")? {
					cli::GpuBackend::Directml
				} else if has_any_cached_runtime("onnxruntime-gpu")? {
					cli::GpuBackend::Cuda
				} else {
					cli::GpuBackend::Directml
				}
			}
			"linux" => cli::GpuBackend::Cuda,
			_ => cli::GpuBackend::Auto
		},
		other => other
	};

	match backend {
		cli::GpuBackend::Directml => {
			if os != "windows" {
				bail!("DirectML backend is only supported on Windows");
			}
			Ok(Plan {
				runtime_package: "onnxruntime-directml",
				ep: Some(PreferredEp::DirectML),
				allow_download
			})
		}
		cli::GpuBackend::Cuda => {
			let ok = (os == "windows" && arch == "x86_64")
				|| (os == "linux" && (arch == "x86_64" || arch == "aarch64"));
			if !ok {
				bail!("CUDA backend not supported on this platform ({os}/{arch})");
			}
			Ok(Plan {
				runtime_package: "onnxruntime-gpu",
				ep: Some(PreferredEp::Cuda),
				allow_download
			})
		}
		cli::GpuBackend::Auto => bail!("GPU backend not supported on this platform ({os}/{arch})")
	}
}

pub fn resolve_plan(args: &cli::Args) -> Result<Plan> {
	let os = env::consts::OS;
	let arch = env::consts::ARCH;

	let mut allow_download = args.yes;

	let want_gpu = match args.device {
		cli::Device::Cpu => false,
		cli::Device::Gpu => true,
		cli::Device::Auto => {
			if os != "windows" {
				false
			} else {
				// If any GPU runtime is already cached, enable GPU without prompting.
				// On Windows, we prefer DirectML when auto-selecting, but CUDA might be cached instead.
				if has_any_cached_runtime("onnxruntime-directml")? || has_any_cached_runtime("onnxruntime-gpu")? {
					true
				} else {
					let msg = "Enable GPU acceleration? This will download a GPU-enabled ONNX Runtime backend.";
					let ok = prompt_yes_no(msg, args.yes)?;
					if ok {
						allow_download = true; // don't ask again for the actual download
					}
					ok
				}
			}
		}
	};

	if !want_gpu {
		return Ok(Plan {
			runtime_package: "onnxruntime",
			ep: None,
			allow_download
		});
	}

	let backend = match args.gpu_backend {
		cli::GpuBackend::Auto => match os {
			"windows" => {
				// Prefer already-cached backend to avoid redundant downloads.
				if has_any_cached_runtime("onnxruntime-directml")? {
					cli::GpuBackend::Directml
				} else if has_any_cached_runtime("onnxruntime-gpu")? {
					cli::GpuBackend::Cuda
				} else {
					cli::GpuBackend::Directml
				}
			}
			"linux" => cli::GpuBackend::Cuda,
			_ => cli::GpuBackend::Auto
		},
		other => other
	};

	match backend {
		cli::GpuBackend::Directml => {
			if os != "windows" {
				bail!("DirectML backend is only supported on Windows");
			}
			Ok(Plan {
				runtime_package: "onnxruntime-directml",
				ep: Some(PreferredEp::DirectML),
				allow_download
			})
		}
		cli::GpuBackend::Cuda => {
			let ok = (os == "windows" && arch == "x86_64")
				|| (os == "linux" && (arch == "x86_64" || arch == "aarch64"));
			if !ok {
				bail!("CUDA backend not supported on this platform ({os}/{arch})");
			}
			Ok(Plan {
				runtime_package: "onnxruntime-gpu",
				ep: Some(PreferredEp::Cuda),
				allow_download
			})
		}
		cli::GpuBackend::Auto => bail!("GPU backend not supported on this platform ({os}/{arch})")
	}
}

pub fn ensure_onnxruntime(plan: &Plan) -> Result<OnnxRuntimeInstall> {
	let os = env::consts::OS;
	let arch = env::consts::ARCH;

	let package = plan.runtime_package;
	let pkg_dir = cache_base_dir()?
		.join("onnxruntime")
		.join(package);

	// 1) If any version is already installed, use it (avoid prompting on new upstream releases).
	if let Some(main_lib) = find_any_installed_lib(os, &pkg_dir)? {
		return Ok(OnnxRuntimeInstall { main_lib });
	}

	// 2) Otherwise, download latest wheel for this platform.
	let proj = pypi::fetch_project(package)?;
	let os_norm = normalize_os(os);
	let arch_norm = normalize_arch(arch);
	let wheel = pypi::select_wheel(&proj, &os_norm, &arch_norm)?;

	let base = pkg_dir.join(&proj.info.version);
	let wheel_path = base.join(&wheel.filename);
	let lib_dir = base.join("lib");

	let msg = match package {
		"onnxruntime" => "Download ONNX Runtime CPU backend now?",
		"onnxruntime-directml" => "Download ONNX Runtime DirectML (GPU) backend now?",
		"onnxruntime-gpu" => "Download ONNX Runtime CUDA (GPU) backend now?",
		_ => "Download ONNX Runtime backend now?"
	};
	if !wheel_path.exists() && !prompt_yes_no(msg, plan.allow_download)? {
		bail!("runtime download cancelled by user");
	}

	if !wheel_path.exists() {
		download::download_to_path(
			&wheel.url,
			&wheel_path,
			download::Digests {
				sha256_hex: Some(wheel.digests.sha256.clone()),
				md5_hex: None
			}
		)
		.with_context(|| format!("download wheel: {}", wheel.filename))?;
	}

	extract_ort_libs_from_wheel(&wheel_path, &lib_dir)?;

	let main_lib = find_main_lib(os, &lib_dir)
		.ok_or_else(|| anyhow::anyhow!("unable to find ONNX Runtime library after extraction in {}", lib_dir.display()))?;

	Ok(OnnxRuntimeInstall { main_lib })
}

pub fn ensure_onnxruntime_noninteractive(
	plan: &Plan,
	mut on_progress: impl FnMut(DownloadProgress<'_>)
) -> Result<OnnxRuntimeInstall> {
	let os = env::consts::OS;
	let arch = env::consts::ARCH;

	let package = plan.runtime_package;
	let pkg_dir = cache_base_dir()?
		.join("onnxruntime")
		.join(package);

	// 1) If any version is already installed, use it (avoid prompting on new upstream releases).
	if let Some(main_lib) = find_any_installed_lib(os, &pkg_dir)? {
		return Ok(OnnxRuntimeInstall { main_lib });
	}

	// 2) Otherwise, download latest wheel for this platform.
	let proj = pypi::fetch_project(package)?;
	let os_norm = normalize_os(os);
	let arch_norm = normalize_arch(arch);
	let wheel = pypi::select_wheel(&proj, &os_norm, &arch_norm)?;

	let base = pkg_dir.join(&proj.info.version);
	let wheel_path = base.join(&wheel.filename);
	let lib_dir = base.join("lib");

	if !wheel_path.exists() && !plan.allow_download {
		bail!("download required: runtime package {} ({})", package, wheel.url);
	}

	if !wheel_path.exists() {
		download::download_to_path_with_progress(
			&wheel.url,
			&wheel_path,
			download::Digests {
				sha256_hex: Some(wheel.digests.sha256.clone()),
				md5_hex: None
			},
			|p| on_progress(DownloadProgress { url: &wheel.url, progress: p })
		)
		.with_context(|| format!("download wheel: {}", wheel.filename))?;
	}

	extract_ort_libs_from_wheel(&wheel_path, &lib_dir)?;

	let main_lib = find_main_lib(os, &lib_dir)
		.ok_or_else(|| anyhow::anyhow!("unable to find ONNX Runtime library after extraction in {}", lib_dir.display()))?;

	Ok(OnnxRuntimeInstall { main_lib })
}

pub fn init_ort(rt: &OnnxRuntimeInstall) -> Result<()> {
	if let Some(p) = ORT_MAIN_LIB.get() {
		if p != &rt.main_lib {
			bail!(
				"ONNX Runtime is already initialized with {}. Restart required to switch to {}.",
				p.display(),
				rt.main_lib.display()
			);
		}
		return Ok(());
	}

	// Must be called before any `Session` is created.
	let builder = ort::init_from(&rt.main_lib)
		.with_context(|| format!("load onnxruntime from {}", rt.main_lib.display()))?;
	builder.commit();
	let _ = ORT_MAIN_LIB.set(rt.main_lib.clone());
	Ok(())
}

fn find_main_lib(os: &str, lib_dir: &Path) -> Option<PathBuf> {
	let prefer = match os {
		"windows" => "onnxruntime.dll",
		"macos" => "libonnxruntime.dylib",
		_ => "libonnxruntime.so"
	};
	let p = lib_dir.join(prefer);
	if p.exists() {
		return Some(p);
	}

	let mut best: Option<(u64, PathBuf)> = None;
	let rd = std::fs::read_dir(lib_dir).ok()?;
	for ent in rd.flatten() {
		let path = ent.path();
		let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();

		let ok = match os {
			"windows" => name == "onnxruntime.dll",
			"macos" => name.starts_with("libonnxruntime") && name.ends_with(".dylib"),
			_ => name.starts_with("libonnxruntime.so")
		};
		if !ok {
			continue;
		}

		let len = ent.metadata().ok().map(|m| m.len()).unwrap_or(0);
		match &best {
			None => best = Some((len, path)),
			Some((best_len, _)) if len > *best_len => best = Some((len, path)),
			_ => {}
		}
	}
	best.map(|(_, p)| p)
}

fn extract_ort_libs_from_wheel(wheel_path: &Path, lib_dir: &Path) -> Result<()> {
	use std::io::Read;

	std::fs::create_dir_all(lib_dir).with_context(|| format!("create lib dir: {}", lib_dir.display()))?;

	let file = std::fs::File::open(wheel_path).with_context(|| format!("open wheel: {}", wheel_path.display()))?;
	let mut zip = zip::ZipArchive::new(file).context("open zip archive")?;

	for i in 0..zip.len() {
		let mut f = zip.by_index(i).context("read zip entry")?;
		if f.is_dir() {
			continue;
		}
		let name = f.name().replace('\\', "/");
		if !name.contains("/capi/") {
			continue;
		}
		if !is_runtime_lib_file(&name) {
			continue;
		}

		let base = Path::new(&name)
			.file_name()
			.and_then(|s| s.to_str())
			.ok_or_else(|| anyhow::anyhow!("invalid zip entry name: {name}"))?;
		let dst = lib_dir.join(base);
		if dst.exists() {
			continue;
		}

		let mut out = std::fs::File::create(&dst).with_context(|| format!("create file: {}", dst.display()))?;
		let mut buf = Vec::new();
		f.read_to_end(&mut buf).context("read zip entry bytes")?;
		out.write_all(&buf).context("write extracted file")?;
	}

	Ok(())
}

fn is_runtime_lib_file(name: &str) -> bool {
	let lower = name.to_ascii_lowercase();
	lower.ends_with(".dll") || lower.ends_with(".so") || lower.contains(".so.") || lower.ends_with(".dylib")
}

fn cache_base_dir() -> Result<PathBuf> {
	let dirs = directories::ProjectDirs::from("rs", "rembg", "rembg-rs")
		.ok_or_else(|| anyhow::anyhow!("unable to resolve user cache directory"))?;
	Ok(dirs.cache_dir().to_path_buf())
}

fn has_any_cached_runtime(package: &str) -> Result<bool> {
	let os = env::consts::OS;
	let base = cache_base_dir()?
		.join("onnxruntime")
		.join(package);
	if !base.exists() {
		return Ok(false);
	}
	let rd = std::fs::read_dir(&base).with_context(|| format!("read cache dir: {}", base.display()))?;
	for ent in rd.flatten() {
		if !ent.path().is_dir() {
			continue;
		}
		let lib_dir = ent.path().join("lib");
		if find_main_lib(os, &lib_dir).is_some() {
			return Ok(true);
		}
	}
	Ok(false)
}

fn find_any_installed_lib(os: &str, pkg_dir: &Path) -> Result<Option<PathBuf>> {
	if !pkg_dir.exists() {
		return Ok(None);
	}

	let mut versions: Vec<PathBuf> = Vec::new();
	let rd = std::fs::read_dir(pkg_dir).with_context(|| format!("read cache dir: {}", pkg_dir.display()))?;
	for ent in rd.flatten() {
		let p = ent.path();
		if p.is_dir() {
			versions.push(p);
		}
	}

	// Prefer higher semantic versions when possible.
	versions.sort_by(|a, b| cmp_version_dir_names(b, a));

	for vdir in versions {
		let lib_dir = vdir.join("lib");
		if let Some(main) = find_main_lib(os, &lib_dir) {
			return Ok(Some(main));
		}

		// If we have a wheel but never extracted libs (interrupted run), extract without prompting.
		let wheel = find_any_wheel(&vdir)?;
		if let Some(wheel_path) = wheel {
			extract_ort_libs_from_wheel(&wheel_path, &lib_dir)?;
			if let Some(main) = find_main_lib(os, &lib_dir) {
				return Ok(Some(main));
			}
		}
	}

	Ok(None)
}

fn find_any_wheel(dir: &Path) -> Result<Option<PathBuf>> {
	if !dir.exists() {
		return Ok(None);
	}
	let rd = std::fs::read_dir(dir).with_context(|| format!("read dir: {}", dir.display()))?;
	for ent in rd.flatten() {
		let p = ent.path();
		if p.is_file() {
			if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
				if ext.eq_ignore_ascii_case("whl") {
					return Ok(Some(p));
				}
			}
		}
	}
	Ok(None)
}

fn cmp_version_dir_names(a: &PathBuf, b: &PathBuf) -> std::cmp::Ordering {
	let a = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
	let b = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
	cmp_versions(a, b)
}

fn cmp_versions(a: &str, b: &str) -> std::cmp::Ordering {
	use std::cmp::Ordering;

	let pa = parse_version_prefix(a);
	let pb = parse_version_prefix(b);

	for i in 0..pa.len().max(pb.len()) {
		let av = pa.get(i).copied().unwrap_or(0);
		let bv = pb.get(i).copied().unwrap_or(0);
		match av.cmp(&bv) {
			Ordering::Equal => continue,
			ord => return ord
		}
	}

	// Fall back to lexicographic for suffixes.
	a.cmp(b)
}

fn parse_version_prefix(s: &str) -> Vec<u64> {
	// Parse `1.24.1` from `1.24.1` or `1.24.1.post1` etc.
	let mut out = Vec::new();
	for part in s.split('.') {
		let mut digits = String::new();
		for ch in part.chars() {
			if ch.is_ascii_digit() {
				digits.push(ch);
			} else {
				break;
			}
		}
		if digits.is_empty() {
			break;
		}
		if let Ok(v) = digits.parse::<u64>() {
			out.push(v);
		} else {
			break;
		}
	}
	out
}

fn normalize_os(os: &str) -> String {
	match os {
		"windows" => "windows".to_string(),
		"macos" => "macos".to_string(),
		_ => "linux".to_string()
	}
}

fn normalize_arch(arch: &str) -> String {
	match arch {
		"x86_64" => "x86_64".to_string(),
		"aarch64" => "aarch64".to_string(),
		_ => arch.to_string()
	}
}

fn prompt_yes_no(msg: &str, assume_yes: bool) -> Result<bool> {
	if assume_yes {
		return Ok(true);
	}
	eprint!("{msg} [y/N] ");
	io::stderr().flush().ok();

	let mut s = String::new();
	io::stdin().read_line(&mut s).context("read user input")?;
	let s = s.trim().to_ascii_lowercase();
	Ok(matches!(s.as_str(), "y" | "yes"))
}
