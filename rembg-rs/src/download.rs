use std::{
	fs,
	fs::File,
	io::{Read, Write},
	path::Path,
	time::Instant
};

use anyhow::{Context, Result, bail};
use sha2::Digest as _;

pub struct Digests {
	pub sha256_hex: Option<String>,
	pub md5_hex: Option<String>
}

#[derive(Debug, Clone, Copy)]
pub struct Progress {
	pub downloaded: u64,
	pub total: Option<u64>,
	pub secs: f64,
	pub done: bool
}

pub fn download_to_path(url: &str, dst: &Path, digests: Digests) -> Result<()> {
	download_to_path_with_progress(url, dst, digests, |p| {
		print_progress(url, p.downloaded, p.total, p.secs);
		if p.done {
			eprintln!();
		}
	})
}

pub fn download_to_path_with_progress(
	url: &str,
	dst: &Path,
	digests: Digests,
	mut on_progress: impl FnMut(Progress)
) -> Result<()> {
	if let Some(parent) = dst.parent() {
		fs::create_dir_all(parent).with_context(|| format!("create dir: {}", parent.display()))?;
	}

	let tmp = dst.with_extension("part");
	let _ = fs::remove_file(&tmp);

	let resp = ureq::get(url)
		.call()
		.with_context(|| format!("GET {url}"))?;

	let status = resp.status().as_u16();
	if status / 100 != 2 {
		bail!("download failed (HTTP {status}): {url}");
	}

	let total_len = resp
		.headers()
		.get(ureq::http::header::CONTENT_LENGTH)
		.and_then(|v| v.to_str().ok())
		.and_then(|s| s.parse::<u64>().ok());

	let mut reader = resp.into_body().into_reader();
	let mut file = File::create(&tmp).with_context(|| format!("create file: {}", tmp.display()))?;

	let mut sha256 = digests.sha256_hex.as_deref().map(|_| sha2::Sha256::new());
	let mut md5 = digests.md5_hex.as_deref().map(|_| md5::Context::new());

	let mut buf = [0u8; 64 * 1024];
	let mut downloaded: u64 = 0;
	let start = Instant::now();
	let mut last = Instant::now();

	loop {
		let n = reader.read(&mut buf).context("read response body")?;
		if n == 0 {
			break;
		}
		downloaded += n as u64;

		if let Some(h) = sha256.as_mut() {
			h.update(&buf[..n]);
		}
		if let Some(h) = md5.as_mut() {
			h.consume(&buf[..n]);
		}
		file.write_all(&buf[..n]).context("write file")?;

		if last.elapsed().as_millis() >= 250 {
			on_progress(Progress {
				downloaded,
				total: total_len,
				secs: start.elapsed().as_secs_f64(),
				done: false
			});
			last = Instant::now();
		}
	}
	file.flush().ok();

	on_progress(Progress {
		downloaded,
		total: total_len,
		secs: start.elapsed().as_secs_f64(),
		done: true
	});

	if let (Some(expected), Some(h)) = (digests.sha256_hex.as_deref(), sha256) {
		let got = hex::encode(h.finalize());
		if !eq_hex(expected, &got) {
			bail!("sha256 mismatch for {url}: expected {expected}, got {got}");
		}
	}
	if let (Some(expected), Some(h)) = (digests.md5_hex.as_deref(), md5) {
		let got = format!("{:x}", h.finalize());
		if !eq_hex(expected, &got) {
			bail!("md5 mismatch for {url}: expected {expected}, got {got}");
		}
	}

	fs::rename(&tmp, dst).with_context(|| format!("rename {} -> {}", tmp.display(), dst.display()))?;
	Ok(())
}

fn print_progress(url: &str, downloaded: u64, total: Option<u64>, secs: f64) {
	let mb = |b: u64| (b as f64) / (1024.0 * 1024.0);
	let speed = if secs > 0.0 { mb(downloaded) / secs } else { 0.0 };

	match total {
		Some(t) if t > 0 => {
			let pct = (downloaded as f64) * 100.0 / (t as f64);
			eprint!("\rDownloading {:.1}/{:.1} MiB ({:.0}%) {:.1} MiB/s  {}", mb(downloaded), mb(t), pct, speed, url);
		}
		_ => {
			eprint!("\rDownloading {:.1} MiB {:.1} MiB/s  {}", mb(downloaded), speed, url);
		}
	}
	let _ = std::io::stderr().flush();
}

fn eq_hex(a: &str, b: &str) -> bool {
	a.trim().trim_start_matches("0x").eq_ignore_ascii_case(b.trim().trim_start_matches("0x"))
}
