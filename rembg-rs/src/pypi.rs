use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct PypiProject {
	pub info: PypiInfo,
	pub releases: HashMap<String, Vec<PypiReleaseFile>>
}

#[derive(Debug, Deserialize)]
pub struct PypiInfo {
	pub version: String
}

#[derive(Debug, Deserialize)]
pub struct PypiReleaseFile {
	pub filename: String,
	pub url: String,
	pub packagetype: String,
	pub digests: PypiDigests
}

#[derive(Debug, Deserialize)]
pub struct PypiDigests {
	pub sha256: String
}

pub fn fetch_project(name: &str) -> Result<PypiProject> {
	let url = format!("https://pypi.org/pypi/{name}/json");
	let resp = ureq::get(&url)
		.call()
		.with_context(|| format!("GET {url}"))?;
	let status = resp.status().as_u16();
	if status / 100 != 2 {
		bail!("pypi request failed (HTTP {status}): {url}");
	}
	let s = resp.into_body().read_to_string().context("read pypi json")?;
	let proj: PypiProject = serde_json::from_str(&s).context("parse pypi json")?;
	Ok(proj)
}

pub fn select_wheel<'a>(proj: &'a PypiProject, os: &str, arch: &str) -> Result<&'a PypiReleaseFile> {
	let version = &proj.info.version;
	let files = proj
		.releases
		.get(version)
		.with_context(|| format!("missing releases entry for version {version}"))?;

	let mut candidates: Vec<&PypiReleaseFile> = files
		.iter()
		.filter(|f| f.packagetype == "bdist_wheel")
		.collect();

	candidates.sort_by(|a, b| a.filename.cmp(&b.filename));

	let f = candidates
		.into_iter()
		.find(|f| wheel_matches(&f.filename, os, arch))
		.with_context(|| format!("no wheel found for {os}/{arch} in {version}"))?;

	Ok(f)
}

fn wheel_matches(filename: &str, os: &str, arch: &str) -> bool {
	// We only need the native runtime library embedded in the wheel; python tags are irrelevant.
	// Platform tags vary a lot on Linux/macOS, so match on the conservative suffix.
	match (os, arch) {
		("windows", "x86_64") => filename.ends_with("win_amd64.whl"),
		("windows", "aarch64") => filename.ends_with("win_arm64.whl"),
		("linux", "x86_64") => filename.ends_with("x86_64.whl") && filename.contains("manylinux"),
		("linux", "aarch64") => filename.ends_with("aarch64.whl") && filename.contains("manylinux"),
		("macos", "aarch64") => filename.ends_with("arm64.whl") && filename.contains("macosx"),
		("macos", "x86_64") => filename.ends_with("x86_64.whl") && filename.contains("macosx"),
		_ => false
	}
}

trait ReadBodyToString {
	fn read_to_string(self) -> Result<String>;
}

impl ReadBodyToString for ureq::Body {
	fn read_to_string(self) -> Result<String> {
		use std::io::Read;
		let mut r = self.into_reader();
		let mut s = String::new();
		r.read_to_string(&mut s).context("read response body")?;
		Ok(s)
	}
}
