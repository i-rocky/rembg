use anyhow::{Context, Result, bail};
use image::{DynamicImage, GrayImage, RgbImage, Rgba, RgbaImage};

pub fn apply_alpha(img: &RgbImage, mask: &GrayImage, threshold: Option<u8>, color_key_tolerance: Option<u8>) -> DynamicImage {
	let (w, h) = (img.width(), img.height());
	let mut out = RgbaImage::new(w, h);

	let bg = color_key_tolerance.and_then(|t| {
		if t == 0 {
			None
		} else {
			Some((estimate_bg_rgb(img), (t as i32) * (t as i32)))
		}
	});

	for y in 0..h {
		for x in 0..w {
			let p = img.get_pixel(x, y);
			let mut a = mask.get_pixel(x, y)[0];
			if let Some(t) = threshold {
				a = if a >= t { 255 } else { 0 };
			}

			if let Some(((br, bgc, bb), tol2)) = bg {
				let dr = p[0] as i32 - br as i32;
				let dg = p[1] as i32 - bgc as i32;
				let db = p[2] as i32 - bb as i32;
				let d2 = dr * dr + dg * dg + db * db;
				if d2 <= tol2 {
					a = 0;
				}
			}

			out.put_pixel(x, y, Rgba([p[0], p[1], p[2], a]));
		}
	}
	DynamicImage::ImageRgba8(out)
}

pub fn composite_over_bg(img: &RgbImage, mask: &GrayImage, threshold: Option<u8>, bgcolor: &str) -> Result<DynamicImage> {
	let (bg_r, bg_g, bg_b) = parse_hex_rgb(bgcolor)?;
	let (w, h) = (img.width(), img.height());
	let mut out = RgbImage::new(w, h);

	for y in 0..h {
		for x in 0..w {
			let p = img.get_pixel(x, y);
			let mut a = mask.get_pixel(x, y)[0];
			if let Some(t) = threshold {
				a = if a >= t { 255 } else { 0 };
			}
			let a = a as u32; // 0..255
			let inv = 255u32 - a;

			let r = (p[0] as u32 * a + bg_r as u32 * inv + 127) / 255;
			let g = (p[1] as u32 * a + bg_g as u32 * inv + 127) / 255;
			let b = (p[2] as u32 * a + bg_b as u32 * inv + 127) / 255;

			out.put_pixel(x, y, image::Rgb([r as u8, g as u8, b as u8]));
		}
	}

	Ok(DynamicImage::ImageRgb8(out))
}

fn estimate_bg_rgb(img: &RgbImage) -> (u8, u8, u8) {
	let w = img.width();
	let h = img.height();
	if w == 0 || h == 0 {
		return (255, 255, 255);
	}

	let patch = 6u32.min(w).min(h);
	let mut sum = [0u64; 3];
	let mut n = 0u64;

	for &(ox, oy) in &[(0u32, 0u32), (w.saturating_sub(patch), 0u32), (0u32, h.saturating_sub(patch)), (w.saturating_sub(patch), h.saturating_sub(patch))] {
		for y in oy..(oy + patch) {
			for x in ox..(ox + patch) {
				let p = img.get_pixel(x, y);
				sum[0] += p[0] as u64;
				sum[1] += p[1] as u64;
				sum[2] += p[2] as u64;
				n += 1;
			}
		}
	}

	if n == 0 {
		return (255, 255, 255);
	}

	((sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8)
}

fn parse_hex_rgb(s: &str) -> Result<(u8, u8, u8)> {
	let s = s.trim();
	let s = s.strip_prefix('#').unwrap_or(s);
	if s.len() != 6 {
		bail!("invalid bgcolor {s:?} (expected RRGGBB or #RRGGBB)");
	}
	let r = u8::from_str_radix(&s[0..2], 16).context("parse r")?;
	let g = u8::from_str_radix(&s[2..4], 16).context("parse g")?;
	let b = u8::from_str_radix(&s[4..6], 16).context("parse b")?;
	Ok((r, g, b))
}
