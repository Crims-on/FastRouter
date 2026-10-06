//! Media helpers shared by executors and the media endpoints.

use base64::Engine;

const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v) => {
            let o = v.octets();
            o[0] == 10 || o[0] == 127 || o[0] == 0 || (o[0] == 172 && (16..=31).contains(&o[1])) || (o[0] == 192 && o[1] == 168) || (o[0] == 169 && o[1] == 254) || (o[0] == 100 && (64..=127).contains(&o[1]))
        }
        std::net::IpAddr::V6(v) => {
            if let Some(v4) = v.to_ipv4_mapped() {
                return is_private_ip(std::net::IpAddr::V4(v4));
            }
            let s = v.segments();
            v.is_loopback() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80
        }
    }
}

/// SSRF guard: every resolved address of `host` must be public.
pub async fn host_is_public(host: &str) -> bool {
    let h = host.to_lowercase();
    if h.is_empty() || h == "localhost" || h == "metadata.google.internal" {
        return false;
    }
    match tokio::net::lookup_host((h.as_str(), 443)).await {
        Ok(addrs) => {
            let v: Vec<_> = addrs.collect();
            !v.is_empty() && v.iter().all(|a| !is_private_ip(a.ip()))
        }
        Err(_) => false,
    }
}

fn sniff_mime(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if b.starts_with(b"GIF8") {
        Some("image/gif")
    } else if b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// Downloads a remote image (public hosts only, ≤10 MB) as a data URI.
pub async fn fetch_image_as_data_url(url: &str) -> Option<String> {
    let u = reqwest::Url::parse(url).ok()?;
    if !matches!(u.scheme(), "http" | "https") || !host_is_public(u.host_str()?).await {
        return None;
    }
    let r = crate::exec::no_redirect_client().get(url).timeout(std::time::Duration::from_secs(15)).send().await.ok()?;
    if !r.status().is_success() {
        return None;
    }
    let ct = r.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(|s| s.split(';').next().unwrap_or("").trim().to_string());
    let bytes = r.bytes().await.ok()?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return None;
    }
    let mime = sniff_mime(&bytes).map(str::to_owned).or(ct.filter(|c| c.starts_with("image/")))?;
    Some(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&bytes)))
}
