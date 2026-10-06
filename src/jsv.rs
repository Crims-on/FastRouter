//! Small helpers that give `serde_json::Value` JavaScript-like semantics.
//! The translation layer is a close port of 9router's JS, so these keep the
//! Rust code readable and behaviourally identical (truthiness, String(x), ...).

use serde_json::{Map, Value, json};

/// JS truthiness.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Returns `Some(v)` when the value is truthy.
pub fn t(v: &Value) -> Option<&Value> {
    truthy(v).then_some(v)
}

/// `String(x)` for display/template purposes.
pub fn js_string(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.abs() < 1e21 {
                    return format!("{}", f as i64);
                }
            }
            n.to_string()
        }
        Value::Array(a) => a.iter().map(|x| if x.is_null() { String::new() } else { js_string(x) }).collect::<Vec<_>>().join(","),
        Value::Object(_) => "[object Object]".into(),
    }
}

/// `String(x ?? "")`
pub fn sn(v: &Value) -> String {
    if v.is_null() { String::new() } else { js_string(v) }
}

/// `x || ""` for strings.
pub fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

/// `x || default` for strings.
pub fn s_or<'a>(v: &'a Value, d: &'a str) -> &'a str {
    match v.as_str() {
        Some(x) if !x.is_empty() => x,
        _ => d,
    }
}

/// JS `typeof x === "number"` → value, else 0.
pub fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

pub fn int(v: &Value) -> i64 {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or(0)
}

/// A JSON number from f64, keeping integers integral (JS has one number type).
pub fn jnum(f: f64) -> Value {
    if f.fract() == 0.0 && f.abs() < 9.0e15 {
        json!(f as i64)
    } else {
        json!(f)
    }
}

pub fn is_obj(v: &Value) -> bool {
    v.is_object()
}

pub fn arr(v: &Value) -> &[Value] {
    v.as_array().map(|a| a.as_slice()).unwrap_or(&[])
}

pub fn arr_mut(v: &mut Value) -> Option<&mut Vec<Value>> {
    v.as_array_mut()
}

/// Deletes a key from an object value (no-op otherwise). Returns the removed value.
pub fn del(v: &mut Value, key: &str) -> Option<Value> {
    v.as_object_mut().and_then(|o| o.shift_remove(key))
}

/// `key in obj` / `obj[key] !== undefined`.
pub fn has(v: &Value, key: &str) -> bool {
    v.as_object().map(|o| o.contains_key(key)).unwrap_or(false)
}

/// Present and not null (JS `!= null`).
pub fn defined(v: &Value, key: &str) -> bool {
    v.get(key).map(|x| !x.is_null()).unwrap_or(false)
}

pub fn obj() -> Value {
    Value::Object(Map::new())
}

/// Shallow merge `{...a, ...b}`.
pub fn spread(a: &Value, b: &Value) -> Value {
    let mut out = a.as_object().cloned().unwrap_or_default();
    if let Some(bo) = b.as_object() {
        for (k, v) in bo {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// Object without the given keys (`const { a, b, ...rest } = obj`).
pub fn omit(v: &Value, keys: &[&str]) -> Value {
    let mut out = v.as_object().cloned().unwrap_or_default();
    for k in keys {
        out.shift_remove(*k);
    }
    Value::Object(out)
}

/// JS `JSON.parse` falling back to `fallback` on error; non-strings pass through.
pub fn safe_parse(v: &Value, fallback: Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or(fallback),
        other => other.clone(),
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn now_s() -> i64 {
    now_ms() / 1000
}

pub fn uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub fn rand_hex(bytes: usize) -> String {
    let mut out = String::new();
    while out.len() < bytes * 2 {
        out.push_str(&uuid::Uuid::new_v4().simple().to_string());
    }
    out.truncate(bytes * 2);
    out
}

/// `Date.now().toString(36)`
pub fn base36(mut n: u64) -> String {
    const D: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".into();
    }
    let mut v = Vec::new();
    while n > 0 {
        v.push(D[(n % 36) as usize]);
        n /= 36;
    }
    v.reverse();
    String::from_utf8(v).unwrap()
}

pub fn sha256_hex(data: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(data.as_bytes()))
}

/// Case-insensitive glob (`*` wildcard) anchored at both ends (pricing.js matchPattern).
pub fn glob_match(pattern: &str, model: &str) -> bool {
    let p = pattern.to_ascii_lowercase();
    let m = model.to_ascii_lowercase();
    let parts: Vec<&str> = p.split('*').collect();
    if parts.len() == 1 {
        return p == m;
    }
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !m.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else if i == parts.len() - 1 {
            return m.len() >= pos + part.len() && m[pos..].ends_with(part);
        } else {
            match m[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

/// Recursively collects JSON values into a JS-style `Map` stand-in: an object keyed by string.
pub fn map_get<'a>(map: &'a Value, key: impl ToString) -> &'a Value {
    &map[key.to_string().as_str()]
}

pub fn map_size(map: &Value) -> usize {
    map.as_object().map(|o| o.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globbing() {
        assert!(glob_match("*claude*opus-4.6*", "claude-opus-4.6-thinking"));
        assert!(glob_match("hy3*", "hy3-preview"));
        assert!(!glob_match("hy3*", "xhy3"));
        assert!(glob_match("*gpt-5*", "GPT-5"));
        assert!(glob_match("glm-5.2", "GLM-5.2"));
        assert!(!glob_match("glm-5.2", "glm-5.2x"));
        assert!(glob_match("MiniMax-*", "minimax-m2"));
    }

    #[test]
    fn strings() {
        assert_eq!(js_string(&json!(5)), "5");
        assert_eq!(js_string(&json!(5.0)), "5");
        assert_eq!(js_string(&json!(true)), "true");
        assert_eq!(base36(35), "z");
        assert!(truthy(&json!([])));
        assert!(!truthy(&json!("")));
    }
}

/// parseInt(s, 10): leading integer prefix (after whitespace), else None.
pub fn parse_int_prefix(s: &str) -> Option<i64> {
    let t = s.trim_start();
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: i64 = digits.parse().ok()?;
    Some(if neg { -n } else { n })
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// ISO-8601 UTC timestamp (like JS `new Date(ms).toISOString()`).
pub fn iso_from_ms(ms: i64) -> String {
    let days = ms.div_euclid(86_400_000);
    let rem = ms.rem_euclid(86_400_000);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3_600_000, rem / 60_000 % 60, rem / 1000 % 60, rem % 1000)
}

/// Parses an ISO-8601 timestamp ("2025-01-02T03:04:05(.sss)(Z|+hh:mm)") → epoch ms.
pub fn parse_iso_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.len() < 10 {
        return None;
    }
    let y: i64 = s.get(0..4)?.parse().ok()?;
    let mo: i64 = s.get(5..7)?.parse().ok()?;
    let d: i64 = s.get(8..10)?.parse().ok()?;
    let mut ms = days_from_civil(y, mo, d) * 86_400_000;
    if s.len() >= 19 {
        let h: i64 = s.get(11..13)?.parse().ok()?;
        let mi: i64 = s.get(14..16)?.parse().ok()?;
        let se: i64 = s.get(17..19)?.parse().ok()?;
        ms += (h * 3600 + mi * 60 + se) * 1000;
        let mut rest = &s[19..];
        if let Some(r) = rest.strip_prefix('.') {
            let frac: String = r.chars().take_while(|c| c.is_ascii_digit()).collect();
            let f3: String = format!("{frac:0<3}").chars().take(3).collect();
            ms += f3.parse::<i64>().unwrap_or(0);
            rest = &r[frac.len()..];
        }
        if rest.len() >= 6 && (rest.starts_with('+') || rest.starts_with('-')) {
            let sign = if rest.starts_with('-') { -1 } else { 1 };
            let oh: i64 = rest.get(1..3)?.parse().ok()?;
            let om: i64 = rest.get(4..6)?.parse().ok()?;
            ms -= sign * (oh * 3600 + om * 60) * 1000;
        }
    }
    Some(ms)
}
