//! Model resolution: maps a requested model (or combo) to an ordered list of
//! upstream targets, honouring priority, round-robin and cooldowns.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::catalog::{self, Provider};
use crate::db::{Db, ProviderConn};

#[derive(Clone)]
pub struct Target {
    pub conn: ProviderConn,
    pub provider: &'static Provider,
    pub model: String,
}

impl Target {
    pub fn base_url(&self) -> String {
        self.conn
            .base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| self.provider.base_url.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    /// The fully-qualified model id clients use for this target.
    pub fn qualified(&self) -> String {
        format!("{}/{}", self.conn.prefix, self.model)
    }
}

#[derive(Default)]
pub struct Router {
    cooldowns: Mutex<HashMap<String, Instant>>,
    rr: Mutex<HashMap<String, usize>>,
}

impl Router {
    pub fn cooldown(&self, key: &str, dur: Duration) {
        self.cooldowns
            .lock()
            .unwrap()
            .insert(key.to_string(), Instant::now() + dur);
    }

    pub fn clear_cooldown(&self, conn_id: &str) {
        self.cooldowns
            .lock()
            .unwrap()
            .retain(|k, _| !k.starts_with(conn_id));
    }

    pub fn cooling_until(&self, key: &str) -> Option<Duration> {
        let map = self.cooldowns.lock().unwrap();
        map.get(key)
            .and_then(|t| t.checked_duration_since(Instant::now()))
    }

    pub fn is_cooling(&self, t: &Target) -> bool {
        self.cooling_until(&t.conn.id).is_some()
            || self
                .cooling_until(&format!("{}|{}", t.conn.id, t.model))
                .is_some()
    }

    /// Resolves a requested model into an ordered list of targets.
    pub fn resolve(&self, db: &Db, requested: &str, strategy: &str) -> Vec<Target> {
        let conns: Vec<ProviderConn> = db
            .list_connections()
            .into_iter()
            .filter(|c| c.enabled)
            .collect();
        let entries = match db.get_combo(requested) {
            Some(combo) => combo.models,
            None => vec![requested.to_string()],
        };
        let mut targets = Vec::new();
        for entry in entries {
            targets.extend(self.resolve_one(&conns, entry.trim(), strategy));
        }
        // Healthy targets first; cooling ones are kept as a last resort.
        let (mut ok, cooling): (Vec<_>, Vec<_>) =
            targets.into_iter().partition(|t| !self.is_cooling(t));
        ok.extend(cooling);
        ok
    }

    fn resolve_one(&self, conns: &[ProviderConn], model: &str, strategy: &str) -> Vec<Target> {
        let build = |group: Vec<&ProviderConn>, model: &str| -> Vec<Target> {
            group
                .into_iter()
                .filter_map(|c| {
                    Some(Target {
                        conn: c.clone(),
                        provider: catalog::get(&c.provider)?,
                        model: model.to_string(),
                    })
                })
                .collect()
        };

        // Explicit `prefix/model`.
        if let Some((prefix, rest)) = model.split_once('/') {
            let group: Vec<&ProviderConn> = conns.iter().filter(|c| c.prefix == prefix).collect();
            if !group.is_empty() && !rest.is_empty() {
                return self.order(prefix, build(group, rest), strategy);
            }
        }

        // Bare model id: find the first prefix whose known models include it.
        let mut prefixes: Vec<&str> = Vec::new();
        for c in conns {
            let known = c.models.iter().any(|m| m == model)
                || catalog::get(&c.provider)
                    .map(|p| p.models.contains(&model))
                    .unwrap_or(false);
            if known && !prefixes.contains(&c.prefix.as_str()) {
                prefixes.push(&c.prefix);
            }
        }
        let Some(prefix) = prefixes.first() else {
            return vec![];
        };
        let group: Vec<&ProviderConn> = conns.iter().filter(|c| c.prefix == *prefix).collect();
        self.order(prefix, build(group, model), strategy)
    }

    fn order(&self, prefix: &str, mut targets: Vec<Target>, strategy: &str) -> Vec<Target> {
        if strategy == "round-robin" && targets.len() > 1 {
            let mut rr = self.rr.lock().unwrap();
            let n = rr.entry(prefix.to_string()).or_insert(0);
            let start = *n % targets.len();
            *n = n.wrapping_add(1);
            targets.rotate_left(start);
        }
        targets
    }
}

/// Every model id the router can currently serve, for `/v1/models`.
pub fn available_models(db: &Db) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for combo in db.list_combos() {
        out.push((combo.name, "combo".into()));
    }
    let mut seen = std::collections::HashSet::new();
    for c in db.list_connections().into_iter().filter(|c| c.enabled) {
        let Some(p) = catalog::get(&c.provider) else {
            continue;
        };
        let models: Vec<String> = if c.models.is_empty() {
            p.models.iter().map(|m| m.to_string()).collect()
        } else {
            c.models.clone()
        };
        for m in models {
            let id = format!("{}/{}", c.prefix, m);
            if seen.insert(id.clone()) {
                out.push((id, c.provider.clone()));
            }
        }
    }
    out
}
