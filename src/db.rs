//! SQLite persistence: provider connections (JSON documents mirroring 9router's
//! connection records), custom provider nodes, combos, model aliases, API keys,
//! settings and usage logs.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Map, Value, json};

pub struct Db {
    conn: Mutex<Connection>,
}

#[derive(Clone, Debug)]
pub struct Combo {
    pub name: String,
    pub models: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    pub key: String,
    pub created_at: i64,
    pub last_used: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct UsageRecord {
    pub ts: i64,
    pub api_key: Option<String>,
    pub requested_model: String,
    pub provider: String,
    pub connection: String,
    pub model: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_tokens: i64,
    pub status: i64,
    pub latency_ms: i64,
    pub cost: f64,
    pub stream: bool,
    pub endpoint: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Totals {
    pub requests: i64,
    pub ok: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct GroupRow {
    pub key: String,
    pub requests: i64,
    pub ok: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cost: f64,
    pub avg_latency: f64,
}

pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Merges `patch` into `doc`: null removes a key, `providerSpecificData` is
/// shallow-merged, everything else replaces.
pub fn merge_doc(doc: &mut Value, patch: &Value) {
    let Some(p) = patch.as_object() else { return };
    if !doc.is_object() {
        *doc = json!({});
    }
    let o = doc.as_object_mut().unwrap();
    for (k, v) in p {
        if v.is_null() {
            o.shift_remove(k);
        } else if k == "providerSpecificData" && v.is_object() {
            let cur = o.entry(k.clone()).or_insert_with(|| json!({}));
            if !cur.is_object() {
                *cur = json!({});
            }
            for (a, b) in v.as_object().unwrap() {
                if b.is_null() {
                    cur.as_object_mut().unwrap().shift_remove(a);
                } else {
                    cur[a] = b.clone();
                }
            }
        } else {
            o.insert(k.clone(), v.clone());
        }
    }
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS provider_connections (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 1,
                is_active INTEGER NOT NULL DEFAULT 1,
                data TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS pc_provider ON provider_connections(provider);
            CREATE TABLE IF NOT EXISTS provider_nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS combos (name TEXT PRIMARY KEY, models TEXT NOT NULL, created_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS model_aliases (alias TEXT PRIMARY KEY, target TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL, last_used INTEGER
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                api_key TEXT,
                requested_model TEXT NOT NULL,
                provider TEXT NOT NULL,
                connection TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost REAL NOT NULL DEFAULT 0,
                stream INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
            "#,
        )?;
        for (col, ddl) in [("cached_tokens", "INTEGER NOT NULL DEFAULT 0"), ("endpoint", "TEXT NOT NULL DEFAULT ''")] {
            let has: bool = conn.prepare("SELECT 1 FROM pragma_table_info('usage') WHERE name = ?1")?.exists([col])?;
            if !has {
                conn.execute_batch(&format!("ALTER TABLE usage ADD COLUMN {col} {ddl}"))?;
            }
        }
        let db = Self { conn: Mutex::new(conn) };
        db.migrate_v1()?;
        Ok(db)
    }

    /// v1 stored one flat `connections` table (api_key/base_url/prefix).
    fn migrate_v1(&self) -> Result<()> {
        let rows: Vec<(String, String, String, String, String, Option<String>, i64, i64, String, i64)> = {
            let c = self.lock();
            let exists: bool = c.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='connections'")?.exists([])?;
            if !exists {
                return Ok(());
            }
            let mut st = c.prepare("SELECT id, provider, name, prefix, api_key, base_url, priority, enabled, models, created_at FROM connections")?;
            let r = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?)))?.flatten().collect();
            r
        };
        for (id, provider, name, prefix, key, base, prio, enabled, _models, created) in rows {
            let mut prov = match provider.as_str() {
                "qwen" => "alicode".to_string(),
                p => p.to_string(),
            };
            let mut psd = json!({});
            if provider == "openai-compatible" || provider == "anthropic-compatible" {
                let node_id = format!("{provider}-{}", uuid::Uuid::new_v4());
                let base = base.clone().unwrap_or_default();
                let ty = provider.clone();
                self.upsert_node(&node_id, &ty, &json!({"name": name, "prefix": prefix, "baseUrl": base, "apiType": "chat"}))?;
                psd = json!({"baseUrl": base, "prefix": prefix, "apiType": "chat", "nodeName": name});
                prov = node_id;
            } else if let Some(b) = base.filter(|b| !b.is_empty()) {
                psd["baseUrl"] = json!(b);
            }
            let doc = json!({"id": id, "provider": prov, "authType": "apikey", "name": name, "apiKey": key, "priority": prio, "isActive": enabled != 0, "providerSpecificData": psd, "createdAt": created});
            self.insert_connection(&doc)?;
        }
        self.lock().execute_batch("ALTER TABLE connections RENAME TO connections_v1_migrated")?;
        Ok(())
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---------- settings ----------

    pub fn get_setting(&self, key: &str) -> Option<String> {
        self.lock().query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional().ok().flatten()
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.lock().execute("INSERT INTO settings(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value", params![key, value])?;
        Ok(())
    }

    pub fn delete_setting(&self, key: &str) -> Result<()> {
        self.lock().execute("DELETE FROM settings WHERE key = ?1", [key])?;
        Ok(())
    }

    /// JSON-valued setting (Null when absent).
    pub fn setting_json(&self, key: &str) -> Value {
        self.get_setting(key).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
    }

    pub fn set_setting_json(&self, key: &str, v: &Value) -> Result<()> {
        self.set_setting(key, &v.to_string())
    }

    // ---------- provider connections ----------

    fn row_doc(id: String, provider: String, priority: i64, active: i64, data: String, created: i64) -> Value {
        let mut d: Value = serde_json::from_str(&data).unwrap_or_else(|_| json!({}));
        d["id"] = json!(id);
        d["provider"] = json!(provider);
        d["priority"] = json!(priority);
        d["isActive"] = json!(active != 0);
        if d.get("createdAt").is_none() {
            d["createdAt"] = json!(created);
        }
        d
    }

    fn query_docs(&self, sql: &str, p: &[&dyn rusqlite::ToSql]) -> Vec<Value> {
        let c = self.lock();
        let Ok(mut st) = c.prepare(sql) else { return vec![] };
        st.query_map(p, |r| Ok(Self::row_doc(r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?))).map(|rows| rows.flatten().collect()).unwrap_or_default()
    }

    pub fn list_connections(&self) -> Vec<Value> {
        self.query_docs("SELECT id, provider, priority, is_active, data, created_at FROM provider_connections ORDER BY provider, priority, created_at", &[])
    }

    /// getProviderConnections({provider, isActive}) sorted by priority.
    pub fn connections_for(&self, provider: &str, active_only: bool) -> Vec<Value> {
        let sql = if active_only {
            "SELECT id, provider, priority, is_active, data, created_at FROM provider_connections WHERE provider = ?1 AND is_active = 1 ORDER BY priority, created_at"
        } else {
            "SELECT id, provider, priority, is_active, data, created_at FROM provider_connections WHERE provider = ?1 ORDER BY priority, created_at"
        };
        self.query_docs(sql, &[&provider])
    }

    pub fn get_connection(&self, id: &str) -> Option<Value> {
        self.query_docs("SELECT id, provider, priority, is_active, data, created_at FROM provider_connections WHERE id = ?1", &[&id]).into_iter().next()
    }

    pub fn insert_connection(&self, doc: &Value) -> Result<String> {
        let id = doc["id"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let provider = doc["provider"].as_str().unwrap_or("").to_string();
        let priority = doc["priority"].as_i64().unwrap_or_else(|| self.connections_for(&provider, false).len() as i64 + 1);
        let active = doc["isActive"].as_bool().unwrap_or(true);
        let mut d = doc.clone();
        d["id"] = json!(id);
        let created = d["createdAt"].as_i64().unwrap_or_else(now);
        d["createdAt"] = json!(created);
        self.lock().execute(
            "INSERT INTO provider_connections(id, provider, priority, is_active, data, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, provider, priority, active as i64, d.to_string(), created],
        )?;
        Ok(id)
    }

    /// updateProviderConnection(id, patch)
    pub fn update_connection(&self, id: &str, patch: &Value) -> Result<Option<Value>> {
        let Some(mut doc) = self.get_connection(id) else { return Ok(None) };
        merge_doc(&mut doc, patch);
        doc["updatedAt"] = json!(now());
        let priority = doc["priority"].as_i64().unwrap_or(1);
        let active = doc["isActive"].as_bool().unwrap_or(true);
        let provider = doc["provider"].as_str().unwrap_or("").to_string();
        self.lock().execute(
            "UPDATE provider_connections SET provider = ?2, priority = ?3, is_active = ?4, data = ?5 WHERE id = ?1",
            params![id, provider, priority, active as i64, doc.to_string()],
        )?;
        Ok(Some(doc))
    }

    pub fn delete_connection(&self, id: &str) -> Result<()> {
        self.lock().execute("DELETE FROM provider_connections WHERE id = ?1", [id])?;
        Ok(())
    }

    // ---------- custom provider nodes ----------

    pub fn upsert_node(&self, id: &str, ty: &str, data: &Value) -> Result<()> {
        let mut d = data.clone();
        d["id"] = json!(id);
        d["type"] = json!(ty);
        self.lock().execute(
            "INSERT INTO provider_nodes(id, type, data, created_at) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(id) DO UPDATE SET type = excluded.type, data = excluded.data",
            params![id, ty, d.to_string(), now()],
        )?;
        Ok(())
    }

    pub fn list_nodes(&self, ty: Option<&str>) -> Vec<Value> {
        let c = self.lock();
        let (sql, has) = match ty {
            Some(_) => ("SELECT data FROM provider_nodes WHERE type = ?1 ORDER BY created_at", true),
            None => ("SELECT data FROM provider_nodes ORDER BY created_at", false),
        };
        let Ok(mut st) = c.prepare(sql) else { return vec![] };
        let map = |r: &rusqlite::Row| r.get::<_, String>(0);
        let rows: Vec<String> = if has { st.query_map([ty.unwrap()], map).map(|r| r.flatten().collect()).unwrap_or_default() } else { st.query_map([], map).map(|r| r.flatten().collect()).unwrap_or_default() };
        rows.into_iter().filter_map(|s| serde_json::from_str(&s).ok()).collect()
    }

    pub fn get_node(&self, id: &str) -> Option<Value> {
        self.lock().query_row("SELECT data FROM provider_nodes WHERE id = ?1", [id], |r| r.get::<_, String>(0)).optional().ok().flatten().and_then(|s| serde_json::from_str(&s).ok())
    }

    pub fn delete_node(&self, id: &str) -> Result<()> {
        let c = self.lock();
        c.execute("DELETE FROM provider_nodes WHERE id = ?1", [id])?;
        c.execute("DELETE FROM provider_connections WHERE provider = ?1", [id])?;
        Ok(())
    }

    // ---------- combos ----------

    pub fn list_combos(&self) -> Vec<Combo> {
        let c = self.lock();
        let Ok(mut st) = c.prepare("SELECT name, models, created_at FROM combos ORDER BY name") else { return vec![] };
        st.query_map([], |r| {
            let m: String = r.get(1)?;
            Ok(Combo { name: r.get(0)?, models: serde_json::from_str(&m).unwrap_or_default(), created_at: r.get(2)? })
        })
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
    }

    pub fn get_combo(&self, name: &str) -> Option<Combo> {
        self.lock()
            .query_row("SELECT name, models, created_at FROM combos WHERE name = ?1", [name], |r| {
                let m: String = r.get(1)?;
                Ok(Combo { name: r.get(0)?, models: serde_json::from_str(&m).unwrap_or_default(), created_at: r.get(2)? })
            })
            .optional()
            .ok()
            .flatten()
    }

    pub fn upsert_combo(&self, name: &str, models: &[String]) -> Result<()> {
        self.lock().execute(
            "INSERT INTO combos(name, models, created_at) VALUES (?1, ?2, ?3) ON CONFLICT(name) DO UPDATE SET models = excluded.models",
            params![name, serde_json::to_string(models)?, now()],
        )?;
        Ok(())
    }

    pub fn delete_combo(&self, name: &str) -> Result<()> {
        self.lock().execute("DELETE FROM combos WHERE name = ?1", [name])?;
        Ok(())
    }

    // ---------- model aliases ----------

    pub fn model_aliases(&self) -> Map<String, Value> {
        let c = self.lock();
        let Ok(mut st) = c.prepare("SELECT alias, target FROM model_aliases ORDER BY alias") else { return Map::new() };
        st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).map(|rows| rows.flatten().map(|(a, t)| (a, json!(t))).collect()).unwrap_or_default()
    }

    pub fn set_model_alias(&self, alias: &str, target: &str) -> Result<()> {
        self.lock().execute("INSERT INTO model_aliases(alias, target) VALUES (?1, ?2) ON CONFLICT(alias) DO UPDATE SET target = excluded.target", params![alias, target])?;
        Ok(())
    }

    pub fn delete_model_alias(&self, alias: &str) -> Result<()> {
        self.lock().execute("DELETE FROM model_aliases WHERE alias = ?1", [alias])?;
        Ok(())
    }

    // ---------- API keys ----------

    pub fn list_api_keys(&self) -> Vec<ApiKey> {
        let c = self.lock();
        let Ok(mut st) = c.prepare("SELECT id, name, key, created_at, last_used FROM api_keys ORDER BY created_at DESC") else { return vec![] };
        st.query_map([], |r| Ok(ApiKey { id: r.get(0)?, name: r.get(1)?, key: r.get(2)?, created_at: r.get(3)?, last_used: r.get(4)? })).map(|rows| rows.flatten().collect()).unwrap_or_default()
    }

    pub fn insert_api_key(&self, k: &ApiKey) -> Result<()> {
        self.lock().execute("INSERT INTO api_keys(id, name, key, created_at) VALUES (?1, ?2, ?3, ?4)", params![k.id, k.name, k.key, k.created_at])?;
        Ok(())
    }

    pub fn delete_api_key(&self, id: &str) -> Result<()> {
        self.lock().execute("DELETE FROM api_keys WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Validates a key; returns its name and stamps last_used.
    pub fn check_api_key(&self, key: &str) -> Option<String> {
        let c = self.lock();
        let name: Option<String> = c.query_row("SELECT name FROM api_keys WHERE key = ?1", [key], |r| r.get(0)).optional().ok().flatten();
        if name.is_some() {
            let _ = c.execute("UPDATE api_keys SET last_used = ?1 WHERE key = ?2", params![now(), key]);
        }
        name
    }

    pub fn has_api_keys(&self) -> bool {
        self.lock().query_row("SELECT COUNT(*) FROM api_keys", [], |r| r.get::<_, i64>(0)).map(|n| n > 0).unwrap_or(false)
    }

    // ---------- usage ----------

    pub fn insert_usage(&self, u: &UsageRecord) {
        let res = self.lock().execute(
            "INSERT INTO usage(ts, api_key, requested_model, provider, connection, model, prompt_tokens,
             completion_tokens, cached_tokens, status, latency_ms, cost, stream, endpoint, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![u.ts, u.api_key, u.requested_model, u.provider, u.connection, u.model, u.prompt_tokens, u.completion_tokens, u.cached_tokens, u.status, u.latency_ms, u.cost, u.stream as i64, u.endpoint, u.error],
        );
        if let Err(e) = res {
            tracing::warn!("failed to record usage: {e}");
        }
    }

    pub fn totals_since(&self, since: i64) -> Totals {
        self.lock()
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(status BETWEEN 200 AND 299), 0), COALESCE(SUM(prompt_tokens), 0),
                 COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0) FROM usage WHERE ts >= ?1",
                [since],
                |r| Ok(Totals { requests: r.get(0)?, ok: r.get(1)?, prompt_tokens: r.get(2)?, completion_tokens: r.get(3)?, cost: r.get(4)? }),
            )
            .unwrap_or_default()
    }

    pub fn group_since(&self, column: &str, since: i64) -> Vec<GroupRow> {
        let column = match column {
            "provider" => "provider",
            "api_key" => "COALESCE(api_key, '(none)')",
            "connection" => "connection",
            _ => "model",
        };
        let sql = format!(
            "SELECT {column} AS k, COUNT(*), COALESCE(SUM(status BETWEEN 200 AND 299), 0), SUM(prompt_tokens),
             SUM(completion_tokens), SUM(cost), AVG(latency_ms)
             FROM usage WHERE ts >= ?1 GROUP BY k ORDER BY COUNT(*) DESC LIMIT 50"
        );
        let c = self.lock();
        let Ok(mut st) = c.prepare(&sql) else { return vec![] };
        st.query_map([since], |r| Ok(GroupRow { key: r.get(0)?, requests: r.get(1)?, ok: r.get(2)?, prompt_tokens: r.get(3)?, completion_tokens: r.get(4)?, cost: r.get(5)?, avg_latency: r.get(6)? }))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
    }

    pub fn histogram(&self, buckets: i64, size: i64) -> Vec<(i64, i64, i64)> {
        let end = (now() / size + 1) * size;
        let start = end - buckets * size;
        let mut out: Vec<(i64, i64, i64)> = (0..buckets).map(|i| (start + i * size, 0, 0)).collect();
        let c = self.lock();
        let Ok(mut st) = c.prepare("SELECT (ts - ?1) / ?2 AS b, COUNT(*), SUM(prompt_tokens + completion_tokens) FROM usage WHERE ts >= ?1 GROUP BY b") else { return out };
        if let Ok(rows) = st.query_map(params![start, size], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))) {
            for (b, n, t) in rows.flatten() {
                if let Some(slot) = out.get_mut(b as usize) {
                    slot.1 = n;
                    slot.2 = t;
                }
            }
        }
        out
    }

    pub fn recent_usage(&self, limit: i64, offset: i64) -> Vec<UsageRecord> {
        let c = self.lock();
        let Ok(mut st) = c.prepare(
            "SELECT ts, api_key, requested_model, provider, connection, model, prompt_tokens, completion_tokens,
             status, latency_ms, cost, stream, error, cached_tokens, endpoint FROM usage ORDER BY id DESC LIMIT ?1 OFFSET ?2",
        ) else {
            return vec![];
        };
        st.query_map(params![limit, offset], |r| {
            Ok(UsageRecord {
                ts: r.get(0)?,
                api_key: r.get(1)?,
                requested_model: r.get(2)?,
                provider: r.get(3)?,
                connection: r.get(4)?,
                model: r.get(5)?,
                prompt_tokens: r.get(6)?,
                completion_tokens: r.get(7)?,
                status: r.get(8)?,
                latency_ms: r.get(9)?,
                cost: r.get(10)?,
                stream: r.get::<_, i64>(11)? != 0,
                error: r.get(12)?,
                cached_tokens: r.get(13)?,
                endpoint: r.get(14)?,
            })
        })
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
    }

    pub fn usage_count(&self) -> i64 {
        self.lock().query_row("SELECT COUNT(*) FROM usage", [], |r| r.get(0)).unwrap_or(0)
    }

    pub fn clear_usage(&self) -> Result<()> {
        self.lock().execute("DELETE FROM usage", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connections_roundtrip() {
        let db = Db::open_in_memory().unwrap();
        let id = db.insert_connection(&json!({"provider": "openai", "authType": "apikey", "apiKey": "k", "providerSpecificData": {"a": 1}})).unwrap();
        let c = db.get_connection(&id).unwrap();
        assert_eq!(c["priority"], 1);
        db.update_connection(&id, &json!({"modelLock_x": "2099-01-01T00:00:00Z", "providerSpecificData": {"b": 2}})).unwrap();
        let c = db.get_connection(&id).unwrap();
        assert_eq!(c["providerSpecificData"], json!({"a": 1, "b": 2}));
        db.update_connection(&id, &json!({"modelLock_x": null, "isActive": false})).unwrap();
        let c = db.get_connection(&id).unwrap();
        assert!(c.get("modelLock_x").is_none());
        assert!(db.connections_for("openai", true).is_empty());
        db.upsert_node("openai-compatible-chat-1", "openai-compatible", &json!({"prefix": "my", "baseUrl": "http://x"})).unwrap();
        assert_eq!(db.list_nodes(Some("openai-compatible"))[0]["prefix"], "my");
        db.set_model_alias("fast", "groq/llama").unwrap();
        assert_eq!(db.model_aliases()["fast"], "groq/llama");
    }
}
