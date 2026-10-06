//! SQLite persistence: connections, combos, API keys, settings and usage logs.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, params};

pub struct Db {
    conn: Mutex<Connection>,
}

#[derive(Clone, Debug)]
pub struct ProviderConn {
    pub id: String,
    pub provider: String,
    pub name: String,
    /// Model prefix used to address this connection (`prefix/model`).
    pub prefix: String,
    pub api_key: String,
    /// Optional base URL override (required for custom providers).
    pub base_url: Option<String>,
    pub priority: i64,
    pub enabled: bool,
    /// Models fetched from the upstream `/models` endpoint.
    pub models: Vec<String>,
    pub created_at: i64,
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
    pub status: i64,
    pub latency_ms: i64,
    pub cost: f64,
    pub stream: bool,
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
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connections (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                name TEXT NOT NULL,
                prefix TEXT NOT NULL,
                api_key TEXT NOT NULL DEFAULT '',
                base_url TEXT,
                priority INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                models TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS combos (
                name TEXT PRIMARY KEY,
                models TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL,
                last_used INTEGER
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
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---------- settings ----------

    pub fn get_setting(&self, key: &str) -> Option<String> {
        self.lock()
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| {
                r.get(0)
            })
            .optional()
            .ok()
            .flatten()
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.lock().execute(
            "INSERT INTO settings(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ---------- connections ----------

    fn row_to_conn(r: &Row) -> rusqlite::Result<ProviderConn> {
        let models: String = r.get("models")?;
        Ok(ProviderConn {
            id: r.get("id")?,
            provider: r.get("provider")?,
            name: r.get("name")?,
            prefix: r.get("prefix")?,
            api_key: r.get("api_key")?,
            base_url: r.get("base_url")?,
            priority: r.get("priority")?,
            enabled: r.get::<_, i64>("enabled")? != 0,
            models: serde_json::from_str(&models).unwrap_or_default(),
            created_at: r.get("created_at")?,
        })
    }

    pub fn list_connections(&self) -> Vec<ProviderConn> {
        let conn = self.lock();
        let mut stmt =
            match conn.prepare("SELECT * FROM connections ORDER BY priority ASC, created_at ASC") {
                Ok(s) => s,
                Err(_) => return vec![],
            };
        stmt.query_map([], Self::row_to_conn)
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub fn get_connection(&self, id: &str) -> Option<ProviderConn> {
        self.lock()
            .query_row(
                "SELECT * FROM connections WHERE id = ?1",
                [id],
                Self::row_to_conn,
            )
            .optional()
            .ok()
            .flatten()
    }

    pub fn insert_connection(&self, c: &ProviderConn) -> Result<()> {
        self.lock().execute(
            "INSERT INTO connections(id, provider, name, prefix, api_key, base_url, priority, enabled, models, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                c.id,
                c.provider,
                c.name,
                c.prefix,
                c.api_key,
                c.base_url,
                c.priority,
                c.enabled as i64,
                serde_json::to_string(&c.models)?,
                c.created_at
            ],
        )?;
        Ok(())
    }

    pub fn update_connection(&self, c: &ProviderConn) -> Result<()> {
        self.lock().execute(
            "UPDATE connections SET name = ?2, prefix = ?3, api_key = ?4, base_url = ?5, priority = ?6,
             enabled = ?7, models = ?8 WHERE id = ?1",
            params![
                c.id,
                c.name,
                c.prefix,
                c.api_key,
                c.base_url,
                c.priority,
                c.enabled as i64,
                serde_json::to_string(&c.models)?
            ],
        )?;
        Ok(())
    }

    pub fn delete_connection(&self, id: &str) -> Result<()> {
        self.lock()
            .execute("DELETE FROM connections WHERE id = ?1", [id])?;
        Ok(())
    }

    // ---------- combos ----------

    pub fn list_combos(&self) -> Vec<Combo> {
        let conn = self.lock();
        let Ok(mut stmt) =
            conn.prepare("SELECT name, models, created_at FROM combos ORDER BY name")
        else {
            return vec![];
        };
        stmt.query_map([], |r| {
            let models: String = r.get(1)?;
            Ok(Combo {
                name: r.get(0)?,
                models: serde_json::from_str(&models).unwrap_or_default(),
                created_at: r.get(2)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn get_combo(&self, name: &str) -> Option<Combo> {
        self.lock()
            .query_row(
                "SELECT name, models, created_at FROM combos WHERE name = ?1",
                [name],
                |r| {
                    let models: String = r.get(1)?;
                    Ok(Combo {
                        name: r.get(0)?,
                        models: serde_json::from_str(&models).unwrap_or_default(),
                        created_at: r.get(2)?,
                    })
                },
            )
            .optional()
            .ok()
            .flatten()
    }

    pub fn upsert_combo(&self, name: &str, models: &[String]) -> Result<()> {
        self.lock().execute(
            "INSERT INTO combos(name, models, created_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET models = excluded.models",
            params![name, serde_json::to_string(models)?, now()],
        )?;
        Ok(())
    }

    pub fn delete_combo(&self, name: &str) -> Result<()> {
        self.lock()
            .execute("DELETE FROM combos WHERE name = ?1", [name])?;
        Ok(())
    }

    // ---------- API keys ----------

    pub fn list_api_keys(&self) -> Vec<ApiKey> {
        let conn = self.lock();
        let Ok(mut stmt) = conn.prepare(
            "SELECT id, name, key, created_at, last_used FROM api_keys ORDER BY created_at DESC",
        ) else {
            return vec![];
        };
        stmt.query_map([], |r| {
            Ok(ApiKey {
                id: r.get(0)?,
                name: r.get(1)?,
                key: r.get(2)?,
                created_at: r.get(3)?,
                last_used: r.get(4)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn insert_api_key(&self, k: &ApiKey) -> Result<()> {
        self.lock().execute(
            "INSERT INTO api_keys(id, name, key, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![k.id, k.name, k.key, k.created_at],
        )?;
        Ok(())
    }

    pub fn delete_api_key(&self, id: &str) -> Result<()> {
        self.lock()
            .execute("DELETE FROM api_keys WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Returns the key's name when valid, and stamps its last-used time.
    pub fn check_api_key(&self, key: &str) -> Option<String> {
        let conn = self.lock();
        let name: Option<String> = conn
            .query_row("SELECT name FROM api_keys WHERE key = ?1", [key], |r| {
                r.get(0)
            })
            .optional()
            .ok()
            .flatten();
        if name.is_some() {
            let _ = conn.execute(
                "UPDATE api_keys SET last_used = ?2 WHERE key = ?1",
                params![key, now()],
            );
        }
        name
    }

    pub fn has_api_keys(&self) -> bool {
        self.lock()
            .query_row("SELECT COUNT(*) FROM api_keys", [], |r| r.get::<_, i64>(0))
            .map(|n| n > 0)
            .unwrap_or(false)
    }

    // ---------- usage ----------

    pub fn insert_usage(&self, u: &UsageRecord) {
        let res = self.lock().execute(
            "INSERT INTO usage(ts, api_key, requested_model, provider, connection, model, prompt_tokens,
             completion_tokens, status, latency_ms, cost, stream, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                u.ts,
                u.api_key,
                u.requested_model,
                u.provider,
                u.connection,
                u.model,
                u.prompt_tokens,
                u.completion_tokens,
                u.status,
                u.latency_ms,
                u.cost,
                u.stream as i64,
                u.error
            ],
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
                |r| {
                    Ok(Totals {
                        requests: r.get(0)?,
                        ok: r.get(1)?,
                        prompt_tokens: r.get(2)?,
                        completion_tokens: r.get(3)?,
                        cost: r.get(4)?,
                    })
                },
            )
            .unwrap_or_default()
    }

    /// Aggregates usage grouped by `model` or `provider` column.
    pub fn group_since(&self, column: &str, since: i64) -> Vec<GroupRow> {
        let column = match column {
            "provider" => "provider",
            "api_key" => "COALESCE(api_key, '(none)')",
            _ => "model",
        };
        let sql = format!(
            "SELECT {column} AS k, COUNT(*), COALESCE(SUM(status BETWEEN 200 AND 299), 0), SUM(prompt_tokens),
             SUM(completion_tokens), SUM(cost), AVG(latency_ms)
             FROM usage WHERE ts >= ?1 GROUP BY k ORDER BY COUNT(*) DESC LIMIT 50"
        );
        let conn = self.lock();
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return vec![];
        };
        stmt.query_map([since], |r| {
            Ok(GroupRow {
                key: r.get(0)?,
                requests: r.get(1)?,
                ok: r.get(2)?,
                prompt_tokens: r.get(3)?,
                completion_tokens: r.get(4)?,
                cost: r.get(5)?,
                avg_latency: r.get(6)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Per-bucket (requests, tokens) for the chart, `buckets` buckets of `size` seconds ending now.
    pub fn histogram(&self, buckets: i64, size: i64) -> Vec<(i64, i64, i64)> {
        let end = (now() / size + 1) * size;
        let start = end - buckets * size;
        let mut out: Vec<(i64, i64, i64)> =
            (0..buckets).map(|i| (start + i * size, 0, 0)).collect();
        let conn = self.lock();
        let Ok(mut stmt) = conn.prepare(
            "SELECT (ts - ?1) / ?2 AS b, COUNT(*), SUM(prompt_tokens + completion_tokens)
             FROM usage WHERE ts >= ?1 GROUP BY b",
        ) else {
            return out;
        };
        let rows = stmt.query_map(params![start, size], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        });
        if let Ok(rows) = rows {
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
        let conn = self.lock();
        let Ok(mut stmt) = conn.prepare(
            "SELECT ts, api_key, requested_model, provider, connection, model, prompt_tokens, completion_tokens,
             status, latency_ms, cost, stream, error FROM usage ORDER BY id DESC LIMIT ?1 OFFSET ?2",
        ) else {
            return vec![];
        };
        stmt.query_map(params![limit, offset], |r| {
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
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn usage_count(&self) -> i64 {
        self.lock()
            .query_row("SELECT COUNT(*) FROM usage", [], |r| r.get(0))
            .unwrap_or(0)
    }

    pub fn clear_usage(&self) -> Result<()> {
        self.lock().execute("DELETE FROM usage", [])?;
        Ok(())
    }
}
