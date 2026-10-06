//! Live dashboard updates over Server-Sent Events.
//!
//! Pages mark regions with `data-region`. While a page is open, this stream
//! re-renders those regions on the server whenever the database changes (and
//! every few seconds so relative times stay fresh) and sends only the regions
//! whose HTML changed. The client swaps them in place.

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use serde::Deserialize;
use serde_json::json;

use crate::AppState;

#[derive(Deserialize, Default, Clone)]
pub struct LiveQuery {
    view: String,
    range: Option<String>,
    page: Option<i64>,
    id: Option<String>,
}

const TICK: Duration = Duration::from_millis(800);
const REFRESH: Duration = Duration::from_secs(10);

/// Current HTML of every live region of a view. A region named `_reload`
/// carries a fingerprint: when it changes the page structure changed and the
/// client reloads instead.
pub async fn regions(st: &AppState, q: &LiveQuery, base: &str) -> Vec<(String, String)> {
    let db = &st.db;
    let mut out: Vec<(String, String)> = vec![];
    match q.view.as_str() {
        "overview" => {
            out.push(("stats".into(), super::overview::stats(db).into_string()));
            out.push(("chart".into(), super::overview::chart(db).into_string()));
            out.push(("recent".into(), super::overview::recent(db, base).into_string()));
            out.push(("health".into(), super::overview::health(db).into_string()));
        }
        "usage" => {
            let r = super::usage::range(q.range.as_deref());
            let page = q.page.unwrap_or(0).max(0);
            out.push(("stats".into(), super::usage::stats(db, &r).into_string()));
            out.push(("chart".into(), super::usage::chart(db, &r).into_string()));
            out.push(("groups".into(), super::usage::groups(db, &r).into_string()));
            out.push(("log".into(), super::usage::log(db, &r, page).into_string()));
        }
        "providers" => {
            let counts = super::providers::connection_counts(db);
            for e in &crate::registry::REG.entries {
                if let Some(id) = e["id"].as_str() {
                    out.push((format!("pc-{id}"), super::providers::card_badge(id, &counts).into_string()));
                }
            }
            for n in db.list_nodes(None) {
                if let Some(id) = n["id"].as_str() {
                    out.push((format!("pc-{id}"), super::providers::card_badge(id, &counts).into_string()));
                }
            }
            let mut nodes: Vec<String> = db.list_nodes(None).iter().filter_map(|n| n["id"].as_str().map(str::to_owned)).collect();
            nodes.sort();
            out.push(("_reload".into(), nodes.join(",")));
        }
        "provider" => {
            let id = q.id.clone().unwrap_or_default();
            let conns = db.connections_for(&id, false);
            out.push(("conn-count".into(), format!("{} configured", conns.len())));
            let mut ids = vec![];
            for c in &conns {
                let cid = c["id"].as_str().unwrap_or("").to_string();
                out.push((format!("cs-{cid}"), super::providers::status_cell(c).into_string()));
                ids.push(cid);
            }
            out.push(("_reload".into(), ids.join(",")));
        }
        "keys" => {
            let keys = db.list_api_keys();
            out.push(("keys".into(), super::keys::keys_table(&keys).into_string()));
        }
        _ => {}
    }
    out
}

pub async fn stream(State(st): State<AppState>, headers: HeaderMap, Query(q): Query<LiveQuery>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let base = super::base_url(&headers);
    let s = async_stream::stream! {
        let mut sent: HashMap<String, String> = regions(&st, &q, &base).await.into_iter().collect();
        let mut version = st.db.change_counter();
        let mut last = Instant::now();
        loop {
            tokio::time::sleep(TICK).await;
            let v = st.db.change_counter();
            if v == version && last.elapsed() < REFRESH {
                continue;
            }
            version = v;
            last = Instant::now();
            for (name, html) in regions(&st, &q, &base).await {
                if sent.get(&name) == Some(&html) {
                    continue;
                }
                if name == "_reload" {
                    yield Ok(Event::default().event("reload").data("1"));
                } else {
                    yield Ok(Event::default().event("region").data(json!({"r": name, "h": html}).to_string()));
                }
                sent.insert(name, html);
            }
        }
    };
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}
