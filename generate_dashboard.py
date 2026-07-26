
#Phase 6: Analyst Dashboard


import json
import pandas as pd
import numpy as np

ALERT_BUDGET_PCT = 0.01   # top 1% of events, matching the assessment's evaluation criteria


def build_dashboard():
    alerts = pd.read_csv("data/alerts_explained.csv")
    raw_logs = pd.read_csv("data/access_logs_unlabeled.csv")

    alerts["top_contributing_features"] = alerts["top_contributing_features"].apply(
        lambda s: eval(s) if isinstance(s, str) and s.strip() else {}
    )

    n_total = len(alerts)
    k = max(1, int(n_total * ALERT_BUDGET_PCT))
    top_alerts = alerts.sort_values("risk_score", ascending=False).head(k).copy()

    n_true_anomalies_in_queue = int((top_alerts["label"] == "anomaly").sum())
    precision = n_true_anomalies_in_queue / len(top_alerts) if len(top_alerts) else 0

    entity_ids_in_queue = top_alerts["entity_id"].unique().tolist()
    history = {}
    for eid in entity_ids_in_queue:
        rows = raw_logs[raw_logs["entity_id"] == eid].sort_values("timestamp").tail(8)
        history[eid] = rows[["timestamp", "source_ip", "geo_location", "resource_accessed",
                              "auth_method", "session_duration_sec", "device_fingerprint"]].to_dict("records")

    alert_records = top_alerts[[
        "session_id", "entity_id", "entity_type", "risk_score", "detection_method",
        "predicted_type", "predicted_confidence", "explanation", "top_contributing_features",
        "label", "anomaly_type",
    ]].to_dict("records")

    summary = {
        "total_sessions": int(n_total),
        "alert_budget": int(k),
        "alert_budget_pct": ALERT_BUDGET_PCT * 100,
        "precision_in_queue": round(precision * 100, 1),
        "true_anomalies_total": int((alerts["label"] == "anomaly").sum()),
        "entities_monitored": int(alerts["entity_id"].nunique()),
    }

    data_json = json.dumps({
        "summary": summary,
        "alerts": alert_records,
        "history": history,
    }, default=str)

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    with open("dashboard/index.html", "w") as f:
        f.write(html)
    print(f"Saved dashboard/index.html  ({len(alert_records)} alerts, "
          f"precision@top{ALERT_BUDGET_PCT*100:.0f}% = {precision*100:.1f}%)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Behavioural Anomaly Detection - Analyst Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --surface: #171a21; --surface2: #1f232c; --border: #2a2f3a;
    --text: #e6e8ec; --text-secondary: #9aa1ad; --text-muted: #6b7280;
    --accent: #4f8cff; --danger: #e5484d; --warning: #f5a524; --success: #17c964;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 32px; line-height: 1.5;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  .subtitle { color: var(--text-secondary); font-size: 13px; margin-bottom: 24px; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 28px;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px;
  }
  .stat-value { font-size: 22px; font-weight: 600; }
  .stat-label { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
  .toolbar { display: flex; gap: 10px; margin-bottom: 12px; align-items: center; }
  input#search {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 13px; width: 260px;
  }
  select#typeFilter {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 13px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    text-align: left; padding: 10px 12px; color: var(--text-secondary);
    font-weight: 500; border-bottom: 1px solid var(--border); font-size: 12px;
  }
  tbody tr {
    border-bottom: 1px solid var(--border); cursor: pointer;
  }
  tbody tr:hover { background: var(--surface2); }
  tbody td { padding: 10px 12px; vertical-align: middle; }
  .risk-bar-bg {
    width: 80px; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden;
  }
  .risk-bar-fill { height: 100%; border-radius: 3px; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500;
  }
  .badge-anomaly { background: rgba(229,72,77,0.15); color: var(--danger); }
  .badge-normal { background: rgba(23,201,100,0.15); color: var(--success); }
  .badge-edge { background: rgba(245,165,36,0.15); color: var(--warning); }
  .detail-row td { background: var(--surface); padding: 16px 20px; }
  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .detail-section h4 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--text-secondary); margin: 0 0 8px;
  }
  .explanation-box {
    background: var(--surface2); border-left: 3px solid var(--accent);
    padding: 10px 14px; border-radius: 4px; font-size: 13px; margin-bottom: 12px;
  }
  .contrib-item {
    display: flex; justify-content: space-between; font-size: 12px;
    padding: 4px 0; border-bottom: 1px solid var(--border);
  }
  .history-item {
    font-size: 12px; padding: 6px 0; border-bottom: 1px solid var(--border); color: var(--text-secondary);
  }
  .history-item .ts { color: var(--text); font-weight: 500; }
  .hidden { display: none; }
</style>
</head>
<body>

<h1>Behavioural anomaly detection - analyst alert queue</h1>
<div class="subtitle">Ranked by risk score, within the top 1% alert budget. Click a row for contributing factors and entity history.</div>

<div class="summary-grid" id="summaryGrid"></div>

<div class="toolbar">
  <input id="search" type="text" placeholder="Search entity id...">
  <select id="typeFilter"><option value="">All predicted types</option></select>
</div>

<table>
  <thead>
    <tr>
      <th>Rank</th><th>Entity</th><th>Type</th><th>Risk</th><th>Detection path</th>
      <th>Predicted attack</th><th>Confidence</th><th>Ground truth</th>
    </tr>
  </thead>
  <tbody id="alertBody"></tbody>
</table>

<script>
const DATA = __DATA_JSON__;

function riskColor(score) {
  if (score > 0.85) return "#e5484d";
  if (score > 0.6) return "#f5a524";
  return "#4f8cff";
}

function badgeFor(label) {
  if (label === "anomaly") return '<span class="badge badge-anomaly">anomaly</span>';
  if (label === "edge_case") return '<span class="badge badge-edge">edge case</span>';
  return '<span class="badge badge-normal">normal</span>';
}

function methodLabel(method) {
  if (method === "sequence_lstm") return "Sequence (LSTM)";
  if (method === "cold_start_isolation_forest") return "Cold-start (Isolation Forest)";
  return method || "-";
}

function renderSummary() {
  const s = DATA.summary;
  const cards = [
    [s.total_sessions.toLocaleString(), "Total sessions scored"],
    [s.alert_budget.toLocaleString(), `Alert budget (top ${s.alert_budget_pct}%)`],
    [s.precision_in_queue + "%", "Precision within queue"],
    [s.true_anomalies_total.toLocaleString(), "True anomalies (all data)"],
    [s.entities_monitored.toLocaleString(), "Entities monitored"],
  ];
  document.getElementById("summaryGrid").innerHTML = cards.map(([v, l]) =>
    `<div class="stat-card"><div class="stat-value">${v}</div><div class="stat-label">${l}</div></div>`
  ).join("");
}

function populateTypeFilter() {
  const types = [...new Set(DATA.alerts.map(a => a.predicted_type))].sort();
  const sel = document.getElementById("typeFilter");
  types.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  });
}

function renderHistory(entityId) {
  const hist = DATA.history[entityId] || [];
  if (hist.length === 0) return '<div class="history-item">No recent history available.</div>';
  return hist.slice().reverse().map(h =>
    `<div class="history-item"><span class="ts">${h.timestamp}</span> - ${h.geo_location}, ${h.resource_accessed}, auth=${h.auth_method}, dur=${h.session_duration_sec}s</div>`
  ).join("");
}

function renderContribs(contribs) {
  const entries = Object.entries(contribs || {});
  if (entries.length === 0) return '<div class="history-item">No strong single-feature driver.</div>';
  return entries.map(([f, z]) =>
    `<div class="contrib-item"><span>${f}</span><span>z = ${z}</span></div>`
  ).join("");
}

function renderTable(filterText = "", filterType = "") {
  const tbody = document.getElementById("alertBody");
  const rows = DATA.alerts.filter(a =>
    (!filterText || a.entity_id.toLowerCase().includes(filterText.toLowerCase())) &&
    (!filterType || a.predicted_type === filterType)
  );

  tbody.innerHTML = "";
  rows.forEach((a, i) => {
    const tr = document.createElement("tr");
    const pct = Math.round(a.risk_score * 100);
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${a.entity_id}</td>
      <td>${a.entity_type}</td>
      <td>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${pct}%;background:${riskColor(a.risk_score)}"></div></div>
        <span style="font-size:11px;color:var(--text-secondary)">${a.risk_score.toFixed(2)}</span>
      </td>
      <td style="font-size:12px;color:var(--text-secondary)">${methodLabel(a.detection_method)}</td>
      <td>${a.predicted_type}</td>
      <td>${(a.predicted_confidence * 100).toFixed(0)}%</td>
      <td>${badgeFor(a.label)}</td>
    `;
    tr.addEventListener("click", () => toggleDetail(tr, a));
    tbody.appendChild(tr);
  });
}

function toggleDetail(tr, alert) {
  const next = tr.nextSibling;
  if (next && next.classList && next.classList.contains("detail-row")) {
    next.remove();
    return;
  }
  document.querySelectorAll(".detail-row").forEach(r => r.remove());

  const detailTr = document.createElement("tr");
  detailTr.className = "detail-row";
  detailTr.innerHTML = `
    <td colspan="8">
      <div class="detail-grid">
        <div class="detail-section">
          <h4>Why this was flagged</h4>
          <div class="explanation-box">${alert.explanation}</div>
          <h4>Top contributing features</h4>
          ${renderContribs(alert.top_contributing_features)}
        </div>
        <div class="detail-section">
          <h4>Recent entity history</h4>
          ${renderHistory(alert.entity_id)}
        </div>
      </div>
    </td>
  `;
  tr.parentNode.insertBefore(detailTr, tr.nextSibling);
}

renderSummary();
populateTypeFilter();
renderTable();

document.getElementById("search").addEventListener("input", e => {
  renderTable(e.target.value, document.getElementById("typeFilter").value);
});
document.getElementById("typeFilter").addEventListener("change", e => {
  renderTable(document.getElementById("search").value, e.target.value);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build_dashboard()