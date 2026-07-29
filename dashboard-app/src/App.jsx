import { useEffect, useMemo, useState, Fragment } from "react";
import axios from "axios";
import exportedData from "./data/alerts.json";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = API.replace("http", "ws") + "/ws/alerts";

function riskColor(score) {
  if (score >= 0.97) return "var(--risk-critical)";
  if (score >= 0.9) return "var(--risk-medium)";
  return "var(--risk-low)";
}

function useLiveData() {
  // Priority: live FastAPI backend (if running) > exported JSON from your real
  // pipeline (alerts.json, written by export_alerts_json.py) > empty state.
  const [summary, setSummary] = useState(exportedData.summary);
  const [alerts, setAlerts] = useState(exportedData.alerts);
  const [entityHistory, setEntityHistory] = useState(exportedData.entity_history || {});
  const [source, setSource] = useState(
    exportedData.alerts && exportedData.alerts.length ? "exported" : "empty"
  );
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    axios
      .get(`${API}/api/summary`, { timeout: 1500 })
      .then((s) => axios.get(`${API}/api/alerts?limit=200`, { timeout: 1500 }).then((a) => [s, a]))
      .then(([s, a]) => {
        if (cancelled) return;
        setSummary(s.data);
        setAlerts(a.data);
        setSource("live");
      })
      .catch(() => {
        // stay on whatever exportedData already gave us
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (source !== "live") return;
    let ws;
    try {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
      ws.onmessage = (event) => {
        const row = JSON.parse(event.data);
        setAlerts((prev) => [row, ...prev].slice(0, 200));
      };
    } catch {
      setWsConnected(false);
    }
    return () => ws && ws.close();
  }, [source]);

  return { summary, alerts, entityHistory, source, wsConnected };
}

export default function App() {
  const { summary, alerts, entityHistory, source, wsConnected } = useLiveData();
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [expandedId, setExpandedId] = useState(null);

  const predictedTypes = useMemo(
    () => ["all", ...new Set(alerts.map((a) => a.predicted_attack).filter(Boolean))],
    [alerts]
  );

  const filtered = useMemo(() => {
    return alerts.filter((a) => {
      const matchesSearch = search === "" || (a.entity_id || "").toLowerCase().includes(search.toLowerCase());
      const matchesType = typeFilter === "all" || a.predicted_attack === typeFilter;
      return matchesSearch && matchesType;
    });
  }, [alerts, search, typeFilter]);

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Behavioural anomaly detection - analyst alert queue</h1>
          <p className="subtitle">Ranked by risk score, within the top 1% alert budget. Click a row for contributing factors and entity history.</p>
        </div>
        <div className="live-indicator">
          <span className={`pulse-dot ${source === "live" && wsConnected ? "" : "disconnected"}`} />
          {source === "live"
            ? wsConnected
              ? "Live feed connected"
              : "Connecting…"
            : source === "exported"
            ? "Showing exported pipeline data"
            : "No data yet — run export_alerts_json.py"}
        </div>
      </div>

      {source !== "live" && (
        <div className="mock-banner">
          {source === "exported"
            ? "Showing data from src/data/alerts.json (run export_alerts_json.py again to refresh it). Once the FastAPI backend is running, this switches to live data automatically."
            : "src/data/alerts.json is empty. Run your pipeline through generate_dashboard.py, then `python export_alerts_json.py` from the project root to populate it."}
        </div>
      )}

      <div className="summary-grid">
        <div className="card">
          <p className="value">{summary.total_sessions.toLocaleString()}</p>
          <p className="label">Total sessions scored</p>
        </div>
        <div className="card">
          <p className="value">{summary.alert_budget.toLocaleString()}</p>
          <p className="label">Alert budget (top 1%)</p>
        </div>
        <div className="card">
          <p className="value">{Math.round((summary.precision_in_queue || 0) * 1000) / 10}%</p>
          <p className="label">Precision within queue</p>
        </div>
        <div className="card">
          <p className="value">{summary.true_anomalies_total.toLocaleString()}</p>
          <p className="label">True anomalies (all data)</p>
        </div>
        <div className="card">
          <p className="value">{summary.entities_monitored}</p>
          <p className="label">Entities monitored</p>
        </div>
      </div>

      <div className="filter-bar">
        <input
          className="search-input"
          placeholder="Search entity id..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select className="type-select" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
          {predictedTypes.map((t) => (
            <option key={t} value={t}>
              {t === "all" ? "All predicted types" : t}
            </option>
          ))}
        </select>
      </div>

      <div className="table-panel">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Entity</th>
              <th>Type</th>
              <th>Risk</th>
              <th>Detection path</th>
              <th>Predicted attack</th>
              <th>Confidence</th>
              <th>Ground truth</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((a) => {
              const isOpen = expandedId === a.session_id;
              const color = riskColor(a.risk_score);
              const history = entityHistory[a.entity_id] || [];
              return (
                <Fragment key={a.session_id}>
                  <tr
                    className={isOpen ? "row-open" : ""}
                    onClick={() => setExpandedId(isOpen ? null : a.session_id)}
                  >
                    <td className="rank-cell">{a.rank}</td>
                    <td className="entity-id">{a.entity_id}</td>
                    <td className="entity-type">{a.entity_type}</td>
                    <td>
                      <div className="risk-bar-wrap">
                        <div className="risk-bar-track">
                          <div className="risk-bar-fill" style={{ width: `${a.risk_score * 100}%`, background: color }} />
                        </div>
                        <span className="risk-label" style={{ color }}>
                          {a.risk_score.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="entity-type">{a.detection_method}</td>
                    <td>{a.predicted_attack || "—"}</td>
                    <td className="entity-type">
                      {a.classifier_confidence != null ? `${Math.round(a.classifier_confidence * 100)}%` : "—"}
                    </td>
                    <td>
                      <span className={`gt-badge gt-${a.ground_truth || "unknown"}`}>{a.ground_truth || "—"}</span>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="detail-row">
                      <td colSpan={8}>
                        <div className="detail-grid">
                          <div>
                            <p className="detail-heading">Why this was flagged</p>
                            <div className="reason-box">{a.reason_string || "No explanation recorded for this session."}</div>
                            <p className="detail-heading">Top contributing features</p>
                            <ul className="feature-list">
                              {(a.top_features || []).length === 0 && <li className="entity-type">No feature attribution recorded.</li>}
                              {(a.top_features || []).map((f, i) => (
                                <li key={i}>
                                  <span>{f.feature}</span>
                                  <span className="z-score">z = {typeof f.z_score === "number" ? f.z_score.toFixed(2) : f.z_score}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="detail-heading">Recent entity history</p>
                            <ul className="history-list">
                              {history.length === 0 && <li>No history exported for this entity yet.</li>}
                              {history.map((h, i) => (
                                <li key={i}>
                                  <strong>{h.ts}</strong> — {h.geo_location}, {h.resource_accessed}, auth={h.auth_method}, dur={h.session_duration_sec}s
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={8} className="empty-cell">
                  No alerts match your search/filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
