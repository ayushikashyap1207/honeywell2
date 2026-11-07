import { useEffect, useState } from "react";
import axios from "axios";

import AlertsTab from "./components/AlertsTab";
import CopilotTab from "./components/CopilotTab";
import DriftTab from "./components/DriftTab";
import EvaluationTab from "./components/EvaluationTab";
import exportedData from "./data/alerts.json";
import evaluationData from "./data/evaluation.json";
import driftData from "./data/drift.json";
import shapData from "./data/shap.json";

// Empty base in production -> same-origin requests proxied by nginx.
// In `npm run dev`, default to the local FastAPI server.
const API_BASE =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");
const API = API_BASE.replace(/\/$/, "");
const WS_URL = API
  ? `${API.replace(/^http/, "ws")}/ws/alerts`
  : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/alerts`;

const EMPTY_SUMMARY = {
  total_sessions: 0,
  alert_budget: 0,
  precision_in_queue: 0,
  true_anomalies_total: 0,
  entities_monitored: 0,
  feedback_pending: 0,
};

function useLiveData() {
  const [summary, setSummary] = useState(exportedData.summary || EMPTY_SUMMARY);
  const [alerts, setAlerts] = useState(exportedData.alerts || []);
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
        setSummary({ ...EMPTY_SUMMARY, ...s.data });
        setAlerts(a.data);
        setSource("live");
      })
      .catch(() => {});
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
        setAlerts((prev) => [{ ...row, rank: "live" }, ...prev].slice(0, 200));
      };
    } catch {
      setWsConnected(false);
    }
    return () => ws && ws.close();
  }, [source]);

  return { summary, alerts, entityHistory, source, wsConnected };
}

const TABS = [
  { id: "alerts", label: "Alert queue" },
  { id: "evaluation", label: "Model evaluation" },
  { id: "drift", label: "Drift" },
  { id: "copilot", label: "Copilot" },
];

export default function App() {
  const { summary, alerts, entityHistory, source, wsConnected } = useLiveData();
  const [tab, setTab] = useState("alerts");
  const [copilotContext, setCopilotContext] = useState(null);

  function askCopilot(alert) {
    setCopilotContext({
      session_id: alert.session_id,
      entity_id: alert.entity_id,
    });
    setTab("copilot");
  }

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Behavioural anomaly detection — analyst workspace</h1>
          <p className="subtitle">
            Ranked alerts, calibrated risk, investigation, model evaluation and
            drift in one place. Click an alert for evidence.
          </p>
        </div>
        <div className="live-indicator">
          <span
            className={`pulse-dot ${
              source === "live" && wsConnected ? "" : "disconnected"
            }`}
          />
          {source === "live"
            ? wsConnected
              ? "Live feed connected"
              : "Connecting…"
            : source === "exported"
            ? "Showing exported pipeline data"
            : "No data yet — run export_alerts_json.py"}
        </div>
      </div>

      <div className="tabs">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            className={`tab ${tab === entry.id ? "active" : ""}`}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
            {entry.id === "alerts" && alerts.length > 0 && (
              <span className="tab-count">{alerts.length}</span>
            )}
            {entry.id === "copilot" && copilotContext?.session_id && (
              <span className="tab-dot" />
            )}
          </button>
        ))}
      </div>

      {tab === "alerts" && (
        <>
          {source !== "live" && (
            <div className="mock-banner">
              {source === "exported"
                ? "Showing data from src/data/alerts.json. Start the FastAPI backend (make demo) to switch to live data and enable analyst feedback."
                : "src/data/alerts.json is empty. Run the pipeline and export_alerts_json.py first."}
            </div>
          )}

          <div className="summary-grid">
            <div className="card">
              <p className="value">{(summary.total_sessions || 0).toLocaleString()}</p>
              <p className="label">Total sessions scored</p>
            </div>
            <div className="card">
              <p className="value">{(summary.alert_budget || 0).toLocaleString()}</p>
              <p className="label">Alert budget (top 1%)</p>
            </div>
            <div className="card">
              <p className="value">
                {Math.round((summary.precision_in_queue || 0) * 1000) / 10}%
              </p>
              <p className="label">Precision within queue</p>
            </div>
            <div className="card">
              <p className="value">{(summary.true_anomalies_total || 0).toLocaleString()}</p>
              <p className="label">True anomalies (all data)</p>
            </div>
            <div className="card">
              <p className="value">{summary.entities_monitored || 0}</p>
              <p className="label">Entities monitored</p>
            </div>
            <div className="card">
              <p className="value">{summary.feedback_pending || 0}</p>
              <p className="label">Feedback awaiting retrain</p>
            </div>
          </div>

          <AlertsTab
            alerts={alerts}
            entityHistory={entityHistory}
            api={API}
            onAskCopilot={askCopilot}
          />
        </>
      )}

      {tab === "evaluation" && (
        <EvaluationTab api={API} fallback={evaluationData} shap={shapData} />
      )}

      {tab === "drift" && <DriftTab api={API} fallback={driftData} />}

      {tab === "copilot" && (
        <CopilotTab api={API} context={copilotContext} />
      )}
    </div>
  );
}
