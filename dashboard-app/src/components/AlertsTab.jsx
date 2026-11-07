import { Fragment, useMemo, useState } from "react";
import axios from "axios";

function riskColor(score) {
  if (score >= 0.97) return "var(--risk-critical)";
  if (score >= 0.9) return "var(--risk-medium)";
  return "var(--risk-low)";
}

const VERDICT_LABELS = {
  confirm: "Confirmed attack",
  false_positive: "False positive",
  escalate: "Escalated",
  acknowledge: "Acknowledged",
};

export default function AlertsTab({
  alerts,
  entityHistory,
  api,
  onAskCopilot,
}) {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [expandedId, setExpandedId] = useState(null);
  const [feedback, setFeedback] = useState({});
  const [busyId, setBusyId] = useState(null);

  const predictedTypes = useMemo(
    () => ["all", ...new Set(alerts.map((a) => a.predicted_attack).filter(Boolean))],
    [alerts]
  );

  const filtered = useMemo(
    () =>
      alerts.filter((a) => {
        const matchesSearch =
          search === "" ||
          (a.entity_id || "").toLowerCase().includes(search.toLowerCase()) ||
          (a.session_id || "").toLowerCase().includes(search.toLowerCase());
        const matchesType = typeFilter === "all" || a.predicted_attack === typeFilter;
        return matchesSearch && matchesType;
      }),
    [alerts, search, typeFilter]
  );

  async function sendFeedback(alert, verdict) {
    if (!api) {
      setFeedback((prev) => ({
        ...prev,
        [alert.session_id]: { error: "Start the live backend to submit feedback." },
      }));
      return;
    }
    setBusyId(alert.session_id);
    try {
      await axios.post(`${api}/api/feedback`, {
        session_id: alert.session_id,
        verdict,
        analyst: "dashboard",
      });
      setFeedback((prev) => ({
        ...prev,
        [alert.session_id]: { verdict },
      }));
    } catch (error) {
      setFeedback((prev) => ({
        ...prev,
        [alert.session_id]: {
          error: error?.response?.data?.detail || "Could not save feedback.",
        },
      }));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <>
      <div className="filter-bar">
        <input
          className="search-input"
          placeholder="Search entity or session id..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select
          className="type-select"
          value={typeFilter}
          onChange={(event) => setTypeFilter(event.target.value)}
        >
          {predictedTypes.map((type) => (
            <option key={type} value={type}>
              {type === "all" ? "All predicted types" : type}
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
              <th>Calibrated</th>
              <th>Detection path</th>
              <th>Predicted attack</th>
              <th>Confidence</th>
              <th>Ground truth</th>
              <th>Feedback</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((alert) => {
              const isOpen = expandedId === alert.session_id;
              const score = Number(alert.risk_score) || 0;
              const color = riskColor(score);
              const history = entityHistory[alert.entity_id] || [];
              const feedbackState = feedback[alert.session_id];
              const verdict = feedbackState?.verdict || alert.feedback_verdict;
              const details = alert.explanation_details || {};
              const counterfactual = details.counterfactual;
              return (
                <Fragment key={alert.session_id}>
                  <tr
                    className={isOpen ? "row-open" : ""}
                    onClick={() => setExpandedId(isOpen ? null : alert.session_id)}
                  >
                    <td className="rank-cell">{alert.rank}</td>
                    <td className="entity-id">{alert.entity_id}</td>
                    <td className="entity-type">{alert.entity_type}</td>
                    <td>
                      <div className="risk-bar-wrap">
                        <div className="risk-bar-track">
                          <div
                            className="risk-bar-fill"
                            style={{ width: `${score * 100}%`, background: color }}
                          />
                        </div>
                        <span className="risk-label" style={{ color }}>
                          {score.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="entity-type">
                      {alert.calibrated_risk != null
                        ? Number(alert.calibrated_risk).toFixed(3)
                        : "—"}
                    </td>
                    <td className="entity-type">{alert.detection_method}</td>
                    <td>{alert.predicted_attack || "—"}</td>
                    <td className="entity-type">
                      {alert.classifier_confidence != null
                        ? `${Math.round(alert.classifier_confidence * 100)}%`
                        : "—"}
                    </td>
                    <td>
                      <span className={`gt-badge gt-${alert.ground_truth || "unknown"}`}>
                        {alert.ground_truth || "—"}
                      </span>
                    </td>
                    <td>
                      {verdict ? (
                        <span className="badge badge-info">{VERDICT_LABELS[verdict] || verdict}</span>
                      ) : (
                        <span className="acked-tag">—</span>
                      )}
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="detail-row">
                      <td colSpan={10}>
                        <div className="detail-grid">
                          <div>
                            <p className="detail-heading">Why this was flagged</p>
                            <div className="reason-box">
                              {alert.reason_string || "No explanation recorded for this session."}
                            </div>

                            <div className="inline-stats">
                              <span>
                                raw risk <strong>{score.toFixed(3)}</strong>
                              </span>
                              <span>
                                calibrated{" "}
                                <strong>
                                  {alert.calibrated_risk != null
                                    ? Number(alert.calibrated_risk).toFixed(3)
                                    : "n/a"}
                                </strong>
                              </span>
                              <span>
                                conformal p{" "}
                                <strong>
                                  {alert.conformal_p_value != null
                                    ? Number(alert.conformal_p_value).toFixed(4)
                                    : "n/a"}
                                </strong>
                              </span>
                            </div>

                            {counterfactual?.text && (
                              <>
                                <p className="detail-heading">Counterfactual</p>
                                <div className="reason-box counterfactual-box">
                                  {counterfactual.text}
                                </div>
                              </>
                            )}

                            <p className="detail-heading">Top contributing features</p>
                            <ul className="feature-list">
                              {(alert.top_features || []).length === 0 && (
                                <li className="entity-type">No feature attribution recorded.</li>
                              )}
                              {(alert.top_features || []).map((feature, index) => (
                                <li key={index}>
                                  <span>{feature.feature}</span>
                                  <span className="z-score">
                                    z ={" "}
                                    {typeof feature.z_score === "number"
                                      ? feature.z_score.toFixed(2)
                                      : feature.z_score}
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div>
                            <p className="detail-heading">Analyst disposition</p>
                            <div className="feedback-row">
                              <button
                                className="feedback-btn feedback-confirm"
                                disabled={busyId === alert.session_id}
                                onClick={() => sendFeedback(alert, "confirm")}
                              >
                                Confirm attack
                              </button>
                              <button
                                className="feedback-btn feedback-false"
                                disabled={busyId === alert.session_id}
                                onClick={() => sendFeedback(alert, "false_positive")}
                              >
                                False positive
                              </button>
                              <button
                                className="feedback-btn feedback-escalate"
                                disabled={busyId === alert.session_id}
                                onClick={() => sendFeedback(alert, "escalate")}
                              >
                                Escalate
                              </button>
                            </div>
                            {feedbackState?.error && (
                              <p className="feedback-error">{feedbackState.error}</p>
                            )}
                            {verdict && !feedbackState?.error && (
                              <p className="feedback-ok">
                                Latest disposition: {VERDICT_LABELS[verdict] || verdict}
                              </p>
                            )}
                            <button
                              className="ack-btn copilot-ask-btn"
                              onClick={() => onAskCopilot?.(alert)}
                            >
                              Ask the copilot about this alert
                            </button>

                            <p className="detail-heading history-heading">Recent entity history</p>
                            <ul className="history-list">
                              {history.length === 0 && (
                                <li>No history exported for this entity yet.</li>
                              )}
                              {history.map((item, index) => (
                                <li key={index}>
                                  <strong>{item.ts || item.timestamp}</strong> —{" "}
                                  {item.geo_location}, {item.resource_accessed}, auth=
                                  {item.auth_method}, dur={item.session_duration_sec}s
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
                <td colSpan={10} className="empty-cell">
                  No alerts match your search/filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
