import { useEffect, useState } from "react";

import { BarChart, statusClass } from "./Charts";

const EMPTY = {
  available: false,
  overall_status: "unknown",
  max_feature_psi: 0,
  drifted_features: 0,
  warning_features: 0,
  features: [],
  score: null,
  reference_rows: 0,
  current_rows: 0,
};

export default function DriftTab({ api, fallback }) {
  const [report, setReport] = useState(fallback || EMPTY);

  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    fetch(`${api}/api/drift`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!cancelled && data && data.available !== false && data.features) {
          setReport(data);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [api]);

  const featureItems = (report.features || [])
    .slice(0, 10)
    .map((item) => ({ label: item.feature, value: item.psi, status: item.status }));

  return (
    <div className="tab-stack">
      <div className="metric-grid">
        <div className="card metric">
          <p className="label">Overall status</p>
          <p className="value">
            <span className={statusClass(report.overall_status)}>
              {report.overall_status}
            </span>
          </p>
          <p className="metric-note">
            PSI warning ≥ 0.10 · significant ≥ 0.25
          </p>
        </div>
        <div className="card metric">
          <p className="label">Max feature PSI</p>
          <p className="value">{report.max_feature_psi ?? "—"}</p>
          <p className="metric-note">population stability index</p>
        </div>
        <div className="card metric">
          <p className="label">Drifted features</p>
          <p className="value">{report.drifted_features ?? 0}</p>
          <p className="metric-note">{report.warning_features ?? 0} in warning band</p>
        </div>
        <div className="card metric">
          <p className="label">Comparison window</p>
          <p className="value">
            {report.reference_rows?.toLocaleString?.() ?? report.reference_rows} /{" "}
            {report.current_rows?.toLocaleString?.() ?? report.current_rows}
          </p>
          <p className="metric-note">reference / current rows</p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h2>Feature drift</h2>
            <p>
              Older reference window vs the most recent window. PSI measures
              distribution shift; KS adds a two-sample test.
            </p>
          </div>
          <span className="badge badge-info">
            {report.generated_at ? report.generated_at.slice(0, 10) : "static report"}
          </span>
        </div>
        <BarChart
          items={featureItems}
          color="var(--risk-medium)"
          maxValue={0.5}
          formatValue={(value) => Number(value).toFixed(3)}
        />
      </div>

      <div className="panel">
        <div className="panel-header compact">
          <h2>Feature table</h2>
        </div>
        <div className="table-panel flat">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>PSI</th>
                <th>KS statistic</th>
                <th>KS p-value</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(report.features || []).map((item) => (
                <tr key={item.feature}>
                  <td className="entity-id">{item.feature}</td>
                  <td>{item.psi}</td>
                  <td>{item.ks_statistic}</td>
                  <td>{item.ks_p_value}</td>
                  <td>
                    <span className={statusClass(item.status)}>{item.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {report.score && (
        <div className="panel">
          <div className="panel-header compact">
            <h2>Score distribution drift</h2>
            <span className={statusClass(report.score.status)}>
              {report.score.status} · PSI {report.score.psi}
            </span>
          </div>
          <p className="panel-footnote">
            If the risk-score distribution shifts significantly, retraining or
            recalibration should be scheduled.
          </p>
        </div>
      )}
    </div>
  );
}
