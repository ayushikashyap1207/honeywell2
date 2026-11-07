import { useEffect, useMemo, useState } from "react";

import { BarChart, LineChart, PALETTE } from "./Charts";

const EMPTY = {
  available: false,
  dataset: {},
  models: [],
  curves: {},
  per_attack_recall: {},
  cold_start: {},
  calibration: null,
  feature_importance: [],
};

const PREFERRED_MODELS = [
  "Hybrid IF+LSTM (pipeline)",
  "Hybrid + isotonic calibration",
  "One-Class SVM",
  "Random forest (supervised upper bound)",
  "PCA reconstruction",
];

function modelColor(name, index) {
  const fixed = {
    "Hybrid IF+LSTM (pipeline)": "#2dd4bf",
    "Hybrid + isotonic calibration": "#34d399",
    "One-Class SVM": "#f5a623",
    "Random forest (supervised upper bound)": "#8b5cf6",
    "PCA reconstruction": "#60a5fa",
  };
  return fixed[name] || PALETTE[index % PALETTE.length];
}

export default function EvaluationTab({ api, fallback, shap }) {
  const [report, setReport] = useState(fallback || EMPTY);
  const [selectedModel, setSelectedModel] = useState(
    "Hybrid IF+LSTM (pipeline)"
  );

  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    fetch(`${api}/api/evaluation`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!cancelled && data && data.available !== false && data.models) {
          setReport(data);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [api]);

  const pipeline = useMemo(
    () =>
      report.models?.find((item) => item.name === "Hybrid IF+LSTM (pipeline)") ||
      report.models?.[0] ||
      {},
    [report]
  );

  const prSeries = useMemo(() => {
    const names = PREFERRED_MODELS.filter((name) => report.curves?.[name]);
    return names.map((name, index) => ({
      name,
      color: modelColor(name, index),
      points: report.curves[name].pr || [],
    }));
  }, [report]);

  const calibrationSeries = useMemo(() => {
    if (!report.calibration?.bins?.length) return [];
    return [
      {
        name: "Calibrated model",
        color: "#2dd4bf",
        points: report.calibration.bins.map((bin) => [
          bin.mean_predicted,
          bin.fraction_positive,
        ]),
      },
      {
        name: "Perfect calibration",
        color: "#5b6478",
        points: [
          [0, 0],
          [1, 1],
        ],
      },
    ];
  }, [report]);

  const shapItems = useMemo(() => {
    const features = shap?.features || report.feature_importance || [];
    return features.slice(0, 10).map((item) => ({
      label: item.feature,
      value: item.importance,
    }));
  }, [shap, report]);

  const attackItems = useMemo(() => {
    const values = report.per_attack_recall?.[selectedModel] || {};
    return Object.entries(values)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value);
  }, [report, selectedModel]);

  const coldStart = report.cold_start?.[selectedModel] || {};
  const modelNames = report.models?.map((item) => item.name) || [];

  return (
    <div className="tab-stack">
      <div className="metric-grid">
        <div className="card metric">
          <p className="label">Test rows</p>
          <p className="value">{(report.dataset?.test_rows || 0).toLocaleString()}</p>
          <p className="metric-note">
            {report.dataset?.test_anomalies || 0} anomalies ·{" "}
            {report.dataset?.cold_start_test_rows || 0} cold-start
          </p>
        </div>
        <div className="card metric">
          <p className="label">Pipeline PR-AUC</p>
          <p className="value">{pipeline.pr_auc ?? "—"}</p>
          <p className="metric-note">top 1% alert budget</p>
        </div>
        <div className="card metric">
          <p className="label">Precision @ top 1%</p>
          <p className="value">{pipeline.precision_at_1 ?? "—"}</p>
          <p className="metric-note">
            recall {pipeline.recall_at_1 ?? "—"} · FPR {pipeline.fpr_at_1 ?? "—"}
          </p>
        </div>
        <div className="card metric">
          <p className="label">Calibration</p>
          <p className="value">
            {report.calibration?.ece != null ? report.calibration.ece : "—"}
          </p>
          <p className="metric-note">
            ECE · Brier {report.calibration?.brier ?? "—"}
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h2>Baseline comparison</h2>
            <p>
              Unsupervised baselines train on normal rows only. Supervised models
              are shown as an upper bound, not as the deployed detector.
            </p>
          </div>
          <span className="badge badge-info">
            {report.generated_at ? report.generated_at.slice(0, 10) : "static report"}
          </span>
        </div>
        <div className="table-panel flat">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Kind</th>
                <th>PR-AUC</th>
                <th>ROC-AUC</th>
                <th>Precision@1%</th>
                <th>Recall@1%</th>
                <th>FPR@1%</th>
              </tr>
            </thead>
            <tbody>
              {(report.models || []).map((model) => (
                <tr
                  key={model.name}
                  className={model.name === selectedModel ? "row-open" : ""}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <td className="entity-id">{model.name}</td>
                  <td>
                    <span className={`badge badge-${model.kind}`}>
                      {model.kind}
                    </span>
                  </td>
                  <td>{model.pr_auc}</td>
                  <td>{model.roc_auc}</td>
                  <td>{model.precision_at_1}</td>
                  <td>{model.recall_at_1}</td>
                  <td>{model.fpr_at_1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="two-col">
        <div className="panel">
          <div className="panel-header compact">
            <h2>Precision–recall</h2>
          </div>
          {prSeries.length ? (
            <LineChart
              series={prSeries}
              xLabel="Recall"
              yLabel="Precision"
              height={250}
            />
          ) : (
            <p className="empty-note">Run evaluation.py to generate curves.</p>
          )}
        </div>

        <div className="panel">
          <div className="panel-header compact">
            <h2>Calibration curve</h2>
          </div>
          {calibrationSeries.length ? (
            <LineChart
              series={calibrationSeries}
              xLabel="Predicted probability"
              yLabel="Observed attack rate"
              height={250}
            />
          ) : (
            <p className="empty-note">No calibration report available.</p>
          )}
        </div>
      </div>

      <div className="two-col">
        <div className="panel">
          <div className="panel-header compact">
            <h2>Per-attack recall @ top 1%</h2>
            <select
              className="type-select"
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              {modelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <BarChart
            items={attackItems}
            maxValue={1}
            color="var(--accent)"
            formatValue={(value) => `${Math.round(value * 100)}%`}
          />
        </div>

        <div className="panel">
          <div className="panel-header compact">
            <h2>Cold-start vs warm entities</h2>
          </div>
          <table className="mini-table">
            <thead>
              <tr>
                <th>Slice</th>
                <th>Rows</th>
                <th>Anomalies</th>
                <th>PR-AUC</th>
                <th>Recall@1%</th>
              </tr>
            </thead>
            <tbody>
              {["cold_start", "warm"].map((slice) => (
                <tr key={slice}>
                  <td>{slice === "cold_start" ? "Cold start" : "Warm"}</td>
                  <td>{coldStart[slice]?.rows ?? "—"}</td>
                  <td>{coldStart[slice]?.anomalies ?? "—"}</td>
                  <td>{coldStart[slice]?.pr_auc ?? "—"}</td>
                  <td>{coldStart[slice]?.recall_at_1 ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="panel-footnote">
            Recall is measured within each slice's own top-1% queue.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header compact">
          <h2>Feature attribution</h2>
          <span className="badge badge-info">
            {shap?.method || "supervised RF importance"}
          </span>
        </div>
        <BarChart
          items={shapItems}
          color="#8b5cf6"
          formatValue={(value) => Number(value).toFixed(3)}
        />
        <p className="panel-footnote">
          Global importance from the offline attribution report. SHAP is used
          automatically when the optional dependency is installed; the fallback
          is RandomForest impurity importance.
        </p>
      </div>
    </div>
  );
}
