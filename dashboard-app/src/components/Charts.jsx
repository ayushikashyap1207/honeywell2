const PALETTE = [
  "#2dd4bf",
  "#f5a623",
  "#f0465b",
  "#8b5cf6",
  "#34d399",
  "#60a5fa",
  "#f472b6",
  "#facc15",
];

export function LineChart({
  series = [],
  xLabel = "",
  yLabel = "",
  height = 240,
  xDomain = [0, 1],
  yDomain = [0, 1],
  formatX = (value) => Number(value).toFixed(1),
  formatY = (value) => Number(value).toFixed(1),
}) {
  const width = 560;
  const padding = { left: 44, right: 14, top: 14, bottom: 34 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const scaleX = (value) =>
    padding.left +
    ((Number(value) - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plotWidth;
  const scaleY = (value) =>
    height -
    padding.bottom -
    ((Number(value) - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * plotHeight;

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        {ticks.map((tick) => {
          const x = scaleX(xDomain[0] + tick * (xDomain[1] - xDomain[0]));
          const y = scaleY(yDomain[0] + tick * (yDomain[1] - yDomain[0]));
          return (
            <g key={tick}>
              <line
                x1={padding.left}
                x2={width - padding.right}
                y1={y}
                y2={y}
                stroke="rgba(255,255,255,0.07)"
                strokeWidth="1"
              />
              <line
                x1={x}
                x2={x}
                y1={padding.top}
                y2={height - padding.bottom}
                stroke="rgba(255,255,255,0.05)"
                strokeWidth="1"
              />
              <text
                x={padding.left - 8}
                y={y + 3}
                textAnchor="end"
                fontSize="9"
                fill="var(--text-muted)"
              >
                {formatY(yDomain[0] + tick * (yDomain[1] - yDomain[0]))}
              </text>
              <text
                x={x}
                y={height - padding.bottom + 15}
                textAnchor="middle"
                fontSize="9"
                fill="var(--text-muted)"
              >
                {formatX(xDomain[0] + tick * (xDomain[1] - xDomain[0]))}
              </text>
            </g>
          );
        })}

        {series.map((entry, index) => {
          const color = entry.color || PALETTE[index % PALETTE.length];
          const points = (entry.points || [])
            .map(([x, y]) => `${scaleX(x)},${scaleY(y)}`)
            .join(" ");
          return (
            <polyline
              key={entry.name || index}
              points={points}
              fill="none"
              stroke={color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}

        <text
          x={width / 2}
          y={height - 2}
          textAnchor="middle"
          fontSize="10"
          fill="var(--text-secondary)"
        >
          {xLabel}
        </text>
        <text
          x={12}
          y={padding.top + 2}
          fontSize="10"
          fill="var(--text-secondary)"
        >
          {yLabel}
        </text>
      </svg>
      <div className="chart-legend">
        {series.map((entry, index) => (
          <span className="legend-item" key={entry.name || index}>
            <span
              className="legend-swatch"
              style={{ background: entry.color || PALETTE[index % PALETTE.length] }}
            />
            {entry.name}
          </span>
        ))}
      </div>
    </div>
  );
}

export function BarChart({
  items = [],
  labelKey = "label",
  valueKey = "value",
  maxItems = 10,
  color = "var(--accent)",
  maxValue,
  formatValue = (value) => Number(value).toFixed(2),
}) {
  const visible = items.slice(0, maxItems);
  const ceiling =
    maxValue ||
    Math.max(1, ...visible.map((item) => Math.abs(Number(item[valueKey]) || 0)));

  return (
    <div className="bar-chart">
      {visible.map((item, index) => {
        const value = Number(item[valueKey]) || 0;
        return (
          <div className="bar-row" key={`${item[labelKey]}-${index}`}>
            <span className="bar-label" title={String(item[labelKey])}>
              {item[labelKey]}
            </span>
            <span className="bar-track">
              <span
                className="bar-fill"
                style={{
                  width: `${Math.min(100, (Math.abs(value) / ceiling) * 100)}%`,
                  background: item.color || color,
                }}
              />
            </span>
            <span className="bar-value">{formatValue(value)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function statusClass(status) {
  if (status === "significant" || status === "critical") return "badge badge-critical";
  if (status === "warning" || status === "medium") return "badge badge-warning";
  if (status === "stable" || status === "ok") return "badge badge-ok";
  return "badge";
}

export { PALETTE };
