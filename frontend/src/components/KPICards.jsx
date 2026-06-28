export default function KPICards({ kpis }) {
  if (!kpis?.length) return null;
  return (
    <div className="kpi-grid">
      {kpis.map((kpi) => (
        <div className="kpi-card" key={kpi.name}>
          <div className="kpi-title">{kpi.name}</div>
          <div className="kpi-value">{kpi.value}</div>
          {kpi.note && <div className="kpi-note">{kpi.note}</div>}
        </div>
      ))}
    </div>
  );
}
