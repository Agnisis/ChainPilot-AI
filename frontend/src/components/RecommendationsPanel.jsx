import ReactMarkdown from 'react-markdown';

export default function RecommendationsPanel({ recommendations, loading, onGenerate, onAsk, askQuery, setAskQuery }) {
  return (
    <section className="panel recommendations-panel animate-fade-up delay-4">
      <div className="panel-header">
        <h2>AI Recommendations</h2>
        <button className="btn primary" onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate Executive Recommendations"}
        </button>
      </div>

      <div className="ask-row">
        <input
          type="text"
          placeholder="Ask about best path, inventory, procurement..."
          value={askQuery}
          onChange={(e) => setAskQuery(e.target.value)}
        />
        <button className="btn secondary" onClick={onAsk} disabled={loading || !askQuery.trim()}>
          Ask RAG + LLM
        </button>
      </div>

      {!recommendations && (
        <p className="muted">Run analysis first, then generate LLM recommendations using your data and uploaded RAG documents.</p>
      )}

      {recommendations && (
        <div className="rec-grid">
        {recommendations.qa_answer && (
          <div className="rec-block highlight" style={{ borderLeft: '4px solid #10b981', marginBottom: '24px', gridColumn: '1 / -1' }}>
            <h3 style={{ color: '#10b981' }}>💬 Ask AI Response</h3>
            <div className="markdown-content">
              <ReactMarkdown>{recommendations.qa_answer}</ReactMarkdown>
            </div>
            {recommendations.qa_sources && recommendations.qa_sources.length > 0 && (
              <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <strong style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Sources: </strong>
                <span style={{ fontSize: '0.85rem', color: '#10b981' }}>{recommendations.qa_sources.join(", ")}</span>
              </div>
            )}
          </div>
        )}
        {recommendations.executive_summary && (
          <div className="rec-block highlight">
            <h3>📋 Executive Summary</h3>
            <div className="markdown-content">
              <ReactMarkdown>{recommendations.executive_summary}</ReactMarkdown>
            </div>
          </div>
        )}
        {recommendations.risks && recommendations.risks.length > 0 && (
          <div className="rec-block danger">
            <h3>🚨 Identified Risks</h3>
            <ul className="markdown-list">{recommendations.risks.map((r, i) => <li key={i}><ReactMarkdown>{r}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.key_insights && recommendations.key_insights.length > 0 && (
          <div className="rec-block">
            <h3>💡 Key Insights</h3>
            <ul className="markdown-list">{recommendations.key_insights.map((k, i) => <li key={i}><ReactMarkdown>{k}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.inventory_recommendations && recommendations.inventory_recommendations.length > 0 && (
          <div className="rec-block">
            <h3>📦 Inventory Action Plan</h3>
            <ul className="markdown-list">{recommendations.inventory_recommendations.map((ir, i) => <li key={i}><ReactMarkdown>{ir}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.procurement_recommendations && recommendations.procurement_recommendations.length > 0 && (
          <div className="rec-block">
            <h3>🤝 Procurement Strategy</h3>
            <ul className="markdown-list">{recommendations.procurement_recommendations.map((pr, i) => <li key={i}><ReactMarkdown>{pr}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.logistics_recommendations && recommendations.logistics_recommendations.length > 0 && (
          <div className="rec-block">
            <h3>🚚 Logistics Optimization</h3>
            <ul className="markdown-list">{recommendations.logistics_recommendations.map((lr, i) => <li key={i}><ReactMarkdown>{lr}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.cost_optimization && recommendations.cost_optimization.length > 0 && (
          <div className="rec-block">
            <h3>💰 Cost Reduction</h3>
            <ul className="markdown-list">{recommendations.cost_optimization.map((co, i) => <li key={i}><ReactMarkdown>{co}</ReactMarkdown></li>)}</ul>
          </div>
        )}
        {recommendations.strategic_path && (
          <div className="rec-block highlight">
            <h3>🚀 Future Business Strategy</h3>
            <div className="markdown-content">
              <ReactMarkdown>{recommendations.strategic_path}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
      )}
    </section>
  );
}

function RecBlock({ title, content, highlight }) {
  return (
    <div className={`rec-block ${highlight ? "highlight" : ""}`}>
      <h3>{title}</h3>
      <p>{content || "—"}</p>
    </div>
  );
}

function RecList({ title, items, danger }) {
  return (
    <div className={`rec-block ${danger ? "danger" : ""}`}>
      <h3>{title}</h3>
      <ul>
        {(items || []).map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
