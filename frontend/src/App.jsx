import { useEffect, useState } from "react";
import {
  createSession,
  getRecommendations,
  getSessionStatus,
  queryRag,
  runAnalysis,
  uploadDataFile,
  uploadDemoDataset,
  uploadRagFile,
} from "./api/client";
import ChartPanel from "./components/ChartPanel";
import DemoDatasetSelector from "./components/DemoDatasetSelector";
import FileUploadSection from "./components/FileUploadSection";
import KPICards from "./components/KPICards";
import ModelComparisonPanel from "./components/ModelComparisonPanel";
import RecommendationsPanel from "./components/RecommendationsPanel";
import ValidationPanel from "./components/ValidationPanel";

export default function App() {
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("sessionId") || "");
  const [status, setStatus] = useState(null);
  const [analysis, setAnalysis] = useState(() => {
    const saved = localStorage.getItem("analysis");
    try { return saved ? JSON.parse(saved) : null; } catch { return null; }
  });
  const [recommendations, setRecommendations] = useState(() => {
    const saved = localStorage.getItem("recommendations");
    try { return saved ? JSON.parse(saved) : null; } catch { return null; }
  });
  const [loading, setLoading] = useState(false); // used for file uploads
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [userName, setUserName] = useState(() => localStorage.getItem("userName") || "");
  const [isLoggedIn, setIsLoggedIn] = useState(() => !!localStorage.getItem("userName"));
  const [error, setError] = useState("");
  const [askQuery, setAskQuery] = useState("");
  const [dateColumn, setDateColumn] = useState("");
  const [targetColumn, setTargetColumn] = useState("");

  // Manage Backend Session Initialization
  useEffect(() => {
    if (!sessionId) {
      createSession()
        .then((data) => {
          setSessionId(data.session_id);
          localStorage.setItem("sessionId", data.session_id);
        })
        .catch((e) => setError("Failed to start backend session: " + e.message));
    } else {
      refreshStatus(sessionId);
    }
  }, [sessionId]);

  // Persist State Changes
  useEffect(() => {
    if (isLoggedIn && userName.trim()) localStorage.setItem("userName", userName);
  }, [isLoggedIn, userName]);

  useEffect(() => {
    if (analysis) localStorage.setItem("analysis", JSON.stringify(analysis));
    else localStorage.removeItem("analysis");
  }, [analysis]);

  useEffect(() => {
    if (recommendations) localStorage.setItem("recommendations", JSON.stringify(recommendations));
    else localStorage.removeItem("recommendations");
  }, [recommendations]);

  const refreshStatus = async (sid = sessionId) => {
    if (!sid) return;
    const s = await getSessionStatus(sid);
    setStatus(s);
  };

  const handleDataUpload = async (files) => {
    setError("");
    setLoading(true);
    try {
      for (const file of files) {
        await uploadDataFile(sessionId, file);
      }
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLoad = async (datasetName) => {
    setError("");
    setLoading(true);
    try {
      await uploadDemoDataset(sessionId, datasetName);
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRagUpload = async (files) => {
    setError("");
    setLoading(true);
    try {
      for (const file of files) {
        await uploadRagFile(sessionId, file);
      }
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    setError("");
    setIsAnalyzing(true);
    setRecommendations(null);
    try {
      const result = await runAnalysis(
        sessionId,
        dateColumn.trim() || null,
        targetColumn.trim() || null
      );
      setAnalysis(result);
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRecommendations = async () => {
    setError("");
    setIsGenerating(true);
    try {
      const rec = await getRecommendations(sessionId);
      setRecommendations(rec);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleAsk = async () => {
    setError("");
    setIsGenerating(true);
    try {
      const res = await queryRag(sessionId, askQuery);
      setRecommendations(prev => ({
        ...(prev || {}),
        qa_answer: res.answer,
        qa_sources: res.sources,
      }));
    } catch (e) {
      setError(e.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="app">
      <nav className="top-nav" style={{ justifyContent: 'space-between' }}>
        <div className="nav-left" style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div className="brand">
            <div className="logo-orb"></div>
            <h2>ChainPilot <span className="text-primary">AI</span></h2>
          </div>
          
          {isLoggedIn && userName && (
            <div className="user-greeting animate-fade-up" style={{ borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '24px', color: '#fff', fontWeight: '500' }}>
              Welcome, {userName}
            </div>
          )}
        </div>
        
        {isLoggedIn && userName && (
          <div className="nav-right animate-fade-up" style={{ display: 'flex', gap: '12px' }}>
            {analysis && (
              <button 
                onClick={() => {
                  setAnalysis(null);
                  setRecommendations(null);
                  setStatus(null);
                  setSessionId("");
                  localStorage.removeItem("sessionId");
                  localStorage.removeItem("analysis");
                  localStorage.removeItem("recommendations");
                }}
                className="btn primary"
                style={{ padding: '4px 12px', fontSize: '0.8rem', borderRadius: '4px' }}
              >
                + New Analysis
              </button>
            )}
            <button 

              onClick={() => {
                localStorage.removeItem("userName");
                localStorage.removeItem("sessionId");
                localStorage.removeItem("analysis");
                localStorage.removeItem("recommendations");
                setUserName("");
                setIsLoggedIn(false);
                setSessionId("");
                setAnalysis(null);
                setRecommendations(null);
                setStatus(null);
              }}
              className="logout-btn"
              style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', color: 'var(--text-muted)', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem', transition: 'all 0.2s' }}
            >
              Logout
            </button>
          </div>
        )}
      </nav>

      {error && <div className="error-banner">{error}</div>}

      {!isLoggedIn ? (
        <div className="onboarding-layout animate-fade-up">
          <header className="hero-header">
            <div className="glow-orb"></div>
            <h1>ChainPilot <span className="text-primary">AI</span></h1>
            <p className="hero-subtitle">State-of-the-art Deep Learning, Optuna tuning, and Context-Aware LLMs for supply chain mastery.</p>
          </header>

          <div className="panel login-panel animate-fade-up delay-1" style={{ maxWidth: '400px', width: '100%', textAlign: 'center', padding: '40px' }}>
            <h2>Welcome to SCMAi</h2>
            <p className="muted" style={{ marginBottom: '24px' }}>Please enter your name to initialize your secure session.</p>
            <input 
              type="text" 
              placeholder="Your Full Name" 
              value={userName} 
              onChange={(e) => setUserName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && userName.trim()) setIsLoggedIn(true) }}
              style={{ width: '100%', padding: '14px', marginBottom: '20px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', fontSize: '1rem', outline: 'none' }}
            />
            <button className="btn primary full giant-btn" onClick={() => { if(userName.trim()) setIsLoggedIn(true); }}>
              Initialize Dashboard
            </button>
          </div>
        </div>
      ) : (!analysis && !isAnalyzing) ? (
        <div className="onboarding-layout animate-fade-up">
          <header className="hero-header">
            <div className="glow-orb"></div>
            <h1>ChainPilot <span className="text-primary">AI</span></h1>
            <p className="hero-subtitle">State-of-the-art Deep Learning, Optuna tuning, and Context-Aware LLMs for supply chain mastery.</p>
          </header>

          <div className="features-grid">
            <div className="feature-card">
              <div className="icon">📈</div>
              <h3>Demand Forecasting</h3>
              <p>PyTorch LSTMs & Auto-ARIMA tuned via Optuna to predict sales.</p>
            </div>
            <div className="feature-card">
              <div className="icon">🚨</div>
              <h3>Anomaly Detection</h3>
              <p>Isolation Forests instantly catch massive supply chain disruptions.</p>
            </div>
            <div className="feature-card">
              <div className="icon">🧠</div>
              <h3>LLM Strategy</h3>
              <p>Gemini RAG analyzes SHAP features to provide actionable insights.</p>
            </div>
          </div>

          <div className="upload-workspace">
            <div className="upload-row">
              <DemoDatasetSelector onSelectDataset={handleDemoLoad} loading={loading} />
              <FileUploadSection
                title="Training Data (CSV)"
                description="Upload supply chain CSV files."
                accept=".csv"
                onUpload={handleDataUpload}
                loading={loading}
                buttonLabel="Upload CSV Data"
              />
              <FileUploadSection
                title="RAG Documents"
                description="Upload PDF, TXT, MD company docs."
                accept=".pdf,.txt,.md,.csv"
                onUpload={handleRagUpload}
                loading={loading}
                buttonLabel="Upload Documents"
              />
            </div>

            {(status?.data_files?.length > 0 || status?.rag_files?.length > 0) && (
              <section className="panel assets-panel animate-fade-up delay-1">
                <h2>Staged Assets</h2>
                <div className="assets-grid">
                  {status.data_files?.length > 0 && (
                    <div className="asset-group">
                      <h4>Training Data</h4>
                      <ul>{status.data_files.map(f => <li key={f}>📄 {f}</li>)}</ul>
                    </div>
                  )}
                  {status.rag_files?.length > 0 && (
                    <div className="asset-group">
                      <h4>RAG Documents</h4>
                      <ul>{status.rag_files.map(f => <li key={f}>📑 {f}</li>)}</ul>
                    </div>
                  )}
                </div>
              </section>
            )}

            <section className="panel run-panel animate-fade-up delay-2">
              <h2>Pipeline Execution</h2>
              <div className="input-group-row">
                <label>
                  Date column
                  <input value={dateColumn} onChange={(e) => setDateColumn(e.target.value)} placeholder="auto-detect" />
                </label>
                <label>
                  Target column
                  <input value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} placeholder="Sales / Demand" />
                </label>
              </div>
              <button className="btn primary full giant-btn" onClick={handleAnalyze} disabled={loading || !status?.has_data}>
                Start AI Analysis Pipeline
              </button>
            </section>
          </div>
        </div>
      ) : (
        <div className="layout dashboard animate-fade-up">
          <aside className="sidebar">
            <DemoDatasetSelector onSelectDataset={handleDemoLoad} loading={loading} />
            <FileUploadSection
              title="Training Data"
              description="Upload new CSV files to swap datasets."
              accept=".csv"
              onUpload={handleDataUpload}
              loading={loading}
              buttonLabel="Upload CSV"
            />
            <FileUploadSection
              title="RAG Documents"
              description="Upload more company docs."
              accept=".pdf,.txt,.md,.csv"
              onUpload={handleRagUpload}
              loading={loading}
              buttonLabel="Upload Documents"
            />
            
            {(status?.data_files?.length > 0 || status?.rag_files?.length > 0) && (
              <section className="panel assets-panel animate-fade-up delay-1" style={{ marginTop: '0' }}>
                <h2>Active Assets</h2>
                <div className="assets-grid" style={{ gridTemplateColumns: '1fr' }}>
                  {status.data_files?.length > 0 && (
                    <div className="asset-group">
                      <h4>Training Data</h4>
                      <ul>{status.data_files.map(f => <li key={f}>📄 {f}</li>)}</ul>
                    </div>
                  )}
                  {status.rag_files?.length > 0 && (
                    <div className="asset-group">
                      <h4>RAG Documents</h4>
                      <ul>{status.rag_files.map(f => <li key={f}>📑 {f}</li>)}</ul>
                    </div>
                  )}
                </div>
              </section>
            )}

            <section className="panel run-panel">
              <button className="btn primary full giant-btn" onClick={handleAnalyze} disabled={isAnalyzing || loading || !status?.has_data}>
                {isAnalyzing ? "Running pipeline..." : "Re-Run Pipeline"}
              </button>
            </section>
          </aside>

          <main className="content">
            {isAnalyzing ? (
              <div className="loader-container animate-fade-up">
                <div className="cyber-spinner"></div>
                <h2>Training Models & Analyzing Data...</h2>
                <p className="muted">Running Deep Learning pipelines, Optuna optimization, and walk-forward cross-validation.</p>
              </div>
            ) : analysis ? (
            <>
              <section className="panel summary-panel animate-fade-up delay-1">
                <h2>Analysis Summary</h2>
                <p>
                  <strong>Source:</strong> {analysis.forecasting_source} · <strong>Best model:</strong>{" "}
                  {analysis.best_model} · <strong>RMSE:</strong> {analysis.best_metrics?.RMSE?.toFixed(2)}
                </p>
                <KPICards kpis={analysis.kpis} />
              </section>

              <section className="panel animate-fade-up delay-2">
                <h2>Forecast & Analytics Charts</h2>
                <div className="charts-grid">
                  {analysis.charts.map((chart) => (
                    <ChartPanel key={chart.id} chart={chart} />
                  ))}
                </div>
              </section>

              <ModelComparisonPanel 
                modelRanking={analysis.model_ranking} 
                bestModel={analysis.best_model} 
              />

              <ValidationPanel analysis={analysis} />

              <RecommendationsPanel
                recommendations={recommendations}
                loading={isGenerating}
                onGenerate={handleRecommendations}
                onAsk={handleAsk}
                askQuery={askQuery}
                setAskQuery={setAskQuery}
              />
            </>
          ) : null}
        </main>
      </div>
      )}
    </div>
  );
}
