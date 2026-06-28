# ChainPilot AI

AI-powered supply chain intelligence platform: upload data, train forecasting models, visualize analytics in React, and get RAG + LLM executive recommendations via FastAPI.

## Architecture

```text
React Frontend (Chart.js)
        │
        ▼
   FastAPI Backend
   ├── Data upload & model training (from notebook pipeline)
   ├── Chart JSON for frontend
   ├── RAG (ChromaDB + uploaded PDF/TXT/CSV)
   └── LLM recommendations (Google Gemini)
```

## Quick Start

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env          # add GEMINI_API_KEY
python run.py
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

## Usage Flow

1. Open the React app — a session is created automatically.
2. **Upload CSV** training data (e.g. `DataCoSupplyChainDataset.csv` or M5 `calendar.csv` + `sales_train_validation.csv`).
3. **Upload RAG documents** (PDF, TXT, MD) — company policies, SOPs, supplier docs.
4. Click **Train Models & Analyze** — runs forecasting (ARIMA, SARIMA, RF, XGBoost, LightGBM), anomaly detection, KPIs.
5. View **Chart.js dashboards** in the browser.
6. Click **Generate Executive Recommendations** — Gemini uses analysis + RAG context.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session` | Create session |
| POST | `/api/upload/data` | Upload CSV for training |
| POST | `/api/upload/rag` | Upload docs for RAG |
| POST | `/api/analyze` | Train models & run pipeline |
| GET | `/api/analysis/{session_id}` | Get charts & KPIs |
| POST | `/api/recommendations` | LLM executive recommendations |
| POST | `/api/rag/query` | Ask RAG + LLM |

## Project Structure

```text
backend/
  app/
    main.py                 # FastAPI entry
    services/               # Notebook logic as Python modules
      data_loader.py
      forecasting.py
      anomaly.py
      kpi.py
      charts.py
      rag_service.py
      llm_service.py
      pipeline.py
    api/routes/             # REST endpoints
frontend/
  src/
    App.jsx                 # Main dashboard
    components/             # Charts, uploads, recommendations
```

## Original Notebook

The Jupyter notebook prototype remains at `AI_Powered_Supply_Chain_Intelligence_Platform.ipynb`. The backend services were extracted from that pipeline.

## Environment Variables

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL_NAME=gemini-1.5-flash
CORS_ORIGINS=http://localhost:5173
```

## License

MIT — Agnisis Dutta
