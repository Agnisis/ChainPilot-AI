# CHAPTER 3

## SYSTEM ARCHITECTURE AND PROPOSED METHODOLOGY

### 3.1 High-Level System Architecture

ChainPilot AI is architected as a decoupled, two-tier web application following modern microservices design principles. The system separates the computationally intensive machine learning backend from the interactive visualization frontend, enabling independent scaling, deployment, and maintenance.

> **[INSERT Figure 3.1: System Architecture Diagram]**
> *(Draw or screenshot a diagram showing: User Browser --> React 18 Frontend (Port 5173) --> Vite Proxy --> FastAPI Backend (Port 8000) --> [ML Pipeline | RAG Pipeline | Data Store])*

The architecture consists of the following primary components:

1. **React 18 Frontend** (JavaScript, Port 5173): A single-page application (SPA) built with React 18 and Vite that provides the user interface, interactive Chart.js visualizations, and session management. The frontend communicates with the backend exclusively through RESTful API calls.

2. **FastAPI Backend** (Python, Port 8000): A high-performance, asynchronous Python server that hosts the machine learning pipeline, data ingestion logic, chart generation, and the RAG/LLM recommendation engine. FastAPI was chosen over Flask due to its native support for asynchronous request handling (`async`/`await`), automatic OpenAPI documentation generation, and Pydantic-based request/response validation.

3. **Data Ingestion Layer**: A flexible data loader that can ingest CSV files through manual browser upload or through an automated demo domain pre-loader that copies datasets from the local filesystem.

4. **ML/DL Training Pipeline**: A sequential execution pipeline that trains six models (Auto-ARIMA, Random Forest, XGBoost, PyTorch LSTM, PyTorch GRU, Isolation Forest), evaluates them via walk-forward cross-validation, and ranks them by RMSE.

5. **RAG + LLM Engine**: A Retrieval-Augmented Generation system using ChromaDB for vector storage and Google Gemini for natural-language recommendation generation.

**Table 3.1: Technology Stack**

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend Framework | React | 18.3.1 | Component-based UI rendering |
| Build Tool | Vite | 5.4.11 | Development server with hot module replacement |
| Charting | Chart.js + react-chartjs-2 | 4.4.6 / 5.2.0 | Interactive data visualizations |
| Markdown Rendering | react-markdown | 10.1.0 | Render LLM responses as formatted text |
| UI Font | Google Fonts (Outfit) | 300-800 | Modern, professional typography |
| Backend Framework | FastAPI | >= 0.115 | Asynchronous REST API server |
| ASGI Server | Uvicorn | >= 0.32 | Production-grade ASGI server |
| Data Processing | Pandas | >= 2.2 | DataFrame operations and CSV parsing |
| Numerical Computing | NumPy | >= 1.26 | Array operations and mathematical functions |
| Classical ML | scikit-learn | >= 1.5 | Random Forest, Isolation Forest, preprocessing |
| Gradient Boosting | XGBoost | >= 2.1 | Extreme Gradient Boosting implementation |
| Statistical Models | statsmodels, pmdarima | >= 0.14, >= 2.0 | ARIMA/SARIMA time-series models |
| Deep Learning | PyTorch | >= 2.0 | LSTM and GRU neural network architectures |
| Hyperparameter Tuning | Optuna | >= 3.6 | Bayesian hyperparameter optimization |
| Model Interpretability | SHAP | >= 0.45 | SHapley Additive exPlanations |
| Vector Database | ChromaDB | >= 0.5 | Embedding storage for RAG |
| Sentence Embeddings | sentence-transformers | >= 3.0 | Document chunk embedding (all-MiniLM-L6-v2) |
| LLM API | google-generativeai | >= 0.8 | Google Gemini API integration |
| PDF Parsing | pypdf | >= 5.0 | Extract text from uploaded PDF documents |

### 3.2 Backend Architecture (FastAPI)

The FastAPI backend is organized into a modular, layered architecture with clear separation of concerns:

```
backend/
  app/
    api/
      routes/
        session.py        # Session creation and status endpoints
        upload.py          # Data and RAG file upload endpoints
        analysis.py        # ML pipeline execution endpoint
        recommendations.py # LLM recommendation endpoint
        rag.py             # RAG query endpoint
    models/
      schemas.py           # Pydantic request/response schemas
    services/
      data_loader.py       # Data ingestion and preprocessing
      features.py          # Feature engineering
      forecasting.py       # ARIMA, RF, XGBoost model training
      deep_learning.py     # PyTorch LSTM/GRU training
      hyperparameter_tuning.py  # Optuna optimization
      anomaly.py           # Isolation Forest
      validation.py        # Walk-forward cross-validation
      explainability.py    # SHAP feature importance
      kpi.py               # KPI calculations
      charts.py            # Chart data generation
      ensemble.py          # Model ensembling
      rag_service.py       # ChromaDB vector operations
      llm_service.py       # Google Gemini integration
      pipeline.py          # Full analysis orchestration
    config.py              # Application configuration
    main.py                # FastAPI app initialization
```

The backend exposes 7 RESTful API endpoints:

**Table 3.2: API Endpoint Specification**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/session` | Create a new user session with unique ID |
| GET | `/api/session/{session_id}` | Get session status (uploaded files, data availability) |
| POST | `/api/upload/data` | Upload CSV training data files |
| POST | `/api/upload/rag` | Upload RAG documents (PDF, TXT, MD, CSV) |
| POST | `/api/upload/demo` | Pre-load a demo domain dataset from local filesystem |
| POST | `/api/analyze` | Execute the full ML/DL analysis pipeline |
| POST | `/api/recommendations` | Generate LLM executive recommendations |
| POST | `/api/rag/query` | Query the RAG knowledge base |

> **[INSERT Figure 3.2: FastAPI Backend Request Flow]**
> *(Draw a flowchart: POST /api/analyze --> data_loader.py --> features.py --> forecasting.py + deep_learning.py --> hyperparameter_tuning.py --> validation.py --> anomaly.py --> explainability.py --> kpi.py --> charts.py --> JSON Response)*

### 3.3 Frontend Architecture (React 18)

The React 18 frontend is a single-page application (SPA) with no external routing library. Navigation between the three major views -- Login, Onboarding/Upload, and Dashboard -- is handled through conditional rendering based on React state variables.

**Component Hierarchy:**

```
<App>
  <TopNav>
    <Brand />           -- ChainPilot AI logo and name
    <UserGreeting />    -- Welcome message
    <NewAnalysisBtn />  -- Reset to home page
    <LogoutBtn />       -- Clear session
  </TopNav>

  {Login View}
    <LoginPanel />      -- Name input + Initialize button

  {Onboarding View}
    <FeatureCards />    -- 3 capability showcase cards
    <DemoDatasetSelector />  -- Domain dropdown pre-loader
    <FileUploadSection />    -- CSV data upload
    <FileUploadSection />    -- RAG document upload
    <StagedAssets />         -- Uploaded file list
    <PipelineExecution />    -- Date/target column inputs + Run button

  {Dashboard View}
    <Sidebar>
      <DemoDatasetSelector />
      <FileUploadSection /> (x2)
      <ActiveAssets />
      <ReRunButton />
    </Sidebar>
    <Main>
      <SummaryPanel />         -- Best model, RMSE, KPIs
      <KPICards />             -- Dynamic KPI grid
      <ChartPanel /> (x10)     -- Chart.js visualizations
      <ModelComparisonPanel /> -- Model ranking table + bar chart
      <ValidationPanel />      -- SHAP, CV scores, Optuna results
      <RecommendationsPanel /> -- LLM recommendations + RAG Q&A
    </Main>
</App>
```

> **[INSERT Figure 3.3: React 18 Component Hierarchy]**
> *(Create a tree diagram or screenshot the component hierarchy above)*

**State Management:** The application uses React's built-in `useState` hooks with `localStorage` persistence. All critical state variables -- `sessionId`, `userName`, `analysis`, and `recommendations` -- are automatically saved to `localStorage` on change, ensuring the dashboard survives page reloads without data loss.

**Styling:** The application employs a dark glassmorphism design system implemented in pure CSS (753 lines). Key design tokens include semi-transparent panel backgrounds with `backdrop-filter: blur(12px)`, a deep navy-to-indigo gradient background (`#020617 --> #0f172a --> #1e1b4b`), sky-blue primary color (`#38bdf8`), and smooth `fadeUp` entrance animations with staggered delay classes.

### 3.4 Data Ingestion and Preprocessing Pipeline

The data ingestion layer (`data_loader.py`) is designed to be completely dataset-agnostic. It implements three tiers of data loading:

**Tier 1: Domain-Specific Loaders**
For the four benchmark datasets, specialized preprocessing functions handle domain-specific data transformations:

- `prepare_rossmann_timeseries()`: Merges `train.csv` with `store.csv` on the `Store` column, aggregates daily sales across all stores, and constructs a time-indexed demand series.
- `prepare_olist_timeseries()`: Joins `orders` and `order_items` tables, extracts freight values as the target metric, and constructs a date-indexed delivery performance series.

**Tier 2: Auto-Detection Loader**
For unknown datasets uploaded via the manual CSV upload, the `prepare_generic_timeseries()` function performs intelligent column detection:
- It searches for date-like columns by scanning column names for patterns such as "date", "timestamp", "order_date", "DateOrders"
- It searches for target columns by scanning for patterns such as "sales", "demand", "quantity", "revenue"
- It automatically parses dates, sorts chronologically, and resamples to daily frequency

**Tier 3: Fallback**
If no date or target column is found, the system falls back to treating the first datetime-parseable column as the date axis and the first numeric column as the target.

### 3.5 Multi-Domain Dataset Profiles

This section details the four benchmark datasets and their specific analytical challenges.

**Dataset 1: M5 Forecasting Accuracy (Walmart)**
- **Source:** Kaggle M5 Competition
- **Scale:** 30,490 products across 10 stores, ~58 million individual data points
- **Files:** `calendar.csv` (1,969 rows x 14 columns), `sales_train_validation.csv` (30,490 x 1,947), `sell_prices.csv` (6,841,121 x 4)
- **Features:** Hierarchical product categories (Food, Household, Hobbies), store locations (CA, TX, WI), calendar events (Super Bowl, Christmas), SNAP eligibility
- **Challenge:** Massive hierarchical time-series forecasting with extreme intermittent demand (many zero-sales days)

**Dataset 2: Rossmann Store Sales**
- **Source:** Kaggle
- **Scale:** 1,017,209 daily sales records across 1,115 European drug stores
- **Files:** `train.csv` (1,017,209 x 9), `store.csv` (1,115 x 10)
- **Features:** Promotions (`Promo`, `Promo2`), school/state holidays, competitor distance, store type (a/b/c/d), product assortment level
- **Challenge:** Modeling the non-linear impact of promotional campaigns and holiday seasonality on daily sales

**Dataset 3: DataCo Smart Supply Chain**
- **Source:** Kaggle (CC0 License)
- **Scale:** 180,519 supply chain records with 53 operational columns
- **Files:** `DataCoSupplyChainDataset.csv` (180,519 x 53)
- **Features:** Shipping modes, delivery status (Late/On-time/Advance), order regions (global), product categories, profit margins, customer segments, latitude/longitude
- **Challenge:** Multivariate logistics optimization, late-delivery risk prediction, and fraud/anomaly detection across global shipping networks

**Dataset 4: Brazilian E-Commerce Olist**
- **Source:** Kaggle
- **Scale:** 99,441 real e-commerce orders across 8 normalized relational tables
- **Files:** 9 CSV files including `olist_orders_dataset.csv` (99,441 x 8), `olist_order_items_dataset.csv` (112,650 x 7), `olist_geolocation_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_customers_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`
- **Features:** Freight values, delivery timestamps, customer reviews, seller locations, product dimensions, payment types
- **Challenge:** End-to-end e-commerce fulfillment analysis requiring SQL-style relational joins across 8 tables

### 3.6 RAG Pipeline: FAISS + Google Gemini

The Retrieval-Augmented Generation (RAG) pipeline allows non-technical users to ask natural-language questions about their supply chain data and receive grounded, context-aware responses.

> **[INSERT Figure 3.4: RAG Pipeline: Document Ingestion to LLM Response]**
> *(Draw a pipeline: PDF/TXT Upload --> Text Extraction (pypdf) --> Chunk Splitting --> Sentence Transformer Embedding --> ChromaDB Vector Store --> Query Embedding --> Top-K Retrieval --> Gemini LLM Prompt Injection --> Natural Language Response)*

**Stage 1: Document Ingestion**
When a user uploads a RAG document (PDF, TXT, MD, or CSV), the `rag_service.py` module extracts the raw text content. For PDFs, the `pypdf` library is used to extract text page by page. The extracted text is then split into overlapping chunks of approximately 500 tokens each.

**Stage 2: Embedding and Storage**
Each text chunk is embedded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model from the `sentence-transformers` library. These embeddings are stored in a ChromaDB persistent collection associated with the user's session.

**Stage 3: Retrieval and Generation**
When the user submits a query through the "Ask RAG + LLM" interface, the query is embedded using the same sentence transformer. ChromaDB performs an approximate nearest-neighbor search to retrieve the top-K (K=5) most semantically similar document chunks. These chunks are injected into a structured prompt template that instructs Google Gemini to generate a response grounded exclusively in the retrieved context.

**Stage 4: Executive Recommendations**
The `llm_service.py` module constructs a comprehensive prompt that includes the model performance metrics, SHAP feature importance rankings, KPI calculations, and anomaly detection results. Google Gemini then generates structured recommendations covering:
- Executive Summary
- Identified Risks
- Key Insights
- Inventory Action Plan
- Procurement Strategy
- Logistics Optimization
- Cost Reduction Opportunities
- Future Business Strategy
