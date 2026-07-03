# CHAPTER 5: WEB APPLICATION & SYSTEM ARCHITECTURE

A fundamental objective of this M.Tech thesis is to bridge the gap between isolated Jupyter Notebook research and real-world deployment. To demonstrate how Artificial Intelligence can be effectively utilized in a corporate environment, the predictive models were integrated into a production-grade Web Application. This platform features a high-performance RESTful backend, a dynamic User Interface, and an advanced Generative AI reasoning layer.

## 5.1 System Architecture Overview

The system was engineered utilizing a decoupled, microservices-inspired architecture. This separation of concerns ensures that the heavy mathematical computations of the ML engine do not block or degrade the user experience of the frontend dashboard. The architecture consists of three primary layers:
1. **The Core ML & Data Pipeline (Python)**
2. **The FastAPI Backend Server**
3. **The React 18 Frontend Dashboard**
4. **The LLM Retrieval-Augmented Generation (RAG) Layer**

## 5.2 The FastAPI Backend Server

The backend was constructed using **FastAPI**, a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. FastAPI was selected over Django or Flask because it natively supports asynchronous request handling (`async`/`await`), which is critical when serving compute-heavy ML models and making external API calls to Large Language Models.

The backend exposes several critical RESTful endpoints:
* `/api/upload/data`: Handles the ingestion of raw, multi-megabyte CSV supply chain datasets.
* `/api/upload/rag`: Handles the ingestion of corporate PDF/TXT documents used to contextualize the LLM.
* `/api/analyze`: Triggers the entire ML pipeline. It ingests the data, executes the Optuna hyperparameter tuning, trains the ARIMA, Random Forest, XGBoost, and PyTorch LSTM models, and evaluates them via walk-forward cross-validation.
* `/api/recommendations`: Triggers the Gemini LLM engine to synthesize the algorithmic outputs.

All internal data models and API schemas were strongly typed using `Pydantic`. This ensured that the multidimensional arrays and evaluation metrics produced by the Machine Learning models were safely serialized into JSON format before being transmitted across the network to the frontend.

## 5.3 The React 18 Frontend Dashboard

The user-facing platform was built using **React 18**, a JavaScript library for building user interfaces. The application was bootstrapped using Vite to ensure rapid hot-module replacement during development.

The frontend is designed as a single-page application (SPA) replicating an enterprise-grade Command Center. The key architectural components of the frontend include:

1. **State Management & Persistence:** The application manages complex state objects (such as the results of the 5 ML models and their associated charts). To ensure data resilience, a custom `useEffect` hook algorithm was implemented to synchronize the application state with the browser's `localStorage`. This guarantees that if an executive accidentally refreshes the browser, the massive machine learning state is instantly restored without requiring a computationally expensive network retraining call.
2. **Dynamic Visualizations:** Raw statistical arrays (e.g., predicted confidence intervals, SHAP feature importance vectors) are parsed and rendered into interactive graphics using **Chart.js** and `react-chartjs-2`. This allows operations managers to visually inspect the forecast boundaries and anomaly locations dynamically.
3. **Model Evaluation Leaderboard:** A dedicated React component was built to dynamically render a comparative leaderboard of all trained models. It extracts the RMSE, MAE, R², and Directional Accuracy scores for ARIMA, XGBoost, and the PyTorch models, and visually highlights the "winning" architecture that the system selected to drive the primary forecast.

## 5.4 LLM Integration & Retrieval-Augmented Generation (RAG)

Presenting an operations manager with an Isolation Forest anomaly score or a PyTorch RMSE metric is often counterproductive; business leaders require actionable strategies, not raw mathematics. To solve this, the platform integrates the **Google Gemini API**.

However, sending raw supply chain metrics to a standard LLM often results in generic or hallucinated advice. To constrain the LLM and make it highly specific to the user's business, a **Retrieval-Augmented Generation (RAG)** pipeline was engineered.

### The RAG Architecture
1. **Document Ingestion:** The user uploads proprietary corporate documents (e.g., supplier contracts, emergency response SOPs, logistics routing rules).
2. **Chunking & Embedding:** The FastAPI backend splits these documents into semantic chunks. Using `SentenceTransformers` (`all-MiniLM-L6-v2`), these chunks are mathematically converted into high-dimensional embedding vectors.
3. **Vector Storage:** The embeddings are stored in a highly efficient `FAISS` (Facebook AI Similarity Search) in-memory vector database.
4. **Semantic Retrieval:** When an anomaly is detected or the user asks a strategic question, the system converts the query into a vector. FAISS performs a cosine-similarity search to instantly retrieve the top-$K$ most mathematically relevant corporate documents.
5. **Contextual Generation:** The raw Machine Learning metrics (the anomaly details, the SHAP feature importance) AND the retrieved corporate documents are injected into a highly structured Prompt Template. This prompt forces the Gemini model to output a response constrained to a specific Markdown schema: Executive Summary, Immediate Action Plan, and Expected Impact.

This RAG integration ensures that the AI's recommendations are mathematically backed by the Machine Learning models and contextually constrained by the organization's actual corporate policies.

## 5.5 Conclusion of Architecture

By seamlessly connecting a complex PyTorch/XGBoost Machine Learning pipeline to a high-speed FastAPI backend, rendering the results on an interactive React 18 Dashboard, and utilizing a FAISS-backed Gemini RAG pipeline to explain the results, this project successfully proves that AI can be extracted from theoretical research and deployed as a highly effective, real-world SaaS application.
