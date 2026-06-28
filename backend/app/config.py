from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = Path(__file__).resolve().parents[1]
_ENV_FILES = (
    _BACKEND / ".env",
    _ROOT / ".env",
    _ROOT / "backend" / ".env",
)

class Settings(BaseSettings):
    app_name: str = "ChainPilot AI"
    debug: bool = False
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Gemini API Configuration
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-2.5-flash"

    # Database Connections (reserved for future use)
    database_url: str | None = None

    # Vector Database Connections
    chroma_host: str | None = None
    chroma_port: int = 8001

    # Local Directory settings
    upload_dir: Path = Path("uploads")
    data_dir: Path = Path("uploads/data")
    rag_dir: Path = Path("uploads/rag")
    chroma_dir: Path = Path("uploads/chroma")
    results_dir: Path = Path("uploads/results")

    # ── Core ML Settings ─────────────────────────────────────────────
    random_state: int = 42
    test_size_days: int = 90
    forecast_horizons: list[int] = [30, 60, 90]
    fast_mode: bool = False
    anomaly_contamination: float = 0.03

    # ── Hyperparameter Tuning (Optuna) ───────────────────────────────
    tuning_mode: bool = True           # Enable Optuna Bayesian optimisation
    tuning_trials: int = 5             # KEEP LOW for CPU execution speed
    cv_folds: int = 3                  # Walk-forward cross-validation folds

    # ── Deep Learning ────────────────────────────────────────────────
    deep_learning_enabled: bool = True  # Enable LSTM / GRU models
    dl_epochs: int = 20                 # Low epochs to stay fast on CPU
    dl_sequence_length: int = 30        # Input window for LSTM/GRU
    dl_hidden_size: int = 64            # Hidden layer dimension
    dl_num_layers: int = 2              # Number of recurrent layers
    dl_learning_rate: float = 0.001     # Initial learning rate
    dl_batch_size: int = 32             # Training batch size
    dl_patience: int = 10               # Early stopping patience

    # ── Ensemble ─────────────────────────────────────────────────────
    ensemble_enabled: bool = True       # Enable ensemble/stacking

    # ── RAG Configuration ────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"   # sentence-transformers model
    rag_chunk_size: int = 512           # Chunk size in tokens (approx)
    rag_chunk_overlap: int = 64         # Chunk overlap
    reranking_enabled: bool = True      # Cross-encoder reranking

    # ── LLM Configuration ────────────────────────────────────────────
    llm_temperature: float = 0.3        # Temperature for recommendations
    llm_qa_temperature: float = 0.7     # Temperature for Q&A
    llm_max_retries: int = 3            # Retry count on failure
    llm_max_context_chars: int = 12000  # Max chars sent to LLM

    model_config = SettingsConfigDict(
        env_file=[str(p) for p in _ENV_FILES if p.exists()] or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

settings = Settings()

# Disable heavy features in fast mode
if settings.fast_mode:
    settings.tuning_mode = False
    settings.dl_epochs = min(settings.dl_epochs, 30)
    settings.dl_patience = min(settings.dl_patience, 7)

# Create necessary directories
for path in (settings.upload_dir, settings.data_dir, settings.rag_dir, settings.chroma_dir, settings.results_dir):
    path.mkdir(parents=True, exist_ok=True)