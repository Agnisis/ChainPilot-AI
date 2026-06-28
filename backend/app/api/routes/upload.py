import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import SessionStatus, UploadResponse
from app.services.pipeline import pipeline
from app.services.rag_service import rag_service
from pydantic import BaseModel

class DemoUploadRequest(BaseModel):
    session_id: str
    dataset_name: str


router = APIRouter(prefix="/api", tags=["upload"])


def _session_data_dir(session_id: str) -> Path:
    return settings.data_dir / session_id


def _session_rag_dir(session_id: str) -> Path:
    return settings.rag_dir / session_id


@router.post("/session", response_model=SessionStatus)
def create_session():
    session_id = uuid.uuid4().hex
    _session_data_dir(session_id).mkdir(parents=True, exist_ok=True)
    _session_rag_dir(session_id).mkdir(parents=True, exist_ok=True)
    return SessionStatus(
        session_id=session_id,
        has_data=False,
        has_rag=False,
        analysis_ready=False,
        data_files=[],
        rag_files=[],
    )


@router.get("/session/{session_id}", response_model=SessionStatus)
def get_session_status(session_id: str):
    data_dir = _session_data_dir(session_id)
    rag_dir = _session_rag_dir(session_id)
    data_files = [f.name for f in data_dir.glob("*")] if data_dir.exists() else []
    rag_files = [f.name for f in rag_dir.glob("*")] if rag_dir.exists() else []
    return SessionStatus(
        session_id=session_id,
        has_data=len(data_files) > 0,
        has_rag=rag_service.has_documents(session_id) or len(rag_files) > 0,
        analysis_ready=pipeline.get_saved_result(session_id) is not None,
        data_files=data_files,
        rag_files=rag_files,
    )


@router.post("/upload/data", response_model=UploadResponse)
async def upload_data_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for training data.")

    dest_dir = _session_data_dir(session_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        file_type="data",
        message="Data file uploaded. Run analysis to train models and generate charts.",
    )


@router.post("/upload/rag", response_model=UploadResponse)
async def upload_rag_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    allowed = {".pdf", ".txt", ".md", ".csv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"RAG supports: {', '.join(sorted(allowed))}")

    dest_dir = _session_rag_dir(session_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (file.filename or f"document{suffix}")

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = rag_service.ingest_file(session_id, dest_path)
    return UploadResponse(
        session_id=session_id,
        filename=file.filename or dest_path.name,
        file_type="rag",
        message=f"RAG document indexed ({chunks} chunks).",
    )

@router.post("/upload/demo", response_model=UploadResponse)
def load_demo_dataset(request: DemoUploadRequest):
    session_id = request.session_id
    dataset_name = request.dataset_name.lower()
    
    dest_dir = _session_data_dir(session_id)
    if dest_dir.exists():
        for f in dest_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Path resolution: routes -> api -> app -> backend -> SCMAi -> Data
    data_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "Data"
    
    if dataset_name == "m5":
        src = data_dir / "m5-forecasting-accuracy Walmart Sales"
    elif dataset_name == "rossmann":
        src = data_dir / "rossmann-store-sales"
    elif dataset_name == "dataco":
        src = data_dir / "DataCo SupplyChain"
    elif dataset_name == "olist":
        src = data_dir / "Brazilian E-Commerce Olist"
    else:
        raise HTTPException(status_code=400, detail="Unknown demo dataset.")
        
    if not src.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset folder not found: {src}")
        
    for f in src.glob("*.csv"):
        shutil.copy2(f, dest_dir / f.name)
        
    return UploadResponse(
        session_id=session_id,
        filename=dataset_name,
        file_type="data",
        message=f"Pre-loaded demo dataset '{dataset_name}'.",
    )

