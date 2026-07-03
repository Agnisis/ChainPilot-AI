from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalysisRequest, AnalysisResult
from app.services.pipeline import pipeline

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResult)
def run_analysis(request: AnalysisRequest):
    try:
        return pipeline.run(
            session_id=request.session_id,
            date_column=request.date_column,
            target_column=request.target_column,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@router.get("/analysis/{session_id}", response_model=AnalysisResult)
def get_analysis(session_id: str):
    result = pipeline.get_saved_result(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis found. Upload data and run analysis first.")
    return result
