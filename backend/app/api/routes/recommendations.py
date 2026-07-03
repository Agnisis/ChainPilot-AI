from fastapi import APIRouter, HTTPException

from app.models.schemas import RagQueryRequest, RagQueryResponse, RecommendationRequest, RecommendationResponse
from app.services.llm_service import answer_rag_query, generate_recommendations
from app.services.pipeline import pipeline

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    summary = pipeline.get_llm_summary(request.session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run analysis before requesting recommendations.")

    result = generate_recommendations(
        session_id=request.session_id,
        analysis_summary=summary,
        question=request.question,
    )

    return RecommendationResponse(
        session_id=request.session_id,
        executive_summary=result.get("executive_summary", ""),
        key_insights=result.get("key_insights", []),
        risks=result.get("risks", []),
        inventory_recommendations=result.get("inventory_recommendations", []),
        procurement_recommendations=result.get("procurement_recommendations", []),
        logistics_recommendations=result.get("logistics_recommendations", []),
        cost_optimization=result.get("cost_optimization", []),
        strategic_path=result.get("strategic_path", ""),
        raw_text=result.get("raw_text", ""),
    )


@router.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest):
    summary = pipeline.get_llm_summary(request.session_id)
    answer, sources = answer_rag_query(request.session_id, request.query, summary)
    return RagQueryResponse(session_id=request.session_id, answer=answer, sources=sources)
