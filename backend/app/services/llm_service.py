import json
import re
from typing import Any

from app.config import settings
from app.services.rag_service import rag_service

import logging
logger = logging.getLogger(__name__)

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def _configure_gemini():
    if not GEMINI_AVAILABLE or not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


def _parse_sections(text: str) -> dict[str, Any]:
    """Parse LLM markdown-style sections into structured fields."""
    sections = {
        "executive_summary": "",
        "key_insights": [],
        "risks": [],
        "inventory_recommendations": [],
        "procurement_recommendations": [],
        "logistics_recommendations": [],
        "cost_optimization": [],
        "strategic_path": "",
    }
    current = None
    for line in text.splitlines():
        lower = line.lower().strip()
        if re.match(r"^#+\s*1\.?\s*executive", lower) or "executive summary" in lower:
            current = "executive_summary"
            continue
        if "key insight" in lower:
            current = "key_insights"
            continue
        if re.match(r"^#+\s*3\.?\s*risk", lower) or lower.startswith("risks"):
            current = "risks"
            continue
        if "inventory" in lower:
            current = "inventory_recommendations"
            continue
        if "procurement" in lower:
            current = "procurement_recommendations"
            continue
        if "logistic" in lower:
            current = "logistics_recommendations"
            continue
        if "cost" in lower:
            current = "cost_optimization"
            continue
        if "strategy" in lower or "best path" in lower or "future business" in lower:
            current = "strategic_path"
            continue
        if not line.strip():
            continue
        cleaned = re.sub(r"^[-*•\d.]+\s*", "", line.strip())
        if not cleaned:
            continue
        if current == "executive_summary":
            sections["executive_summary"] += cleaned + " "
        elif current == "strategic_path":
            sections["strategic_path"] += cleaned + " "
        elif current and current in sections:
            sections[current].append(cleaned)

    sections["executive_summary"] = sections["executive_summary"].strip()
    sections["strategic_path"] = sections["strategic_path"].strip() or sections["executive_summary"]
    return sections


def _extract_severity_scores(text: str) -> dict[str, str]:
    """Extract optional severity scores from LLM text (High, Medium, Low)."""
    scores = {}
    if "high risk" in text.lower() or "critical" in text.lower():
        scores["overall_risk"] = "High"
    elif "medium risk" in text.lower() or "moderate" in text.lower():
        scores["overall_risk"] = "Medium"
    else:
        scores["overall_risk"] = "Low"
    return scores


def generate_recommendations(
    session_id: str,
    analysis_summary: dict[str, Any],
    question: str | None = None,
) -> dict[str, Any]:
    
    # RAG Retrieval
    rag_docs, rag_sources = rag_service.query(
        session_id,
        question or "supply chain inventory procurement logistics recommendations strategy",
        top_k=5,
    )
    rag_context = "\n\n".join(rag_docs) if rag_docs else "No uploaded RAG documents available."
    
    # Condense analysis payload to stay within token limits while preserving rich DS info
    condensed_analysis = {
        "best_model": analysis_summary.get("best_model"),
        "best_metrics": analysis_summary.get("best_metrics"),
        "anomalies": analysis_summary.get("anomaly_summary"),
        "kpis": analysis_summary.get("supply_chain_kpis"),
        "feature_importance": analysis_summary.get("feature_importance"),
        "statistical_tests": analysis_summary.get("statistical_tests")
    }

    prompt = f"""
You are a Senior Data Science & Supply Chain Consultant for ChainPilot AI.

Analyze the forecasting output, SHAP feature importance, statistical tests, anomaly detection, KPIs, and uploaded company documents.
Provide actionable executive guidance that bridges Data Science insights with Supply Chain operations.

Instead of long paragraphs, structure your answer into concrete SCENARIOS, IMMEDIATE ACTIONS, and EXPECTED IMPACTS.
Format your response exactly with these sections (using Markdown):

# 1. Executive Summary
Give a brief, punchy high-level overview.

# 2. Key Insights
For each insight, provide:
* **Observation:** (e.g., Based on SHAP features, Lag7 is highly influential)
* **Actionable Step:** (What should the company do right now?)

# 3. Risks
For each risk, provide:
* **Identified Risk:** (e.g., High demand anomaly detected)
* **Mitigation Strategy:** (How to prevent disruption)

# 4. Inventory Recommendations
# 5. Procurement Recommendations
# 6. Logistics Recommendations
# 7. Cost Optimization Suggestions
For sections 4-7, structure each recommendation exactly like this:
* **Scenario:** [The business context]
* **Action Plan:** [A strict bullet point of the step]
* **Expected Impact:** [What this action achieves]

# 8. Future Business Strategy
Provide the best strategic path forward based on this data.

User question: {question or "Provide full supply chain improvement recommendations based on the ML analysis."}

Data Science & Forecasting Data:
{json.dumps(condensed_analysis, indent=2, default=str)}

Relevant uploaded document context (RAG):
{rag_context[:settings.llm_max_context_chars]}
"""

    client = _configure_gemini()
    if client is None:
        fallback = _fallback_recommendations(analysis_summary)
        fallback["raw_text"] = "Gemini API key not configured. Set GEMINI_API_KEY in backend/.env"
        return fallback

    for attempt in range(settings.llm_max_retries):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model_name,
                contents=prompt,
                config={'temperature': settings.llm_temperature}
            )
            raw_text = response.text or ""
            break
        except Exception as exc:
            logger.warning(f"Gemini call failed (attempt {attempt+1}): {exc}")
            if attempt == settings.llm_max_retries - 1:
                fallback = _fallback_recommendations(analysis_summary)
                fallback["raw_text"] = f"Gemini call failed after {settings.llm_max_retries} attempts: {exc}"
                return fallback

    parsed = _parse_sections(raw_text)
    parsed["raw_text"] = raw_text
    parsed["rag_sources"] = rag_sources
    parsed["severity_scores"] = _extract_severity_scores(raw_text)
    return parsed


def answer_rag_query(session_id: str, query: str, analysis_summary: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    docs, sources = rag_service.query(session_id, query, top_k=5)
    context = "\n\n".join(docs) if docs else "No matching document chunks found."
    
    analysis_block = ""
    if analysis_summary:
        # Include SHAP features in Q&A context
        condensed = {
            "metrics": analysis_summary.get("best_metrics"),
            "features": analysis_summary.get("feature_importance")
        }
        analysis_block = f"\n\nCurrent Data Science Analysis:\n{json.dumps(condensed, indent=2, default=str)[:2000]}"

    prompt = f"""
You are ChainPilot AI data science and supply chain assistant. Answer using uploaded documents and analysis context.
Be concise, practical, and business-focused. If asked about model features, refer to the SHAP importance.

Structure your answer using Markdown with the following clear formats:
### 🏢 Business Scenario
[Context of the answer]
### ⚡ Immediate Action Plan
* **Step 1:** ...
* **Step 2:** ...
### 📈 Expected Impact
[What this solves]

Question: {query}

Document context:
{context}
{analysis_block}
"""

    client = _configure_gemini()
    if client is None:
        if docs:
            return f"Based on uploaded documents:\n\n{context[:2000]}", sources
        return "Configure GEMINI_API_KEY for intelligent answers.", sources

    try:
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=prompt,
            config={'temperature': settings.llm_qa_temperature}
        )
        return response.text or "No response generated.", sources
    except Exception as exc:
        return f"LLM error: {exc}", sources


def _fallback_recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_model", "N/A")
    rmse = summary.get("best_metrics", {}).get("RMSE", "N/A")
    anom_dict = summary.get("anomaly_summary", {})
    anomalies = anom_dict.get("demand_anomaly_count", 0) if isinstance(anom_dict, dict) else 0
    
    return {
        "executive_summary": f"Best forecast model: {best} (RMSE: {rmse}). {anomalies} demand anomalies detected.",
        "key_insights": [
            "Advanced Deep Learning and ML pipeline completed.",
            "Review SHAP feature importance for demand drivers.",
        ],
        "risks": ["Late delivery and demand volatility require monitoring."],
        "inventory_recommendations": ["Align safety stock with forecast prediction intervals."],
        "procurement_recommendations": ["Use 30/60/90-day forecast totals for supplier planning."],
        "logistics_recommendations": ["Investigate regions with highest shipment delay rates."],
        "cost_optimization": ["Reduce overstock on low-velocity SKUs identified in analysis."],
        "strategic_path": "Continue data-driven forecasting utilizing ensemble and deep learning models.",
        "rag_sources": [],
        "severity_scores": {"overall_risk": "Medium"}
    }