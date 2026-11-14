"""
main.py
-------
FastAPI backend for the LLM-based Neuro-Symbolic Legal Judgment Prediction
framework (IPC criminal cases, 213 sections). See pipeline.py for the steps.

Run with:  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import JudgmentPipeline
from symbolic_reasoner import punishment_summary

app = FastAPI(title="Neuro-Symbolic Legal Judgment Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = JudgmentPipeline()


class CaseInput(BaseModel):
    case_text: str = Field(..., max_length=20000)
    use_llm: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "sections": len(pipeline.kb.sections),
            "llm_available": pipeline.llm is not None, "llm_error": pipeline.llm_error}


@app.get("/sections")
def sections():
    return [{"section": s, "title": e["title"], "category": e["category"], "severity": e["severity"],
             "punishment": punishment_summary(e["punishment"])}
            for s, e in pipeline.kb.sections.items()]


@app.get("/sections/{section_id}")
def section(section_id: str):
    entry = pipeline.kb.get(section_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Section {section_id} not in knowledge base")
    return entry


@app.post("/predict")
def predict(payload: CaseInput):
    case_text = payload.case_text.strip()
    if len(case_text) < 30:
        raise HTTPException(status_code=400, detail="Please describe the case in at least a sentence or two.")
    return pipeline.predict(case_text, use_llm=payload.use_llm)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
