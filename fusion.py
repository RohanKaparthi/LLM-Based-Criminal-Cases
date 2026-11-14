"""
fusion.py
---------
Neuro-symbolic fusion layer. Two separate fusions, because "which offence?"
and "will the accused be convicted?" are different questions:

  1. Candidate ranking (which offence):
       fused_score = w_r * retrieval_score   (symbolic: KB similarity / named offence)
                   + w_e * element_fit       (symbolic: share of ingredients proved)
                   + w_l * llm_fit           (neural: LLM's overall fit judgment)

  2. Conviction probability (verdict):
       p = alpha * symbolic_probability + (1 - alpha) * llm_probability

Weights live in data/fusion_weights.json and can be re-fitted with
`python evaluate.py --calibrate`. Defaults are used if that file is absent.
"""
import json
from pathlib import Path
from typing import List, Optional

WEIGHTS_PATH = Path(__file__).parent / "data" / "fusion_weights.json"
DEFAULT_WEIGHTS = {
    "rank": {"retrieval": 0.35, "elements": 0.40, "llm_fit": 0.25},
    "alpha": 0.5,
    "threshold": 0.5,
}
FIT_VALUE = {"strong": 1.0, "partial": 0.5, "none": 0.0}


def load_weights() -> dict:
    if WEIGHTS_PATH.exists():
        loaded = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_WEIGHTS, **loaded, "rank": {**DEFAULT_WEIGHTS["rank"], **loaded.get("rank", {})}}
    return DEFAULT_WEIGHTS


def rank_candidates(evaluated: List[dict], weights: dict) -> List[dict]:
    w = weights["rank"]
    for c in evaluated:
        c["fused_score"] = round(w["retrieval"] * c["retrieval_score"]
                                 + w["elements"] * c["element_fit"]
                                 + w["llm_fit"] * FIT_VALUE.get(c["llm_fit"], 0.0), 4)
    return sorted(evaluated, key=lambda c: c["fused_score"], reverse=True)


def fuse_probability(symbolic_prob: float, llm_prob: Optional[float], weights: dict) -> float:
    if llm_prob is None:
        return round(symbolic_prob, 3)
    alpha = weights["alpha"]
    return round(alpha * symbolic_prob + (1 - alpha) * max(0.0, min(1.0, float(llm_prob))), 3)


def confidence_level(prob: float, threshold: float, charge_margin: float) -> str:
    """How far the decision is from the threshold, and how clearly one charge won."""
    distance = abs(prob - threshold)
    if distance >= 0.3 and charge_margin >= 0.1:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"
