"""
pipeline.py
-----------
End-to-end judgment pipeline for one case narrative:

  1. Fact analysis (LLM, fast model)          -> facts + offences the facts suggest
  2. Candidate retrieval (symbolic, TF-IDF)   -> top sections from the 213-section KB
  3. Element assessment (LLM, reasoning model) -> each ingredient proved / not proved / unclear
  4. Charge selection (symbolic rules)         -> primary charge, extra charges, reductions
  5. Verdict (symbolic rules + fused probability)
  6. Punishment (curated KB lookup)

If the LLM is unavailable or fails, steps 1 and 3 fall back to regex rules
(mode = "symbolic_only").
"""
import logging
from typing import Optional

from fusion import load_weights, rank_candidates, fuse_probability, confidence_level
from knowledge_base import KnowledgeBase
from retriever import Retriever
from symbolic_reasoner import SymbolicReasoner

log = logging.getLogger(__name__)
TOP_K = 8
MAX_CANDIDATES = 12


class JudgmentPipeline:
    def __init__(self, use_llm: bool = True):
        self.kb = KnowledgeBase()
        self.retriever = Retriever(self.kb)
        self.weights = load_weights()
        self.reasoner = SymbolicReasoner(self.kb, threshold=self.weights["threshold"])
        self.llm = None
        self.llm_error: Optional[str] = None
        if use_llm:
            try:
                from llm_engine import LLMEngine
                self.llm = LLMEngine()
            except Exception as e:  # missing key, missing package, ...
                self.llm_error = repr(e)
                log.warning("LLM unavailable, running symbolic-only: %s", e)

    def _add_lesser_offences(self, retrieved: list) -> list:
        """Add the lesser-included offences of the top matches, so that if a grave charge fails on
        one ingredient the reasoner can still convict for the lesser offence (rule R3)."""
        present = {self.kb.charge_section(r["section"]) for r in retrieved}
        added = []
        for r in retrieved[:3]:
            for lesser in self.kb.lesser_of(r["section"]):
                charge = self.kb.charge_section(lesser)
                if charge not in present and len(retrieved) + len(added) < MAX_CANDIDATES:
                    present.add(charge)
                    added.append({"section": charge, "title": self.kb.sections[charge]["title"],
                                  "retrieval_score": round(0.8 * r["retrieval_score"], 4), "raw_score": 0.0,
                                  "tfidf": 0.0, "signals": [f"lesser offence of s.{r['section']}"]})
        return retrieved + added

    def predict(self, case_text: str, use_llm: bool = True) -> dict:
        trace = []
        llm_on = use_llm and self.llm is not None
        mode = "neuro_symbolic" if llm_on else "symbolic_only"

        # 1. Fact analysis
        analysis = {}
        if llm_on:
            try:
                analysis = self.llm.analyze_case(case_text)
                trace.append("LLM extracted the facts and proposed candidate offences.")
            except Exception as e:
                log.warning("analyze_case failed: %s", e)
                trace.append(f"LLM fact analysis failed ({type(e).__name__}); continuing with retrieval only.")
        proposed = [s for s in (self.kb.normalize(o.get("section"))
                                for o in analysis.get("suspected_offences", []) if isinstance(o, dict)) if s]

        # 2. Retrieval
        retrieved = self.retriever.retrieve(case_text, top_k=TOP_K, extra=proposed)
        best_raw = max((r["raw_score"] for r in retrieved), default=0.0)
        retrieved = self._add_lesser_offences(retrieved)
        candidates = [{**r, **{k: self.kb.sections[r["section"]][k]
                               for k in ("ingredients", "distinguishing_note", "title")}} for r in retrieved]
        trace.append(f"Retrieved {len(candidates)} candidate sections from the knowledge base: "
                     + ", ".join(c["section"] for c in candidates) + ".")

        # 3. Element assessment
        assessment = None
        if llm_on:
            try:
                assessment = self.llm.assess_elements(case_text, analysis.get("facts", {}), candidates)
                trace.append(f"LLM ({assessment.get('_model')}) checked each legal ingredient of every "
                             "candidate against the narrative.")
            except Exception as e:
                log.warning("assess_elements failed: %s", e)
                trace.append(f"LLM element check failed ({type(e).__name__}); used rule-based fallback.")
                mode = "symbolic_fallback"
        if assessment is None:
            assessment = self.reasoner.heuristic_assessment(case_text, candidates)
            if mode == "symbolic_only":
                trace.append("Rule-based (regex) assessment used for elements and evidence.")

        # 4. Charges
        evaluated = self.reasoner.evaluate_candidates(candidates, assessment)
        ranked = rank_candidates(evaluated, self.weights)
        charges = self.reasoner.select_charges(ranked)
        primary = charges["primary"]
        if primary:
            trace.append(f"Primary charge: s.{primary['charge_section']} (all essential ingredients "
                         f"{'proved' if primary['unclear'] == 0 else 'proved or not contradicted'}).")
        if charges["reduced_from"]:
            rf = charges["reduced_from"]
            trace.append(f"Charge reduced from s.{rf['charge_section']} because: " + "; ".join(rf["failed_elements"]))

        # 5. Verdict
        evidence = self.reasoner.evidence_strength(assessment.get("evidence", {}))
        exception = self.reasoner.check_exception(assessment.get("general_exception"))
        sym_prob = self.reasoner.symbolic_conviction_probability(primary, evidence, exception)
        llm_prob = assessment.get("conviction_probability") if primary else None
        conv_prob = fuse_probability(sym_prob, llm_prob, self.weights)
        verdict = self.reasoner.decide_verdict(charges, conv_prob, exception, best_raw,
                                               analysis.get("is_criminal_matter"))
        if verdict["verdict"] == "INCONCLUSIVE":
            conv_prob = None
        trace.append(f"Verdict: {verdict['verdict']} ({verdict['basis'].replace('_', ' ')}).")

        # 6. Punishment
        charged = [primary] + charges["additional"] if primary else []
        alleged = primary or charges["failed_top"]
        sentencing = [self.reasoner.sentencing(c) for c in charged]
        if not sentencing and alleged:
            sentencing = [self.reasoner.sentencing(alleged)]

        margin = (ranked[0]["fused_score"] - ranked[1]["fused_score"]) if len(ranked) > 1 else 1.0
        return {
            "mode": mode,
            "verdict": verdict["verdict"],
            "verdict_basis": verdict["basis"],
            "verdict_reason": verdict["reason"],
            "conviction_probability": conv_prob,
            "symbolic_probability": sym_prob if primary else None,
            "neural_probability": llm_prob,
            "confidence_level": confidence_level(conv_prob, self.weights["threshold"], margin)
                                if conv_prob is not None else "Low",
            "charges": [_charge_view(c) for c in charged],
            "alleged_charge": _charge_view(alleged) if alleged else None,
            "reduced_from": _charge_view(charges["reduced_from"]) if charges["reduced_from"] else None,
            "punishment": sentencing,
            "punishment_applies": verdict["verdict"] == "CONVICTED",
            "evidence": evidence,
            "general_exception": exception,
            "rationale": assessment.get("rationale") or _rule_rationale(verdict, primary, evidence),
            "summary": analysis.get("summary"),
            "facts": analysis.get("facts"),
            "candidates": [_candidate_view(c) for c in ranked],
            "reasoning_trace": trace,
            "disclaimer": "Decision-support prediction for research/education. Not legal advice. "
                          "IPC was replaced by the Bharatiya Nyaya Sanhita for offences after 1 July 2024.",
        }


def _charge_view(c: dict) -> dict:
    return {"section": c["charge_section"], "matched_section": c["section"], "title": c["title"],
            "elements": c["elements"], "element_fit": c["element_fit"], "fused_score": c["fused_score"],
            "failed_elements": c["failed_elements"], "note": c["note"]}


def _candidate_view(c: dict) -> dict:
    return {k: c[k] for k in ("section", "charge_section", "title", "retrieval_score", "element_fit",
                              "llm_fit", "fused_score", "applicable", "proved", "unclear",
                              "not_proved", "signals", "note")}


def _rule_rationale(verdict: dict, primary: Optional[dict], evidence: dict) -> str:
    parts = [verdict["reason"]]
    if evidence["prosecution"]:
        parts.append("Prosecution evidence: " + ", ".join(e["type"].replace("_", " ") for e in evidence["prosecution"]) + ".")
    if evidence["defence"]:
        parts.append("Weaknesses: " + ", ".join(e["type"].replace("_", " ") for e in evidence["defence"]) + ".")
    if evidence.get("note"):
        parts.append(evidence["note"])
    return " ".join(parts)
