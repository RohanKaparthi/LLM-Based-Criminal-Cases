"""
symbolic_reasoner.py
--------------------
Symbolic half of the neuro-symbolic pipeline. Every decision here is a fixed,
auditable rule over the knowledge base and the element findings:

  R1  A section applies only if none of its ingredients is "not_proved", at least
      one is proved, and at least half are proved (unclear counts half).
  R2  If a greater offence and one of its lesser-included offences both apply,
      only the greater offence is charged.
  R3  If the best-matching offence fails on an ingredient but a lesser-included
      offence applies, the charge is reduced to that lesser offence.
  R4  Definition sections are charged under their punishing section
      (s.378 theft -> s.379, s.300 murder -> s.302).
  R5  A general exception (IPC ss. 76-106) that is established leads to acquittal.
  R6  Conviction requires the fused conviction probability to reach the threshold;
      otherwise the accused gets the benefit of doubt.
  R7  Punishment is read from the curated knowledge base only.

It also contains a regex-based fallback assessment used when no LLM is available.
"""
import re
from typing import Dict, List, Optional

from knowledge_base import KnowledgeBase

STRENGTH = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}
PROSECUTION_TYPE_WEIGHT = {
    "eyewitness": 1.0, "medical": 0.9, "forensic": 1.0, "documentary": 0.8, "recovery": 0.8,
    "electronic": 0.9, "confession": 0.5, "circumstantial": 0.6, "police_testimony": 0.6, "other": 0.5,
}
DEFENCE_TYPE_WEIGHT = {
    "hostile_witness": 0.9, "contradiction": 0.9, "alibi": 1.0, "delay": 0.4,
    "lack_of_corroboration": 0.7, "unreliable_evidence": 0.9, "procedural_lapse": 0.4,
    "false_implication": 0.6, "other": 0.5,
}
EXCEPTION_LAW = {
    "private_defence": "Right of private defence (IPC ss. 96-106)",
    "accident": "Accident in doing a lawful act (IPC s. 80)",
    "insanity": "Act of a person of unsound mind (IPC s. 84)",
    "intoxication": "Involuntary intoxication (IPC s. 85)",
    "child": "Act of a child (IPC ss. 82-83)",
    "mistake_of_fact": "Mistake of fact (IPC ss. 76, 79)",
    "consent": "Act done with consent (IPC ss. 87-89)",
    "necessity": "Act to prevent other harm (IPC s. 81)",
    "judicial_act": "Judicial act / act under judgment (IPC ss. 77-78)",
    "duress": "Act done under threats (IPC s. 94)",
    "other": "General exception (IPC Chapter IV)",
}
FIT_VALUE = {"strong": 1.0, "partial": 0.5, "none": 0.0}

# ---- Regex cues for the no-LLM fallback --------------------------------------
_DEFENCE_CUES = [
    (r"turned hostile|hostile witness", "hostile_witness", "strong"),
    (r"contradict|inconsisten|discrepanc", "contradiction", "strong"),
    (r"\balibi\b", "alibi", "strong"),
    (r"delay(ed)? in (lodging|filing|registering)|belated (fir|complaint)", "delay", "moderate"),
    (r"insufficient (corroborative )?evidence|lack of (evidence|corroboration)|no independent witness|uncorroborated", "lack_of_corroboration", "strong"),
    (r"unreliable|not reliable|doubtful|not trustworthy", "unreliable_evidence", "strong"),
    (r"benefit of (the )?doubt|not proved|unproven|could not (be )?(conclusively )?(establish|prove)|failed to (prove|establish)|not satisfied", "unreliable_evidence", "strong"),
    (r"rested (largely|mainly|solely) on circumstantial", "lack_of_corroboration", "moderate"),
    (r"falsely implicated|false implication|planted", "false_implication", "moderate"),
]
_PROSECUTION_CUES = [
    (r"eye ?witness(es)?\b(?! informed)|witness(es)? (saw|identified|testified)|witness (accounts|testimony|depositions)", "eyewitness", "strong"),
    (r"post[- ]?mortem|medical (report|evidence|examination)|injur(y|ies) report", "medical", "strong"),
    (r"forensic|fingerprint|\bdna\b|ballistic", "forensic", "strong"),
    (r"documentary (evidence|records)|bank records|documents (show|proved)", "documentary", "moderate"),
    (r"recover(ed|y) (of |the )?|recovered material|seized", "recovery", "strong"),
    (r"cctv|video footage|call (detail|records)|electronic evidence", "electronic", "strong"),
    (r"confess", "confession", "moderate"),
    (r"caught red[- ]handed|caught in the act|arrested on the spot", "eyewitness", "strong"),
    (r"corroborat(ed|ion) by|consistent (picture|testimony|witness)|credib(le|ility)|no reasonable doubt|clearly established|found reliable|satisfied with the evidence|failed to shake", "other", "strong"),
    (r"circumstantial and direct evidence", "circumstantial", "moderate"),
]
_EXCEPTION_CUES = [
    (r"self[- ]defen[cs]e|private defen[cs]e|defend(ing|ed) (himself|herself|themselves)", "private_defence"),
    (r"\baccident(al|ally)?\b", "accident"),
    (r"(accused|he|she) (was|is) (of unsound mind|insane|mentally ill)|plea of insanity|insanity defen[cs]e", "insanity"),
    (r"(aged|age of|years old|a child of) (\d|six|five|four|three)\b", "child"),
]


class SymbolicReasoner:
    def __init__(self, kb: KnowledgeBase, threshold: float = 0.5):
        self.kb = kb
        self.threshold = threshold

    # ------------------------------------------------------------------ fallback
    def heuristic_assessment(self, case_text: str, candidates: List[dict]) -> dict:
        """Regex-only stand-in for the LLM element assessment (no API needed)."""
        text = case_text.lower()
        defence = self._scan(text, _DEFENCE_CUES)
        # Remove defence phrases first so "not proved beyond reasonable doubt" is not
        # read as prosecution evidence.
        stripped = text
        for pattern, _, _ in _DEFENCE_CUES:
            stripped = re.sub(pattern, " ", stripped)
        prosecution = self._scan(stripped, _PROSECUTION_CUES)
        p_total = sum(STRENGTH[e["strength"]] for e in prosecution)
        d_total = sum(STRENGTH[e["strength"]] for e in defence)

        exc = next((t for pattern, t in _EXCEPTION_CUES if re.search(pattern, text)), "none")
        out_candidates = []
        for c in candidates:
            named = bool({"offence named in text", "section cited in text"} & set(c.get("signals", [])))
            if named:
                # The charge is named; whether it is proved depends on which side's evidence prevails.
                status = "proved" if p_total > d_total else ("not_proved" if d_total > p_total else "unclear")
            else:
                status = "proved" if self._ingredient_overlap(text, c) else "unclear"
            out_candidates.append({
                "section": c["section"],
                "elements": [{"id": i["id"], "status": status, "evidence": ""} for i in c["ingredients"]],
                "fit": "strong" if named else ("partial" if c["retrieval_score"] >= 0.6 else "none"),
                "note": "Rule-based match (no LLM): " + (", ".join(c.get("signals")) or "keyword similarity"),
            })
        return {
            "candidates": out_candidates,
            "evidence": {"prosecution": prosecution, "defence": defence},
            "general_exception": {"type": exc, "established": "unclear" if exc != "none" else "no",
                                  "detail": "Claimed in the narrative; cannot be verified without the LLM." if exc != "none" else ""},
            "conviction_probability": None,
            "rationale": None,
        }

    @staticmethod
    def _scan(text: str, cues) -> List[dict]:
        found, seen = [], set()
        for pattern, etype, strength in cues:
            m = re.search(pattern, text)
            if m and (etype, m.group(0)) not in seen:
                seen.add((etype, m.group(0)))
                found.append({"type": etype, "strength": strength, "detail": m.group(0)})
        return found

    @staticmethod
    def _ingredient_overlap(text: str, cand: dict) -> bool:
        words = {w[:6] for i in cand["ingredients"] for w in re.findall(r"[a-z]{5,}", i["text"].lower())}
        words -= {"accus", "victi", "perso", "prope", "inten", "knowl", "commi", "offen", "anoth"}
        if not words:
            return False
        hits = sum(1 for w in words if w in text)
        return hits / len(words) >= 0.4

    # --------------------------------------------------------- element evaluation
    def evaluate_candidates(self, candidates: List[dict], assessment: dict) -> List[dict]:
        findings = {self.kb.normalize(c.get("section")): c for c in assessment.get("candidates", [])}
        evaluated = []
        for c in candidates:
            f = findings.get(c["section"], {})
            statuses = {e.get("id"): e for e in f.get("elements", [])}
            elements = []
            for ing in c["ingredients"]:
                e = statuses.get(ing["id"], {})
                status = e.get("status") if e.get("status") in ("proved", "not_proved", "unclear") else "unclear"
                elements.append({"id": ing["id"], "text": ing["text"], "status": status,
                                 "evidence": e.get("evidence", "")})
            n = max(len(elements), 1)
            proved = sum(e["status"] == "proved" for e in elements)
            unclear = sum(e["status"] == "unclear" for e in elements)
            failed = [e for e in elements if e["status"] == "not_proved"]
            element_fit = (proved + 0.5 * unclear) / n if elements else 0.0
            llm_fit = f.get("fit", "none") if f else "none"
            evaluated.append({
                **{k: c[k] for k in ("section", "title", "retrieval_score", "signals")},
                "charge_section": self.kb.charge_section(c["section"]),
                "severity": self.kb.sections[self.kb.charge_section(c["section"])]["severity"],
                "elements": elements,
                "element_fit": round(element_fit, 3),
                "proved": proved, "unclear": unclear, "not_proved": len(failed),
                "failed_elements": [e["text"] for e in failed],
                "llm_fit": llm_fit if llm_fit in FIT_VALUE else "none",
                "note": f.get("note", ""),
                # R1
                "applicable": not failed and proved >= 1 and element_fit >= 0.5 and llm_fit != "none",
            })
        return evaluated

    # ------------------------------------------------------------ charge selection
    def select_charges(self, ranked: List[dict]) -> dict:
        """`ranked` is sorted by fused candidate score (see fusion.rank_candidates)."""
        applicable = [c for c in ranked if c["applicable"]]

        # R2: drop lesser-included offences when a greater offence is made out (no failed
        # ingredient, and either a strong overall fit or at least as many ingredients proved);
        # also collapse definition/punishing duplicates (R4).
        kept = []
        for c in applicable:
            dominated = any(
                g is not c and (g["llm_fit"] == "strong" or g["element_fit"] >= c["element_fit"])
                and self.kb.is_lesser(c["section"], g["section"])
                and not self.kb.is_lesser(g["section"], c["section"])   # ignore inconsistent cycles
                for g in applicable)
            duplicate = any(self.kb.same_offence(c["section"], k["section"]) for k in kept)
            if not dominated and not duplicate:
                kept.append(c)

        # Charge the gravest offence among those that match about as well as the best one.
        primary = None
        if kept:
            pool = [k for k in kept if k["fused_score"] >= 0.8 * kept[0]["fused_score"]]
            # Severity only breaks ties among strong fits; weaker matches go by score.
            primary = max(pool, key=lambda k: (FIT_VALUE[k["llm_fit"]],
                                               k["severity"] if k["llm_fit"] == "strong" else 0,
                                               k["fused_score"]))
            kept.remove(primary)
            kept.insert(0, primary)
        # Offences that matched the narrative well but failed on an ingredient, best match first.
        # (Their failed ingredient lowers their fused score, so rank them by match strength instead.)
        failed = sorted((c for c in ranked if not c["applicable"] and c["not_proved"] and c["llm_fit"] != "none"),
                        key=lambda c: c["retrieval_score"] + FIT_VALUE[c["llm_fit"]], reverse=True)

        # R3: a greater offence failed on an ingredient, but a lesser-included one holds.
        reduced_from = next((f for f in failed if primary and self.kb.is_lesser(primary["section"], f["section"])), None)

        # The best-matching (alleged) offence failed: don't swap in an unrelated weak match.
        if failed and primary and not reduced_from and primary["llm_fit"] != "strong":
            f = failed[0]
            if f["retrieval_score"] + FIT_VALUE[f["llm_fit"]] > primary["retrieval_score"] + FIT_VALUE[primary["llm_fit"]]:
                kept.remove(primary)
                primary = None

        additional = [c for c in kept[1:]
                      if c["element_fit"] == 1.0 and c["llm_fit"] == "strong"
                      and primary and c["fused_score"] >= 0.5 * primary["fused_score"]][:2]

        return {"primary": primary, "additional": additional, "reduced_from": reduced_from,
                "failed_top": failed[0] if failed else None}

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def evidence_strength(evidence: dict) -> dict:
        pros = evidence.get("prosecution", []) or []
        dfn = evidence.get("defence", []) or []
        p = sum(STRENGTH.get(e.get("strength"), 0.3) * PROSECUTION_TYPE_WEIGHT.get(e.get("type"), 0.5) for e in pros)
        d = sum(STRENGTH.get(e.get("strength"), 0.3) * DEFENCE_TYPE_WEIGHT.get(e.get("type"), 0.5) for e in dfn)
        if not pros and not dfn:
            score, note = 0.65, "No evidence is described; the narrated facts are assumed to be proved."
        else:
            score, note = 0.5 + 0.2 * p - 0.25 * d, None
        return {"score": round(min(0.97, max(0.03, score)), 3), "prosecution_weight": round(p, 2),
                "defence_weight": round(d, 2), "prosecution": pros, "defence": dfn, "note": note}

    @staticmethod
    def check_exception(exc: Optional[dict]) -> dict:
        exc = exc or {}
        etype = exc.get("type") or "none"
        established = exc.get("established") or "no"
        if etype == "none":
            established = "no"
        return {"type": etype, "established": established, "detail": exc.get("detail", ""),
                "law": EXCEPTION_LAW.get(etype)}

    # ------------------------------------------------------------------ verdict
    @staticmethod
    def symbolic_conviction_probability(primary: Optional[dict], evidence: dict, exception: dict) -> float:
        if not primary:
            return 0.0
        prob = primary["element_fit"] * evidence["score"]
        if exception["established"] == "yes":
            prob = min(prob, 0.05)
        elif exception["established"] == "unclear":
            prob *= 0.6
        return round(prob, 3)

    def decide_verdict(self, charges: dict, conviction_prob: float, exception: dict,
                       best_retrieval_raw: float, is_criminal: Optional[bool]) -> dict:
        primary = charges["primary"]
        failed = charges["failed_top"]
        if (primary or failed) and exception["established"] == "yes":   # R5
            return {"verdict": "ACQUITTED", "basis": "general_exception",
                    "reason": f"{exception['law']}: {exception['detail']}".strip(": ")}
        if not primary:
            if failed:
                return {"verdict": "ACQUITTED", "basis": "charge_not_proved",
                        "reason": f"Essential ingredient(s) of s.{failed['charge_section']} not proved: "
                                  + "; ".join(failed["failed_elements"])}
            if is_criminal is False or best_retrieval_raw < 0.25:
                return {"verdict": "INCONCLUSIVE", "basis": "no_offence_identified",
                        "reason": "The narrative does not clearly disclose any of the 213 IPC offences in the knowledge base."}
            return {"verdict": "INCONCLUSIVE", "basis": "insufficient_facts",
                    "reason": "The facts given are not enough to establish the ingredients of any offence. Add details of the act, intention and evidence."}
        if primary["proved"] == 0:
            return {"verdict": "INCONCLUSIVE", "basis": "insufficient_facts",
                    "reason": f"s.{primary['charge_section']} ({primary['title']}) is the closest match, but none of its "
                              "ingredients could be confirmed from the narrative. Add details of the act, intention and evidence."}
        if conviction_prob >= self.threshold:   # R6
            return {"verdict": "CONVICTED", "basis": "ingredients_proved",
                    "reason": f"All essential ingredients of s.{primary['charge_section']} are made out and the evidence meets the required standard."}
        return {"verdict": "ACQUITTED", "basis": "benefit_of_doubt",
                "reason": "The prosecution evidence is not strong enough to prove the charge beyond reasonable doubt; the accused gets the benefit of doubt."}

    # ------------------------------------------------------------------ sentencing
    def sentencing(self, cand: dict) -> dict:
        """R7: prescribed punishment for a charge, straight from the knowledge base."""
        sec = cand["charge_section"]
        entry = self.kb.sections[sec]
        p = entry["punishment"]
        # Punishing section outside the 213-section KB (e.g. 171B bribery -> 171E).
        external = p.get("punishable_under") if p.get("punishable_under") not in self.kb else None
        return {
            "section": sec,
            "title": entry["title"],
            "defined_in": cand["section"] if cand["section"] != sec else None,
            "punishable_under": external,
            "punishable_under_title": p.get("punishable_under_title") if external else None,
            "punishment_text": p["text"],
            "summary": punishment_summary(p),
            "max_years": p["max_years"], "min_years": p["min_years"],
            "life": p["life"], "death": p["death"], "fine": p["fine"],
            "imprisonment_kind": p["kind"],
            "cognizable": p["cognizable"], "bailable": p["bailable"], "triable_by": p["triable_by"],
            "note": p.get("note"),
            "severity": entry["severity"],
        }


def _years(y: float) -> str:
    if y is None:
        return ""
    if y < 1 / 12:
        return f"{round(y * 365)} days" if y * 365 >= 2 else "24 hours"
    if y < 1:
        m = round(y * 12)
        return f"{m} month" + ("s" if m != 1 else "")
    return f"{y:g} year" + ("s" if y != 1 else "")


def punishment_summary(p: dict) -> str:
    """One-line structured summary, e.g. 'Death or life imprisonment + fine'."""
    parts = []
    if p["death"]:
        parts.append("Death")
    if p["life"]:
        parts.append("Life imprisonment")
    kind = {"rigorous": "rigorous ", "simple": "simple "}.get(p["kind"] or "", "")
    if p["min_years"] and p["life"] and not p["max_years"]:
        # e.g. "Minimum 7 years, up to life imprisonment"
        parts = (["Death"] if p["death"] else []) + \
                [f"Minimum {_years(p['min_years'])} {kind}imprisonment, up to life imprisonment"]
    elif p["min_years"] and p["max_years"]:
        parts.append(f"{kind}imprisonment {_years(p['min_years'])} to {_years(p['max_years'])}".capitalize())
    elif p["min_years"]:
        parts.append(f"Minimum {_years(p['min_years'])} {kind}imprisonment")
    elif p["max_years"]:
        parts.append(f"Up to {_years(p['max_years'])} {kind}imprisonment")
    s = " or ".join(parts) if parts else ""
    if p["fine"] == "and":
        s += " + fine"
    elif p["fine"] == "or":
        s += ", or fine, or both" if s else "Fine"
    elif p["fine"] == "only":
        s = "Fine only"
    return s or p["text"]
