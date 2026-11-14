"""
llm_engine.py
-------------
Neural half of the neuro-symbolic pipeline (Groq-hosted open models).

The LLM is used only for what rules cannot do well: reading a free-text
narrative. It never decides the verdict or the punishment directly.

  1. analyze_case()     [fast model]      - structured facts + offences the facts suggest
                                            (used to widen candidate retrieval)
  2. assess_elements()  [reasoning model] - for each candidate section, mark every legal
                                            ingredient proved / not_proved / unclear with a
                                            quote; grade the evidence; check general
                                            exceptions; give an independent conviction estimate
"""
import json
from typing import List

from groq_client import GroqJSONClient, LLMError, FAST_MODEL, REASONING_MODEL

ANALYZE_SYSTEM = """You are a legal analyst for Indian criminal law (IPC, 1860).
You read case narratives and extract facts precisely. You never invent facts that are
not in the narrative. Answer with a single JSON object."""

ANALYZE_USER = """Read the case narrative and return JSON with:
- "is_criminal_matter": true if the narrative describes conduct that could be an offence
- "summary": 1-2 sentence neutral summary of what happened
- "accused": list of names/descriptions of accused persons ([] if unknown)
- "victims": list of names/descriptions of victims ([] if unknown)
- "facts": {{
    "acts": list of the accused's acts, as stated,
    "mental_state": what the narrative says about intention, knowledge, planning, motive, rashness or negligence,
    "consequences": e.g. death, injuries (simple/grievous), property loss, fear, reputational harm,
    "property": property involved, if any,
    "weapon_or_means": weapon or means used, if any,
    "relationship": relationship between accused and victim, if stated,
    "consent": what the narrative says about the victim's consent, if relevant,
    "circumstances": place, time, number of offenders, public servant involvement, etc.
  }}
- "suspected_offences": up to 5 IPC sections the facts most plausibly disclose, most likely first,
  each {{"section": "302", "reason": "short reason"}}. Use the precise section that
  prescribes the punishment (e.g. 379 for theft, 302 for murder).

Case narrative:
<<<CASE
{case_text}
CASE>>>"""

ASSESS_SYSTEM = """You are an Indian criminal-trial judge applying the Indian Penal Code, 1860.
Your job is to check, ingredient by ingredient, whether the narrative establishes each
candidate offence, and to grade the evidence. Be strict: the prosecution must prove every
ingredient beyond reasonable doubt. Base every finding only on the narrative.
Answer with a single JSON object."""

ASSESS_USER = """Case narrative:
<<<CASE
{case_text}
CASE>>>

Facts extracted earlier (may be incomplete):
{facts}

Candidate IPC sections, each with the ingredients the prosecution must prove:
{candidates}

For EVERY candidate, mark EVERY ingredient (use the given ids) as:
  "proved"     - the narrative states facts or evidence establishing it
  "not_proved" - the narrative shows it is absent, disproved, or the evidence for it failed
                 (e.g. contradicted, witnesses hostile, found unreliable)
  "unclear"    - the narrative does not address it
If the narrative states that the evidence established the accused's guilt of the named
offence, treat that offence's ingredients as proved unless something contradicts it.

General exceptions (IPC ss. 76-106, e.g. private defence ss. 96-106, accident s.80, insanity s.84):
the accused needs to show them only on a preponderance of probabilities, not beyond reasonable
doubt. Mark "established": "yes" if the narrative's facts satisfy the exception's conditions
(for private defence: a reasonable apprehension of death or grievous hurt from the victim's
attack, and force not clearly excessive), "no" if the facts show it does not apply, and
"unclear" only if the narrative does not give enough facts to judge.

Return JSON:
{{
  "candidates": [
    {{"section": "<id>",
      "elements": [{{"id": "<ingredient id>", "status": "proved|not_proved|unclear",
                    "evidence": "short quote or paraphrase from the narrative, or empty"}}],
      "fit": "strong|partial|none",
      "note": "one sentence: why this section does or does not fit"}}
  ],
  "evidence": {{
    "prosecution": [{{"type": "eyewitness|medical|forensic|documentary|recovery|confession|electronic|circumstantial|police_testimony|other",
                     "strength": "strong|moderate|weak", "detail": "short"}}],
    "defence":     [{{"type": "hostile_witness|contradiction|alibi|delay|lack_of_corroboration|unreliable_evidence|procedural_lapse|false_implication|other",
                     "strength": "strong|moderate|weak", "detail": "short"}}]
  }},
  "general_exception": {{"type": "none|private_defence|accident|insanity|intoxication|child|mistake_of_fact|consent|necessity|judicial_act|duress|other",
                        "established": "yes|no|unclear", "detail": "short"}},
  "conviction_probability": <0.0-1.0, your estimate that a court convicts the accused of the best-fitting offence>,
  "rationale": "3-5 sentences of legal reasoning citing the facts and the ingredients"
}}"""


class LLMEngine:
    def __init__(self, api_key: str = None):
        self.client = GroqJSONClient(api_key)

    def analyze_case(self, case_text: str) -> dict:
        return self.client.chat_json(
            FAST_MODEL, ANALYZE_SYSTEM, ANALYZE_USER.format(case_text=case_text),
            max_tokens=2500, reasoning_effort="low", max_wait=30)

    def assess_elements(self, case_text: str, facts: dict, candidates: List[dict]) -> dict:
        cand_lines = []
        for c in candidates:
            ingredients = "\n".join(f'      - {i["id"]}: {i["text"]}' for i in c["ingredients"])
            note = f'\n    note: {c["distinguishing_note"]}' if c.get("distinguishing_note") else ""
            cand_lines.append(f'  - section "{c["section"]}": {c["title"]}{note}\n    ingredients:\n{ingredients}')
        facts_str = json.dumps(facts, ensure_ascii=False) if facts else "(none)"
        user = ASSESS_USER.format(case_text=case_text, facts=facts_str, candidates="\n".join(cand_lines))
        # Each Groq model has its own quota: if the reasoning model is rate-limited or out of
        # daily tokens, the smaller model does the element check instead of the regex fallback.
        try:
            result = self.client.chat_json(REASONING_MODEL, ASSESS_SYSTEM, user,
                                           max_tokens=5000, reasoning_effort="low", max_wait=20)
            result["_model"] = REASONING_MODEL
        except LLMError:
            result = self.client.chat_json(FAST_MODEL, ASSESS_SYSTEM, user,
                                           max_tokens=5000, reasoning_effort="medium", max_wait=60)
            result["_model"] = FAST_MODEL
        return result
