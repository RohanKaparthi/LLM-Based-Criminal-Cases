"""
gen_narratives.py
-----------------
Generates a section-prediction test set that does NOT leak its labels: for every
IPC section in the knowledge base, two realistic case narratives that describe the
facts and evidence without naming the offence or section.

  label 1 - evidence clearly establishes every ingredient (conviction expected)
  label 0 - same kind of allegation, but the evidence has real weaknesses (acquittal expected)

Output: data/eval_narratives.csv  (ipc_section, label, text)

Caveat for reporting: the narratives are LLM-generated and the pipeline also uses an
LLM, so results may be optimistic; spot-check a sample by hand.

Usage (from backend/):  python gen_narratives.py [--sections 302 379 ...]
"""
import argparse
import csv
import json
from pathlib import Path

from groq_client import GroqJSONClient, REASONING_MODEL
from knowledge_base import KnowledgeBase

OUT = Path(__file__).parent / "data" / "eval_narratives.csv"
BATCH = 3

SYSTEM = """You write realistic case narratives for testing an Indian criminal-law (IPC)
reasoning system. Answer with a single JSON object."""

USER = """For EACH IPC section below write two case narratives of 90-160 words, in the style of a
police / trial summary from India, with Indian names and places:
  - label 1: the evidence clearly establishes every ingredient of the offence
  - label 0: the same kind of allegation, but the evidence has real weaknesses (e.g. hostile or
    contradictory witnesses, alibi, no corroboration, unreliable documents) so a court would acquit

Rules:
- Describe concrete acts, the accused's state of mind, the consequences and the evidence.
- Do NOT mention any section number, and do NOT name the offence by its legal name
  (e.g. avoid "murder", "theft", "robbery", "cheating", "forgery", "dacoity") - describe what happened.
- Do NOT state the verdict or outcome of the trial.

Sections:
{sections}

Return JSON: {{"cases": [{{"section": "<id>", "label": 1, "text": "..."}}, {{"section": "<id>", "label": 0, "text": "..."}}, ...]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*")
    args = ap.parse_args()

    kb = KnowledgeBase()
    client = GroqJSONClient()
    existing = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            existing = {r["ipc_section"] for r in csv.DictReader(f)}
    todo = [s for s in (args.sections or kb.sections) if s in kb and s not in existing]

    new_file = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ipc_section", "label", "text"])
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            listing = "\n".join(f'- "{s}": {kb.sections[s]["title"]} - {kb.sections[s]["description"]}' for s in batch)
            print(f"[gen] {i + len(batch)}/{len(todo)}: {', '.join(batch)}", flush=True)
            try:
                out = client.chat_json(REASONING_MODEL, SYSTEM, USER.format(sections=listing),
                                       max_tokens=5000, reasoning_effort="low", temperature=0.7)
            except Exception as e:
                print(f"  failed: {e}")
                continue
            for case in out.get("cases", []):
                sec = kb.normalize(case.get("section"))
                if sec in batch and case.get("text") and case.get("label") in (0, 1, "0", "1"):
                    w.writerow([sec, int(case["label"]), " ".join(case["text"].split())])
            f.flush()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
