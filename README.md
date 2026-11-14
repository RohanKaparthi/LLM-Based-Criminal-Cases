# LLM-Based Neuro-Symbolic Legal Judgment Prediction (Criminal Cases)

Given a criminal case narrative, the system identifies the applicable IPC (Indian
Penal Code) section(s) out of **213 sections**, predicts the **verdict**
(convicted / acquitted / inconclusive), and states the **punishment** prescribed by law,
with an ingredient-by-ingredient explanation.

It combines:

- **Neural reasoning:** Groq-hosted LLMs (`gpt-oss-20b`, `gpt-oss-120b`) read the
  free-text narrative. They extract facts and judge whether each legal ingredient is proved.
- **Symbolic reasoning:** a 213-section knowledge base and fixed legal rules decide the
  charge, the verdict and the punishment. The LLM never decides the punishment.

## Architecture

```
Case narrative
   │
   ▼
1. Fact analysis          LLM (gpt-oss-20b)       facts + offences the facts suggest
   │
   ▼
2. Candidate retrieval    symbolic (TF-IDF)       top ~8-12 of 213 sections:
   │                                              similarity, offence names, "u/s 302" citations,
   │                                              LLM proposals, lesser-included offences
   ▼
3. Element assessment     LLM (gpt-oss-120b)      each legal ingredient of each candidate:
   │                                              proved / not_proved / unclear + quote;
   │                                              evidence grading; general exceptions
   ▼
4. Charge selection       symbolic rules          R1 all ingredients hold   R2 greater absorbs lesser
   │                                              R3 failed grave charge → lesser offence
   │                                              R4 definition → punishing section (378 → 379)
   ▼
5. Verdict                symbolic + fusion       R5 general exception → acquittal
   │                                              R6 p = α·p_symbolic + (1-α)·p_llm ≥ threshold → convicted
   ▼
6. Punishment             curated KB lookup       R7 punishment, min/max, fine, cognizable,
                                                  bailable, trial court
```

If the LLM is unavailable (no key, rate limit, error), steps 1 and 3 fall back to
regex rules and the response reports `mode: "symbolic_only"`.

## Knowledge base

`backend/data/ipc_kb.json` is built by `python -m kb.build_kb` from these sources, most authoritative first:

| Source | What it provides | How it was made |
|---|---|---|
| `kb/punishments.py` | punishment text, min/max term, life/death, fine, cognizable, bailable, trial court, definition→punishing section | written by hand from the IPC First Schedule |
| `kb/ingredient_overrides.py` | legal ingredients for 66 frequently charged sections | written by hand |
| `data/kb_drafts.json` | description, ingredients, keywords, category for all 213 sections | drafted once by `gpt-oss-120b`; cached; editable |
| `knowledge_base.py` `CURATED_LESSER` | greater → lesser-included offence hierarchy | written by hand, merged with LLM suggestions |

Severity (1–10) is computed from the punishment, not guessed.

**Review needed before relying on it:** check the LLM-drafted ingredients for sections not in
`ingredient_overrides.py` against the bare act. Entries where cognizable/bailable is `None` are
marked "Verify" in the UI.

## Evaluation

```
cd backend
python evaluate.py --data dataset --mask sections --no-llm   # all 4,260 rows, ~1 min
python evaluate.py --data dataset --mask full --no-llm       # shows the label leakage
python gen_narratives.py                                      # build leak-free test set (LLM)
python evaluate.py --data narratives --limit 100              # LLM pipeline on narratives
python evaluate.py --data narratives --limit 100 --calibrate  # also fit fusion weights
```

**About `custom_ipc_full_dataset_large.csv`:** every row contains its answer. The text names the
section ("convicted under Section 420 IPC"), the offence title, and the verdict, and is
generated from just 21 template sentences. `evaluate.py` masks section numbers and verdict
words. Even so, scores on this file are an upper bound. With the offence name also removed,
section accuracy drops to chance. `data/eval_narratives.csv` (from `gen_narratives.py`)
describes facts without naming the offence, so it is the fairer test. It is LLM-generated,
though, so spot-check it.

Reports are written to `backend/reports/`.

## Project structure

```
backend/
├── main.py               FastAPI app: /predict, /health, /sections, /sections/{id}
├── pipeline.py           end-to-end judgment pipeline (steps 1–6)
├── llm_engine.py         LLM prompts: fact analysis, element assessment
├── groq_client.py        Groq wrapper: JSON mode, retries on rate limits, SQLite response cache
├── retriever.py          TF-IDF + offence-name + citation candidate retrieval
├── symbolic_reasoner.py  rules R1–R7, evidence scoring, regex fallback
├── knowledge_base.py     KB loader, section normalisation, offence hierarchy
├── fusion.py             candidate-score and conviction-probability fusion
├── evaluate.py           benchmarking + weight calibration
├── gen_narratives.py     leak-free narrative test-set generator
├── kb/                   build_kb.py, punishments.py, ingredient_overrides.py
├── data/                 ipc_kb.json, kb_drafts.json, ipc_rules.json (legacy), eval_narratives.csv
└── tests/                unit tests for the symbolic layer (no API calls)
frontend/
└── src/App.jsx           UI: verdict, charges + punishment, ingredient checklist, evidence, trace
```

## How to run

### 1. Groq API key
Sign up at console.groq.com and create a key. The free tier allows 8,000 tokens per
minute per model, so one case takes about 5–30 s. LLM responses are cached in
`backend/.cache/`, so a repeated case is answered instantly.

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt
copy .env.example .env             # then put your GROQ_API_KEY in .env
python -m kb.build_kb --no-llm     # rebuild KB from the committed drafts (no API calls)
python -m unittest discover -s tests
uvicorn main:app --reload --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

Describe the case: what the accused did, their intention, the result, and the evidence
(witnesses, medical reports, recoveries, contradictions, defences). Then click **Assess case**.

## Limitations

- This is a research prototype for decision support and is not legal advice.
- The IPC was replaced by the Bharatiya Nyaya Sanhita (BNS) on 1 July 2024. The IPC still governs
  offences committed before that date. An IPC → BNS mapping is a natural next step.
- Coverage is limited to the 213 sections in the dataset. Abetment (s.109), common intention (s.34)
  and conspiracy (s.120B) are not modelled.
- The system gives the prescribed punishment range. The actual sentence within that range is left
  to the court, which weighs aggravating and mitigating factors.
- The evidence grading is a heuristic. It is not a substitute for the Indian Evidence Act rules on
  admissibility; for example, confessions to police are inadmissible under s.25.
