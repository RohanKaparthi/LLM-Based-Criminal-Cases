"""
retriever.py
------------
Candidate-section retrieval over the 213-section knowledge base.

Narrows the search space before any reasoning, so the LLM only ever sees a short
list of plausible sections. Three signals are combined:

  1. TF-IDF similarity between the case text and each section's
     title / description / ingredients / keywords (word + character n-grams).
  2. Offence-name matches: the case text names an offence title
     ("accused of criminal breach of trust"); longer, more specific titles win.
  3. Explicit section references ("u/s 420 IPC") - the strongest signal.
"""
import re
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

from knowledge_base import KnowledgeBase


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.ids = list(kb.sections)
        docs = []
        for sec in self.ids:
            e = kb.sections[sec]
            ingredients = " ".join(i["text"] for i in e.get("ingredients", []))
            keywords = " ; ".join(e.get("keywords", []))
            docs.append(" ".join([e["title"]] * 3 + [e.get("description", ""), ingredients,
                                                     keywords, keywords, e.get("distinguishing_note", "")]))
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.doc_matrix = normalize(hstack([self.word_vec.fit_transform(docs),
                                            0.5 * self.char_vec.fit_transform(docs)]).tocsr())
        self.titles = {sec: _norm_text(kb.sections[sec]["title"]) for sec in self.ids}

    def _title_matches(self, text_n: str) -> Dict[str, float]:
        hits = {sec: len(t) for sec, t in self.titles.items()
                if len(t) >= 4 and re.search(rf"\b{re.escape(t)}\b", text_n)}
        if not hits:
            return {}
        longest = max(hits.values())
        # The most specific (longest) matching title gets the full boost.
        return {sec: (0.6 if n == longest else 0.3) for sec, n in hits.items()}

    def retrieve(self, case_text: str, top_k: int = 10, extra: List[str] = None) -> List[dict]:
        q = normalize(hstack([self.word_vec.transform([case_text]),
                              0.5 * self.char_vec.transform([case_text])]).tocsr())
        sims = (self.doc_matrix @ q.T).toarray().ravel()
        top_sim = sims.max() or 1.0

        title_boost = self._title_matches(_norm_text(case_text))
        mentioned = set(self.kb.find_section_mentions(case_text))
        extra = set(extra or [])

        results = []
        for i, sec in enumerate(self.ids):
            tfidf = sims[i] / top_sim
            score = 0.5 * tfidf + title_boost.get(sec, 0.0)
            signals = []
            if title_boost.get(sec, 0) >= 0.6:
                signals.append("offence named in text")
            elif title_boost.get(sec):
                signals.append("part of offence name in text")
            if sec in mentioned:
                score += 1.0
                signals.append("section cited in text")
            if sec in extra:
                score += 0.4
                signals.append("proposed by LLM fact analysis")
            results.append({"section": sec, "retrieval_score": float(score),
                            "tfidf": round(float(sims[i]), 4), "signals": signals})

        results.sort(key=lambda r: r["retrieval_score"], reverse=True)
        top = results[:top_k]
        # Make sure every explicitly cited / LLM-proposed section is considered.
        for r in results[top_k:]:
            if r["signals"] and r["section"] not in {t["section"] for t in top}:
                top.append(r)
        best = top[0]["retrieval_score"] if top else 1.0
        for r in top:
            r["raw_score"] = round(r["retrieval_score"], 4)
            r["retrieval_score"] = round(min(1.0, r["retrieval_score"] / max(best, 1e-9)), 4)
            r["title"] = self.kb.sections[r["section"]]["title"]
        return top
