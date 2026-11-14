"""
evaluate.py
-----------
Benchmarks the judgment pipeline on section prediction and verdict prediction.

Datasets
  --data dataset     custom_ipc_full_dataset_large.csv (213 sections x 20 cases).
                     Every row literally contains its answer ("...convicted X under
                     Section 420 IPC"), so the text is masked before prediction:
                       --mask sections : remove section numbers + verdict words
                                         (keeps the offence name, like an FIR/charge-sheet)
                       --mask full     : also remove the offence name
                                         (shows the dataset has no other section signal)
  --data narratives  data/eval_narratives.csv from gen_narratives.py: realistic
                     narratives that describe facts without naming the offence.

Usage (from backend/)
  python evaluate.py --data dataset --no-llm                 # all 4,260 rows, fast
  python evaluate.py --data dataset --limit 60               # LLM mode (rate-limited)
  python evaluate.py --data narratives --limit 100 --calibrate
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from fusion import rank_candidates, fuse_probability, WEIGHTS_PATH, load_weights
from pipeline import JudgmentPipeline

BACKEND = Path(__file__).parent
DATASET = BACKEND.parent / "custom_ipc_full_dataset_large.csv"
NARRATIVES = BACKEND / "data" / "eval_narratives.csv"
REPORTS = BACKEND / "reports"

_SECTION_RE = re.compile(r"(?:under\s+)?Section\s+\w+(?:\s+(?:IPC|of the Indian Penal Code))?", re.I)
_VERDICT_RE = re.compile(
    r"\b(?:convict(?:ed|ion|s)?|acquit(?:ted|tal|s)?|(?:not\s+)?guilty|"
    r"(?:giving|extended)\s+(?:the\s+)?benefit\s+of\s+(?:the\s+)?doubt|benefit\s+of\s+(?:the\s+)?doubt)\b", re.I)


def mask_text(text: str, title: str, level: str) -> str:
    t = _SECTION_RE.sub("[...]", text)
    t = _VERDICT_RE.sub("[...]", t)
    if level == "full":
        t = re.sub(re.escape(title), "[offence]", t, flags=re.I)
    return t


def load_data(args) -> pd.DataFrame:
    if args.data == "dataset":
        df = pd.read_csv(DATASET)
        df["ipc_section"] = df["ipc_section"].astype(str)
        df["input_text"] = [mask_text(t, s, args.mask) for t, s in zip(df.text, df.section_title)]
    else:
        if not NARRATIVES.exists():
            raise SystemExit(f"{NARRATIVES} missing - run: python gen_narratives.py")
        df = pd.read_csv(NARRATIVES)
        df["ipc_section"] = df["ipc_section"].astype(str)
        df["input_text"] = df["text"]
    if args.limit and args.limit < len(df):
        # Spread the sample across sections instead of taking the first rows.
        df = (df.sample(frac=1, random_state=args.seed)
                .assign(_r=lambda d: d.groupby("ipc_section").cumcount())
                .sort_values(["_r", "ipc_section"]).head(args.limit).drop(columns="_r"))
    return df.reset_index(drop=True)


def section_hit(kb, gold: str, predicted) -> bool:
    return predicted is not None and kb.same_offence(gold, predicted)


def run(args):
    pipe = JudgmentPipeline(use_llm=not args.no_llm)
    if not args.no_llm and pipe.llm is None:
        raise SystemExit(f"LLM unavailable: {pipe.llm_error}")
    df = load_data(args)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        out = pipe.predict(r.input_text, use_llm=not args.no_llm)
        ranked_sections = [c["section"] for c in out["candidates"]]
        if out["alleged_charge"]:
            ranked_sections.insert(0, out["alleged_charge"]["matched_section"])
        primary = out["charges"][0]["matched_section"] if out["charges"] else (
            out["alleged_charge"]["matched_section"] if out["alleged_charge"] else None)
        rows.append({
            "case_id": r.get("case_id", i), "gold_section": r.ipc_section, "gold_label": int(r.label),
            "pred_section": primary, "pred_verdict": out["verdict"],
            "section_top1": section_hit(pipe.kb, r.ipc_section, primary),
            "section_top3": any(section_hit(pipe.kb, r.ipc_section, s) for s in ranked_sections[:3]),
            "verdict_correct": (out["verdict"] == "CONVICTED") == (int(r.label) == 1),
            "symbolic_prob": out["symbolic_probability"], "neural_prob": out["neural_probability"],
            "fused_prob": out["conviction_probability"], "mode": out["mode"],
            "_candidates": out["candidates"],
        })
        if (i + 1) % 25 == 0 or not args.no_llm:
            done = pd.DataFrame(rows)
            print(f"[{i + 1}/{len(df)}] top1={done.section_top1.mean():.3f} "
                  f"verdict={done.verdict_correct.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    metrics = summarize(res)
    tag = f"{args.data}{'_' + args.mask if args.data == 'dataset' else ''}_{'symbolic' if args.no_llm else 'llm'}"
    REPORTS.mkdir(exist_ok=True)
    res.drop(columns="_candidates").to_csv(REPORTS / f"eval_{tag}.csv", index=False)
    report = {"tag": tag, "n": len(res), "metrics": metrics, "weights": pipe.weights}
    if args.calibrate:
        report["calibration"] = calibrate(pipe, res, write=not args.dry_run)
    (REPORTS / f"eval_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.data == "dataset":
        print("\nNOTE: dataset rows are generated from 21 template sentences; the evidence sentences "
              "map one-to-one to the verdict, so verdict accuracy here is an upper bound, not a "
              "realistic estimate. Use --data narratives for a fairer test.")


def summarize(res: pd.DataFrame) -> dict:
    y, pred = res.gold_label == 1, res.pred_verdict == "CONVICTED"
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    tn = (~y & ~pred).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec_n = tn / (tn + fn) if tn + fn else 0.0
    rec_n = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n else 0.0
    return {
        "section_top1_accuracy": round(res.section_top1.mean(), 4),
        "section_top3_accuracy": round(res.section_top3.mean(), 4),
        "verdict_accuracy": round(res.verdict_correct.mean(), 4),
        "verdict_macro_f1": round((f1_pos + f1_neg) / 2, 4),
        "inconclusive_rate": round((res.pred_verdict == "INCONCLUSIVE").mean(), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def calibrate(pipe, res: pd.DataFrame, write: bool) -> dict:
    """Grid-search fusion weights on the evaluated cases (re-uses stored features; no API calls)."""
    from symbolic_reasoner import SymbolicReasoner
    best_rank, best_rank_acc = None, -1
    grid = [x / 10 for x in range(0, 11)]
    for wr in grid:
        for we in grid:
            wl = round(1 - wr - we, 2)
            if wl < 0:
                continue
            w = {"rank": {"retrieval": wr, "elements": we, "llm_fit": wl}}
            hits = 0
            for _, r in res.iterrows():
                ranked = rank_candidates([dict(c) for c in r["_candidates"]], w)
                charges = pipe.reasoner.select_charges(ranked)
                p = charges["primary"] or charges["failed_top"]
                hits += section_hit(pipe.kb, r.gold_section, p["section"] if p else None)
            if hits > best_rank_acc:
                best_rank, best_rank_acc = w["rank"], hits

    best_fuse, best_fuse_acc = None, -1
    scored = res[res.symbolic_prob.notna()]
    for a in [x / 10 for x in range(11)]:
        for th in [x / 20 for x in range(4, 17)]:
            preds = [fuse_probability(r.symbolic_prob, None if pd.isna(r.neural_prob) else r.neural_prob,
                                      {"alpha": a}) >= th for _, r in scored.iterrows()]
            acc = sum(p == (g == 1) for p, g in zip(preds, scored.gold_label))
            if acc > best_fuse_acc:
                best_fuse, best_fuse_acc = {"alpha": a, "threshold": th}, acc

    new = {**load_weights(), "rank": best_rank, **(best_fuse or {})}
    if write:
        WEIGHTS_PATH.write_text(json.dumps(new, indent=2), encoding="utf-8")
    return {"weights": new, "section_top1_at_best": round(best_rank_acc / len(res), 4),
            "verdict_acc_at_best": round(best_fuse_acc / max(len(scored), 1), 4),
            "written": write,
            "warning": "Weights fitted on the same cases they are scored on; report results on a held-out split."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["dataset", "narratives"], default="dataset")
    ap.add_argument("--mask", choices=["sections", "full"], default="sections")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="calibrate without writing weights")
    run(ap.parse_args())
