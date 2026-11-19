"""
Filter ablation: what does removing retracted papers from the results change? (read-only on data; no LLM)

For every MIRAGE question, fetch the top 100 BM25 hits, then build three result lists of up to 32:
  baseline      the plain top 32 (same as the main retrieval run)
  drop_retracted     retracted papers removed, next-ranked papers fill the gaps (an index-time filter)
  drop_retracted_cites  also drop papers that cite a retracted paper (a strict filter)
Reports question-level exposure at k=5 and k=32, per dataset, and how much the top 5 changes.

Run from the project root:   python src/filter_ablation.py
Writes: results/filter_ablation.csv and results/filter_ablation.txt
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.bm25_search import search, get_searcher  # noqa: E402

BENCHMARK = Path("third_party/MIRAGE/benchmark.json")
FLAGS = Path("data/processed/subset_flags.parquet")
OUT_CSV = Path("results/filter_ablation.csv")
OUT_TXT = Path("results/filter_ablation.txt")
FETCH, K_MAIN, KS = 100, 32, (5, 32)
EXPECTED = {"questions": 7663,
            ("baseline", 5): (35, 714), ("baseline", 32): (262, 2404)}   # (retracted, one-hop) question counts, verified earlier

lines = []
def log(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))

def load_questions():
    bench = json.load(open(BENCHMARK))
    out = []
    for ds, items in bench.items():
        for qid, item in items.items():
            out.append((f"{ds}/{qid}", ds, item["question"]))
    return out

def run_query(query):
    try:
        return search(query, k=FETCH)
    except Exception as e:
        if "clause" not in str(e).lower() and "toomanyclauses" not in type(e).__name__.lower():
            raise
    return search(" ".join(query.split()[:300]), k=FETCH)

def main():
    questions = load_questions()
    assert len(questions) == EXPECTED["questions"], len(questions)
    fl = pd.read_parquet(FLAGS, columns=["id", "is_retracted", "one_hop"])
    ret = set(fl.loc[fl["is_retracted"], "id"])
    oh = set(fl.loc[fl["one_hop"], "id"])
    get_searcher()

    variants = {
        "baseline": lambda hs: hs,
        "drop_retracted": lambda hs: [h for h in hs if h["docid"] not in ret],
        "drop_retracted_cites": lambda hs: [h for h in hs if h["docid"] not in ret and h["docid"] not in oh],
    }
    rows = []          # one row per (question, variant, k)
    t0 = time.time()
    for i, (qkey, ds, q) in enumerate(questions, 1):
        hits = run_query(q)
        base32 = [h["docid"] for h in hits[:K_MAIN]]
        base5 = set(base32[:5])
        for vname, f in variants.items():
            ids = [h["docid"] for h in f(hits)][:K_MAIN]
            for k in KS:
                top = ids[:k]
                rows.append((qkey, ds, vname, k,
                             any(d in ret for d in top), any(d in oh for d in top),
                             len(top), len(set(top) & set(base32[:k])) / max(1, min(k, len(base32))),
                             set(ids[:5]) != base5 if k == 5 else False))
        if i % 1000 == 0:
            log(f"  {i:,}/{len(questions):,} questions ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows, columns=["qkey", "dataset", "variant", "k", "retracted", "one_hop", "n_results", "kept_from_baseline", "top5_changed"])
    ds_n = df[df.variant == "baseline"].groupby("dataset")["qkey"].nunique()
    recs = []
    for (v, k), g in df.groupby(["variant", "k"]):
        for scope, sub in [("overall", g)] + [(d, g[g.dataset == d]) for d in ds_n.index]:
            n = sub["qkey"].nunique()
            recs.append(dict(variant=v, k=k, scope=scope, questions=n,
                             retracted_q=int(sub["retracted"].sum()), retracted_pct=100 * sub["retracted"].mean(),
                             one_hop_q=int(sub["one_hop"].sum()), one_hop_pct=100 * sub["one_hop"].mean(),
                             kept_pct=100 * sub["kept_from_baseline"].mean(),
                             top5_changed_pct=100 * sub["top5_changed"].mean() if k == 5 else float("nan")))
    out = pd.DataFrame(recs)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    log("\n=== Filter ablation (question-level, top-100 fetched, up to 32 kept) ===")
    ov = out[out.scope == "overall"].sort_values(["variant", "k"])
    for _, r in ov.iterrows():
        extra = f" | top-5 list changed for {r.top5_changed_pct:.2f}% of questions" if r.k == 5 else ""
        log(f"{r.variant:<22} k={int(r.k):<2} retracted {r.retracted_q:>5,} ({r.retracted_pct:.2f}%)  "
            f"one-hop {r.one_hop_q:>5,} ({r.one_hop_pct:.2f}%)  kept from baseline {r.kept_pct:.1f}%{extra}")
    log("\n--- sanity check against the main retrieval run (should match) ---")
    for k in KS:
        b = ov[(ov.variant == "baseline") & (ov.k == k)].iloc[0]
        exp = EXPECTED[("baseline", k)]
        ok = (int(b.retracted_q), int(b.one_hop_q)) == exp
        log(f"baseline k={k}: got ({int(b.retracted_q)}, {int(b.one_hop_q)}), expected {exp} -> {'MATCH' if ok else 'MISMATCH'}")
    short = df[(df.variant != "baseline") & (df.k == 32) & (df.n_results < 32)]
    log(f"questions with fewer than 32 results after filtering: {short['qkey'].nunique()}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"\nSaved {OUT_CSV} and {OUT_TXT}")

if __name__ == "__main__":
    main()