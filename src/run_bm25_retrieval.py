"""
Run BM25 retrieval (top-32) for all MIRAGE questions and flag the hits.

1. Load third_party/MIRAGE/benchmark.json; query = "question" field only (no options);
   key = "<dataset>/<id>". Print question count per dataset.
2. Load the searcher once (src.bm25_search) and call search(query, k=32) per question.
   On a too-many-clauses error, truncate the query to its first 300 words and retry.
3. Save data/processed/retrieval_bm25.parquet (one row per question and rank).
   Resumable: chunks of 500 questions are saved in data/processed/retrieval_partial/.
4. Join with data/processed/subset_flags.parquet (docid = id) and save
   data/processed/retrieval_bm25_flagged.parquet.
5. Print and save exposure summary to results/retrieval_exposure.txt.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.metadata
import json
import pandas as pd
from src.bm25_search import search, get_searcher

BENCHMARK_PATH = Path("third_party/MIRAGE/benchmark.json")
FLAGS_PATH = Path("data/processed/subset_flags.parquet")
PARTIAL_DIR = Path("data/processed/retrieval_partial")
OUT_PATH = Path("data/processed/retrieval_bm25.parquet")
FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
RESULTS_PATH = Path("results/retrieval_exposure.txt")

K = 32
CHUNK = 500
EXPECTED_QUESTIONS = 7663
MAX_WORDS = 300
STAT_FLAGS = ["is_retracted", "one_hop", "is_notice"]
FLAG_COLS = ["is_retracted", "is_notice", "one_hop", "in_situ", "n_retracted_cited"]

output_lines = []


def log(msg=""):
    print(msg, flush=True)
    output_lines.append(str(msg))


def is_too_many_clauses(exc):
    return "clause" in str(exc).lower() or "toomanyclauses" in type(exc).__name__.lower()


def run_query(query):
    """Return (hits, truncated). Retry once with a truncated query on too-many-clauses."""
    try:
        return search(query, k=K), False
    except Exception as e:
        if not is_too_many_clauses(e):
            raise
    short = " ".join(query.split()[:MAX_WORDS])
    return search(short, k=K), True


def write_atomic(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def load_questions():
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)
    questions = []
    log("--- Questions per dataset ---")
    for dataset, items in bench.items():
        log(f"{dataset}: {len(items):,}")
        for qid, item in items.items():
            questions.append((f"{dataset}/{qid}", dataset, item["question"]))
    log(f"total: {len(questions):,}")
    assert len(questions) == EXPECTED_QUESTIONS, len(questions)
    return questions


def run_chunks(questions):
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    get_searcher()  # load once
    n_chunks = (len(questions) + CHUNK - 1) // CHUNK
    n_run = 0
    run_seconds = 0.0
    n_resumed = 0

    for c in range(n_chunks):
        part_path = PARTIAL_DIR / f"part_{c:04d}.parquet"
        meta_path = PARTIAL_DIR / f"meta_{c:04d}.parquet"
        chunk = questions[c * CHUNK:(c + 1) * CHUNK]
        if part_path.exists() and meta_path.exists():
            n_resumed += len(chunk)
            continue

        t0 = time.time()
        rows, meta = [], []
        for qkey, dataset, query in chunk:
            hits, truncated = run_query(query)
            meta.append({"qkey": qkey, "dataset": dataset, "n_hits": len(hits), "truncated": truncated})
            for h in hits:
                rows.append({
                    "qkey": qkey, "dataset": dataset, "rank": h["rank"],
                    "docid": h["docid"], "pmid": h["pmid"], "score": h["score"],
                })
        # meta is written last, so a chunk only counts as done once both files exist
        write_atomic(pd.DataFrame(rows, columns=["qkey", "dataset", "rank", "docid", "pmid", "score"]), part_path)
        write_atomic(pd.DataFrame(meta), meta_path)

        dt = time.time() - t0
        n_run += len(chunk)
        run_seconds += dt
        done = min((c + 1) * CHUNK, len(questions))
        log(f"[progress] {done:,}/{len(questions):,} questions "
            f"(chunk {c + 1}/{n_chunks}, {dt / len(chunk) * 100:.1f}s per 100 q)")

    return n_run, run_seconds, n_resumed


def assemble(n_chunks):
    parts = [pd.read_parquet(PARTIAL_DIR / f"part_{c:04d}.parquet") for c in range(n_chunks)]
    metas = [pd.read_parquet(PARTIAL_DIR / f"meta_{c:04d}.parquet") for c in range(n_chunks)]
    return pd.concat(parts, ignore_index=True), pd.concat(metas, ignore_index=True)


def exposure_table(df, meta, flags_base):
    log("\n--- Exposure (share of retrieved rows | share of questions with >=1 hit) ---")
    datasets = list(meta["dataset"].drop_duplicates())
    for flag in STAT_FLAGS:
        log(f"\n{flag}  (corpus base rate: {flags_base[flag]:.4%})")
        qany = df.groupby("qkey")[flag].any()
        meta = meta.assign(any_hit=meta["qkey"].map(qany).fillna(False).astype(bool))
        log(f"  {'overall':<10} rows {df[flag].mean():.4%} | questions {meta['any_hit'].mean():.4%}")
        for ds in datasets:
            rows_ds = df[df["dataset"] == ds]
            q_ds = meta[meta["dataset"] == ds]
            log(f"  {ds:<10} rows {rows_ds[flag].mean():.4%} | questions {q_ds['any_hit'].mean():.4%}")


def main():
    log("=== BM25 retrieval over MIRAGE ===")
    log(f"Pyserini version: {importlib.metadata.version('pyserini')} (BM25 defaults), k={K}")

    questions = load_questions()
    n_chunks = (len(questions) + CHUNK - 1) // CHUNK
    n_run, run_seconds, n_resumed = run_chunks(questions)

    df, meta = assemble(n_chunks)
    assert meta["qkey"].is_unique and len(meta) == EXPECTED_QUESTIONS
    write_atomic(df, OUT_PATH)

    flags = pd.read_parquet(FLAGS_PATH, columns=["id"] + FLAG_COLS)
    flagged = df.merge(flags, how="left", left_on="docid", right_on="id", validate="many_to_one")
    assert flagged["id"].notna().all(), "some retrieved docids missing from subset_flags"
    assert len(flagged) == len(df)
    flagged = flagged.drop(columns=["id"])
    write_atomic(flagged, FLAGGED_PATH)

    log("\n--- Run summary ---")
    log(f"Questions run: {len(meta):,} (expected {EXPECTED_QUESTIONS:,})")
    log(f"Total rows: {len(df):,} (expected {EXPECTED_QUESTIONS * K:,})")
    log(f"Queries with fewer than {K} hits: {int((meta['n_hits'] < K).sum()):,}")
    log(f"  of which zero hits: {int((meta['n_hits'] == 0).sum()):,}")
    log(f"Truncated queries (first {MAX_WORDS} words): {int(meta['truncated'].sum()):,}")
    if n_run:
        log(f"Time per 100 questions: {run_seconds / n_run * 100:.1f}s "
            f"(over {n_run:,} questions run this session; {n_resumed:,} resumed from partial files)")
    else:
        log(f"Time per 100 questions: n/a (all {n_resumed:,} questions resumed from partial files)")

    flags_base = flags[STAT_FLAGS].mean()
    exposure_table(flagged, meta, flags_base)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
