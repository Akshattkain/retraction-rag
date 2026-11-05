"""
Verification script for BM25 Lucene index on PubMed pilot subset.

1. Check indexed document count and disk size against expected 1,224,889 docs.
2. Smoke test: search "eosinophilic meningitis Angiostrongylus" (k=3).
3. Recall check: for the 23 PubMedQA questions with gold PMID in corpus subset,
   search question text (k=32), reporting top-32 recall and hit ranks.
4. Save full printed output to results/bm25_index_check.txt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import subprocess
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from src.bm25_search import search, get_searcher, get_id_to_pmid_map


def main():
    index_path = Path("data/processed/bm25_subset")
    subset_docs_path = Path("data/processed/subset_docs.parquet")
    benchmark_path = Path("third_party/MIRAGE/benchmark.json")
    output_path = Path("results/bm25_index_check.txt")

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=== BM25 Lucene Index Verification (Pilot Subset) ===")

    # 1. Index document count and disk size
    log("\n--- Step 1: Index Stats & Size Verification ---")
    searcher = get_searcher()
    num_docs = searcher.num_docs
    expected_docs = 1224889

    # Size on disk
    du_res = subprocess.run(["du", "-sh", str(index_path)], capture_output=True, text=True, check=True)
    index_size = du_res.stdout.strip().split()[0]

    log(f"Index location: {index_path}")
    log(f"Indexed document count: {num_docs:,}")
    log(f"Expected document count: {expected_docs:,}")
    mismatch = num_docs != expected_docs
    log(f"Count mismatch flagged: {mismatch}")
    if mismatch:
        log(f"WARNING: Indexed doc count ({num_docs}) does not match expected ({expected_docs})!")
    else:
        log(f"SUCCESS: Indexed doc count exactly matches the corpus scan total ({expected_docs:,}).")
    log(f"Index size on disk: {index_size}")

    # 2. Smoke Test
    log("\n--- Step 2: Smoke Test ---")
    smoke_query = "eosinophilic meningitis Angiostrongylus"
    log(f"Query: \"{smoke_query}\" (k=3)")
    smoke_hits = search(smoke_query, k=3)
    log(f"{'Rank':<6} {'PMID':<12} {'DocID':<24} {'Score':<10}")
    log("-" * 52)
    for h in smoke_hits:
        log(f"{h['rank']:<6} {str(h['pmid']):<12} {h['docid']:<24} {h['score']:<10.4f}")

    # 3. Recall Check on 23 PubMedQA Questions
    log("\n--- Step 3: PubMedQA Recall Check (Gold PMID in Corpus Subset) ---")
    log(f"Loading {benchmark_path}...")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        mirage_data = json.load(f)

    pubmedqa = mirage_data.get("pubmedqa", {})
    df_subset = pd.read_parquet(subset_docs_path)
    subset_pmids = set(df_subset["pmid"])

    # Identify 23 target questions
    target_questions = []
    for qid, item in pubmedqa.items():
        raw_pmids = item.get("PMID", [])
        if not isinstance(raw_pmids, list):
            raw_pmids = [raw_pmids]
        for p in raw_pmids:
            try:
                p_int = int(p)
                if p_int in subset_pmids:
                    target_questions.append({
                        "qid": str(qid),
                        "gold_pmid": p_int,
                        "question": item.get("question", ""),
                    })
            except (ValueError, TypeError):
                continue

    log(f"Found {len(target_questions)} PubMedQA questions whose gold PMID is in the corpus subset.")
    assert len(target_questions) == 23, f"Expected 23 questions, found {len(target_questions)}"

    hits_found = 0
    hit_ranks = []
    results_table = []

    for idx, q_info in enumerate(target_questions, start=1):
        qid = q_info["qid"]
        gold_pmid = q_info["gold_pmid"]
        q_text = q_info["question"]

        retrieved = search(q_text, k=32)
        retrieved_pmids = [h["pmid"] for h in retrieved]

        hit_rank = None
        hit_score = None
        if gold_pmid in retrieved_pmids:
            hit_index = retrieved_pmids.index(gold_pmid)
            hit_rank = retrieved[hit_index]["rank"]
            hit_score = retrieved[hit_index]["score"]
            hits_found += 1
            hit_ranks.append(hit_rank)
        else:
            hit_ranks.append(None)

        results_table.append({
            "idx": idx,
            "qid": qid,
            "gold_pmid": gold_pmid,
            "hit": hit_rank is not None,
            "rank": hit_rank,
            "score": hit_score,
            "question": q_text,
        })

    log(f"\nPer-Question Retrieval Results (k=32):")
    header = f"{'#':<4} {'QID':<26} {'Gold PMID':<12} {'Hit@32':<8} {'Rank':<6} {'Score':<8}"
    log(header)
    log("-" * len(header))
    for row in results_table:
        hit_str = "YES" if row["hit"] else "NO"
        rank_str = str(row["rank"]) if row["rank"] is not None else "-"
        score_str = f"{row['score']:.4f}" if row["score"] is not None else "-"
        log(f"{row['idx']:<4} {row['qid']:<26} {row['gold_pmid']:<12} {hit_str:<8} {rank_str:<6} {score_str:<8}")

    recall_32 = hits_found / len(target_questions) * 100.0
    valid_ranks = [r for r in hit_ranks if r is not None]
    mrr_32 = sum(1.0 / r for r in valid_ranks) / len(target_questions) if target_questions else 0.0

    log(f"\n=== Summary Metrics ===")
    log(f"Total PubMedQA questions evaluated: {len(target_questions)}")
    log(f"Gold PMIDs retrieved in Top 32: {hits_found} / {len(target_questions)} ({recall_32:.2f}%)")
    log(f"MRR@32: {mrr_32:.4f}")
    if valid_ranks:
        rank_1_count = sum(1 for r in valid_ranks if r == 1)
        rank_5_count = sum(1 for r in valid_ranks if r <= 5)
        rank_10_count = sum(1 for r in valid_ranks if r <= 10)
        log(f"Recall@1:  {rank_1_count} / {len(target_questions)} ({rank_1_count / len(target_questions) * 100:.2f}%)")
        log(f"Recall@5:  {rank_5_count} / {len(target_questions)} ({rank_5_count / len(target_questions) * 100:.2f}%)")
        log(f"Recall@10: {rank_10_count} / {len(target_questions)} ({rank_10_count / len(target_questions) * 100:.2f}%)")
        log(f"Ranks of hits: {sorted(valid_ranks)}")

    # 4. Save to results/bm25_index_check.txt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"\nSaved full output log to: {output_path}")


if __name__ == "__main__":
    main()
