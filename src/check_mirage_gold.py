"""
Check MIRAGE gold-evidence papers against retraction labels and corpus subset.

1. Load third_party/MIRAGE/benchmark.json. For pubmedqa and bioasq, each item has a
   "PMID" list of gold evidence papers. Collect all gold PMIDs (cast to int) per
   dataset, with question ids.
2. Load data/processed/retracted.parquet and data/processed/subset_docs.parquet.
3. Print per dataset:
   - Number of questions;
   - Number of unique gold PMIDs;
   - How many gold PMIDs are in the retracted set (list question id, PMID, in_rw,
     in_pubmed_tag, has_reinstatement, retraction_date);
   - How many gold PMIDs exist in the corpus subset.
4. Save the printed output to results/mirage_gold_check.txt.
"""

from pathlib import Path
import json
import pandas as pd


def format_date(val):
    if pd.isna(val) or val is None:
        return "None"
    return str(val).split("T")[0].split(" ")[0]


def main():
    benchmark_path = Path("third_party/MIRAGE/benchmark.json")
    retracted_path = Path("data/processed/retracted.parquet")
    subset_docs_path = Path("data/processed/subset_docs.parquet")
    output_txt_path = Path("results/mirage_gold_check.txt")

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=== MIRAGE Gold Evidence Retraction and Subset Check ===")

    # 1. Load benchmark.json
    log(f"Loading {benchmark_path}...")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        mirage_data = json.load(f)

    # 2. Load retracted.parquet and subset_docs.parquet
    log(f"Loading {retracted_path}...")
    df_retracted = pd.read_parquet(retracted_path)
    retracted_dict = df_retracted.set_index("pmid").to_dict(orient="index")
    all_retracted_pmids = set(df_retracted["pmid"])

    log(f"Loading {subset_docs_path}...")
    df_subset = pd.read_parquet(subset_docs_path)
    subset_pmids = set(df_subset["pmid"])
    log(f"Loaded {len(df_retracted):,} retracted PMIDs and {len(subset_pmids):,} corpus subset PMIDs.\n")

    datasets = ["pubmedqa", "bioasq"]

    for ds in datasets:
        log(f"--------------------------------------------------")
        log(f"Dataset: {ds.upper()}")
        log(f"--------------------------------------------------")

        ds_items = mirage_data.get(ds, {})
        num_questions = len(ds_items)

        # Collect gold PMIDs with question IDs
        question_pmid_pairs = []
        unique_gold_pmids = set()

        for qid, item in ds_items.items():
            raw_pmids = item.get("PMID", [])
            if raw_pmids is None:
                continue
            if not isinstance(raw_pmids, list):
                raw_pmids = [raw_pmids]

            for p in raw_pmids:
                try:
                    p_int = int(p)
                    question_pmid_pairs.append((str(qid), p_int))
                    unique_gold_pmids.add(p_int)
                except (ValueError, TypeError):
                    continue

        # Check against retracted set
        retracted_gold_pmids = unique_gold_pmids.intersection(all_retracted_pmids)
        retracted_matches = []
        for qid, pmid in question_pmid_pairs:
            if pmid in retracted_dict:
                info = retracted_dict[pmid]
                retracted_matches.append({
                    "question_id": qid,
                    "pmid": pmid,
                    "in_rw": info["in_rw"],
                    "in_pubmed_tag": info["in_pubmed_tag"],
                    "has_reinstatement": info["has_reinstatement"],
                    "retraction_date": format_date(info["retraction_date"]),
                })

        # Check against corpus subset
        subset_gold_pmids = unique_gold_pmids.intersection(subset_pmids)
        subset_gold_pairs = [(qid, pmid) for qid, pmid in question_pmid_pairs if pmid in subset_pmids]

        # Log statistics
        log(f"Number of questions: {num_questions:,}")
        log(f"Total question-PMID gold pairs: {len(question_pmid_pairs):,}")
        log(f"Unique gold PMIDs: {len(unique_gold_pmids):,}")
        log(f"Gold PMIDs in retracted set (unique PMIDs): {len(retracted_gold_pmids):,}")
        log(f"Gold PMID occurrences in retracted set: {len(retracted_matches):,}")

        if retracted_matches:
            log("\nList of retracted gold evidence papers:")
            header = f"  {'Question ID':<26} {'PMID':<10} {'in_rw':<8} {'in_tag':<8} {'reinstated':<12} {'retraction_date':<12}"
            log(header)
            log("  " + "-" * (len(header) - 2))
            for m in retracted_matches:
                log(f"  {m['question_id']:<26} {m['pmid']:<10} {str(m['in_rw']):<8} {str(m['in_pubmed_tag']):<8} {str(m['has_reinstatement']):<12} {m['retraction_date']:<12}")
        else:
            log("  No gold PMIDs found in the retracted set.")

        log(f"\nGold PMIDs in corpus subset:")
        pct_unique = (len(subset_gold_pmids) / len(unique_gold_pmids) * 100.0) if unique_gold_pmids else 0.0
        pct_pairs = (len(subset_gold_pairs) / len(question_pmid_pairs) * 100.0) if question_pmid_pairs else 0.0
        log(f"  Unique gold PMIDs in corpus subset: {len(subset_gold_pmids):,} / {len(unique_gold_pmids):,} ({pct_unique:.2f}%)")
        log(f"  Question-PMID pairs in corpus subset: {len(subset_gold_pairs):,} / {len(question_pmid_pairs):,} ({pct_pairs:.2f}%)")
        log()

    # 4. Save printed output to results/mirage_gold_check.txt
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    log(f"Results saved to {output_txt_path}")


if __name__ == "__main__":
    main()
