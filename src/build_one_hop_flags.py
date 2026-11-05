"""
Build one-hop citation flags for the PubMed corpus subset.

1. Load:
   - data/processed/icite_citing_pairs.parquet
   - data/processed/retracted.parquet
   - data/processed/notice_exclude_pmids.txt (36,649 PMIDs)
   - data/processed/notice_pmids.txt
   - data/processed/subset_docs.parquet
2. Filter pairs in order, reporting pairs and unique citing PMIDs:
   (a) all pairs
   (b) drop pairs whose citing PMID is in the exclusion list
   (c) keep only pairs whose citing PMID is in the corpus subset
3. Build per-document flags for every subset doc:
   - is_retracted (from subset_docs)
   - is_notice (PMID in notice_exclude_pmids.txt)
   - in_situ (retracted AND in notice_pmids.txt)
   - one_hop (not retracted, not is_notice, and citing PMID in step (c) pairs)
   - n_retracted_cited (distinct retracted PMIDs cited from step (c) pairs)
   Write to data/processed/subset_flags.parquet.
4. Print and save summary statistics to results/one_hop_table.txt.
"""

from pathlib import Path
from typing import List, Set, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# File paths
ICITE_PAIRS_PATH = Path("data/processed/icite_citing_pairs.parquet")
RETRACTED_PATH = Path("data/processed/retracted.parquet")
NOTICE_EXCLUDE_PATH = Path("data/processed/notice_exclude_pmids.txt")
NOTICE_PMIDS_PATH = Path("data/processed/notice_pmids.txt")
SUBSET_DOCS_PATH = Path("data/processed/subset_docs.parquet")

OUTPUT_FLAGS_PATH = Path("data/processed/subset_flags.parquet")
OUTPUT_LOG_PATH = Path("results/one_hop_table.txt")


def load_pmid_file(path: Path) -> Set[int]:
    """Load a newline-delimited text file of PMIDs into a set of ints."""
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def main():
    output_lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("=== Corpus Subset One-Hop Flag Generation ===")

    # 1. Load data files
    log("\n--- Step 1: Loading Input Data ---")
    log(f"Loading {ICITE_PAIRS_PATH}...")
    df_pairs = pd.read_parquet(ICITE_PAIRS_PATH)
    log(f"  Loaded {len(df_pairs):,} citation pairs.")

    log(f"Loading {RETRACTED_PATH}...")
    df_retracted = pd.read_parquet(RETRACTED_PATH)
    retracted_pmids: Set[int] = set(df_retracted["pmid"].dropna().astype(int))
    log(f"  Loaded {len(retracted_pmids):,} retracted PMIDs.")

    log(f"Loading {NOTICE_EXCLUDE_PATH}...")
    notice_exclude_pmids: Set[int] = load_pmid_file(NOTICE_EXCLUDE_PATH)
    log(f"  Loaded {len(notice_exclude_pmids):,} notice exclusion PMIDs.")
    assert len(notice_exclude_pmids) == 36649, (
        f"Expected 36,649 notice exclusion PMIDs, got {len(notice_exclude_pmids)}"
    )

    log(f"Loading {NOTICE_PMIDS_PATH}...")
    notice_pmids: Set[int] = load_pmid_file(NOTICE_PMIDS_PATH)
    log(f"  Loaded {len(notice_pmids):,} notice PMIDs.")

    log(f"Loading {SUBSET_DOCS_PATH}...")
    df_subset = pd.read_parquet(SUBSET_DOCS_PATH)
    subset_pmids: Set[int] = set(df_subset["pmid"].dropna().astype(int))
    log(f"  Loaded {len(df_subset):,} subset docs ({len(subset_pmids):,} unique PMIDs).")

    # 2. Sequential pair filtering
    log("\n--- Step 2: Sequential Citation Pair Filtering ---")

    # (a) All pairs
    pairs_a = df_pairs
    unique_citing_a = pairs_a["citing_pmid"].nunique()
    log(f"(a) All pairs: {len(pairs_a):,} pairs, {unique_citing_a:,} unique citing PMIDs")

    # (b) Drop pairs whose citing PMID is in the exclusion list
    pairs_b = pairs_a[~pairs_a["citing_pmid"].isin(notice_exclude_pmids)]
    unique_citing_b = pairs_b["citing_pmid"].nunique()
    log(f"(b) After dropping notice exclusion list: {len(pairs_b):,} pairs, {unique_citing_b:,} unique citing PMIDs")

    # (c) Keep only pairs whose citing PMID is in the corpus subset
    pairs_c = pairs_b[pairs_b["citing_pmid"].isin(subset_pmids)]
    unique_citing_c = pairs_c["citing_pmid"].nunique()
    log(f"(c) After keeping only corpus subset citing PMIDs: {len(pairs_c):,} pairs, {unique_citing_c:,} unique citing PMIDs")

    # Distinct retracted PMIDs cited per citing PMID in step (c)
    n_ret_cited_map: Dict[int, int] = (
        pairs_c.groupby("citing_pmid")["retracted_pmid"].nunique().to_dict()
    )
    citing_pmids_c: Set[int] = set(pairs_c["citing_pmid"])

    # 3. Build flags for every subset doc
    log("\n--- Step 3: Building Subset Flags ---")
    df_flags = pd.DataFrame()
    df_flags["id"] = df_subset["id"]
    df_flags["pmid"] = df_subset["pmid"].astype(int)
    df_flags["is_retracted"] = df_subset["is_retracted"].astype(bool)
    df_flags["is_notice"] = df_flags["pmid"].isin(notice_exclude_pmids)
    df_flags["in_situ"] = df_flags["is_retracted"] & df_flags["pmid"].isin(notice_pmids)
    df_flags["one_hop"] = (
        (~df_flags["is_retracted"])
        & (~df_flags["is_notice"])
        & df_flags["pmid"].isin(citing_pmids_c)
    )
    df_flags["n_retracted_cited"] = df_flags["pmid"].map(n_ret_cited_map).fillna(0).astype(int)

    # Write data/processed/subset_flags.parquet
    OUTPUT_FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_flags, preserve_index=False)
    pq.write_table(table, OUTPUT_FLAGS_PATH, compression="snappy")
    log(f"Successfully wrote {len(df_flags):,} rows to {OUTPUT_FLAGS_PATH}.")

    # 4. Summary & Verification Metrics
    log("\n--- Step 4: Summary Metrics & Sanity Checks ---")
    total_docs = len(df_flags)
    retracted_docs = int(df_flags["is_retracted"].sum())
    notice_docs = int(df_flags["is_notice"].sum())
    in_situ_docs = int(df_flags["in_situ"].sum())
    one_hop_docs = int(df_flags["one_hop"].sum())
    non_retracted_docs = total_docs - retracted_docs

    log(f"Total subset documents: {total_docs:,}")
    log(f"Retracted documents: {retracted_docs:,} (expected 1,528)")
    assert retracted_docs == 1528, f"Expected 1,528 retracted docs, got {retracted_docs}"

    log(f"is_notice documents: {notice_docs:,} (expected 234)")
    assert notice_docs == 234, f"Expected 234 notice docs, got {notice_docs}"

    log(f"in_situ documents: {in_situ_docs:,} (expected 57)")
    assert in_situ_docs == 57, f"Expected 57 in_situ docs, got {in_situ_docs}"

    share_all = (one_hop_docs / total_docs * 100.0) if total_docs > 0 else 0.0
    share_non_ret = (one_hop_docs / non_retracted_docs * 100.0) if non_retracted_docs > 0 else 0.0
    log(f"\nOne-hop documents: {one_hop_docs:,}")
    log(f"  Share of all subset documents: {share_all:.4f}% ({one_hop_docs:,} / {total_docs:,})")
    log(f"  Share of non-retracted documents: {share_non_ret:.4f}% ({one_hop_docs:,} / {non_retracted_docs:,})")

    # Distribution of n_retracted_cited among one-hop docs (1, 2, 3+)
    one_hop_counts = df_flags.loc[df_flags["one_hop"], "n_retracted_cited"]
    c_1 = int((one_hop_counts == 1).sum())
    c_2 = int((one_hop_counts == 2).sum())
    c_3plus = int((one_hop_counts >= 3).sum())

    log(f"\nDistribution of n_retracted_cited among one-hop documents ({len(one_hop_counts):,} total):")
    log(f"  n_retracted_cited = 1:  {c_1:,} ({c_1 / one_hop_docs * 100.0:.2f}%)")
    log(f"  n_retracted_cited = 2:  {c_2:,} ({c_2 / one_hop_docs * 100.0:.2f}%)")
    log(f"  n_retracted_cited >= 3: {c_3plus:,} ({c_3plus / one_hop_docs * 100.0:.2f}%)")

    # Retracted docs cited by another retracted paper
    # Check 1: Subset retracted docs cited by any retracted paper (global)
    pairs_ret_cites_ret_global = df_pairs[df_pairs["citing_pmid"].isin(retracted_pmids)]
    subset_ret_pmids = set(df_flags.loc[df_flags["is_retracted"], "pmid"])
    subset_ret_cited_global = subset_ret_pmids.intersection(set(pairs_ret_cites_ret_global["retracted_pmid"]))

    # Check 2: Within subset citing pairs (step c), retracted docs cited by another retracted doc in subset
    pairs_c_ret_cites_ret = pairs_c[pairs_c["citing_pmid"].isin(subset_ret_pmids)]
    subset_ret_cited_by_subset_ret = subset_ret_pmids.intersection(set(pairs_c_ret_cites_ret["retracted_pmid"]))
    subset_ret_citing_subset_ret = set(pairs_c_ret_cites_ret["citing_pmid"])

    log(f"\nRetracted papers cited by another retracted paper:")
    log(f"  Global (all iCite pairs): {len(set(pairs_ret_cites_ret_global['retracted_pmid'])):,} retracted PMIDs are cited by {len(set(pairs_ret_cites_ret_global['citing_pmid'])):,} retracted PMIDs ({len(pairs_ret_cites_ret_global):,} pairs)")
    log(f"  Subset retracted docs cited by any retracted paper: {len(subset_ret_cited_global):,} docs")
    log(f"  Within corpus subset (step c): {len(pairs_c_ret_cites_ret):,} pairs where citing doc is a subset retracted doc")
    log(f"    - Subset retracted docs that cite another retracted paper: {len(subset_ret_citing_subset_ret):,} docs")
    log(f"    - Subset retracted docs cited by another subset retracted paper: {len(subset_ret_cited_by_subset_ret):,} docs")

    # How many notice docs would have been flagged one-hop without the exclusion
    pairs_subset_raw = df_pairs[df_pairs["citing_pmid"].isin(subset_pmids)]
    notice_citing_in_subset = pairs_subset_raw[
        pairs_subset_raw["citing_pmid"].isin(notice_exclude_pmids)
    ]
    notice_pmids_citing_retracted = set(notice_citing_in_subset["citing_pmid"])
    notice_docs_flagged_without_exclusion = df_flags[
        df_flags["is_notice"] & (~df_flags["is_retracted"]) & df_flags["pmid"].isin(notice_pmids_citing_retracted)
    ]
    log(f"\nNotice documents without exclusion filter:")
    log(f"  Notice docs that cite a retracted paper in the corpus subset: {len(notice_docs_flagged_without_exclusion):,} docs")
    log(f"  (These {len(notice_docs_flagged_without_exclusion):,} notice docs would have been falsely flagged as one-hop papers without the exclusion list)")

    # 5. Save printed output to results/one_hop_table.txt
    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"\nSaved printed report to {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()
