"""
Scan the PubMed corpus subset and map retraction labels.

1. Load all .jsonl files in data/raw/pubmed_subset/chunk/. Cast PMID to int;
   count and report any missing or non-numeric PMIDs.
2. Print total docs and unique PMIDs.
3. Load data/processed/retracted.parquet. Count docs whose PMID is in the retracted
   set: overall, and split into RW-only / tag-only / both. Print retracted docs / total
   docs as a percentage.
4. Write data/processed/subset_docs.parquet with columns id, pmid, is_retracted.
5. Save the printed output to results/corpus_scan.txt.
"""

from pathlib import Path
import glob
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    chunk_dir = Path("data/raw/pubmed_subset/chunk")
    retracted_parquet_path = Path("data/processed/retracted.parquet")
    output_parquet_path = Path("data/processed/subset_docs.parquet")
    output_txt_path = Path("results/corpus_scan.txt")

    chunk_files = sorted(glob.glob(str(chunk_dir / "*.jsonl")))
    print(f"Found {len(chunk_files)} chunk files in {chunk_dir}.")

    # Load retracted labels
    print(f"Loading {retracted_parquet_path}...")
    df_retracted = pd.read_parquet(retracted_parquet_path)

    rw_only_pmids = set(df_retracted[df_retracted["in_rw"] & ~df_retracted["in_pubmed_tag"]]["pmid"])
    tag_only_pmids = set(df_retracted[~df_retracted["in_rw"] & df_retracted["in_pubmed_tag"]]["pmid"])
    both_pmids = set(df_retracted[df_retracted["in_rw"] & df_retracted["in_pubmed_tag"]]["pmid"])
    all_retracted_pmids = set(df_retracted["pmid"])

    # Sanity check label partitions
    assert len(all_retracted_pmids) == len(rw_only_pmids) + len(tag_only_pmids) + len(both_pmids)

    # Containers for subset doc metadata
    doc_ids = []
    doc_pmids = []
    is_retracted_list = []

    missing_pmid_count = 0
    non_numeric_pmid_count = 0

    rw_only_doc_count = 0
    tag_only_doc_count = 0
    both_doc_count = 0
    retracted_doc_count = 0

    unique_pmids = set()
    retracted_unique_pmids = set()

    # Iterate through chunks
    for file_path in chunk_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = record.get("id")
                raw_pmid = record.get("PMID")

                pmid_int = None
                if raw_pmid is None or raw_pmid == "":
                    missing_pmid_count += 1
                else:
                    try:
                        pmid_int = int(raw_pmid)
                    except (ValueError, TypeError):
                        non_numeric_pmid_count += 1

                if pmid_int is not None:
                    unique_pmids.add(pmid_int)
                    is_ret = pmid_int in all_retracted_pmids
                    if is_ret:
                        retracted_doc_count += 1
                        retracted_unique_pmids.add(pmid_int)
                        if pmid_int in both_pmids:
                            both_doc_count += 1
                        elif pmid_int in rw_only_pmids:
                            rw_only_doc_count += 1
                        elif pmid_int in tag_only_pmids:
                            tag_only_doc_count += 1
                else:
                    is_ret = False

                doc_ids.append(doc_id)
                doc_pmids.append(pmid_int)
                is_retracted_list.append(is_ret)

    total_docs = len(doc_ids)
    retracted_pct = (retracted_doc_count / total_docs * 100) if total_docs > 0 else 0.0

    # Build report lines
    report = []
    report.append("=== PubMed Corpus Subset Scan ===")
    report.append(f"Chunk files processed: {len(chunk_files)}")
    report.append(f"Total documents: {total_docs:,}")
    report.append(f"Unique valid PMIDs: {len(unique_pmids):,}")
    report.append(f"Missing PMIDs: {missing_pmid_count:,}")
    report.append(f"Non-numeric PMIDs: {non_numeric_pmid_count:,}")
    report.append("")
    report.append("=== Retraction Match Summary ===")
    report.append(f"Total retracted documents: {retracted_doc_count:,} ({retracted_pct:.4f}% of total docs)")
    report.append(f"Unique retracted PMIDs found in subset: {len(retracted_unique_pmids):,}")
    report.append(f"  - Both (RW & PubMed tag): {both_doc_count:,} docs")
    report.append(f"  - Retraction Watch only: {rw_only_doc_count:,} docs")
    report.append(f"  - PubMed tag only: {tag_only_doc_count:,} docs")
    report.append(f"  - Sum of components: {both_doc_count + rw_only_doc_count + tag_only_doc_count:,} docs")
    report.append("")
    report.append("=== Output Files ===")
    report.append(f"Parquet table: {output_parquet_path}")
    report.append(f"Summary log: {output_txt_path}")

    report_str = "\n".join(report)
    print("\n" + report_str)

    # Save output to results/corpus_scan.txt
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
    print(f"\nSaved summary to {output_txt_path}")

    # Write data/processed/subset_docs.parquet
    print(f"Writing {output_parquet_path}...")
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_arrays(
        [
            pa.array(doc_ids, type=pa.string()),
            pa.array(doc_pmids, type=pa.int64()),
            pa.array(is_retracted_list, type=pa.bool_()),
        ],
        names=["id", "pmid", "is_retracted"],
    )
    pq.write_table(table, output_parquet_path, compression="snappy")
    print(f"Successfully wrote {table.num_rows:,} rows to {output_parquet_path}")


if __name__ == "__main__":
    main()
