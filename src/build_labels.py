"""
Build the retraction label table and run sanity checks.

1. Load data/raw/retraction-watch-data/retraction_watch.csv:
   - Drop the empty trailing column.
   - Filter RetractionNature == "Retraction".
   - Parse RetractionDate with format "%m/%d/%Y %H:%M".
   - Treat OriginalPaperPubMedID of 0 or blank as missing.
   - Lowercase OriginalPaperDOI and treat blank/"unavailable" as missing.
   - Keep Reason as a list (split on ";").
2. Load data/raw/pubmed_retracted_pmids.txt.
3. Write data/processed/retracted.parquet with one row per retracted PMID:
   pmid (int), doi (str), retraction_date (datetime), reasons (list[str]), in_rw (bool), in_pubmed_tag (bool).
   Write RW rows with DOI but no PMID to data/processed/retracted_doi_only.parquet.
4. Print summary counts and save to results/label_summary.txt.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def clean_pmid(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    if not val_str or val_str == "0":
        return None
    try:
        ival = int(val_str)
        return ival if ival > 0 else None
    except ValueError:
        return None


def clean_doi(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    if not val_str or val_str in {"unavailable", "nan", "none"}:
        return None
    return val_str


def clean_reasons(val):
    if pd.isna(val):
        return []
    return [r.strip() for r in str(val).split(";") if r.strip()]


def main():
    rw_csv_path = Path("data/raw/retraction-watch-data/retraction_watch.csv")
    pubmed_txt_path = Path("data/raw/pubmed_retracted_pmids.txt")
    out_retracted_path = Path("data/processed/retracted.parquet")
    out_doi_only_path = Path("data/processed/retracted_doi_only.parquet")
    summary_txt_path = Path("results/label_summary.txt")

    out_retracted_path.parent.mkdir(parents=True, exist_ok=True)
    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Retraction Watch CSV
    df_raw = pd.read_csv(rw_csv_path, low_memory=False)
    rw_rows_total = len(df_raw)

    # Drop empty trailing column (e.g. 'Unnamed: 20')
    if df_raw.columns[-1].startswith("Unnamed:") and df_raw.iloc[:, -1].isna().all():
        df_raw = df_raw.iloc[:, :-1]

    # Filter RetractionNature == "Retraction"
    df_rw = df_raw[df_raw["RetractionNature"] == "Retraction"].copy()
    rw_after_nature_filter = len(df_rw)

    # Parse RetractionDate
    df_rw["retraction_date"] = pd.to_datetime(
        df_rw["RetractionDate"], format="%m/%d/%Y %H:%M"
    )

    # Parse and clean PMID, DOI, Reasons
    df_rw["pmid"] = df_rw["OriginalPaperPubMedID"].apply(clean_pmid)
    df_rw["doi"] = df_rw["OriginalPaperDOI"].apply(clean_doi)
    df_rw["reasons"] = df_rw["Reason"].apply(clean_reasons)

    # Collect PMIDs with RetractionNature == "Reinstatement"
    df_reinstatements = df_raw[df_raw["RetractionNature"] == "Reinstatement"]
    reinstatement_pmids = set(
        df_reinstatements["OriginalPaperPubMedID"].apply(clean_pmid).dropna().astype(int)
    )

    # RW PMID counts
    rw_valid_pmid_mask = df_rw["pmid"].notna()
    rw_rows_with_valid_pmid = int(rw_valid_pmid_mask.sum())
    rw_pmids_series = df_rw.loc[rw_valid_pmid_mask, "pmid"].astype(int)
    rw_unique_pmids = set(rw_pmids_series.unique())

    # 2. Load PubMed retracted PMIDs
    with open(pubmed_txt_path, "r") as f:
        pubmed_tag_pmids = set(
            int(line.strip()) for line in f if line.strip().isdigit()
        )
    pubmed_tag_count = len(pubmed_tag_pmids)

    # Overlap and union calculations
    overlap = rw_unique_pmids & pubmed_tag_pmids
    union_pmids = rw_unique_pmids | pubmed_tag_pmids
    rw_only = rw_unique_pmids - pubmed_tag_pmids
    tag_only = pubmed_tag_pmids - rw_unique_pmids

    # 3. Build data/processed/retracted.parquet
    # Deduplicate RW entries with valid PMID:
    # Earliest retraction date, combined unique reasons, first non-null DOI
    rw_by_pmid = {}
    for _, row in df_rw[rw_valid_pmid_mask].iterrows():
        p = int(row["pmid"])
        if p not in rw_by_pmid:
            rw_by_pmid[p] = {
                "doi": row["doi"],
                "retraction_date": row["retraction_date"],
                "reasons": list(row["reasons"]),
            }
        else:
            # Combine reasons preserving order
            for r in row["reasons"]:
                if r not in rw_by_pmid[p]["reasons"]:
                    rw_by_pmid[p]["reasons"].append(r)
            # Earliest retraction date
            if pd.notna(row["retraction_date"]):
                curr_date = rw_by_pmid[p]["retraction_date"]
                if pd.isna(curr_date) or row["retraction_date"] < curr_date:
                    rw_by_pmid[p]["retraction_date"] = row["retraction_date"]
            # Fallback for DOI if first was None
            if not rw_by_pmid[p]["doi"] and row["doi"]:
                rw_by_pmid[p]["doi"] = row["doi"]

    retracted_records = []
    for p in sorted(union_pmids):
        in_rw = p in rw_unique_pmids
        in_tag = p in pubmed_tag_pmids
        has_reinstatement = p in reinstatement_pmids

        if in_rw:
            doi = rw_by_pmid[p]["doi"]
            r_date = rw_by_pmid[p]["retraction_date"]
            reasons = rw_by_pmid[p]["reasons"]
        else:
            doi = None
            r_date = pd.NaT
            reasons = []

        retracted_records.append(
            {
                "pmid": p,
                "doi": doi,
                "retraction_date": r_date,
                "reasons": reasons,
                "in_rw": in_rw,
                "in_pubmed_tag": in_tag,
                "has_reinstatement": has_reinstatement,
            }
        )

    df_retracted = pd.DataFrame(retracted_records)
    # Ensure types
    df_retracted["pmid"] = df_retracted["pmid"].astype("int64")
    df_retracted["in_rw"] = df_retracted["in_rw"].astype("bool")
    df_retracted["in_pubmed_tag"] = df_retracted["in_pubmed_tag"].astype("bool")
    df_retracted["has_reinstatement"] = df_retracted["has_reinstatement"].astype("bool")
    df_retracted["retraction_date"] = pd.to_datetime(df_retracted["retraction_date"])

    df_retracted.to_parquet(out_retracted_path, index=False)

    # 3b. Build data/processed/retracted_doi_only.parquet
    # RW rows that have a DOI but no PMID
    doi_only_mask = df_rw["doi"].notna() & df_rw["pmid"].isna()
    df_doi_only = df_rw[doi_only_mask][
        ["Record ID", "doi", "retraction_date", "reasons", "Title", "Journal"]
    ].copy().rename(
        columns={"Record ID": "record_id", "Title": "title", "Journal": "journal"}
    )
    df_doi_only.to_parquet(out_doi_only_path, index=False)

    # Analyze tag-only PMIDs appearing in RW under any other RetractionNature
    df_non_ret = df_raw[df_raw["RetractionNature"] != "Retraction"].copy()
    df_non_ret["pmid"] = df_non_ret["OriginalPaperPubMedID"].apply(clean_pmid)
    df_non_ret_valid = df_non_ret[df_non_ret["pmid"].notna()].copy()
    df_non_ret_valid["pmid"] = df_non_ret_valid["pmid"].astype(int)
    tag_only_in_other = df_non_ret_valid[df_non_ret_valid["pmid"].isin(tag_only)]

    counts_by_nature_unique = tag_only_in_other.groupby("RetractionNature")["pmid"].nunique()
    counts_by_nature_rows = tag_only_in_other["RetractionNature"].value_counts()

    retracted_with_reinstatement_count = int(df_retracted["has_reinstatement"].sum())
    doi_only_row_count = len(df_doi_only)

    # 4 & 5. Print and save summary
    summary_lines = [
        f"RW rows total: {rw_rows_total}",
        f"After RetractionNature filter: {rw_after_nature_filter}",
        f"RW rows with valid PMID: {rw_rows_with_valid_pmid} (unique PMIDs: {len(rw_unique_pmids)})",
        f"PubMed-tag count: {pubmed_tag_count}",
        f"Overlap: {len(overlap)}",
        f"Union size: {len(union_pmids)}",
        f"RW-only count: {len(rw_only)}",
        f"Tag-only count: {len(tag_only)}",
        f"Retracted PMIDs with has_reinstatement: {retracted_with_reinstatement_count}",
        f"Row count of retracted_doi_only.parquet: {doi_only_row_count}",
        f"Tag-only PMIDs appearing in Retraction Watch under other RetractionNature: "
        f"{tag_only_in_other['pmid'].nunique()} unique PMIDs ({len(tag_only_in_other)} rows):",
    ]
    for nature in counts_by_nature_unique.index:
        summary_lines.append(
            f"  - {nature}: {counts_by_nature_unique[nature]} unique PMIDs ({counts_by_nature_rows.get(nature, 0)} rows)"
        )

    summary_output = "\n".join(summary_lines)
    print(summary_output)

    with open(summary_txt_path, "w") as f:
        f.write(summary_output + "\n")


if __name__ == "__main__":
    main()
