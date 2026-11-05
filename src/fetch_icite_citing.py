"""
Fetch citing papers for retracted PMIDs from the NIH iCite API.

1. Load data/processed/retracted.parquet (37,201 PMIDs).
2. Query https://icite.od.nih.gov/api/pubs?pmids=<up to 200 comma-separated>&fl=pmid,year,cited_by
   - On the first successful response, print top-level keys and first record.
   - Assert "data" key exists and has at least one record before caching anything.
   - Check pmid, year, and cited_by fields are present. Treat null cited_by as empty.
3. Batch 200 PMIDs per request, sleep 0.5s between requests, retry with exponential backoff.
   - Only write batch file to data/raw/icite/ if response has a "data" list.
   - Log batches that return fewer records than PMIDs requested.
   - Skip batches already saved on disk to resume reruns.
4. Write data/processed/icite_citing_pairs.parquet (columns: retracted_pmid, citing_pmid).
   - Drop duplicate (retracted_pmid, citing_pmid) pairs before writing and report count dropped.
5. Print and save summary statistics to results/icite_fetch.txt:
   - Batches done and failed;
   - Batches returning fewer records than requested;
   - Total records returned vs 37,201 requested;
   - Retracted PMIDs requested vs returned by iCite;
   - How many have at least one citing paper;
   - Total pairs; unique citing PMIDs;
   - Median and max citing papers per retracted PMID;
   - Top 5 retracted PMIDs with the most citing papers.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Base configuration
ICITE_API_URL = "https://icite.od.nih.gov/api/pubs"
BATCH_SIZE = 200
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
REQUEST_TIMEOUT = 30  # seconds

RETRACTED_PARQUET_PATH = Path("data/processed/retracted.parquet")
RAW_ICITE_DIR = Path("data/raw/icite")
OUTPUT_PARQUET_PATH = Path("data/processed/icite_citing_pairs.parquet")
OUTPUT_LOG_PATH = Path("results/icite_fetch.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("icite_fetch")


def parse_citing_pmids(val: Any) -> List[int]:
    """Parse citing PMIDs from iCite cited_by field into a list of integers."""
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            if x is not None and str(x).strip():
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
        return out
    if isinstance(val, str):
        out = []
        for x in val.replace(",", " ").split():
            try:
                out.append(int(x))
            except (ValueError, TypeError):
                pass
        return out
    return []


def fetch_batch_with_retry(
    session: requests.Session,
    pmids: List[int],
    batch_idx: int,
) -> Dict[str, Any]:
    """Fetch one batch of PMIDs from iCite API with exponential backoff on errors/429/5xx."""
    pmids_str = ",".join(str(p) for p in pmids)
    params = {
        "pmids": pmids_str,
        "fl": "pmid,year,cited_by",
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(ICITE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return data

            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    f"Batch {batch_idx:04d}: received HTTP {resp.status_code} "
                    f"(attempt {attempt}/{MAX_RETRIES}). Backing off for {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0
            else:
                resp.raise_for_status()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(
                f"Batch {batch_idx:04d}: request error '{e}' "
                f"(attempt {attempt}/{MAX_RETRIES}). Backing off for {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError(f"Batch {batch_idx:04d} failed after {MAX_RETRIES} attempts.")


def main():
    output_lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("=== NIH iCite Citing Papers Fetch for Retracted PMIDs ===")

    # 1. Load retracted PMIDs
    log(f"Loading retracted PMIDs from {RETRACTED_PARQUET_PATH}...")
    df_retracted = pd.read_parquet(RETRACTED_PARQUET_PATH)
    all_retracted_pmids: List[int] = sorted(df_retracted["pmid"].dropna().astype(int).unique().tolist())
    total_requested = len(all_retracted_pmids)
    log(f"Loaded {total_requested:,} unique retracted PMIDs.")

    # Prepare batches
    batches: List[List[int]] = [
        all_retracted_pmids[i : i + BATCH_SIZE]
        for i in range(0, total_requested, BATCH_SIZE)
    ]
    num_batches = len(batches)
    log(f"Prepared {num_batches} batches of up to {BATCH_SIZE} PMIDs.")

    RAW_ICITE_DIR.mkdir(parents=True, exist_ok=True)

    # 2 & 3. Fetch batches
    batches_done = 0
    batches_failed = 0
    batches_from_cache = 0
    batches_network_fetched = 0
    batches_fewer_than_requested = 0
    total_records_returned = 0

    checked_first_response = False
    session = requests.Session()

    for batch_idx, batch_pmids in enumerate(batches):
        batch_file = RAW_ICITE_DIR / f"cited_by_batch_{batch_idx:04d}.json"

        batch_data = None
        loaded_from_disk = False

        if batch_file.exists():
            try:
                with open(batch_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)
                loaded_from_disk = True
                batches_from_cache += 1
            except Exception as e:
                logger.warning(f"Failed to read cached batch file {batch_file}: {e}. Refetching.")
                batch_data = None

        if batch_data is None:
            try:
                batch_data = fetch_batch_with_retry(session, batch_pmids, batch_idx)
                batches_network_fetched += 1
            except Exception as e:
                logger.error(f"Batch {batch_idx:04d} permanently failed: {e}")
                batches_failed += 1
                continue

        # Check first response BEFORE writing any file if newly fetched
        if not checked_first_response:
            if not isinstance(batch_data, dict):
                raise TypeError(f"Response is not a JSON object: {type(batch_data)}")
            top_keys = list(batch_data.keys())
            log(f"\nFirst response top-level JSON keys: {top_keys}")
            
            assert "data" in batch_data, f"Top-level 'data' key missing from iCite response! Keys: {top_keys}"
            records = batch_data.get("data")
            assert isinstance(records, list), f"'data' field is not a list: {type(records)}"
            assert len(records) > 0, f"'data' list is empty in first response!"
            
            log(f"First record in response:\n{json.dumps(records[0], indent=2)}")
            
            sample_keys = set(records[0].keys())
            for req_field in ["pmid", "year", "cited_by"]:
                assert req_field in sample_keys, (
                    f"Field '{req_field}' missing from iCite record! Available: {sample_keys}"
                )
            log("Verified required fields (pmid, year, cited_by) are present in first record.\n")
            checked_first_response = True

        # Only write batch file to disk if newly fetched and has "data" list
        if not loaded_from_disk:
            if isinstance(batch_data, dict) and isinstance(batch_data.get("data"), list):
                with open(batch_file, "w", encoding="utf-8") as f:
                    json.dump(batch_data, f)
            else:
                logger.error(f"Batch {batch_idx:04d} response does not have a 'data' list. Not caching.")
                batches_failed += 1
                continue
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        batches_done += 1

        # Check records returned vs requested
        rec_list = batch_data.get("data", [])
        rec_count = len(rec_list)
        total_records_returned += rec_count
        if rec_count < len(batch_pmids):
            logger.info(f"Batch {batch_idx:04d}: returned {rec_count} records for {len(batch_pmids)} requested PMIDs.")
            batches_fewer_than_requested += 1

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            logger.info(
                f"Progress: {batch_idx + 1}/{num_batches} batches processed "
                f"(Done: {batches_done}, Failed: {batches_failed}, Fewer: {batches_fewer_than_requested})"
            )

    log(f"\n--- Batch Fetch Summary ---")
    log(f"Total batches requested: {num_batches}")
    log(f"Batches completed successfully: {batches_done}")
    log(f"Batches failed: {batches_failed}")
    log(f"Batches loaded from cache: {batches_from_cache}")
    log(f"Batches fetched from network: {batches_network_fetched}")
    log(f"Batches returning fewer records than requested: {batches_fewer_than_requested} / {num_batches} ({batches_fewer_than_requested / num_batches * 100:.2f}%)")
    log(f"Total records returned: {total_records_returned:,} vs {total_requested:,} requested ({total_records_returned / total_requested * 100:.2f}%)")

    # 4. Parse all raw batch files and build pair table
    log(f"\nProcessing all batch files to build citing pairs table...")
    retracted_pmid_set = set(all_retracted_pmids)
    pmids_returned_by_icite: Set[int] = set()

    # Track citation counts per retracted PMID: pmid -> set of citing PMIDs
    citations_per_retracted: Dict[int, Set[int]] = {p: set() for p in all_retracted_pmids}

    pair_retracted_list: List[int] = []
    pair_citing_list: List[int] = []

    for batch_idx in range(num_batches):
        batch_file = RAW_ICITE_DIR / f"cited_by_batch_{batch_idx:04d}.json"
        if not batch_file.exists():
            continue

        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        records = batch_data.get("data", [])
        for rec in records:
            raw_p = rec.get("pmid")
            if raw_p is None:
                continue
            try:
                ret_pmid = int(raw_p)
            except (ValueError, TypeError):
                continue

            pmids_returned_by_icite.add(ret_pmid)
            citing_list = parse_citing_pmids(rec.get("cited_by"))

            # Update tracking
            if ret_pmid in citations_per_retracted:
                citations_per_retracted[ret_pmid].update(citing_list)

            for c_pmid in citing_list:
                pair_retracted_list.append(ret_pmid)
                pair_citing_list.append(c_pmid)

    raw_pairs_count = len(pair_retracted_list)
    log(f"Extracted {raw_pairs_count:,} raw (retracted_pmid, citing_pmid) pairs.")

    # 3. Drop duplicate pairs before writing Parquet
    log("Deduplicating pairs...")
    df_pairs = pd.DataFrame({
        "retracted_pmid": pd.Series(pair_retracted_list, dtype="int64"),
        "citing_pmid": pd.Series(pair_citing_list, dtype="int64"),
    })
    df_pairs_unique = df_pairs.drop_duplicates()
    dropped_duplicates = raw_pairs_count - len(df_pairs_unique)
    total_pairs = len(df_pairs_unique)
    log(f"Duplicate pairs dropped: {dropped_duplicates:,}")
    log(f"Final unique pairs: {total_pairs:,}")

    # Write Parquet table
    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_pairs_unique, preserve_index=False)
    pq.write_table(table, OUTPUT_PARQUET_PATH, compression="snappy")
    log(f"Saved pair table to {OUTPUT_PARQUET_PATH} ({len(df_pairs_unique):,} rows).")

    # 5. Statistics & Summary
    log(f"\n=== Summary Metrics ===")
    unique_citing_pmids = df_pairs_unique["citing_pmid"].nunique()
    retracted_returned_count = len(pmids_returned_by_icite.intersection(retracted_pmid_set))

    # Citation counts per requested retracted PMID
    citing_counts_all = pd.Series([len(citations_per_retracted[p]) for p in all_retracted_pmids])
    with_citations_count = (citing_counts_all > 0).sum()

    median_all = citing_counts_all.median()
    max_all = citing_counts_all.max()

    # Also compute for only returned PMIDs
    citing_counts_returned = pd.Series([len(citations_per_retracted[p]) for p in pmids_returned_by_icite if p in citations_per_retracted])
    median_returned = citing_counts_returned.median() if len(citing_counts_returned) > 0 else 0.0

    log(f"Retracted PMIDs requested: {total_requested:,}")
    log(f"Retracted PMIDs returned by iCite: {retracted_returned_count:,} ({retracted_returned_count / total_requested * 100:.2f}%)")
    log(f"Retracted PMIDs with at least one citing paper: {with_citations_count:,} ({with_citations_count / total_requested * 100:.2f}%)")
    log(f"Total unique (retracted_pmid, citing_pmid) pairs: {total_pairs:,}")
    log(f"Unique citing PMIDs: {unique_citing_pmids:,}")
    log(f"Citing papers per retracted PMID (across all {total_requested:,} requested):")
    log(f"  - Median: {median_all:.1f}")
    log(f"  - Max: {max_all:,}")
    log(f"Citing papers per retracted PMID (across {retracted_returned_count:,} returned):")
    log(f"  - Median: {median_returned:.1f}")

    # Top 5 retracted PMIDs with the most citing papers
    sorted_top = sorted(citations_per_retracted.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    log(f"\nTop 5 retracted PMIDs with the most citing papers:")
    log(f"{'Rank':<6} {'Retracted PMID':<16} {'Citing Papers Count':<20}")
    log("-" * 42)
    for rank, (pmid, cit_set) in enumerate(sorted_top, start=1):
        log(f"{rank:<6} {pmid:<16} {len(cit_set):<20,}")

    # 6. Save printed output to results/icite_fetch.txt
    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"\nPrinted summary saved to {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()
