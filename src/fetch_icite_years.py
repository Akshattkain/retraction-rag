"""
Fetch publication years for the one-hop docs retrieved at ranks 1-32 from the NIH iCite API.
The only network access is the iCite API.

1. Take the distinct pmid of rows in data/processed/retrieval_bm25_flagged.parquet where one_hop is True
   (expect 2,296 distinct docids; the distinct PMID count is reported).
2. Query https://icite.od.nih.gov/api/pubs?pmids=<up to 200>&fl=pmid,year
   - On the first response, print the top-level keys and first record, and check pmid and year are present.
   - Batch 200 PMIDs, sleep 0.5 s between requests, retry with exponential backoff on errors, 429 and 5xx.
   - Raw responses are saved to data/raw/icite_years/batch_XXXX.json; batches already saved are skipped.
3. Write data/processed/citing_years.parquet (pmid int64, year Int64; one row per record iCite returned).
   Print requested vs returned, null years, and min/max year.
4. Save printed output to results/icite_years_fetch.txt.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ICITE_API_URL = "https://icite.od.nih.gov/api/pubs"
BATCH_SIZE = 200
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
REQUEST_TIMEOUT = 30  # seconds
EXPECTED_DOCIDS = 2296

FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
RAW_DIR = Path("data/raw/icite_years")
OUTPUT_PARQUET_PATH = Path("data/processed/citing_years.parquet")
OUTPUT_LOG_PATH = Path("results/icite_years_fetch.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("icite_years")

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def fetch_batch_with_retry(session, pmids, batch_idx):
    """Fetch one batch from iCite with exponential backoff on errors, 429 and 5xx."""
    params = {"pmids": ",".join(str(p) for p in pmids), "fl": "pmid,year"}
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(ICITE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Batch {batch_idx:04d}: HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES}); "
                               f"backing off {backoff:.1f}s")
            else:
                resp.raise_for_status()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"Batch {batch_idx:04d}: {e} (attempt {attempt}/{MAX_RETRIES}); backing off {backoff:.1f}s")
        time.sleep(backoff)
        backoff *= 2.0
    raise RuntimeError(f"Batch {batch_idx:04d} failed after {MAX_RETRIES} attempts")


def check_first_response(data):
    """Print and verify the structure of the first response."""
    assert isinstance(data, dict), f"response is not a JSON object: {type(data)}"
    log(f"First response top-level JSON keys: {list(data.keys())}")
    assert isinstance(data.get("data"), list) and len(data["data"]) > 0, "'data' missing, not a list, or empty"
    log(f"First record:\n{json.dumps(data['data'][0], indent=2)}")
    missing = {"pmid", "year"} - set(data["data"][0].keys())
    assert not missing, f"fields missing from first record: {missing}"
    log("Verified fields pmid and year are present in the first record.")


def main():
    log("=== iCite publication years for retrieved one-hop docs (ranks 1-32) ===")
    log(f"Source: {ICITE_API_URL} (fl=pmid,year); fetched/processed {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    df = pd.read_parquet(FLAGGED_PATH, columns=["docid", "pmid", "one_hop"])
    oh = df[df["one_hop"]]
    n_docids = oh["docid"].nunique()
    pmids = sorted(oh["pmid"].astype(int).unique().tolist())
    log(f"One-hop rows: {len(oh):,}; distinct docids: {n_docids:,} (expected {EXPECTED_DOCIDS:,}); "
        f"distinct PMIDs: {len(pmids):,}")
    if n_docids != EXPECTED_DOCIDS:
        log(f"WARNING: distinct docids {n_docids:,} != expected {EXPECTED_DOCIDS:,}")

    batches = [pmids[i:i + BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]
    log(f"{len(batches)} batches of up to {BATCH_SIZE} PMIDs")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    checked, failed, cached, fetched = False, 0, 0, 0
    for idx, batch in enumerate(batches):
        path = RAW_DIR / f"batch_{idx:04d}.json"
        data, from_disk = None, False
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                from_disk = True
                cached += 1
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}; refetching")
        if data is None:
            try:
                data = fetch_batch_with_retry(session, batch, idx)
            except Exception as e:
                logger.error(f"Batch {idx:04d} permanently failed: {e}")
                failed += 1
                continue
            fetched += 1
        if not checked:
            check_first_response(data)
            checked = True
        if not from_disk:
            if not (isinstance(data, dict) and isinstance(data.get("data"), list)):
                logger.error(f"Batch {idx:04d} has no 'data' list; not caching")
                failed += 1
                fetched -= 1
                continue
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    log(f"\nBatches: {len(batches)} total, {fetched} fetched, {cached} from cache, {failed} failed")

    rows = []
    for idx in range(len(batches)):
        path = RAW_DIR / f"batch_{idx:04d}.json"
        if path.exists():
            for rec in json.loads(path.read_text(encoding="utf-8")).get("data", []):
                if rec.get("pmid") is not None:
                    rows.append((int(rec["pmid"]), rec.get("year")))
    out = pd.DataFrame(rows, columns=["pmid", "year"]).drop_duplicates("pmid")
    out["pmid"] = out["pmid"].astype("int64")
    out["year"] = pd.to_numeric(out["year"]).astype("Int64")
    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    returned = set(out["pmid"])
    log(f"Wrote {OUTPUT_PARQUET_PATH} ({len(out):,} rows)")
    log(f"PMIDs requested: {len(pmids):,}; returned by iCite: {len(returned & set(pmids)):,}; "
        f"not returned: {len(set(pmids) - returned):,}; returned but not requested: {len(returned - set(pmids)):,}")
    log(f"Null years among returned: {int(out['year'].isna().sum()):,}")
    if out["year"].notna().any():
        log(f"Year range: min {int(out['year'].min())}, max {int(out['year'].max())}")

    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
