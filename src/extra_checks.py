"""
Extra checks on BM25 retrieval exposure (read-only; no network).

Reads data/processed/retrieval_bm25_flagged.parquet (plus the MIRAGE benchmark for the
per-dataset question counts that form the denominators, 7,663 in total).

1. Question-level exposure by k (1, 3, 5, 10, 32): share and count of questions with at
   least one is_retracted / one_hop / is_notice hit within ranks <= k, overall and per dataset.
2. For is_retracted rows: look up title and content by docid in
   data/raw/pubmed_subset/chunk/*.jsonl, count docs containing "retract" (case-insensitive)
   in title or content, and print those titles.
3. One-hop hubs: distinct one-hop docids, questions per doc (median, max, number in >=5
   questions), share of one-hop rows covered by the 10 and 100 most-retrieved one-hop
   docs, all ranks and ranks 1-5 only.
4. Save printed output to results/extra_checks.txt.
"""

import json
from pathlib import Path

import pandas as pd

FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
BENCHMARK_PATH = Path("third_party/MIRAGE/benchmark.json")
CHUNK_DIR = Path("data/raw/pubmed_subset/chunk")
RESULTS_PATH = Path("results/extra_checks.txt")

KS = [1, 3, 5, 10, 32]
FLAGS = ["is_retracted", "one_hop", "is_notice"]
TOP_NS = [10, 100]
EXPECTED_QUESTIONS = 7663

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def question_counts():
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)
    counts = {ds: len(items) for ds, items in bench.items()}
    assert sum(counts.values()) == EXPECTED_QUESTIONS, counts
    return counts


def lookup_docs(docids):
    """Map docid -> (title, content), scanning only the chunk files that hold those ids."""
    docids = set(docids)
    by_file = {}
    for d in docids:
        by_file.setdefault(d.rsplit("_", 1)[0], set()).add(d)
    found = {}
    for stem, wanted in by_file.items():
        remaining = set(wanted)
        with open(CHUNK_DIR / f"{stem}.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in remaining:
                    found[rec["id"]] = (rec["title"] or "", rec["content"] or "")
                    remaining.discard(rec["id"])
                    if not remaining:
                        break
    missing = docids - set(found)
    assert not missing, f"docs not found: {sorted(missing)[:5]} ({len(missing)} total)"
    return found


def exposure_by_k_section(df, counts):
    log("=== 1. Question-level exposure by k (denominator = all questions in group) ===")
    datasets = list(counts)
    groups = [("overall", EXPECTED_QUESTIONS, df)] + [
        (ds, counts[ds], df[df["dataset"] == ds]) for ds in datasets
    ]
    for flag in FLAGS:
        log(f"\n{flag}: questions with >=1 hit within ranks <= k  (count, share)")
        log(f"  {'group':<10}{'n_q':>7}" + "".join(f"{'k=' + str(k):>16}" for k in KS))
        for gname, n_q, g in groups:
            cells = []
            for k in KS:
                n = g[(g["rank"] <= k) & g[flag]]["qkey"].nunique()
                cells.append(f"{n:>6,} {n / n_q:>7.2%}")
            log(f"  {gname:<10}{n_q:>7,}" + "".join(f"{c:>16}" for c in cells))


def retract_mention_section(df):
    log("\n=== 2. Retracted docs whose title/content mention 'retract' ===")
    ret = df[df["is_retracted"]]
    docids = sorted(ret["docid"].unique())
    docs = lookup_docs(docids)
    pmids = ret.drop_duplicates("docid").set_index("docid")["pmid"]
    n_rows = ret.groupby("docid").size()
    hits = [d for d in docids if "retract" in (docs[d][0] + " " + docs[d][1]).lower()]
    in_title = [d for d in hits if "retract" in docs[d][0].lower()]
    log(f"Distinct retracted docids: {len(docids):,} ({len(ret):,} rows)")
    log(f"Containing 'retract' in title or content: {len(hits):,} docs "
        f"({int(n_rows[hits].sum()) if hits else 0:,} rows)")
    log(f"  of which in title: {len(in_title):,}")
    for d in hits:
        where = "title" if d in in_title else "content"
        log(f"  {d}  pmid {pmids[d]}  [{where}]  rows {n_rows[d]}  | {docs[d][0]}")


def hub_stats(one_hop, label):
    n_rows = len(one_hop)
    log(f"\n{label}: {n_rows:,} one-hop rows")
    per_doc_q = one_hop.groupby("docid")["qkey"].nunique().sort_values(ascending=False)
    per_doc_rows = one_hop.groupby("docid").size().sort_values(ascending=False)
    log(f"  distinct one-hop docids: {len(per_doc_q):,}")
    log(f"  questions per doc: median {per_doc_q.median():.1f}, max {per_doc_q.max()}, "
        f"in >=5 questions: {int((per_doc_q >= 5).sum()):,}")
    for n in TOP_NS:
        covered = per_doc_rows.head(n).sum()
        log(f"  top {n} most-retrieved docs cover {covered:,} rows ({covered / n_rows:.2%})")


def hubs_section(df):
    log("\n=== 3. One-hop hubs ===")
    one_hop = df[df["one_hop"]]
    hub_stats(one_hop, "All ranks (1-32)")
    hub_stats(one_hop[one_hop["rank"] <= 5], "Ranks 1-5 only")


def main():
    log("=== Extra checks (BM25, MIRAGE) ===")
    df = pd.read_parquet(FLAGGED_PATH)
    counts = question_counts()
    log(f"Rows: {len(df):,}, questions with >=1 row: {df['qkey'].nunique():,}, "
        f"questions in benchmark: {sum(counts.values()):,}\n")

    exposure_by_k_section(df, counts)
    retract_mention_section(df)
    hubs_section(df)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
