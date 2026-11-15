"""
Exposure analysis of BM25 retrieval over MIRAGE (read-only; no network).

Reads data/processed/retrieval_bm25_flagged.parquet and data/processed/retracted.parquet
(plus subset_flags.parquet for corpus base rates and the MIRAGE benchmark for question text).

1. Exposure by rank bucket (1-5, 6-16, 17-32): share of rows that are is_retracted and
   one_hop, overall and per dataset, next to the corpus base rates.
2. Distinct retracted docs retrieved: count, number of questions each appears in,
   10 most-retrieved retracted docs.
3. Five case studies of questions with a retracted hit (2 bioasq, 3 from other datasets,
   seed 0): qkey, question (first 200 chars), rank, pmid, score, title (from
   data/raw/pubmed_subset/chunk/*.jsonl), retraction reasons and date, in_situ.
4. Share of one-hop rows with n_retracted_cited >= 3, overall and for bioasq.
5. Save printed output to results/exposure_analysis.txt.
"""

import json
import random
from pathlib import Path

import pandas as pd

FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
RETRACTED_PATH = Path("data/processed/retracted.parquet")
FLAGS_PATH = Path("data/processed/subset_flags.parquet")
BENCHMARK_PATH = Path("third_party/MIRAGE/benchmark.json")
CHUNK_DIR = Path("data/raw/pubmed_subset/chunk")
RESULTS_PATH = Path("results/exposure_analysis.txt")

SEED = 0
BUCKETS = [("1-5", 1, 5), ("6-16", 6, 16), ("17-32", 17, 32)]
FLAGS = ["is_retracted", "one_hop"]
N_BIOASQ_CASES = 2
N_OTHER_CASES = 3
MIN_CITED = 3

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def bucket_of(rank):
    for name, lo, hi in BUCKETS:
        if lo <= rank <= hi:
            return name
    raise ValueError(f"rank out of range: {rank}")


def lookup_titles(docids):
    """Map docid -> title by scanning only the chunk files that contain those ids."""
    by_file = {}
    for d in docids:
        by_file.setdefault(d.rsplit("_", 1)[0], set()).add(d)
    titles = {}
    for stem, wanted in by_file.items():
        path = CHUNK_DIR / f"{stem}.jsonl"
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in wanted:
                    titles[rec["id"]] = rec["title"]
                    if len(titles) == len(docids):
                        return titles
    missing = set(docids) - set(titles)
    assert not missing, f"titles not found for: {sorted(missing)}"
    return titles


def rank_bucket_section(df, base):
    log("=== 1. Exposure by rank bucket (share of rows) ===")
    log("Corpus base rate: " + ", ".join(f"{f} {base[f]:.4%}" for f in FLAGS))
    df = df.assign(bucket=df["rank"].map(bucket_of))
    groups = [("overall", df)] + [(ds, df[df["dataset"] == ds]) for ds in df["dataset"].drop_duplicates()]
    for flag in FLAGS:
        log(f"\n{flag}  (base {base[flag]:.4%})")
        log(f"  {'group':<10}" + "".join(f"{name:>12}" for name, _, _ in BUCKETS))
        for gname, g in groups:
            shares = g.groupby("bucket")[flag].mean()
            log(f"  {gname:<10}" + "".join(f"{shares[name]:>12.4%}" for name, _, _ in BUCKETS))


def distinct_retracted_section(df, retracted_hits):
    log("\n=== 2. Distinct retracted docs retrieved ===")
    n_q = retracted_hits.groupby("docid")["qkey"].nunique().sort_values(ascending=False)
    n_rows = retracted_hits.groupby("docid").size()
    log(f"Retracted rows: {len(retracted_hits):,} of {len(df):,}")
    log(f"Distinct retracted docids: {len(n_q):,}")
    log(f"Questions with >=1 retracted hit: {retracted_hits['qkey'].nunique():,}")
    log("Questions per retracted doc: "
        f"min {n_q.min()}, median {n_q.median():.1f}, mean {n_q.mean():.2f}, max {n_q.max()}")
    log(f"  appear in exactly 1 question: {int((n_q == 1).sum()):,}")
    log(f"  appear in >=2 questions: {int((n_q >= 2).sum()):,}")
    log(f"  appear in >=5 questions: {int((n_q >= 5).sum()):,}")
    top = n_q.head(10)
    titles = lookup_titles(list(top.index))
    pmids = retracted_hits.drop_duplicates("docid").set_index("docid")["pmid"]
    log("\nTop 10 most-retrieved retracted docs:")
    for docid, nq in top.items():
        log(f"  {docid}  pmid {pmids[docid]}  questions {nq}  rows {n_rows[docid]}  | {titles[docid][:100]}")


def case_study_section(retracted_hits, retracted):
    log("\n=== 3. Case studies (seed 0) ===")
    rng = random.Random(SEED)
    qkeys = retracted_hits.groupby("qkey")["dataset"].first()
    bio = sorted(qkeys[qkeys == "bioasq"].index)
    other = sorted(qkeys[qkeys != "bioasq"].index)
    chosen = rng.sample(bio, N_BIOASQ_CASES) + rng.sample(other, N_OTHER_CASES)

    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)
    # best-ranked retracted hit per chosen question
    hits = (retracted_hits[retracted_hits["qkey"].isin(chosen)]
            .sort_values(["qkey", "rank"]))
    n_hits = hits.groupby("qkey").size()
    best = hits.drop_duplicates("qkey").set_index("qkey")
    titles = lookup_titles(list(best["docid"]))
    ret = retracted.set_index("pmid")

    for i, qkey in enumerate(chosen, start=1):
        row = best.loc[qkey]
        dataset, qid = qkey.split("/", 1)
        question = bench[dataset][qid]["question"]
        info = ret.loc[int(row["pmid"])]
        reasons = "; ".join(info["reasons"]) if info["reasons"] is not None else "n/a"
        date = info["retraction_date"]
        date = "n/a" if pd.isna(date) else date.strftime("%Y-%m-%d")
        log(f"\nCase {i}: {qkey}")
        log(f"  question: {question[:200]!r}")
        log(f"  retracted hits for this question: {n_hits[qkey]} (showing best-ranked)")
        log(f"  rank {row['rank']}, pmid {row['pmid']}, docid {row['docid']}, score {row['score']:.4f}")
        log(f"  title: {titles[row['docid']]}")
        log(f"  retraction reasons: {reasons}")
        log(f"  retraction date: {date}")
        log(f"  in_situ: {bool(row['in_situ'])}")


def n_cited_section(df, flags):
    log(f"\n=== 4. One-hop rows with n_retracted_cited >= {MIN_CITED} ===")
    one_hop = df[df["one_hop"]]
    log(f"Overall: {(one_hop['n_retracted_cited'] >= MIN_CITED).mean():.4%} "
        f"of {len(one_hop):,} one-hop rows")
    bio = one_hop[one_hop["dataset"] == "bioasq"]
    log(f"bioasq: {(bio['n_retracted_cited'] >= MIN_CITED).mean():.4%} of {len(bio):,} one-hop rows")
    corpus = flags[flags["one_hop"]]
    log(f"Corpus one-hop docs (for comparison): "
        f"{(corpus['n_retracted_cited'] >= MIN_CITED).mean():.4%} of {len(corpus):,}")


def main():
    log("=== Exposure analysis (BM25, MIRAGE) ===")
    df = pd.read_parquet(FLAGGED_PATH)
    retracted = pd.read_parquet(RETRACTED_PATH)
    flags = pd.read_parquet(FLAGS_PATH, columns=FLAGS + ["n_retracted_cited"])
    log(f"Rows: {len(df):,}, questions: {df['qkey'].nunique():,}\n")

    rank_bucket_section(df, flags[FLAGS].mean())
    retracted_hits = df[df["is_retracted"]]
    distinct_retracted_section(df, retracted_hits)
    case_study_section(retracted_hits, retracted)
    n_cited_section(df, flags)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
