"""
Inspect one-hop "hub" documents in BM25 retrieval over MIRAGE (read-only; no network).

Reads data/processed/retrieval_bm25_flagged.parquet, data/processed/icite_citing_pairs.parquet,
data/processed/retracted.parquet (plus subset_flags.parquet for base rates and the MIRAGE
benchmark for the per-dataxset question counts used as denominators).

1. Top 25 one-hop docs by number of questions (all ranks) and top 25 for ranks 1-5:
   docid, pmid, title, content word count, questions, rows, questions per dataset,
   median rank, n_retracted_cited, PMIDs of the retracted papers it cites (iCite pairs).
2. One-hop question-level exposure at k=5 and k=32 (denominator: all questions, overall and
   per dataset) when excluding: nothing, top 1 doc, top 10 docs, every doc in >=5 questions.
   Excluded docs are dropped from the ranked lists without back-filling, so this is a
   lower bound on exposure after replacing them with the next-best documents.
3. Distinct retrieved docids: total, and share that are is_retracted / one_hop, next to
   corpus base rates.
4. Save printed output to results/hub_inspect.txt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.extra_checks import lookup_docs, question_counts

FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
PAIRS_PATH = Path("data/processed/icite_citing_pairs.parquet")
RETRACTED_PATH = Path("data/processed/retracted.parquet")
FLAGS_PATH = Path("data/processed/subset_flags.parquet")
RESULTS_PATH = Path("results/hub_inspect.txt")

TOP_N = 25
KS = [5, 32]
EXCLUDE_TOP_NS = [1, 10]
HUB_MIN_QUESTIONS = 5
EXPECTED_QUESTIONS = 7663

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def rank_hubs(one_hop):
    """One row per one-hop docid, sorted by questions desc, rows desc, docid asc."""
    g = one_hop.groupby("docid")
    hubs = pd.DataFrame({
        "pmid": g["pmid"].first(),
        "questions": g["qkey"].nunique(),
        "rows": g.size(),
        "median_rank": g["rank"].median(),
        "n_retracted_cited": g["n_retracted_cited"].first(),
    })
    hubs = hubs.reset_index().sort_values(["questions", "rows", "docid"], ascending=[False, False, True])
    return hubs.reset_index(drop=True)


def cited_retracted_map(pairs, retracted_pmids, pmids):
    sub = pairs[pairs["citing_pmid"].isin(pmids) & pairs["retracted_pmid"].isin(retracted_pmids)]
    return sub.groupby("citing_pmid")["retracted_pmid"].agg(lambda s: sorted(set(s))).to_dict()


def print_hub_list(title, one_hop, docs, cited):
    log(f"\n--- {title} ---")
    hubs = rank_hubs(one_hop).head(TOP_N)
    for i, h in hubs.iterrows():
        docid = h["docid"]
        sub = one_hop[one_hop["docid"] == docid]
        per_ds = sub.groupby("dataset")["qkey"].nunique().sort_values(ascending=False)
        per_ds_s = ", ".join(f"{ds} {n}" for ds, n in per_ds.items())
        t, content = docs[docid]
        cited_pmids = cited.get(int(h["pmid"]), [])
        # consistency with the flag built in build_one_hop_flags.py
        if len(cited_pmids) != h["n_retracted_cited"]:
            log(f"  WARNING: {docid} lists {len(cited_pmids)} cited retracted PMIDs "
                f"but n_retracted_cited = {h['n_retracted_cited']}")
        log(f"{i + 1:>2}. {docid}  pmid {h['pmid']}")
        log(f"    title: {t}")
        log(f"    content words: {len(content.split()):,}")
        log(f"    questions {h['questions']}, rows {h['rows']}, median rank {h['median_rank']:.1f}")
        log(f"    questions per dataset: {per_ds_s}")
        log(f"    n_retracted_cited {h['n_retracted_cited']}; cites retracted PMIDs: {cited_pmids}")


def exposure_section(one_hop, counts, hubs):
    log("\n=== 2. One-hop question-level exposure with hub exclusion ===")
    log("Excluded docs are removed without back-filling (lower bound on exposure).")
    scenarios = [("nothing", set())]
    for n in EXCLUDE_TOP_NS:
        scenarios.append((f"top {n} doc{'s' if n > 1 else ''}", set(hubs["docid"].head(n))))
    big = set(hubs.loc[hubs["questions"] >= HUB_MIN_QUESTIONS, "docid"])
    scenarios.append((f"all docs in >={HUB_MIN_QUESTIONS} questions ({len(big)} docs)", big))

    datasets = list(counts)
    groups = [("overall", EXPECTED_QUESTIONS, None)] + [(ds, counts[ds], ds) for ds in datasets]
    for k in KS:
        log(f"\nk = {k}: share (count) of questions with >=1 one-hop hit within ranks <= {k}")
        log(f"  {'excluding':<42}" + "".join(f"{g[0]:>18}" for g in groups))
        for name, excluded in scenarios:
            sub = one_hop[(one_hop["rank"] <= k) & ~one_hop["docid"].isin(excluded)]
            cells = []
            for _, n_q, ds in groups:
                s = sub if ds is None else sub[sub["dataset"] == ds]
                n = s["qkey"].nunique()
                cells.append(f"{n / n_q:.2%} ({n:,})")
            log(f"  {name:<42}" + "".join(f"{c:>18}" for c in cells))


def distinct_section(df, flags):
    log("\n=== 3. Distinct retrieved docids vs corpus base rates ===")
    distinct = df.drop_duplicates("docid")
    log(f"Distinct retrieved docids: {len(distinct):,} (of {len(flags):,} in corpus subset, "
        f"{len(distinct) / len(flags):.2%})")
    for flag in ["is_retracted", "one_hop"]:
        log(f"  {flag}: {distinct[flag].mean():.4%} of distinct retrieved docs "
            f"({int(distinct[flag].sum()):,}) | corpus base rate {flags[flag].mean():.4%}")


def main():
    log("=== Hub inspection (BM25, MIRAGE) ===")
    df = pd.read_parquet(FLAGGED_PATH)
    pairs = pd.read_parquet(PAIRS_PATH)
    retracted_pmids = set(pd.read_parquet(RETRACTED_PATH, columns=["pmid"])["pmid"].astype(int))
    flags = pd.read_parquet(FLAGS_PATH, columns=["is_retracted", "one_hop"])
    counts = question_counts()
    log(f"Rows: {len(df):,}; one-hop rows: {int(df['one_hop'].sum()):,}\n")

    one_hop = df[df["one_hop"]]
    top_all = rank_hubs(one_hop).head(TOP_N)
    top_5 = rank_hubs(one_hop[one_hop["rank"] <= 5]).head(TOP_N)

    needed = set(top_all["docid"]) | set(top_5["docid"])
    docs = lookup_docs(needed)
    pmids = set(one_hop.loc[one_hop["docid"].isin(needed), "pmid"].astype(int))
    cited = cited_retracted_map(pairs, retracted_pmids, pmids)

    log(f"=== 1. Top {TOP_N} one-hop docs ===")
    print_hub_list(f"Top {TOP_N} by number of questions (ranks 1-32)", one_hop, docs, cited)
    print_hub_list(f"Top {TOP_N} by number of questions (ranks 1-5 only)",
                   one_hop[one_hop["rank"] <= 5], docs, cited)

    # exclusion ranking: all ranks, by number of questions
    exposure_section(one_hop, counts, rank_hubs(one_hop))
    distinct_section(df, flags)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
