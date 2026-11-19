"""
Retraction severity and exposure (BM25, MIRAGE). Read-only except for the two outputs; no network.

Reads data/processed/retracted.parquet, retrieval_bm25_flagged.parquet, icite_citing_pairs.parquet,
notice_exclude_pmids.txt, subset_docs.parquet (question denominators from benchmark.json).
Writes data/processed/retracted_classes.parquet (pmid, reason_class, source_group) and
results/severity_exposure.txt.

Class of a paper = highest of A > B > C > D among its reasons, "U" if none (see reason_map.py).
Views: strict = A; broad = A or B; all = every retracted paper.
source_group: both (RW and PubMed tag), rw_only, tag_only.

1. Class sizes and source-group sizes: all retracted PMIDs; those in the corpus subset; the
   distinct retracted docs retrieved.
2. Direct exposure: share (count) of questions with >=1 retrieved retracted doc, k=5 and k=32,
   for strict / broad / all and by source group; overall and per dataset.
3. One-hop: strongest class among the retracted papers each corpus doc cites (iCite pairs after
   dropping notice_exclude_pmids, restricted to subset docs). Share (count) of questions with a
   one-hop doc (not retracted, not a notice) whose strongest cited class is in strict / broad /
   all, k=5 and k=32, with and without the docs that appear in >=5 questions.
   Excluded docs are dropped without back-filling (lower bound), as in hub_inspect.py.
4. One-hop docs citing only class D or U retracted papers; top 10 hub docs with strongest class.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.extra_checks import question_counts
from src.hub_inspect import rank_hubs
from src.reason_map import CLASSES, CLASS_RANK, RANK_CLASS, check_against_parquet, paper_class

RETRACTED_PATH = Path("data/processed/retracted.parquet")
FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
PAIRS_PATH = Path("data/processed/icite_citing_pairs.parquet")
NOTICE_EXCLUDE_PATH = Path("data/processed/notice_exclude_pmids.txt")
SUBSET_DOCS_PATH = Path("data/processed/subset_docs.parquet")
CLASSES_PATH = Path("data/processed/retracted_classes.parquet")
RESULTS_PATH = Path("results/severity_exposure.txt")

KS = [5, 32]
HUB_MIN_QUESTIONS = 5
EXPECTED_QUESTIONS = 7663
EXPECTED_RETRACTED = 37201
EXPECTED_SUBSET_RETRACTED = 1528
EXPECTED_RETRIEVED_RETRACTED = 189
EXPECTED_HUBS = 23
TOP_HUBS = 10

# view name -> minimum class rank (strict: A; broad: A or B; all: everything)
VIEWS = {"strict": CLASS_RANK["A"], "broad": CLASS_RANK["B"], "all": 0}
SOURCE_GROUPS = ["both", "rw_only", "tag_only"]

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def source_group(row):
    if row["in_rw"] and row["in_pubmed_tag"]:
        return "both"
    if row["in_rw"]:
        return "rw_only"
    assert row["in_pubmed_tag"], f"pmid {row['pmid']} is in neither source"
    return "tag_only"


def build_classes():
    df = pd.read_parquet(RETRACTED_PATH)
    assert df["pmid"].is_unique and len(df) == EXPECTED_RETRACTED, len(df)
    out = pd.DataFrame({
        "pmid": df["pmid"].astype("int64"),
        "reason_class": df["reasons"].map(paper_class),
        "source_group": df.apply(source_group, axis=1),
    })
    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CLASSES_PATH, index=False)
    return out


def size_table(title, cls):
    """Class and source-group sizes for a subset of retracted_classes rows."""
    n = len(cls)
    log(f"\n{title} (n = {n:,} PMIDs)")
    by_class = cls["reason_class"].value_counts()
    log("  class:        " + "  ".join(f"{c} {int(by_class.get(c, 0)):,} ({by_class.get(c, 0) / n:.1%})"
                                       for c in CLASSES))
    by_src = cls["source_group"].value_counts()
    log("  source group: " + "  ".join(f"{g} {int(by_src.get(g, 0)):,} ({by_src.get(g, 0) / n:.1%})"
                                       for g in SOURCE_GROUPS))


def exposure_table(title, sets, counts):
    """sets: {row label: DataFrame of qualifying rows with qkey, dataset, rank}."""
    groups = [("overall", EXPECTED_QUESTIONS, None)] + [(ds, n, ds) for ds, n in counts.items()]
    for k in KS:
        log(f"\n{title}, k = {k}: share (count) of questions")
        log(f"  {'':<34}" + "".join(f"{g[0]:>18}" for g in groups))
        for label, rows in sets.items():
            sub = rows[rows["rank"] <= k]
            cells = []
            for _, n_q, ds in groups:
                s = sub if ds is None else sub[sub["dataset"] == ds]
                n = s["qkey"].nunique()
                cells.append(f"{n / n_q:.2%} ({n:,})")
            log(f"  {label:<34}" + "".join(f"{c:>18}" for c in cells))


def main():
    log("=== Retraction severity and exposure (BM25, MIRAGE) ===")
    n_reasons = check_against_parquet()
    log(f"Reason map check: {n_reasons} distinct reasons in retracted.parquet, all mapped, no extra keys")

    cls = build_classes()
    log(f"Wrote {CLASSES_PATH} ({len(cls):,} rows)")
    cls_rank = cls["reason_class"].map(CLASS_RANK)
    pmid_rank = dict(zip(cls["pmid"], cls_rank))
    pmid_group = dict(zip(cls["pmid"], cls["source_group"]))

    docs = pd.read_parquet(SUBSET_DOCS_PATH)
    subset_pmids = set(docs["pmid"].astype(int))
    subset_retracted = set(docs.loc[docs["is_retracted"].astype(bool), "pmid"].astype(int))
    assert len(subset_retracted) == EXPECTED_SUBSET_RETRACTED, len(subset_retracted)
    assert subset_retracted <= set(cls["pmid"])

    df = pd.read_parquet(FLAGGED_PATH)
    counts = question_counts()
    assert sum(counts.values()) == EXPECTED_QUESTIONS

    # ---- 1. sizes
    log("\n=== 1. Class and source-group sizes ===")
    ret_docs = df.loc[df["is_retracted"], ["docid", "pmid"]].drop_duplicates("docid")
    assert len(ret_docs) == EXPECTED_RETRIEVED_RETRACTED, len(ret_docs)
    size_table("All retracted PMIDs", cls)
    size_table("Retracted PMIDs in the corpus subset", cls[cls["pmid"].isin(subset_retracted)])
    # one row per distinct retracted doc (a PMID shared by two docids would count twice)
    retrieved = cls.set_index("pmid").loc[ret_docs["pmid"].astype(int)].reset_index()
    size_table("Distinct retracted docs retrieved (ranks 1-32)", retrieved)

    # ---- 2. direct exposure
    log("\n=== 2. Direct exposure by severity view and source group ===")
    log("Questions with >=1 retrieved retracted doc (is_retracted); denominator = all questions in group.")
    direct = df[df["is_retracted"]].copy()
    direct["rank_class"] = direct["pmid"].astype(int).map(pmid_rank)
    direct["group"] = direct["pmid"].astype(int).map(pmid_group)
    assert direct["rank_class"].notna().all() and direct["group"].notna().all()
    sets = {f"{v} ({'A' if v == 'strict' else 'A or B' if v == 'broad' else 'every paper'})":
            direct[direct["rank_class"] >= m] for v, m in VIEWS.items()}
    exposure_table("Direct, by severity view", sets, counts)
    sets = {g: direct[direct["group"] == g] for g in SOURCE_GROUPS}
    exposure_table("Direct, by source group", sets, counts)

    # ---- 3. one-hop strongest cited class
    log("\n=== 3. One-hop exposure by strongest cited class ===")
    notice_exclude = {int(x) for x in NOTICE_EXCLUDE_PATH.read_text().split()}
    pairs = pd.read_parquet(PAIRS_PATH)
    pairs = pairs[~pairs["citing_pmid"].isin(notice_exclude) & pairs["citing_pmid"].isin(subset_pmids)]
    pairs = pairs.assign(cited_rank=pairs["retracted_pmid"].map(pmid_rank))
    assert pairs["cited_rank"].notna().all(), "iCite pair cites a PMID missing from retracted.parquet"
    log(f"iCite pairs after dropping notice_exclude_pmids and restricting to subset docs: "
        f"{len(pairs):,} pairs, {pairs['citing_pmid'].nunique():,} citing PMIDs")
    strongest = pairs.groupby("citing_pmid")["cited_rank"].max().astype(int)

    one_hop = df[df["one_hop"]].copy()
    one_hop["strongest"] = one_hop["pmid"].astype(int).map(strongest)
    assert one_hop["strongest"].notna().all(), "one-hop row with no cited retracted pair"
    n_cited = pairs.groupby("citing_pmid")["retracted_pmid"].nunique()
    assert (one_hop["pmid"].astype(int).map(n_cited) == one_hop["n_retracted_cited"]).all(), \
        "n_retracted_cited disagrees with iCite pairs"

    hubs = rank_hubs(one_hop)
    big = set(hubs.loc[hubs["questions"] >= HUB_MIN_QUESTIONS, "docid"])
    assert len(big) == EXPECTED_HUBS, len(big)

    for name, excluded in [("all one-hop docs", set()),
                           (f"without the {len(big)} docs in >={HUB_MIN_QUESTIONS} questions", big)]:
        base = one_hop[~one_hop["docid"].isin(excluded)]
        sets = {f"{v} (strongest >= {RANK_CLASS[m] if m else 'any'})": base[base["strongest"] >= m]
                for v, m in VIEWS.items()}
        exposure_table(f"One-hop, {name}", sets, counts)

    # ---- 4. D/U-only docs and hubs
    log("\n=== 4. One-hop docs citing only class D or U retracted papers ===")
    corpus_oh = strongest[~strongest.index.isin(subset_retracted)]
    log(f"Corpus one-hop docs (cite >=1 retracted paper, not retracted, not notice): {len(corpus_oh):,}")
    retrieved_oh = one_hop.drop_duplicates("docid")
    for label, ranks in [("Corpus subset", corpus_oh), ("Retrieved (ranks 1-32), distinct docids",
                                                         retrieved_oh["strongest"])]:
        n = len(ranks)
        vc = ranks.value_counts()
        dist = "  ".join(f"{RANK_CLASS[r]} {int(vc.get(r, 0)):,}" for r in sorted(CLASS_RANK.values(), reverse=True))
        d_u = int((ranks <= CLASS_RANK["D"]).sum())
        log(f"{label}: {n:,} docs; strongest cited class: {dist}")
        log(f"  cite only class D or U: {d_u:,} ({d_u / n:.2%})")

    log(f"\nTop {TOP_HUBS} one-hop hub docs (by questions, ranks 1-32)")
    for i, h in hubs.head(TOP_HUBS).iterrows():
        s = RANK_CLASS[int(strongest[int(h["pmid"])])]
        log(f"{i + 1:>2}. {h['docid']}  pmid {h['pmid']}  questions {h['questions']}  "
            f"n_retracted_cited {h['n_retracted_cited']}  strongest cited class {s}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
