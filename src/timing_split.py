"""
Timing split for one-hop exposure: does the retrieved doc cite a retracted paper before, in the same
year as, or after the retraction? (read-only apart from the output; no network).

Reads data/processed/icite_citing_pairs.parquet, retracted.parquet, citing_years.parquet,
retrieval_bm25_flagged.parquet, notice_exclude_pmids.txt, retracted_classes.parquet
(question denominators from benchmark.json via src/extra_checks.py).

1. Pairs: drop notice_exclude_pmids from citing_pmid, keep citing PMIDs present in citing_years (the
   retrieved one-hop docs). Join the citing year and the retraction year (retraction_date, dated papers
   only). Label: before (citing year < retraction year), same_year (=), after (>). Pairs whose retracted
   paper has no date are "undated" (tag-only) and are excluded from the split.
2. Pair counts per label; one-hop docs with at least one undated pair.
3. Doc level, dated pairs only: any_after (>=1 pair after), all_before (every dated pair before),
   same_year_only_mixed (the rest: no pair after, at least one same_year). all_undated = no dated pair.
   Check: any_after + all_before + same_year_only_mixed = docs with >=1 dated pair.
4. Question level (denominator 7,663, overall and per dataset), k=5 and k=32: questions with >=1 one-hop
   doc in each doc category; questions whose one-hop docs within k are ALL undated. Without and with the
   docs that appear in >=5 questions removed (dropped, no back-fill).
5. The same doc-level and question-level split restricted to class A cited papers (retracted_classes).
   A doc with no class-A pair drops out of that analysis. The removed hub docs are the same ones as above.
6. Pair labels of hub doc 25032021.
7. Output saved to results/timing_split.txt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.extra_checks import question_counts
from src.hub_inspect import rank_hubs

PAIRS_PATH = Path("data/processed/icite_citing_pairs.parquet")
RETRACTED_PATH = Path("data/processed/retracted.parquet")
YEARS_PATH = Path("data/processed/citing_years.parquet")
FLAGGED_PATH = Path("data/processed/retrieval_bm25_flagged.parquet")
NOTICE_EXCLUDE_PATH = Path("data/processed/notice_exclude_pmids.txt")
CLASSES_PATH = Path("data/processed/retracted_classes.parquet")
RESULTS_PATH = Path("results/timing_split.txt")

KS = [5, 32]
HUB_MIN_QUESTIONS = 5
EXPECTED_QUESTIONS = 7663
EXPECTED_DOCS = 2296
EXPECTED_HUBS = 23
HUB_DOC_PMID = 25032021
PAIR_LABELS = ["before", "same_year", "after", "undated"]
DOC_CATS = ["any_after", "all_before", "same_year_only_mixed", "all_undated"]

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def label_pairs(pairs):
    cy, ry = pairs["citing_year"], pairs["retraction_year"]
    lab = np.where(ry.isna(), "undated",
                   np.where(cy < ry, "before", np.where(cy == ry, "same_year", "after")))
    return pairs.assign(label=lab)


def doc_categories(pairs):
    """Series: citing_pmid -> doc category, from the labelled pairs."""
    g = pairs.assign(dated=pairs["label"] != "undated", is_after=pairs["label"] == "after",
                     is_before=pairs["label"] == "before").groupby("citing_pmid")
    n_dated, n_after, n_before = g["dated"].sum(), g["is_after"].sum(), g["is_before"].sum()
    cat = pd.Series("same_year_only_mixed", index=n_dated.index)
    cat[n_before == n_dated] = "all_before"
    cat[n_after > 0] = "any_after"
    cat[n_dated == 0] = "all_undated"
    return cat


def question_section(rows, counts, scenarios):
    """rows: one-hop retrieval rows with a 'cat' column (docs outside the analysis already dropped)."""
    groups = [("overall", EXPECTED_QUESTIONS, None)] + [(ds, n, ds) for ds, n in counts.items()]
    for k in KS:
        for name, excluded in scenarios:
            sub = rows[(rows["rank"] <= k) & ~rows["docid"].isin(excluded)]
            log(f"\nk = {k}, {name}: share (count) of questions")
            log(f"  {'':<40}" + "".join(f"{g[0]:>18}" for g in groups))
            lines = [("any one-hop doc", sub)] + [(f"has a doc: {c}", sub[sub["cat"] == c]) for c in DOC_CATS]
            for label, s in lines:
                cells = []
                for _, n_q, ds in groups:
                    t = s if ds is None else s[s["dataset"] == ds]
                    n = t["qkey"].nunique()
                    cells.append(f"{n / n_q:.2%} ({n:,})")
                log(f"  {label:<40}" + "".join(f"{c:>18}" for c in cells))
            # questions whose one-hop docs within k are all undated
            per_q = sub.assign(u=sub["cat"] == "all_undated").groupby("qkey").agg(
                u=("u", "all"), dataset=("dataset", "first"))
            cells = []
            for _, n_q, ds in groups:
                t = per_q if ds is None else per_q[per_q["dataset"] == ds]
                n = int(t["u"].sum())
                cells.append(f"{n / n_q:.2%} ({n:,})")
            log(f"  {'all one-hop docs undated':<40}" + "".join(f"{c:>18}" for c in cells))


def run_split(title, pairs, one_hop, counts, scenarios):
    log(f"\n{'=' * 8} {title} {'=' * 8}")
    log("\n--- Pair counts per label ---")
    vc = pairs["label"].value_counts()
    log(f"Pairs: {len(pairs):,}; retrieved one-hop docs with >=1 pair: {pairs['citing_pmid'].nunique():,}")
    for lab in PAIR_LABELS:
        log(f"  {lab:<10} {int(vc.get(lab, 0)):>8,}")
    n_undated_docs = pairs.loc[pairs["label"] == "undated", "citing_pmid"].nunique()
    log(f"One-hop docs with >=1 undated pair: {n_undated_docs:,}")

    cat = doc_categories(pairs)
    log("\n--- Doc level (dated pairs only) ---")
    for c in DOC_CATS:
        log(f"  {c:<22} {int((cat == c).sum()):>8,}")
    n_with_dated = pairs.loc[pairs["label"] != "undated", "citing_pmid"].nunique()
    parts = int(cat.isin(DOC_CATS[:3]).sum())
    ok = parts == n_with_dated
    log(f"  any_after + all_before + same_year_only_mixed = {parts:,}; docs with >=1 dated pair = "
        f"{n_with_dated:,}  {'MATCH' if ok else 'MISMATCH'}")
    assert ok

    rows = one_hop[one_hop["pmid"].isin(cat.index)].assign(cat=lambda d: d["pmid"].map(cat))
    log("\n--- Question level ---")
    question_section(rows, counts, scenarios)
    return cat


def main():
    log("=== Timing split for one-hop exposure (BM25, MIRAGE) ===")
    counts = question_counts()
    assert sum(counts.values()) == EXPECTED_QUESTIONS

    df = pd.read_parquet(FLAGGED_PATH)
    one_hop = df[df["one_hop"]].copy()
    one_hop["pmid"] = one_hop["pmid"].astype("int64")
    assert one_hop["docid"].nunique() == EXPECTED_DOCS, one_hop["docid"].nunique()

    hubs = rank_hubs(one_hop)
    big = set(hubs.loc[hubs["questions"] >= HUB_MIN_QUESTIONS, "docid"])
    assert len(big) == EXPECTED_HUBS, len(big)
    scenarios = [("no docs removed", set()),
                 (f"{len(big)} docs in >={HUB_MIN_QUESTIONS} questions removed (no back-fill)", big)]

    # ---- 1. build and label pairs
    notice_exclude = {int(x) for x in NOTICE_EXCLUDE_PATH.read_text().split()}
    years = pd.read_parquet(YEARS_PATH)
    assert years["pmid"].is_unique and years["year"].notna().all()
    assert set(years["pmid"]) == set(one_hop["pmid"]), "citing_years does not match the retrieved one-hop docs"
    pairs = pd.read_parquet(PAIRS_PATH)
    n0 = len(pairs)
    pairs = pairs[~pairs["citing_pmid"].isin(notice_exclude)]
    n1 = len(pairs)
    pairs = pairs[pairs["citing_pmid"].isin(set(years["pmid"]))]
    log(f"Pairs: {n0:,} total; {n1:,} after dropping notice_exclude citing PMIDs; "
        f"{len(pairs):,} with a citing year (retrieved one-hop docs: {pairs['citing_pmid'].nunique():,})")
    assert pairs["citing_pmid"].nunique() == EXPECTED_DOCS

    ret = pd.read_parquet(RETRACTED_PATH, columns=["pmid", "retraction_date"]).set_index("pmid")
    cls = pd.read_parquet(CLASSES_PATH, columns=["pmid", "reason_class"]).set_index("pmid")
    assert pairs["retracted_pmid"].isin(ret.index).all() and pairs["retracted_pmid"].isin(cls.index).all()
    pairs = pairs.assign(
        citing_year=pairs["citing_pmid"].map(years.set_index("pmid")["year"]).astype(float),
        retraction_year=pairs["retracted_pmid"].map(ret["retraction_date"].dt.year.astype("float")),
        reason_class=pairs["retracted_pmid"].map(cls["reason_class"]))
    pairs = label_pairs(pairs)

    # ---- 2-4. all cited retracted papers
    run_split("All cited retracted papers", pairs, one_hop, counts, scenarios)

    # ---- 5. class A cited papers only
    run_split("Class A cited papers only", pairs[pairs["reason_class"] == "A"], one_hop, counts, scenarios)

    # ---- 6. hub doc
    log(f"\n{'=' * 8} Hub doc {HUB_DOC_PMID}: pair labels {'=' * 8}")
    hub_pairs = pairs[pairs["citing_pmid"] == HUB_DOC_PMID]
    assert len(hub_pairs) > 0
    docids = sorted(one_hop.loc[one_hop["pmid"] == HUB_DOC_PMID, "docid"].unique())
    log(f"docid(s): {docids}; pairs: {len(hub_pairs)}")
    log(f"  {'retracted_pmid':<15}{'class':<7}{'citing year':>12}{'retraction year':>17}  label")
    for _, r in hub_pairs.iterrows():
        ry = "none" if pd.isna(r["retraction_year"]) else f"{int(r['retraction_year'])}"
        log(f"  {r['retracted_pmid']:<15}{r['reason_class']:<7}{int(r['citing_year']):>12}{ry:>17}  {r['label']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
