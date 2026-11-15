"""
Retraction reason inventory, stage 1 (read-only; no network).

Reads data/processed/retracted.parquet.

1. Every distinct value in the reasons lists with its count of papers (a paper counts once
   per reason), sorted descending, no cutoff. Also the number of retracted PMIDs with no
   reasons (tag-only).
2. The same reason counts per source group: in both sources, Retraction Watch only,
   PubMed tag only.
3. The 15 most common distinct combinations of reasons (order-independent: reasons are
   sorted within a paper; papers with no reasons are excluded here and reported separately).
4. Save printed output to results/reason_list.txt.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

RETRACTED_PATH = Path("data/processed/retracted.parquet")
RESULTS_PATH = Path("results/reason_list.txt")
TOP_COMBOS = 15

output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(str(msg))


def reason_set(reasons):
    """Distinct reasons of one paper (empty if missing)."""
    if reasons is None:
        return frozenset()
    return frozenset(reasons)


def print_reason_counts(sets, n_papers):
    counts = Counter(r for s in sets for r in s)
    n_empty = sum(1 for s in sets if not s)
    log(f"Papers: {n_papers:,}; with no reasons: {n_empty:,}; with >=1 reason: {n_papers - n_empty:,}")
    log(f"Distinct reason values: {len(counts):,}")
    for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        log(f"  {n:>7,}  {reason}")


def main():
    log("=== Retraction reasons (stage 1) ===")
    df = pd.read_parquet(RETRACTED_PATH)
    df["reason_set"] = df["reasons"].map(reason_set)
    log(f"Retracted PMIDs: {len(df):,}\n")

    log("=== 1. All reasons, count of papers ===")
    print_reason_counts(list(df["reason_set"]), len(df))
    n_tag_only_empty = int((df["reason_set"].map(len) == 0).sum())
    log(f"\nRetracted PMIDs with no reasons (tag-only): {n_tag_only_empty:,}")

    log("\n=== 2. Reasons by source group ===")
    groups = [
        ("In both sources (Retraction Watch and PubMed tag)", df["in_rw"] & df["in_pubmed_tag"]),
        ("Retraction Watch only", df["in_rw"] & ~df["in_pubmed_tag"]),
        ("PubMed tag only", ~df["in_rw"] & df["in_pubmed_tag"]),
    ]
    assert sum(int(m.sum()) for _, m in groups) == len(df), "source groups do not partition the table"
    for name, mask in groups:
        log(f"\n--- {name} ---")
        print_reason_counts(list(df.loc[mask, "reason_set"]), int(mask.sum()))

    log(f"\n=== 3. {TOP_COMBOS} most common reason combinations ===")
    combos = Counter(tuple(sorted(s)) for s in df["reason_set"] if s)
    log(f"Distinct non-empty combinations: {len(combos):,}")
    for combo, n in sorted(combos.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_COMBOS]:
        log(f"  {n:>7,}  {' | '.join(combo)}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
