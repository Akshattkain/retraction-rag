"""
Figures for the retracted-paper exposure analysis (read-only; no model, no network).
Run from the project root:   python src/make_figures.py
Reads : data/processed/retrieval_bm25_flagged.parquet, data/processed/subset_flags.parquet,
        data/processed/retracted_classes.parquet, data/processed/icite_citing_pairs.parquet,
        third_party/MIRAGE/benchmark.json, and results/filter_ablation.csv if it exists.
Writes: figures/f1..f8 *.png
Figures:
  f1  questions with a retracted / citing paper by number of results read, vs random retrieval
  f2  the same at top 32, by question set
  f3  share of retrieved results and distinct papers vs the corpus base rate
  f4  one-hop papers retrieved for the most questions
  f5  one-hop exposure with the most repeated papers removed
  f6  share of retrieved results by rank bucket vs the corpus rate
  f7  effect of the index-time retraction filter on direct and one-hop exposure
  f8  exposure by severity: one-hop (all), one-hop (class A only) and direct, top 5 and top 32
Every figure title states its finding; a check at the end compares the numbers with the ones already
verified in results/*.txt and prints MATCH or MISMATCH.
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FLAGGED = Path("data/processed/retrieval_bm25_flagged.parquet")
FLAGS = Path("data/processed/subset_flags.parquet")
BENCH = Path("third_party/MIRAGE/benchmark.json")
ABL = Path("results/filter_ablation.csv")
CLASSES = Path("data/processed/retracted_classes.parquet")
PAIRS = Path("data/processed/icite_citing_pairs.parquet")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

GREY, HI, INK, SOFT = "#8C8772", "#B04E2C", "#3D3929", "#73705F"
TEAL = "#2F6B7A"
KS = (1, 3, 5, 10, 32)
NAMES = {"bioasq": "BioASQ", "medmcqa": "MedMCQA", "pubmedqa": "PubMedQA", "mmlu": "MMLU", "medqa": "MedQA"}
EXPECT_RET = {1: 11, 3: 20, 5: 35, 10: 81, 32: 262}
EXPECT_OH = {1: 200, 3: 459, 5: 714, 10: 1144, 32: 2404}
EXPECT_DS = {"bioasq": 57, "medmcqa": 153, "pubmedqa": 18, "mmlu": 26, "medqa": 8}
EXPECT_OH_ALL = {5: 714, 32: 2404}
EXPECT_OH_STRICT = {5: 272, 32: 1296}
EXPECT_DIRECT = {5: 35, 32: 262}
PILOT = "Pilot: 5% sample of PubMed, keyword search (BM25), 7,663 MIRAGE questions"

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def finish(fig, title, name, note=PILOT):
    title = textwrap.fill(title, 64)
    two = "\n" in title
    fig.suptitle(title, fontsize=22, x=0.02, ha="left", y=0.985, va="top")
    fig.text(0.02, 0.012, textwrap.fill(note, 150), fontsize=11, color=SOFT, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.88 if two else 0.93))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("wrote", FIG / name)

def bar_labels(ax, bars, fmt, inside=False):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="top" if inside else "bottom", fontsize=13, color="white" if inside else INK,
                    xytext=(0, -6 if inside else 3), textcoords="offset points", zorder=6,
                    bbox=None if inside else dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

def contrast_vs_white(hex_color):
    """WCAG contrast ratio of a #RRGGBB colour against a white background."""
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 1.05 / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] + 0.05)

def main():
    df = pd.read_parquet(FLAGGED, columns=["qkey", "dataset", "rank", "docid", "pmid", "is_retracted", "one_hop"])
    fl = pd.read_parquet(FLAGS, columns=["is_retracted", "one_hop"])
    counts = {ds: len(items) for ds, items in json.load(open(BENCH)).items()}
    N = sum(counts.values())
    p = {"is_retracted": fl["is_retracted"].mean(), "one_hop": fl["one_hop"].mean()}

    def qn(sub, flag, k):                      # number of questions with >=1 hit inside the top k
        return sub.loc[(sub["rank"] <= k) & sub[flag], "qkey"].nunique()

    # ---- F1: exposure by depth, against what random retrieval would give
    obs = {f: [100 * qn(df, f, k) / N for k in KS] for f in ("is_retracted", "one_hop")}
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    for ax, f, label in zip(axes, ("is_retracted", "one_hop"), ("a retracted paper", "a paper citing retracted work")):
        bars = ax.bar(range(len(KS)), obs[f], color=[GREY] * (len(KS) - 1) + [HI], width=0.62)
        chance = [100 * (1 - (1 - p[f]) ** k) for k in KS]
        ax.plot(range(len(KS)), chance, "o--", color=INK, lw=2, label="if results were random", zorder=3)
        bar_labels(ax, bars, lambda v: f"{v:.2f}%" if f == "is_retracted" else f"{v:.1f}%")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"top {k}" for k in KS])
        ax.set_ylabel("% of questions"); ax.set_title(f"Questions retrieving\n{label}", fontsize=15, loc="left")
        ax.legend(frameon=False, loc="upper left")
    finish(fig, "How exposure grows with the number of results read, against random retrieval", "f1_exposure_by_depth.png")

    # ---- F2: by question set (top 32)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    top_ds = []
    for ax, f, label in zip(axes, ("is_retracted", "one_hop"), ("a retracted paper", "a paper citing retracted work")):
        vals = {ds: 100 * qn(df[df.dataset == ds], f, 32) / counts[ds] for ds in counts}
        top_ds.append(max(vals, key=vals.get))
        order = sorted(vals, key=vals.get)
        ax.barh([f"{NAMES[d]} ({counts[d]:,})" for d in order], [vals[d] for d in order],
                color=[GREY] * (len(order) - 1) + [HI], height=0.6)
        for i, d in enumerate(order):
            ax.text(vals[d], i, f"  {vals[d]:.2f}%" if f == "is_retracted" else f"  {vals[d]:.1f}%", va="center", fontsize=13)
        ax.set_xlim(0, max(vals.values()) * 1.2); ax.set_xlabel("% of questions (top 32 results)")
        ax.set_title(f"Questions retrieving\n{label}", fontsize=15, loc="left")
    finish(fig, "BioASQ is highest on both measures; the other question sets are lower" if top_ds == ["bioasq", "bioasq"]
           else "Exposure by question set (top 32 results)", "f2_by_question_set.png")

    # ---- F3: share of results vs corpus base rate
    distinct = df.drop_duplicates("docid")
    series = {"corpus (what chance gives)": [100 * p["is_retracted"], 100 * p["one_hop"]],
              "all retrieved results": [100 * df["is_retracted"].mean(), 100 * df["one_hop"].mean()],
              "distinct retrieved papers": [100 * distinct["is_retracted"].mean(), 100 * distinct["one_hop"].mean()]}
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    for i, (ax, label, dec) in enumerate(zip(axes, ("Retracted papers", "Papers citing a retracted paper"), (3, 2))):
        vals = [series[s][i] for s in series]
        bars = ax.bar(range(3), vals, color=[SOFT, HI, GREY], width=0.6)
        bar_labels(ax, bars, lambda v, d=dec: f"{v:.{d}f}%")
        ax.set_xticks(range(3)); ax.set_xticklabels(["corpus", "retrieved\nresults", "distinct\nretrieved"])
        ax.set_ylabel("share (%)"); ax.set_title(label, fontsize=16, loc="left")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.text(0.5, 0.97, f"retrieved results / corpus = {vals[1] / vals[0]:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=14, color=SOFT)
    finish(fig, "Search returns retracted papers at about their share of the corpus", "f3_share_vs_base_rate.png")

    # ---- F4: hub papers among one-hop results
    oh = df[df["one_hop"]]
    hub = oh.groupby("docid").agg(questions=("qkey", "nunique"), rows=("qkey", "size"), pmid=("pmid", "first"))
    hub = hub.sort_values(["questions", "rows"], ascending=False)
    top = hub.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.barh([f"PMID {int(x)}" for x in top["pmid"]], top["questions"], color=[GREY] * (len(top) - 1) + [HI], height=0.65)
    for i, v in enumerate(top["questions"]):
        ax.text(v, i, f"  {int(v)}", va="center", fontsize=13)
    cover10 = 100 * hub["rows"].head(10).sum() / hub["rows"].sum()
    ax.set_xlabel("number of questions the paper was retrieved for (top 32)")
    ax.text(0.98, 0.1, f"Top 10 papers cover {cover10:.1f}% of all one-hop results\n{len(hub):,} distinct papers in total",
            transform=ax.transAxes, ha="right", fontsize=15, color=SOFT, bbox=dict(facecolor="white", edgecolor="none"))
    finish(fig, f"One paper is retrieved for {int(hub['questions'].iloc[0]):,} of {N:,} questions", "f4_hub_papers.png")

    # ---- F5: removing repeat papers (dropped without back-filling, so these are lower bounds)
    top1 = {hub.index[0]}
    many = set(hub.index[hub["questions"] >= 5])
    def exposure(excl, k, sub, n):
        s = sub[(sub["rank"] <= k) & sub["one_hop"] & ~sub["docid"].isin(excl)]
        return 100 * s["qkey"].nunique() / n
    groups = [("All questions\ntop 5", df, N, 5), ("All questions\ntop 32", df, N, 32),
              ("MedQA only\ntop 5", df[df.dataset == "medqa"], counts["medqa"], 5)]
    scen = [("all papers", set()), ("without the top paper", top1), (f"without papers seen in 5+ questions ({len(many)})", many)]
    fig, ax = plt.subplots(figsize=(13.33, 7.5)); w = 0.26
    for j, (lab, excl) in enumerate(scen):
        vals = [exposure(excl, k, sub, n) for _, sub, n, k in groups]
        bars = ax.bar([i + (j - 1) * w for i in range(3)], vals, width=w, color=[GREY, "#BDB8A6", HI][j], label=lab)
        bar_labels(ax, bars, lambda v: f"{v:.1f}%")
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("% of questions with a one-hop paper"); ax.legend(frameon=False, loc="upper left", fontsize=13)
    rel = [1 - exposure(many, k, sub, n) / max(exposure(set(), k, sub, n), 1e-9) for _, sub, n, k in groups]
    claim = rel[2] == max(rel) and rel[0] > 0.2
    finish(fig, "Take out a few repeat papers and one-hop exposure drops a lot, most of all for MedQA" if claim
           else "One-hop exposure with and without the most repeated papers", "f5_repeat_papers_removed.png",
           note=PILOT + ". Removed papers are not replaced, so these are lower bounds.")

    # ---- F6: where in the ranking
    bins = [(1, 5, "ranks 1-5"), (6, 16, "ranks 6-16"), (17, 32, "ranks 17-32")]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    pattern = []
    for ax, f, label, dec in zip(axes, ("is_retracted", "one_hop"), ("Retracted papers", "Papers citing a retracted paper"), (3, 2)):
        vals = [100 * df.loc[(df["rank"] >= a) & (df["rank"] <= b), f].mean() for a, b, _ in bins]
        pattern.append(vals[0] < 100 * p[f] if f == "is_retracted" else vals[0] > 100 * p[f])
        bars = ax.bar(range(3), vals, color=[HI, GREY, GREY], width=0.6)
        ax.axhline(100 * p[f], color=INK, ls="--", lw=2, label="corpus rate"); ax.legend(frameon=False, loc="upper right", fontsize=13)
        ax.set_ylim(0, max(max(vals), 100 * p[f]) * 1.3)
        bar_labels(ax, bars, lambda v, d=dec: f"{v:.{d}f}%", inside=True)
        ax.set_xticks(range(3)); ax.set_xticklabels([b[2] for b in bins]); ax.set_ylabel("share of retrieved results (%)")
        ax.set_title(label, fontsize=16, loc="left")
    finish(fig, "Retracted papers are not concentrated at the top ranks; one-hop papers are, slightly" if all(pattern)
           else "Share of retrieved results by rank, against the corpus rate", "f6_by_rank.png")

    # ---- F7: filter ablation (only if filter_ablation.py has been run)
    if ABL.exists():
        a = pd.read_csv(ABL); a = a[a.scope == "overall"]
        order = ["baseline", "drop_retracted", "drop_retracted_cites"]
        labels = ["no filter", "drop retracted\npapers", "drop retracted +\npapers citing them"]
        fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
        for ax, k in zip(axes, (5, 32)):
            sub = a[a.k == k].set_index("variant").loc[order]
            for j, (col, lab, colr) in enumerate((("retracted_pct", "retracted paper", HI), ("one_hop_pct", "paper citing a retracted one", GREY))):
                bars = ax.bar([i + (j - 0.5) * 0.36 for i in range(3)], sub[col], width=0.36, color=colr, label=lab)
                bar_labels(ax, bars, lambda v: f"{v:.2f}%" if v < 1 else f"{v:.1f}%")
            ax.set_ylim(0, float(sub[["retracted_pct", "one_hop_pct"]].values.max()) * 1.55)
            ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=13); ax.set_ylabel("% of questions")
            ax.set_title(f"Top {k} results", fontsize=16, loc="left"); ax.legend(frameon=False, fontsize=13, loc="upper right")
        ch = a[(a.k == 5) & (a.variant == "drop_retracted_cites")]["top5_changed_pct"].iloc[0]
        b0 = a.set_index(["variant", "k"])
        unchanged = all(b0.loc[("drop_retracted", k), "retracted_q"] == 0 and
                        abs(b0.loc[("drop_retracted", k), "one_hop_q"] - b0.loc[("baseline", k), "one_hop_q"]) <= 0.005 * b0.loc[("baseline", k), "one_hop_q"]
                        for k in (5, 32))
        finish(fig, "Dropping retracted papers removes direct exposure and leaves one-hop exposure unchanged" if unchanged
               else "What filtering retracted papers out of the results changes", "f7_filter_ablation.png",
               note=PILOT + f". Filtered papers are replaced by the next-ranked ones. The strict filter changes the top 5 for {ch:.1f}% of questions.")

    # ---- F8: exposure by severity of the cited retracted paper
    # strict = the retracted paper's reason class is A (see src/reason_map.py); a one-hop paper counts as
    # strict if it cites at least one class-A retracted paper.
    cls = pd.read_parquet(CLASSES, columns=["pmid", "reason_class"])
    class_a = set(cls.loc[cls["reason_class"] == "A", "pmid"].astype(int))
    oh_pmids = set(df.loc[df["one_hop"], "pmid"].astype(int))
    pairs = pd.read_parquet(PAIRS, columns=["citing_pmid", "retracted_pmid"])
    pairs = pairs[pairs["citing_pmid"].isin(oh_pmids)]
    strict_citers = set(pairs.loc[pairs["retracted_pmid"].isin(class_a), "citing_pmid"].astype(int))
    df["one_hop_strict"] = df["one_hop"] & df["pmid"].astype(int).isin(strict_citers)
    k8 = (5, 32)
    a5, a32 = (100 * qn(df, "one_hop_strict", k) / qn(df, "one_hop", k) for k in k8)
    series8 = [("one-hop (all)", "one_hop", SOFT), ("one-hop (strict, class A only)", "one_hop_strict", HI),
               ("direct (all)", "is_retracted", TEAL)]
    assert all(contrast_vs_white(c) >= 4.5 for _, _, c in series8)
    fig, ax = plt.subplots(figsize=(13.33, 7.5)); w = 0.26
    for j, (lab, flag, colr) in enumerate(series8):
        vals = [100 * qn(df, flag, k) / N for k in k8]
        bars = ax.bar([i + (j - 1) * w for i in range(len(k8))], vals, width=w, color=colr, label=lab)
        bar_labels(ax, bars, lambda v: f"{v:.2f}%" if v < 1 else f"{v:.1f}%")
    ax.set_xticks(range(len(k8))); ax.set_xticklabels([f"top {k}" for k in k8])
    ax.set_ylabel(f"% of {N:,} questions"); ax.legend(frameon=False, loc="upper left", fontsize=13)
    finish(fig, f"Class A papers account for {a5:.0f}% of one-hop exposure at top 5 and {a32:.0f}% at top 32",
           "f8_severity.png",
           note=PILOT + ". Class A = at least one Retraction Watch reason about data, results, images, methods or "
                "materials (see src/reason_map.py). Papers with no listed reason are not counted as class A, so strict "
                "is a lower bound.")

    # ---- numbers check
    log = []
    for k in KS:
        for name, got, exp in (("retracted", qn(df, "is_retracted", k), EXPECT_RET[k]), ("one-hop", qn(df, "one_hop", k), EXPECT_OH[k])):
            log.append((f"{name} questions, top {k}", got, exp))
    for ds in counts:
        log.append((f"retracted questions, {NAMES[ds]} top 32", qn(df[df.dataset == ds], "is_retracted", 32), EXPECT_DS[ds]))
    log.append(("rows", len(df), 245121)); log.append(("top paper's questions", int(hub["questions"].iloc[0]), 521))
    for k in k8:
        log.append((f"one-hop (all) questions, top {k}", qn(df, "one_hop", k), EXPECT_OH_ALL[k]))
        log.append((f"one-hop (strict) questions, top {k}", qn(df, "one_hop_strict", k), EXPECT_OH_STRICT[k]))
        log.append((f"direct (all) questions, top {k}", qn(df, "is_retracted", k), EXPECT_DIRECT[k]))
    print("\n--- check against numbers already verified in results/*.txt ---")
    bad = 0
    for name, got, exp in log:
        ok = got == exp; bad += not ok
        print(f"{name:<38} got {got:>7,}  expected {exp:>7,}  {'MATCH' if ok else 'MISMATCH'}")
    print("ALL MATCH" if not bad else f"{bad} MISMATCH(ES): numbers differ from results/*.txt; investigate before using the figures")

if __name__ == "__main__":
    main()
