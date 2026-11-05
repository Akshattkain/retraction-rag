# Retraction exposure in medical RAG

How often does a medical RAG system pull in retracted papers, or papers that cite them, and does filtering retracted papers out of the index fix it? This repo measures that on a PubMed subset using Retraction Watch and PubMed retraction labels. It's a pilot, so all numbers are preliminary.

## Background

Retracted papers keep getting cited, usually without a note. In a PubMed Central study of 7,813 retracted papers, only 722 of 13,252 post-retraction citation contexts (5.4%) acknowledged the retraction ([Hsiao & Schneider, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9520488/)). A 2025 stem-cell study found 14 of 334 (4.2%) ([BMC Medicine](https://doaj.org/article/1e794786dc2d48d29ab835d1fe8e2d28)). Chatbots show the same gap: ChatGPT-5.0 used 72 of 117 retracted stem-cell articles in its answers, and 30 of those 72 answers included a retraction notice ([Zhang et al., 2025](https://ideas.repec.org/a/spr/scient/v130y2025i11d10.1007_s11192-025-05484-y.html)).

The LLM-side work I found is small chatbot audits ([Yao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12583397/), [Labenbacher et al.](https://www.jmir.org/2026/1/e88766), [Thelwall](https://arxiv.org/pdf/2604.16872)). I haven't found one that measures this at the retrieval level in a standard medical RAG pipeline.

## What I'm measuring

- Direct exposure: a retrieved paper is itself retracted.
- One-hop exposure: a retrieved paper isn't retracted but cites a retracted one. This is a risk flag, not proof the citing paper is wrong. Retraction notices are excluded from the citing side.

Hypotheses:

1. Retrievers return retracted papers, and papers citing them, at a rate different from the base rate of the corpus they search.
2. An index-time retraction filter removes direct exposure but leaves one-hop exposure, which should be several times more common.
3. Removing a retracted source changes the answer in a measurable share of exposed questions.

If retrieval turns out to be neutral, exposure just tracks base rates, and that's a result too.

## Setup

- Corpus: a uniform ~5% sample of [MedRAG's PubMed corpus](https://arxiv.org/pdf/2402.13178) (every 20th baseline file, 58 files, 1,224,889 abstracts). The full corpus is 23.9M snippets from a Feb 2024 snapshot, about 70 GB of text, so I'm piloting on the sample and may scale up later.
- Questions: MIRAGE, 7,663 questions.
- Retriever: BM25 through Pyserini with default parameters, k = 32.
- Retraction labels: Retraction Watch (`RetractionNature == "Retraction"`) combined with PubMed's "Retracted Publication" tag.
- Retraction notices: PubMed's notice list combined with Retraction Watch's `RetractionPubMedID`. Notices that are not themselves retracted papers are excluded from the citing side.
- Citation links: iCite API (`cited_by`). iCite only has PubMed-to-PubMed links.
- Base rates are computed on the same subset the retriever searches.

## Layout

```
CONTEXT.md        rules and verified data facts, read by the coding agent before each task
README.md
requirements.txt
src/              one script per stage
results/          printed summary from each stage
data/             gitignored
third_party/      gitignored (MIRAGE, MedRAG code)
```

## Reproduce so far

Needs Python 3.12 and Java (tested on 23; Pyserini targets 21).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Retraction Watch (shallow clone; the full history is ~4 GB)
git clone --depth 1 https://gitlab.com/crossref/retraction-watch-data data/raw/retraction-watch-data

# PubMed retraction tag and retraction notices (needs NCBI EDirect)
esearch -db pubmed -query '"Retracted Publication"[pt]' | efetch -format uid > data/raw/pubmed_retracted_pmids.txt
esearch -db pubmed -query 'hasretractionof OR "retraction notice"[pt]' | efetch -format uid > data/raw/pubmed_retraction_notice_pmids.txt

# MIRAGE
git clone --depth 1 https://github.com/Teddy-XiongGZ/MIRAGE third_party/MIRAGE

# Corpus subset: every 20th baseline file of MedRAG/pubmed (~3.5 GB)
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
os.makedirs("data/raw/pubmed_subset", exist_ok=True)
for i in range(20, 1300, 20):
    try:
        hf_hub_download("MedRAG/pubmed", f"chunk/pubmed23n{i:04d}.jsonl",
                        repo_type="dataset", local_dir="data/raw/pubmed_subset")
    except Exception as e:
        print("skip", i, type(e).__name__)
EOF
```

Then, in order:

```bash
python src/build_labels.py
python src/scan_corpus.py
python src/check_mirage_gold.py
python -m pyserini.index.lucene --collection JsonCollection \
  --input data/raw/pubmed_subset/chunk --index data/processed/bm25_subset \
  --generator DefaultLuceneDocumentGenerator --threads 4 --storeRaw
python src/check_bm25_index.py
python src/fetch_icite_citing.py
```

Build the notice lists (the union, then the exclusion list: notices that are not retracted papers):

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/raw/retraction-watch-data/retraction_watch.csv", low_memory=False)
rw = pd.to_numeric(df["RetractionPubMedID"], errors="coerce")
rw = set(rw[rw > 0].astype(int))
pm = set(int(x) for x in open("data/raw/pubmed_retraction_notice_pmids.txt") if x.strip().isdigit())
union = rw | pm
open("data/processed/notice_pmids.txt", "w").write("\n".join(map(str, sorted(union))))
retracted = set(pd.read_parquet("data/processed/retracted.parquet")["pmid"])
excl = union - retracted
open("data/processed/notice_exclude_pmids.txt", "w").write("\n".join(map(str, sorted(excl))))
print(len(union), len(excl))
EOF
python src/build_one_hop_flags.py
```

## Results so far

- Labels: 37,201 retracted PMIDs. Retraction Watch has 30,546, the PubMed tag has 35,274, and 28,619 are in both. 21 also have a reinstatement record in Retraction Watch and are flagged.
- Corpus subset: 1,224,889 abstracts, 1,528 of them retracted (0.12%).
- MIRAGE gold evidence: 0 of PubMedQA's 500 gold papers are retracted, and 6 of BioASQ's 3,467. That's about what chance predicts at this base rate (expected ~4.3). Only ~5% of gold papers are in the subset (23 for PubMedQA, 194 for BioASQ).
- BM25 index: 1,224,889 documents, 1.5 GB, count matches the corpus scan. All 23 PubMedQA gold papers that are in the subset come back at rank 1. That's an easy test, so it checks the index and PMID mapping, not retrieval quality on real queries.
- Notice lists: 34,200 from PubMed and 31,668 unique from Retraction Watch, 38,142 combined (27,726 in both). NLM renamed "Retraction of Publication" to "Retraction Notice" in 2026, so the query uses the new name ([NLM Technical Bulletin](https://www.nlm.nih.gov/pubs/techbull/ma26/ma26_pubmed_update_MeSH_changes.html)). 1,493 of the combined list are also retracted papers (the notice reuses the paper's PMID), so the exclusion list drops those and keeps 36,649. In the subset there are 291 notices, 57 of them also retracted, leaving 234 to exclude.
- Citation links: iCite returned 37,135 of the 37,201 retracted PMIDs, with 735,647 (retracted, citing) pairs from 597,694 unique citing papers. The median retracted paper has 7 citing papers and the max is 4,217. After dropping notices, 719,914 pairs remain, and 27,491 of them have a citing paper in the subset.
- One-hop: 21,696 subset abstracts (1.77%) cite a retracted paper. 85.1% cite exactly one, 10.5% cite two, and 4.4% cite three or more. Without the notice exclusion, 73 more notice abstracts would have been flagged.
- Retracted papers citing retracted papers: 402 of the 1,528 retracted abstracts in the subset (26%) cite another retracted paper, and overall it's 9,688 of 37,201 (26.0%). For non-retracted abstracts it's 1.8%. This is an observation, not a finding. Retractions cluster by topic and in networks of papers that cite each other.
- Rough chance baselines: with 32 snippets per question and random retrieval, about 3.9% of questions would contain a retracted abstract and about 44% a one-hop abstract (arithmetic from the rates above). Retrieval isn't random, so these are reference points, not the test.

## Limitations

- The subset is ~5% of the corpus, so each question has less competition than in the full corpus.
- BM25 only, with no dense retriever comparison yet.
- Retraction dates and reasons exist only for papers in Retraction Watch (30,546 of 37,201, 82%). Papers that only carry the PubMed tag have neither.
- Labels are current but the corpus snapshot is from Feb 2024, so some papers were retracted after the index was built.
- Snippets are abstracts, not full text. 62 PMIDs have two snippet rows, and counts are per snippet.
- One-hop is a lower bound, since iCite only has PubMed-to-PubMed links. It's also a risk flag: a paper that cites a retracted one to criticize it still counts.
- The subset holds ~5% of MIRAGE's gold evidence, so RAG answers can't be scored against the benchmark's answer labels. The answer-change test compares answers with and without a retracted source instead.
- Only known retractions are flagged.
- Unverified lead: BioASQ question IDs look like MongoDB ObjectIds, whose first 8 hex digits decode to a timestamp. If that holds, 3 of the 5 dated retracted gold papers were already retracted when the question was written. Needs checking against BioASQ documentation and PubMed notice dates.

## To do

- [x] Repo, environment, data downloads
- [x] Retraction label table
- [x] Corpus subset scan and base rate
- [x] MIRAGE gold-evidence check
- [x] BM25 index and recall check
- [x] Retraction-notice lists and exclusion rule
- [x] Fetch citing papers from iCite for all retracted PMIDs
- [x] One-hop flags for the corpus subset
- [ ] Fetch publication years to split citations before and after retraction
- [ ] Run BM25 (k = 32) on all 7,663 MIRAGE questions
- [ ] Measure direct and one-hop exposure against subset base rates
- [ ] Filter ablation (index-time vs query-time)
- [ ] Answer-change test (design to be decided)
- [ ] Sensitivity check excluding reinstated papers
- [ ] Decide whether to run on the full corpus

## Updates

- Set up the repo and environment (Python 3.12, Pyserini 2.4.0).
- Downloaded Retraction Watch (shallow clone, 79 MB), the PubMed retraction tag PMIDs (35,274), MIRAGE, and the 5% corpus subset (3.5 GB).
- Built the retraction label table (37,201 retracted PMIDs) and added a reinstatement flag.
- Scanned the corpus subset: 1,528 retracted abstracts (0.12%).
- Checked MIRAGE's gold-evidence papers against the labels.
- Built the BM25 index and checked recall on PubMedQA.
- Downloaded retraction-notice lists from PubMed (34,200) and Retraction Watch (31,668 unique), and combined them (38,142). 1,493 are also retracted papers, so the exclusion list keeps 36,649.
- Fetched citing papers from iCite: 37,135 of 37,201 retracted PMIDs returned, 735,647 (retracted, citing) pairs, 597,694 unique citing papers.
- Built one-hop flags: 21,696 of 1,224,889 subset abstracts (1.77%) cite a retracted paper, 85% of them exactly one. 402 of 1,528 retracted abstracts (26%) cite another retracted paper.

## References

- Hsiao & Schneider (2021), Continued use of retracted papers, *Quantitative Science Studies* 2(4): <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9520488/>
- BMC Medicine stem-cell study (DOI 10.1186/s12916-025-03965-8): <https://doaj.org/article/1e794786dc2d48d29ab835d1fe8e2d28>
- Zhang et al. (2025), *Scientometrics* 130(11): <https://ideas.repec.org/a/spr/scient/v130y2025i11d10.1007_s11192-025-05484-y.html>
- Yao et al. (2025), *Naturwissenschaften* 112:85: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12583397/>
- Labenbacher et al. (2026), *JMIR* 28:e88766: <https://www.jmir.org/2026/1/e88766>
- Thelwall (2026): <https://arxiv.org/pdf/2604.16872>
- Xiong et al. (2024), MedRAG and MIRAGE: <https://arxiv.org/pdf/2402.13178>
- Retraction Watch data via Crossref: <https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/>
- iCite API and bulk data: <https://support.icite.nih.gov/hc/en-us/articles/9513563045787-Bulk-Data-and-API>