# Retracted-paper exposure in medical RAG

Goal: measure how often a standard medical RAG pipeline retrieves retracted papers
(direct) or non-retracted papers that cite retracted ones (one-hop), and whether an
index-time retraction filter leaves one-hop exposure in place.

## Rules
- Real data only. No synthetic data, no manual labeling.
- Log every data source version/download date and every random seed.
- Retrieval first. An LLM is only needed later for the answer-change test.
- Don't use OpenAlex's is_retracted flag (known data-quality problem).
- Never derive PMIDs from DOIs by guessing.
- Never initialize MedRAG with the full PubMed corpus (it auto-downloads 70 GB).

## Definitions
- direct exposure: a retrieved PMID is retracted
- one-hop exposure: a retrieved PMID is not retracted but cites a retracted PMID
- also record whether the citation predates the retraction date

## Hypotheses
1. Retrievers return retracted papers, and papers citing them, at a rate different
   from base rate (computed on the same corpus subset).
2. An index-time filter removes direct exposure but leaves one-hop exposure.
3. Removing a retracted source changes the answer in a measurable share of exposed
   questions.

## Plan
labels -> BM25 index -> iCite one-hop table -> retrieval over MIRAGE (7,663 questions)
-> exposure analysis -> filter ablation -> answer-change test

## Actual setup (pilot)
- Corpus: uniform ~5% subset of MedRAG PubMed: every 20th baseline file, 58 files in
  data/raw/pubmed_subset/chunk/ (3.5 GB). Compute ALL base rates on this subset.
- Retrieval: BM25 only (Pyserini, default parameters). MedCPT is out of scope locally.
- Retraction Watch: data/raw/retraction-watch-data/retraction_watch.csv (shallow clone).
  Header has a trailing comma. Dates look like "3/25/2026 0:00".
  OriginalPaperPubMedID of 0 or blank means missing. OriginalPaperDOI can be
  blank/"unavailable". Reason is a semicolon-separated list. Keep RetractionNature
  == "Retraction" only.
- PubMed tag list: data/raw/pubmed_retracted_pmids.txt (35,274 PMIDs)
- Questions: third_party/MIRAGE/benchmark.json
- Citation links: iCite API only (no bulk download), 200 PMIDs per request, fields
  cited_by and references.
- Retraction labels = Retraction Watch UNION PubMed "Retracted Publication" tag.

## Verified data facts
- Corpus records: {id, title, content, contents, PMID}; PMID is a string. Cast to int.
- MIRAGE benchmark.json: dict of medqa(1273), medmcqa(4183), pubmedqa(500),
  bioasq(618), mmlu(1089). pubmedqa and bioasq items have a "PMID" list of gold
  evidence papers.
- Pyserini BM25 indexing of one chunk file worked with: python -m pyserini.index.lucene
  --collection JsonCollection --input <dir> --index <idx>
  --generator DefaultLuceneDocumentGenerator --threads 2 --storeRaw
  (Java 23, Python 3.12, Pyserini 2.4.0)

## Label Table Summary
- `data/processed/retracted.parquet`: 37,201 unique retracted PMIDs (union of RW Retractions and PubMed tags).
  Columns: pmid (int64), doi, retraction_date, reasons, in_rw (bool), in_pubmed_tag (bool), has_reinstatement (bool).
- `data/processed/retracted_doi_only.parquet`: 32,260 records from Retraction Watch with DOI but no valid PMID.
- Overlap / counts:
  - RW rows total: 73,444
  - RetractionNature == "Retraction": 68,002
  - RW rows with valid PMID: 30,570 (30,546 unique PMIDs)
  - PubMed retracted tag list: 35,274 unique PMIDs
  - Overlap: 28,619 PMIDs
  - Union size: 37,201 PMIDs
  - RW-only: 1,927 PMIDs
  - PubMed-tag only: 6,655 PMIDs
  - Retracted PMIDs with has_reinstatement: 21
  - Tag-only PMIDs in Retraction Watch under other RetractionNature: 62 unique PMIDs (343 rows: 26 Expression of concern, 19 Reinstatement, 17 Correction)

## BM25 Index & Retrieval Helper
- Index path: `data/processed/bm25_subset`
  - Built over `data/raw/pubmed_subset/chunk` (58 files, 1,224,889 documents) using Lucene JsonCollection with raw document storage.
  - Document count: 1,224,889 (matches corpus scan total exactly; 0 unindexable, 0 errors).
  - Index size on disk: ~1.5 GB.
- Search helper: `src/bm25_search.py`
  - Function: `search(query: str, k: int = 32) -> list[dict]`
  - Returns list of `{docid: str, pmid: int, score: float, rank: int}` with 1-indexed ranks.
  - Maps `docid` to integer `pmid` via cached `data/processed/subset_docs.parquet` (asserts `id` is unique).
  - Smoke test: Top hit for "eosinophilic meningitis Angiostrongylus" is PMID 579615 (score 17.6820).
  - Recall test: 23 / 23 (100%) for PubMedQA questions whose gold PMID is in the pilot subset (100% Recall@1, MRR@32 = 1.0000).
- Detailed verification log: `results/bm25_index_check.txt`



## BM25 Retrieval over MIRAGE (stage: retrieval)
- Script: `src/run_bm25_retrieval.py` (run as `caffeinate -i .venv/bin/python src/run_bm25_retrieval.py`). Pyserini 2.4.0, BM25 defaults, k=32, deterministic (no seed). Query = `question` field only, key `<dataset>/<id>`.
- Outputs: `data/processed/retrieval_bm25.parquet` (qkey, dataset, rank, docid, pmid, score), `data/processed/retrieval_bm25_flagged.parquet` (+ is_retracted, is_notice, one_hop, in_situ, n_retracted_cited), resumable chunks in `data/processed/retrieval_partial/`. Log: `results/retrieval_exposure.txt`.
- Run: 7,663 questions (medqa 1,273; medmcqa 4,183; pubmedqa 500; bioasq 618; mmlu 1,089). 245,121 rows (not 245,216: 95 short). 0 truncated queries. ~0.7s per 100 questions.
- 4 queries returned <32 hits, all medmcqa, all very short (2-3 words) = one rare drug/term plus function words ("is", "a"), not stopwords-only: "Arestin is:" (0 hits), "Algophobia is:" (0), "Prucalopride is a" (19), "Dicor is" (14). Likely fewer than 32 subset docs contain the term (5% corpus subset).
- Exposure (share of retrieved rows | questions with >=1 hit), corpus base rate in brackets:
  - is_retracted: 0.113% | 3.42% [0.125%]. Bioasq highest (0.324% | 9.22%), medqa lowest (0.020% | 0.63%).
  - one_hop: 1.50% | 31.4% [1.77%]. Bioasq 4.71% | 60.0%.
  - is_notice: 0.006% | 0.18% [0.019%].
  - Row-level exposure is at or below base rate overall (bioasq is the exception for retracted and one-hop).

## Exposure Analysis and Extra Checks (BM25, MIRAGE)
- `src/analyze_exposure.py` -> `results/exposure_analysis.txt`; `src/extra_checks.py` -> `results/extra_checks.txt`. Both read `retrieval_bm25_flagged.parquet`; base rates from `subset_flags.parquet`; question denominators from `benchmark.json` (7,663; the 2 zero-hit questions count as unexposed).
- Rank buckets, share of retrieved rows (corpus base rate: is_retracted 0.1247%, one_hop 1.7713%). Only 276 rows are retracted, so the is_retracted cells are small-count.

  | is_retracted | 1-5 | 6-16 | 17-32 |
  |---|---|---|---|
  | overall | 0.0914% | 0.1222% | 0.1126% |
  | medqa | 0.0157% | 0.0143% | 0.0245% |
  | medmcqa | 0.1052% | 0.1326% | 0.1122% |
  | pubmedqa | 0.1200% | 0.1455% | 0.1000% |
  | bioasq | 0.2265% | 0.3383% | 0.3439% |
  | mmlu | 0.0367% | 0.0751% | 0.0918% |

  | one_hop | 1-5 | 6-16 | 17-32 |
  |---|---|---|---|
  | overall | 2.0102% | 1.5095% | 1.3358% |
  | medqa | 3.9906% | 1.4711% | 0.9525% |
  | medmcqa | 1.0667% | 1.0589% | 1.0229% |
  | pubmedqa | 1.2800% | 1.1818% | 1.4250% |
  | bioasq | 5.1780% | 4.7661% | 4.5206% |
  | mmlu | 1.8549% | 1.5861% | 1.1364% |

- Question-level exposure by k: questions with >=1 hit within ranks <= k, count (share of all questions in group).

  | is_retracted | n_q | k=1 | k=3 | k=5 | k=10 | k=32 |
  |---|---|---|---|---|---|---|
  | overall | 7,663 | 11 (0.14%) | 20 (0.26%) | 35 (0.46%) | 81 (1.06%) | 262 (3.42%) |
  | medqa | 1,273 | 0 (0.00%) | 0 (0.00%) | 1 (0.08%) | 1 (0.08%) | 8 (0.63%) |
  | medmcqa | 4,183 | 7 (0.17%) | 13 (0.31%) | 22 (0.53%) | 55 (1.31%) | 153 (3.66%) |
  | pubmedqa | 500 | 2 (0.40%) | 2 (0.40%) | 3 (0.60%) | 4 (0.80%) | 18 (3.60%) |
  | bioasq | 618 | 2 (0.32%) | 3 (0.49%) | 7 (1.13%) | 16 (2.59%) | 57 (9.22%) |
  | mmlu | 1,089 | 0 (0.00%) | 2 (0.18%) | 2 (0.18%) | 5 (0.46%) | 26 (2.39%) |

  | one_hop | n_q | k=1 | k=3 | k=5 | k=10 | k=32 |
  |---|---|---|---|---|---|---|
  | overall | 7,663 | 200 (2.61%) | 459 (5.99%) | 714 (9.32%) | 1,144 (14.93%) | 2,404 (31.37%) |
  | medqa | 1,273 | 97 (7.62%) | 188 (14.77%) | 248 (19.48%) | 337 (26.47%) | 514 (40.38%) |
  | medmcqa | 4,183 | 38 (0.91%) | 116 (2.77%) | 209 (5.00%) | 389 (9.30%) | 1,007 (24.07%) |
  | pubmedqa | 500 | 3 (0.60%) | 14 (2.80%) | 31 (6.20%) | 54 (10.80%) | 150 (30.00%) |
  | bioasq | 618 | 34 (5.50%) | 80 (12.94%) | 130 (21.04%) | 199 (32.20%) | 371 (60.03%) |
  | mmlu | 1,089 | 28 (2.57%) | 61 (5.60%) | 96 (8.82%) | 165 (15.15%) | 362 (33.24%) |

  | is_notice | n_q | k=1 | k=3 | k=5 | k=10 | k=32 |
  |---|---|---|---|---|---|---|
  | overall | 7,663 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 3 (0.04%) | 14 (0.18%) |
  | medqa | 1,273 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
  | medmcqa | 4,183 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 2 (0.05%) | 9 (0.22%) |
  | pubmedqa | 500 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 1 (0.20%) | 2 (0.40%) |
  | bioasq | 618 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 3 (0.49%) |
  | mmlu | 1,089 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

- "retract" mention check: of 189 distinct retracted docids (276 rows), 1 doc (1 row) contains "retract" (case-insensitive) in title or content: pmid 35080547, content only. 0 in titles.
- One-hop hub concentration (one-hop rows by distinct docid):
  - Ranks 1-32: 3,679 rows, 2,296 distinct docids; questions per doc median 1, max 521, 23 docs in >=5 questions; top 10 docs cover 758 rows (20.60%), top 100 cover 1,097 rows (29.82%).
  - Ranks 1-5 only: 770 rows, 416 distinct docids; questions per doc median 1, max 283, 3 docs in >=5 questions; top 10 cover 332 rows (43.12%), top 100 cover 454 rows (58.96%).
- `src/hub_inspect.py` (identity of the top one-hop docs, exposure with hubs excluded) has been run; results below.

## Hub Inspection (BM25, MIRAGE)
- `src/hub_inspect.py` -> `results/hub_inspect.txt`. Reads `retrieval_bm25_flagged.parquet`, `icite_citing_pairs.parquet`, `retracted.parquet`; base rates from `subset_flags.parquet`; titles and content looked up by docid in `data/raw/pubmed_subset/chunk/*.jsonl`.
- Top one-hop hub: docid `pubmed23n0800_14851`, pmid 25032021 ("Breathlessness with pulmonary metastases: a multimodal approach"). 521 questions (medqa 402, mmlu 95, medmcqa 23, pubmedqa 1), median rank 5.0, 283 questions within ranks 1-5. Content 1,021 words. Cites one retracted paper, pmid 18425927.
- One-hop question-level exposure with hubs excluded (lower bounds: excluded docs are dropped from the ranked lists without back-filling). Share (count) of questions, denominator all questions in group:

  | k=5 | overall | medqa | medmcqa | pubmedqa | bioasq | mmlu |
  |---|---|---|---|---|---|---|
  | nothing | 9.32% (714) | 19.48% (248) | 5.00% (209) | 6.20% (31) | 21.04% (130) | 8.82% (96) |
  | top 1 doc | 5.73% (439) | 2.28% (29) | 4.85% (203) | 6.00% (30) | 21.04% (130) | 4.32% (47) |
  | top 10 docs | 5.21% (399) | 0.71% (9) | 4.59% (192) | 5.80% (29) | 21.04% (130) | 3.58% (39) |
  | all docs in >=5 questions (23 docs) | 5.01% (384) | 0.71% (9) | 4.30% (180) | 5.80% (29) | 20.87% (129) | 3.40% (37) |

  | k=32 | overall | medqa | medmcqa | pubmedqa | bioasq | mmlu |
  |---|---|---|---|---|---|---|
  | nothing | 31.37% (2,404) | 40.38% (514) | 24.07% (1,007) | 30.00% (150) | 60.03% (371) | 33.24% (362) |
  | top 1 doc | 26.36% (2,020) | 16.97% (216) | 23.62% (988) | 30.00% (150) | 60.03% (371) | 27.09% (295) |
  | top 10 docs | 23.66% (1,813) | 6.21% (79) | 22.76% (952) | 29.80% (149) | 60.03% (371) | 24.06% (262) |
  | all docs in >=5 questions (23 docs) | 23.10% (1,770) | 5.97% (76) | 22.09% (924) | 29.60% (148) | 59.39% (367) | 23.42% (255) |

- Distinct retrieved docids: 147,741 (12.06% of the 1,224,889 corpus subset docs). is_retracted: 0.1279% of distinct retrieved docs (189) vs corpus base rate 0.1247%. one_hop: 1.5541% (2,296) vs corpus base rate 1.7713%.
