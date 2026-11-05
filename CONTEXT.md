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

