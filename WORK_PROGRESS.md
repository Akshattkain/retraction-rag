# Work progress

Dated build record for this project. Newest entries go last.

## 10/28/2026 — Akshatt Kain

- Work done: Set up the repo and environment (Python 3.12, Pyserini 2.4.0). Downloaded Retraction Watch, the PubMed "Retracted Publication" PMID list (35,274), MIRAGE, and a uniform 5% sample of the MedRAG PubMed corpus (every 20th baseline file, 58 files, 3.5 GB).
- Results / findings: A full clone of the Retraction Watch repo was 4.2 GB, almost all git history. A shallow clone is 79 MB. The full MedRAG corpus is about 70 GB, which is why the pilot uses the 5% sample.
- Blockers / next steps: Build the retraction label table.

## 10/30/2026 — Akshatt Kain

- Work done: Built the retraction label table from Retraction Watch (`RetractionNature == "Retraction"`) combined with the PubMed tag. Added a reinstatement flag.
- Results / findings: 37,201 retracted PMIDs. Retraction Watch has 30,546, the PubMed tag has 35,274, and 28,619 are in both. 6,655 are tag-only and have no retraction date or reason. 21 also have a reinstatement record and are flagged. 32,260 Retraction Watch rows have a DOI but no PMID and are kept in a separate file.
- Blockers / next steps: Scan the corpus subset.

## 11/2/2026 — Akshatt Kain

- Work done: Scanned the corpus subset for retracted abstracts. Checked MIRAGE's gold-evidence papers (PubMedQA and BioASQ) against the labels.
- Results / findings: The subset has 1,224,889 abstracts, 1,528 of them retracted (0.12%). 0 of PubMedQA's 500 gold papers are retracted, and 6 of BioASQ's 3,467. That is about what chance predicts (expected ~4.3). Only ~5% of gold papers are in the subset, so RAG answers can't be scored against the benchmark's answer labels. The answer-change test will compare answers with and without a retracted source instead.
- Blockers / next steps: Build the BM25 index.

## 11/2/2026 — Akshatt Kain

- Work done: Built the BM25 index over the subset with Pyserini (default parameters) and a search helper that maps documents to PMIDs.
- Results / findings: 1,224,889 documents indexed, matching the corpus scan. Index size 1.5 GB, build time 26 seconds. All 23 PubMedQA gold papers that are in the subset come back at rank 1. That is an easy test, so it checks the index and PMID mapping, not retrieval quality.
- Blockers / next steps: Citation links and retraction-notice lists.

## 11/4/2026 — Akshatt Kain

- Work done: Built retraction-notice lists (PubMed and Retraction Watch) and an exclusion rule so notices don't count as citing papers. Fetched citing papers from the iCite API for all retracted PMIDs.
- Results / findings: NLM renamed the "Retraction of Publication" type to "Retraction Notice" in 2026, so the old query returned nothing. The combined notice list has 38,142 PMIDs. 1,493 of them are also retracted papers (the notice reuses the paper's PMID), so the exclusion list keeps 36,649. iCite returned 37,135 of 37,201 retracted PMIDs, with 735,647 (retracted, citing) pairs from 597,694 unique citing papers. The median retracted paper has 7 citing papers and the maximum is 4,217.
- Blockers / next steps: Flag one-hop documents in the subset.

## 11/6/2026 — Akshatt Kain

- Work done: Flagged one-hop documents in the subset (non-retracted papers that cite a retracted one).
- Results / findings: 21,696 snippets (1.77%) cite a retracted paper. That is 21,695 distinct PMIDs, because one PMID (35083036) has two snippets. 85.1% cite exactly one retracted paper. 402 of the 1,528 retracted abstracts (26%) cite another retracted paper, against 1.8% for non-retracted abstracts. This is an observation, not a finding, since retractions cluster by topic and in citation networks.
- Blockers / next steps: Run retrieval over the MIRAGE questions.

## 11/9/2026 — Akshatt Kain

- Work done: Ran BM25 retrieval (k = 32) over all 7,663 MIRAGE questions and flagged every retrieved paper as retracted, one-hop, or a notice. Computed exposure by depth, question set and rank.
- Results / findings: 245,121 retrieved rows. 3.42% of questions (262) retrieved a retracted abstract at top 32, and 0.46% (35) at top 5. One-hop exposure is 31.4% (2,404) at top 32 and 9.3% (714) at top 5. Retracted abstracts are 0.113% of retrieved rows against a 0.125% corpus rate, so retrieval looks neutral. BioASQ is highest on both measures (9.2% direct and 60.0% one-hop at top 32). Four very short MedMCQA questions returned fewer than 32 hits.
- Blockers / next steps: Check what drives the one-hop number.

## 11/11/2026 — Akshatt Kain

- Work done: Inspected the most-retrieved one-hop papers and checked the cited retraction on PubMed.
- Results / findings: One paper (PMID 25032021, a case-study-style review on breathlessness) is retrieved for 521 questions, 283 of them in the top 5. It cites one retracted paper, a 2008 Cochrane review (PMID 18425927), withdrawn in November 2013 as out of date. The notice is a separate PubMed record (PMID 24272974), which is why the review has a Retraction Watch entry but not the PubMed retraction tag. Removing the 23 papers that appear in 5 or more questions lowers one-hop exposure from 31.4% to 23.1% at top 32 and from 9.3% to 5.0% at top 5. For MedQA at top 5 it falls from 19.5% to 0.7%.
- Blockers / next steps: Test what a retraction filter does.

## 11/13/2026 — Akshatt Kain

- Work done: Filter ablation. Fetched the top 100 for each question and rebuilt lists of up to 32 under three rules.
- Results / findings: Dropping retracted papers removes all direct exposure (262 questions at top 32, 35 at top 5) and leaves one-hop unchanged (2,404 and 714). This is partly built in, since a filter on retracted status can't touch papers that aren't retracted. Also dropping papers that cite retracted ones removes both, but changes the top 5 for 9.64% of questions (0.46% for the plain filter).
- Blockers / next steps: Split exposure by retraction reason.

## 11/14/2026 — Akshatt Kain

- Work done: Grouped the 110 Retraction Watch reasons into severity classes A (data, results, image, method or material problems), B (misconduct, ethics, authorship, peer review), C (investigation, unclear), D (administrative), plus U for papers with no listed reason. Recomputed direct and one-hop exposure by class.
- Results / findings: 55% of the 189 distinct retracted papers retrieved are class A. Class A papers account for 38% of one-hop questions at top 5 (3.55% of all questions) and 54% at top 32 (16.91%). The hub paper's strongest cited class is D, which is why the top-5 headline falls under the strict view. 16.2% of corpus one-hop papers cite only class D or U retracted papers. The class grouping is mine. Papers with no reason are not counted as class A, so strict figures are lower bounds.
- Blockers / next steps: Split one-hop exposure by citation timing.

## 11/17/2026 — Akshatt Kain

- Work done: Added figures f1 to f8 with a built-in check against the verified results. Reconciled 21,696 one-hop snippets with 21,695 distinct PMIDs.
- Results / findings: All plotted counts match the results files.
- Blockers / next steps: Fetch publication years for the timing split.

## 11/19/2026 — Akshatt Kain

- Work done: Fetched publication years from iCite for the 2,296 retrieved one-hop papers. Labelled each (retracted, citing) pair as cited before, in the same year as, or after the retraction year.
- Results / findings: iCite returned a year for all 2,296 papers (1980 to 2024). 2,537 pairs involve a dated retraction: 66% were cited before the retraction year, 11% in the same year, 23% after. 339 pairs involve tag-only papers with no date and are excluded. 10.0% of questions at top 32 (767) and 1.80% at top 5 (138) have a one-hop paper with a citation after the retraction year. The top-5 figure is 1.42% without the 23 repeat papers, so it does not depend on the hub paper. Timing is year-level only, and a citation after the retraction year doesn't show the authors knew.
- Blockers / next steps: Answer-change test (needs an LLM and a budget decision), topic-matched baselines and confidence intervals, query-time vs index-time filter, reinstated-paper sensitivity check, scaling check on a larger corpus fraction.