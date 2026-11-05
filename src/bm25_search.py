"""
BM25 Search Helper for PubMed Corpus Subset.

Provides search(query, k=32) over data/processed/bm25_subset.
Maps Lucene document IDs to PubMed IDs (pmid: int) using
data/processed/subset_docs.parquet.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

INDEX_PATH = Path("data/processed/bm25_subset")
SUBSET_DOCS_PATH = Path("data/processed/subset_docs.parquet")

_SEARCHER: Optional[LuceneSearcher] = None
_ID_TO_PMID: Optional[Dict[str, int]] = None


def _init_resources():
    global _SEARCHER, _ID_TO_PMID
    if _SEARCHER is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Lucene index not found at {INDEX_PATH}")
        _SEARCHER = LuceneSearcher(str(INDEX_PATH))

    if _ID_TO_PMID is None:
        if not SUBSET_DOCS_PATH.exists():
            raise FileNotFoundError(f"Subset docs parquet not found at {SUBSET_DOCS_PATH}")
        df = pd.read_parquet(SUBSET_DOCS_PATH, columns=["id", "pmid"])
        assert df["id"].is_unique, f"Column 'id' in {SUBSET_DOCS_PATH} is not unique"
        _ID_TO_PMID = dict(zip(df["id"], df["pmid"].astype(int)))


def get_searcher() -> LuceneSearcher:
    """Return the cached LuceneSearcher instance."""
    _init_resources()
    assert _SEARCHER is not None
    return _SEARCHER


def get_id_to_pmid_map() -> Dict[str, int]:
    """Return the cached id -> pmid dictionary."""
    _init_resources()
    assert _ID_TO_PMID is not None
    return _ID_TO_PMID


def search(query: str, k: int = 32) -> List[Dict[str, Any]]:
    """
    Search the BM25 index with the given query.

    Args:
        query: Search query string.
        k: Number of hits to return (default: 32).

    Returns:
        List of dicts: {'docid': str, 'pmid': int, 'score': float, 'rank': int}
        Rank is 1-indexed (1 to len(hits)).
    """
    _init_resources()
    assert _SEARCHER is not None
    assert _ID_TO_PMID is not None

    hits = _SEARCHER.search(query, k=k)
    results = []
    for rank, hit in enumerate(hits, start=1):
        docid = hit.docid
        pmid = _ID_TO_PMID.get(docid)
        results.append({
            "docid": docid,
            "pmid": int(pmid) if pmid is not None else None,
            "score": float(hit.score),
            "rank": rank,
        })
    return results


if __name__ == "__main__":
    searcher = get_searcher()
    print(f"Index: {INDEX_PATH}")
    print(f"Num docs in index: {searcher.num_docs:,}")
    mapping = get_id_to_pmid_map()
    print(f"Loaded {len(mapping):,} id -> pmid mappings.")

    test_query = "eosinophilic meningitis Angiostrongylus"
    print(f"\nSmoke test query: '{test_query}' (k=3)")
    hits = search(test_query, k=3)
    for h in hits:
        print(f"Rank {h['rank']}: docid={h['docid']}, pmid={h['pmid']}, score={h['score']:.4f}")
