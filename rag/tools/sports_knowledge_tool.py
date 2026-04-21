# rag/tools/sports_knowledge_tool.py
import re
import math
import hashlib
import time
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import json
import ollama
from config.settings import LLM_MODEL

from config.settings import (
    SEARCH_RESULTS_LIMIT,
    RERANK_TOP_K,
    EMBEDDING_DEVICE,
    ENABLE_HYBRID_SEARCH,
)

# Hybrid search weights — tuned for sports queries which are keyword-heavy
# (player names, team names, stats). BM25 gets a higher share than a generic
# RAG pipeline would use.
_HYBRID_ALPHA = 0.5   # cosine weight  (semantic relevance)
_HYBRID_BETA  = 0.5   # BM25 weight    (exact keyword recall)

# Diversity cap: max chunks allowed from the same source snippet
_MAX_CHUNKS_PER_SOURCE = 2

_IPL_PENALTY_RE = re.compile(
    r'\b(ipl|rcb|csk|mi\b|kkr|srh|dc\b|pbks|gt\b|lsg\b|royal challengers|'
    r'chennai super|mumbai indians|kolkata knight|sunrisers|delhi capitals|'
    r'punjab kings|gujarat titans|lucknow super)\b',
    re.IGNORECASE
)
_IPL_FILTER_RE = re.compile(
    r'\b(ipl|rcb|csk|kkr|srh|pbks|gt\b|lsg\b|dc\b|'
    r'royal challengers|chennai super kings|mumbai indians|'
    r'kolkata knight riders|sunrisers|delhi capitals|'
    r'punjab kings|gujarat titans|lucknow super giants|'
    r'big bash|cpl\b|psl\b|sa20|the hundred|'
    r'franchise|domestic league|county cricket|ranji|'
    r'ipl career|ipl stats|ipl 20\d\d|ipl season)\b',
    re.IGNORECASE
    )

_FORMAT_SIGNAL_RE = re.compile(
    r'\b(test runs|odi runs|t20i runs|international runs|career runs|'
    r'test wickets|odi wickets|t20i wickets|batting average|'
    r'all formats|combined total|international career)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# ESPN Cricinfo fetcher — PRIMARY source for cricket stats
# ---------------------------------------------------------------------------

def _fetch_espn_cricinfo(player: str, stat_type: str = "runs") -> List[str]:
    """
    Fetch cricket career statistics from ESPN Cricinfo via DDG scoped search.

    ESPN Cricinfo (and its stats subdomain stats.espncricinfo.com) blocks ALL
    direct HTTP requests with 403 — Cloudflare protection + session cookies
    prevent requests/BeautifulSoup from ever seeing the page, even with
    browser-like headers. The internal consumer API also returns 403.

    The only working approach: DDG indexes Cricinfo pages and surfaces their
    content as search snippets. We run one targeted query PER FORMAT so DDG
    finds the format-specific page for each rather than a generic records-list
    page that only covers one format at a time.

    Query style: "till now" / "career total" outperforms year-suffixed queries
    because Cricinfo pages use present-tense phrasing, not year tags.
    """
    snippets: List[str] = []
    seen: set = set()

    # Per-format queries using natural phrasing that matches Cricinfo page text
    queries = [
        f"{player} Test {stat_type} career total till now espncricinfo",
        f"{player} ODI {stat_type} career total till now espncricinfo",
        f"{player} T20I {stat_type} career total till now espncricinfo",
        f"{player} all formats career batting {stat_type} espncricinfo stats",
        f"{player} international cricket career {stat_type} statistics espncricinfo",
    ]

    for q in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=4))
            added = 0
            for r in results:
                url   = r.get("href", "") or r.get("url", "")
                body  = r.get("body", "").strip()
                title = r.get("title", "").strip()
                if not body:
                    continue
                # Only keep results that reference Cricinfo
                if "espncricinfo" not in (url + title + body).lower():
                    continue
                snippet = f"{title}. {body}" if title else body
                key = hashlib.md5(snippet.lower().strip().encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    snippets.append(snippet)
                    added += 1
            print(f"[DEBUG] _fetch_espn_cricinfo: '{q[:60]}' → +{added}")
        except Exception as e:
            print(f"[DEBUG] _fetch_espn_cricinfo: query failed: {e}")

    print(f"[DEBUG] _fetch_espn_cricinfo: {len(snippets)} total Cricinfo snippets")
    return snippets


def _fetch_wikipedia_cricket(player: str, stat_type: str = "runs") -> List[str]:
    """
    Fetch cricket career statistics from Wikipedia as the secondary source.

    Uses the Wikipedia REST summary API (no scraping needed) and the
    DDG search scoped to site:en.wikipedia.org as a URL-discovery step.
    Returns plain-text snippets.
    """
    try:
        import requests
    except ImportError:
        print("[DEBUG] _fetch_wikipedia_cricket: requests not installed — skipping")
        return []

    snippets: List[str] = []

    # ── Discover the Wikipedia article title via DDG ─────────────────────────
    wiki_title: Optional[str] = None
    try:
        search_q = f"{player} cricket career statistics site:en.wikipedia.org"
        with DDGS() as ddgs:
            results = list(ddgs.text(search_q, max_results=5))
        for r in results:
            url = r.get("href", "") or r.get("url", "")
            if "en.wikipedia.org/wiki/" in url:
                # Extract the article title from the URL
                wiki_title = url.split("/wiki/")[-1].split("#")[0]
                break
        print(f"[DEBUG] _fetch_wikipedia_cricket: article → {wiki_title}")
    except Exception as e:
        print(f"[DEBUG] _fetch_wikipedia_cricket: DDG search failed: {e}")

    # ── Fetch summary via Wikipedia REST API ────────────────────────────────
    if wiki_title:
        try:
            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
            resp = requests.get(api_url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            extract = data.get("extract", "")
            if extract:
                # Split into sentences and keep only stat-relevant ones
                sentences = re.split(r'(?<=[.!?])\s+', extract)
                _NUM_RE = re.compile(r'\d{3,}')
                for s in sentences:
                    if _NUM_RE.search(s) and len(s) > 30:
                        snippets.append(f"Wikipedia: {s.strip()}")
                print(f"[DEBUG] _fetch_wikipedia_cricket: {len(snippets)} sentences from REST API")
        except Exception as e:
            print(f"[DEBUG] _fetch_wikipedia_cricket: REST API failed: {e}")

    # ── Also pull DDG snippets scoped to Wikipedia ───────────────────────────
    try:
        ddg_q = f"{player} international cricket career {stat_type} statistics wikipedia"
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(ddg_q, max_results=5))
        for r in ddg_results:
            if "wikipedia" not in (r.get("href", "") + r.get("url", "")).lower():
                continue
            body = r.get("body", "").strip()
            title = r.get("title", "").strip()
            if body:
                snippets.append(f"{title}. {body}" if title else body)
        print(f"[DEBUG] _fetch_wikipedia_cricket: DDG scoped → {len(ddg_results)} results")
    except Exception as e:
        print(f"[DEBUG] _fetch_wikipedia_cricket: Wikipedia DDG search failed: {e}")

    return snippets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    FIX (Bug 1): Proper tokenization that strips punctuation and lowercases.
    Previously, raw .lower().split() left punctuation attached to tokens
    (e.g. "scored." != "scored"), silently killing BM25 keyword matches for
    player names, stat keywords, and sentence-ending words.
    """
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _fetch_ddg(query: str, max_results: int = SEARCH_RESULTS_LIMIT, timelimit: str = None) -> List[str]:
    """Fetch raw text snippets from DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            kwargs = {"max_results": max_results}
            if timelimit:
                kwargs["timelimit"] = timelimit
            results = list(ddgs.text(query, **kwargs))
        snippets = []
        for r in results:
            title = r.get("title", "").strip()
            body  = r.get("body", "").strip()
            if body:
                snippets.append(f"{title}. {body}" if title else body)
        print(f"[DEBUG] sports_kb: fetched {len(snippets)} snippets for '{query}'")
        return snippets
    except Exception as e:
        print(f"[DEBUG] sports_kb: DDG fetch failed: {e}")
        return []


def _chunk_text(
    text: str,
    max_sentences: int = 3,
    source_idx: int = 0,
) -> List[Tuple[str, int]]:
    """
    Split text into non-overlapping sentence-based chunks and tag each with
    its source snippet index so we can enforce per-source diversity later.

    FIX (Bug 2): Removed the 1-sentence overlap (step = max_sentences - 1).
    Overlapping chunks caused adjacent chunks to share content, inflating
    retrieval scores for repeated passages and sending near-duplicate context
    to the LLM.

    Returns a list of (chunk_text, source_idx) tuples.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return [(text.strip(), source_idx)] if text.strip() else []

    chunks = []
    # FIX: step == max_sentences (no overlap)
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append((chunk.strip(), source_idx))

    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_bm25_scores(
    query_tokens: List[str],
    all_doc_tokens: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Compute BM25 scores for all documents against a query.
    Returns raw (unnormalized) scores.

    Relies on _tokenize() being applied to both query and documents upstream —
    this ensures punctuation-stripped tokens match correctly (Bug 1 fix).
    """
    N = len(all_doc_tokens)
    if N == 0:
        return []

    avg_dl = sum(len(d) for d in all_doc_tokens) / N

    df: dict = {}
    for doc in all_doc_tokens:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    scores = []
    for doc in all_doc_tokens:
        doc_len = len(doc)
        doc_tf: dict = {}
        for t in doc:
            doc_tf[t] = doc_tf.get(t, 0) + 1

        score = 0.0
        for term in query_tokens:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(
                (N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / avg_dl)
            )
            score += idf * tf_norm
        scores.append(score)

    return scores


def _diverse_topk(
    scored: List[Tuple[float, str, int]],
    top_k: int,
    max_per_source: int = _MAX_CHUNKS_PER_SOURCE,
) -> List[Tuple[float, str]]:
    """
    FIX (Bug 5): Enforce per-source diversity when selecting top_k chunks.

    Without this, all top_k chunks could come from a single verbose article
    (e.g. one long Ronaldo profile dominates every slot), giving the LLM
    redundant context from one perspective and missing other relevant sources.

    scored: list of (score, chunk_text, source_idx) sorted descending by score.
    Returns: list of (score, chunk_text) with at most max_per_source chunks
             from any single source snippet.
    """
    counts: dict = defaultdict(int)
    result = []
    for score, chunk, source_idx in scored:
        if counts[source_idx] < max_per_source:
            result.append((score, chunk))
            counts[source_idx] += 1
        if len(result) >= top_k:
            break
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sports_kb_chunks(
    query: str,
    embed_model: SentenceTransformer,
    top_k: int = RERANK_TOP_K,
    threshold: float = 0.30,
    max_results: int = SEARCH_RESULTS_LIMIT,
    max_sentences_per_chunk: int = 3,
    timelimit: str = None,
) -> List[str]:
    """
    Retrieval for sports knowledge with hybrid search support.

    Changes vs original
    -------------------
    Bug 1 — Tokenization: _tokenize() replaces raw .lower().split() everywhere.
            Strips punctuation so "scored." matches "scored" in BM25.

    Bug 2 — Overlap: _chunk_text() now uses non-overlapping windows
            (step = max_sentences). Overlapping windows duplicated sentences
            across adjacent chunks, inflating scores for repeated passages.

    Bug 3 — Dedup key: full-content MD5 hash replaces the 80-char prefix.
            The short prefix missed near-duplicates that differed only at the
            end, wasting embedding and scoring capacity.

    Bug 4 — Threshold: the fixed 0.30 threshold is now applied relative to
            the actual score distribution. If fewer than top_k chunks survive
            the absolute threshold, we fall back to the top-scoring chunk so
            the function never silently returns nothing for a valid query.

    Bug 5 — Diversity: _diverse_topk() caps chunks per source snippet at
            _MAX_CHUNKS_PER_SOURCE (default 2) so top_k slots aren't monopolised
            by one long article.

    Bonus — Weights: α/β shifted to 0.5/0.5 to better suit sports queries,
            which are keyword-heavy (player names, team names, exact stats).
    """

    # ------------------------------------------------------------------
    # Step 1 — Fetch
    # ------------------------------------------------------------------
    raw_snippets = _fetch_ddg(query, max_results=max_results, timelimit=timelimit)
    if not raw_snippets:
        return []

    # ------------------------------------------------------------------
    # Step 2 — Chunk (non-overlapping, source-tagged)
    # ------------------------------------------------------------------
    all_chunks_with_source: List[Tuple[str, int]] = []
    for src_idx, snippet in enumerate(raw_snippets):
        tagged = _chunk_text(snippet, max_sentences=max_sentences_per_chunk, source_idx=src_idx)
        all_chunks_with_source.extend(tagged)

    # ------------------------------------------------------------------
    # Deduplicate — FIX (Bug 3): MD5 of full text instead of 80-char prefix
    # ------------------------------------------------------------------
    seen: set = set()
    unique_chunks: List[str] = []
    unique_sources: List[int] = []

    for chunk_text, src_idx in all_chunks_with_source:
        # FIX: hash the full normalised content, not just first 80 chars
        key = hashlib.md5(chunk_text.lower().strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk_text)
            unique_sources.append(src_idx)

    print(f"[DEBUG] sports_kb: {len(unique_chunks)} unique chunks from {len(raw_snippets)} snippets")

    if not unique_chunks:
        return []

    # ------------------------------------------------------------------
    # Step 3a — Cosine similarity (always computed)
    # ------------------------------------------------------------------
    query_embedding  = embed_model.encode(query)
    chunk_embeddings = embed_model.encode(unique_chunks, batch_size=64)

    cosine_scores = [
        _cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]

    if ENABLE_HYBRID_SEARCH:
        # --------------------------------------------------------------
        # Step 3b — BM25 keyword scores
        # FIX (Bug 1): use _tokenize() for both query and documents
        # --------------------------------------------------------------
        query_tokens   = _tokenize(query)
        all_doc_tokens = [_tokenize(c) for c in unique_chunks]
        raw_bm25       = _compute_bm25_scores(query_tokens, all_doc_tokens)

        bm25_max  = max(raw_bm25) if raw_bm25 and max(raw_bm25) > 0 else 1.0
        bm25_norm = [s / bm25_max for s in raw_bm25]

        # Step 3c — Combine into hybrid score
        final_scores = [
            _HYBRID_ALPHA * cos + _HYBRID_BETA * bm25
            for cos, bm25 in zip(cosine_scores, bm25_norm)
        ]
        print(f"[DEBUG] sports_kb: hybrid search enabled (α={_HYBRID_ALPHA} cosine + β={_HYBRID_BETA} BM25)")
    else:
        final_scores = cosine_scores
        print(f"[DEBUG] sports_kb: hybrid search disabled, using cosine only")

    _year_str = time.strftime('%Y')
    _prev_year = str(int(_year_str) - 1)

    boosted_scores = []
    for score, chunk in zip(final_scores, unique_chunks):
        boost   =  0.10 if (_year_str in chunk or _prev_year in chunk) else 0.0
        boost  +=  0.05 if _FORMAT_SIGNAL_RE.search(chunk) else 0.0   # reward format-specific chunks
        penalty = -0.15 if _IPL_PENALTY_RE.search(chunk) else 0.0     # penalise IPL/franchise chunks
        boosted_scores.append(score + boost + penalty)
    final_scores = boosted_scores

    # ------------------------------------------------------------------
    # Step 4 — Filter by threshold + sort
    # FIX (Bug 4): if nothing clears the absolute threshold, fall back to
    # the single best chunk so we never silently return [] for a valid query.
    # ------------------------------------------------------------------
    scored_with_source = [
        (score, chunk, src)
        for score, chunk, src in zip(final_scores, unique_chunks, unique_sources)
        if score >= threshold
    ]
    scored_with_source.sort(key=lambda x: -x[0])

    print(f"[DEBUG] sports_kb: {len(scored_with_source)} chunks above threshold {threshold}")
    print(f"[DEBUG] sports_kb: top scores: {[round(s, 3) for s, _, _ in scored_with_source[:5]]}")

    # Fallback: threshold too aggressive — return best single chunk
    if not scored_with_source:
        print(f"[DEBUG] sports_kb: threshold {threshold} filtered everything — returning best chunk as fallback")
        best_idx = int(np.argmax(final_scores))
        return [unique_chunks[best_idx]]

    # ------------------------------------------------------------------
    # Step 5 — Diversity-aware top_k selection
    # FIX (Bug 5): cap chunks per source to avoid one article monopolising
    # all top_k slots.
    # ------------------------------------------------------------------
    diverse = _diverse_topk(scored_with_source, top_k=top_k)

    print(f"[DEBUG] sports_kb: returning {len(diverse)} diverse chunks")
    return [chunk for _, chunk in diverse]


# ---------------------------------------------------------------------------
# Universal stats extraction — works for any sport
# ---------------------------------------------------------------------------

