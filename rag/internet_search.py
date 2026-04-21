# rag/internet_search.py

import re
import time
import numpy as np
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse
from typing import List, Optional, Tuple
from config.settings import SEARCH_RESULTS_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BS_TIMEOUT      = 6
BS_MAX_URLS     = 3
MIN_SNIPPET_LEN = 30
DDG_OVERFETCH   = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NOISE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "img",
    "nav", "footer", "header", "aside", "form", "button",
    "input", "select", "textarea", "meta", "link", "head",
}

# Compiled regex — done once at import time
_RE_URL        = re.compile(r'https?://\S+')
_RE_WHITESPACE = re.compile(r'\s+')
_RE_REPEATED   = re.compile(r'(.{20,}?)\1+')
_RE_LONE_JUNK  = re.compile(r'^[\d\s\|\-–—•·]+$')
_RE_COOKIE_MSG = re.compile(
    r'(cookie|privacy policy|accept all|we use cookies'
    r'|gdpr|consent|subscribe to our newsletter)',
    re.IGNORECASE,
)

# Sentence boundary — split on . ! ? followed by whitespace + uppercase letter
# Keeps abbreviations like "Dr.", "U.S.A.", "e.g." intact (no capital after)
_RE_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Surface-level clean:
      - Strip URLs
      - Collapse whitespace / newlines into single spaces
      - Strip leading/trailing whitespace
    """
    text = _RE_URL.sub('', text)
    text = _RE_WHITESPACE.sub(' ', text)
    return text.strip()


def _deep_clean(text: str) -> str:
    """
    Content-aware clean applied AFTER _clean_text():
      - Remove repeated sentence fragments (boilerplate copy-paste artefacts)
      - Drop lines that are only numbers, pipes, bullets
      - Normalise Unicode dashes and smart quotes to ASCII
    """
    text = _RE_REPEATED.sub(r'\1', text)
    text = _RE_LONE_JUNK.sub('', text)
    text = (
        text
        .replace('\u2019', "'").replace('\u2018', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2013', '-').replace('\u2014', '--')
        .replace('\u00a0', ' ')     # non-breaking space
    )
    return text.strip()


def _is_noise_line(text: str) -> bool:
    """
    Return True for cookie banners, GDPR notices, and very short nav labels.
    """
    if _RE_COOKIE_MSG.search(text):
        return True
    if len(text.split()) < 5:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# URL UTILITIES — deduplication + domain diversity
# ─────────────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """
    Extract root domain from any URL, stripping subdomains.

    Examples:
        https://blog.medium.com/article  →  medium.com
        https://docs.python.org/3/       →  python.org
    """
    try:
        host  = urlparse(url).netloc.lower()
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return url


def _deduplicate_urls(urls: List[str]) -> List[str]:
    """
    Remove exact duplicate URLs (case-insensitive, ignoring trailing slash).
    Preserves order — first occurrence wins.
    """
    seen, unique = set(), []
    for url in urls:
        key = url.rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            unique.append(url)
    return unique


def _diversify_urls(urls: List[str], max_urls: int) -> List[str]:
    """
    Select up to max_urls URLs with ONE URL per root domain.

    Two-pass strategy:
      Pass 1 — best (highest DDG-ranked) URL per unique domain
      Pass 2 — fill remaining slots with second-pick URLs

    Prevents BeautifulSoup slots being wasted on the same site
    (e.g. 3 of 3 slots all going to medium.com).
    """
    domain_map: dict = {}
    for url in urls:
        domain = _extract_domain(url)
        domain_map.setdefault(domain, []).append(url)

    # Pass 1: one per domain
    selected  = [url_list[0] for url_list in domain_map.values()]
    remaining = max_urls - len(selected)

    # Pass 2: fill with runner-up URLs
    if remaining > 0:
        second_picks = [
            url_list[1]
            for url_list in domain_map.values()
            if len(url_list) > 1
        ]
        selected += second_picks[:remaining]

    return selected[:max_urls]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — DDG: FAST RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _ddg_search(query: str, top_k: int, timelimit: str = None) -> List[dict]:
    """
    Hit DuckDuckGo and return raw result dicts.
    Over-fetches by DDG_OVERFETCH so downstream stages have richer input.

    Args:
        query:     Search query string.
        top_k:     Desired result count (actual fetch = top_k * DDG_OVERFETCH).
        timelimit: DDG time filter — 'd' (day), 'w' (week), 'm' (month),
                   'y' (year), or None (no filter).
    """
    with DDGS() as ddgs:
        kwargs = {"max_results": top_k * DDG_OVERFETCH}
        if timelimit:
            kwargs["timelimit"] = timelimit
        return list(ddgs.text(query, **kwargs))


def _ddg_snippets(results: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract cleaned text snippets AND source URLs from DDG result dicts.

    Returns:
        snippets:  cleaned "title + body" strings
        urls:      corresponding page hrefs (fed to BeautifulSoup)
    """
    snippets, urls = [], []

    for r in results:
        text = " ".join(filter(None, [r.get("title", ""), r.get("body", "")]))
        text = _clean_text(text)
        text = _deep_clean(text)

        if len(text) > MIN_SNIPPET_LEN:
            snippets.append(text[:500])

        href = r.get("href", "").strip()
        if href and href.startswith("http"):
            urls.append(href)

    return snippets, urls


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — BEAUTIFULSOUP: HIGH-QUALITY DEEP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_html(url: str, timeout: int = BS_TIMEOUT) -> Optional[str]:
    """
    GET a page and return its raw HTML, or None on any error.
    Uses a browser-like User-Agent to reduce 403 / bot-block responses.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _bs_extract(html: str) -> str:
    """
    BeautifulSoup extraction pipeline:

    1. Parse HTML (lxml preferred, html.parser fallback).
    2. Remove all NOISE_TAGS in-place (scripts, nav, footer, ads).
    3. Locate best semantic content root:
           <article> → <main> → role="main" → id~content/main → <body>
    4. Collect <p>, <li>, <h1>-<h3>, <blockquote> in document order.
    5. Apply _clean_text() + _deep_clean() + _is_noise_line() filter.

    Returns a single cleaned string of the page's core editorial content.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Step 2: remove noise tags in-place
    for tag in soup(list(NOISE_TAGS)):
        tag.decompose()

    # Step 3: locate best content root
    root = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", id=re.compile(r'content|main|article', re.I))
        or soup.find("div", class_=re.compile(r'content|main|article', re.I))
        or soup.body
        or soup
    )

    # Step 4: collect prose tags in document order
    prose_tags = root.find_all(["p", "li", "h1", "h2", "h3", "blockquote"])
    raw_lines  = [tag.get_text(separator=" ", strip=True) for tag in prose_tags]

    # Step 5: clean + filter
    cleaned = []
    for line in raw_lines:
        line = _clean_text(line)
        line = _deep_clean(line)
        if len(line) > MIN_SNIPPET_LEN and not _is_noise_line(line):
            cleaned.append(line)

    return " ".join(cleaned)


def _bs_fetch_urls(urls: List[str], max_urls: int = BS_MAX_URLS) -> List[str]:
    """
    Fetch and deep-extract the best max_urls pages via BeautifulSoup.

    Pre-processing before any HTTP request:
      1. _deduplicate_urls() — drop exact duplicate hrefs
      2. _diversify_urls()   — enforce one URL per root domain

    Silently skips pages that error or yield < 100 chars.
    0.3 s polite delay between requests avoids rate-limiting.
    """
    urls = _deduplicate_urls(urls)
    urls = _diversify_urls(urls, max_urls)

    docs = []
    for url in urls:
        html = _fetch_html(url)
        if html:
            content = _bs_extract(html)
            if len(content) > 100:
                docs.append(content)
        time.sleep(0.3)

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — CHUNKING (sentence-based with overlap)
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split a block of text into individual sentences using the
    _RE_SENTENCE boundary pattern (. ! ? followed by space + capital).

    Handles common abbreviations intact:
      "Dr. Smith went..."  →  NOT split at "Dr."  (no capital after space)
      "U.S.A. reported..." →  NOT split at "A."   (no capital after space)
      "He won. She lost."  →  Split correctly at "won."
    """
    raw = _RE_SENTENCE.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s and len(s.split()) >= 3:   # drop fragments shorter than 3 words
            sentences.append(s)
    return sentences


def _chunk_text(
    docs: List[str],
    chunk_size: int = 5,    # number of SENTENCES per chunk
    overlap: int = 1,       # number of sentences shared between chunks
) -> List[str]:
    """
    Sentence-aware sliding window chunker.

    Why sentence-based (not character-based):
      - Embeddings encode complete thoughts → better cosine similarity
      - LLM receives coherent sentences → fewer hallucinations
      - Never cuts mid-word or mid-name ("Cristiano Ro" + "naldo")

    Args:
        docs:       cleaned document strings.
        chunk_size: sentences per chunk (default 5).
        overlap:    sentences shared with the next chunk (default 1).
                    Ensures context isn't hard-cut at boundaries.
    """
    chunks = []

    for doc in docs:
        sentences = _split_sentences(doc)
        if not sentences:
            continue

        # Short doc — emit as one chunk
        if len(sentences) <= chunk_size:
            chunk = " ".join(sentences).strip()
            if len(chunk) > MIN_SNIPPET_LEN:
                chunks.append(chunk)
            continue

        # Sliding window over sentences
        start = 0
        while start < len(sentences):
            window = sentences[start: start + chunk_size]
            chunk  = " ".join(window).strip()
            if len(chunk) > MIN_SNIPPET_LEN:
                chunks.append(chunk)
            start += chunk_size - overlap   # slide forward with overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — EMBEDDINGS + COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: List[str], model) -> np.ndarray:
    """
    Encode a list of texts into a float32 matrix of shape (N, dim).
    Works with any encoder that exposes .encode(texts) → array-like
    (sentence-transformers, FastEmbed, etc.)
    """
    return np.array(
        model.encode(texts, show_progress_bar=False),
        dtype=np.float32,
    )


def _cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    L2-normalised dot product between one query vector and N chunk vectors.
    Result range [-1, 1] — higher = more semantically similar.

    Args:
        query_vec:  shape (dim,)
        chunk_vecs: shape (N, dim)
    Returns:
        scores:     shape (N,)
    """
    eps = 1e-10
    q = query_vec  / (np.linalg.norm(query_vec) + eps)
    c = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + eps)
    return c @ q


def _rank_chunks(
    query: str,
    chunks: List[str],
    model,
    top_k: int,
) -> List[str]:
    """
    Embed query + all chunks in one batch, score with cosine similarity,
    return the top-k most relevant chunks in ranked order.
    """
    if not chunks:
        return []

    all_vecs   = _embed([query] + chunks, model)
    query_vec  = all_vecs[0]
    chunk_vecs = all_vecs[1:]
    scores     = _cosine_similarity(query_vec, chunk_vecs)
    top_idx    = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(texts: List[str]) -> List[str]:
    """
    Remove near-duplicate text chunks using first 80 chars as fingerprint.
    Preserves order — first occurrence wins.
    """
    seen, unique = set(), []
    for t in texts:
        key = t[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch(query: str, top_k: int = SEARCH_RESULTS_LIMIT,
          timelimit: str = None) -> List[str]:
    """
    Fast path — DDG only, no BeautifulSoup, no embedding.

    Pipeline:
        DDG Search → clean snippets → deduplicate → return top_k

    Use when latency matters more than content depth.

    Args:
        query:     Search query string.
        top_k:     Number of results to return.
        timelimit: DDG time filter — 'd' (day), 'w' (week), 'm' (month),
                   'y' (year), or None (no filter, default).
    """
    results         = _ddg_search(query, top_k, timelimit=timelimit)
    snippets, _urls = _ddg_snippets(results)
    return _deduplicate(snippets)[:top_k]


def fetch_and_rerank(
    query: str,
    reranker=None,                      # legacy — NOT used (see note below)
    top_k: int = SEARCH_RESULTS_LIMIT,
    embed_model=None,
    chunk_size: int = 5,                # sentences per chunk
    chunk_overlap: int = 1,             # sentences of overlap between chunks
    use_bs: bool = True,
    bs_max_urls: int = BS_MAX_URLS,
    timelimit: str = None,              # DDG time filter: 'd','w','m','y', or None
) -> List[str]:
    """
    Full dual-source pipeline:

        DDG Search
          ↓  fast snippets + source URLs
        URL pre-processing
          ↓  deduplicate_urls() → diversify_urls() (one per domain)
        BeautifulSoup deep-fetch
          ↓  full page text, noise-stripped, deep-cleaned
        Merge: BS docs (richer) + DDG snippets (fast fallback)
          ↓  deduplicate by 80-char fingerprint
        Sentence-based chunking  (sliding window, sentence overlap)
          ↓  deduplicate chunks
        Embeddings  (query + all chunks in one batch)
          ↓
        Cosine similarity  →  return top-k ranked chunks

    Args:
        query:         Search query string.
        reranker:      Legacy param — intentionally NOT called here.
                       A single rerank pass runs in build_context() after
                       all tool outputs are merged, saving ~400-500 ms.
        top_k:         Number of final chunks to return.
        embed_model:   Encoder with .encode(texts) → array-like.
                       If None, returns plain deduped chunks (no ranking).
        chunk_size:    Sentences per chunk (default 5).
        chunk_overlap: Sentences shared between consecutive chunks (default 1).
        use_bs:        Set False to skip BeautifulSoup (faster, lower quality).
        bs_max_urls:   How many URLs to deep-fetch via BS (default 3).
        timelimit:     DDG time filter — 'd' (day), 'w' (week), 'm' (month),
                       'y' (year), or None (no filter, default).
                       Useful for news/current-events queries.

    Returns:
        List of the most query-relevant text chunks, ranked best-first.
    """

    # ── Step 1: DDG → fast snippets + URLs ──────────────────────────────
    raw_results        = _ddg_search(query, top_k, timelimit=timelimit)
    ddg_snippets, urls = _ddg_snippets(raw_results)

    # ── Step 2: BeautifulSoup → domain-diverse deep content ──────────────
    bs_docs = []
    if use_bs and urls:
        bs_docs = _bs_fetch_urls(urls, max_urls=bs_max_urls)

    # ── Step 3: Merge — BS content first (richer), DDG as fill-in ────────
    all_docs = _deduplicate(bs_docs + ddg_snippets)

    if not all_docs:
        return []

    # ── Step 4: Sentence-based chunking ──────────────────────────────────
    chunks = _deduplicate(
        _chunk_text(all_docs, chunk_size=chunk_size, overlap=chunk_overlap)
    )

    if not chunks:
        return []

    # ── Step 5 & 6: Embed + cosine rank ──────────────────────────────────
    if embed_model is not None:
        return _rank_chunks(query, chunks, embed_model, top_k=top_k)

    # Fallback: no model — return deduped chunks up to top_k
    return chunks[:top_k]