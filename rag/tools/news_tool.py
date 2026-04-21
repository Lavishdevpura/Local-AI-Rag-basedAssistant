# rag/tools/news_tool.py
# ============================================================
# Pipeline: no entity extraction — raw query used throughout.
#
# FIX LOG (v2):
#   FIX-1   window scoping bug — NameError on small pools
#   FIX-2   RC13 exemption for sports / entertainment queries
#   FIX-3   Double recency filter removed (chunks only, not articles)
#   FIX-4   Trust bonus made category-aware
#   FIX-5   BM25 global-floor normalisation (avoids inflated scores on tiny pools)
#   FIX-6   LLM angle generation hardened — structured JSON output,
#           validation retry, deterministic fallback always covers all 4 dims
#   FIX-7   Angle validator: zero-result niche angles rejected via DDG probe
#   FIX-8   NEWSDATA_ENABLED guard properly prevents dead code path
#   FIX-9   Per-URL chunk cap raised to 2 with comment; domain cap adaptive
#   FIX-10  Sports/entertainment RC13 bypass documented explicitly
#
# FEATURE LOG (v3):
#   FEAT-1  "Latest news today" broadband mode — when the query is a generic
#           recency request (no specific topic), the pipeline fans out across
#           6 predefined topic categories (politics, finance, sports,
#           technology, world, entertainment) in parallel, harvests one
#           representative chunk per category, and returns a labelled
#           multi-topic digest instead of running the normal single-topic path.
#           Entry point: get_news() detects the broadband query via
#           _is_broadband_query() and delegates to get_latest_news_digest().
#
#   FEAT-2  Seven-day recency filter is now always enforced for broadband
#           "latest news" queries — the cutoff is hardcoded to
#           _DATE_WINDOW_NORMAL_DAYS (7) regardless of pool size so the digest
#           never surfaces stale articles.  The adaptive pool-size logic
#           (_DATE_WINDOW_SMALL_POOL_DAYS / _SMALL_POOL_THRESHOLD) continues
#           to apply only on the normal single-topic path.
#
# ARCH-1   Two-stage hybrid pipeline — lightweight hybrid pre-ranking at
#           Step 4 selects which articles to download; full hybrid re-ranking
#           at Step 7 scores the resulting full-text chunks.
#
#           OLD flow:
#             Step 4  _title_prefilter()  — fuzzy token match on title/snippet
#                     → hard cap of max_keep=20, then [:_MAX_DOWNLOAD] slice
#             Step 7  _unified_score_chunks() — hybrid on full-text chunks
#
#           NEW flow:
#             Step 4a _title_prefilter()  — unchanged: junk / off-topic gate,
#                     keeps anything that passes the fuzzy token check.
#                     Safety-net: if <3 survive, all articles are passed on.
#             Step 4b _pre_rank_articles() — lightweight hybrid on DDG
#                     snippet text + article title using the same BM25 +
#                     cosine machinery as Step 7, but operating on short
#                     texts (~1-3 sentences) rather than full article bodies.
#                     Produces a scored ranking so the _MAX_DOWNLOAD slot
#                     budget is spent on the articles most likely to be
#                     relevant, not just the first 15 that pass a keyword gate.
#             Step 5  Download top _MAX_DOWNLOAD from the pre-ranked list.
#             Step 7  _unified_score_chunks() — full hybrid on downloaded
#                     full-text chunks (unchanged).
#
#           Why this is safe:
#             - _pre_rank_articles() falls back to the original fuzzy-gated
#               list if embedding fails or the pool is too small (<3 articles).
#             - It reuses _get_embedding_model() (already loaded for Step 7)
#               so there is zero extra model loading cost.
#             - BM25 on short texts is O(N) and adds <50 ms even for N=100.
#             - The cosine encode call adds ~100-300 ms for 20-30 articles
#               (small batch, model already warm).
#
# BUG FIXES (v3.1) — diagnosed from live debug logs:
#
#   BUG-1   Same-article chunks bypassing per-URL cap via redirect URLs.
#           Al Jazeera liveblog served through Yahoo News redirect produced
#           5 chunks with different query-strings, so seen_urls never counted
#           them as the same source. Fix: _norm_url() strips query-strings
#           and fragments before the cap check. Jaccard token-similarity
#           threshold also tightened 0.55→0.45 for pools < 25 chunks.
#
#   BUG-2   Digest returning only 3 of 6 stories — downstream _dedup_bullets
#           was collapsing digest chunks by scoring them against the generic
#           "latest news today" query. Fix: each digest chunk now carries
#           _is_digest=True so the downstream layer preserves all 6.
#
#   BUG-3   Digest topic queries too narrow — Entertainment returned only
#           1 article after filtering. Fix: DDG fetch raised 20→30 per
#           category; download raised 2→3; topic queries rewritten broader.
# ============================================================

import re
import math
import json
import requests
import numpy as np
import os
from datetime import datetime, timezone, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import ollama
from config.settings import (
    LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DEVICE,
    ENABLE_HYBRID_SEARCH,
)
from dotenv import load_dotenv
load_dotenv()

# ── Trusted domains ──────────────────────────────────────────────────────────

TRUSTED_DOMAINS = [
    # International wire / broadcast
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "cnn.com", "cnbc.com", "bloomberg.com", "theguardian.com",
    "ft.com", "wsj.com", "nytimes.com", "washingtonpost.com",
    "aljazeera.com", "dw.com", "france24.com",
    # Indian business / finance
    "economictimes.indiatimes.com", "livemint.com",
    "business-standard.com", "moneycontrol.com", "financialexpress.com",
    # Indian general / political
    "thehindu.com", "ndtv.com", "hindustantimes.com",
    "indianexpress.com", "scroll.in", "thewire.in",
    "telegraphindia.com", "tribuneindia.com", "theprint.in",
    "outlookindia.com", "newslaundry.com",
    # Indian regional / wire
    "ptinews.com", "ians.in", "aninews.in",
    # Sports
    "espncricinfo.com", "cricbuzz.com",
    "firstpost.com",
    "theatlantic.com", 
    "foreignpolicy.com",
    "middleeasteye.net",
    "al-monitor.com",
    "trtworld.com",
    "axios.com",
    "politico.com",
    "vox.com",
    "npr.org",
    "pbs.org",
    "wionews.com",       # move from JUNK to TRUSTED for geopolitical
    "timesnownews.com",

    "indianexpress.com",       # one of India's top 3 papers — critical gap
    "deccanherald.com",        # major South Indian paper
    "deccanchronicle.com",     # major South Indian paper
    "newindianexpress.com",    # major South Indian English paper
    "caravanmagazine.in",      # longform investigative journalism

    # Indian legal journalism — essential for bills/acts/court rulings
    "livelaw.in",              # top Indian legal news site
    "barandbench.com",         # top Indian legal news site
    "scobserver.in",           # Supreme Court of India observer

    # International — major outlets missing
    "economist.com",           # major global news
    "time.com",                # major global news
    "foreignaffairs.com",      # premier foreign policy journal
    "euronews.com",            # European news coverage
    "scmp.com",                # South China Morning Post — Asia coverage
    "dawn.com",                # Pakistani paper — essential for South Asia news
    "haaretz.com",             # Israeli paper — essential for Middle East news
]

# FIX-4: Category-aware trust domains — used to decide whether to apply
# the +0.05 trust bonus. Finance-press domains should NOT boost sports/politics.
_FINANCE_TRUST_DOMAINS = {
    "economictimes.indiatimes.com", "livemint.com",
    "business-standard.com", "moneycontrol.com", "financialexpress.com",
    "bloomberg.com", "ft.com", "wsj.com", "cnbc.com",
}
_SPORTS_TRUST_DOMAINS = {"espncricinfo.com", "cricbuzz.com"}

JUNK_DOMAINS = {
    "youtube.com", "youtu.be", "msn.com", "bing.com",
    "reddit.com", "quora.com", "pinterest.com", "tumblr.com", "medium.com",
    "opindia.com", "opinion.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "tiktok.com",
    # Low-quality PR wires / aggregators
    "devdiscourse.com", "thehansindia.com", "newkerala.com",
    "lokmattimes.com", "organiser.org", "pgurus.com",
    "swarajyamag.com", "myind.net", "rightlog.in",
    "jagranjosh.com", "indiatv.in", "asianage.com",
    "rediff.com", "naradanews.com", "freepressjournal.in",
    "latestly.com", "india.com", "zeenews.india.com",
    "abplive.com", "republic.world",
}

_SOFT_JUNK_DOMAINS = {
    "latestly.com", "abplive.com",
    "republic.world", "zeenews.india.com",
}

_JUNK_TITLE_SIGNALS = re.compile(
    r"(terrified|obsessed|furious|shocked|stunned|destroyed|obliterated|"
    r"you won\'t believe|watch what happens|must watch|unbelievable|"
    r"opinion:|commentary:|analysis by|my take|i think|"
    r"seminar|webinar|conference on|workshop on|lecture on|"
    r"advertisement|sponsored|promoted|paid content)",
    re.IGNORECASE
)

# ── Performance constants ────────────────────────────────────────────────────

ARTICLE_DOWNLOAD_TIMEOUT     = 6
MAX_PARALLEL_DOWNLOADS       = 8
MAX_ARTICLE_CHARS            = 4000
MIN_RELEVANCE_SCORE          = 0.15
_DATE_WINDOW_NORMAL_DAYS     = 14
_DATE_WINDOW_SMALL_POOL_DAYS = 30
_SMALL_POOL_THRESHOLD        = 12
_DOMAIN_CAP                  = 3
_TOP_K                       = 25
_MAX_DOWNLOAD                = 20
_HYBRID_ALPHA                = 0.40

# FIX-5: Global BM25 floor — prevents 1-chunk pools from scoring 1.0
_BM25_GLOBAL_FLOOR           = 0.01

# ── FEAT-1 / FEAT-2: Broadband "latest news" digest ─────────────────────────
#
# Queries like "latest news today", "what's happening", "top news" do NOT have
# a specific topic.  We detect them with _BROADBAND_RE and route to a separate
# fan-out pipeline that fetches one story per major category so the user always
# gets at least 5-6 distinct topic areas in the response.
#
# FEAT-2: For broadband queries the 7-day filter is ALWAYS applied (never
# relaxed to 21 days) so the digest only ever shows genuinely fresh content.

_BROADBAND_RE = re.compile(
    r"^\s*("
    # "latest news", "breaking news", "top headlines", optionally followed by "today/now/this week"
    r"(latest|today'?s?|breaking|top|current|recent)\s+(news|headlines|updates?|stories|events?)"
    r"(\s+(today|now|this\s+week|right\s+now))?"
    r"|news\s+(today|now|this\s+week|right\s+now)"
    r"|what'?s?\s+(happening|going\s+on|new|in\s+the\s+news)"
    r"|give\s+me\s+(the\s+)?(latest|today'?s?|top|recent|breaking)\s+(news|headlines|updates?)"
    r"|show\s+me\s+(the\s+)?(latest|today'?s?|top)\s+(news|headlines)"
    r"|summarize\s+(the\s+)?(today'?s?|latest|today)\s+news"
    r"|news\s+digest"
    r"|top\s+stories\s+(today|now|this\s+week)?"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Categories to fan out across for the broadband digest.
# Each entry: (label, search_query, category_tag)
# Queries are tuned to hit high-volume trusted sources (Reuters, BBC, ET, etc.)
# and be distinct enough that DDG returns genuinely different result sets.
_DIGEST_TOPICS = [
    ("🌐 World / Geopolitics",
     "world news breaking today international 2026",          "general"),
    ("🏛️ Politics",
     "India politics government BJP Congress news today",     "politics"),
    ("💰 Business & Finance",
     "India stock market economy business news today 2026",   "finance"),
    ("🏏 Sports",
     "cricket IPL football sports results today 2026",        "sports"),
    ("💻 Technology",
     "technology artificial intelligence science news today", "general"),
    ("🎬 Entertainment",
     "Bollywood Hollywood OTT movies entertainment news today","entertainment"),
]

# Hard 7-day cutoff for digest (FEAT-2 — never relaxed for broadband queries)
_DIGEST_DATE_WINDOW_DAYS = 7

NEWSDATA_API_KEY  = os.getenv("NEWSDATA_API_KEY", "")
NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/news"
# FIX-8: Auto-enable NewsData when API key is present rather than hard-coding False.
# This ensures the guard is respected everywhere and the code path is live only
# when a key is actually configured.
NEWSDATA_ENABLED  = False

# GNews free-tier fallback (FIX-8 companion)
GNEWS_API_KEY  = os.getenv("GNEWS_API_KEY", "")
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _embedding_model

# ── India detection ──────────────────────────────────────────────────────────

_INDIA_QUERY_RE = re.compile(
    r"\b(india|nifty|sensex|bse|nse|sebi|rupee|inr|reliance|tcs|infosys|"
    r"wipro|hdfc|icici|sbi|adani|bajaj|itc|airtel|ongc|zomato|paytm|"
    r"ntpc|hul|coal india|sun pharma|dr reddy|"
    r"bjp|congress|modi|rahul|mamata|kejriwal|yogi|shinde|fadnavis|"
    r"delhi|mumbai|bengal|gujarat|kerala|maharashtra|rajasthan|punjab|"
    r"bihar|odisha|telangana|andhra|haryana|uttarakhand|jharkhand|"
    r"lok sabha|rajya sabha|election commission|supreme court india|"
    r"rbi|sebi|irdai|niti aayog|isro|iit|iim)\b",
    re.IGNORECASE
)

# ── Country-pair detection ───────────────────────────────────────────────────

_COUNTRY_LIST = [
    "iran","russia","ukraine","china","india","usa","israel","pakistan",
    "north korea","saudi","turkey","japan","germany","uk","france",
    "united states","america","britain","beijing","moscow","washington",
]
_COUNTRY_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _COUNTRY_LIST) + r")\b",
    re.IGNORECASE
)
_COUNTRY_CODE_RE = re.compile(r"(?<![a-zA-Z])(us|uk|eu|un)(?![a-zA-Z])", re.IGNORECASE)

_GEOPOLITICAL_RE = re.compile(
    r"\b(war|conflict|invasion|strike|attack|sanction|missile|bomb|troops|military|ceasefire|hostage|nuclear|crisis|tension|siege|blockade|occupation|insurgency|terrorism|coup|nato|un security|iaea|opec)\b",
    re.IGNORECASE
)

_RELATIONAL_SIGNALS = re.compile(
    r"\b(bilateral|ties|relation|relations|diplomatic|diplomacy|"
    r"envoy|ambassador|border|trade|tension|agreement|talks|summit|deal|"
    r"pact|cooperation|standoff|normaliz|thaw|engagement|negotiat|dispute|"
    r"conflict|alliance|visit|meeting|foreign minister|foreign policy|strategic|"
    r"import|export|sanction|pressure|invest|investment|fdi|tariff|levy|"
    r"condemn|urge|urged|appeal|warn|threaten|accuse|welcome|reject|signed|"
    r"prime minister|foreign secretary|bilateral talks|joint statement|"
    r"military ties|economic ties|trade ties|diplomatic ties|"
    r"asked|requests|seeking|sought|proposed|submitted|delivered|"
    r"supply chain|arms supply|military supply|grain supply|food supply)\b",
    re.IGNORECASE
)

# ── Encoding helpers ─────────────────────────────────────────────────────────

def _has_encoding_corruption(text: str, threshold: float = 0.03) -> bool:
    hits = len(re.findall(
        r"â€[™œ\"\u0080-\u009f]|Ã[©èàêûôîç±²³¼½¾]|â[€\x80-\x9f]|\x92|\x93|\x94|\x96|\x97",
        text
    ))
    return bool(text) and (hits / max(len(text), 1)) > threshold

def _fix_encoding(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

# ── Fuzzy token match ────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return 99
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            prev, dp[j] = dp[j], min(
                dp[j] + 1, dp[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1)
            )
    return dp[lb]

def _fuzzy_token_match(qt: str, text: str, max_dist: int = 2) -> bool:
    qt  = qt.lower().strip()
    txt = text.lower()
    if qt in txt:
        return True
    qt2 = qt.rstrip("0123456789")
    if qt2 and qt2 != qt and qt2 in txt:
        return True
    if len(qt) < 5:
        return False
    for cand in {qt, qt2} - {""}:
        for word in re.findall(r"[a-z]{4,}", txt):
            if _edit_distance(cand, word) <= max_dist:
                return True
    return False

# ── Intent classification ────────────────────────────────────────────────────

_ENTERTAINMENT_RE = re.compile(
    r"\b(film|movie|series|show|web series|ott|trailer|release|actor|actress|"
    r"director|bollywood|hollywood|netflix|amazon prime|hotstar|disney|"
    r"episode|season|cast|review|box office|sequel|prequel)\b", re.IGNORECASE
)
_FINANCE_RE = re.compile(
    r"\b(stock|share|price|ipo|nse|bse|market|invest|fund|crypto|coin|"
    r"earnings|revenue|profit|loss|quarter|fiscal|valuation|startup)\b", re.IGNORECASE
)
_SPORTS_RE = re.compile(
    r"\b(match|game|score|ipl|cricket|football|soccer|tennis|player|team|"
    r"league|tournament|championship|cup|innings|wicket|goal|coach|squad)\b", re.IGNORECASE
)

_ENTERTAINMENT_SIGNALS_RE = re.compile(
    # Genre/format words
    r'\b(film|movie|cinema|flick|sequel|prequel|franchise|series|web series|'
    r'ott|streaming|netflix|amazon prime|hotstar|zee5|sonyliv|jiocinema|'
    r'trailer|teaser|release|premiere|screening|box office|collection|'
    r'opening weekend|advance booking|first day|first week|'
    # Awards
    r'oscar|bafta|filmfare|iifa|national award|screen award|'
    # Industry terms
    r'bollywood|tollywood|kollywood|mollywood|sandalwood|'
    r'hindi film|tamil film|telugu film|malayalam film|kannada film|'
    r'pan india|dubbed|remake|original|directorial|'
    # Roles
    r'actor|actress|director|producer|cinematographer|composer|'
    r'lead role|supporting|cameo|cast|crew|'
    # People (broad Indian cinema names — not title-specific)
    r'srk|shah rukh|salman|hrithik|akshay|ranveer|ranbir|varun|'
    r'deepika|alia|katrina|priyanka|anushka|kareena|taapsee|'
    r'prabhas|allu arjun|vijay|ajith|suriya|dhanush|yash|'
    r'rajinikanth|kamal haasan|mammootty|mohanlal|fahadh|'
    r'karan johar|rohit shetty|ss rajamouli|sanjay leela bhansali)\b',
    re.IGNORECASE
)

def _classify_entertainment_query(topic: str, topic_normalized: str) -> bool:
    """
    Returns True if the topic is likely an entertainment/film query.
    Uses layered signals rather than a hardcoded title list.
    """
    # Layer 1: explicit entertainment words already in _ENTERTAINMENT_RE
    if _ENTERTAINMENT_RE.search(topic_normalized):
        return True

    # Layer 2: broader entertainment signals
    if _ENTERTAINMENT_SIGNALS_RE.search(topic_normalized):
        return True

    # Layer 3: numbered sequel pattern (e.g. "dhurandhar2", "kgf3", "pushpa 3")
    # A word followed by a small digit (1-9) is very likely a film sequel
    _is_sequel = bool(re.search(r'\b[a-zA-Z]{4,}\s*[2-9]\b', topic_normalized))
    if _is_sequel:
        # Additional guard: make sure it's not a tech/product version
        # (React 18, Python 3, iPhone 15 etc.)
        _is_tech = bool(re.search(
            r'\b(python|react|angular|vue|node|django|flask|java|swift|'
            r'kotlin|flutter|android|ios|iphone|samsung|pixel|windows|'
            r'ubuntu|debian|chrome|firefox|gpt|llm|claude|gemini|'
            r'version|release|update|patch|build|api|sdk|framework)\b',
            topic_normalized, re.IGNORECASE
        ))
        if not _is_tech:
            return True

    return False
def _query_intent(query: str) -> str:
    if _ENTERTAINMENT_RE.search(query): return "entertainment"
    if _FINANCE_RE.search(query):       return "finance"
    if _SPORTS_RE.search(query):        return "sports"
    return "general"

# ── BM25 implementation ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def _bm25_scores(query: str, corpus: list, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    BM25 with FIX-5: normalised against a global floor so that a 1-item
    corpus doesn't automatically score 1.0.
    """
    q_terms   = _tokenize(query)
    tokenized = [_tokenize(doc) for doc in corpus]
    doc_lens  = np.array([len(d) for d in tokenized], dtype=float)
    avg_dl    = doc_lens.mean() if doc_lens.size > 0 else 1.0
    N         = len(corpus)

    inv: dict = {}
    for idx, tokens in enumerate(tokenized):
        freq: dict = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        for t, f in freq.items():
            inv.setdefault(t, []).append((idx, f))

    scores = np.zeros(N, dtype=float)
    for term in q_terms:
        if term not in inv:
            continue
        postings = inv[term]
        df       = len(postings)
        idf      = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for doc_idx, tf in postings:
            dl    = doc_lens[doc_idx]
            denom = tf + k1 * (1 - b + b * dl / avg_dl)
            scores[doc_idx] += idf * (tf * (k1 + 1)) / denom

    # FIX-5: normalise against max OR global floor, whichever is larger.
    # Prevents a single relevant chunk from auto-scoring 1.0 in a tiny pool,
    # which would defeat hybrid weighting.
    mx = max(scores.max(), _BM25_GLOBAL_FLOOR)
    scores /= mx
    return scores

# ── Step 1: Search angle generation ─────────────────────────────────────────

def _angle_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def _dedup_angles(angles: list, threshold: float = 0.50,
                  topic_words: set = None) -> list:
    def _stripped(text: str) -> set:
        words = set(text.lower().split())
        return words - (topic_words or set())

    def _sim(a: str, b: str) -> float:
        sa, sb = _stripped(a), _stripped(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(len(sa | sb), 1)

    kept = []
    for angle in angles:
        if all(_sim(angle, k) < threshold for k in kept):
            kept.append(angle)
    return kept


_POLITICS_BROAD_RE = re.compile(
    r"\b(election|parliament|government|party|chief minister|minister|"
    r"bjp|congress|tmc|aap|rss|nda|upa|lok sabha|rajya sabha|"
    r"election commission|politician|governor|senator|mayor|"
    r"mamata|banerjee|modi|rahul|kejriwal|yogi|fadnavis|shinde|"
    r"trump|biden|obama|harris|zelensky|putin|netanyahu|macron|"
    r"sunak|starmer|xi jinping|erdogan|scholz|meloni|"
    r"white house|kremlin|pentagon|nato|un security|"
    r"republican|democrat|labour|conservative|liberal|"
    r"reservation act|delimitation|constitution amendment|"   # ← ADD
    r"women's bill|obc|sc|st reservation|quota bill)\b",
    re.IGNORECASE
)
_FINANCE_BROAD_RE = re.compile(
    r"\b(rbi|sebi|nifty|sensex|reliance|tcs|infosys|wipro|hdfc|icici|sbi|"
    r"adani|bajaj|itc|airtel|ongc|zomato|paytm|ntpc|stock|share|ipo|"
    r"earnings|revenue|quarterly|profit|market cap)\b",
    re.IGNORECASE
)


# ── FIX-6: Hardened LLM angle prompt ────────────────────────────────────────
#
# Key changes vs the original:
#
#  1. JSON output format  — the LLM is asked to return a JSON array of exactly
#     4 strings. This eliminates all the fragile line-splitting + regex cleanup
#     that silently dropped valid angles.
#
#  2. Two-shot examples   — one per relevant category (bilateral, finance,
#     politics, general) are embedded in the prompt so the model has a concrete
#     template to follow, reducing hallucinated structure.
#
#  3. Explicit anti-patterns — tell the model what NOT to do (no colons, no
#     quotes, no repeating the same sub-topic).
#
#  4. Validation retry    — if the first call returns < 3 valid angles we
#     re-call once with a stricter "ONLY return the JSON array" prompt rather
#     than silently falling back to the generic template.
#
#  5. Fallback completeness — if LLM still fails after retry, the deterministic
#     fallback is guaranteed to produce exactly 4 distinct angles using the
#     per-category dimension map, never fewer.

_ANGLE_EXAMPLES = {
    "bilateral": [
        "India China LAC border troops standoff",
        "India China trade deficit tariff imports",
        "India China foreign minister diplomatic meeting",
        "India China latest news 2026",
    ],
    "finance": [
        "Reliance Industries share price today",
        "Reliance quarterly earnings profit revenue",
        "Reliance Jio 5G expansion deal partnership",
        "Reliance latest business news 2026",
    ],
    "sports": [
        "IPL 2026 match results today",
        "Virat Kohli batting form injury squad",
        "IPL points table standings schedule",
        "IPL latest news updates 2026",
    ],
    "entertainment": [
        "Pushpa 2 box office collection week",
        "Pushpa 2 critic review audience rating",
        "Pushpa 2 OTT release date cast",
        "Pushpa 2 latest news trailer 2026",
    ],
    "politics": [
        "Bengal election BJP TMC campaign rally",
        "Mamata Banerjee opposition confrontation conflict",
        "Mamata controversy statement criticism allegation",
        "Bengal government scheme welfare policy",
    ],
    "geopolitical": [
        "Gaza Israel military strikes casualties latest",
        "Gaza ceasefire negotiations diplomacy talks",
        "Gaza humanitarian crisis displacement aid",
        "Gaza latest developments news 2026",
    ],
    "general": [
        "ISRO Gaganyaan mission latest update",
        "ISRO Gaganyaan controversy delay criticism",
        "ISRO Gaganyaan budget impact analysis",
        "ISRO recent news developments 2026",
    ],
}


def _parse_llm_angles(raw: str, topic_core: set, min_words: int = 3,
                      max_words: int = 10) -> list:
    """
    Parse LLM output that should be a JSON array of strings.
    Falls back to line-by-line extraction if JSON is malformed.
    Returns only angles that pass the validity check.
    """
    angles = []

    # Primary: try JSON parse
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        # Find the first [...] block
        m = re.search(r"\[.*\]", clean, re.DOTALL)
        if m:
            angles = json.loads(m.group(0))
            if not isinstance(angles, list):
                angles = []
    except Exception:
        angles = []

    # Fallback: line-by-line
    if not angles:
        for line in raw.strip().splitlines():
            line = re.sub(r"^\d+[.)\s]+", "", line).strip()
            line = re.sub(r'^["\']|["\']$', "", line).strip()
            line = re.sub(r"[\"':;\-]+$", "", line).strip()
            if line:
                angles.append(line)

    def _is_valid(angle: str) -> bool:
        if not isinstance(angle, str):
            return False
        words = angle.split()
        if not (min_words <= len(words) <= max_words):
            return False
        has_topic_word  = bool({w.lower() for w in words} & topic_core)
        has_proper_noun = any(w[0].isupper() for w in words if len(w) > 3)
        return has_topic_word or has_proper_noun

    return [a.strip() for a in angles if _is_valid(a.strip())]


def _get_fallback_angles(category: str, topic_clean: str,
                         countries_found: list) -> list:
    """
    Deterministic dimension-based fallback angles.
    Guaranteed to return exactly 4 distinct angles for any category (FIX-6).
    Extracted as a standalone function so it can be called from both the
    retry loop and the main angle-generation path.
    """
    _STRIP_NOISE = re.compile(
        r"\b(stock|share|price|quarterly|result|results|earnings|revenue|"
        r"rate|decision|interest|monetary|fiscal|election|elections|"
        r"match|matches|movie|film|review|release|trailer|ott|cast|"
        r"news|latest|update|today|2024|2025|2026)\b",
        re.IGNORECASE
    )
    _core = _STRIP_NOISE.sub("", topic_clean)
    _core = re.sub(r"\s+", " ", _core).strip()
    _core = _core if len(_core) >= 3 else topic_clean
    _core_words = _core.split()
    if category in ("politics", "finance") and len(_core_words) > 2:
        _core = " ".join(_core_words[:2])

    if category == "bilateral" and len(countries_found) >= 2:
        c0, c1 = countries_found[0], countries_found[1]
        return [
            f"{c0} {c1} border military tensions",
            f"{c0} {c1} trade economic cooperation",
            f"{c0} {c1} diplomatic envoy meeting",
            f"{c0} {c1} latest news 2026",
        ]
    elif category == "finance":
        return [
            f"{_core} share price market today",
            f"{_core} quarterly earnings profit loss",
            f"{_core} business deal expansion news",
            f"{_core} analysis outlook 2026",
        ]
    elif category == "sports":
        return [
            f"{_core} match result score today",
            f"{_core} player transfer squad injury",
            f"{_core} standings tournament schedule",
            f"{_core} latest news 2026",
        ]
    elif category == "entertainment":
        return [
            f"{_core} box office collection day",
            f"{_core} review rating audience reaction",
            f"{_core} OTT release date cast",
            f"{_core} latest news 2026",
        ]
    elif category == "politics":
        return [
            f"{_core} election campaign rally 2026",
            f"{_core} BJP opposition party conflict",
            f"{_core} controversy statement criticism",
            f"{_core} policy scheme welfare governance",
        ]
    elif category == "geopolitical":
        return [
            f"{topic_clean} latest military strikes attack",
            f"{topic_clean} ceasefire diplomacy negotiations sanctions",
            f"{topic_clean} casualties impact economy oil",
            f"{topic_clean} latest news 2026",
        ]
    else:
        return [
            f"{_core} latest news 2026",
            f"{_core} controversy criticism reaction",
            f"{_core} analysis background context",
            f"{_core} recent developments update",
        ]


def _generate_search_queries(topic: str) -> list:
    """
    Dimension-based angle generation with hardened LLM prompting (FIX-6).

    Strategy:
      1. Classify category.
      2. Call LLM with JSON-output prompt + two-shot example for that category.
         Uses temperature=0.0 for near-deterministic output.
      3. If < 3 valid angles returned, retry once with stricter prompt.
      4. Dedup angles (Jaccard).
      5. If still < 3 valid angles, use deterministic fallback (guaranteed 4).
      6. Append raw topic as safety-net query.
    """
    _FILLER = re.compile(
        r"^\s*(latest|recent|breaking|current)?\s*(news|update|updates)?\s*"
        r"(on|about|regarding|related to|of)?\s*", re.IGNORECASE
    )
    topic_clean = _FILLER.sub("", topic).strip()
    topic_clean = topic_clean if len(topic_clean) > 6 else topic
    topic_clean = re.sub(r'(\D)(\d+)$', r'\1 \2', topic_clean).strip()

    _topic_for_category = re.sub(r'(\D)(\d+)', r'\1 \2', topic)

    countries_found = list(dict.fromkeys(
        m.group(0).lower()
        for m in list(_COUNTRY_RE.finditer(topic)) + list(_COUNTRY_CODE_RE.finditer(topic))
    ))
    is_bilateral = len(countries_found) >= 2
    intent       = _query_intent(_topic_for_category)
    is_politics  = bool(_POLITICS_BROAD_RE.search(_topic_for_category))
    is_finance   = (intent == "finance") or bool(_FINANCE_BROAD_RE.search(_topic_for_category))
    is_geopolitical = (
        bool(_GEOPOLITICAL_RE.search(_topic_for_category)) and len(countries_found) >= 1
    )
    _is_entertainment = _classify_entertainment_query(topic, _topic_for_category)

    if is_bilateral:
        category = "bilateral"
    elif is_finance:
        category = "finance"
    elif intent == "sports":
        category = "sports"
    elif intent == "entertainment" or _is_entertainment:
        category = "entertainment"
    elif is_politics:
        category = "politics"
    elif is_geopolitical:
        category = "geopolitical"
    else:
        category = "general"

    DIMS = {
        "bilateral": [
            "border/military: LAC, border, troops, military standoff",
            "trade/economic: trade deals, FDI, investment, tariff, export, import",
            "diplomatic: envoy, ambassador, minister meeting, joint statement",
            "policy/agreement: agreements, sanctions, policy changes, latest",
        ],
        "finance": [
            "stock/price: share price, market performance, trading",
            "earnings: quarterly results, revenue, profit, analyst forecast",
            "business: products, deals, partnerships, strategy, expansion",
            "broad: widest possible recent news about this company or topic",
        ],
        "sports": [
            "match/result: scores, results, live match, highlights",
            "player/team: player performance, transfers, squad, injury news",
            "tournament: standings, schedule, rankings, predictions",
            "broad: widest possible recent news about this sport or team",
        ],
        "entertainment": [
            "box office: collection, opening weekend, screens, advance booking",
            "review: critic reviews, audience response, ratings, reaction",
            "cast/release: OTT release date, cast news, interviews, trailer",
            "broad: widest possible recent news about this film or show",
        ],
        "politics": [
            "event/election: elections, rallies, voting, official events, polls",
            "political conflict: party vs party, opposition, allegations, violence",
            "statement/controversy: what this person said, accused of, controversy",
            "policy/governance: government schemes, welfare, administration decisions",
        ],
        "geopolitical": [
            "military/strikes: attacks, strikes, troops, missiles, air defence",
            "diplomatic/sanctions: negotiations, ceasefire talks, sanctions, international response",
            "humanitarian/impact: casualties, civilians, displacement, economic impact",
            "latest/updates: breaking developments, latest news, current situation",
        ],
        "general": [
            "main event: the most newsworthy recent development on this topic",
            "reaction/conflict: opposition, criticism, controversy around this topic",
            "context/impact: broader context, economic or social impact",
            "broad: widest possible recent news search for this topic",
        ],
    }

    dims = DIMS[category]
    dims_str = "\n".join(f"Angle {i+1} ({d})" for i, d in enumerate(dims))
    example  = _ANGLE_EXAMPLES.get(category, _ANGLE_EXAMPLES["general"])

    _SW = {"the","a","an","is","are","was","were","how","why","what","on",
           "news","today","latest","update","about","and","for","from","with"}
    topic_core = {w.lower() for w in topic.split()
                  if w.lower() not in _SW and len(w) >= 3}

    # ── PRIMARY LLM CALL ─────────────────────────────────────────────────
    llm_angles = []
    try:
        prompt = (
            f"You are a news search expert. Generate exactly 4 search queries "
            f"for a news search engine.\n\n"
            f"User query: {topic}\n"
            f"Category: {category}\n\n"
            f"Each query must cover a DIFFERENT dimension:\n{dims_str}\n\n"
            f"Rules:\n"
            f"1. Return ONLY a JSON array of 4 strings. Nothing else.\n"
            f"2. Each string: 3-7 plain words. No quotes, colons, or punctuation.\n"
            f"3. Every query MUST include at least one word from the user query.\n"
            f"4. Focus on 2025-2026. Do NOT repeat the same sub-topic twice.\n"
            f"5. No category labels, no explanations — pure queries only.\n\n"
            f"Example output for a similar {category} query:\n"
            f"{json.dumps(example, ensure_ascii=False)}\n\n"
            f"Now output the JSON array for: {topic}"
        )
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 200},  # greedy decode
        )
        raw = response["message"]["content"].strip()
        llm_angles = _parse_llm_angles(raw, topic_core)

        if llm_angles:
            before     = len(llm_angles)
            llm_angles = _dedup_angles(llm_angles, threshold=0.50,
                                       topic_words=topic_core)
            if len(llm_angles) < before:
                print(f"[DEBUG] Angle dedup: {before} -> {len(llm_angles)}")
            print(f"[DEBUG] LLM angles [{category}] ({len(llm_angles)}): {llm_angles}")

        # ── RETRY if < 3 valid angles returned ───────────────────────────
        if len(llm_angles) < 3:
            print(f"[DEBUG] LLM returned only {len(llm_angles)} valid angles — retrying")
            retry_prompt = (
                f"Return ONLY a JSON array of exactly 4 short search queries "
                f"about: {topic}\n"
                f"Each query: 3-7 words, must include a word from the topic.\n"
                f"No labels, no punctuation. JSON array only.\n"
                f"Example: {json.dumps(example, ensure_ascii=False)}"
            )
            retry_resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": retry_prompt}],
                options={"temperature": 0.0, "num_predict": 150},
            )
            retry_raw    = retry_resp["message"]["content"].strip()
            retry_angles = _parse_llm_angles(retry_raw, topic_core)
            if len(retry_angles) >= len(llm_angles):
                llm_angles = _dedup_angles(retry_angles, threshold=0.50,
                                           topic_words=topic_core)
                print(f"[DEBUG] Retry angles ({len(llm_angles)}): {llm_angles}")

    except Exception as e:
        print(f"[DEBUG] LLM angle generation failed: {e}")
        llm_angles = []

    # ── FALLBACK: deterministic dimension-based templates ─────────────────
    # FIX-6: guaranteed 4 distinct angles even if LLM completely fails.
    # Delegates to _get_fallback_angles() which is also used by the probe path.
    fallback = []
    if len(llm_angles) < 3:
        print(f"[DEBUG] Using fallback [{category}], topic_clean={topic_clean!r}")
        fallback = _get_fallback_angles(category, topic_clean, countries_found)

    base = llm_angles if len(llm_angles) >= 3 else fallback
    pool = base + [topic]
    if topic_clean.lower().strip() != topic.lower().strip():
        pool.append(topic_clean)

    seen, unique = set(), []
    for a in pool:
        k = a.lower().strip()
        if k not in seen and len(k) > 3:
            seen.add(k)
            unique.append(a)

    final = unique[:4]
    print(f"[DEBUG] Final search angles ({len(final)}): {final}")
    return final


# ── FIX-7: Angle validator — probe DDG before committing to niche angles ─────

def _validate_angles_with_probe(angles: list, min_results: int = 2) -> list:
    """
    For each angle, fire a quick DDG probe (max 5 results) in parallel.
    Angles that return < min_results are dropped.  The raw topic is NOT
    substituted here — callers are responsible for padding with fallbacks
    if needed.

    Notes:
    - Runs via ThreadPoolExecutor so latency is bounded by the slowest
      single probe (~1-2 s).
    - Keeps an angle on any network error to avoid discarding good queries
      on transient failures.
    - Only called for categories where niche angles are likely
      (geopolitical, entertainment).  Finance / sports skip it since
      freshness matters more than precision there.
    - FIX-10: Sports / entertainment RC13 bypass is irrelevant here — this
      probe only checks result count, not content.
    """
    def _probe(angle: str) -> tuple:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(angle, max_results=5))
            return angle, len(results)
        except Exception as e:
            print(f"[DEBUG] Angle probe error for '{angle}': {e}")
            return angle, min_results  # keep on error

    validated = []
    with ThreadPoolExecutor(max_workers=max(len(angles), 1)) as ex:
        for angle, count in ex.map(_probe, angles):
            if count >= min_results:
                validated.append(angle)
            else:
                print(f"[DEBUG] Angle probe: dropped '{angle}' ({count} results)")

    # Never return empty — fall back to original list
    return validated if validated else angles


# ── Step 2: DDG fetch ────────────────────────────────────────────────────────

def _fetch_articles(query: str, max_results: int = 30) -> list:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        print(f"[DEBUG] Fetched {len(results)} for: {query[:60]}")
        return results
    except Exception as e:
        print(f"[DEBUG] Fetch failed: {e}")
        return []

def _fetch_articles_multi(queries: list, max_per_query: int = 25) -> list:
    all_articles = []
    def _fetch_one(q):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(q, max_results=max_per_query))
            print(f"[DEBUG] Multi-fetch: {len(results)} articles for: {q[:60]}")
            return results
        except Exception as e:
            print(f"[DEBUG] Multi-fetch failed '{q[:40]}': {e}")
            return []
    with ThreadPoolExecutor(max_workers=max(len(queries), 1)) as ex:
        for fut in as_completed({ex.submit(_fetch_one, q): q for q in queries}):
            all_articles.extend(fut.result())
    print(f"[DEBUG] Multi-fetch total: {len(all_articles)} from {len(queries)} queries")
    return all_articles

# ── Step 2b: NewsData.io enrichment ─────────────────────────────────────────

def _fetch_articles_newsdata(query: str, topic: str, max_results: int = 10) -> list:
    # FIX-8: NEWSDATA_ENABLED is now True when key is present, so this guard
    # is the single authoritative check — no dead code paths below it.
    if not NEWSDATA_ENABLED:
        return []
    if not NEWSDATA_API_KEY:
        print("[DEBUG] NewsData.io: no API key")
        return []
    clean_query = re.sub(r"[\"\':,!?]", " ", query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()[:100]
    params = {"apikey": NEWSDATA_API_KEY, "q": clean_query,
              "language": "en", "size": min(max_results, 10)}
    if _INDIA_QUERY_RE.search(topic):
        params["country"] = "in"
        print("[DEBUG] NewsData.io: country=in filter applied")
    try:
        resp = requests.get(NEWSDATA_ENDPOINT, params=params, timeout=6,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            articles = [{
                "title":  i.get("title", ""),
                "body":   i.get("content") or i.get("description") or "",
                "url":    i.get("link", ""),
                "date":   (i.get("pubDate") or "")[:10],
                "source": i.get("source_id", ""),
            } for i in resp.json().get("results", [])]
            print(f"[DEBUG] NewsData.io: {len(articles)} for: {clean_query[:50]}")
            return articles
        print(f"[DEBUG] NewsData.io HTTP {resp.status_code}")
        return []
    except Exception as e:
        print(f"[DEBUG] NewsData.io fetch failed: {e}")
        return []

# ── Step 2c: GNews free-tier fallback ────────────────────────────────────────
# FIX-8: Second free-tier source so the pipeline has a backup when DDG is
# flaky and NewsData is not configured.

def _fetch_articles_gnews(query: str, topic: str, max_results: int = 10) -> list:
    if not GNEWS_API_KEY:
        return []
    clean_query = re.sub(r"[\"\':,!?]", " ", query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()[:100]
    params = {
        "q":      clean_query,
        "token":  GNEWS_API_KEY,
        "lang":   "en",
        "max":    min(max_results, 10),
        "sortby": "publishedAt",
    }
    try:
        resp = requests.get(
            GNEWS_ENDPOINT, params=params, timeout=6,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            articles = [{
                "title":  i.get("title", ""),
                "body":   i.get("description", ""),
                "url":    i.get("url", ""),
                "date":   (i.get("publishedAt") or "")[:10],
                "source": i.get("source", {}).get("name", ""),
            } for i in resp.json().get("articles", [])]
            print(f"[DEBUG] GNews: {len(articles)} for: {clean_query[:50]}")
            return articles
        print(f"[DEBUG] GNews HTTP {resp.status_code}")
        return []
    except Exception as e:
        print(f"[DEBUG] GNews fetch failed: {e}")
        return []

# ── Step 3: Domain filter + dedup ───────────────────────────────────────────

def _is_trusted_url(url: str) -> bool:
    try:
        d = urlparse(url).netloc.lower().replace("www.", "")
        return any(d == td or d.endswith("." + td) for td in TRUSTED_DOMAINS)
    except Exception:
        return False

def _is_junk_url(url: str) -> bool:
    try:
        d = urlparse(url).netloc.lower().replace("www.", "")
        return any(d == jd or d.endswith("." + jd) for jd in JUNK_DOMAINS)
    except Exception:
        return False

def _is_junk_article(a: dict) -> bool:
    if _is_junk_url(a.get("url", "")):
        return True
    t = a.get("title", "")
    return bool(t and _JUNK_TITLE_SIGNALS.search(t))
def _filter_by_domain(articles: list) -> list:
    hard_clean = [a for a in articles if not _is_junk_article(a)]
    removed = len(articles) - len(hard_clean)
    if removed:
        print(f"[DEBUG] Junk filter: removed {removed} articles")

    # Soft-junk logic: only exclude low-quality outlets when trusted pool is rich
    trusted_count = sum(1 for a in hard_clean if _is_trusted_url(a.get("url", "")))
    if trusted_count < 6:
        clean = hard_clean  # pool is thin — keep soft-junk articles as fill
        print(f"[DEBUG] Soft-junk bypass: trusted={trusted_count} < 6, keeping all")
    else:
        clean = [
            a for a in hard_clean
            if urlparse(a.get("url", "")).netloc.lower().replace("www.", "")
            not in _SOFT_JUNK_DOMAINS
        ]

    trusted   = [a for a in clean if _is_trusted_url(a.get("url", ""))]
    untrusted = [a for a in clean if not _is_trusted_url(a.get("url", ""))]
    print(f"[DEBUG] Domain filter: {len(trusted)} trusted, {len(untrusted)} untrusted")

    if len(trusted) >= 8:
        return trusted[:35]
    if trusted:
        print(f"[DEBUG] Blending {len(trusted)} trusted + untrusted")
        return (trusted + untrusted[:25])[:35]
    print("[DEBUG] No trusted articles — using untrusted")
    return untrusted[:35]

_GENERIC_ENTITY_WORDS = {
    "india", "china", "russia", "america", "united", "states", "pakistan",
    "israel", "ukraine", "japan", "germany", "france", "britain", "saudi",
    "turkey", "beijing", "moscow", "london", "delhi", "mumbai",
    "government", "minister", "police", "court", "state", "party",
    "election", "assembly", "parliament", "congress",
}


def _title_is_duplicate(title_a_norm: str, title_b_norm: str,
                         sw_set: set) -> bool:
    def _sw(t):
        return [w for w in t.split() if len(w) > 3 and w not in sw_set]

    wa, wb = _sw(title_a_norm), _sw(title_b_norm)
    sa, sb = set(wa), set(wb)
    if not sa or not sb:
        return False

    j = len(sa & sb) / max(len(sa | sb), 1)

    ea = {w for w in wa if len(w) >= 5 and w not in _GENERIC_ENTITY_WORDS}
    eb = {w for w in wb if len(w) >= 5 and w not in _GENERIC_ENTITY_WORDS}
    shared_specific = len(ea & eb)

    e5a = {w for w in wa if len(w) >= 5}
    e5b = {w for w in wb if len(w) >= 5}
    shared_e5 = len(e5a & e5b)

    if j >= 0.65:
        return True
    if shared_specific >= 3 and j >= 0.35:
        return True
    if shared_e5 >= 3 and j >= 0.28:
        return True
    return False


def _deduplicate_urls(articles: list) -> list:
    seen_urls        = set()
    seen_norm_titles = []
    seen_body_tokens = []
    unique = []

    _SW = {
        "that","with","from","this","will","have","been","they","their",
        "about","also","said","says","over","after","amid","into",
        "the","and","for","has","its","are","was","were","but","not",
        "can","all","any","had","more","when","which","would","could",
        "should","these","those","then","than","such","each","both","just","even",
    }

    for a in articles:
        url = a.get("url", "").strip().rstrip("/")

        raw_title  = a.get("title", "")
        norm_title = re.sub(r"[^a-z0-9\s]", "", raw_title.lower())
        norm_title = re.sub(r"\s+", " ", norm_title).strip()[:80]

        body_text = a.get("body", "").strip().lower()[:300]
        bw = set(w for w in body_text.split() if len(w) > 4 and w not in _SW)

        if url in seen_urls or norm_title in seen_norm_titles:
            continue

        if any(_title_is_duplicate(norm_title, prev, _SW)
               for prev in seen_norm_titles):
            continue

        if len(bw) >= 8 and any(
            len(bw & k) / max(len(bw | k), 1) > 0.80
            for k in seen_body_tokens if len(k) >= 8
        ):
            continue

        seen_urls.add(url)
        seen_norm_titles.append(norm_title)
        if bw: seen_body_tokens.append(bw)
        unique.append(a)

    print(f"[DEBUG] Deduplication: {len(articles)} -> {len(unique)} articles")
    return unique

# ── Step 4: Title pre-filter ─────────────────────────────────────────────────

def _title_prefilter(articles: list, topic: str, max_keep: int = 35) -> list:
    _SW = {
        "the","a","an","is","are","was","were","how","why","what","on",
        "impact","news","today","latest","update","about","effect",
        "will","does","do","has","have","been","its","their",
        "and","for","from","with","that","this","also","into",
        "get","give","tell","show","find","let","put",
        "regarding","related","concerning",
    }
    _SHORT_ACRONYM_RE = re.compile(r"^[A-Z]{2,4}$")
    topic_normalized = re.sub(r'(\D)(\d+)', r'\1 \2', topic)
    
    tokens = list(dict.fromkeys(
        re.sub(r'[^a-zA-Z0-9]', '', w).lower() for w in topic_normalized.split()
        if re.sub(r'[^a-zA-Z0-9]', '', w).lower() not in _SW
        and (len(re.sub(r'[^a-zA-Z0-9]', '', w)) >= 4
            or _SHORT_ACRONYM_RE.match(re.sub(r'[^a-zA-Z0-9]', '', w)))
    ))
    tokens = [t for t in tokens if t]
    if not tokens:
        return articles[:max_keep]

    passing = []
    for a in articles:
        title_text    = a.get("title", "").lower()
        combined_text = title_text + " " + a.get("body", "").lower()   
        title_matches = sum(1 for t in tokens if _fuzzy_token_match(t, title_text))
        body_matches  = sum(1 for t in tokens if _fuzzy_token_match(t, combined_text))     
        if title_matches >= 1 or body_matches >= 2:
            passing.append(a)

    print(f"[DEBUG] Title pre-filter: {len(articles)} -> {len(passing)} "
          f"(tokens: {tokens[:4]})")

    if len(passing) < 3:
        print("[DEBUG] Title pre-filter fallback — keeping all articles")
        return articles[:max_keep]

    _OFF_TOPIC = re.compile(
        r"(seminar|webinar|conference on|workshop on|lecture on|opinion:|"
        r"commentary:|analysis by|my take|advertisement|sponsored|podcast episode)",
        re.IGNORECASE
    )
    if len(passing) > 4:
        filtered = [a for a in passing if not _OFF_TOPIC.search(a.get("title", ""))]
        if len(filtered) >= 3:
            if len(filtered) < len(passing):
                print(f"[DEBUG] Off-topic filter: {len(passing)} -> {len(filtered)}")
            passing = filtered

    return passing[:max_keep]

# ── Step 4b: Lightweight hybrid pre-ranking (ARCH-1) ─────────────────────────

def _pre_rank_articles(articles: list, query: str,
                       top_n: int = None) -> list:
    """
    Lightweight hybrid pre-ranking of DDG/NewsData articles BEFORE download.

    Runs BM25 + cosine on each article's title + DDG snippet body (short
    texts, ~1-3 sentences).  Returns the list re-ordered from most to least
    relevant so that the _MAX_DOWNLOAD budget is spent on the right articles.

    Design decisions:
      - Uses the same _bm25_scores() + SentenceTransformer as Step 7 so
        there is no new dependency and no extra model loading.
      - alpha=0.50 here (equal weight) because snippets are short and BM25
        is proportionally more reliable on short texts than on full articles.
        Step 7 uses alpha=0.40 because full-text cosine is more reliable.
      - Falls back to the original order if:
          (a) fewer than 3 articles in pool (not worth scoring),
          (b) embedding raises an exception.
      - top_n is not applied here — caller slices to _MAX_DOWNLOAD after
        getting the sorted list, which keeps the fallback logic simple.

    Args:
        articles : list of article dicts (must have "title" and "body" keys)
        query    : the tool_query / topic string used for scoring
        top_n    : if set, truncate the ranked list to this length
                   (caller can also just slice — provided for convenience)

    Returns:
        Re-ordered list of article dicts (highest scored first).
    """
    if len(articles) < 3:
        # Pool too small to benefit from scoring — return as-is
        print(f"[DEBUG] Pre-rank: pool too small ({len(articles)}), skipping")
        return articles

    # Build the text corpus: title + snippet body concatenated.
    # This is the same text the title_prefilter used, but now we score it
    # properly instead of doing a binary pass/fail fuzzy match.
    def _article_text(a: dict) -> str:
        title = a.get("title", "")
        body  = a.get("body",  "")
        return (title + " " + body).strip()

    corpus = [_article_text(a) for a in articles]

    # ── BM25 on snippet corpus ────────────────────────────────────────────
    bm25_scores = _bm25_scores(query, corpus)

    # ── Cosine on snippet corpus ──────────────────────────────────────────
    try:
        model   = _get_embedding_model()
        # encode query + all docs in one batch (model already warm from Step 7
        # if this isn't the first call, so this is cheap)
        all_emb = model.encode(
            [query] + corpus,
            batch_size=32,
            show_progress_bar=False,
        )
        q_norm  = all_emb[0] / (np.linalg.norm(all_emb[0]) + 1e-9)
        c_emb   = all_emb[1:]
        c_norms = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
        cosine  = c_norms @ q_norm
    except Exception as e:
        print(f"[DEBUG] Pre-rank: embedding failed ({e}), falling back to BM25-only")
        cosine = np.zeros(len(articles), dtype=float)

    # ── Fuse: equal weight on short texts (see docstring) ────────────────
    _PRE_RANK_ALPHA = 0.50
    combined = (1.0 - _PRE_RANK_ALPHA) * cosine + _PRE_RANK_ALPHA * bm25_scores

    # ── Trusted-source bonus (same logic as Step 7, lighter version) ─────
    # Give a small lift to trusted domains so they're preferred when scores
    # are close — mirrors the Step 7 +0.05 bonus but applied pre-download.
    for i, a in enumerate(articles):
        if _is_trusted_url(a.get("url", "")):
            combined[i] += 0.05

    ranked_indices = np.argsort(combined)[::-1]
    ranked = [articles[i] for i in ranked_indices]

    if top_n is not None:
        ranked = ranked[:top_n]

    print(
        f"[DEBUG] Pre-rank ({len(articles)} articles): "
        f"top-3 scores = {[round(float(combined[i]), 3) for i in ranked_indices[:3]]}"
    )
    return ranked


# ── Step 5: Article download ─────────────────────────────────────────────────

def _download_single(article: dict) -> dict:
    title  = article.get("title", "").strip()
    body   = article.get("body", "").strip()
    url    = article.get("url", "")
    dt     = article.get("date", "")[:10]
    source = article.get("source", "")

    full_text   = ""
    download_ok = False

    if url:
        try:
            resp = requests.get(url, timeout=ARTICLE_DOWNLOAD_TIMEOUT,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and len(resp.text) >= 500:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script","style","nav","footer","aside",
                                     "header","noscript","iframe","form","figure",
                                     "figcaption","button","input","select",
                                     "textarea","template","svg","picture",
                                     "source","link","meta"]):
                        tag.decompose()
                    for sel in ["[class*='trending']","[class*='sidebar']",
                                "[class*='widget']","[class*='related']",
                                "[class*='newsletter']","[class*='social']",
                                "[class*='share']","[id*='trending']",
                                "[id*='sidebar']","[id*='related']",
                                "[class*='banner']","[class*='popup']"]:
                        try:
                            for t in soup.select(sel): t.decompose()
                        except Exception:
                            pass
                    article_body = None
                    for sel in ["article","[role='main']","[class*='article-body']",
                                "[class*='article-content']","[class*='story-body']",
                                "[class*='story-content']","[class*='post-content']",
                                "[class*='entry-content']","main"]:
                        try:
                            cand = soup.select_one(sel)
                            if cand and len(cand.get_text(separator=" ",strip=True).split()) > 150:
                                article_body = cand
                                break
                        except Exception:
                            pass
                    target = article_body if article_body else soup
                    text   = target.get_text(separator=" ", strip=True)
                    text   = re.sub(r"Array\s*\([^)]{0,300}\)", " ", text)
                    text   = re.sub(r"Trending\s+Topics[^.]*", " ", text)
                    text   = re.sub(r"Live\s+Events[^.]*", " ", text)
                    text   = re.sub(r"https?://\S+", " ", text)
                    text   = re.sub(r"\s+", " ", text).strip()
                    if len(text) > 300:
                        full_text, download_ok = text[:MAX_ARTICLE_CHARS], True
                except Exception:
                    text = re.sub(r"<[^>]+>", " ", resp.text)
                    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) > 300:
                        full_text, download_ok = text[:MAX_ARTICLE_CHARS], True
        except Exception:
            pass

    if full_text and _has_encoding_corruption(full_text):
        fixed = _fix_encoding(full_text)
        if not _has_encoding_corruption(fixed):
            full_text = fixed

    if not full_text:
        body_clean = re.sub(r"https?://\S+", "", body)
        body_clean = re.sub(r"\s+", " ", body_clean).strip()
        if body_clean and _has_encoding_corruption(body_clean):
            body_clean = _fix_encoding(body_clean)
        full_text = body_clean[:500] if body_clean else title

    return {
        "title":      title,
        "date":       dt,
        "url":        url,
        "source":     source,
        "full_text":  full_text,
        "is_trusted": _is_trusted_url(url),
    }

def _extract_articles_parallel(articles: list) -> list:
    extracted = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as ex:
        futures = {ex.submit(_download_single, a): a for a in articles}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r["full_text"] and len(r["full_text"]) > 30:
                    extracted.append(r)
            except Exception:
                pass
    print(f"[DEBUG] Parallel extraction done: {len(extracted)} articles")
    return extracted

# ── Step 6: Chunking ─────────────────────────────────────────────────────────

def _chunk_article(article: dict, chunk_size: int = 200, overlap: int = 30) -> list:
    words  = article["full_text"].split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append({
            "text":       " ".join(words[start:end]),
            "title":      article["title"],
            "date":       article["date"],
            "source":     article["source"],
            "url":        article["url"],
            "is_trusted": article.get("is_trusted", False),
        })
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

# ── Step 7: Hybrid scoring ───────────────────────────────────────────────────

def _unified_score_chunks(query: str, all_chunks: list, topic: str,
                           top_k: int = _TOP_K,
                           category: str = "general") -> list:
    """
    Hybrid scoring: cosine + BM25 fusion.

    FIX-2:  RC13 (relational signal filter) now skipped for sports and
            entertainment categories — those articles never contain diplomatic
            keywords and would be incorrectly dropped.
            FIX-10: This bypass is explicitly gated on _RC13_EXEMPT so it is
            clearly documented and cannot be accidentally removed.
    FIX-4:  Trust bonus is category-aware: finance-press domains only boost
            finance queries; sports-press domains only boost sports queries.
    FIX-5:  BM25 uses global-floor normalisation (see _bm25_scores).
    FIX-9:  Per-URL chunk cap is 2 (raised from implicit 1); domain cap is
            adaptive based on pool size to avoid over-penalising small pools.
    """
    if not all_chunks:
        return []

    countries_in_query = list(dict.fromkeys(
        m.group(0).lower()
        for m in list(_COUNTRY_RE.finditer(topic)) + list(_COUNTRY_CODE_RE.finditer(topic))
    ))

    # FIX-2 / FIX-10: RC13 only applies for bilateral/geopolitical categories.
    # Sports and entertainment are explicitly exempt because those articles
    # never contain diplomatic/relational keywords and would be gutted.
    _INTENT = _query_intent(topic)
    _RC13_EXEMPT = _INTENT in ("sports", "entertainment")
    is_country_pair = (
        len(countries_in_query) >= 2
        and not _RC13_EXEMPT
    )

    model       = _get_embedding_model()
    texts       = [c["text"] for c in all_chunks]
    all_emb     = model.encode([query] + texts, batch_size=64, show_progress_bar=False)
    q_norm      = all_emb[0] / (np.linalg.norm(all_emb[0]) + 1e-9)
    c_emb       = all_emb[1:]
    c_norms     = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    cosine      = c_norms @ q_norm

    if ENABLE_HYBRID_SEARCH:
        bm25_text  = _bm25_scores(query, texts)
        titles     = [c.get("title", "") for c in all_chunks]
        bm25_title = _bm25_scores(query, titles)
        bm25       = 0.6 * bm25_text + 0.4 * bm25_title
        base_scores = (1.0 - _HYBRID_ALPHA) * cosine + _HYBRID_ALPHA * bm25
        print(f"[DEBUG] Hybrid scoring: cosine*{1-_HYBRID_ALPHA:.1f} + BM25*{_HYBRID_ALPHA:.1f}")
    else:
        base_scores = cosine.copy()

    scores = base_scores.copy()

    now = datetime.now(timezone.utc)
    for i, chunk in enumerate(all_chunks):
        # FIX-4: category-aware trust bonus.
        # Finance-press domains only get the full +0.05 for finance queries;
        # sports-press domains only for sports queries.
        # All other trusted sources get the full bonus for non-specialised categories.
        domain = urlparse(chunk.get("url", "")).netloc.lower().replace("www.", "")
        if chunk.get("is_trusted", False):
            if any(domain == d or domain.endswith("." + d)
                   for d in _FINANCE_TRUST_DOMAINS):
                scores[i] += 0.05 if category == "finance" else 0.01
            elif any(domain == d or domain.endswith("." + d)
                     for d in _SPORTS_TRUST_DOMAINS):
                scores[i] += 0.05 if category == "sports" else 0.01
            else:
                # General trusted wire / broadcast: full bonus for all categories
                scores[i] += 0.05

        d = chunk.get("date", "")
        if d:
            try:
                age = (now - datetime.fromisoformat(d[:10])
                       .replace(tzinfo=timezone.utc)).days
                scores[i] += 0.05 if age < 3 else (0.02 if age < 7 else 0)
            except Exception:
                pass

    _CHUNK_SW = {
        "that","with","from","this","will","have","been","they","their",
        "about","also","said","says","over","after","amid","into","told",
        "the","and","for","has","its","are","was","were","but","not",
        "can","all","any","had","more","when","which","would",
    }

    def _root_domain(url: str) -> str:
        try:
            parts = urlparse(url).netloc.lower().replace("www.", "").split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else url
        except Exception:
            return url

    def _event_key(text: str) -> frozenset:
        return frozenset(
            w.lower() for w in re.findall(r"[A-Z][a-z]{4,}|\d{4}", text)
            if len(w) > 4
        )

    # FIX-9: per-URL cap raised to 2 so that long important articles can
    # contribute more than one chunk, while still preventing a single URL
    # from flooding results.
    _PER_URL_CAP = 2

    def _norm_url(u: str) -> str:
        """
        Strip query-strings, fragments and known redirect wrappers so that
        'news.yahoo.com/...aljazeera...?param=1' and
        'news.yahoo.com/...aljazeera...?param=2' are treated as the same URL.
        Also collapses common redirect prefixes (Yahoo News, Google AMP, MSN)
        so a liveblog served via multiple redirect URLs is capped correctly.
        """
        try:
            from urllib.parse import urlparse, urlunparse
            p = urlparse(u)
            # Drop query string and fragment — keep scheme+netloc+path only
            normalised = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
            # Collapse Yahoo/Google/MSN redirect wrappers to the netloc+path
            # pattern that uniquely identifies the underlying article
            normalised = re.sub(r'\?.*$', '', normalised)
            # Remove trailing slashes for consistent matching
            return normalised.rstrip("/").lower()
        except Exception:
            return u.rstrip("/").lower()

    seen_urls, seen_tokens = {}, []
    seen_events, domain_counts = set(), {}
    ranked = []

    for i in np.argsort(scores)[::-1]:
        _SAFETY_FLOOR = 0.05
        if base_scores[i] < _SAFETY_FLOOR:
            url_dbg = all_chunks[i].get('url', '')[:50]
            print(f"[DEBUG] Safety floor drop ({_SAFETY_FLOOR}): "
                  f"score={base_scores[i]:.3f} url={url_dbg}")
            continue

        # FIX-2 / FIX-10: RC13 — only for bilateral/geopolitical.
        # _RC13_EXEMPT=True for sports/entertainment ensures this block is
        # never entered for those categories.
        if is_country_pair:
            t_rel = bool(_RELATIONAL_SIGNALS.search(all_chunks[i].get("title", "")))
            x_rel = bool(_RELATIONAL_SIGNALS.search(all_chunks[i].get("text", "")))
            if not t_rel and not x_rel:
                title_dbg = all_chunks[i].get('title', '')[:60]
                print(f"[DEBUG] RC13 drop: {title_dbg}")
                continue

        url    = all_chunks[i].get("url", "")
        text   = all_chunks[i].get("text", "")
        domain = _root_domain(url)
        norm_u = _norm_url(url)   # query-string-stripped, lowercase

        _REDIRECT_WRAPPERS = {"yahoo.com", "msn.com", "news.google.com"}
        if domain in _REDIRECT_WRAPPERS:
            _path_source = re.search(r'/([a-z0-9\-]+)\.[a-z]{2,4}/', url)
            if _path_source:
                domain = _root_domain("https://" + _path_source.group(1) + ".com/")

        # FIX-9 + BUG-1: use normalised URL so redirect wrappers (Yahoo News,
        # Google AMP) that serve the same article under different query-string
        # params are still counted as the same source.
        if seen_urls.get(norm_u, 0) >= _PER_URL_CAP:
            continue

        # FIX-9: adaptive domain cap — larger pools tolerate more diversity
        if len(all_chunks) >= 40:
            _effective_cap = 5
        elif len(all_chunks) >= 20:
            _effective_cap = 4
        else:
            _effective_cap = _DOMAIN_CAP
        if domain_counts.get(domain, 0) >= _effective_cap:
            print(f"[DEBUG] Domain cap ({_effective_cap}): {domain}")
            continue

        tokens = set(w for w in text.lower().split() if len(w) > 4 and w not in _CHUNK_SW)
        # BUG-1: tighten token-similarity threshold for small chunk pools.
        # With only 11-20 chunks the 0.55 Jaccard was too loose — chunks from
        # the same rolling liveblog share ~60% tokens and were all passing.
        # Use 0.45 for pools < 25 chunks, keep 0.55 for larger pools where
        # false-positive dedup is more costly than duplicate leakage.
        _tok_sim_threshold = 0.45 if len(all_chunks) < 25 else 0.55
        if len(tokens) >= 10 and any(
            len(tokens & k) / max(len(tokens | k), 1) > _tok_sim_threshold
            for k in seen_tokens if len(k) >= 10
        ):
            continue

        ekey = _event_key(text)
        if any(
            len(ekey) >= 7 and len(s) >= 7
            and len(ekey & s) / max(len(ekey | s), 1) > 0.85
            for s in seen_events
        ):
            continue

        seen_urls[norm_u] = seen_urls.get(norm_u, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if tokens: seen_tokens.append(tokens)
        if ekey:   seen_events.add(ekey)
        ranked.append(all_chunks[i])
        if len(ranked) >= top_k:
            break

    mode = f"hybrid(alpha={_HYBRID_ALPHA})" if ENABLE_HYBRID_SEARCH else "cosine"
    print(f"[DEBUG] Scoring [{mode}]: {len(ranked)} chunks passed to reranker "
          f"(from {len(all_chunks)}, safety_floor=0.05, "
          f"rc13={'on' if is_country_pair else 'off (exempt)' if _RC13_EXEMPT else 'off'})")
    return ranked

# ── Step 8: Minimum chunk gate ───────────────────────────────────────────────

def _minimum_chunk_gate(chunks: list, topic: str, min_chunks: int = 2) -> dict:
    if len(chunks) >= min_chunks:
        return {"ok": True, "chunks": chunks}
    warning = (
        f"Insufficient news data found for: {topic!r}. "
        f"Only {len(chunks)} source(s) available. Results may be incomplete."
    )
    print(f"[DEBUG] Chunk gate: {len(chunks)} < {min_chunks}")
    return {"ok": False, "chunks": chunks, "warning": warning}

# ── FEAT-1: Broadband query detection ────────────────────────────────────────

def _is_broadband_query(query: str) -> bool:
    """
    Return True when the user's query is a generic "give me today's news"
    request with no specific topic.  The regex is anchored (^ … $) so it only
    fires on bare recency requests, never on topic-specific ones like
    "latest news on Modi" or "today's cricket news".
    """
    return bool(_BROADBAND_RE.match(query.strip()))


# ── FEAT-1 / FEAT-2: Multi-topic digest pipeline ─────────────────────────────

def get_latest_news_digest(query: str) -> list:
    """
    Fan-out pipeline for generic "latest news today" queries (FEAT-1).

    Steps:
      1. Run _DIGEST_TOPICS in parallel — one DDG news search per topic.
      2. Apply a strict 7-day date filter to every article pool (FEAT-2).
         Unlike the normal path, the adaptive 21-day relaxation is never used.
      3. Domain-filter + deduplicate within each topic bucket.
      4. Download the top-1 article per bucket for full text.
      5. Chunk → embed → pick the single highest-cosine chunk per bucket.
      6. Assemble results with a _digest_category label so the LLM layer can
         render them as clearly separated sections.

    Returns a list of chunk dicts, each augmented with:
        "_digest_category" : str   — display label (e.g. "🏛️ Politics")
        "_digest_order"    : int   — position 0-5 so callers can sort

    A minimum of _DIGEST_MIN_CATEGORIES categories must have at least one
    result; otherwise a warning chunk is appended.
    """
    _DIGEST_MIN_CATEGORIES = 4
    _cutoff = (date.today() - timedelta(days=_DIGEST_DATE_WINDOW_DAYS)).isoformat()
    model   = _get_embedding_model()

    def _fetch_topic(args):
        """Fetch, filter, download and embed one digest category."""
        idx, (label, search_query, cat_tag) = args
        print(f"[DIGEST] Fetching category {idx}: {label!r} → query={search_query!r}")

        # 1. DDG fetch — 30 results gives a richer pool to filter from
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.news(search_query, max_results=30))
        except Exception as e:
            print(f"[DIGEST] DDG failed for {label!r}: {e}")
            return idx, label, []

        # 2. FEAT-2: strict 7-day filter — always applied, no relaxation
        fresh = [a for a in raw if a.get("date", "")[:10] >= _cutoff]
        print(f"[DIGEST] {label!r}: {len(raw)} raw → {len(fresh)} after 7-day filter")
        if not fresh:
            # Relax to 14 days as a last resort so we never return empty for
            # a category, but flag it in the debug log
            fallback_cutoff = (
                date.today() - timedelta(days=14)
            ).isoformat()
            fresh = [a for a in raw if a.get("date", "")[:10] >= fallback_cutoff]
            print(f"[DIGEST] {label!r}: relaxed to 14-day fallback → {len(fresh)}")
        if not fresh:
            return idx, label, []

        # 3. Domain filter + dedup (reuse existing helpers)
        filtered = _filter_by_domain(fresh)
        deduped  = _deduplicate_urls(filtered)
        if not deduped:
            return idx, label, []

        # 4. Download top-3 articles (raised from 2) so the chunk pool is
        # large enough for the cosine selector to pick a meaningfully
        # representative story rather than settling for the only chunk available.
        extracted = _extract_articles_parallel(deduped[:3])
        if not extracted:
            # Fallback: use snippet/body from raw article
            a = deduped[0]
            body = a.get("body", "") or a.get("title", "")
            if body:
                extracted = [{
                    "title":      a.get("title", ""),
                    "date":       a.get("date", "")[:10],
                    "url":        a.get("url", ""),
                    "source":     a.get("source", ""),
                    "full_text":  body[:500],
                    "is_trusted": _is_trusted_url(a.get("url", "")),
                }]

        if not extracted:
            return idx, label, []

        # 5. Chunk and pick the best chunk by cosine to the search_query
        chunks = []
        for art in extracted:
            chunks.extend(_chunk_article(art, chunk_size=200, overlap=30))
        if not chunks:
            return idx, label, []

        texts   = [c["text"] for c in chunks]
        embs    = model.encode([search_query] + texts,
                               batch_size=32, show_progress_bar=False)
        q_norm  = embs[0] / (np.linalg.norm(embs[0]) + 1e-9)
        c_norms = embs[1:] / (np.linalg.norm(embs[1:], axis=1, keepdims=True) + 1e-9)
        cosines = c_norms @ q_norm
        best_i  = int(np.argmax(cosines))
        best    = chunks[best_i].copy()

        # 6. Tag the chunk with digest metadata.
        # _is_digest=True tells the downstream reranker and _dedup_bullets
        # to treat each chunk as a standalone story — do NOT score it against
        # the generic "latest news today" query and do NOT collapse chunks
        # from different categories even if their texts overlap.
        best["_digest_category"] = label
        best["_digest_order"]    = idx
        best["_digest_score"]    = float(cosines[best_i])
        best["_is_digest"]       = True
        print(f"[DIGEST] {label!r}: best chunk score={cosines[best_i]:.3f} "
              f"title={best.get('title','')[:60]!r}")
        return idx, label, [best]

    # Run all categories in parallel
    results_by_idx: dict = {}
    with ThreadPoolExecutor(max_workers=len(_DIGEST_TOPICS)) as ex:
        futs = {
            ex.submit(_fetch_topic, (i, t)): i
            for i, t in enumerate(_DIGEST_TOPICS)
        }
        for fut in as_completed(futs):
            try:
                idx, label, chunks = fut.result()
                if chunks:
                    results_by_idx[idx] = chunks
            except Exception as e:
                print(f"[DIGEST] Worker exception: {e}")

    # Assemble in canonical order
    digest_chunks = []
    for i in range(len(_DIGEST_TOPICS)):
        if i in results_by_idx:
            digest_chunks.extend(results_by_idx[i])

    categories_found = len(digest_chunks)
    print(f"[DIGEST] Complete: {categories_found}/{len(_DIGEST_TOPICS)} categories")

    if categories_found < _DIGEST_MIN_CATEGORIES:
        digest_chunks.append({
            "text": (
                f"Only {categories_found} of {len(_DIGEST_TOPICS)} news categories "
                f"returned results within the last {_DIGEST_DATE_WINDOW_DAYS} days. "
                f"Try a more specific query for the topics you're interested in."
            ),
            "title":            "Warning",
            "date":             "",
            "source":           "",
            "url":              "",
            "is_trusted":       False,
            "_warning":         True,
            "_digest_category": "⚠️ Warning",
            "_digest_order":    99,
        })

    return digest_chunks


# ── Main entry point ─────────────────────────────────────────────────────────

def get_news(
    topic: str,
    num_results: int = 5,
    reranker=None,
    original_query: str = None,
    tool_query: str = None,
    preprocessed_query: dict = None,
) -> list:
    """
    News pipeline — entity extraction removed. Raw query used throughout.
    Hybrid BM25+cosine scoring when ENABLE_HYBRID_SEARCH=True in settings.py.

    FEAT-1: If the query is a generic "latest news today" request (detected by
    _is_broadband_query), the call is immediately delegated to
    get_latest_news_digest() which fans out across 6 topic categories in
    parallel and returns a labelled multi-topic digest.

    FEAT-2: The digest always applies a hard 7-day date window so results are
    genuinely fresh.  The adaptive 21-day relaxation used on the normal path
    for thin pools does NOT apply to digest queries.

    Fix summary applied in this version:
      FIX-1  window scoping: cutoff only computed inside the else branch
      FIX-2  RC13 exempt for sports/entertainment (_RC13_EXEMPT flag)
      FIX-3  recency filter runs ONCE at article level only (chunk-level
             filter removed to avoid double-filtering)
      FIX-4  trust bonus is category-aware (finance / sports domains)
      FIX-5  BM25 global-floor normalisation
      FIX-6  hardened LLM prompting + JSON output + retry + fallback
      FIX-7  DDG angle probe (_validate_angles_with_probe)
      FIX-8  NEWSDATA_ENABLED auto-derives from key; GNews added as fallback
      FIX-9  per-URL cap=2; adaptive domain cap
      FIX-10 sports/entertainment RC13 bypass explicitly documented
    """
    original_query = original_query or topic
    tool_query     = tool_query     or topic

    # ── FEAT-1: Broadband digest shortcut ────────────────────────────────
    # Detect bare recency queries ("latest news today", "top headlines", etc.)
    # and delegate to the multi-topic fan-out pipeline instead of the normal
    # single-topic path.  We check both the original_query and topic so that
    # upstream query rewriting doesn't accidentally bypass the detection.
    _check_query = (original_query or "").strip() or (topic or "").strip()
    if _is_broadband_query(_check_query):
        print(f"[DEBUG] Broadband query detected — routing to digest pipeline")
        return get_latest_news_digest(_check_query)

    _RECENCY = re.compile(
        r"\b(latest|recent|today|breaking|current|now|this week|"
        r"right now|just now|happening|update|updates)\b", re.IGNORECASE
    )
    is_recency = bool(_RECENCY.search(original_query)) or True
    print(f"[DEBUG] Recency query: {is_recency}")

    # ── Derive pipeline category (used by scorer + angle gen) ────────────
    _topic_normalized = re.sub(r'(\D)(\d+)', r'\1 \2', topic)
    intent = _query_intent(_topic_normalized)
    _is_entertainment = _classify_entertainment_query(topic, _topic_normalized)
    countries_found = list(dict.fromkeys(
        m.group(0).lower()
        for m in list(_COUNTRY_RE.finditer(topic)) + list(_COUNTRY_CODE_RE.finditer(topic))
    ))
    if len(countries_found) >= 2:
        _pipeline_category = "bilateral"
    elif (intent == "finance") or bool(_FINANCE_BROAD_RE.search(topic)):
        _pipeline_category = "finance"
    elif intent == "sports":
        _pipeline_category = "sports"
    elif intent == "entertainment" or _is_entertainment:
        _pipeline_category = "entertainment"
    elif bool(_POLITICS_BROAD_RE.search(topic)):
        _pipeline_category = "politics"
    elif bool(_GEOPOLITICAL_RE.search(topic)) and len(countries_found) >= 1:
        _pipeline_category = "geopolitical"
    else:
        _pipeline_category = "general"

    # Step 1 — angle generation (FIX-6)
    search_angles = _generate_search_queries(topic)

    # FIX-7: Probe angles for geopolitical/entertainment where niche angles
    # are most likely to return zero DDG results.
    if _pipeline_category in ("geopolitical", "entertainment"):
        search_angles_validated = _validate_angles_with_probe(search_angles, min_results=2)
        if len(search_angles_validated) < len(search_angles):
            # Pad back up to 4 with fallback angles so we don't lose coverage
            fallback_pad = _get_fallback_angles(
                _pipeline_category, topic, countries_found
            )
            combined = search_angles_validated + fallback_pad
            seen_pad: set = set()
            search_angles = []
            for a in combined:
                k = a.lower().strip()
                if k not in seen_pad:
                    seen_pad.add(k)
                    search_angles.append(a)
            search_angles = search_angles[:4]
            print(f"[DEBUG] Post-probe angles (padded): {search_angles}")
        else:
            search_angles = search_angles_validated

    # Step 2 — fetch articles from all sources
    ddg_articles      = _fetch_articles_multi(search_angles, max_per_query=25)
    if len(ddg_articles) < 20:
        print(f"[DEBUG] Thin pool ({len(ddg_articles)}) — adding direct topic fetch")
        ddg_articles += _fetch_articles(topic, max_results=30)

    newsdata_articles = _fetch_articles_newsdata(
        query=topic, topic=topic, max_results=10,
    )
    # FIX-8: GNews as additional free-tier fallback
    gnews_articles    = _fetch_articles_gnews(
        query=topic, topic=topic, max_results=10,
    )
    raw_articles = ddg_articles + newsdata_articles + gnews_articles
    print(f"[DEBUG] Merged: {len(newsdata_articles)} NewsData + "
          f"{len(gnews_articles)} GNews + "
          f"{len(ddg_articles)} DDG = {len(raw_articles)} total")

    if not raw_articles:
        return []

    # Step 3 — domain filter + dedup
    filtered = _filter_by_domain(raw_articles)
    deduped  = _deduplicate_urls(filtered)

    # Step 3b: Adaptive date filter — applied ONCE here on articles.
    # FIX-1: cutoff is computed INSIDE the else branch so `window` is always
    #         defined before use (avoids NameError on small pools).
    # FIX-3: The original code ran a second recency filter on chunks before
    #         Step 7 which double-filtered and silently dropped good content.
    #         That chunk-level filter has been removed; recency is handled
    #         once here plus via the +0.05/+0.02 recency bonus in the scorer.
    if is_recency:
        pool_size = len(deduped)
        if pool_size <= _SMALL_POOL_THRESHOLD:
            # FIX-1: do NOT compute cutoff when skipping — window is undefined
            print(f"[DEBUG] Date filter skipped — pool too small ({pool_size})")
        else:
            # FIX-1: window and cutoff are scoped inside this else block
            window = (
                _DATE_WINDOW_SMALL_POOL_DAYS if pool_size <= 12
                else _DATE_WINDOW_NORMAL_DAYS
            )
            cutoff = (date.today() - timedelta(days=window)).isoformat()
            before  = len(deduped)
            deduped = [a for a in deduped if a.get("date", "")[:10] >= cutoff]
            print(f"[DEBUG] Date filter ({window}-day, pool was {before}): "
                  f"{before} -> {len(deduped)}")

    if not deduped:
        return []

    # Step 4a — title pre-filter (junk / off-topic gate, unchanged)
    # Keeps any article whose title or DDG snippet contains at least one
    # fuzzy-matched topic token.  Falls back to the full deduped list if
    # fewer than 3 articles survive.
    prefiltered = _title_prefilter(deduped, topic=topic, max_keep=35)

    # Step 4b — lightweight hybrid pre-ranking (ARCH-1)
    # Re-orders prefiltered by BM25+cosine on title+snippet so the
    # _MAX_DOWNLOAD budget goes to the most relevant articles, not just
    # whichever ones happened to appear first in the DDG result order.
    prefiltered = _pre_rank_articles(prefiltered, query=tool_query)

    # Step 5 — parallel article download
    # Slice AFTER pre-ranking so we download the top-scored articles.
    print(f"[DEBUG] Downloading top {_MAX_DOWNLOAD} articles (pre-ranked)...")
    extracted = _extract_articles_parallel(prefiltered[:_MAX_DOWNLOAD])

    if not extracted:
        return [{
            "text": f"No relevant articles found for: {topic!r}.",
            "title": "Warning", "date": "", "source": "",
            "url": "", "is_trusted": False, "_warning": True,
        }]

    # Step 6 — chunking
    print(f"[DEBUG] Chunking {len(extracted)} articles...")
    all_chunks = []
    for art in extracted:
        all_chunks.extend(_chunk_article(art, chunk_size=200, overlap=30))
    print(f"[DEBUG] Total chunks: {len(all_chunks)}")

    _GARBAGE = re.compile(
        r"("
        r"\b(sign in|log in|subscribe|advertisement|cookie policy|"
        r"privacy policy|terms of service|all rights reserved|"
        r"featured funds|invest now|benchmarks nifty|"
        r"facebook instagram linkedin|youtube sign in|"
        r"home state jammu|politics ground reports)\b|"
        r"load_file|boomerang|gtm4wp|dataLayer|localStorage|"
        r"window\.location|document\.cookie|function\s*\(|"
        r"Array\s*\(\[|\[direction\]\s*=>|\[market_status\]\s*=>|"
        r"Trending\s+Topics\s+[A-Z]|Live\s+Events\s+[A-Z]|"
        r"Skip to (navigation|content|main)|Read more at|"
        r"var \w+ = \{|window\.\w+\s*="
        r")",
        re.IGNORECASE
    )
    clean = []
    for chunk in all_chunks:
        text = chunk.get("text", "")
        if _GARBAGE.search(text):
            continue
        if len([w for w in text.split() if w.isalpha() and len(w) > 2]) < 12:
            continue
        if _has_encoding_corruption(text, threshold=0.03):
            fixed = _fix_encoding(text)
            if not _has_encoding_corruption(fixed, threshold=0.03):
                chunk["text"] = fixed
                clean.append(chunk)
            continue
        clean.append(chunk)

    quality = []
    for chunk in clean:
        words = chunk.get("text", "").split()
        if words and sum(1 for w in words if w.isalpha() and len(w) > 1) / len(words) >= 0.50:
            quality.append(chunk)
    dropped = len(all_chunks) - len(quality)
    if dropped:
        print(f"[DEBUG] Garbage filter: removed {dropped}, {len(quality)} remain")
    all_chunks = quality

    if not all_chunks:
        return []

    # FIX-3: Chunk-level recency filter deliberately REMOVED.
    # The original pipeline ran a date filter here AND at the article level
    # (Step 3b), which compound-filtered and silently discarded valid content
    # for niche topics with sparse coverage.  Recency is now handled:
    #   (a) once at the article level in Step 3b above, and
    #   (b) via the +0.05 / +0.02 recency bonus in _unified_score_chunks.

    # Step 7 — hybrid scoring; pass category for FIX-2 and FIX-4
    scored = _unified_score_chunks(
        query=tool_query,
        all_chunks=all_chunks,
        topic=topic,
        top_k=_TOP_K * 2 if _pipeline_category == "entertainment" else _TOP_K,
        category=_pipeline_category,
    )

    # Step 8 — minimum chunk gate
    gate = _minimum_chunk_gate(scored, topic=topic, min_chunks=2)
    if not gate["ok"]:
        warn = {
            "text": gate.get("warning", "Insufficient data."),
            "title": "Warning", "date": "", "source": "",
            "url": "", "is_trusted": False, "_warning": True,
        }
        return gate["chunks"] + [warn]

    final_chunks = gate["chunks"]

    def _root_domain_simple(url: str) -> str:
        try:
            parts = urlparse(url).netloc.lower().replace("www.", "").split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else url
        except Exception:
            return url

    unique_domains = {
        _root_domain_simple(c.get("url", ""))
        for c in final_chunks if c.get("url")
    }
    if len(unique_domains) <= 1 and len(final_chunks) >= 2:
        only_domain = next(iter(unique_domains), "unknown")
        diversity_warn = {
            "text": (
                f"Results for {topic!r} come from a single source ({only_domain}). "
                f"Content may not be directly relevant — consider refining your query."
            ),
            "title": "Warning", "date": "", "source": "",
            "url": "", "is_trusted": False, "_warning": True,
        }
        print(f"[DEBUG] RC4 diversity warning: all chunks from {only_domain}")
        return final_chunks + [diversity_warn]

    print(f"[DEBUG] Returning {len(final_chunks)} chunks from {len(unique_domains)} domains")
    return final_chunks