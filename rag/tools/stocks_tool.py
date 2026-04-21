# rag/tools/stocks_tool.py

import logging
import re
import requests
import yfinance as yf
from difflib import get_close_matches
from typing import Optional

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_NOISE_WORDS = re.compile(
    r"\b(share|shares|stock|stocks|price|today|current|live|rate|"
    r"equity|listed|market|trading|value|cost|quote|nse|bse)\b",
    re.IGNORECASE,
)

_KNOWN_TERMS = [
    "finance", "financial", "bank", "banking", "industries", "industry",
    "enterprises", "energy", "power", "services", "motors", "auto",
    "technologies", "tech", "capital", "investments", "holdings",
    "pharmaceuticals", "chemicals", "steel", "cement", "insurance",
]


def _clean_query(query: str) -> str:
    cleaned = _NOISE_WORDS.sub("", query)
    return re.sub(r"\s+", " ", cleaned).strip()


def _fix_typos(query: str) -> str:
    words = query.split()
    fixed = []
    for word in words:
        if len(word) < 4:
            fixed.append(word)
            continue
        matches = get_close_matches(word.lower(), _KNOWN_TERMS, n=1, cutoff=0.75)
        fixed.append(matches[0] if matches else word)
    return " ".join(fixed)


def _stem(word: str) -> str:
    """Strip common suffixes so variants match: greens->green, financial->financ"""
    word = word.lower()
    for suffix in ("ial", "ing", "ion", "ies", "ers", "ed", "al", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _get_keywords(query: str) -> list[str]:
    stopwords = {"and", "the", "of", "for", "in", "ltd", "limited", "inc",
                 "corp", "india", "nse", "bse"}
    return [_stem(w) for w in query.split()
            if w.lower() not in stopwords and len(w) > 3]


def _keyword_hits(name: str, keywords: list[str]) -> int:
    """Count keyword hits using stemming on both sides so variants match."""
    stemmed_name = " ".join(_stem(w) for w in name.lower().split())
    return sum(1 for kw in keywords if kw in stemmed_name)


def _fetch_price(symbol: str) -> Optional[tuple]:
    try:
        info = yf.Ticker(symbol).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        name = info.get("longName") or info.get("shortName") or symbol
        currency = info.get("currency", "USD")
        if price:
            return name, currency, float(price)
    except Exception:
        pass
    return None


def _search_ticker(query: str, require_indian: bool = False) -> Optional[str]:
    """
    Search Yahoo Finance. Returns the symbol whose company name has the
    most keyword hits from the query.

    require_indian=True  → only consider .NS / .BO symbols
    require_indian=False → prefer .NS/.BO if they have keyword hits, else any equity
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": 10, "newsCount": 0, "enableFuzzyQuery": True}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        if resp.status_code != 200:
            return None

        quotes = resp.json().get("quotes", [])
        equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]
        keywords = _get_keywords(query)

        def score(q: dict) -> int:
            name = (q.get("longname") or q.get("shortname") or "")
            return _keyword_hits(name, keywords)

        if require_indian:
            # Among all .NS/.BO results, pick the one with most keyword hits
            indian = [q for q in equities
                      if q.get("symbol", "").endswith(".NS")
                      or q.get("symbol", "").endswith(".BO")]
            if not indian:
                return None
            best = max(indian, key=score)
            # Reject if zero keywords matched — means it's an unrelated company
            return best.get("symbol") if score(best) > 0 else None

        else:
            # Pick best Indian result if it has keyword hits
            indian = [q for q in equities
                      if q.get("symbol", "").endswith(".NS")
                      or q.get("symbol", "").endswith(".BO")]
            if indian:
                best_indian = max(indian, key=score)
                if score(best_indian) > 0:
                    return best_indian.get("symbol")

            # Otherwise best global equity
            if equities:
                best_global = max(equities, key=score)
                if score(best_global) > 0:
                    return best_global.get("symbol")
                # No keyword hits at all — just return top result (direct ticker input)
                return equities[0].get("symbol")

            # Index fallback (nifty, sensex, nasdaq)
            indices = [q for q in quotes if q.get("quoteType") == "INDEX"]
            return indices[0].get("symbol") if indices else None

    except Exception:
        return None


def get_stock_price(ticker: str, reranker=None) -> str:
    """
    Resolution order:
    1. Raw symbol as-is
    2. Indian match — original query (most keyword hits among .NS/.BO)
    3. Indian match — typo-fixed query
    4. Global search — original query
    5. Global search — typo-fixed query
    6. .NS / .BO suffix guess
    """
    ticker_clean = ticker.strip()
    symbol_upper = ticker_clean.upper()

    # ── Step 1: Raw symbol ───────────────────────────────────────────────
    result = _fetch_price(symbol_upper)
    if result:
        name, currency, price = result
        return _format(name, symbol_upper, currency, price)

    search_query = _clean_query(ticker_clean)
    fixed_query = _fix_typos(search_query)

    # ── Step 2: Indian match — original query ────────────────────────────
    symbol = _search_ticker(search_query + " india", require_indian=True)
    if symbol:
        result = _fetch_price(symbol)
        if result:
            name, currency, price = result
            return _format(name, symbol, currency, price)

    # ── Step 3: Indian match — typo-fixed query ──────────────────────────
    if fixed_query != search_query:
        symbol = _search_ticker(fixed_query + " india", require_indian=True)
        if symbol:
            result = _fetch_price(symbol)
            if result:
                name, currency, price = result
                return _format(name, symbol, currency, price)

    # ── Step 4: Global search — original query ───────────────────────────
    symbol = _search_ticker(search_query, require_indian=False)
    if symbol:
        result = _fetch_price(symbol)
        if result:
            name, currency, price = result
            return _format(name, symbol, currency, price)

    # ── Step 5: Global search — typo-fixed query ─────────────────────────
    if fixed_query != search_query:
        symbol = _search_ticker(fixed_query, require_indian=False)
        if symbol:
            result = _fetch_price(symbol)
            if result:
                name, currency, price = result
                return _format(name, symbol, currency, price)

    # ── Step 6: .NS / .BO suffix guess ───────────────────────────────────
    base = symbol_upper.replace(".NS", "").replace(".BO", "").replace(" ", "")
    for suffix in (".NS", ".BO"):
        candidate = base + suffix
        result = _fetch_price(candidate)
        if result:
            name, currency, price = result
            return _format(name, candidate, currency, price)

    return (
        f"Could not find stock data for: '{ticker}'. "
        "Please check the company name or ticker symbol."
    )


def _format(name: str, symbol: str, currency: str, price: float) -> str:
    return (
        f"## {name}\n"
        f"- **Symbol:** {symbol}\n"
        f"- **Price:** {currency} {price:,.2f}"
    )