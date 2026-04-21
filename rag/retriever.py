from ast import keyword
import gc
import json
import hashlib
from multiprocessing import context
import re
import time
from unittest import result
from langchain_core import tools
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import re
import time

import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sqlalchemy import text
from yfinance import ticker

from rag import internet_search
from rag.reranker import Reranker
from rag.tools.sports_knowledge_tool import get_sports_kb_chunks
from rag.tools.weather_tool import get_weather
from rag.tools.stocks_tool import get_stock_price
from rag.tools.news_tool import get_news
from rag.tools.crypto_tool import get_crypto_price
from rag.tools.sports_tool import get_sports_scores
from system_commands.system_tools import system_command
from rag.tools.github_tool import handle_github
from config.settings import (
    CHROMA_DB_DIR,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    SIMILARITY_THRESHOLD,
    SPORTS_KB_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    ENABLE_HYBRID_SEARCH,
    MAX_TOKENS,
    MAX_MEMORY_TURNS,
    MAX_CONTEXT_TOKENS,
    LLM_MODEL,
    TEMPERATURE,
    SYSTEM_PROMPT,
    SEARCH_RESULTS_LIMIT,
    _KB_PRECHECK_THRESHOLD,
    SESSION_TIMEOUT_MINUTES,
)



# -------------------------
# Tool descriptions for semantic routing
# -------------------------
TOOL_DESCRIPTIONS = {
    "news": (
        "Tool: news. "
        "Use this tool when the user wants to READ or KNOW ABOUT recent real-world events, "
        "current affairs, breaking stories, or how global events are impacting the world. "
        "This is a READ-ONLY information tool — not for performing actions. "
        "Covers: politics, conflicts, international relations, economic impact of world events, "
        "business news, policy changes, natural disasters, elections, social issues. "
        "Trigger phrases: latest news, breaking news, current events, what happened, "
        "what is happening, situation in, conflict between, impact of war, effect of sanctions. "
        "Examples: "
        "latest news about iran, what is happening in ukraine, "
        "impact of us china trade war on india, breaking news today, "
        "russia ukraine war oil prices, iran missile strike news, "
        "effect of us sanctions on indian economy, geopolitical tensions, "
        "what happened in the middle east today, news about reliance industries, "
        "how does the conflict affect supply chains, today top headlines, "
        "modi latest news, us election results, earthquake news, flood news india"
    ),

    "weather": (
        "Tool: weather. "
        "Use this tool when the user wants to know atmospheric conditions, temperature, "
        "rainfall, humidity, wind, or forecast for a specific city or location. "
        "Always requires a place name or location. "
        "Trigger phrases: weather in, temperature in, forecast for, will it rain, "
        "is it cold, is it hot, climate in, should i carry umbrella. "
        "Examples: "
        "weather in london, temperature in delhi today, will it rain tomorrow in mumbai, "
        "weekly forecast for new york, humidity and wind in tokyo, is it cold in shimla, "
        "should i carry an umbrella in bangalore, monsoon forecast india, "
        "weather conditions in dubai this week, heatwave alert in rajasthan, "
        "is it snowing in manali, weather update for chennai"
    ),

    "stocks": (
        "Tool: stocks. "
        "Use this tool when the user wants the current or recent PRICE, VALUE, or PERFORMANCE "
        "of a publicly listed company stock, share, or financial market index. "
        "Trigger phrases: stock price, share price, market cap, index performance, "
        "how is X trading, what is X stock at, sensex, nifty, dow jones. "
        "Examples: "
        "apple stock price, tesla share price today, sensex today, s&p 500 performance, "
        "microsoft stock, reliance industries share price, tcs stock, infosys share today, "
        "nifty 50 index, adani stock price, hdfc bank shares, what is aapl trading at, "
        "stock market update, how is the dow jones performing, amazon stock today, "
        "zomato share price, paytm stock"
    ),

    "crypto": (
        "Tool: crypto. "
        "Use this tool when the user wants the current PRICE, VALUE, or MARKET DATA "
        "of a cryptocurrency or digital asset. "
        "Trigger phrases: crypto price, coin price, token value, btc, eth, bitcoin, ethereum. "
        "Examples: "
        "bitcoin price today, ethereum market cap, solana price, crypto market update, "
        "btc vs eth performance, dogecoin price, what is xrp worth, bnb coin price, "
        "how much is 1 bitcoin in rupees, crypto market crash, best performing coin today, "
        "cardano price, shiba inu coin value, usdt price, polygon matic price"
    ),

    "sports": (
        "Tool: sports. "
        "Use this tool ONLY for live scores, match results, fixtures, upcoming matches, "
        "and current standings — things that change day to day. "
        "Trigger phrases: score today, live score, match result, who won last night, "
        "today match, live, playing now, upcoming match, next game, fixture, schedule. "
        "Examples: "
        "ipl score today, who won the last india vs australia match, "
        "premier league live score, nba game results today, "
        "today cricket match score, who is playing today, "
        "next ipl match, upcoming champions league fixtures, "
        "barcelona latest score, real madrid vs manchester united score"
    ),

    # ------------------------------------------------------------------ #
    #  NEW — sports_knowledge: facts, stats, records, rules, history      #
    # ------------------------------------------------------------------ #
    "sports_knowledge": (
        "Tool: sports_knowledge. "
        "Use this tool when the user wants to LEARN OR KNOW facts, statistics, records, "
        "rules, history, biography, career stats, or achievements about sports, "
        "players, teams, or tournaments. "
        "This is a KNOWLEDGE tool — not for live scores or upcoming matches. "
        "Trigger phrases: how many, career total, batting average, most goals, most wickets, "
        "most centuries, world record, who holds, all time, career stats, "
        "explain the rules, how does offside work, biography of, who is, "
        "ballon d or, golden boot, history of, which team won the world cup, "
        "tournament winner, ipl winner 2022, who scored most, highest scorer. "
        "Examples: "
        "how many runs did virat kohli score, messi career goals, "
        "most wickets in test cricket all time, batting average of sachin tendulkar, "
        "who won ipl 2022, who won the 2023 cricket world cup, "
        "what is the offside rule in football, how does drs work in cricket, "
        "biography of ms dhoni, history of the premier league, "
        "who has the most grand slams in tennis, roger federer career wins, "
        "ballon d or winners list, golden boot football 2024, "
        "most goals in champions league history, nba all time scoring record, "
        "explain the lbw rule, what is a hat trick, "
        "which country has won most cricket world cups"
    ),

    "system": (
        "Tool: system. "
        "Use this tool when the user wants to PERFORM AN ACTION or EXECUTE A COMMAND "
        "on their local computer. This is an ACTION tool — the user is telling the computer "
        "to DO something, not asking for information. "
        "File and folder operations: "
        "create file, create folder, delete file, delete folder, remove file, remove folder, "
        "move file, move folder, rename file, copy file, open file. "
        "The file or folder name can be ANYTHING — news, work, projects, music, photos, videos, "
        "downloads, documents, code, data, backup, archive, temp, src, build. "
        "App operations: open app, launch app, close app, quit app, start app, switch to app. "
        "System controls: volume up, volume down, mute, unmute, "
        "brightness up, brightness down, take screenshot, "
        "lock screen, sleep, restart, shutdown, empty trash. "
        "Time and scheduling: set timer, set alarm, remind me, reminder, what time is it, "
        "current time, what is today date, current date. "
        "Reminder operations: remind me to, set a reminder, reminder at, remind me at, "
        "reminder for, remind me in, set reminder. "
        "Calendar actions: mark date, add event, schedule meeting, create event, add birthday. "
        "Examples: "
        "open spotify, open chrome, close safari, launch vscode, "
        "delete news folder, delete downloads folder, delete main.py, "
        "remove folder named work, remove projects folder, "
        "create folder named reports on desktop, create a python file called app, "
        "move main.py from downloads to documents, "
        "uninstall spotify, uninstall chrome, "
        "set timer for 5 minutes, set alarm at 7 30 AM, "
        "remind me to call john in 30 minutes, remind me at 5pm to leave, "
        "set a reminder at 10:30 to drink water, reminder at 9am to take medicine, "
        "what time is it, "
        "take screenshot, increase volume, decrease brightness, mute, "
        "mark 25th march for friend birthday, add meeting on friday at 3pm, "
        "create folder with name news, create folder called photos in documents"
    ),

    "kb": (
        "Tool: kb. "
        "Use this tool when the user wants to LEARN HOW TO DO something practical — "
        "programming tutorials, coding syntax, terminal commands, software installation, "
        "configuration steps, debugging errors, git commands, docker setup. "
        "This is a HANDS-ON HOW-TO tool — step-by-step instructions and commands only. "
        "Trigger phrases: how to install, how to create, how to delete, what command, "
        "fix error, debug, configure, setup, git command, bash script. "
        "Examples: "
        "how to install python on mac, how to create a virtual environment, "
        "git commit command, fix pip installation error, how to use docker, "
        "how to write a for loop in python, sql query examples, linux commands, "
        "how to configure nginx, bash script example, how to merge branches in git"
    ),

    "internet_search": (
        "Tool: internet_search. "
        "Use this tool for GENERAL KNOWLEDGE questions — definitions, explanations, concepts, "
        "technology overviews, science, history, culture, biography, geography, and any "
        "topic where the user wants to UNDERSTAND or LEARN ABOUT something. "
        "Covers: cloud computing, AI, machine learning, blockchain, cybersecurity, "
        "data science, quantum computing, software concepts, business concepts, "
        "economics, medicine, law, philosophy, and all non-sports factual knowledge. "
        "Trigger phrases: what is, what are, explain, how does, why is, who is, who was, "
        "tell me about, describe, overview of, benefits of, difference between, "
        "types of, examples of, history of, meaning of, definition of. "
        "Examples: "
        "what is cloud computing, how does machine learning work, "
        "explain blockchain technology, benefits of cloud computing, "
        "what are microservices, how does kubernetes work, "
        "what is quantum computing, explain neural networks, "
        "who is elon musk, history of artificial intelligence, "
        "what is the population of china, explain climate change, "
        "what are the seven wonders of the world, who wrote harry potter, "
        "difference between tcp and udp, types of cloud services, "
        "Who is elon musk,Tell me something about elon musk, "
        "what is devops, explain saas paas iaas"
    ),

    "github": (
        "Tool: github. "
        "Use this tool when the user wants to interact with GitHub — uploading files to a repo, "
        "creating repositories, listing repos, viewing branches, issues, pull requests, "
        "contributors, or getting Git command help. "
        "Trigger phrases: upload to github, push to repo, create repo, list repos, "
        "git push, pull request, open issues, git commands, git help. "
        "Examples: "
        "upload main.py to github, push config.json to my repo, create a new repo, "
        "delete my repo, remove repository, delete github repo, "
        "delete Lavish from my github repository, "       # ← add this
        "remove index.html from my repo, "               # ← add this
        "add file to github repository, "
        "new repository called my-project, list my github repos, show branches in my project, "
        "open issues in my repo, git commands help, git cheatsheet, who contributed to my repo, "
        "push file to github, create github repository, view pull requests"
    ),
}

_YEAR = time.strftime('%Y')
_PREV_YEAR = str(int(_YEAR) - 1)

# =============================================================================
# FIX 1 & 3 HELPER — shared guard used by both _clean_query and
# _preprocess_query_fallback to reject LLM-hallucinated "corrections"
# =============================================================================

_CLEAN_GUARD_JACCARD_THRESHOLD = 0.75  # strict: proper nouns must share ≥75% chars

def _char_jaccard(a: str, b: str) -> float:
    """Character-level Jaccard similarity (set of characters)."""
    sa, sb = set(a.lower()), set(b.lower())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _is_new_proper_noun(word: str, orig_tokens: set) -> bool:
    w = word.strip("'\"?!.,-")
    if not w or len(w) <= 2:
        return False
    if not w[0].isupper():
        return False
    parts = [p.lower() for p in w.split('-') if len(p) > 2]
    if not parts:
        return False
    for part in parts:
        if any(part in orig or orig in part
               for orig in orig_tokens if len(orig) > 2):
            continue
        best_j = max(
            (_char_jaccard(part, orig) for orig in orig_tokens if len(orig) > 2),
            default=0.0
        )
        if best_j >= _CLEAN_GUARD_JACCARD_THRESHOLD:
            continue
        return True
    return False


def _clean_query_guard(original: str, cleaned: str) -> str:
    words = original.split()
    cleaned_words = cleaned.split()
    orig_tokens = set(w.lower().strip("'\"?!.,") for w in words)
    for orig_tok in orig_tokens:
        if re.search(r'\d+$', orig_tok):  # original token ends in digits
            stripped = re.sub(r'\d+$', '', orig_tok)
            if stripped and stripped in cleaned.lower() and orig_tok not in cleaned.lower():
                print(f"[DEBUG] _clean_query_guard: rejected — "
                      f"digit stripped from '{orig_tok}'")
                return original
    expanded_count = sum(len(w.split('-')) for w in cleaned_words)
    if expanded_count > len(words) + 1:
        print(f"[DEBUG] _clean_query_guard: rejected — "
              f"word count {expanded_count} (expanded) vs {len(words)} original")
        return original
    new_proper_nouns = [
        w for w in cleaned_words
        if _is_new_proper_noun(w, orig_tokens)
    ]
    if new_proper_nouns:
        print(f"[DEBUG] _clean_query_guard: rejected — "
              f"new proper nouns: {new_proper_nouns}")
        return original
    return cleaned

_DIGEST_CATEGORY_RE = re.compile(
    r'(🌐|🏛|💰|🏏|💻|🎬|⚠️)\s*(World|Politics|Business|Sports|Technology|Entertainment|Warning)',
    re.IGNORECASE
)

def _dedup_bullets(bullets: list) -> list:
    print(f"[DEBUG] _dedup_bullets: input={len(bullets)} lines")
    print(f"[DEBUG] raw bullets:\n" + "\n".join(bullets[:5]))
    kept = []
    kept_title_tokens = []
    kept_body_tokens  = []
    _STOPWORDS = {
        "that","with","from","this","will","have","been","they",
        "their","about","once","again","face","face-off","polls","names",
        "said","says","also","after","amid","when","which","would",
        "president","donald","trump","said","says",  # high-freq news words
    }
    
    for bullet in bullets:
        if _DIGEST_CATEGORY_RE.search(bullet):
            kept.append(bullet)
            continue
        m = re.search(r'\*\*(.+?)\*\*', bullet)
        if not m:
            kept.append(bullet)
            continue
        title = m.group(1).lower()
        title_words = set(w for w in title.split() if len(w) > 3 and w not in _STOPWORDS)

        # Full bullet body (everything after the em-dash)
        body = re.sub(r'.*?—\s*', '', bullet, count=1)
        body_words = set(w for w in body.lower().split() if len(w) > 4 and w not in _STOPWORDS)

        title_dup = any(
            len(title_words & prev) / max(len(title_words | prev), 1) > 0.60
            for prev in kept_title_tokens if prev
        )
        body_dup = bool(body_words) and any(
            len(body_words & prev) / max(len(body_words | prev), 1) > 0.65
            for prev in kept_body_tokens if prev
        )

        if not title_dup and not body_dup:
            kept.append(bullet)
            kept_title_tokens.append(title_words)
            kept_body_tokens.append(body_words)

    print(f"[DEBUG] _dedup_bullets: output={len(kept)} lines")
    return kept
# =============================================================================
# MODULE-LEVEL NEWS TOPIC JUNK FILTER
# =============================================================================

_NEWS_TOPIC_JUNK = {
    "clean search topic", "search topic", "topic", "main topic",
    "core subject", "subject", "none", "n/a",
}


# =============================================================================
# SPORTS KNOWLEDGE ROUTING HELPERS
# =============================================================================

# Signals that mean the user wants LIVE / UPCOMING data — must stay in `sports`
_LIVE_SPORTS_SIGNALS = re.compile(
    r'\b(score|scorecard|latest score|latest result|match result|'
    r'final score|live score|today score|today match|playing now|'
    r'who won last night|who won yesterday|last match|last game|'
    r'recent match|yesterday.*match|match today|live|right now|'
    r'upcoming|next match|next game|fixture|fixtures|schedule|'
    r'when will|when does|will play|coming up|when is the next)\b',
    re.IGNORECASE
)

# Signals that mean the user wants KNOWLEDGE / FACTS about sports
_SPORTS_KNOWLEDGE_SIGNALS = re.compile(
    r'\b(how many|how much|career|total runs|batting average|bowling average|'
    r'most runs|most wickets|most goals|most centuries|most assists|'
    r'most appearances|tally|all.?time|in history|record holder|'
    r'highest scorer|highest wicket|most points|who scored most|'
    r'who has most|who holds|world record|career stats|career total|'
    r'career goals|career wickets|career runs|biography|born|nationality|'
    r'explain the|what is the offside|offside rule|lbw rule|drs|var|'
    r'hat trick|penalty rule|what are the rules|how does.*work|'
    r'ballon d.?or|golden boot|golden glove|golden ball|'
    r'hall of fame|grand slam|winner of|who won the.*cup|'
    r'who won the.*league|who won the.*tournament|who won the.*series|'
    r'who won the.*championship|ipl winner|world cup winner|'
    r'champions league winner|history of|founded|established)\b',
    re.IGNORECASE
)

# Sports-topic keywords — used to confirm the query is about sports
_SPORTS_TOPIC_SIGNALS = re.compile(
    r'\b(cricket|football|soccer|basketball|tennis|rugby|hockey|baseball|'
    r'golf|f1|formula 1|formula one|nba|nfl|nhl|mlb|ipl|bbl|psl|cpl|'
    r'premier league|la liga|bundesliga|serie a|ligue 1|champions league|'
    r'icc|fifa|uefa|bcci|odi|t20|t20i|test match|test cricket|'
    r'messi|ronaldo|kohli|sachin|dhoni|federer|nadal|djokovic|'
    r'lebron|jordan|kobe|bolt|pele|maradona|neymar|mbappe|haaland|'
    r'virat|rohit|bumrah|anderson|stokes|warner|smith|babar|'
    r'real madrid|barcelona|manchester|liverpool|arsenal|chelsea|'
    r'juventus|inter milan|ac milan|napoli|Bayern|dortmund|psg|'
    r'lakers|celtics|warriors|bulls|heat|knicks|'
    r'player|team|match|tournament|league|cup|series|innings|'
    r'wicket|century|goal|assist|penalty|offside|drs|lbw|'
    r'grand slam|wimbledon|us open|french open|australian open)\b',
    re.IGNORECASE
)

def _is_sports_knowledge_query(query: str, intent: str) -> bool:
    """
    Returns True if the query is about sports KNOWLEDGE/FACTS (not live scores).

    Rules:
    1. If intent is live_sports or upcoming_sports → always False (keep in sports tool)
    2. If LIVE_SPORTS_SIGNALS found → False (explicit score/fixture request)
    3. If SPORTS_KNOWLEDGE_SIGNALS + SPORTS_TOPIC_SIGNALS found → True
    4. If intent is facts/knowledge AND SPORTS_TOPIC_SIGNALS found → True
    """
    if intent in ("live_sports", "upcoming_sports"):
        return False

    q = query.lower()

    # Hard block — explicit live/upcoming signals override everything
    if _LIVE_SPORTS_SIGNALS.search(query):
        # Exception: "who won the 2022 world cup" has "who won" but is historical
        _is_historical = bool(
            re.search(r'\bwho won\b', q) and
            re.search(r'\b(20\d{2})\b|\b(world cup|tournament|series|championship|ipl|final|league)\b', q) and
            not re.search(r'\b(last night|yesterday|today|this week|just now|recent|last match|last game|live)\b', q)
        )
        if not _is_historical:
            return False

    has_sports_topic   = bool(_SPORTS_TOPIC_SIGNALS.search(query))
    has_knowledge_sig  = bool(_SPORTS_KNOWLEDGE_SIGNALS.search(query))

    # Strong signal: explicit knowledge keyword + sports topic
    if has_knowledge_sig and has_sports_topic:
        return True

    # Intent-based: facts/knowledge/sports_knowledge intent with sports topic
    if intent in ("facts", "knowledge", "sports_knowledge", "general") and has_sports_topic:
        return True

    return False

# =========================================================================
# HOW-DOES-X-WORK QUERY CLASSIFIER
# =========================================================================

_HOW_DOES_WORK_PATTERN = re.compile(
    r'^\s*how\s+(?:does|do|did|is|are|was|were|the|a|an|this|that)?\s*(.+?)\s+'
    r'(?:works?|functions?|operates?|processes?|runs?|executes?|performs?|behaves?|'
    r'happens?|occurs?|forms?|develops?|evolves?|grows?|spreads?|moves?|travels?)\??$',
    re.IGNORECASE
)

# Subject → tool/intent mapping for "how does X work" queries
_HOW_DOES_WORK_ROUTING = {
    # Technical / programming → kb
    
    # Scientific / natural phenomena → scientific intent via internet_search
    "scientific": re.compile(
        r'\b(photosynthesis|gravity|electricity|magnetism|nuclear|atom|'
        r'dna|gene|cell|evolution|immune system|brain|heart|digestion|'
        r'black hole|solar system|star|planet|orbit|tectonic|earthquake|'
        r'volcano|tsunami|hurricane|tornado|lightning|rainbow|aurora|'
        r'tide|moon|sun|photon|electron|proton|neutron|quantum|laser|'
        r'mri|x.?ray|vaccine|virus|bacteria|antibiotics|cancer|'
        r'climate|greenhouse|ozone|atmosphere|ocean|water cycle|'
        r'combustion|nuclear fusion|nuclear fission|radioactivity)\b',
        re.IGNORECASE
    ),

    # Sports rules/mechanics → sports_knowledge
    "sports_knowledge": re.compile(
        r'\b(cricket|football|soccer|basketball|tennis|rugby|hockey|'
        r'baseball|golf|formula|f1|drs|var|lbw|offside|penalty|'
        r'scoring in|match in|tournament|league|playoffs|seeding|'
        r'handicap|par|over|innings|wicket|serve|rally|slam|'
        r'transfer window|salary cap|draft|fantasy|betting odds)\b',
        re.IGNORECASE
    ),

    # Financial / economic → internet_search with facts intent
    "financial": re.compile(
        r'\b(stock market|share market|sensex|nifty|bond|mutual fund|'
        r'hedge fund|options|futures|derivatives|forex|currency|'
        r'inflation|deflation|gdp|recession|interest rate|repo rate|'
        r'central bank|rbi|fed|quantitative easing|ipo|dividend|'
        r'cryptocurrency|blockchain|defi|nft|bitcoin mining|'
        r'bank|loan|mortgage|credit|insurance|pension|tax|gst|'
        r'supply chain|trade deficit|balance of payments)\b',
        re.IGNORECASE
    ),

    # Medical / health → internet_search with scientific intent
    "medical": re.compile(
        r'\b(medicine|drug|antibiotic|vaccine|surgery|anesthesia|'
        r'chemotherapy|dialysis|transplant|prosthetic|pacemaker|'
        r'insulin|diabetes|blood pressure|cholesterol|allergy|'
        r'pain killer|sleeping pill|antidepressant|mental health|'
        r'therapy|meditation|yoga|fasting|intermittent fasting)\b',
        re.IGNORECASE
    ),

    # Technology concepts → internet_search with knowledge intent
    "technology": re.compile(
        r'\b(internet|wifi|wi-fi|networking|network|router|firewall|'
        r'web|website|browser|http|https|dns|tcp|ip|protocol|'
        r'bandwidth|latency|server|client|proxy|vpn|'
        r'cybersecurity|operating\s*system|cpu|gpu|processor|'
        r'semiconductor|transistor|memory|storage|database|'
        r'compiler|programming|software|hardware|algorithm|'
        r'data\s*structure|binary|bit|byte|'
        r'internet|wifi|5g|4g|bluetooth|gps|satellite|radar|sonar|'
        r'cloud computing|edge computing|quantum computing|ai|'
        r'artificial intelligence|machine learning|deep learning|'
        r'neural network|nlp|computer vision|autonomous|self.driving|'
        r'electric vehicle|battery|solar panel|wind turbine|'
        r'search engine|recommendation algorithm|social media algorithm|'
        r'facial recognition|fingerprint|encryption|blockchain|'
        r'3d printing|augmented reality|virtual reality|metaverse)\b',
        re.IGNORECASE
    ),
}

# Intent to assign per routing category
_HOW_DOES_WORK_INTENT_MAP = {  
    "scientific":       "scientific",
    "sports_knowledge": "sports_knowledge",
    "financial":        "facts",
    "medical":          "scientific",
    "technology":       "scientific",
}

def _classify_how_does_work(query: str) -> dict | None:
    """
    If the query is a 'how does X work' type question, classify it precisely.
    Returns dict with 'tool', 'intent', 'subject' or None if not matched.
    
    Priority order: kb > scientific > sports_knowledge > medical > financial > technology
    Falls back to LLM classification for unknown subjects.
    """
    if not _HOW_DOES_WORK_PATTERN.search(query):
        if not re.search(
            r'\bhow\s+(?:does|do|did|is|are|what|the|a|an|this|that)?\s*.{2,60}'
            r'\b(?:works?|functions?|operates?|processes?|working)\b',
            query, re.IGNORECASE
        ):
            return None

    m = _HOW_DOES_WORK_PATTERN.search(query)
    subject = m.group(1) if m else query

    # Check each hardcoded category first
    priority_order = [
        "scientific", "sports_knowledge",
        "medical", "financial", "technology",
    ]
    for category in priority_order:
        pattern = _HOW_DOES_WORK_ROUTING[category]
        if pattern.search(query):
            tool = "kb" if category == "kb" else (
                "sports_knowledge" if category == "sports_knowledge"
                else "internet_search"
            )
            intent = _HOW_DOES_WORK_INTENT_MAP[category]
            print(f"[DEBUG] _classify_how_does_work: "
                  f"subject='{subject}' → category={category}, "
                  f"tool={tool}, intent={intent}")
            return {
                "tool":     tool,
                "intent":   intent,
                "subject":  subject,
                "category": category,
            }

    # No hardcoded category matched — subject is unknown.
    # Return a sentinel so query_agent knows to use LLM classification.
    print(f"[DEBUG] _classify_how_does_work: "
          f"subject='{subject}' → no regex match, flagging for LLM classification")
    return {
        "tool":     "internet_search",
        "intent":   "__llm_classify__",   # sentinel — triggers LLM below
        "subject":  subject,
        "category": "unknown",
    }
# =========================================================================
# STRUCTURAL QUERY DETECTOR — catches explanation queries missed by
# _classify_how_does_work when the subject is outside known regex patterns
# =========================================================================

_EXPLAIN_INTENT_TRIGGERS = re.compile(
    r'('
    # "how the/a/an X works", "how X works"
    r'how\s+(?:the|a|an|this|that)\s+\w.{2,40}\s+works?\b|'
    r'how\s+\w.{2,40}\s+works?\s*$|'
    # "what is the working/mechanism/concept of X"
    r'what\s+is\s+(?:the\s+)?(?:working|mechanism|principle|concept|architecture|theory|overview|fundamentals?|basics?)\s+of\b|'
    r'what\s+are\s+(?:the\s+)?(?:working|mechanisms?|principles?|concepts?|architecture|theory|overview|fundamentals?|basics?)\s+of\b|'
    # "give me the working of X", "show me the working of X", "explain the working of X"
    r'(?:give\s+me|show\s+me|tell\s+me|explain)\s+(?:the\s+)?(?:working|mechanism|concept|overview|architecture|fundamentals?|basics?|principle)\s+of\b|'
    # "working of X", "mechanism of X" etc — standalone
    r'\bworking\s+(?:of|behind|principle\s+of)\b|'
    r'\bmechanism\s+of\b|'
    r'\bprinciple\s+of\b|'
    r'\bconcept\s+of\b|'
    r'\bfundamentals?\s+of\b|'
    r'\bbasics?\s+of\b|'
    r'\boverview\s+of\b|'
    r'\barchitecture\s+of\b|'
    r'\binternals?\s+of\b|'
    r'\btheory\s+of\b|'
    r'\bintroduction\s+to\b|'
    r'explain\s+(?:the\s+)?(?:concept|working|mechanism|principle|theory|architecture)\s+of\b|'
    # "X explained", "X overview", "X architecture"
    r'.{3,40}\s+(?:explained|overview|architecture|internals|fundamentals|basics)'
    r')',
    re.IGNORECASE
)

_SUBJECT_TO_CATEGORY = [
    # Each entry: (regex to match subject, category, tool, intent)
    (re.compile(
        r'\b(git|docker|kubernetes|nginx|linux|bash|python|javascript|'
        r'sql|api|rest|graphql|redis|mongodb|webpack|compiler|recursion|'
        r'hash\s*table|linked\s*list|binary\s*search|sorting|algorithm\s+in)\b',
        re.IGNORECASE), "kb", "kb", "knowledge"),

    (re.compile(
        r'\b(machine\s*learning|deep\s*learning|neural\s*network|transformer|'
        r'attention\s*mechanism|bert|gpt|llm|diffusion|backpropagation|'
        r'gradient|convolutional|lstm|rnn|reinforcement\s*learning|'
        r'artificial\s*intelligence|computer\s*vision|nlp|'
        r'cloud\s*computing|edge\s*computing|quantum\s*computing|'
        r'blockchain|encryption|5g|satellite|gps|radar|'
        r'recommendation\s*algorithm|facial\s*recognition|'
        r'augmented\s*reality|virtual\s*reality|metaverse|'
        r'internet\s*of\s*things|iot|devops|microservices|serverless|'
        r'containerization|virtualization|'
        # ADD THESE ↓
        r'internet|wifi|wi-fi|networking|network|router|firewall|'
        r'web|website|browser|http|https|dns|tcp|ip|protocol|'
        r'bandwidth|latency|packet|server|client|proxy|vpn|'
        r'cybersecurity|operating\s*system|cpu|gpu|processor|'
        r'semiconductor|transistor|memory|storage|database|'
        r'compiler|programming|software|hardware|algorithm|'
        r'data\s*structure|binary|hexadecimal|bit|byte)\b',
        re.IGNORECASE), "technology", "internet_search", "scientific"),

    (re.compile(
        r'\b(photosynthesis|gravity|electricity|magnetism|atom|dna|gene|'
        r'cell|evolution|immune|brain|heart|digestion|black\s*hole|'
        r'solar\s*system|quantum|nuclear|vaccine|virus|bacteria|cancer|'
        r'climate|greenhouse|combustion|radioactivity|relativity|'
        r'thermodynamics|ecosystem)\b',
        re.IGNORECASE), "scientific", "internet_search", "scientific"),

    (re.compile(
        r'\b(stock\s*market|share\s*market|inflation|gdp|recession|'
        r'interest\s*rate|central\s*bank|cryptocurrency|blockchain|'
        r'mutual\s*fund|options|futures|forex|ipo|bond|mortgage)\b',
        re.IGNORECASE), "financial", "internet_search", "facts"),

    (re.compile(
        r'\b(cricket|football|soccer|basketball|tennis|offside|lbw|drs|'
        r'var|hat\s*trick|penalty|innings|wicket|over|serve|slam)\b',
        re.IGNORECASE), "sports_knowledge", "sports_knowledge", "sports_knowledge"),

    (re.compile(
        r'\b(vaccine|surgery|chemotherapy|dialysis|insulin|diabetes|'
        r'blood\s*pressure|cholesterol|antibiotic|anesthesia|drug|'
        r'medicine|therapy|mental\s*health)\b',
        re.IGNORECASE), "medical", "internet_search", "scientific"),
]

def _detect_explanation_query(query: str) -> dict | None:
    """
    Catches explanation-style queries that _classify_how_does_work missed.
    For unknown subjects, flags for LLM classification instead of guessing.
    """
    if not _EXPLAIN_INTENT_TRIGGERS.search(query):
        return None

    print(f"[DEBUG] _detect_explanation_query: matched structural pattern for '{query}'")

    for subject_pattern, category, tool, intent in _SUBJECT_TO_CATEGORY:
        if subject_pattern.search(query):
            print(f"[DEBUG] _detect_explanation_query: "
                  f"category={category}, tool={tool}, intent={intent}")
            return {
                "tool":     tool,
                "intent":   intent,
                "subject":  query,
                "category": category,
            }
    # ── NEW: smarter fallback for unknown subjects ────────────────────
    # Instead of always returning knowledge, infer intent from query structure
    _q = query.lower()

    _SCIENTIFIC_CLUES = re.compile(
        r'\b(how|why|what|mechanism|process|formation|evolution|'
        r'reaction|cycle|system|organism|phenomenon|natural|'
        r'chemical|biological|physical|geological|astronomical)\b',
        re.IGNORECASE
    )
    _TECH_CLUES = re.compile(
        r'\b(works|runs|operates|executes|processes|functions|'
        r'protocol|system|network|software|hardware|digital|'
        r'device|signal|data|compute|program|code|platform)\b',
        re.IGNORECASE
    )
    _FINANCIAL_CLUES = re.compile(
        r'\b(market|trade|price|value|fund|invest|economy|'
        r'bank|stock|currency|rate|exchange|tax|revenue)\b',
        re.IGNORECASE
    )

    if _SCIENTIFIC_CLUES.search(query):
        inferred_intent = "scientific"
    elif _TECH_CLUES.search(query):
        inferred_intent = "scientific"   # still uses scientific format
    elif _FINANCIAL_CLUES.search(query):
        inferred_intent = "facts"
    else:
        inferred_intent = "knowledge"

    # Subject not recognized — default to internet_search/knowledge
    # This is the most important case: unknown subjects still get
    # routed correctly based on query STRUCTURE, not subject keywords
    print(f"[DEBUG] _detect_explanation_query: unknown subject → flagging for LLM classification")
    return {
        "tool":     "internet_search",
        "intent":   "__llm_classify__",   # sentinel
        "subject":  query,
        "category": "unknown",
    }


class HybridRetriever:
    def __init__(self):
        print("Loading embedding model for reranker...")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL, device=EMBEDDING_DEVICE
        )


        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        try:
            self.collection = self.client.get_collection(name="knowledge_base")
        except Exception:
            print("Collection 'knowledge_base' not found. Run ingestion first.")
            raise

        print("Loading documents for keyword search...")
        data = self.collection.get(include=["documents"])
        self.documents: List[str] = data["documents"]
        tokenized_docs = [doc.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        self.conv_memory: deque = deque(maxlen=MAX_MEMORY_TURNS)

        self._kv_cache: Optional[List[dict]] = None
        self._kv_cache_key: Optional[str] = None

        self._last_interaction_time = time.time()
        self._session_timeout = SESSION_TIMEOUT_MINUTES * 60

        print("Loading reranker...")
        self.reranker = Reranker()

        print("Embedding tool descriptions for semantic routing...")
        self.tool_embeddings = {
            tool: self.embedding_model.encode(desc)
            for tool, desc in TOOL_DESCRIPTIONS.items()
        }

        print("Caching intent embeddings...")
        self.command_embedding = self.embedding_model.encode(
            "open launch start run close quit execute perform do action "
            "increase decrease mute unmute toggle restart shutdown sleep "
            "screenshot lock empty trash control operate"
        )
        self.explanation_embedding = self.embedding_model.encode(
            "what is explain define describe tell me about how does why is "
            "meaning difference between overview summary biography history "
            "give me points facts about who is"
        )

        self.live_sports_embedding = self.embedding_model.encode(
            "score today live match result who won last night latest cricket football score"
        )
        self.career_stats_embedding = self.embedding_model.encode(
            "how many runs career total wickets batting average all time record most goals tally"
        )

        # Sports knowledge embedding — for semantic routing boost
        self.sports_knowledge_embedding = self.embedding_model.encode(
            "career stats records facts biography rules history sports knowledge "
            "tournament winner all time most goals wickets runs batting average "
            "world record explain rules offside lbw how does cricket football work"
        )

        self.ollama_model = LLM_MODEL
        print("Retriever ready.")


    # =========================================================================
    # SPORTS KNOWLEDGE TOOL EXECUTOR
    # =========================================================================
    def _run_sports_knowledge(
        self, query: str, intent: str,
                                 preprocessed_query: dict,
                                 length_instruction: str = "",
                                 num_points: int = None) -> str:
        """
        Dedicated handler for sports knowledge queries.
        Uses get_sports_kb_chunks() as primary source + internet search as secondary.
        Never calls get_sports_scores() — purely factual/statistical.
        """
        import hashlib
        print(f"[DEBUG] _run_sports_knowledge: query={repr(query)}, intent={intent}")
        _year = time.strftime('%Y')        # ← MOVE HERE
        _prev_year = str(int(_year) - 1)  # ← MOVE HERE 
        user_lower = query.lower()          # ← ADD THIS
        _is_football = bool(re.search(      # ← ADD THIS
        
            r'\b(football|soccer|goals?|premier league|champions league|'
            r'la liga|bundesliga|serie a|ligue 1|fifa|uefa|'
            r'messi|ronaldo|pele|maradona|neymar|mbappe|haaland|'
            r'striker|forward|hat trick in football)\b',
            query, re.IGNORECASE
        ))  

        if intent not in ("knowledge", "facts", "sports_knowledge", "scientific"):
            intent = "sports_knowledge"
            print(f"[DEBUG] _run_sports_knowledge: intent normalised to 'sports_knowledge'")
        _query_props = self._classify_sports_query(query)
        print(f"[DEBUG] Query props: {_query_props}")
        if _query_props.get("sport") == "cricket" and _query_props.get("stat_type") == "goals":
            _query_props = dict(_query_props)
            _query_props["stat_type"] = "runs"
            print(f"[DEBUG] Cricket stat_type normalized: goals → runs")
        if (
            _query_props.get("is_entity_specific")
            and _query_props.get("entity_type") == "national"
        ):
            _query_props = dict(_query_props)   # don't mutate the cached original
            _query_props["is_entity_specific"] = False
            if _query_props.get("sport") == "cricket":
                _query_props["cricket_format"] = "all"
            print(f"[DEBUG] entity_type=national → is_entity_specific reset to False")
            _player_name = query.split()[2] if len(query.split()) > 2 else query
            _cricket_format_queries = [
                f"{_player_name} Test runs career total {_year}",
                f"{_player_name} ODI runs career total {_year}",
                f"{_player_name} T20I runs career total {_year}",
                f"{_player_name} international cricket runs all formats {_year}",
            ]
        # Determine if this is a multi-format / career stats query
        from config.settings import (
            SEARCH_RESULTS_LIMIT, RERANK_TOP_K,
            SPORTS_KB_SIMILARITY_THRESHOLD, MAX_TOKENS
        )
        from rag.tools.sports_knowledge_tool import get_sports_kb_chunks
        _is_multi_format = bool(re.search(
            r'\b(all formats?|across formats?|test and odi|odi and test|'
            r'combined formats?)\b',
            user_lower, re.IGNORECASE
        ))
        _is_career = self._is_career_stats_query(query)
        _is_generic_record = bool(re.search(
            r'\b(most|highest|all.?time|in history|in the history|ever|'
            r'record holder|who has|who scored most|who holds|'
            r'leading scorer|top scorer|best ever)\b',
            query, re.IGNORECASE
        )) and bool(re.search(
            r'\b(football|soccer|cricket|tennis|basketball|nba|nfl|rugby|'
            r'hockey|baseball|golf|f1|formula|goals?|runs?|wickets?|'
            r'points?|assists?|grand slam|titles?|wins?)\b',
            query, re.IGNORECASE
        ))

        _is_tournament_query = bool(re.search(
            r'\b(squad|playing|participating|teams? in|groups?|which teams?|'
            r'how many teams?|who (are|is|will) (playing|participating)|'
            r'qualified|fixtures|schedule of|format of|venues?|host(ed by)?|'
            r'draw|seeding|bracket|knockout|group stage)\b',
            query, re.IGNORECASE
        ))

        if _is_multi_format:
            _fetch_top_k = SEARCH_RESULTS_LIMIT * 3
        elif _is_career or _is_generic_record:
            _fetch_top_k = SEARCH_RESULTS_LIMIT * 2
        else:
            _fetch_top_k = SEARCH_RESULTS_LIMIT
 
        _tl = self._get_timelimit(query, intent, _query_props)
        # ── Step 1: Build search queries ──────────────────────────────
        if _is_career or _is_generic_record or _is_tournament_query:
            search_queries = self._build_career_search_queries(query, _query_props)

        else:
            search_queries = [query, self._expand_query(query)]
            if _is_multi_format:
                search_queries += self._generate_format_subqueries(query)
        # ── CRITICAL: Force year into queries regardless of LLM output ────────
        _is_cricket_fixed = (
            _query_props.get("sport") == "cricket" and
            not _query_props.get("is_generic_record") and
            not _query_props.get("is_entity_specific")
        )
        has_year = any(_year in sq or _prev_year in sq for sq in search_queries)
        if not has_year and (_is_career or _is_generic_record) and not _is_cricket_fixed:
            search_queries.insert(1, search_queries[0] + f" {_year}")
            print(f"[DEBUG] Force year injection in _run_sports_knowledge")
        
        _is_football_generic = _is_generic_record and bool(re.search(
            r'\b(football|soccer|goals?)\b', query, re.IGNORECASE
        ))
        if _is_football_generic and not bool(re.search(
            r'\b(ronaldo|messi|pele)\b', query, re.IGNORECASE
        )):
            _ronaldo_query = f"Cristiano Ronaldo career goals total official {_year}"
            _messi_query   = f"Lionel Messi career goals total official {_year}"
            _pele_query    = f"Pele 767 official goals why not record competitive matches only"
            _record_query  = f"most goals football history officially recognised competitive {_year}"
            search_queries.extend([_ronaldo_query, _messi_query, _record_query, _pele_query])
            print(f"[DEBUG] Football generic record: injected Ronaldo/Messi verification queries")

        # ── Step 2: Sports KB chunks (DDG-backed, hybrid scored) ──────
        kb_query = search_queries[0]
        sports_kb_chunks = get_sports_kb_chunks(
            query=kb_query,
            embed_model=self.embedding_model,
            top_k=RERANK_TOP_K,
            threshold=SPORTS_KB_SIMILARITY_THRESHOLD,
        )
        kb_blocks = [f"[SOURCE]\n{chunk}" for chunk in sports_kb_chunks]
        kb_keys   = {hashlib.md5(c.lower().strip().encode()).hexdigest() for c in sports_kb_chunks}

        # ── Step 3: Internet search snippets ─────────────────────────
        # For career stats, always append current year to at least one query
        # so DDG doesn't return 2-year-old cached pages

        raw_snippets = []
        for sq in search_queries:
            fetched = self.internet_search_tool(sq, top_k=_fetch_top_k, timelimit=_tl)
            raw_snippets.extend(fetched)
        raw_snippets = list(dict.fromkeys(
            s for s in raw_snippets
            if hashlib.md5(s.lower().strip().encode()).hexdigest() not in kb_keys
        ))

        year_snippets = [s for s in raw_snippets if _year in s or str(int(_year)-1) in s]
        other_snippets = [s for s in raw_snippets if s not in year_snippets]
        raw_snippets = year_snippets + other_snippets

        web_blocks = self.format_tool_output(
            tool_output=raw_snippets,
            tool_name="internet_search",
        )

        # KB blocks first — they get priority in reranker
        all_blocks = kb_blocks + web_blocks

        _context_max_tokens = 4000 if (_is_career or _is_generic_record) else MAX_TOKENS

        # Rerank
        _rerank_top_k = RERANK_TOP_K * 4 if (_is_career or _is_generic_record) else RERANK_TOP_K
        if self.reranker and len(all_blocks) > 1:
            try:
                all_blocks = self.reranker.rerank(
                    query, all_blocks, top_k=RERANK_TOP_K
                )
            except Exception as e:
                print(f"[DEBUG] reranker failed: {e}")
        import hashlib as _hlib
        seen = set()
        deduped = []
        for block in all_blocks:
            content = re.sub(r'\[\w+\][^\n]*\n', '', block).strip()
            fp = _hlib.md5(content.lower().strip().encode()).hexdigest()
            if fp not in seen:
                seen.add(fp)
                deduped.append(block)
        packed = ""
        total_tokens = 0
        for block in deduped:
            tokens = len(block.split())
            if total_tokens + tokens > _context_max_tokens:
                break
            packed += block + "\n\n"
            total_tokens += tokens
 
        context = re.sub(r'\[(SPORTS KB|INTERNET SEARCH|SOURCE)\]\s*', '', packed).strip()
        if not context or len(context.strip()) < 50:
            context = "No reliable sports information found for this query."
 
        print(f"[DEBUG] _run_sports_knowledge: {len(kb_blocks)} KB + {len(web_blocks)} web blocks, "
        f"{total_tokens} context tokens (budget={_context_max_tokens})")


        if _is_career or _is_generic_record or intent == "sports_knowledge":
            _verified_facts = self._extract_stats_from_context(query, context, _query_props)
            _fact_lines = [
                l.strip() for l in (_verified_facts or "").splitlines()
                if l.strip().startswith("FACT:") and len(l.strip()) > 10
            ]

            if _fact_lines:
                context = (
                    f"=== VERIFIED FACTS EXTRACTED FROM SOURCES ===\n"
                    f"{chr(10).join(_fact_lines)}\n"
                    f"=== USE ONLY THESE NUMBERS IN YOUR ANSWER ===\n\n"
                    f"{context}"
                )
                print(f"[DEBUG] Verified facts prepended ({len(_fact_lines)} lines)")
            else:
                _context_no_numbers = re.sub(r'\b\d[\d,]*\b', '[NUMBER REDACTED]', context)
                context = (
                    f"=== WARNING: NO VERIFIED FACTS FOUND IN SOURCES ===\n"
                    f"The sources do not contain a clear answer to this question.\n"
                    f"You MUST say: 'The exact figure is not available in my current sources.'\n"
                    f"Do NOT substitute numbers from training memory under any circumstances.\n"
                    f"=== END WARNING ===\n\n"
                    f"{_context_no_numbers}"
                )
                print(f"[DEBUG] No facts extracted — training memory warning prepended")
        # ── Step 5: LLM answer ────────────────────────────────────────
        result = self._llm_call(
            user_input=query,
            context=context,
            intent=intent,
            length_instruction=length_instruction,
            num_points=num_points,
            context_label="Context from Sports KB and internet search",
        )
        return result


    def _detect_intent(self, query: str) -> dict:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f'Classify this query into exactly one intent.\n\n'
                        f'Query: "{query}"\n\n'
                        f'INTENTS:\n'
                        f'  live_sports   — today score, last match result, who won recently\n'
                        f'  facts         — career stats, how many X did Y score/take/win, '
                        f'                  all time record, most X in history, batting average\n'
                        f'  knowledge     — biography, history, how does X work, who is X\n'
                        f'  news          — latest news, breaking news, current events, war, conflict\n'
                        f'  weather       — temperature, forecast, rain, weather in city\n'
                        f'  scientific    — physics, chemistry, biology, space, how universe works\n'
                        f'  ownership     — CEO, founder, who owns company\n'
                        f'  general       — everything else\n\n'
                        f'CRITICAL RULE: "how many runs/goals/wickets did X score/take" = facts\n'
                        f'CRITICAL RULE: "score today / live score / who won last night" = live_sports\n'
                        f'The word "score" as a VERB (did X score) = facts, not live_sports\n\n'
                        f'Reply with ONLY this JSON, nothing else:\n'
                        f'{{"intent": "facts", "format": "default", "num_points": null}}'
                    )
                }],
                options={"temperature": 0.0, "num_predict": 30}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                raw = match.group(0)
            return json.loads(raw)
        except Exception as e:
            print(f"[DEBUG] _detect_intent failed: {e}")
            return {"intent": "general", "format": "default", "num_points": None}

    def _semantic_intent_check(self, query: str) -> str | None:
        qe = self.embedding_model.encode(query)

        def sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        career_sim = sim(qe, self.career_stats_embedding)
        live_sim   = sim(qe, self.live_sports_embedding)

        print(f"[DEBUG] semantic_intent: career={career_sim:.3f} live={live_sim:.3f}")

        if career_sim > 0.38 and career_sim > live_sim + 0.04:
            return "facts"
        if live_sim > 0.50 and live_sim > career_sim + 0.06:
            return "live_sports"

        return None


    def _detect_entities(self, query: str, intent: str) -> dict:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f'Query: "{query}"\n'
                        f'Intent: {intent}\n\n'
                        f"Return ONLY this JSON:\n"
                        f'{{"primary": ["main subject"], "context": [], "tool_params": {{}}}}\n\n'
                        f"Rules:\n"
                        f"- primary: the core subject the user wants info about\n"
                        f"  If intent is knowledge/facts/scientific → person or topic name\n"
                        f"  If intent is news → country, company, or event\n"
                        f"  If intent is weather → city name\n"
                        f"  If intent is live_sports/upcoming_sports → team or player\n"
                        f"- context: related background entity, if present\n"
                        f"- tool_params: only fill relevant key\n"
                        f"  weather → {{\"location\": \"city\"}}\n"
                        f"  stocks  → {{\"ticker\": \"SYMBOL\"}}\n"
                        f"  crypto  → {{\"coin\": \"coin-id\"}}\n"
                        f"  news    → {{\"topic\": \"search topic\"}}\n"
                        f"  others  → {{}}\n\n"
                        f"Examples:\n"
                        f'  "how many runs did virat kohli score for india" + intent=knowledge\n'
                        f'  → {{"primary":["virat kohli"],"context":["india"],"tool_params":{{}}}}\n\n'
                        f'  "latest news about iran war" + intent=news\n'
                        f'  → {{"primary":["iran"],"context":[],"tool_params":{{"topic":"iran war"}}}}\n\n'
                        f'  "weather in tokyo" + intent=weather\n'
                        f'  → {{"primary":["tokyo"],"context":[],"tool_params":{{"location":"Tokyo"}}}}\n\n'
                        f'  "reliance share price" + intent=general\n'
                        f'  → {{"primary":["reliance"],"context":[],"tool_params":{{"ticker":"RELIANCE.NS"}}}}\n\n'
                        f'  "how does iran war affect reliance" + intent=news\n'
                        f'  → {{"primary":["reliance"],"context":["iran war"],"tool_params":{{"topic":"reliance iran war impact"}}}}\n\n'
                        f"- For records/stats queries ('most X in Y', 'highest X', 'who scored most X'):\n"
                        f"    primary = the SPORT or COMPETITION (football, cricket, tennis)\n"
                        f"    context = [] (no context needed)\n"
                        f"- NEVER use abstract words as entities: history, record, world, all-time, career,\n"
                        f"    most, highest, best, goals, runs, wickets, points, assists\n"
                        f"- The entity must be a NAMED THING: person, team, sport, country, company, city\n"
                        f"Only the JSON. Nothing else."
                    )
                }],
                options={"temperature": 0.0, "num_predict": 80}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            _PLACEHOLDER_STRINGS = {
                "background entity if any", "supporting context",
                "background entity", "context", "supporting detail",
                "related background entity", "no context", "none",
            }
            primary = [e.strip() for e in parsed.get("primary", []) if e.strip()]
            context = [
                e.strip() for e in parsed.get("context", [])
                if e.strip() and e.strip().lower() not in _PLACEHOLDER_STRINGS
            ]
            tool_params = parsed.get("tool_params", {})

            primary_set = {e.lower() for e in primary}
            context = [e for e in context if e.lower() not in primary_set]

            if not primary:
                primary = [query]

            print(f"[DEBUG] _detect_entities: PRIMARY={primary} | CONTEXT={context} | PARAMS={tool_params}")
            return {"primary": primary, "context": context, "tool_params": tool_params}
        except Exception as e:
            print(f"[DEBUG] _detect_entities failed: {e}")
            return {"primary": [query], "context": [], "tool_params": {}}

    # =========================================================================
    # QUERY PREPROCESSING
    # =========================================================================
    def _preprocess_query(self, query: str) -> dict:
        cleaned = re.sub(r'\s+', ' ', query.strip())
        cleaned = self._clean_query(cleaned)

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Analyze this query and return a JSON object.\n"
                        f'Query: "{query}"\n\n'
                        f"Return ONLY this JSON structure:\n"
                        f"{{\n"
                        f'  "cleaned": "normalized lowercase query, filler words removed",\n'
                        f'  "entities": {{\n'
                        f'    "primary": ["main subject — named entity ONLY (country, company, person, topic)\n'
                        f'               NEVER include: regarding, about, related, latest, news, the, what"],\n'
                        f'    "context": []  // leave empty if no real background entity\n'
                        f'  }},\n'
                        f'  "intent": "knowledge",\n'
                        f'  // MUST be exactly ONE of:\n'
                        f'  // ownership, facts, knowledge, sports_knowledge, scientific, general,\n'
                        f'  // weather, live_sports, upcoming_sports, news\n'
                        f'  "classification": "informational|transactional|navigational",\n'
                        f'  "num_points": null,\n'
                        f'  "format": "detailed|brief|summary|points|definition|default",\n'
                        f'  "tool_params": {{"location": null, "ticker": null, "coin": null, "topic": null}}\n'
                        f"}}\n\n"
                        f"INTENT RULES — read every rule before deciding:\n"
                        f"  intent=sports_knowledge: ANY query about sports FACTS, STATS, RECORDS, RULES,\n"
                        f"                        HISTORY, or BIOGRAPHY. Career totals, batting/bowling\n"
                        f"                        averages, most goals/wickets/runs all-time, world records,\n"
                        f"                        tournament winners (past years), player bios, game rules,\n"
                        f"                        awards (Ballon d'Or, Golden Boot), offside, LBW, DRS,\n"
                        f"                        history of leagues or tournaments.\n"
                        f"  intent=scientific:    ANY query about physics, chemistry, biology, astronomy,\n"
                        f"                        geology, mathematics, medicine, natural phenomena, space,\n"
                        f"                        evolution, genetics, climate science, quantum mechanics.\n"
                        f"  intent=live_sports:   ANY query about a score, result, who won, latest score,\n"
                        f"                        last match, recent match, today match, live score.\n"
                        f"  intent=upcoming_sports: upcoming match, next game, fixture, schedule.\n"
                        f"  intent=news:          latest news, breaking news, what happened, current events,\n"
                        f"                        conflict, war, election, sanctions, policy, diplomacy,\n"
                        f"                        trade relations, bilateral relations, geopolitics,\n"
                        f"                        economy news, market news, company news, person in the news.\n"
                        f"                        Covers: country relations, government decisions,\n"
                        f"                        corporate announcements, political events.\n"
                        f"  intent=ownership:     who owns, CEO, founder, chairman — companies only.\n"
                        f"  intent=facts:         non-sports historical facts, specific dates/venues.\n"
                        f"  intent=knowledge:     definitions, explanations, history (non-sports),\n"
                        f"                        biographies of non-sports people, political titles.\n"
                        f"  intent=weather:       weather, temperature, forecast, rain, humidity.\n"
                        f"  intent=general:       everything else.\n\n"
                        f"PRIORITY ORDER (resolve conflicts top-to-bottom):\n"
                        f"  1. Score / result / who won today / live score → live_sports\n"
                        f"  2. Upcoming / fixture / schedule               → upcoming_sports\n"
                        f"  3. News article / breaking news / conflict     → news\n"
                        f"  4. Sports stats/records/rules/history/awards   → sports_knowledge\n"
                        f"  5. Ownership / CEO / founder (company only)    → ownership\n"
                        f"  6. Physics / chemistry / biology / space       → scientific\n"
                        f"  7. Political title / biography (non-sports)    → knowledge\n"
                        f"  8. Everything else                             → general\n\n"
                        f"EXAMPLES:\n"
                        f'  "what is quantum computing"              → intent=knowledge\n'
                        f'  "what is a black hole"                   → intent=scientific\n'
                        f'  "how does photosynthesis work"           → intent=scientific\n'
                        f'  "history of the roman empire"            → intent=knowledge\n'
                        f'  "who is elon musk"                       → intent=knowledge\n'
                        f'  "latest score of juventus"               → intent=live_sports\n'
                        f'  "next ipl match"                         → intent=upcoming_sports\n'
                        f'  "latest news on iran war"                → intent=news\n'
                        f'  "latest news regarding india china relations" → intent=news, primary=["india china relations"]\n'
                        f'  "what is happening in russia ukraine"    → intent=news, primary=["russia", "ukraine"]\n'
                        f'  "who is ceo of reliance"                 → intent=ownership\n'
                        f'  "president of usa"                       → intent=knowledge\n'
                        f'  "weather in jaipur"                      → intent=weather\n'
                        f'  "give me 5 points on climate change"     → num_points=5, format=points\n'
                        f'  "how many runs did virat kohli score"    → intent=sports_knowledge\n'
                        f'  "who won ipl 2022"                       → intent=sports_knowledge\n'
                        f'  "ballon d or winner 2024"                → intent=sports_knowledge\n'
                        f'  "what is the offside rule"               → intent=sports_knowledge\n\n'
                        f'  "summarize bitcoin for me"                → format=summary\n'
                        f'  "give me a summary of machine learning"   → format=summary\n'
                        f"OTHER RULES:\n"
                        f"  tool_params.location: city name for weather, else null\n"
                        f"  tool_params.ticker: stock ticker for stocks, else null\n"
                        f"  tool_params.coin: crypto id for crypto, else null\n"
                        f"  tool_params.topic: clean search topic for news, else null\n"
                        f"  num_points: integer if user asked for N points/tips/facts, else null\n"
                        f"  NEVER put generic words in entities: stock, price, war, news, score, history\n"
                        f"  Only the JSON. No explanation."
                    )
                }],
                options={"temperature": 0.0, "num_predict": 600}
           )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            if raw and not raw.rstrip().endswith('}'):
                open_braces   = raw.count('{') - raw.count('}')
                open_brackets = raw.count('[') - raw.count(']')
                stripped = raw.rstrip()
                if stripped and stripped[-1] not in ('}', ']', '"', ','):
                    last_comma = max(raw.rfind(','), raw.rfind('{'))
                    if last_comma > 0:
                        raw = raw[:last_comma]
                raw = raw.rstrip().rstrip(',')
                raw += ']' * open_brackets + '}' * open_braces
                print(f"[DEBUG] _preprocess_query: repaired truncated JSON")

            parsed = json.loads(raw)

            _PLACEHOLDERS = {
                "main subject", "supporting context", "actual main topic",
                "supporting detail", "topic", "subject", "context",
                "background entity if any", "background entity",
                "related background entity", "no context", "none",
            }
            _INTENT_WORDS = {
                "give","get","show","tell","find","fetch","search","look","check",
                "list","explain","describe","summarize","compare","need","want",
                "latest","recent","current","today","now","new","old","best","top",
                "about","please","update","the","a","an","in","on","of","for",
                "and","or","me","my","is","are","was","were","will","would",
                "could","should","do","does","did","news","weather","forecast",
                "temperature","stock","share","price","market","score","result",
                "match","sports","crypto","coin",
                # filler prepositions/connectors that leak into entities
                "regarding","related","concerning","with","from","by","at","its",
                "impact","effect","affects","affecting","between","against",
                "around","across","through","within","without","into","onto",
                "what","which","who","how","why","when","where","this","that",
                "these","those","their","there","they","them","then","than",
            }
            _PROMPT_LEAK = (
                "semantically meaningful","intent/action/filler",
                "examples of valid entities","never include",
                "supporting background","core subject","the user wants info",
                )
            _QUESTION_WORDS = {
                'how','what','why','when','where','who','which',
                'is','are','was','were','will','can','do','does',
            }

            def _is_valid_entity(e: str) -> bool:
                if not e or not isinstance(e, str):
                    return False
                e = e.strip('\'".,;:!? ')
                el = e.lower().strip()
                if not el or len(el.split()) > 5 or len(el) > 50:
                    return False
                if re.match(r'^[^a-z0-9]', el):
                    return False
                if el in _QUESTION_WORDS or el in _PLACEHOLDERS or el in _INTENT_WORDS:
                    return False
                if any(sig in el for sig in _PROMPT_LEAK):
                    return False
                return True

            raw_entities = parsed.get("entities", {"primary": [], "context": []})
            seen_entities: set = set()
            primary_clean: list = []
            for e in raw_entities.get("primary", []):
                if _is_valid_entity(e):
                    key = e.strip('\'".,;:!? ').lower()
                    if key not in seen_entities:
                        seen_entities.add(key)
                        primary_clean.append(e.strip('\'".,;:!? '))

            context_clean: list = []
            for e in raw_entities.get("context", []):
                if _is_valid_entity(e):
                    key = e.strip('\'".,;:!? ').lower()
                    if key not in seen_entities:
                        seen_entities.add(key)
                        context_clean.append(e.strip('\'".,;:!? '))

            validated_entities = {"primary": primary_clean, "context": context_clean}

            if not validated_entities["primary"]:
                _FALLBACK_STOP = {
                    'what','who','when','where','why','how','the','and','for',
                    'from','with','give','latest','recent','about','please',
                    'tell','show','find','get','me','my','a','an','is','are',
                    'was','were','will','would','could','should','do','does','did',
                    'news','update','weather','stock','price','score','result',
                    'crypto','coin','match','market',
                }
                words = [w for w in cleaned.lower().split() if w not in _FALLBACK_STOP]
                validated_entities["primary"] = words[:2] if words else [cleaned]

            raw_intent = parsed.get("intent", "general")
            _VALID_INTENTS = {
                "ownership","facts","knowledge","sports_knowledge","general","weather",
                "live_sports","upcoming_sports","news","scientific",
            }
            if raw_intent not in _VALID_INTENTS:
                print(f"[DEBUG] Invalid intent '{raw_intent}' → defaulting to 'general'")
                raw_intent = "general"

            if re.search(
                r'\b(latest news|breaking news|news about|news related to|'
                r'news regarding|what is happening|what happened|current events)\b',
                query, re.IGNORECASE
            ):
                if raw_intent not in ("news",):
                    print(f"[DEBUG] intent corrected: {raw_intent} → news (news keyword)")
                    raw_intent = "news"

            elif re.search(
                r'\b(how many|how may|how much|career|total runs|batting average|'
                r'bowling average|most runs|most wickets|most goals|most centuries|'
                r'tally|all.?time|in history|record holder|highest scorer|'
                r'most points|most assists|who scored most|who has most|who holds|'
                r'offside rule|lbw rule|ballon d|golden boot|biography of|'
                r'who won the|tournament winner|world record)\b',
                query, re.IGNORECASE
            ):
                if _SPORTS_TOPIC_SIGNALS.search(query) and raw_intent in (
                    "live_sports", "general", "knowledge", "facts"
                ):
                    print(f"[DEBUG] intent corrected: {raw_intent} → sports_knowledge (career/records/rules)")
                    raw_intent = "sports_knowledge"
                elif raw_intent in ("live_sports", "general"):
                    print(f"[DEBUG] intent corrected: {raw_intent} → knowledge (career stats, no sports topic)")
                    raw_intent = "knowledge"

            elif re.search(
                r'\b(score|scorecard|latest score|latest result|match result|'
                r'final score|live score|today score|today match|playing now|'
                r'who won|last match|last game|recent match)\b'
                r'|\b(vs\.?|versus)\b',
                query, re.IGNORECASE
            ) and not re.search(
                r'\b(news|breaking|headline|stock|share|price|crypto|'
                r'bitcoin|ethereum|compare|impact|affect|economy|market)\b',
                query, re.IGNORECASE
            ):
                if raw_intent in ("knowledge", "general", "facts"):
                    print(f"[DEBUG] intent corrected: {raw_intent} → live_sports (score/vs keyword)")
                    raw_intent = "live_sports"

            elif raw_intent in ("general", "knowledge"):
                if re.search(
                r'\b(physics|chemistry|biology|astronomy|geology|mathematics|'
                r'quantum|evolution|genetics|photosynthesis|gravity|atom|molecule|'
                r'dna|gene|cell|ecosystem|black hole|big bang|relativity|'
                r'thermodynamics|nuclear|solar system|galaxy|nebula|volcano|'
                r'earthquake|orbit|atmosphere|earth|planet|universe|space)\b',
                query, re.IGNORECASE
                ) and re.search(
                    r'\b(how did|how does|how do|how was|what causes|why does|'
                    r'why do|why is|what is .{2,40} made of|explain|theory of|law of)\b',
                    query, re.IGNORECASE
                ):
                    print(f"[DEBUG] intent corrected: {raw_intent} → scientific")
                    raw_intent = "scientific"

            elif raw_intent == "general":
                if re.search(
                    r'^\s*(what is|what are|who is|who was|explain|define|'
                    r'tell me about|describe|how does|how do|why is|why does|'
                    r'history of|meaning of|biography of)\b',
                    query, re.IGNORECASE
                ):
                    print(f"[DEBUG] intent corrected: general → knowledge (definitional query)")
                    raw_intent = "knowledge"
            if re.search(r'\bnet\s*worth\b', query, re.IGNORECASE):
                if raw_intent in ("ownership", "general", "live_sports"):
                    print(f"[DEBUG] intent corrected: {raw_intent} → knowledge (net worth query)")
                    raw_intent = "knowledge"

            raw_intent = _fix_sports_intent(query, raw_intent)

            _PROMPT_LEAK_CLEANED = (
                "normalized lowercase query","filler words removed",
                "core subject","intent/action","the user wants info",
                "supporting background","semantically meaningful",
            )
            _raw_cleaned = parsed.get("cleaned", cleaned)
            if any(sig in _raw_cleaned.lower() for sig in _PROMPT_LEAK_CLEANED) or len(_raw_cleaned) > 300:
                print(f"[DEBUG] Prompt leak in 'cleaned' field — using original query")
                _raw_cleaned = query

            return {
                "original":       query,
                "cleaned":        _raw_cleaned,
                "entities":       validated_entities,
                "intent":         raw_intent,
                "classification": parsed.get("classification", "informational"),
                "num_points":     parsed.get("num_points"),
                "format":         parsed.get("format", "default"),
                "tool_params":    parsed.get("tool_params", {}),
            }

        except Exception as e:
            print(f"[DEBUG] _preprocess_query LLM failed: {e}, using fallback")
            return self._preprocess_query_fallback(query, cleaned)

    def _preprocess_query_fallback(self, query: str, cleaned: str) -> dict:
        q = cleaned.lower()
        seen_fb = set()

        intent = "general"
        if any(w in q for w in ["weather","temperature","forecast","rain","humidity","wind"]):
            intent = "weather"
        elif re.search(
            r'\b(physics|chemistry|biology|astronomy|geology|mathematics|medicine|'
            r'ecology|quantum|evolution|genetics|photosynthesis|gravity|atom|molecule|'
            r'dna|gene|cell|ecosystem|climate science|black hole|big bang|relativity|'
            r'thermodynamics|electromagnetic|nuclear|solar system|galaxy|nebula|'
            r'tectonic|volcano|earthquake|hurricane|tsunami|orbit|atmosphere|'
            r'human|humans|life|origin|species|fossil|homo sapiens|primate|ape|organism)\b',
            q, re.IGNORECASE
        ) or re.search(
            r'\b(how did .{2,40} form|how does .{2,40} work|what causes|'
            r'why does .{2,40} happen|how was .{2,40} created|'
            r'came into existence|originate|evolved|how did .{2,40} evolve|'
            r'what is .{2,40} made of|explain .{2,40} process)\b',
            q, re.IGNORECASE
        ):
            intent = "scientific"
        elif (
            (re.search(
                r'\b(career|all.?time|ever|in history|overall|total|tally|'
                r'in his|in her|in their|t20i|t20is|odis|tests|test cricket|'
                r'odi cricket|batting average|bowling average|strike rate|'
                r'economy rate|statistics|stats\b|records?\b|biography|'
                r'how many .{2,40}\b(score|scored|took|made|hit|won)\b|'
                r'who has (most|highest|best)|most .{2,30} in|'
                r'goals for|goals in career|wickets in career|runs in career)\b',
                q, re.IGNORECASE
            ) and not re.search(
                r'\b(today|yesterday|last night|last match|last game|'
                r'this match|this innings|live|right now|score today|result today)\b',
                q, re.IGNORECASE
            )) or any(phrase in q for phrase in [
                "what is the offside","offside rule","what is a","what are the rules",
                "how many players","how many world cups","how many times",
                "how does","how do","explain","define","meaning","biography",
                "ballon d or","ballon d'or","golden boot","golden glove","hat trick",
                "how many ballon","world record","record holder",
                "who scored most","who has scored most","who has the most",
                "who holds the record","all time","all-time","in history",
            ])
        ):
            intent = "sports_knowledge"
        elif (
            "who won" in q and
            re.search(r'\b(20\d{2})\b|\b(world cup|tournament|series|championship|final)\b', q) and
            not re.search(r'\b(last night|yesterday|today|this week|just now|recent|last match|last game|live)\b', q)
        ):
            intent = "sports_knowledge"
        elif any(w in q for w in ["score","result","live score","playing now","today match",
                                   "who won","last match","last game","recent match","latest score",
                                   "final score","scorecard"]):
            intent = "live_sports"
        elif any(w in q for w in ["upcoming","next match","next game","fixture","when will","when is the next"]):
            intent = "upcoming_sports"
        elif re.search(r'\bnews\b', q) or any(w in q for w in [
            "latest news", "breaking news", "current events", "what happened",
            "news about", "war", "conflict", "election"
        ]):
            intent = "news"
        elif re.search(
            r'\b(ceo|founder|co-founder|chairman|chief executive|who owns|'
            r'who runs|who leads|who started|who created|who built|'
            r'founded by|owner of|who is behind)\b',
            q, re.IGNORECASE
        ):
            intent = "ownership"
        
        elif any(w in q for w in ["award","winner","world record","ballon","tournament winner",
                                   "who won the","won the cup","won the ipl","won the league"]):
            intent = "sports_knowledge"
        elif any(w in q for w in ["how to","what is","explain","define","meaning","history",
                                   "president of","prime minister","biography"]):
            intent = "knowledge"

        classification = "informational"
        if any(w in q for w in ["open","launch","price","score","upload","create","weather in"]):
            classification = "transactional"
        elif any(w in q for w in ["show me","find","list","fetch","search for"]):
            classification = "navigational"

        points_match = re.search(
            r'\b(\d+)\s*(?:key\s+|main\s+|important\s+)?(?:points?|facts?|tips?|steps?|reasons?|ways?|examples?)\b'
            r'|\b(?:give|list|tell|show)\s+(?:me\s+)?(?:in\s+)?(\d+)\b',
            q
        )
        num_points = int(points_match.group(1) or points_match.group(2)) if points_match else None

        fmt = "default"
        if num_points: fmt = "points"
        elif any(w in q for w in ["summarize", "give me a summary", "summary of"]): fmt = "summary"
        elif any(w in q for w in ["brief","short","quick","summary","summarize"]): fmt = "brief"
        elif any(w in q for w in ["detailed","in detail","explain","in-depth"]): fmt = "detailed"
        elif any(w in q for w in ["definition","meaning","define"]): fmt = "definition"

        _INTENT_WORDS_FB = {
            "give", "get", "show", "tell", "find", "fetch", "search",
            "look", "check", "list", "explain", "describe", "summarize",
            "compare", "need", "want",
            "latest", "recent", "current", "today", "now", "new", "old",
            "best", "top", "big", "last", "first", "next", "good", "bad",
            "about", "please", "update", "updates",
            "the", "a", "an", "in", "on", "of", "for", "and", "or",
            "me", "my", "is", "are", "was", "were", "will", "would",
            "could", "should", "do", "does", "did",
            "news", "weather", "forecast", "temperature",
            "stock", "share", "price", "market",
            "score", "result", "match", "sports",
            "crypto", "coin",
            "regarding", "related", "concerning", "with", "from", "by", "at",
            "impact", "effect", "affects", "affecting", "between", "against",
            "around", "across", "through", "within", "without", "into",
            "what", "which", "who", "how", "why", "when", "where",
            "this", "that", "these", "those", "their", "there", "its",
        }
        _clean_q_for_entities = self._clean_query(query)
        _clean_q_for_entities = _clean_query_guard(query, _clean_q_for_entities)

        _FB_STOP = {
            'what','who','when','where','why','how','the','and','for','from',
            'with','give','latest','recent','about','please','tell','show',
            'find','get','me','my','a','an','is','are','was','were','will',
            'would','could','should','do','does','did','news','update',
            'weather','stock','price','score','result','crypto','coin',
            'match','market',
        }
        _words = [w for w in _clean_q_for_entities.lower().split()
            if w not in _FB_STOP and len(w) > 2]
        raw_entities = {
            "primary": _words[:3] if _words else [_clean_q_for_entities],
            "context": [],
        }

        _ENTITY_FILTER = {
            "give","get","show","tell","find","fetch","search","look","check",
            "list","describe","summarize","compare","need","want","explain",
            "latest","recent","current","now","new","old","best","top","big",
            "last","first","next","good","bad","about","please","update","updates",
            "the","a","an","in","on","of","for","and","or","me","my",
            "is","are","was","were","will","would","could","should","do","does","did",
            "news","weather","forecast","temperature","stock","share",
            "price","market","score","result","match","sports","crypto","coin",
            "regarding","related","concerning","with","from","by","at","its",
            "impact","effect","affects","affecting","between","against",
            "around","across","through","within","without","into","onto",
            "what","which","who","how","why","when","where",
            "this","that","these","those","their","there","then","than",
        }

        _QUESTION_WORDS_FB = {
            'how', 'what', 'why', 'when', 'where', 'who', 'which',
            'is', 'are', 'was', 'were', 'will', 'can', 'do', 'does',
        }
        def _clean_entity_fb(e: str) -> str:
            return e.strip('\'".,;:!? ')
        primary_fb: list = []
        for e in raw_entities.get("primary", []):
            ce = _clean_entity_fb(e)
            cl = ce.lower()
            if (ce and len(ce) > 1
                    and cl not in _ENTITY_FILTER
                    and cl not in _QUESTION_WORDS_FB
                    and cl not in seen_fb
                    and not re.match(r'^[^a-z0-9]', cl)
                    and not re.search(r'\b(regardd|regardin|aboutt|abuot|teh|hte)\b', cl)):
                seen_fb.add(cl)
                primary_fb.append(ce)

        context_fb: list = []
        for e in raw_entities.get("context", []):
            ce = _clean_entity_fb(e)
            cl = ce.lower()
            if (ce and len(ce) > 1
                    and cl not in _ENTITY_FILTER
                    and cl not in _QUESTION_WORDS_FB
                    and cl not in seen_fb):
                seen_fb.add(cl)
                context_fb.append(ce)

        entities = {"primary": primary_fb, "context": context_fb}
        if not entities["primary"]:
            _FB_STOPWORDS = {
                'what','who','when','where','why','how',
                'the','and','for','from','with',
                'give','latest','recent','about','please',
                'tell','show','find','get','me','my',
                'a','an','is','are','was','were','will',
                'would','could','should','do','does','did',
                'news','update','weather','stock','price',
                'score','result','crypto','coin','match','market',
            }
            words = [w for w in _clean_q_for_entities.lower().split()
                     if w not in _FB_STOPWORDS and len(w) > 2]
            entities["primary"] = words[:3] if words else [_clean_q_for_entities]

        tool_params = {}
        if intent == "weather":
            tool_params = self._extract_params_fallback("weather", query)
        elif intent == "news":
            _TOPIC_STOP = {'what','is','the','latest','news','regarding','about','related',
                'to','give','me','a','an','on','for','of','in','with','some',
                'tell','show','get','find','any',}
            topic_words = [w for w in query.split() if w.lower() not in _TOPIC_STOP]
            tool_params = {"topic": " ".join(topic_words) or query}
        elif intent == "general" and any(w in q for w in [
            "latest news", "breaking news", "news about", "what happened",
            "current events", "conflict", "war", "election",
        ]):
            tool_params = self._extract_params_fallback("news", query)
        elif intent == "general" and any(w in q for w in [
            "stock","share","price","sensex","nifty","nasdaq"
        ]):
            tool_params = self._extract_params_fallback("stocks", query)
        elif intent == "general" and any(w in q for w in [
            "bitcoin","btc","ethereum","eth","crypto","coin","solana","dogecoin"
        ]):
            tool_params = self._extract_params_fallback("crypto", query)

        return {
            "original":       query,
            "cleaned":        cleaned,
            "entities":       entities,
            "intent":         intent,
            "classification": classification,
            "num_points":     num_points,
            "format":         fmt,
            "tool_params":    tool_params,
        }

    def _classify_query_type(self, query_lower: str) -> str:
        result = self._preprocess_query_fallback(query_lower, query_lower)
        return result["classification"]

    def _classify_intent(self, query: str) -> str:
        query_lower = query.lower()

        if re.search(
            r'\b(latest news|breaking news|news related to|news about|'
            r'what is the news|current news|recent news|'
            r'what happened|what is happening|situation in)\b',
            query_lower
        ):
            return "news"

        if re.search(
            r'\bhow many\b|\bhow much\b|\bcareer\b|\btotal runs\b|'
            r'\bbatting average\b|\bbowling average\b|\bmost runs\b|'
            r'\bmost wickets\b|\bmost goals\b|\bmost centuries\b|'
            r'\btally\b|\ball.?time\b|\bin history\b|\brecord holder\b|'
            r'\bhighest run\b|\bhighest scorer\b|\bhighest wicket\b|'
            r'\bmost points\b|\bmost assists\b|\bmost appearances\b|'
            r'\bwho scored most\b|\bwho has most\b|\bwho holds\b',
            query_lower
        ):
            if _SPORTS_TOPIC_SIGNALS.search(query):
                return "sports_knowledge"
            return "knowledge"
        if re.search(
            r'\b(score|scorecard|latest score|latest result|latest match|final score|'
            r'live score|today score|today match|playing now|match result)\b'
            r'|\bscore\b(?!d)',
            query_lower
        ):
            if not re.search(
                r'\b(career|all.?time|till now|so far|total|overall|'
                r'how many .{2,40}(score|scored)|batting average|'
                r'most runs|most wickets|most centuries)\b',
                query_lower
            ):
                return "live_sports"

        if (
            re.search(r'\b(vs\.?|versus)\b', query_lower) and
            not re.search(
                r'\b(stock|share|price|crypto|bitcoin|ethereum|compare|impact|'
                r'affect|news|breaking|war|politics|economy|market|weather)\b',
                query_lower
            )
        ):
            return "live_sports"

        if any(phrase in query_lower for phrase in [
            "player of the tournament", "player of the series",
            "man of the tournament", "man of the series",
            "mvp", "most valuable", "golden boot", "golden glove",
            "best player", "best bowler", "best batter",
            "who won the player", "who won the award",
            "who won the world cup", "who won the tournament",
            "who won the series", "who won the ipl", "who won the league",
            "who won the cup", "who won the champions",
            "tournament winner", "world cup winner", "champion of",
            "award winner", "ballon d or", "ballon d'or",
        ]) or (
            re.search(
                r'\bwho won\b.{1,40}\b(world cup|ipl|league|cup|tournament|series|final|championship)\b',
                query_lower
            ) and not re.search(
                r'\b(last night|yesterday|today|this week|just now|recent|last match|last game)\b',
                query_lower
            )
        ):
            return "sports_knowledge"

        if any(word in query_lower for word in [
            "upcoming", "next match", "next game", "fixture", "fixtures",
            "schedule", "scheduled", "when will", "will play", "coming up",
            "when does", "when is the next",
        ]) or re.search(r'next.{1,20}(match|game|fixture|cricket|football|ipl|test|odi|t20)', query_lower):
            return "upcoming_sports"

        _CAREER_SIGNALS = re.compile(
            r'\b(career|all.?time|ever|in history|overall|total|tally|'
            r'in his|in her|in their|t20i|t20is|odis|tests|test cricket|'
            r'odi cricket|batting average|bowling average|strike rate|'
            r'economy rate|statistics|stats\b|records?\b|biography|'
            r'how many .{2,40}\b(score|scored|took|made|hit|won)\b|'
            r'who has (most|highest|best)|most .{2,30} in|'
            r'goals for|goals in career|goals in champions|'
            r'wickets in career|runs in career)\b',
            re.IGNORECASE
        )
        _LIVE_SIGNALS_KP = re.compile(
            r'\b(today|yesterday|last night|last match|last game|'
            r'this match|this innings|current|live|right now|'
            r'in the match|score today|result today)\b',
            re.IGNORECASE
        )
        _has_career = bool(_CAREER_SIGNALS.search(query))
        _has_live   = bool(_LIVE_SIGNALS_KP.search(query))

        _RULES_PHRASES = [
            "what is the offside", "offside rule", "what is a", "what is the",
            "how many players", "how many world cups", "how many times",
            "how does", "how do", "explain", "define", "meaning of",
            "what are the rules", "drs", "var", "hat trick", "penalty rule",
            "ballon d or", "ballon d'or", "golden boot", "golden glove",
            "how many ballon", "biography", "born", "nationality",
            "world record", "record holder", "who holds the record",
            "who scored most", "who has scored most", "who has the most",
            "all time", "all-time", "in history", "in the history",
        ]
        if (_has_career and not _has_live) or any(p in query_lower for p in _RULES_PHRASES):
            if _SPORTS_TOPIC_SIGNALS.search(query):
                return "sports_knowledge"
            return "knowledge"
        _SCIENTIFIC_TOPICS_CI = re.compile(
            r'\b(physics|chemistry|biology|astronomy|geology|mathematics|medicine|'
            r'ecology|quantum|evolution|genetics|photosynthesis|gravity|atom|molecule|'
            r'dna|gene|cell|ecosystem|black hole|big bang|relativity|'
            r'thermodynamics|nuclear|solar system|galaxy|nebula|'
            r'tectonic|volcano|earthquake|hurricane|tsunami|orbit|atmosphere|'
            r'earth|planet|universe|space|star|sun|moon|'
            r'human|humans|life|origin of life|species|fossil|prehistoric|'
            r'energy|force|light|sound|wave|radiation|element|compound|reaction)\b',
        re.IGNORECASE
        )
        _SCIENTIFIC_VERBS_CI = re.compile(
            r'\b(how did|how does|how do|how was|how were|what causes|'
            r'why does|why do|why is|what is .{2,40} made of|'
            r'came into existence|come into existence|originate|originated|'
            r'explain .{2,30} process|theory of|law of|laws of)\b',
            re.IGNORECASE
        )
        if _SCIENTIFIC_TOPICS_CI.search(query_lower) and _SCIENTIFIC_VERBS_CI.search(query_lower):
            return "scientific"

        if (_has_career and not _has_live) or any(p in query_lower for p in _RULES_PHRASES):
            return "knowledge"

        if any(word in query_lower for word in [
            "score", "scorecard", "latest score", "latest result",
            "latest match", "result", "final score",
            "live score", "today match", "today score", "playing now",
            "who won", "last match", "last game", "recent match",
            "yesterday match", "match result",
        ]):
            return "live_sports"

        POLITICAL_GUARDS = [
            "president of usa", "president of india", "president of china",
            "president of russia", "president of france", "president of the",
            "president of pakistan", "president of iran", "president of turkey",
            "president of brazil", "president of mexico", "president of germany",
            "prime minister", "head of state", "head of government",
            "vice president", "governor of", "minister of",
            "secretary of state", "chancellor of",
        ]
        is_political = any(phrase in query_lower for phrase in POLITICAL_GUARDS)
        _CEO_PATTERN = re.compile(
            r'\b(ceo|founder|chairman|co-founder|chief executive|who owns|owner of|'
            r'founded by|head of|who runs|who leads|who started|who created|'
            r'who made|who built|who is behind)\b',
            re.IGNORECASE
        )

        if not is_political and _CEO_PATTERN.search(query_lower):
            return "ownership"

        if any(word in query_lower for word in [
            "when", "date", "venue", "where is", "start date",
        ]):
            return "facts"

        if any(word in query_lower for word in [
            "weather", "temperature", "forecast", "rain",
            "humidity", "wind", "climate", "sunny", "cloudy",
            "monsoon", "heatwave", "cold in", "hot in",
            "umbrella", "raining", "drizzle", "snow",
        ]):
            return "weather"

        _SCIENTIFIC_TOPICS = re.compile(
            r'\b(physics|chemistry|biology|astronomy|geology|mathematics|medicine|'
            r'ecology|quantum|evolution|genetics|photosynthesis|gravity|atom|molecule|'
            r'dna|gene|cell|ecosystem|black hole|big bang|relativity|'
            r'thermodynamics|nuclear|solar system|galaxy|nebula|'
            r'tectonic|volcano|earthquake|hurricane|tsunami|orbit|atmosphere|'
            r'earth|planet|universe|space|star|sun|moon|'
            r'energy|force|light|sound|wave|radiation|element|compound|reaction)\b',
            re.IGNORECASE
        )
        _SCIENTIFIC_VERBS = re.compile(
            r'\b(how did|how does|how do|how was|how were|what causes|'
            r'why does|why do|why is|what is .{2,40} made of|'
            r'explain .{2,30} process|theory of|law of|laws of)\b',
            re.IGNORECASE
        )
        if _SCIENTIFIC_TOPICS.search(query_lower) and _SCIENTIFIC_VERBS.search(query_lower):
            return "scientific"

        if any(word in query_lower for word in [
            "how to", "what is", "explain", "define", "meaning", "what are",
        ]):
            return "knowledge"

        if is_political:
            return "knowledge"

        return "general"

    # =========================================================================
    # LLM ROUTERS
    # =========================================================================
    def _llm_route_tool(self, query: str) -> str:
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                "role": "user",
                "content": (
                    f'Which tool answers this query? Reply with ONE word only.\n\n'
                    f'Tools: news, weather, stocks, crypto, sports, sports_knowledge, system, kb, internet_search, github\n\n'
                    f'Rules:\n'
                    f'- score/result/who won today/live score → sports\n'
                    f'- upcoming match/next game/fixture → sports\n'
                    f'- career stats/how many X did Y/records/rules/tournament winner → sports_knowledge\n'
                    f'- latest news/breaking/conflict → news\n'
                    f'- upload/push/create/delete repo → github\n'
                    f'- stock price/share price → stocks\n'
                    f'- bitcoin/crypto price → crypto\n'
                    f'- open/close/create/delete/remind → system\n'
                    f'- how to install/code/terminal → kb\n'
                    f'- who is/what is/biography (non-sports) → internet_search\n\n'
                    f"User query: \"{query}\"\n\n"
                    f"Reply with ONLY the tool name. Nothing else."
                )
            }],
            options={"temperature": 0.0, "num_predict": 5}
        )
        raw = response["message"]["content"].strip().lower()
        valid_tools = {"news", "weather", "stocks", "crypto", "sports", "sports_knowledge",
                       "system", "kb", "internet_search", "github"}
        aliases = {
            "search": "internet_search",
            "internet": "internet_search",
            "internet search": "internet_search",
            "knowledge": "internet_search",
            "kb": "kb",
            "system": "system",
            "sports knowledge": "sports_knowledge",
            "github": "github",
        }
        if raw in valid_tools:
            return raw
        if raw in aliases:
            return aliases[raw]
        for t in valid_tools:
            if t in raw:
                return t
        return "internet_search"

    def _llm_detect_multi_tool(self, query: str) -> List[str]:
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                "role": "user",
                "content": (
                    f"A user asked: \"{query}\"\n\n"
                    f"List every tool needed to fully answer this question. Available tools:\n"
                    f"  news           — current events, breaking news, geopolitical or market impact\n"
                    f"  stocks         — stock / share price of a specific company or index\n"
                    f"  crypto         — cryptocurrency prices and market data\n"
                    f"  weather        — weather forecast for a city or location\n"
                    f"  sports         — live scores, match results today, upcoming fixtures\n"
                    f"  sports_knowledge — career stats, records, rules, tournament history, player facts\n"
                    f"  system         — open/close/launch apps, take screenshot, adjust volume or brightness,\n"
                    f"                   set timer/alarm/reminder, remind me, create/delete/move files or folders\n"
                    f"  github         — list repos, show branches, issues, pull requests, git help\n"
                    f"  kb             — how-to, programming, terminal commands, software setup\n"
                    f"  internet_search — general knowledge, definitions, history, science\n\n"
                    f"Rules:\n"
                    f"- Use sports ONLY for live/today scores and upcoming fixtures.\n"
                    f"- Use sports_knowledge for career stats, records, rules, historical winners.\n"
                    f"- Include a tool ONLY if the query explicitly or clearly needs it.\n"
                    f"- 'how does X impact Y price' or 'effect of X on Y' → news only, NOT stocks.\n"
                    f"- Use stocks ONLY when user explicitly asks for current price or value.\n"
                    f"- Return ONLY tool names separated by commas. No explanation.\n\n"
                    f"Examples:\n"
                    f"  'open youtube and show latest news' → system,news\n"
                    f"  'reliance share price and iran war impact on it' → stocks,news\n"
                    f"  'ipl score today and how many ipl titles has mumbai indians won' → sports,sports_knowledge\n"
                    f"  'apple stock price' → stocks\n"
                    f"  'what is machine learning' → internet_search\n"
                    f"  'messi career goals and barcelona latest score' → sports_knowledge,sports\n\n"
                    f"Answer:"
                )
            }],
            options={"temperature": 0.0, "num_predict": 20}
        )
        raw = response["message"]["content"].strip().lower()
        valid = {"news", "weather", "stocks", "crypto", "sports", "sports_knowledge",
                 "system", "kb", "internet_search", "github"}
        seen = set()
        tools = []
        for t in raw.split(","):
            t = t.strip().replace(" ", "_")
            if t in valid and t not in seen:
                tools.append(t)
                seen.add(t)
        print(f"[DEBUG] LLM multi-tool decision: {tools}")
        return tools if tools else ["internet_search"]
    

    # =========================================================================
    # SEMANTIC TOOL ROUTER
    # =========================================================================
    def _route_tool(self, query: str) -> str:
        ql = query.lower().strip()

        if re.search(
            r'(upload|push file|git push|create repo|new repo|make repo'
            r'|delete repo|delete repository|remove repo|remove repository'
            r'|delete.*\bgithub\b|\bgithub\b.*delete'
            r'|delete.*\brepo\b|\bremove.*\brepo\b'
            r'|from\s+(my\s+)?(github|repo|repository)'
            r'|(?:add|push|upload|move|rename)\s+.{1,30}\s+(?:to|from|on)\s+(?:my\s+)?(?:github|repo)'
            r'|list repos|my repos|pull request|open issues'
            r'|git help|git commands|git cheatsheet|github)\b'
            r'|push\s+\w+\.\w+',
            ql
        ):
            return "github"
        
        if re.search(
            r'\b(impact|affect|affects|effect|influence|how does|how will|'
            r'what happens|due to|because of|result of)\b', ql
        ) and re.search(
            r'\b(war|conflict|sanction|crisis|geopolit|iran|russia|ukraine|'
            r'oil price|commodity|inflation|recession|tension)\b', ql
        ):
            print(f"[DEBUG] Geopolitical impact query → news")
            return "news"

        if re.search(r'\b(share price|stock price|share value|trading at|listed on nse|listed on bse)\b', ql):
            if not re.search(r'\b(impact|affect|affects|effect|influence|due to|because of|how does|how will|what happens)\b', ql):
                return "stocks"

        if re.search(
            r'\byoutube\b|'
            r'\bplay\s+.+\s+(?:on|in)\s+youtube\b|'
            r'\bsearch\s+.+\s+(?:on|in)\s+youtube\b|'
            r'\bwatch\s+.+\s+(?:on|in)\s+youtube\b|'
            r'\bopen\s+youtube\b',
            ql
        ):
            print(f"[DEBUG] YouTube hard rule matched → system")
            return "system"

        if re.search(
            r'\b(remind\s+me|set\s+a\s+reminder|set\s+reminder|reminder\s+at'
            r'|remind\s+me\s+at|remind\s+me\s+in|reminder\s+for'
            r'|reminder\s+to|remind\s+me\s+to)\b',
            ql
        ):
            print(f"[DEBUG] Reminder hard rule matched → system")
            return "system"

        if re.search(
            r'^\s*(create|make|new)\b.{0,30}\b(file|folder|directory|dir'
            r'|python|java|javascript|js|typescript|ts|html|css|cpp|ruby|bash'
            r'|go|rust|markdown|json|yaml|sql|txt|text)\b'
            r'|^\s*(create|make|new)\s+(?:a\s+)?(?:new\s+)?folder\b',
            ql
        ):
            print(f"[DEBUG] File/folder creation hard rule matched → system")
            return "system"

        # ── Sports knowledge hard rules ────────────────────────────────
        # These must come BEFORE the sports score hard rule
        _is_historical_winner = bool(
            re.search(r'\bwho won\b', ql) and
            re.search(r'\b(20\d{2})\b|\b(world cup|tournament|series|championship|final|league|ipl)\b', ql) and
            not re.search(r'\b(last night|yesterday|today|this week|just now|recent|last match|last game|live)\b', ql)
        )
        _is_career_stats = bool(re.search(
            r'\b(career|all.?time|till now|so far|total|overall|'
            r'how many .{2,40}\b(score|scored|took|made|hit|won)\b|'
            r'batting average|bowling average|most runs|most wickets|'
            r'most centuries|most goals|record|how many runs|how many wickets|'
            r'how many goals|biography|explain the|what is the offside|'
            r'offside rule|lbw|drs|var|hat trick|golden boot|ballon d|'
            r'grand slam|all time|in history|world record)\b',
            ql
        ))

        if _is_historical_winner or _is_career_stats:
            if _SPORTS_TOPIC_SIGNALS.search(query):
                print(f"[DEBUG] Sports knowledge hard rule → sports_knowledge "
                      f"(historical={_is_historical_winner}, career={_is_career_stats})")
                return "sports_knowledge"

        # ── Sports score hard rules ────────────────────────────────────
        if not _is_historical_winner and not _is_career_stats and re.search(
            r'\b(scorecard|latest score|latest result|match result|'
            r'live score|today score|final score|who won|result of|'
            r'yesterday.+match|last match|last game)\b',
            ql
        ) and not re.search(r'\b(news|breaking|headline|article)\b', ql):
            if re.search(
                r'\b(ipl|cricket|football|soccer|premier league|la liga|bundesliga|'
                r'serie a|ligue 1|champions league|nba|nfl|nhl|mlb|f1|formula|'
                r'tennis|rugby|hockey|basketball|baseball|'
                r'match|game|team|league|cup|tournament|'
                r'juventus|real madrid|barcelona|arsenal|chelsea|liverpool|'
                r'manchester|milan|inter|napoli|dortmund|psg|ajax|'
                r'india|australia|england|pakistan|south africa|new zealand|'
                r'west indies|sri lanka|bangladesh|afghanistan|'
                r'lakers|celtics|warriors|bulls|knicks|heat|nets|nuggets|suns|'
                r'chiefs|patriots|cowboys|eagles|giants|rams)\b',
                ql
            ):
                print(f"[DEBUG] Sports score hard rule matched → sports")
                return "sports"

        _is_knowledge_question = bool(re.search(
            r'^\s*(how|what|why|when|where|can|should|is|are|do|does|did'
            r'|will|would|could|tell me|explain|describe|give me|list)',
            ql
        ))

        if re.search (r'\bnet\s*worth\b|\bpersonal\s+wealth\b|\bhow\s+rich\s+is\b', ql):
            print(f"[DEBUG] Net worth query → internet_search")
            return "internet_search"

        query_embedding = self.embedding_model.encode(query)

        scores = {}
        for tool, tool_embedding in self.tool_embeddings.items():
            scores[tool] = float(
                np.dot(query_embedding, tool_embedding) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding))
            )

        # ── Boost sports_knowledge for knowledge-style sports queries ──
        _sk_sim = float(
            np.dot(query_embedding, self.sports_knowledge_embedding) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(self.sports_knowledge_embedding))
        )
        if _SPORTS_KNOWLEDGE_SIGNALS.search(query) and _SPORTS_TOPIC_SIGNALS.search(query):
            scores["sports_knowledge"] = min(1.0, scores.get("sports_knowledge", 0) + 0.15)
            print(f"[DEBUG] Sports knowledge signal boost: sports_knowledge={scores['sports_knowledge']:.3f}")

        _ACTION_VERB = re.search(
            r'^\s*(open|launch|close|quit|start|run|delete|remove|uninstall'
            r'|create|make|move|copy|rename|set\s+(timer|alarm|reminder)'
            r'|remind\s+me|take\s+screenshot|screenshot|mute|unmute'
            r'|increase|decrease|volume|brightness|shutdown|restart|sleep'
            r'|lock\s+screen|empty\s+trash|mark\s+.+\s+(for|as)'
            r'|add\s+(event|birthday|meeting)|schedule\s+.+\s+on)',
            ql
        )
        if _ACTION_VERB and not _is_knowledge_question:
            scores["system"] = min(1.0, scores["system"] + 0.15)
            print(f"[DEBUG] Action verb boost applied: +0.15 → system={scores['system']:.3f}")

        if _is_knowledge_question:
            scores["system"] = 0.0

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        best_tool, best_score = sorted_scores[0]
        second_tool, second_score = sorted_scores[1]
        gap = best_score - second_score

        print(f"[DEBUG] Semantic scores: { {k: f'{v:.3f}' for k, v in sorted_scores[:5]} }")
        print(f"[DEBUG] Best: {best_tool} ({best_score:.3f}), Gap: {gap:.3f}")

        MIN_CONFIDENCE = 0.15
        CLEAR_GAP      = 0.06

        if best_score >= MIN_CONFIDENCE and gap >= CLEAR_GAP:
            print(f"[DEBUG] Clear winner → {best_tool}")
            return best_tool

        print(f"[DEBUG] Ambiguous (gap={gap:.3f}, score={best_score:.3f}) → LLM router")
        llm_tool = self._llm_route_tool(query)
        print(f"[DEBUG] LLM selected → {llm_tool}")
        return llm_tool

    def _route_tools(self, query: str) -> List[str]:
        q = query.lower()

        # VS queries → sports (live H2H)
        if (
            re.search(r'(vs\.?|versus)', q) and
            not re.search(
                r'(stock|share|price|crypto|bitcoin|ethereum|compare|impact|'
                r'affect|news|breaking|war|politics|economy|market|weather)', q
            )
        ):
            print(f"[DEBUG] Sports VS query early-exit → [sports]")
            return ["sports"]

        # Historical tournament winner → sports_knowledge
        if (
            re.search(r'\bwho won\b', q) and
            re.search(r'\b(20\d{2})\b|\b(world cup|tournament|series|championship|final|ipl|league)\b', q) and
            not re.search(r'\b(last night|yesterday|today|this week|just now|recent|last match|last game|live)\b', q)
        ):
            print(f"[DEBUG] Historical winner query → sports_knowledge")
            return ["sports_knowledge"]
        if re.search(r'\bnet\s*worth\b|\bpersonal\s+wealth\b|\bhow\s+rich\s+is\b', q):
            print(f"[DEBUG] Net worth → internet_search early exit")
            return ["internet_search"]


        GITHUB_INTERACTIVE = re.compile(
            r'\b(upload|push\s+file|git\s+push|create\s+repo|new\s+repo|make\s+repo'
            r'|delete\s+repo|delete\s+repository|remove\s+repo|remove\s+repository'
            r'|list\s+repos|my\s+repos|pull\s+request|open\s+issues'
            r'|git\s+help|git\s+commands|git\s+cheatsheet|github)\b'
            r'|push\s+\w+\.\w+'
            # All orderings of delete + repo
            r'|\b(delete|remove)\b.{0,50}\b(repo|repository)\b'
            r'|\b(repo|repository)\b.{0,50}\b(delete|remove)\b'
            r'|\bfrom\b.{0,30}\b(repo|repository)\b.{0,30}\b(delete|remove)\b',
            re.IGNORECASE
        )
        if re.search(GITHUB_INTERACTIVE, q) and not any(
            re.search(p, q) for p in [r'\band\b', r'\balso\b', r'\balong with\b',
                                       r'\bplus\b', r'\btogether\b', r'\bas well as\b']
        ):
            return ["github"]

        if re.search(
            r'\b(remind\s+me|set\s+a\s+reminder|set\s+reminder|reminder\s+at'
            r'|remind\s+me\s+at|remind\s+me\s+in|reminder\s+for'
            r'|reminder\s+to|remind\s+me\s+to)\b',
            q
        ):
            print(f"[DEBUG] Reminder early-exit → [system]")
            return ["system"]

        if re.search(
            r'^\s*(create|make|new)\b.{0,30}\b(file|folder|directory|dir'
            r'|python|java|javascript|js|typescript|ts|html|css|cpp|ruby|bash'
            r'|go|rust|markdown|json|yaml|sql|txt|text)\b'
            r'|^\s*(create|make|new)\s+(?:a\s+)?(?:new\s+)?folder\b',
            q
        ):
            print(f"[DEBUG] File/folder creation early-exit → [system]")
            return ["system"]

        SYSTEM_INTERACTIVE = r'\b(shutdown|restart|reboot|sleep|hibernate|format|delete|remove)\b'
        if re.search(SYSTEM_INTERACTIVE, q) and not any(
            re.search(p, q) for p in [r'\band\b', r'\balso\b', r'\balong with\b',
                               r'\bplus\b', r'\btogether\b', r'\bas well as\b']
        ):
            _is_github_context = bool(re.search(
                r'\b(repo|repository|github)\b.{0,50}\b(delete|remove)\b'
                r'|\b(delete|remove)\b.{0,50}\b(repo|repository|github)\b'
                r'|\bfrom\b.{0,30}\b(repo|repository|github)\b',
                q, re.IGNORECASE
            ))
            if not _is_github_context:
                return ["system"]

        STRONG_COMPOUND = [
            r'\band\b', r'\balso\b', r'\balong with\b',
            r'\bplus\b', r'\btogether\b', r'\bas well as\b',
            r'\bcombined with\b',
        ]
        WEAK_COMPOUND = [
            r'\bimpact\b', r'\baffect\b', r'\baffects\b',
            r'\bcompare\b', r'\bvs\b', r'\bversus\b',
            r'\bboth\b',
        ]

        has_strong_compound = any(re.search(p, q) for p in STRONG_COMPOUND)

        _is_sports_h2h = has_strong_compound and bool(re.search(
            r'(score|result|match|game|vs|between|fixture|upcoming|next|won|goal|played)',
            q
        )) and bool(re.search(
            r'(ipl|cricket|football|soccer|premier league|la liga|bundesliga|'
            r'serie a|ligue 1|champions league|nba|nfl|nhl|mlb|'
            r'match|game|team|league|cup|tournament|'
            r'india|australia|england|pakistan|'
            r'real madrid|barcelona|arsenal|chelsea|liverpool|manchester|'
            r'milan|inter|juventus|dortmund|psg|atletico|'
            r'lakers|celtics|warriors|bulls)',
            q
        ))

        if has_strong_compound and not _is_sports_h2h:
            tools = self._llm_detect_multi_tool(query)
            return tools

        if _is_sports_h2h:
            print(f"[DEBUG] Sports H2H query — skipping multi-tool, routing to sports only")
            return [self._route_tool(query)]

        has_weak_compound = any(re.search(p, q) for p in WEAK_COMPOUND)

        if has_weak_compound:
            DOMAIN_SIGNALS = {
                "stocks":          r'\b(stock price|share price|sensex|nifty|nasdaq|s&p|market cap|aapl|tsla)\b',          
                "news":            r'\b(news|war|conflict|sanction|crisis|geopolit|event|happen|impact|affect|latest|breaking|headline|tension|attack)\b',
                "crypto":          r'\b(bitcoin|btc|ethereum|eth|crypto|coin|blockchain|token|solana|dogecoin|ripple|xrp)\b',
                "weather":         r'\b(weather|temperature|forecast|rain|humidity|wind|climate|monsoon|heatwave|cold|hot|sunny)\b',
                "sports":          r'\b(score today|live score|match today|playing now|latest score|last match result)\b',
                "sports_knowledge":r'\b(career|all.?time|most goals|most wickets|most runs|record|rules|offside|lbw|biography|how many goals|how many runs|how many wickets|tournament winner|world cup winner|ballon d)\b',
                "internet_search": r'\b(who is|what is|explain|biography|history|remarkable|achievement|works|founded|invented|discover|definition|meaning)\b',
                "kb":              r'\b(how to|install|code|script|terminal|command|python|git|docker|debug|error|setup|configure|programming)\b',
                "github":          r'\b(github|repo|repository|branch|pull request|issue|commit|push|upload file)\b',
            }
            matched_domains = [
                domain for domain, pattern in DOMAIN_SIGNALS.items()
                if re.search(pattern, q)
            ]
            print(f"[DEBUG] Weak compound domains matched: {matched_domains}")

            if len(matched_domains) >= 2:
                tools = self._llm_detect_multi_tool(query)
                return tools
            return [self._route_tool(query)]

        return [self._route_tool(query)]

    # =========================================================================
    # QUERY EXPANSION & DECOMPOSITION
    # =========================================================================
    def _expand_query(self, query: str) -> str:
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                "role": "user",
                "content": (
                    f"Convert this question into a short, precise search engine query "
                    f"of 3-8 keywords. Remove question words (who, what, how, why). "
                    f"Keep all specific names, places, years, and topics. "
                    f"If the query is about current status or recent data, append {time.strftime('%Y')} to the search query. "
                    f"Return ONLY the search query — no explanation, no prefix, no punctuation at end.\n\n"
                    f"Examples:\n"
                    f"'who is the president of india' → president India {time.strftime('%Y')}\n"
                    f"'who is the ceo of openai' → OpenAI CEO {time.strftime('%Y')}\n"
                    f"'highest goal scorer 2025 football' → top football goal scorer 2025\n"
                    f"'who stopped most penalties in football history' → most penalty saves football history record\n"
                    f"'what is machine learning' → machine learning definition overview\n\n"
                    f"Question: {query}\n"
                    f"Search query:"
                )
            }],
            options={"temperature": 0.0, "num_predict": 20}
        )
        expanded = response["message"]["content"].strip()
        expanded = re.sub(
            r"^(expanded query:|search query:|answer:|query:)\s*",
            "", expanded, flags=re.IGNORECASE
        ).strip().rstrip("?.!")
        print(f"[DEBUG] Expanded query: {expanded}")
        return expanded if expanded and len(expanded) > 3 else query

    def _clean_query(self, query: str) -> str:
        q = query.strip()
        words = q.split()
        if len(words) <= 1:
            return q

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Fix ONLY clear spelling/typing errors in this query.\n"
                        f"STRICT RULES:\n"
                        f"- Do NOT change meaning, intent, or structure.\n"
                        f"- Do NOT add new words that are not in the original.\n"
                        f"- Do NOT expand abbreviations (e.g. 'pm' stays 'pm', not 'prime minister').\n"
                        f"- Do NOT reinterpret topic names — 'iran war' must stay 'iran war',\n"
                        f"  NEVER change it to 'Iran-Iraq War' or any other interpretation.\n"
                        f"- Proper nouns (country names, company names, people's names) must\n"
                        f"  stay EXACTLY as typed unless they are an obvious letter-scramble.\n"
                        f"- If there are no clear typos, return the query exactly as-is.\n"
                        f"Return ONLY the corrected query. No explanation. No punctuation changes.\n\n"
                        f"Examples:\n"
                        f"  'what is the latest news regardding lpg shortage in india' "
                        f"→ 'what is the latest news regarding lpg shortage in india'\n"
                        f"  'waether in dlehi today' → 'weather in delhi today'\n"
                        f"  'realme madrid scor lasst nite' → 'real madrid score last night'\n"
                        f"  'news related to iran war' → 'news related to iran war'\n"
                        f"  'what is the news related to iran war' → 'what is the news related to iran war'\n"
                        f"  'who si the pm of india' → 'who is the pm of india'\n\n"
                        f"Query: {query}\n"
                        f"Corrected:"
                    )
                }],
                options={"temperature": 0.0, "num_predict": 50}
            )
            cleaned = response["message"]["content"].strip()
            cleaned = cleaned.strip('\'"')
            cleaned = re.sub(
                r'^(query:|corrected:|answer:|fixed:|output:)\s*',
                '', cleaned, flags=re.IGNORECASE
                ).strip()
            cleaned = _clean_query_guard(query, cleaned)
            if cleaned and len(cleaned.split()) >= 1:
                if cleaned.lower() != query.lower():
                    print(f"[DEBUG] Query cleaned: {repr(query)} → {repr(cleaned)}")
                return cleaned
        except Exception:
            pass
        return q

    def _correct_sports_team_name(self, query: str) -> str:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Fix ONLY misspelled sports team or player names in this query.\n"
                        f"RULES:\n"
                        f"- Only fix team names and player names that are clearly misspelled.\n"
                        f"- Do NOT change the intent, structure, or any other words.\n"
                        f"- Do NOT add or remove words.\n"
                        f"- If no team/player name is misspelled, return the query exactly as-is.\n"
                        f"- Return ONLY the corrected query. Nothing else.\n\n"
                        f"Examples:\n"
                        f"  'what is the latest score of new castel' → 'what is the latest score of newcastle'\n"
                        f"  'barcalona vs reel madrid score' → 'barcelona vs real madrid score'\n"
                        f"  'liverpoool latest score' → 'liverpool latest score'\n"
                        f"  'manchster city score today' → 'manchester city score today'\n"
                        f"  'spoting cp latest score' → 'sporting cp latest score'\n"
                        f"  'chealsea score yesterday' → 'chelsea score yesterday'\n"
                        f"  'who won india vs austrlia' → 'who won india vs australia'\n"
                        f"  'next ipl match' → 'next ipl match'\n\n"
                        f"Query: {query}\n"
                        f"Corrected:"
                    )
                }],
                options={"temperature": 0.0, "num_predict": 60}
            )
            corrected = response["message"]["content"].strip().strip('\'"')
            corrected = re.sub(
                r'^(query:|corrected:|answer:|fixed:|output:)\s*',
                '', corrected, flags=re.IGNORECASE
            ).strip()
            if len(corrected.split()) > len(query.split()) + 2:
                return query
            if corrected and corrected.lower() != query.lower():
                print(f"[DEBUG] Sports team name corrected: {repr(query)} → {repr(corrected)}")
            return corrected if corrected else query
        except Exception as e:
            print(f"[DEBUG] _correct_sports_team_name failed: {e}")
            return query
        
    def _classify_sports_query(self, query: str) -> dict:
        """
        Single LLM call. Classifies all query properties needed by the three
        sports methods. Result is passed as props= argument to each method.
 
        Returns safe defaults if LLM call fails — nothing downstream breaks.
        """
        _SAFE_DEFAULTS = {
            "sport": "other",
            "stat_type": "other",
            "is_generic_record": False,
            "is_entity_specific": False,
            "is_tournament": False,
            "cricket_format": "all",
            "specific_year": None,
            "entity_type":  "none",
        }
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f'Classify this sports query. Return ONLY valid JSON, nothing else.\n\n'
                        f'Query: "{query}"\n\n'
                        f'{{\n'
                        f'  "sport": "<sport in lowercase: football, cricket, tennis, basketball, rugby, f1, hockey, baseball, golf, swimming, athletics, volleyball, kabaddi, or other>",\n'
                        f'  "stat_type": "<stat in lowercase: goals, runs, wickets, points, assists, titles, wins, centuries, tries, aces, touchdowns, appearances, or other>",\n'
                        f'  "is_generic_record": <true if asking who has MOST/HIGHEST/ALL-TIME across the sport — false otherwise>,\n'
                        f'  "is_entity_specific": <true if asking stats FOR or AT a specific club/team/country — false otherwise>,\n'
                        f'  "is_tournament": <true if asking about teams/squads/fixtures/schedule of a specific tournament — false otherwise>,\n'
                        f'  "cricket_format": "<test, odi, t20, or all — use all when no format mentioned>",\n'
                        f'  "specific_year": "<4-digit year if mentioned in query, else null>",\n'
                        f'  "entity_type": "<ONLY when is_entity_specific=true: national, club, or tournament — national=country/national team, club=club/franchise/team, tournament=competition venue>"\n'
                        '}}\n\n'
                        f'RULES:\n'
                        f'- is_entity_specific MUST be false when is_generic_record is true\n'
                        f'- cricket_format is always "all" when query says career/total/international or no format mentioned\n'
                        f'- entity_type is "none" when is_entity_specific is false\n'
                        f'- entity_type "national": India, England, Australia, Portugal, Brazil, Argentina,\n'
                        f'  France, Germany, USA, New Zealand, Pakistan, Sri Lanka, West Indies, etc.\n'
                        f'- entity_type "club": RCB, CSK, Mumbai Indians, Barcelona, Real Madrid, Man City,\n'
                        f'  PSG, Bayern, Lakers, Celtics, Warriors, Mercedes, Ferrari, etc.\n'
                        f'- entity_type "tournament": IPL, Champions League, World Cup, Wimbledon, US Open, etc.\n\n'
                        f'EXAMPLES:\n'
                        f'"most goals in football history" → is_generic_record=true, entity_type="none"\n'
                        f'"messi goals for barcelona" → is_entity_specific=true, entity_type="club"\n'
                        f'"messi goals for argentina" → is_entity_specific=true, entity_type="national"\n'
                        f'"kohli runs for india" → is_entity_specific=true, entity_type="national", cricket_format="all"\n'
                        f'"kohli runs for rcb" → is_entity_specific=true, entity_type="club"\n'
                        f'"kohli test runs" → cricket_format="test", entity_type="none"\n'
                        f'"mbappe goals for real madrid" → is_entity_specific=true, entity_type="club"\n'
                        f'"rashford goals for england" → is_entity_specific=true, entity_type="national"\n'
                        f'"hamilton wins for mercedes" → is_entity_specific=true, entity_type="club"\n'
                        f'"teams in ipl 2024" → is_tournament=true, specific_year="2024"\n'
                        f'"most tries in rugby history" → sport="rugby", is_generic_record=true\n'
                        f'"federer wins at wimbledon" → is_entity_specific=true, entity_type="tournament"\n'
                        f'Only the JSON. Nothing else.'
                    )
                }],
                options={"temperature": 0.0, "num_predict": 150}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            # Extract JSON object even if LLM added surrounding text
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            parsed = json.loads(raw)
 
            # Enforce: entity_specific cannot be true when generic_record is true
            if parsed.get("is_generic_record") and parsed.get("is_entity_specific"):
                parsed["is_entity_specific"] = False
 
            # Fill any missing keys with safe defaults
            for key, default in _SAFE_DEFAULTS.items():
                if key not in parsed:
                    parsed[key] = default
 
            # Normalize sport to lowercase string
            parsed["sport"] = str(parsed.get("sport") or "other").lower().strip()
            parsed["stat_type"]  = str(parsed.get("stat_type")  or "stat" ).lower().strip()
            parsed["entity_type"]= str(parsed.get("entity_type")or "none" ).lower().strip()
 
            # Normalize cricket_format
            cf = str(parsed.get("cricket_format") or "all").lower().strip()
            parsed["cricket_format"] = cf if cf in ("test", "odi", "t20", "all") else "all"
 
            # Normalize specific_year to string or None
            if not parsed.get("is_entity_specific"):
                parsed["entity_type"] = "none"
 
            # Validate entity_type value
            if parsed["entity_type"] not in ("national", "club", "tournament", "none"):
                parsed["entity_type"] = "club"
            yr = parsed.get("specific_year")
            if yr:
                try:
                    parsed["specific_year"] = str(int(str(yr).strip()))
                except Exception:
                    parsed["specific_year"] = None
            else:
                parsed["specific_year"] = None
 
            # Normalize stat_type
            parsed["stat_type"] = str(parsed.get("stat_type") or "stat").lower().strip()
 
            print(f"[DEBUG] _classify_sports_query: {parsed}")
            return parsed
 
        except Exception as e:
            print(f"[DEBUG] _classify_sports_query failed: {e} — using safe defaults")
            return _SAFE_DEFAULTS


    def _get_timelimit(self, query: str, intent: str, props: dict = None):
        """
        Returns DDG timelimit string or None.
        Props come from _classify_sports_query.
        Falls back gracefully when props not provided.
        """
        props = props or {}
        _current_year = int(time.strftime('%Y'))
 
        # Rule 1: Past year in query → fetch that era, no recency filter
        specific_year = props.get("specific_year")
        if specific_year:
            try:
                if int(specific_year) < _current_year - 1:
                    print(f"[DEBUG] _get_timelimit: past year {specific_year} → None")
                    return None
            except Exception:
                pass
 
        # Rule 2: Scientific facts never change
        if intent == "scientific":
            print(f"[DEBUG] _get_timelimit: scientific → None")
            return None
 
        # Rule 3: Any known sport or tournament → past month (freshest DDG results)
        sport = props.get("sport", "other")
        if sport != "other" or props.get("is_tournament"):
            print(f"[DEBUG] _get_timelimit: sport={sport} → 'm'")
            return "m"
 
        # Rule 4: No props provided — minimal regex fallback for obvious sports
        if not props or props == {}:
            if re.search(
                r'\b(football|soccer|cricket|tennis|basketball|rugby|hockey|'
                r'baseball|golf|f1|formula|ipl|nba|nfl|nhl|mlb|atp|wta|'
                r'world cup|tournament|championship|league|series|grand slam)\b',
                query, re.IGNORECASE
            ):
                return "m"
 
        # Rule 5: General knowledge intents → past year
        if intent in ("sports_knowledge", "facts", "knowledge", "ownership", "news"):
            print(f"[DEBUG] _get_timelimit: intent={intent} → 'y'")
            return "y"
 
        return None
    

    def _decompose_query(self, query: str, tools: List[str]) -> dict:
        tool_descriptions = {
            "stocks":          "stock prices, share price, market indices",
            "news":            "current events, breaking news, geopolitical impact on markets",
            "crypto":          "cryptocurrency prices and market data",
            "weather":         "weather forecast for a location",
            "sports":          "live scores, match results today, upcoming fixtures",
            "sports_knowledge":"career stats, records, rules, tournament history, player facts, bios",
            "system":          "computer actions: open/close apps, move files, create files or folders, timer, alarm, reminder, time, date, volume, brightness, screenshot",
            "github":          "GitHub repos, branches, issues, pull requests, git help",
            "kb":              "technical how-to, programming, terminal commands",
            "internet_search": "general knowledge, definitions, history, science",
        }

        tool_lines = "\n".join(
            f"  {t} — {tool_descriptions.get(t, t)}"
            for t in tools
        )

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"A user asked: \"{query}\"\n\n"
                        f"Split this into exactly {len(tools)} focused sub-questions, "
                        f"one per tool listed below. Each sub-question must:\n"
                        f"- Be self-contained and answerable on its own\n"
                        f"- Preserve ALL specific entity names (company, country, coin, city) "
                        f"from the original query\n"
                        f"- Be targeted only at what that tool does\n\n"
                        f"Tools:\n{tool_lines}\n\n"
                        f"Reply in this EXACT format (tool name then colon then sub-question), "
                        f"one per line, nothing else:\n"
                        + "\n".join(f"{t}: <sub-question>" for t in tools)
                        + f"\n\nExamples:\n"
                        f"Query: 'reliance stock price and how does iran war impact it'\n"
                        f"stocks: What is the current stock price of Reliance Industries today?\n"
                        f"news: How does the Iran war impact Reliance Industries share price?\n\n"
                        f"Query: 'ipl score today and how many ipl titles has mumbai indians won'\n"
                        f"sports: What is the IPL score today?\n"
                        f"sports_knowledge: How many IPL titles has Mumbai Indians won in total?\n\n"
                        f"Now decompose: \"{query}\""
                    )
                }],
                options={"temperature": 0.0, "num_predict": 120}
            )

            raw = response["message"]["content"].strip()
            print(f"[DEBUG] Query decomposition raw output:\n{raw}")

            sub_questions = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                for t in tools:
                    if re.match(rf'^{re.escape(t)}\s*[:\-—]\s*', line, re.IGNORECASE):
                        sub_q = re.sub(rf'^{re.escape(t)}\s*[:\-—]\s*', '', line, flags=re.IGNORECASE).strip()
                        if sub_q:
                            sub_questions[t] = sub_q
                        break

            for t in tools:
                if t not in sub_questions:
                    print(f"[DEBUG] Decomposition missing tool '{t}' — using original query")
                    sub_questions[t] = query

            print(f"[DEBUG] Sub-questions: {sub_questions}")
            return sub_questions

        except Exception as e:
            print(f"[DEBUG] Decomposition failed ({e}) — using original query for all tools")
            return {t: query for t in tools}

    def _decompose_single_query(self, query: str, tool: str) -> List[str]:
        COMPOUND_SIGNALS = [
            r'\band\b', r'\balso\b', r'\balong with\b', r'\bplus\b',
            r'\btogether\b', r'\bas well as\b',
            r'\bwhat are\b', r'\bwhat is\b.*\band\b',
            r'\bwho is\b.*\band\b', r'\bhow does\b.*\band\b',
        ]
        has_compound = any(re.search(p, query.lower()) for p in COMPOUND_SIGNALS)
        if not has_compound:
            return [query]

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"A user asked: \"{query}\"\n\n"
                        f"Split this into 2-3 focused retrieval search phrases "
                        f"(short keyword phrases, NOT full questions).\n\n"
                        f"Rules:\n"
                        f"- Each phrase covers a DIFFERENT aspect of the question\n"
                        f"- Preserve ALL specific names (person, company, country, place)\n"
                        f"- Keep each phrase to 3-8 keywords\n"
                        f"- If the query is truly single-aspect, return only 1 line\n\n"
                        f"Reply with ONLY numbered lines, nothing else.\n"
                        f"Query: \"{query}\""
                    )
                }],
                options={"temperature": 0.0, "num_predict": 80}
            )
            raw = response["message"]["content"].strip()
            print(f"[DEBUG] Single-query decomposition:\n{raw}")

            sub_queries = []
            for line in raw.splitlines():
                line = line.strip()
                line = re.sub(r"^\d+[:.)]\s*", "", line).strip()
                if line and len(line) > 3:
                    sub_queries.append(line)

            if sub_queries:
                print(f"[DEBUG] Sub-queries: {sub_queries}")
                return sub_queries[:3]

        except Exception as e:
            print(f"[DEBUG] Single-query decomposition failed ({e})")

        return [query]

    def _generate_format_subqueries(self, query: str) -> List[str]:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                "role": "user",
                "content": (
                    f"A user asked: \"{query}\"\n\n"
                    f"This query spans multiple sub-categories or formats.\n"
                    f"Generate 3-4 specific search queries, one per sub-category,\n"
                    f"that would help find the record holder for each sub-category.\n\n"
                    f"Rules:\n"
                    f"- Each query must be 3-7 keywords\n"
                    f"- Each query targets a DIFFERENT sub-category\n"
                    f"- Keep the core topic (wickets/runs/goals/saves etc.) intact\n"
                    f"- Return ONLY numbered lines, nothing else\n\n"
                    f"Examples:\n"
                    f"Query: 'most wickets in cricket across all formats'\n"
                    f"1. most wickets test cricket all time record\n"
                    f"2. most wickets ODI cricket all time record\n"
                    f"3. most wickets T20I cricket all time record\n\n"
                    f"Query: \"{query}\"\n"
                    f"Sub-queries:"
                )
                }],
                options={"temperature": 0.0, "num_predict": 80}
            )
            raw = response["message"]["content"].strip()
            sub_queries = []
            for line in raw.splitlines():
                line = line.strip()
                line = re.sub(r"^\d+[:.)]\s*", "", line).strip()
                if line and len(line) > 3:
                    sub_queries.append(line)
            print(f"[DEBUG] _generate_format_subqueries: {sub_queries}")
            return sub_queries[:4]
        except Exception as e:
            print(f"[DEBUG] _generate_format_subqueries failed: {e}")
            return []

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    def semantic_search(self, query: str, k=RETRIEVAL_TOP_K) -> List[str]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"][0]

    def _is_career_stats_query(self, query: str) -> bool:
        return bool(re.search(
            r'\b(how many|how much|career|total|all.?time|most|highest|'
            r'record|tally|ever|in history|most goals|most runs|most wickets|'
            # ADD these tournament-related patterns:
            r'squad|which teams?|how many teams?|participating teams?|'
            r'teams? (in|for|at)|who (are|is) (playing|participating)|'
            r'qualified for|fixture|group stage|knockout|draw|'
            r'batting average|bowling average|who scored most|who has most|'
            r'who holds|who has the most|who has scored|who has taken|'
            r'most points|most assists|most appearances|most caps|most titles|'
            r'most grand slams|most championships|most wins|most matches|'
            r'most centuries|most fifties|most hat tricks|most penalties|'
            r'most clean sheets|most saves|most yellow cards|most red cards|'
            r'most in history|most in the history|most of all time|'
            r'record holder|record for most|holds the record|'
            r'all time record|all-time record|all time leading|'
            r'leading scorer|top scorer|highest scorer|'
            r'best batting|best bowling|best strike rate|'
            r'fastest century|fastest fifty|fastest goal|'
            r'youngest|oldest player|first player to|'
            r'how many times|how many seasons|how many tournaments)\b',
            query.lower()
        ))

    def _build_career_search_queries(self, query: str, props: dict = None):
        """
        Builds up to 5 targeted search queries.
        Works for any sport, any stat, any entity name.
        Uses year extracted from query when user asks about a specific past year.
        """
        props = props or {}
        _current_year = time.strftime('%Y')
        _prev_year = str(int(_current_year) - 1)
 
        # Determine search year — use year from query if it refers to the past
        specific_year = props.get("specific_year")
        if specific_year:
            try:
                _search_year = (
                    specific_year
                    if int(specific_year) < int(_current_year) - 1
                    else _current_year
                )
            except Exception:
                _search_year = _current_year
        else:
            _search_year = _current_year
 
        base_query = query.strip().rstrip('?.')
        is_tournament = props.get("is_tournament", False)
        is_entity_specific = props.get("is_entity_specific", False)
 
        # ── Tournament queries ────────────────────────────────────────────────
        if is_tournament:
            queries = [
                f"{base_query} {_search_year}",
                f"{base_query} official list {_search_year}",
                f"{base_query} complete {_search_year}",
                f"{base_query} all teams {_search_year}",
                f"{base_query} {_search_year} wikipedia",
            ]
            print(f"[DEBUG] Tournament queries (year={_search_year}): {queries}")
            return self._finalize_queries(queries, _search_year, _prev_year)
        _is_cricket_intl_career = (
            props.get("sport") == "cricket" and
            not props.get("is_generic_record") and
            not is_entity_specific
        )
        if _is_cricket_intl_career:
            _STOP = {'how', 'many', 'runs', 'has', 'scored', 'for', 'india',
                    'total', 'career', 'much', 'wickets', 'goals', 'international',
                    'what', 'is', 'the', 'a', 'an', 'in', 'of', 'odi', 'test',
                    't20', 't20i', 'across', 'all', 'formats', 'cricket', 'odis'}
            _words = [w for w in query.lower().split() if w not in _STOP and len(w) > 2]
            _player = " ".join(_words[:2]).title() if _words else "the player"

            _q = query.lower()
            _wants_t20  = bool(re.search(r'\bt20i?\b|\bt20 international', _q))
            _wants_test = bool(re.search(r'\btest(s)?\b', _q))
            _wants_odi  = bool(re.search(r'\bodi(s)?\b|\bone.?day\b', _q))

            if _wants_t20 and not _wants_test and not _wants_odi:
                queries = [f"How many runs has {_player} scored in T20 Internationals"]
            elif _wants_test and not _wants_t20 and not _wants_odi:
                queries = [f"How many runs has {_player} scored in Tests"]
            elif _wants_odi and not _wants_t20 and not _wants_test:
                queries = [f"How many runs has {_player} scored in ODIs"]
            else:
                queries = [
                    f"How many runs has {_player} scored in T20 Internationals",
                    f"How many runs has {_player} scored in Tests",
                    f"How many runs has {_player} scored across all formats of cricket",
                ]

            print(f"[DEBUG] Cricket fixed queries for '{_player}': {queries}")
            return queries

 
        # ── Entity-specific queries ───────────────────────────────────────────
        _is_past_club = False
        if is_entity_specific and specific_year is None:
            _is_past_club = bool(re.search(
                r'\b(career at|time at|during his|while at|'
                r'spell at|stint at|played for|his years at|'
                r'when he was at|during his time at|his stint)\b',
                query, re.IGNORECASE
            ))
        if is_entity_specific:
            if _is_past_club:
                queries = [
                    f"__YEARFREE__{base_query} career total all competitions",
                    f"{base_query} {_search_year}",
                    f"__YEARFREE__{base_query} stats career",
                    f"__YEARFREE__{base_query} all competitions total",
                    f"{base_query} career breakdown {_search_year}",
                ]
            else:
                queries = [
                    f"{base_query} {_search_year}",
                    f"__YEARFREE__{base_query} all time total",  # ← sentinel
                    f"{base_query} stats {_search_year}",
                    f"{base_query} total goals all competitions {_search_year}",
                    f"{base_query} career {_search_year}",  
                ]
            print(f"[DEBUG] Entity-specific queries: {queries}")
            return self._finalize_queries(queries, _search_year, _prev_year)
        

 
        # ── Generic record and player-specific: LLM generates queries ─────────
        # No sport-specific templates — works for any sport
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f'Convert this question into 3 short search engine queries.\n'
                        f'Search year: {_search_year}\n\n'
                        f'Rules:\n'
                        f'- Each query must be 4-8 keywords\n'
                        f'- Each query MUST include the year {_search_year}\n'
                        '- Remove question words (who, what, how, did, does)\n'
                        f'- Keep ALL names, sport names, and stat keywords from the question\n'
                        f'- Do NOT add any names or words not in the question\n'
                        f'- Return ONLY numbered lines, nothing else\n\n'
                        f'Question: "{query}"\n'
                        f'Queries:'
                    )
                }],
                options={"temperature": 0.0, "num_predict": 100}
            )
            raw = response["message"]["content"].strip()
            llm_queries = []
            for line in raw.splitlines():
                line = re.sub(r"^\d+[:.)]\s*", "", line.strip()).strip()
                if line and len(line) > 3:
                    if _search_year not in line and _prev_year not in line:
                        line = f"{line} {_search_year}"
                    llm_queries.append(line)
            llm_queries = llm_queries[:3]
        except Exception as e:
            print(f"[DEBUG] _build_career_search_queries LLM failed: {e}")
            llm_queries = []
 
        all_queries = [
            f"{base_query} {_search_year}",
            f"{base_query} latest updated {_search_year}",
        ] +llm_queries
 
        print(f"[DEBUG] _build_career_search_queries raw: {all_queries}")
        return self._finalize_queries(all_queries, _search_year, _prev_year)
 
 
    def _finalize_queries(self, queries: list, year: str, prev_year: str) -> list:
        """Deduplicate. Queries prefixed __YEARFREE__ are kept without year injection."""
        seen = set()
        final = []
        for q in queries:
            is_year_free = q.startswith("__YEARFREE__")
            if is_year_free:
                q = q[len("__YEARFREE__"):]   # strip sentinel
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                if not is_year_free and year not in q and prev_year not in q:
                    q = f"{q} {year}"
                final.append(q)
        return final[:5]
    

    def keyword_search(self, query: str, k=RETRIEVAL_TOP_K) -> List[str]:
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        return [self.documents[i] for i in ranked_indices[:k]]

    def hybrid_kb_search(self, query: str, k=RETRIEVAL_TOP_K) -> List[str]:
        semantic_docs = self.semantic_search(query, k)
        if ENABLE_HYBRID_SEARCH:
            keyword_docs = self.keyword_search(query, k)
            combined = list(dict.fromkeys(semantic_docs + keyword_docs))
            return combined[:k]
        return semantic_docs[:k]

    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================
    def add_to_memory(self, user_input: str, assistant_response: str):
        self.conv_memory.append({
            "user": user_input,
            "assistant": assistant_response
        })

    def clear_memory(self):
        self.conv_memory.clear()
        self._kv_cache = None
        self._kv_cache_key = None
        print("Conversational memory and KV cache cleared.")

    def delete_last_turn(self):
        if self.conv_memory:
            removed = self.conv_memory.pop()
            self._kv_cache = None
            self._kv_cache_key = None
            print(f"Deleted last turn — User: {removed['user'][:50]}...")
        else:
            print("Memory is already empty.")

    def get_memory_summary(self) -> str:
        if not self.conv_memory:
            return "Memory is empty."
        lines = []
        for i, turn in enumerate(self.conv_memory, 1):
            lines.append(f"[Turn {i}] User: {turn['user'][:60]}...")
            lines.append(f"         Assistant: {turn['assistant'][:60]}...")
        return "\n".join(lines)

    # =========================================================================
    # SESSION TIMEOUT
    # =========================================================================
    def _check_session_timeout(self):
        current_time = time.time()
        elapsed = current_time - self._last_interaction_time
        if elapsed > self._session_timeout and self.conv_memory:
            print("\n[Session timeout — memory auto-cleared]\n")
            self.clear_memory()
        self._last_interaction_time = current_time

    # =========================================================================
    # KV CACHE
    # =========================================================================
    def _build_message_list(self, user_input: str) -> List[dict]:
        cache_key = str([
            (t["user"], t["assistant"]) for t in self.conv_memory
        ])

        if self._kv_cache is not None and self._kv_cache_key == cache_key:
            messages = self._kv_cache.copy()
            messages.append({"role": "user", "content": user_input})
            return messages

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in self.conv_memory:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        self._kv_cache = messages.copy()
        self._kv_cache_key = cache_key
        messages.append({"role": "user", "content": user_input})
        return messages

    def _trim_messages_to_token_limit(self, messages: List[dict]) -> List[dict]:
        system_message = messages[0]
        last_message = messages[-1]
        history = messages[1:-1]

        trimmed = []
        total_tokens = 0
        reserved = (
            len(last_message["content"].split()) +
            len(system_message["content"].split())
        )

        for msg in reversed(history):
            tokens = len(msg["content"].split())
            if total_tokens + tokens + reserved > MAX_CONTEXT_TOKENS:
                break
            trimmed.insert(0, msg)
            total_tokens += tokens

        return [system_message] + trimmed + [last_message]

    # =========================================================================
    # CONTEXT COMPRESSION
    # =========================================================================
    def compress_context(self, context: str) -> str:
        tokens = context.split()
        if len(tokens) <= MAX_TOKENS:
            return context
        half = MAX_TOKENS // 2
        compressed_tokens = tokens[:half] + tokens[-half:]
        return " ".join(compressed_tokens)

    # =========================================================================
    # FORMAT TOOL OUTPUT
    # =========================================================================
    def format_tool_output(self, tool_output, tool_name: str = "") -> list:
        label = tool_name.upper().replace("_", " ") if tool_name else "DATA"

        if isinstance(tool_output, list) and tool_output and isinstance(tool_output[0], dict):
            blocks = []
            warning_text = None
            for chunk in tool_output:
                if chunk.get("_warning"):
                    warning_text = chunk["text"]
                    continue
                trust_label    = "TRUSTED" if chunk.get("is_trusted") else "unverified"
                reranker_score = chunk.get("reranker_score", "")
                score_str      = f" [score: {reranker_score}]" if reranker_score else ""
                block = (
                    f"[{label}][{trust_label}]{score_str} "
                    f"{chunk.get('title', '')} "
                    f"({chunk.get('source', '')}, {chunk.get('date', '')})\n"
                    f"{chunk.get('text', '')}"
                )
                blocks.append(block)
            if warning_text and not blocks:
                return [f"[{label}] ⚠ {warning_text}"]
            if warning_text:
                blocks.append(f"[{label}] ⚠ {warning_text}")
            return blocks

        if isinstance(tool_output, str):
            text = tool_output.strip()
            if not text or len(text) < 10:
                return [f"[{label}] No data available."]
            return [f"[{label}]\n{text}"]

        if isinstance(tool_output, list) and tool_output and isinstance(tool_output[0], str):
            return [f"[{label}]\n{doc}" for doc in tool_output if doc.strip()]

        return [f"[{label}] No relevant data found."]

    # =========================================================================
    # BUILD CONTEXT
    # =========================================================================
    def build_context(self, blocks: list, original_query: str = "",
                      preprocessed_query: dict = None) -> str:
        if not blocks:
            return "No relevant data found."

        query_for_rerank = original_query or ""

        if self.reranker and len(blocks) > 1 and query_for_rerank:
            try:
                blocks = self.reranker.rerank(
                    query_for_rerank, blocks, top_k=RERANK_TOP_K
                )
                print(f"[DEBUG] build_context: reranker applied on {len(blocks)} blocks")
            except Exception as e:
                print(f"[DEBUG] build_context reranker failed: {e}")

        seen_blocks = set()
        deduped_blocks = []
        for block in blocks:
            content_part = re.sub(r'\[\w+\][^\n]*\n', '', block).strip()
            fp = hashlib.md5(content_part.lower().strip().encode()).hexdigest()
            if fp and fp not in seen_blocks:
                seen_blocks.add(fp)
                deduped_blocks.append(block)
        blocks = deduped_blocks

        packed = ""
        total_tokens = 0
        for block in blocks:
            tokens = len(block.split())
            if total_tokens + tokens > MAX_TOKENS:
                break
            packed += block + "\n\n"
            total_tokens += tokens

        return self.compress_context(packed.strip())

    # =========================================================================
    # KB-SPECIFIC CONTEXT BUILDER
    # =========================================================================
    def build_kb_context(self, kb_docs: list, original_query: str = "",is_document: bool = False) -> str:
        if not kb_docs:
            return "No relevant KB documents found."
        
        _token_budget = MAX_TOKENS * 3 if is_document else MAX_TOKENS

        if self.reranker and len(kb_docs) > 1 and original_query and not is_document:
            try:
                kb_docs = self.reranker.rerank(
                    original_query, kb_docs, top_k=RERANK_TOP_K
                )
                print(f"[DEBUG] build_kb_context: reranked {len(kb_docs)} docs")
            except Exception as e:
                print(f"[DEBUG] build_kb_context reranker failed: {e}")

        packed = ""
        total_tokens = 0
        for doc in kb_docs:
            tokens = len(doc.split())
            if total_tokens + tokens > _token_budget:
                break
            packed += doc + "\n"
            total_tokens += tokens

        return self.compress_context(packed.strip())

    # =========================================================================
    # INTERNET SEARCH
    # =========================================================================
    def internet_search_tool(self, query: str, top_k: int = None,
                              timelimit: str = None) -> List[str]:
        kwargs = {
            "top_k": top_k or SEARCH_RESULTS_LIMIT,
            "embed_model": self.embedding_model,
        }
        if timelimit is not None:
            kwargs["timelimit"] = timelimit
        try:
            snippets = internet_search.fetch_and_rerank(query, **kwargs)
        except TypeError as e:
            if "timelimit" in str(e) or "unexpected keyword argument" in str(e):
                print(f"[DEBUG] internet_search_tool: timelimit not supported, retrying without it")
                kwargs.pop("timelimit", None)
                snippets = internet_search.fetch_and_rerank(query, **kwargs)
            else:
                raise
        if not snippets:
            return ["No relevant internet results found."]
        return snippets

    def _parallel_retrieve(self, sub_queries: List[str], tool: str,
                           top_k: int = None) -> List[str]:
        if len(sub_queries) == 1:
            expanded = self._expand_query(sub_queries[0])
            return self.internet_search_tool(expanded, top_k=top_k or SEARCH_RESULTS_LIMIT)

        def _fetch_one(sub_q: str) -> str:
            expanded = self._expand_query(sub_q)
            return self.internet_search_tool(expanded, top_k=(top_k or SEARCH_RESULTS_LIMIT))
        all_snippets: List[str] = []
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
            futures = {executor.submit(_fetch_one, sq): sq for sq in sub_queries}
            for future in as_completed(futures):
                sq = futures[future]
                try:
                    snippets = future.result()
                    if isinstance(snippets, list):
                        all_snippets.extend(snippets)
                except Exception as e:
                    print(f"[DEBUG] Parallel retrieve failed for '{sq}': {e}")
        seen, unique = set(), []
        for s in all_snippets:
            key = hashlib.md5(s.lower().strip().encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        print(f"[DEBUG] _parallel_retrieve: {len(sub_queries)} sub-queries → "
          f"{len(unique)} unique snippets")
        return unique if unique else ["No relevant information found."]

    # =========================================================================
    # FORMAT HELPERS
    # =========================================================================
    def _get_format_instruction(self, intent: str, query: str, num_points: int = None, fmt: str = None) -> str:
        if num_points:
            return (
                f"## {query.strip().title()}\n"
                f"Give up to {num_points} numbered points about this topic.\n"
                f"Rules:\n"
                f"- Each point must have a bold heading and 1-2 sentences of explanation.\n"
                f"- If the context has fewer than {num_points} distinct facts, give as many as you have.\n"
                f"- If you have fewer points than requested, add this note at the end:\n"
                f"  '> Note: Only X points available from current context.'\n"
                f"- NEVER pad with vague or repeated points just to hit the number.\n"
                f"- NEVER say you cannot answer — always give what you have.\n\n"
                f"Format each point as:\n"
                f"1. **[Heading]** — [explanation]\n"
                f"2. **[Heading]** — [explanation]\n"
            )

        if intent == "ownership":
            return (
                "## Ownership\n"
                "List ALL owners clearly with their roles and stake if mentioned.\n"
                "Format each owner as:\n"
                "- **[Name]** — [Role/Stake]\n"
            )
        elif intent == "knowledge":
            return (
               "Answer directly with concrete facts. Do NOT describe what sources say.\n"
                "WRONG: 'This article discusses...' / 'The text covers...'\n"
                "RIGHT: State the actual facts as if you know them.\n\n"
                "DEPTH RULE: Provide at least 5-6 Key Facts. Each must be a full sentence with specific details.\n"
                "Cover: definition, how it works, key components, advantages, challenges, applications.\n\n"
                "Format:\n"
                "## [Topic Name]\n"
                "**Answer:** [Direct 2-3 sentence answer to the question]\n\n"
                "**Key Facts:**\n"
                "- **[Fact heading]** — [full sentence with specific details]\n"
                "- **[Fact heading]** — [full sentence with specific details]\n"
                "- **[Fact heading]** — [full sentence with specific details]\n"
                "- **[Fact heading]** — [full sentence with specific details]\n"
                "- **[Fact heading]** — [full sentence with specific details]\n"
                
            )
        elif intent == "facts":
            return (
                "Format:\n"
                "## [Fact Type]\n"
                "- **[Label]:** [exact fact from context]\n\n"
                "RULES:\n"
                "- Only state facts explicitly present in the context.\n"
                "- If sources disagree, list each answer separately.\n"
                "- Never guess or fill gaps. If uncertain, say so.\n"
            )
        elif intent == "sports_knowledge":
            import re as _re_fmt
            _q = (query or "").lower()
            _DISCLAIMER = (
                "\n> ⚠️ *Disclaimer: These figures are sourced from the internet and may "
                "not be fully accurate. Please verify from an official source like "
                "ESPNcricinfo, FIFA, or the relevant sports governing body.*\n"
            )
            _is_rule = bool(_re_fmt.search(
                r'\b(what is|explain|how does|what are the rules|offside|lbw|drs|'
                r'var|hat trick|penalty|free kick|corner|handball|'
                r'yellow card|red card|innings|over|wicket|no ball|wide|'
                r'how is|what happens|when is|rule of|definition)\b', _q
            ))
            _is_winner = bool(_re_fmt.search(
                r'\b(who won|winner|champion|won the|ballon|golden boot|'
                r'golden glove|mvp|player of|award|title|trophy)\b', _q
            ))
            _is_bio = bool(_re_fmt.search(
                r'\b(who is|biography|born|nationality|age|career of|profile)\b', _q
            ))

            if _is_rule:
                return (
                    "Give a thorough explanation of this sports rule.\n"
                    "Do NOT use bracket placeholders in output.\n"
                    "Do NOT cite source labels like [SPORTS KB].\n\n"
                    "Format your response as:\n"
                    "One paragraph definition.\n\n"
                    "How it works:\n"
                    "- First condition with precise explanation\n"
                    "- Second condition with precise explanation\n"
                    "- Third condition with precise explanation\n"
                    "- Fourth condition with precise explanation\n\n"
                    "Common examples:\n"
                    "- Real match example\n"
                    "- Exception or edge case\n\n"
                    "RULES:\n"
                    "- Minimum 4 bullet points in how it works section.\n"
                    "- Only include examples and exceptions from context.\n"
                    "- Do NOT invent examples not in context.\n"
                    f"- After your final line, add exactly this disclaimer:\n{_DISCLAIMER}"
                )
            elif _is_winner:
                return (
                    "State the winner directly. Only include details present in context.\n"
                    "Do NOT use bracket placeholders. Do NOT cite source labels.\n"
                    "Do NOT add headings with brackets like ## [Award].\n"
                    "Do NOT invent years, scores, or opponents not in context.\n"
                    f"After answering, add exactly this disclaimer:\n{_DISCLAIMER}\n"
                    "Stop after the disclaimer.\n\n"
                    "EXAMPLE:\n"
                    "Mumbai Indians won IPL 2019, defeating Chennai Super Kings by 1 run.\n"
                )
            elif _is_bio:
                return (
                    "Answer the question directly using only facts from context.\n"
                    "Do NOT use bracket placeholders. Do NOT cite source labels.\n"
                    "Do NOT add headings with brackets. Only include fields present in context.\n"
                    f"After answering, add exactly this disclaimer:\n{_DISCLAIMER}\n"
                    "Stop after the disclaimer.\n"
                )
            else:
                _is_cricket_q = bool(_re_fmt.search(
                    r'\b(cricket|runs?|wickets?|batting|bowling|odi|test|t20|'
                    r'centuries|fifties|innings)\b', _q
                ))
                _is_club_q = bool(_re_fmt.search(
                    r'\b(for|at|with)\s+[a-z]', _q
                )) and not bool(_re_fmt.search(
                    r'\b(india|england|australia|pakistan|portugal|brazil|'
                    r'argentina|france|germany|spain|italy|international|'
                    r'south africa|new zealand|west indies|sri lanka)\b', _q
                ))
                _is_record_q = bool(_re_fmt.search(
                    r'\b(most|highest|all.?time|in history|ever|record|'
                    r'who scored most|leading scorer|top scorer)\b', _q
                ))

                if _is_cricket_q and not _is_club_q:
                    return (
                        "The FACT lines contain a computed total. Write it as the first sentence.\n"
                        "Format: '[Player] has scored [computed total] international runs ([format breakdown]).\n"
                        "Then on the next three lines write the format breakdown:\n"
                        "- Tests: [number]\n"
                        "- ODIs: [number]\n"
                        "- T20Is: [number]\n"
                        "RULES:\n"
                        "- The computed total comes from the line starting with "
                        "'FACT: Total international' — use that number exactly.\n"
                        "- Only include format lines that exist in the FACT lines.\n"
                        "- Do NOT add or subtract numbers yourself.\n"
                        f"- After the last format line, add exactly this disclaimer:\n{_DISCLAIMER}"
                        "- Stop after the disclaimer.\n"
                    )
                elif _is_club_q:
                    return (
                        "Write one sentence stating the player name and how many goals or runs they scored for that specific club.\n"
                        "Then write dash lines for appearances and seasons only if those numbers exist in the FACT lines.\n"
                        f"After the last dash line, add exactly this disclaimer:\n{_DISCLAIMER}"
                        "Stop after the disclaimer."
                    )
                elif _is_record_q:
                    return (
                        "State the record holder and their career total in one sentence.\n"
                        "If a second highest exists in FACT lines, add it on the next line with a dash.\n"
                        f"After that, add exactly this disclaimer:\n{_DISCLAIMER}"
                        "Stop after the disclaimer."
                    )
                else:
                    return (
                        "Write one sentence stating the player name and their total career stat.\n"
                        "Then write dash lines for any breakdown figures that exist in the FACT lines.\n"
                        f"After the last dash line, add exactly this disclaimer:\n{_DISCLAIMER}"
                        "Stop after the disclaimer."
                    )

        elif intent == "live_sports":
            # Detect what kind of stats answer is needed from the query
                _is_cricket_q = bool(_re_fmt.search(
                    r'\b(cricket|runs?|wickets?|batting|bowling|odi|test|t20|'
                    r'centuries|fifties|innings)\b', _q
                ))
                _is_club_q = bool(_re_fmt.search(
                    r'\b(for|at|with)\s+[a-z]', _q
                )) and not bool(_re_fmt.search(
                    r'\b(india|england|australia|pakistan|portugal|brazil|'
                    r'argentina|france|germany|spain|italy|international|'
                    r'south africa|new zealand|west indies|sri lanka)\b', _q
                ))
                _is_record_q = bool(_re_fmt.search(
                    r'\b(most|highest|all.?time|in history|ever|record|'
                    r'who scored most|leading scorer|top scorer)\b', _q
                ))
 
                if _is_cricket_q and not _is_club_q:
                    # Cricket international career — show format breakdown
                    return (
                        "Use ONLY the numbers from the FACT lines above.\n"
                        "Show the grand total first, then each format.\n\n"
                        "Format:\n"
                        "[Player] has scored [total] international runs/wickets.\n"
                        "- Tests: [Test number]\n"
                        "- ODIs: [ODI number]\n"
                        "- T20Is: [T20I number]\n\n"
                        "RULES:\n"
                        "- Use ONLY numbers from FACT lines. No other numbers.\n"
                        "- If a format is missing from FACT lines, omit that line.\n"
                        "- If only a partial total exists, state it as partial.\n"
                        "- Do NOT cite source names. Stop after the stats.\n"
                    )
                elif _is_club_q:
                    # Club/entity-specific stats
                    return (
                        "Use ONLY the numbers from the FACT lines above.\n"
                        "State the specific club/entity stat, not career total.\n\n"
                        "Format:\n"
                        "[Player] scored/took [number] [stat] for [club].\n"
                        "- [Detail from FACT lines if present]\n"
                        "- [Detail from FACT lines if present]\n\n"
                        "RULES:\n"
                        "- Use ONLY numbers explicitly linked to the named entity in FACT lines.\n"
                        "- Do NOT substitute the overall career total.\n"
                        "- Do NOT cite source names. Stop after the stats.\n"
                    )
                elif _is_record_q:
                    # All-time record query
                    return (
                        "Use ONLY the numbers from the FACT lines above.\n"
                        "State the official record holder and their total.\n\n"
                        "Format:\n"
                        "[Record holder] holds the all-time record with [number] [stat].\n"
                        "- [Second highest if in FACT lines]\n"
                        "- [Any relevant note from FACT lines]\n\n"
                        "RULES:\n"
                        "- Use ONLY numbers from FACT lines. No training memory.\n"
                        "- Official record = competitive matches only.\n"
                        "- If FACT lines show conflicting totals, state both with their labels.\n"
                        "- Do NOT cite source names. Stop after the stats.\n"
                    )
                else:
                    # General career stats (football career, any other sport)
                    return (
                        "Use ONLY the numbers from the FACT lines above.\n"
                        "State the total first, then any breakdown available.\n\n"
                        "Format:\n"
                        "[Player] has scored/taken [total] [stat] in their career.\n"
                        "- [Club/competition breakdown from FACT lines]\n"
                        "- [International breakdown from FACT lines]\n"
                        "- [Any other detail from FACT lines]\n\n"
                        "RULES:\n"
                        "- Use ONLY numbers from FACT lines. No training memory.\n"
                        "- Only include breakdown lines that exist in FACT lines.\n"
                        "- Do NOT cite source names. Stop after the stats.\n"
                    )
        elif intent == "upcoming_sports":
            return (
                "## Upcoming Matches\n"
                "List each upcoming match as a bullet. Format:\n"
                "- **[Team1] vs [Team2]** — [Date] | [Venue if available]\n"
                "Rules:\n"
                "- One bullet per match. No scores (match not played yet).\n"
                "- Never invent dates or venues.\n"
            )
        elif intent == "weather":
            return (
                "## Weather in [City] — [Date]\n"
                "- **Temperature:** [current temp °C / °F]\n"
                "- **Condition:** [weather condition]\n"
                "- **High / Low:** [high] / [low]\n"
                "- **Humidity:** [humidity if present]\n"
                "- **Wind:** [wind speed and direction if present]\n"
                "- **Forecast:** [short forecast if present]\n"
                "- **Last Updated:** [timestamp or date from context]\n"
                "CRITICAL: Only include fields that have data.\n"
            )
        elif intent == "news":
            return (
                "Write one bullet per article. Use ONLY this exact format:\n"
                "- **[headline]** *(source, date)* — [2-3 sentence summary]\n\n"
                "Rules:\n"
                "- Copy the headline, source name, and date exactly from the context.\n"
                "- Source and date appear in parentheses after each title in the context.\n"
                "- Summary: what happened, who, key number or date, why it matters.\n"
                "- No sub-bullets. No + symbols. No ## headings. Only - bullets.\n"
                "- Never omit *(source, date)*. Never skip the **headline**.\n"
                "- CRITICAL: Each bullet MUST end with a full stop before the next bullet.\n"
                "- CRITICAL: NEVER run one bullet's text into the next bullet's header.\n"
                "  Every bullet starts with '- **' on its own line and ends before the next '- **'.\n\n"        
                "Example:\n"
                "- **Mamata accuses Modi of weaponising SIR** *(aninews, 2026-03-22)* — "
                "West Bengal CM Mamata Banerjee accused PM Modi of using the Strategic "
                "Investment Region to spread communal tension in West Bengal.\n"
            )
        elif intent == "scientific":
            import re as _re_sci
            _q = (query or "").lower()

        # Detect scientific sub-type
            _is_origin = bool(_re_sci.search(
                r'\b(how did|where did|when did|how was|how were|'
                r'came into existence|come into existence|originate|'
                r'how did .{2,30} form|how did .{2,30} evolve|'
                r'how did .{2,30} begin|how did .{2,30} start|'
                r'how did .{2,30} come|creation of|origin of|'
                r'how was .{2,30} created|how was .{2,30} formed|'
                r'how was .{2,30} born)\b', _q
            ))
            _is_cause = bool(_re_sci.search(
                r'\b(what causes|why does|why do|why is|why are|'
                r'what makes|what causes|reason for|reason behind|'
                r'responsible for|what triggers|what leads to)\b', _q
            ))  
            _is_process = bool(_re_sci.search(
                r'\b(how does|how do|how is|how are|what happens|'
                r'what happens when|what happens during|'
                r'how does .{2,30} work|how does .{2,30} function|'
                r'how does .{2,30} travel|how does .{2,30} move|'
                r'how does .{2,30} spread|how does .{2,30} grow)\b', _q
            ))
            _is_biology = bool(_re_sci.search(
                r'\b(human|body|brain|cell|dna|gene|blood|heart|'
                r'immune|virus|bacteria|organ|muscle|bone|nerve|'
                r'sleep|dream|age|aging|cancer|disease|digest|'
                r'evolution|species|animal|plant|organism|life)\b', _q
            ))
            _is_astronomy = bool(_re_sci.search(
                r'\b(star|planet|moon|sun|galaxy|universe|space|'
                r'black hole|solar system|orbit|gravity|cosmos|'
                r'nebula|asteroid|comet|supernova|light year)\b', _q
            ))
            _is_physics = bool(_re_sci.search(
                r'\b(light|sound|wave|energy|force|motion|speed|'
                r'electricity|magnetic|radiation|nuclear|quantum|'
                r'particle|atom|electron|proton|neutron|heat|'
                r'temperature|pressure|entropy|relativity)\b', _q
            ))
            _is_earth = bool(_re_sci.search(
                r'\b(earthquake|volcano|ocean|tide|season|weather|'
                r'climate|atmosphere|tectonic|erosion|glacier|'
                r'water cycle|northern lights|aurora|fossil|'
                r'mountain|river|soil|rock|mineral|cave)\b', _q
            ))

            # Pick section headers based on sub-type
            if _is_origin and _is_biology:
                mechanism_header = "**Evolutionary / Historical Timeline:**"
                mechanism_hint   = "stages of development, time periods (millions of years ago), key ancestors or milestones"
                facts_hint       = "specific dates (e.g. 3.8 billion years ago), species names, fossil evidence, locations"
            elif _is_origin and _is_astronomy:
                mechanism_header = "**Formation Process (Step by Step):**"
                mechanism_hint   = "physical processes, temperatures, pressures, time spans in billions of years"
                facts_hint       = "distances in light-years, temperatures in Kelvin, ages in billion years, key events"
            elif _is_origin:
                mechanism_header = "**How It Came to Be (Step by Step):**"
                mechanism_hint   = "chronological stages, key conditions, causes, time periods"
                facts_hint       = "specific dates, measurements, named events, key figures or discoveries"
            elif _is_cause and _is_physics:
                mechanism_header = "**Physics Behind It:**"
                mechanism_hint   = "wavelengths (nm), frequencies (Hz), forces (N), speeds (m/s), equations if simple"
                facts_hint       = "exact wavelengths, angles, speeds, named laws or effects (e.g. Rayleigh scattering)"
            elif _is_cause and _is_biology:
                mechanism_header = "**Biological Mechanism:**"
                mechanism_hint   = "cellular/molecular processes, organs involved, chemical signals, timescales"
                facts_hint       = "specific molecules, hormone names, cell types, measurable biological values"
            elif _is_cause and _is_earth:
                mechanism_header = "**Geological / Atmospheric Process:**"
                mechanism_hint   = "tectonic forces, atmospheric layers, chemical reactions, time scales"
                facts_hint       = "depths (km), temperatures (°C), pressures (atm), speeds (km/h), named plates or layers"
            elif _is_process and _is_biology:
                mechanism_header = "**Biological Process (Step by Step):**"
                mechanism_hint   = "organs/cells/molecules involved, sequence of events, chemical signals, durations"
                facts_hint       = "specific molecule names, organ sizes, speeds, temperatures, durations"
            elif _is_process and _is_physics:
                mechanism_header = "**Physical Process (Step by Step):**"
                mechanism_hint   = "forces, energy transfers, speeds, units (Joules, Watts, m/s, Hz)"
                facts_hint       = "exact values with units, named laws (Newton, Faraday), measurable quantities"
            elif _is_astronomy:
                mechanism_header = "**Astrophysical Process:**"
                mechanism_hint   = "forces (gravity, fusion), temperatures, distances, time spans in billions of years"
                facts_hint       = "distances in light-years or AU, masses in solar masses, temperatures in Kelvin"
            elif _is_earth:
                mechanism_header = "**Earth Science Process:**"
                mechanism_hint   = "tectonic activity, atmospheric/chemical reactions, time scales, locations"
                facts_hint       = "depths (km), speeds (cm/year), temperatures (°C), named plates or layers"
            else:
            # Generic scientific fallback
                mechanism_header = "**How It Works / Mechanism:**"
                mechanism_hint   = "step-by-step process, measurements, units, named concepts or laws"
                facts_hint       = "specific measurable facts with units — dates, distances, temperatures, durations"

            return (
                "Answer with precise scientific facts. Go deep — do not give a brief summary.\n"
                f"Format:\n"
                f"## [Scientific Topic]\n"
                f"**Summary:** [2-3 sentence direct answer — include at least one specific number, date, or measurement]\n\n"
                f"{mechanism_header}\n"
                f"- **[Step or Concept]** — [{mechanism_hint}]\n"
                f"- **[Step or Concept]** — [precise explanation with specific detail]\n"
                f"- **[Step or Concept]** — [precise explanation with specific detail]\n"
                f"- **[Step or Concept]** — [precise explanation with specific detail]\n"
                f"- **[Step or Concept]** — [precise explanation with specific detail]\n\n"
                f"**Key Facts:**\n"
                f"- [{facts_hint}]\n"
                f"- [specific measurable fact with units]\n"
                f"- [specific measurable fact with units]\n\n"
                "RULES:\n"
                f"- Minimum 5 bullet points in the middle section. Never stop early.\n"
                f"- Every bullet must include at least one specific detail (number, unit, date, name).\n"
                f"- Do NOT write 'see above' or repeat the summary in bullets.\n"
                f"- Do NOT use vague phrases like 'over time' or 'many years ago' — use exact figures.\n"
                f"- ONLY use numbers and statistics that are explicitly present in the context.\n"
                f"- NEVER invent statistics, market sizes, percentages, or dollar figures.\n" 
                f"- If no specific numbers are in the context, describe the concept without numbers.\n"
            )
        elif intent == "document":
            return (
                "Answer the question using ONLY the text in the context above.\n"
                "Do NOT add, infer, or invent any information not explicitly present.\n\n"
                "Write in natural prose or bullet points — do NOT output empty section headings.\n"
                "Only include a section if the context actually contains that information.\n"
                "If a field is absent from context, simply omit it — do not write 'Not mentioned'.\n\n"
                "RULES:\n"
                "- Never use bracket placeholders like [Section Name] or [content] in output.\n"
                "- Never output a heading followed immediately by nothing or 'Not mentioned'.\n"
                "- Copy facts exactly — do not paraphrase or embellish.\n"
                "- Stop immediately after presenting what the context contains.\n"
            )
        

        elif intent == "summary" or fmt == "summary":
            return (
                "Provide a clear and concise summary.\n"
                "Format:\n"
                "## Summary: [Topic]\n"
                "- **[Key Point]** — [1-2 sentence explanation]\n"
                "- **[Key Point]** — [1-2 sentence explanation]\n"
                "- **[Key Point]** — [1-2 sentence explanation]\n\n"
                "RULES:\n"
                "- 3-5 bullet points maximum.\n"
                "- Each bullet must be a distinct point — no repetition.\n"
                "- Do not pad with vague or filler content.\n"
                "- Stop after the summary. No disclaimers.\n"
            )

        else:
            return ""

    def _clean_response(self, response: str) -> str:
        cutoff_patterns = [
            "\n(I don't", "\nCorrecting",
            "\ni.e.,", "\nSo the answer", "(Stop)", "[raw_data",
            "\nNote: I", "\n(Note: I",
        ]
        for pattern in cutoff_patterns:
            if pattern in response:
                response = response[:response.index(pattern)].strip()
        lines = response.split("\n")
        normalized = []
        for line in lines:
            if re.match(r'^\s*\*(?!\*)\s+', line):
                line = re.sub(r'^(\s*)\*(\s+)', r'\1-\2', line)
            normalized.append(line)
        response = "\n".join(normalized)

        import re as _re_cr
        lines = response.split("\n")
        seen_lines = set()
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                clean_lines.append(line)
                continue
            content_after_dash = _re_cr.sub(r'^[\-\*\s]*\*\*[^\*]+\*\*[\s\-—]*', '', stripped)
            if content_after_dash and content_after_dash != stripped:
                fp = content_after_dash.lower().strip()[:80]
            else:
                fp = _re_cr.sub(r'[\*\-\#\>\`]+', '', stripped).lower().strip()[:60]
            fp = _re_cr.sub(r'\s+', ' ', fp).strip()
            if fp and fp not in seen_lines:
                seen_lines.add(fp)
                clean_lines.append(line)

        response = "\n".join(clean_lines)

        sentences = response.split(". ")
        seen_sentences = set()
        clean_sentences = []
        for s in sentences:
            key = s.strip().lower()[:60]
            if key and key not in seen_sentences:
                seen_sentences.add(key)
                clean_sentences.append(s)

        return ". ".join(clean_sentences).strip()

    def get_kb_confidence(self, query: str) -> float:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["distances"]
        )
        if not results["distances"] or not results["distances"][0]:
            return 0.0
        distance = results["distances"][0][0]
        return max(0.0, 1.0 - distance)
    

    def _should_use_kb(self, confidence: float, docs: list, query: str) -> bool:

        if not docs or confidence < 0.20:
            print(f"[DEBUG] KB gate: FAIL stage 1 (confidence={confidence:.3f})")
            return False

        q_vec = self.embedding_model.encode(query)
        scored_docs = []
        for doc in docs[:5]:
            d_vec = self.embedding_model.encode(doc[:512])
            sim = float(
                np.dot(q_vec, d_vec) /
                (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            )
            scored_docs.append((sim, doc))

        scored_docs.sort(key=lambda x: -x[0])
        best_sim = scored_docs[0][0]
        print(f"[DEBUG] KB gate best_sim={best_sim:.3f}")

        if best_sim < 0.15:
            print(f"[DEBUG] KB gate: FAIL stage 2 (sim too low)")
            return False

        if not self.reranker:
            return best_sim >= 0.40

        top_docs = [doc for _, doc in scored_docs[:5]]

        try:
            scored_results = self.reranker.rerank_with_scores(query, top_docs, top_k=1)
            if not scored_results:
                return False

            top_score, top_doc = scored_results[0]
            print(f"[DEBUG] KB gate reranker score={top_score:.3f} | {top_doc[:60]}")

            # ── RAISED THRESHOLD ─────────────────────────────────────────
            # Old: -2.5 (too permissive — passes resume for "how does RAG work")
            # New: 2.0 (cross-encoder must be confident the doc answers the query)
            _KB_SCORE_THRESHOLD = 2.0
        
            if top_score < _KB_SCORE_THRESHOLD:
                print(f"[DEBUG] KB gate: FAIL reranker (score={top_score:.3f} < {_KB_SCORE_THRESHOLD})")
                return False

            # ── SECOND CHECK: ask LLM if the doc actually answers the question ──
            # This catches cases where reranker scores high due to topic overlap
            # but the document doesn't contain the actual answer
            try:
                _doc_preview = top_doc[:800]
                _check = ollama.chat(
                    model=self.ollama_model,
                    messages=[{
                        "role": "user",
                        "content": (
                            f'Does the document below directly answer this question?\n\n'
                            f'Question: "{query}"\n\n'
                            f'Document:\n{_doc_preview}\n\n'
                            f'Reply with ONLY "yes" or "no".'
                        )
                    }],
                    options={"temperature": 0.0, "num_predict": 3}
                )
                _answer = _check["message"]["content"].strip().lower()
                _passes = _answer.startswith("yes")
                print(f"[DEBUG] KB gate LLM check: '{_answer}' → {'PASS' if _passes else 'FAIL'}")
                return _passes
            except Exception as e:
                print(f"[DEBUG] KB gate LLM check failed: {e} — using reranker score only")
                return top_score >= _KB_SCORE_THRESHOLD

        except AttributeError:
            try:
                reranked = self.reranker.rerank(query, top_docs, top_k=1)
                return bool(reranked)
            except Exception as e:
                print(f"[DEBUG] KB gate reranker failed: {e}")
                return best_sim >= 0.35

        except Exception as e:
            print(f"[DEBUG] KB gate reranker failed: {e}")
            return best_sim >= 0.35

    # =========================================================================
    # LLM CALL HELPER
    # =========================================================================
    def _llm_call(self, user_input: str, context: str, intent: str,
              length_instruction: str = "", num_points: int = None,
              temperature: float = None, context_label: str = "Context",
              fmt: str = None, news_article_count: int = 0) -> str:
        intent_instructions = {
            "document": (          # ← PASTE THIS ENTRY
                "Extract and present ONLY what is explicitly written in the context. "
                "Do NOT add anything from training memory under any circumstance. "
                "Do NOT invent names, dates, companies, skills, or achievements. "
                "Do NOT output empty headings or 'Not mentioned' for absent fields — simply omit them. "
                "If the entire context is irrelevant to the question, say: "
                "'This information is not in the knowledge base.'"
            ),
            "ownership":       "List ALL owners, founders or executives mentioned in context. Do not omit any names.",
            "live_sports":     "Answer only about scores, results or match outcomes.",
            "upcoming_sports": "List only upcoming scheduled matches. Never include scores.",
            "facts":           "Answer only the specific fact asked. Be precise with numbers and dates.",
            "sports_knowledge": (
                "Answer with precise sports facts, statistics, rules, and verified records. "
                "For RULE questions: give a thorough step-by-step explanation with conditions, "
                "examples, and exceptions — not a one-line definition. "
                "For STATS questions: state the exact number first, then break down by format. "
                "For WINNER questions: name the winner, year, and key details. "
                "Prioritise [SPORTS KB] blocks over web snippets. "
                "Minimum 4 detailed bullet points for rule and knowledge questions."
            ),
            "knowledge": (
                    "Give a comprehensive, detailed explanation using all relevant facts from the context. "
                    "Cover what it is, how it works, why it matters, key components, and real-world applications. "
                    "Do NOT stop after a brief summary — elaborate on each point with specifics. "
                    "Use all available context to give the most complete answer possible."
                ),
            "news": (
                "Summarise each news article as a separate bullet. "
                "Each bullet must be 2-3 sentences: what happened, who is involved, "
                "key numbers or figures, and why it matters. "
                "Do NOT collapse multiple articles into one bullet. "
                "Include the source name and date for each bullet. "
                "Never repeat the same fact across bullets."
            ),
            "scientific": (
                "Answer with precise scientific facts, measurements, and processes. "
                "Use correct scientific terminology with exact numbers, dates, and units. "
                "Provide a thorough explanation — cover the mechanism step by step, "
                "include cause and effect, key discoveries, and real-world significance. "
                "Aim for at least 6-8 detailed bullet points. "
                "Do not stop after a brief summary — elaborate fully on each concept."
            ),

            "general":         "Answer directly and confidently. Stick to what the context says.",
        }
        intent_instruction = intent_instructions.get(intent, "Answer directly and concisely.")

        intent_temperature = {
            "document":        0.0,
            "ownership":       0.1,
            "live_sports":     0.1,
            "upcoming_sports": 0.1,
            "facts":           0.5,
            "sports_knowledge":0.3,
            "knowledge":       0.4,
            "scientific":      0.35,
            "news":            0.4,
            "general":         0.2,
        }
        if temperature is None:
            temperature = intent_temperature.get(intent, TEMPERATURE)

        if num_points:
            num_predict = min(50 + num_points * 100, 1200)
        elif intent == "document":   # ← PASTE THIS
            num_predict = 800
        elif intent == "news":
            _news_per_block = max(300, news_article_count * 300)
            num_predict = max(1500, _news_per_block)
        elif intent == "scientific":
            num_predict = 1000
        elif intent == "sports_knowledge":
            num_predict = 2000
        elif intent == "knowledge":
            num_predict = 1500
        elif intent == "facts":
            num_predict = 600
        else:
            num_predict = 400

        format_instruction = self._get_format_instruction(intent, user_input, num_points=num_points, fmt=fmt)
        if intent == "document":
            messages = [{
                "role": "system",
                "content": (
                    "You are a document reader. Your ONLY job is to extract and present "
                    "information explicitly written in the context the user provides. "
                    "You have NO other knowledge. Never invent, infer, or add anything. "
                    "If something is not in the context, do not mention it at all."
            )
            }, {"role": "user", "content": ""}]
        elif intent in ("facts", "ownership", "knowledge", "sports_knowledge", "news", "live_sports", "upcoming_sports", "scientific"):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]
        else:
            messages = self._build_message_list(user_input)
            messages = self._trim_messages_to_token_limit(messages)

        is_search_context = "internet search" in context_label.lower() or context_label == "Context"

        if is_search_context:
            rules = (
                f"0. [SPORTS KB] blocks are pre-verified facts — prioritise them over [INTERNET SEARCH] blocks.\n"
                f"   Prefer the MOST RECENT data available. The current year is {time.strftime('%Y')}.\n"
                f"   If context has both old and new figures for the same stat, always use the newest one.\n"
                f"1. Extract and state FACTS directly from the context. Use EXACT numbers — never round or estimate.\n"
                f"2. NEVER repeat the same fact twice.\n"
                f"3. If different snippets give DIFFERENT answers to the same question,\n"
                f"   report ALL answers found and note which source says what.\n"
                f"4. If the context contains partial or conflicting information,\n"
                f"   clearly say 'sources disagree' or 'data is incomplete'.\n"
                f"5. NEVER invent specific numbers, names, or dates not in the context.\n"
                f"   NEVER write 'more than X' or 'around X' if an exact figure exists in context.\n"
                f"5b. If the context starts with === VERIFIED FACTS === and contains a 'computed' line,\n"
                f"   use that computed total as the answer — it has already been verified.\n"
                f"   Only avoid adding numbers yourself if NO computed total exists in the verified facts.\n"
                f"5c. NEVER use numbers from your training knowledge. Only use numbers from the\n"
                f"   context above. If context figures differ from what you remember, trust context.\n"
                f"6. If a sub-category is mentioned in the question, only use facts that explicitly match it.\n"
                f"7. If a specific number is not present in the context, say 'not available in context'\n"
                f"   rather than estimating or approximating.\n"
                f"8. Only say you cannot answer if the context has ZERO relevant facts.\n"
                f"9. Stop after answering. No disclaimers."
            )
        else:
            rules = (
                f"1. Read the context above carefully before answering.\n"
                f"2. Answer ONLY using facts explicitly written in the context.\n"
                f"3. NEVER use your training memory for any fact, name, date, or number.\n"
                f"4. NEVER combine the subject from the question with unrelated context.\n"
                f"   If context is about X but question asks about Y, say not available.\n"
                f"5. NEVER invent or infer information not present in the context.\n"
                f"6. If the context does not answer the question, say exactly:\n"
                f"   'This information is not available in the knowledge base.'\n"
                f"7. Stop immediately after answering. No extra commentary.\n"
            )
        
        if intent == "news" and news_article_count > 0:
            rules += (
                f"\n\nCRITICAL — FABRICATION PREVENTION:\n"
                f"There are EXACTLY {news_article_count} article block(s) in the context above.\n"
                f"Write EXACTLY {news_article_count} bullet(s) — one per article block. No more.\n"
                f"Every bullet MUST be grounded in an article block present above.\n"
                f"DO NOT invent, paraphrase from memory, or add any bullet not supported "
                f"by the text in context. If you cannot find {news_article_count} distinct facts, "
                f"write fewer bullets rather than fabricating content.\n"
            )
        if intent == "document":
            messages[-1]["content"] = (
                f"Context (this is the ONLY source you may use):\n{context}\n\n"
                f"Question: {user_input}\n\n"
                f"Instructions:\n"
                f"- Answer using ONLY the text above.\n"
                f"- Do not output empty headings.\n"
                f"- Do not say 'not mentioned' or 'not available'.\n"
                f"- Do not use your training knowledge for any fact.\n"
                f"- Only write about what is actually in the context above.\n"
                f"- Stop after answering."
            )       

        # For sports_knowledge: question first, then verified facts context,
        # then short imperative rules — helps small models not drift to training memory
        elif intent == "sports_knowledge":
            _is_entity_q = bool(re.search(
                r'\b(for|at|with|in)\s+[A-Z]', user_input
            ))
            if _is_entity_q:
                rules += (
                    "\n10. ENTITY-SPECIFIC RULE: The question contains 'for/at/with [name]'.\n"
                    "    Answer with ONLY the stat for that specific entity.\n"
                    "    Do NOT substitute the overall career total as the answer.\n"
                    "\n11. HALLUCINATION PREVENTION: Do NOT invent competition names, "
                        "season years, or goal breakdowns.\n"
                    "    Only state figures that appear word-for-word in the FACT lines or context.\n"
                    "    If a breakdown is not in context, do not include it — state the total only.\n"
            )
            # Detect if this is a generic football record query
            _is_football_record_q = bool(re.search(
                r'\b(most|highest|record|all.?time|ever|in history)\b',
                user_input, re.IGNORECASE
            )) and bool(re.search(
                r'\b(football|soccer|goals?|free kick|header|penalties|'
                r'champions league|premier league|la liga)\b',
                user_input, re.IGNORECASE
            ))

            _football_disambiguation = ""
            if _is_football_record_q:
                _football_disambiguation = (
                    f"\n\nCRITICAL: FOR THIS FOOTBALL RECORD QUESTION:\n"
                    f"The === VERIFIED FACTS === block above contains the answer from current sources.\n"
                    f"Use ONLY those FACT lines for your answer.\n"
                    f"Do NOT use your training memory for any name or number.\n"
                    f"If the FACT lines show NOT_FOUND, say the information is not available.\n\n"
                )
            messages[-1]["content"] = (
                f"ANSWER THIS QUESTION USING ONLY THE NUMBERS IN THE CONTEXT BELOW.\n"
                f"DO NOT USE YOUR TRAINING MEMORY FOR ANY NUMBER OR FACT.\n\n"
                f"DO NOT output any bracket placeholders like [Player], [figure], [category] — "
                f"these are instructions for you, not text to print.\n\n"
                f"DO NOT print source labels like [SPORTS KB] or [INTERNET SEARCH] in your answer.\n\n"
                f"DO NOT print === headers === in your answer.\n\n"
                f"DO NOT add headings with brackets like ## [Player / Team / Topic].\n\n"
                f"CRITICAL: If the context starts with === VERIFIED FACTS ===, "
                f"use ONLY those FACT: lines for ALL numbers in your answer — "
                f"including format breakdowns. Do NOT use numbers from the body context.\n\n"
                f"{_football_disambiguation}"
                f"Question: {user_input}\n\n"
                f"{context_label}:\n{context}\n\n"
                f"RULES:\n{rules}\n\n"
                f"FORMAT YOUR RESPONSE LIKE THIS:\n{format_instruction}"
            )
            
        
        else:
            messages[-1]["content"] = (
                f"{context_label}:\n{context}\n\n"
                f"Question: {user_input}\n\n"
                f"RULES:\n{rules}\n\n"
                f"Intent: {intent_instruction}\n\n"
                f"FORMAT YOUR RESPONSE LIKE THIS:\n{format_instruction}"
            )

        if intent != "document":
            if length_instruction:
                messages[0]["content"] = SYSTEM_PROMPT + f"\n\nRESPONSE FORMAT: {length_instruction}"
            else:
                messages[0]["content"] = SYSTEM_PROMPT

        response = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
                "repeat_penalty": 1.1 if intent in ("knowledge",) else (1.05 if intent == "sports_knowledge" else 1.1),
                "repeat_last_n": 64,
                "num_keep": 0,
            }
        )

        result = ""
        for chunk in response:
            result += chunk["message"]["content"]

        return self._clean_response(result)
    
    def _extract_stats_from_context(self, query: str, context: str,
                                 props: dict = None) -> str:
        """
        Extracts FACT lines from context using a targeted LLM call.
        Works universally for any sport, any player, any entity.

        Uses props["entity_type"] (from _classify_sports_query) instead of
        hardcoded country/team name lists — fully dynamic, no regex lists.
        """
        try:
            props = props or {}
            _ctx = context[:15000]
            _current_year = time.strftime('%Y')
            _prev_year = str(int(_current_year) - 1)

            sport          = str(props.get("sport",          "other")).lower()
            stat_type      = str(props.get("stat_type",       "stat" )).lower()
            cricket_format = str(props.get("cricket_format",  "all"  )).lower()
            is_entity_specific = bool(props.get("is_entity_specific", False))
            is_generic_record  = bool(props.get("is_generic_record",  False))
            entity_type        = str(props.get("entity_type", "none")).lower()

            _query_lower = query.lower()

            # ── Helper: does query contain "for [specific entity]"? ───────────
            # Only returns True for actual club/team names.
            # Returns False for "in football history", "in cricket" etc.
            _SPORT_WORDS_RE = re.compile(
                r'\b(football|soccer|cricket|tennis|basketball|rugby|hockey|'
                r'baseball|golf|formula|f1|swimming|athletics|volleyball|'
                r'kabaddi|boxing|wrestling|cycling|badminton|history|'
                r'the world|all time|career|world)\b',
                re.IGNORECASE
            )
            def _has_for_entity(q: str) -> bool:
                for m in re.finditer(
                    r'\bfor\s+([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\s*$|\s*\?)',
                    q, re.IGNORECASE
                ):
                    candidate = m.group(1).strip()
                    if not _SPORT_WORDS_RE.search(candidate):
                        return True
                return False

            # ── Override 1: detect generic record when LLM missed it ──────────
            if not is_generic_record and not is_entity_specific and not _has_for_entity(query):
                if bool(re.search(
                    r'\b(most|highest|all.?time|in history|ever|'
                    r'who scored most|who has most|leading scorer|top scorer|'
                    r'most goals|most runs|most wickets|most points|most tries|'
                    r'record holder|world record)\b',
                    _query_lower
                )):
                    is_generic_record = True
                    print(f"[DEBUG] _extract_stats: is_generic_record override → True")

            # ── Override 2: national team → reset entity-specific ─────────────
            # Uses entity_type from _classify_sports_query (dynamic, no hardcoded list).
            # If entity_type is "national", the query is about international career stats,
            # not a specific club filter — treat as full career/all-formats query.
            if is_entity_specific and entity_type == "national":
                is_entity_specific = False
                if sport == "cricket":
                    cricket_format = "all"
                print(f"[DEBUG] _extract_stats: entity_type=national → is_entity_specific=False")

            # ── Build extraction instructions ─────────────────────────────────

            if sport == "cricket" and is_entity_specific:
                # Club/franchise (RCB, CSK, MI, SRH, BBL team, etc.)
                _em = re.search(
                    r'\bfor\s+([a-zA-Z][a-zA-Z\s]{1,30}?)(?:\s*$|\s*\?)',
                    query.strip(), re.IGNORECASE
                )
                _entity = _em.group(1).strip() if _em else "the mentioned club"
                instructions = (
                    f"The question asks about cricket {stat_type} FOR '{_entity}' specifically.\n"
                    f"Search the context for '{_entity}' near a {stat_type} figure.\n\n"
                    f"FACT: Club/franchise name — [{_entity}]\n"
                    f"FACT: {stat_type.title()} for {_entity} only — "
                    f"[number explicitly attributed to {_entity} in context]\n"
                    f"FACT: Matches for {_entity} — [number if stated]\n"
                    f"FACT: Seasons at {_entity} — [number if stated]\n\n"
                    f"CRITICAL RULES:\n"
                    f"- The number MUST be explicitly linked to '{_entity}' in context.\n"
                    f"- Do NOT extract overall international career totals.\n"
                    f"- Do NOT extract stats from any other team.\n"
                    f"- If context only has career total without {_entity} breakdown: NOT_FOUND.\n"
                )

            elif sport == "cricket" and cricket_format == "test":
                instructions = (
                    f"Extract ONLY Test cricket {stat_type}. Ignore ODI and T20I figures.\n"
                    f"Look for numbers explicitly labelled 'Test' or 'Tests' in context.\n\n"
                    f"FACT: Test {stat_type} — [exact number]\n"
                    f"FACT: Test matches played — [number if stated]\n"
                    f"FACT: Test innings — [number if stated]\n"
                    f"FACT: Test average — [number if stated]\n"
                    f"FACT: Player name — [name]\n"
                )

            elif sport == "cricket" and cricket_format == "odi":
                instructions = (
                    f"Extract ONLY ODI cricket {stat_type}. Ignore Test and T20I figures.\n"
                    f"Look for numbers explicitly labelled 'ODI' or 'One Day' in context.\n\n"
                    f"FACT: ODI {stat_type} — [exact number]\n"
                    f"FACT: ODIs played — [number if stated]\n"
                    f"FACT: ODI innings — [number if stated]\n"
                    f"FACT: ODI average — [number if stated]\n"
                    f"FACT: Player name — [name]\n"
                )

            elif sport == "cricket" and cricket_format == "t20":
                instructions = (
                    f"Extract ONLY T20I cricket {stat_type}. Ignore Test and ODI figures.\n"
                    f"Look for numbers explicitly labelled 'T20I' or 'T20 International'.\n\n"
                    f"FACT: T20I {stat_type} — [exact number]\n"
                    f"FACT: T20Is played — [number if stated]\n"
                    f"FACT: T20I innings — [number if stated]\n"
                    f"FACT: T20I average — [number if stated]\n"
                    f"FACT: Player name — [name]\n"
                )

            elif sport == "cricket":
                # All international formats — extract each separately
                instructions = (
                    f"Extract cricket {stat_type} for ALL THREE international formats.\n"
                    f"Each format MUST be on its own separate line.\n\n"
                    f"FACT: Test {stat_type} — [number explicitly labelled 'Test' in context]\n"
                    f"FACT: ODI {stat_type} — [number explicitly labelled 'ODI' in context]\n"
                    f"FACT: T20I {stat_type} — [number explicitly labelled 'T20I' in context]\n"
                    f"FACT: Total international {stat_type} — "
                    f"[overall career total only if explicitly stated]\n"
                    f"FACT: Player name — [name]\n\n"
                    f"STRICT RULES:\n"
                    f"- Copy numbers EXACTLY — never round or estimate.\n"
                    f"- Each format number must be EXPLICITLY labelled in context.\n"
                    f"- If a format is not in context: FACT: [format] {stat_type} — NOT_FOUND\n"
                    f"- Do NOT combine formats into one line.\n"
                    f"- Do NOT compute totals yourself.\n"
                    f"- Prefer {_current_year} or {_prev_year} data.\n"
                )

            elif sport == "football" and is_entity_specific:
                _em = re.search(
                    r'\bfor\s+([a-zA-Z][a-zA-Z\s]{1,30}?)(?:\s*$|\s*\?)',
                    query.strip(), re.IGNORECASE
                )
                _entity = _em.group(1).strip() if _em else "the mentioned club"
                instructions = (
                    f"The question asks about {stat_type} scored FOR '{_entity}' ONLY.\n"
                    f"Search context for '{_entity}' near a {stat_type} figure.\n\n"
                    f"FACT: Club name — [{_entity}]\n"
                    f"FACT: {stat_type.title()} for {_entity} only — "
                    f"[number explicitly linked to {_entity} in context]\n"
                    f"FACT: Appearances for {_entity} — [number if stated]\n"
                    f"FACT: Seasons at {_entity} — [number if stated]\n\n"
                    f"CRITICAL RULES:\n"
                    f"- The number MUST be explicitly attributed to '{_entity}' in context.\n"
                    f"- Do NOT use the overall career total.\n"
                    f"- Do NOT use {stat_type} from any other club or country.\n"
                    f"- If context only has career total without '{_entity}' breakdown:\n"
                    f"  output: FACT: {stat_type.title()} for {_entity} — NOT_FOUND\n"
                    f"- Copy numbers EXACTLY. No ranges, no estimates.\n"
                )

            elif sport == "football" and is_generic_record:
                instructions = (
                    f"Find the ALL-TIME record holder for most {stat_type} in football.\n"
                    f"CAREER TOTAL = {stat_type} across ALL official club + international "
                    f"matches combined over entire career.\n\n"
                    f"How to identify career total vs single-competition figure:\n"
                    f"- Career total: labelled 'career', 'all-time', 'total', "
                    f"'across all competitions', or a large cumulative figure.\n"
                    f"- Single-competition: labelled 'in Champions League', "
                    f"'for Portugal', 'in La Liga' — these are NOT career totals.\n"
                    f"- Unofficial: labelled 'including friendlies', 'unofficial', "
                    f"'disputed' — report separately.\n\n"
                    f"FACT: Official record holder — [name as in context]\n"
                    f"FACT: Official career total {stat_type} — "
                    f"[number labelled career/all-time/official in context]\n"
                    f"FACT: Second highest — [name and number if stated]\n"
                    f"FACT: Unofficial note — "
                    f"[only if context explicitly mentions unofficial counts]\n\n"
                    f"RULES:\n"
                    f"- Copy numbers EXACTLY. No ranges, no guessing.\n"
                    f"- Do NOT assume a number is wrong because it seems small or large.\n"
                    f"- If conflicting totals exist, report all with their labels.\n"
                    f"- Prefer {_current_year} or {_prev_year} data.\n"
                )

            elif sport == "football":
                # Detect if this is an international/national team query
                _is_intl_query = bool(re.search(
                    r'\b(for\s+(?:portugal|brazil|argentina|france|germany|spain|'
                    r'england|italy|netherlands|belgium|croatia|uruguay|mexico|'
                    r'colombia|chile|japan|south korea|iran|nigeria|ghana|senegal|'
                    r'egypt|morocco|usa|australia|sweden|denmark|norway|'
                    r'[a-z]+ national team|his country|international|national team))\b',
                    _query_lower
                ))

                if _is_intl_query:
                    # Extract international goals specifically
                    _country_m = re.search(
                        r'\bfor\s+([a-zA-Z][a-zA-Z\s]{2,25}?)(?:\s*$|\s*\?)',
                        query.strip(), re.IGNORECASE
                    )
                    _country = _country_m.group(1).strip() if _country_m else "their national team"
                    instructions = (
                        f"The question asks about {stat_type} scored FOR '{_country}' "
                        f"(international / national team) ONLY.\n"
                        f"Search context for '{_country}' near a {stat_type} figure.\n\n"
                        f"FACT: Player name — [name]\n"
                        f"FACT: International {stat_type} for {_country} — "
                        f"[number explicitly linked to {_country} in context]\n"
                        f"FACT: International caps — [number of appearances if stated]\n\n"
                        f"CRITICAL RULES:\n"
                        f"- Extract ONLY {stat_type} for '{_country}' national team matches.\n"
                        f"- Do NOT use overall career total (which includes club goals).\n"
                        f"- Do NOT use {stat_type} from club matches.\n"
                        f"- Copy numbers EXACTLY. No ranges or estimates.\n"
                        f"- Prefer {_current_year} or {_prev_year} data.\n"
                    )
                else:
                    instructions = (
                        f"Extract football career {stat_type} for the player in the question.\n"
                        f"Prefer {_current_year}/{_prev_year} data.\n\n"
                        f"FACT: Player name — [name]\n"
                        f"FACT: Total career {stat_type} all competitions — "
                        f"[most recent cumulative figure from context]\n"
                        f"FACT: Club career {stat_type} — [number if stated]\n"
                        f"FACT: International {stat_type} — [number if stated]\n"
                        f"FACT: Champions League {stat_type} — [number if stated]\n\n"
                        f"RULES:\n"
                        f"- Copy numbers EXACTLY as they appear in context.\n"
                        f"- Do NOT use training memory for any number.\n"
                        f"- Prefer {_current_year} or {_prev_year} data.\n"
                    )

            # ── LLM extraction call ──────────────────────────────────────────
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract facts from the context to answer this question.\n"
                        f"Prefer data from {_current_year} or {_prev_year}.\n\n"
                        f"Question: {query}\n\n"
                        f"Context:\n{_ctx}\n\n"
                        f"Instructions:\n{instructions}\n"
                        f"Global rules:\n"
                        f"- Every output line MUST start with 'FACT:'\n"
                        f"- Copy numbers EXACTLY as they appear — never round or estimate\n"
                        f"- Do NOT use training memory for any number\n"
                        f"- Missing value: FACT: NOT_FOUND — [category]\n"
                        f"Extract now:"
                    )
                }],
                options={"temperature": 0.0, "num_predict": 400}
            )

            extracted = response["message"]["content"].strip()
            if not extracted:
                return ""

            # Keep valid FACT lines, drop NOT_FOUND
            fact_lines = [
                line.strip() for line in extracted.splitlines()
                if line.strip().startswith("FACT:") and "NOT_FOUND" not in line
            ]

            # ── Fallback: LLM ignored FACT: prefix ───────────────────────────
            if not fact_lines and extracted:
                number_pattern = re.findall(
                    r'([A-Z][a-zA-Z\s]{2,25})[:\-]\s*(\d[\d,]+)\s*'
                    r'(goals?|runs?|wickets?|points?|wins?|titles?|'
                    r'slams?|centuries?|tries?|aces?|assists?|strikes?)?\b',
                    extracted
                )
                if number_pattern:
                    for name, num, stat in number_pattern[:4]:
                        fact_lines.append(
                            f"FACT: {name.strip()} — {num} {stat or 'total'}"
                        )
                else:
                    nums  = re.findall(r'\b(\d[\d,]+)\b', extracted)
                    names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', extracted)
                    if nums and names:
                        fact_lines = [f"FACT: {names[0]} — {nums[0]} (fallback)"]

            if not fact_lines:
                print(f"[DEBUG] _extract_stats_from_context: no facts extracted")
                return ""

            # ── Cricket: auto-compute total from format breakdown ─────────────
            if sport == "cricket" and cricket_format == "all" and not is_entity_specific:
                has_combined = any(
                    any(w in l.lower() for w in
                        ["total", "overall", "combined", "all format",
                         "international", "career total", "aggregate"])
                    for l in fact_lines
                )
                if not has_combined:
                    format_nums = {}
                    for line in fact_lines:
                        cleaned_line = line.replace(',', '')
                        big_nums = [
                            int(n) for n in re.findall(r'\b(\d+)\b', cleaned_line)
                            if int(n) > 100
                        ]
                        if not big_nums:
                            continue
                        n  = big_nums[0]
                        ll = line.lower()
                        if 'test' in ll and 'test' not in format_nums:
                            format_nums['test'] = n
                        elif ('odi' in ll or 'one day' in ll) and 'odi' not in format_nums:
                            format_nums['odi'] = n
                        elif (
                            ('t20i' in ll or 't20 international' in ll)
                            and 't20' not in format_nums
                        ):
                            format_nums['t20'] = n
                        elif (
                            't20' in ll
                            and 't20' not in format_nums
                            and 'ipl' not in ll
                        ):
                            format_nums['t20'] = n

                    if len(format_nums) == 3:
                        computed = sum(format_nums.values())
                        fact_lines.insert(0,
                            f"FACT: Total international {stat_type} — {computed:,} "
                            f"(Tests {format_nums['test']:,} + "
                            f"ODIs {format_nums['odi']:,} + "
                            f"T20Is {format_nums['t20']:,}) — computed"
                        )
                        print(f"[DEBUG] Cricket total computed: {computed:,}")

                    elif len(format_nums) == 2:
                        partial = sum(format_nums.values())
                        missing = next(
                            f for f in ['test', 'odi', 't20']
                            if f not in format_nums
                        )
                        parts = " + ".join(
                            f"{k.upper()}s {v:,}" for k, v in format_nums.items()
                        )
                        fact_lines.insert(0,
                            f"FACT: Partial international {stat_type} — {partial:,} "
                            f"({parts}). {missing.upper()} data not found in sources."
                        )
                        print(f"[DEBUG] Cricket partial: {partial:,} (missing {missing})")

                    elif len(format_nums) == 1:
                        k, v = list(format_nums.items())[0]
                        fact_lines.insert(0,
                            f"FACT: Only {k.upper()} {stat_type} found — {v:,}. "
                            f"Other formats not in current sources."
                        )

            # ── Football: clarify Pele/Bican if Ronaldo/Messi absent ──────────
            if sport == "football" and is_generic_record:
                fact_text = " ".join(fact_lines).lower()
                has_pele_bican = bool(re.search(
                    r'\b(pele|pelé|bican|friedenreich)\b', fact_text
                ))
                has_ronaldo_messi = bool(re.search(
                    r'\b(ronaldo|cristiano|messi|lionel)\b', fact_text
                ))
                if has_pele_bican and not has_ronaldo_messi:
                    fact_lines.append(
                        "FACT: NOTE — Pele and Bican totals include unofficial "
                        "friendly matches. The officially recognised competitive "
                        "record per FIFA and Guinness World Records is Cristiano Ronaldo."
                    )
                    print(f"[DEBUG] Pele/Bican without Ronaldo — clarification added")

            print(
                f"[DEBUG] _extract_stats_from_context: {len(fact_lines)} facts "
                f"(sport={sport}, entity={is_entity_specific}, entity_type={entity_type}, "
                f"generic={is_generic_record}, format={cricket_format})"
            )
            return "\n".join(fact_lines)

        except Exception as e:
            print(f"[DEBUG] _extract_stats_from_context failed: {e}")
            return ""
    
    # =========================================================================
    # ENTITY EXTRACTORS
    # =========================================================================
    def _extract_tool_params(self, tool: str, query: str) -> dict:
        prompts = {
            "weather": (
                f'Extract the city or location the user wants weather for.\n'
                f'Query: "{query}"\n'
                f'Reply with ONLY a JSON object: {{"location": "city name"}}\n'
                f'If no location found, use {{"location": "India"}}\n'
                f'Only the JSON. No explanation.'
            ),
            "stocks": (
                f'Extract the stock ticker symbol for the company or index mentioned.\n'
                f'Query: "{query}"\n'
                f'Reply with ONLY a JSON object: {{"ticker": "SYMBOL", "company": "name"}}\n'
                f'Use NSE format for Indian stocks (e.g. RELIANCE.NS, TCS.NS, INFY.NS).\n'
                f'Use standard symbols for US stocks (e.g. AAPL, MSFT, TSLA, GOOGL).\n'
                f'Use ^BSESN for Sensex, ^NSEI for Nifty, ^DJI for Dow Jones, ^GSPC for S&P 500.\n'
                f'Only the JSON. No explanation.'
            ),
            "crypto": (
                f'Extract the cryptocurrency name from the query.\n'
                f'Query: "{query}"\n'
                f'Reply with ONLY a JSON object: {{"coin": "coin-id"}}\n'
                f'Use CoinGecko IDs: bitcoin, ethereum, solana, ripple, dogecoin, '
                f'binancecoin, cardano, polkadot, litecoin, shiba-inu, matic-network, avalanche-2.\n'
                f'Only the JSON. No explanation.'
            ),
            "news": (
                f'Extract the main topic or subject the user wants news about.\n'
                f'Query: "{query}"\n'
                f'Reply with ONLY a JSON object: {{"topic": "<extracted topic>"}}\n'
                f'Strip ALL filler/intent words. Keep ONLY the core subject.\n'
                f'Only the JSON. No explanation.'
            ),
        }

        if tool not in prompts:
            return {}

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompts[tool]}],
                options={"temperature": 0.0, "num_predict": 40}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed_params = json.loads(raw)

            if tool == "news" and "topic" in parsed_params:
                _TOPIC_NOISE = {
                    "give", "get", "show", "tell", "find", "fetch",
                    "latest", "recent", "current", "today", "now",
                    "news", "update", "updates", "about", "please",
                    "the", "a", "an", "me", "my",
                }
                topic_words = [
                    w for w in parsed_params["topic"].split()
                    if w.lower() not in _TOPIC_NOISE
                ]
                if topic_words:
                    parsed_params["topic"] = " ".join(topic_words)
                else:
                    parsed_params["topic"] = query

            return parsed_params
        except Exception as e:
            print(f"[DEBUG] _extract_tool_params failed for {tool}: {e}")
            return self._extract_params_fallback(tool, query)

    def _extract_params_fallback(self, tool: str, query: str) -> dict:
        q = query.lower()
        if tool == "weather":
            _WX_NOISE = {
                "weather","temperature","forecast","rain","raining","humidity",
                "wind","climate","sunny","cloudy","cold","hot","warm","snow",
                "snowing","heatwave","monsoon","umbrella","drizzle","will","it",
                "is","the","in","at","for","of","today","tomorrow","this","week",
                "should","i","carry","what","how","does","a","an","right","now",
                "currently","be","like",
            }
            m = re.search(
                r'\b(?:in|at|for|of)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
                query, re.IGNORECASE
            )
            if m:
                loc = m.group(1).strip().rstrip("?,.")
                loc_words = [w for w in loc.split() if w.lower() not in _WX_NOISE]
                loc = " ".join(loc_words).strip()
                if loc and len(loc) > 1:
                    return {"location": loc}
            words = query.split()
            location_words = []
            for w in reversed(words):
                clean = w.strip("?,.")
                if clean and clean[0].isupper() and clean.lower() not in _WX_NOISE:
                    location_words.insert(0, clean)
                elif location_words:
                    break
            if location_words:
                return {"location": " ".join(location_words)}
            remaining = [w for w in q.split() if w not in _WX_NOISE and len(w) > 2]
            return {"location": remaining[-1].title() if remaining else "India"}

        if tool == "stocks":
            COMPANY_MAP = {
                "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
                "wipro": "WIPRO.NS", "hdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS",
                "sbi": "SBIN.NS", "adani": "ADANIENT.NS", "itc": "ITC.NS",
                "sensex": "^BSESN", "nifty": "^NSEI",
                "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
                "tesla": "TSLA", "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
            }
            for name, ticker in sorted(COMPANY_MAP.items(), key=lambda x: -len(x[0])):
                if name in q:
                    return {"ticker": ticker, "company": name.title()}
            m = re.search(r'\b([A-Z]{2,5})\b', query)
            return {"ticker": m.group(1) if m else query.strip(), "company": query.strip()}

        if tool == "crypto":
            COIN_MAP = {
                "bitcoin": "bitcoin", "btc": "bitcoin",
                "ethereum": "ethereum", "eth": "ethereum",
                "solana": "solana", "sol": "solana",
                "ripple": "ripple", "xrp": "ripple",
                "dogecoin": "dogecoin", "doge": "dogecoin",
                "bnb": "binancecoin", "shiba": "shiba-inu",
                "cardano": "cardano", "ada": "cardano",
            }
            for name, coin in COIN_MAP.items():
                if name in q:
                    return {"coin": coin}
            return {"coin": "bitcoin"}

        if tool == "news":
            return {"topic": query}

        return {}

    def _extract_entities_from_query(self, query: str) -> dict:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract only NAMED entities from this query.\n"
                        f'Query: "{query}"\n\n'
                        f"Return ONLY this JSON:\n"
                        f'{"primary": [...], "context": [...]}\n\n'
                        f"Rules:\n"
                        f"- primary: the COUNTRY, COMPANY, PERSON, or TOPIC the user wants news about\n"
                        f"- context: a second named entity providing background — leave [] if none\n"
                        f"- NEVER extract prepositions, question words, or filler words\n"
                        f"  Words to NEVER include: regarding, about, related, concerning,\n"
                        f"  latest, news, the, what, how, why, with, from, between\n\n"
                        f"Examples:\n"
                        f'  "latest news regarding india china relations"\n'
                        f'  → {{"primary":["india china relations"],"context":[]}}\n'
                        f'  "how iran war affects reliance stock"\n'
                        f'  → {{"primary":["reliance"],"context":["iran war"]}}\n'
                        f'  "latest bitcoin news today"\n'
                        f'  → {{"primary":["bitcoin"],"context":[]}}\n\n'
                        f"Only the JSON. No explanation."
                    )
                }],
                options={"temperature": 0.0, "num_predict": 80}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else {}

            _NOISE = {
                # question words
                'how', 'what', 'why', 'when', 'where', 'who', 'which',
                # aux verbs
                'is', 'are', 'was', 'were', 'will', 'can', 'do', 'does',
                'did', 'has', 'have', 'had', 'could', 'would', 'should',
                # articles / short prepositions
                'the', 'a', 'an', 'in', 'on', 'of', 'for', 'and', 'or',
                'at', 'by', 'to', 'up', 'its', 'my', 'me', 'us',
                # filler prepositions that leak into entities
                'regarding', 'related', 'concerning', 'about', 'with',
                'from', 'between', 'against', 'around', 'through', 'into',
                # generic news/query words
                'latest', 'recent', 'news', 'update', 'updates', 'today',
                'current', 'now', 'new', 'get', 'show', 'tell', 'give',
                'find', 'fetch', 'please', 'impact', 'affect', 'effect',
                'war', 'conflict', 'situation', 'issue', 'this', 'that',
                'these', 'those', 'their', 'there', 'then', 'than',
            }

            def _clean(e):
                return e.strip('\'".,;:!? ') if isinstance(e, str) else ""

            seen: set = set()
            primary: list = []
            for e in parsed.get("primary", []):
                ce = _clean(e)
                cl = ce.lower()
                if ce and len(ce) > 1 and cl not in _NOISE and cl not in seen and not re.match(r'^[^a-z0-9]', cl):
                    seen.add(cl)
                    primary.append(ce)

            context: list = []
            for e in parsed.get("context", []):
                ce = _clean(e)
                cl = ce.lower()
                if ce and len(ce) > 1 and cl not in _NOISE and cl not in seen:
                    seen.add(cl)
                    context.append(ce)

            print(f"[DEBUG] _extract_entities_from_query: PRIMARY={primary} | CONTEXT={context}")
            return {"primary": primary, "context": context}

        except Exception as ex:
            print(f"[DEBUG] _extract_entities_from_query failed: {ex}")
            _FALLBACK_COUNTRIES = {
                "iran", "russia", "ukraine", "china", "india", "usa",
                "israel", "pakistan", "saudi", "turkey", "japan", "uk",
            }
            q_low = query.lower()
            countries = [c for c in _FALLBACK_COUNTRIES if c in q_low]
            if countries:
                primary = list(dict.fromkeys(countries))[:2]
            else:
                words = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                primary = list(dict.fromkeys(w.lower() for w in words))[:3]
            print(f"[DEBUG] _extract_entities_from_query fallback: PRIMARY={primary}")
            return {"primary": primary, "context": []}

    # =========================================================================
    # TOOL EXECUTORS
    # =========================================================================
    def _run_tool(self, tool: str, user_input: str, original_query: str = None,
                  tool_query: str = None, preprocessed_query: dict = None,confirmed : bool = False) -> tuple:
        original_query     = original_query     or user_input
        tool_query         = tool_query         or user_input
        preprocessed_query = preprocessed_query or {}
        user_lower         = user_input.lower()

        try:
            if tool == "system":
                result = system_command(user_input.lower())
                if not result:
                    result = "[System command executed but returned no output]"

            elif tool == "github":
                result = handle_github(user_input,confirmed=confirmed)

            elif tool == "news":
                num_results = 5
                num_match = re.search(r'\b(\d+)\s*(point|result|article|news|item)s?\b', user_lower)
                if num_match:
                    num_results = min(int(num_match.group(1)), 10)

                _tp = (preprocessed_query or {}).get("tool_params", {})
                news_topic = _tp.get("topic") or self._extract_tool_params("news", tool_query).get("topic", tool_query)

                # Apply _cleaned override before junk check
                _cleaned = preprocessed_query.get("cleaned", "")
                if _cleaned and _cleaned != user_input and len(_cleaned) >= 4:
                    news_topic = _cleaned

                # Single junk check — catches bad LLM output AND bad _cleaned values
                if not news_topic or news_topic.lower().strip() in _NEWS_TOPIC_JUNK or len(news_topic) < 3:
                    news_topic = original_query

                print(f"[DEBUG] News topic: '{news_topic}'")
                _news_pq = dict(preprocessed_query or {})
                _existing = _news_pq.get("entities", {})
                if not _existing.get("primary"):
                    _news_pq["entities"] = self._extract_entities_from_query(original_query)
                result = get_news(
                    topic=news_topic,
                    num_results=num_results,
                    reranker=self.reranker,
                    original_query=original_query,
                    tool_query=news_topic,
                    preprocessed_query=_news_pq,
                )
            elif tool == "stocks":
                _tp = (preprocessed_query or {}).get("tool_params", {})
                ticker = _tp.get("ticker") or self._extract_tool_params("stocks", tool_query).get("ticker", "")
                # Fallback to original query if tool_query extraction failed
                if not ticker or len(ticker) < 2:
                    ticker = self._extract_tool_params("stocks", user_input).get("ticker", tool_query.strip())
                print(f"[DEBUG] Stocks ticker: '{ticker}'")
                result = get_stock_price(ticker)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[STOCKS FETCH FAILED for '{ticker}' — price unavailable]"

            elif tool == "crypto":
                _tp = (preprocessed_query or {}).get("tool_params", {})
                coin = _tp.get("coin") or self._extract_tool_params("crypto", tool_query).get("coin", "bitcoin")
                print(f"[DEBUG] Crypto coin: '{coin}'")
                result = get_crypto_price(coin)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[CRYPTO FETCH FAILED for '{coin}' — price unavailable]"

            elif tool == "weather":
                _tp = (preprocessed_query or {}).get("tool_params", {})
                location = _tp.get("location") or self._extract_tool_params("weather", tool_query).get("location", "India")
                print(f"[DEBUG] Weather location: '{location}'")
                result = get_weather(location, query=user_lower)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[WEATHER FETCH FAILED for '{location}']"

            elif tool == "sports":
                _sports_q = (preprocessed_query or {}).get("cleaned") or user_input
                _sports_q = self._correct_sports_team_name(_sports_q)
                result = get_sports_scores(_sports_q)

            elif tool == "sports_knowledge":
                # sports_knowledge is handled via _run_sports_knowledge,
                # but _run_tool is called from multi-tool paths — return raw snippets
                intent = (preprocessed_query or {}).get("intent", "knowledge")
                search_queries = self._build_career_search_queries(tool_query) if self._is_career_stats_query(tool_query) else [tool_query, self._expand_query(tool_query)]
                _tl = self._get_timelimit(tool_query, intent)
                raw_snippets: List[str] = []
                for sq in search_queries:
                    raw_snippets.extend(self.internet_search_tool(sq, top_k=SEARCH_RESULTS_LIMIT, timelimit=_tl))
                sports_kb_chunks = get_sports_kb_chunks(
                    query=search_queries[0],
                    embed_model=self.embedding_model,
                    top_k=RERANK_TOP_K,
                    threshold=SPORTS_KB_SIMILARITY_THRESHOLD,
                )
                kb_keys = {hashlib.md5(c.lower().strip().encode()).hexdigest() for c in sports_kb_chunks}
                raw_snippets = list(dict.fromkeys(
                    s for s in raw_snippets 
                    if hashlib.md5(s.lower().strip().encode()).hexdigest() not in kb_keys
                ))
                result = [f"[SPORTS KB]\n{chunk}" for chunk in sports_kb_chunks] + raw_snippets

            elif tool == "internet_search":
                expanded = self._expand_query(tool_query)
                result = self.internet_search_tool(expanded)
            else:
                result = f"[{tool} tool not available]"

        except Exception as e:
            result = f"[{tool} tool error: {e}]"

        print(f"[DEBUG] Tool '{tool}' completed.")
        return (tool, result)

    def _run_tool_with_subquery(self, tool: str, sub_question: str,
                                 original_query: str, preprocessed_query: dict = None, confirmed: bool = False) -> tuple:
        preprocessed_query = preprocessed_query or {}
        sub_lower          = sub_question.lower()

        try:
            if tool == "system":
                result = system_command(sub_question.lower())
                if not result:
                    result = "[System command executed but returned no output]"

            elif tool == "github":
                result = handle_github(sub_question)

            elif tool == "news":
                num_results = 5
                num_match = re.search(r'\b(\d+)\s*(point|result|article|news|item)s?\b', sub_lower)
                if num_match:
                    num_results = min(int(num_match.group(1)), 10)

                _tp = (preprocessed_query or {}).get("tool_params", {})
                news_topic = _tp.get("topic") or self._extract_tool_params("news", sub_question).get("topic", sub_question)

                # Junk guard — same as _run_tool
                if not news_topic or news_topic.lower().strip() in _NEWS_TOPIC_JUNK or len(news_topic) < 3:
                    news_topic = original_query

                print(f"[DEBUG] News topic (sub): '{news_topic}'")
                _news_pq = dict(preprocessed_query or {})
                _existing = _news_pq.get("entities", {})
                if not _existing.get("primary"):
                    _news_pq["entities"] = self._extract_entities_from_query(original_query)
                result = get_news(
                    topic=news_topic,
                    num_results=num_results,
                    reranker=self.reranker,
                    original_query=original_query,
                    tool_query=news_topic,
                    preprocessed_query=_news_pq,
                )
            elif tool == "stocks":
                params = self._extract_tool_params("stocks", sub_question)
                ticker = params.get("ticker", sub_question.strip())
                if not ticker or len(ticker) < 2:
                    params = self._extract_tool_params("stocks", original_query)
                    ticker = params.get("ticker", sub_question.strip())
                print(f"[DEBUG] Stocks ticker (sub): '{ticker}'")
                result = get_stock_price(ticker)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[STOCKS FETCH FAILED for '{ticker}' — price unavailable]"

            elif tool == "crypto":
                params = self._extract_tool_params("crypto", sub_question)
                coin = params.get("coin", "bitcoin")
                print(f"[DEBUG] Crypto coin (sub): '{coin}'")
                result = get_crypto_price(coin)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[CRYPTO FETCH FAILED for '{coin}' — price unavailable]"

            elif tool == "weather":
                params = self._extract_tool_params("weather", sub_question)
                location = params.get("location", "India")
                print(f"[DEBUG] Weather location (sub): '{location}')")
                result = get_weather(location, query=sub_lower)
                if not result or len(result.strip()) < 10 or result.startswith("["):
                    result = f"[WEATHER FETCH FAILED for '{location}']"

            elif tool == "sports":
                _sports_q = (preprocessed_query or {}).get("cleaned") or original_query
                result = get_sports_scores(self._correct_sports_team_name(_sports_q))

            elif tool == "sports_knowledge":
                intent = (preprocessed_query or {}).get("intent", "knowledge")
                search_queries = (self._build_career_search_queries(sub_question)
                                  if self._is_career_stats_query(sub_question)
                                  else [sub_question, self._expand_query(sub_question)])
                _tl = self._get_timelimit(sub_question, intent)
                raw_snippets: List[str] = []
                for sq in search_queries:
                    raw_snippets.extend(self.internet_search_tool(sq, top_k=SEARCH_RESULTS_LIMIT, timelimit=_tl))
                sports_kb_chunks = get_sports_kb_chunks(
                    query=search_queries[0],
                    embed_model=self.embedding_model,
                    top_k=RERANK_TOP_K,
                    threshold=SPORTS_KB_SIMILARITY_THRESHOLD,
                )
                kb_keys = {hashlib.md5(c.lower().strip().encode()).hexdigest() for c in sports_kb_chunks}
                raw_snippets = list(dict.fromkeys(
                s for s in raw_snippets
                if hashlib.md5(s.lower().strip().encode()).hexdigest() not in kb_keys
                ))
                result = [f"[SPORTS KB]\n{chunk}" for chunk in sports_kb_chunks] + raw_snippets

            elif tool == "internet_search":
                expanded = self._expand_query(sub_question)
                result = self.internet_search_tool(expanded)

            else:
                result = f"[{tool} tool not available]"

        except Exception as e:
            result = f"[{tool} tool error: {e}]"

        print(f"[DEBUG] Tool '{tool}' completed (sub-question: '{sub_question[:60]}...')")
        return (tool, result)

    # =========================================================================
    # MERGE TOOL RESULTS
    # =========================================================================
    def _merge_tool_results(self, user_input: str, tool_contexts: dict,
                             sub_questions: dict = None) -> str:
        sub_questions = sub_questions or {}

        sections = []
        for tool, context_str in tool_contexts.items():
            label = tool.upper().replace("_", " ")
            sub_q = sub_questions.get(tool, "")
            sub_q_line = f"[Retrieved to answer: \"{sub_q}\"]\n" if sub_q else ""
            sections.append(f"=== {label} ===\n{sub_q_line}{context_str}")
        combined = "\n\n".join(sections)

        tool_list = ", ".join(tool_contexts.keys())
        num_tools = len(tool_contexts)

        messages = self._build_message_list(user_input)
        messages = self._trim_messages_to_token_limit(messages)
        messages[-1]["content"] = (
            f"The user asked: {user_input}\n\n"
            f"You have data from {num_tools} independent sources ({tool_list}).\n"
            f"Each section below was retrieved to answer a specific sub-question.\n\n"
            f"{combined}\n\n"
            f"STRICT RULES:\n"
            f"1. START your response with the stock/crypto/weather price from the direct data section.\n"
            f"   Use the exact figures shown. Do not round or modify them.\n"
            f"2. Then give the news bullets from the NEWS section — one bullet per article.\n"
            f"   Each bullet: **[headline]** *(source, date)* — [2-3 sentence summary].\n"
            f"3. Each === section === is the sole authority for its domain.\n"
            f"4. NEVER cross-validate sections.\n"
            f"5. NEVER say 'not mentioned' or 'not available' for data that IS present.\n"
            f"6. FETCH FAILED rule: if a section contains '[FETCH FAILED]' or '[tool error]', "
            f"   write: 'Data could not be fetched at this time.'\n"
            f"7. Do NOT invent facts.\n"
            f"8. Give each section its own ### heading.\n"
            f"9. Write at least 3-5 news bullets if available. Do NOT stop after the price.\n"
            f"10. Stop immediately after the last news bullet.\n\n"
            f"FORMAT:\n"
            f"## {user_input.strip().title()}\n"
            f"### Share Price\n"
            f"[exact price line]\n\n"
            f"### Latest News\n"
            f"- **[headline]** *(source, date)* — [summary]\n"
            f"- **[headline]** *(source, date)* — [summary]\n"
        )

        response = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.1,
                "num_predict": 2000,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
            }
        )
        result = ""
        for chunk in response:
            result += chunk["message"]["content"]
        return self._clean_response(result)

    # =========================================================================
    # MAIN QUERY AGENT
    # =========================================================================
    def query_agent(self, user_input: str, confirmed: bool = False) -> str:

        print(f"[DEBUG PRE] confidence={self.get_kb_confidence(user_input):.3f}")
        docs = self.hybrid_kb_search(user_input, k=5)
        print(f"[DEBUG PRE] top doc: {docs[0][:100] if docs else 'EMPTY'}")

        self._check_session_timeout()
        user_lower = user_input.lower()

        # ── Abbreviation expansion for KB matching ────────────────────────────
        _ABBREV_MAP = {
            r'\br\.?a\.?g\.?\b':     'retrieval augmented generation',
            r'\bllm\b':              'large language model',
            r'\bml\b':               'machine learning',
            r'\bdl\b':               'deep learning',
            r'\bnlp\b':              'natural language processing',
            r'\bcv\b':               'computer vision',
            r'\bapi\b':              'application programming interface',
            r'\brag\b':              'retrieval augmented generation',
        }
        _expanded_input = user_input
        for pattern, expansion in _ABBREV_MAP.items():
            _expanded_input = re.sub(pattern, expansion, _expanded_input, flags=re.IGNORECASE)

        if _expanded_input != user_input:
            print(f"[DEBUG] Abbreviation expanded: {repr(user_input)} → {repr(_expanded_input)}")

        _is_knowledge_q = bool(re.search(
            r'^\s*(how|what|why|when|where|can|should|is|are|do|does|did'
            r'|will|would|could|tell me|explain|describe|give me|list)\b',
            user_lower
        ))
        _ACTION_LEADING = not _is_knowledge_q and bool(re.search(
            r'^\s*(open\s+(?!ai\b)|launch|close|quit|start|delete|remove|uninstall'
            r'|create|make|move|copy|rename|set|remind|reminder|schedule'
            r'|take\s+screenshot|screenshot|mute|unmute'
            r'|increase|decrease|shutdown|restart|sleep'
            r'|lock|empty|mark|add\s+event|add\s+birthday'
            r'|play|watch|search\s+.+\s+(?:on|in)\s+youtube'
            r'|open\s+youtube|youtube)\b',
            user_lower
        ))

        _is_ownership_q = bool(re.search(
            r'\b(ceo|founder|co-founder|chairman|chief executive|who owns|'
            r'who runs|who leads|who started|who created|who built|'
            r'founded by|owner of|who is behind)\b',
            user_lower, re.IGNORECASE
        ))
        _is_github_op = bool(re.search(
            r'\b(repo|repository|github)\b.{0,50}\b(delete|remove|add|push|upload|rename|create)\b'
            r'|\b(delete|remove|add|push|upload|rename|create)\b.{0,50}\b(repo|repository|github)\b'
            r'|\bfrom\b.{0,30}\b(repo|repository|github)\b',
            user_lower, re.IGNORECASE
        ))
        if _ACTION_LEADING and not _is_ownership_q and not _is_github_op:
            _sys_result = system_command(user_lower)
            if _sys_result:
                self.add_to_memory(user_input, _sys_result)
                return _sys_result

        FIXED_RESPONSES = [
            (["hello", "hi", "hey", "hii"],
             "Hello! How can I assist you today?", "word"),
            (["how are you", "how r you", "how are u"],
             "I'm doing great and ready to help!", "phrase"),
            (["bye", "goodbye", "see you", "see ya"],
             "Goodbye! Have a productive day.", "word"),
            (["thanks", "thank you", "thx", "thankyou"],
             "You're welcome! Let me know if you need any more help.", "word"),
            (["sorry", "apologies", "my bad"],
             "No worries! Let's continue with your tasks.", "word"),
            (["what can you do", "your capabilities", "what do you do"],
             (
                 "I can answer questions from my knowledge base, assist with Git commands, "
                 "Python automation, macOS terminal commands, weather, stocks, crypto, "
                 "sports scores, sports facts & records, news, and general troubleshooting."
             ), "phrase"),
        ]

        for triggers, response, match_type in FIXED_RESPONSES:
            for trigger in triggers:
                if match_type == "phrase":
                    if trigger in user_lower:
                        self.add_to_memory(user_input, response)
                        return response
                else:
                    if re.search(rf'\b{re.escape(trigger)}\b', user_lower):
                        self.add_to_memory(user_input, response)
                        return response
        # ── KB pre-check — runs BEFORE all other routing ─────────────────────
        # If KB has high-confidence relevant content, use it directly
        # This makes RAG work for ANY document you ingest without keyword config
        # ── KB pre-check — two-stage, fully general ──────────────────────────
        # In query_agent, after FIXED_RESPONSES loop
        # ── KB pre-check — semantic gate using reranker ───────────────────────
        # ── KB pre-check ──────────────────────────────────────────────────────
        _kb_query = _expanded_input
        _kb_pre_confidence = self.get_kb_confidence(_kb_query)
        _kb_pre_docs = self.hybrid_kb_search(_kb_query, k=RETRIEVAL_TOP_K * 3)
        print(f"[DEBUG] KB pre-confidence: {_kb_pre_confidence:.3f}")


        if self._should_use_kb(_kb_pre_confidence, _kb_pre_docs, _kb_query):
            print(f"[DEBUG] KB pre-check HIT")
            preprocessed_query = self._preprocess_query(user_input)
            fmt        = preprocessed_query.get("format", "default")
            num_points = preprocessed_query.get("num_points")
            intent     = "document"

            _fmt_map = {
                "brief":      "Be brief. Only list the most relevant facts from the document.",
                "summary":    "Summarise only what is in the document. 3-5 bullet points max.",
                "detailed":   "List all sections found in the document with their content.",
                "definition": "Give a one paragraph description based only on the document.",
                "points":     f"Give exactly {num_points} points only." if num_points else "",
                "default":    "",
            }

            # For document queries, pack ALL retrieved chunks without reranking.
            # Reranking drops relevant chunks (e.g. education section scores lower
            # than header chunk for "what is education" query).
            # The LLM will find the relevant part itself from the full context.
            context = self.build_kb_context(
                kb_docs=_kb_pre_docs,   # ← all docs, no reranking
                original_query=user_input,
                is_document=True,
            )

            if context and len(context.strip()) > 50:
                result = self._llm_call(
                    user_input=user_input,
                    context=context,
                    intent=intent,
                    length_instruction=_fmt_map.get(fmt, ""),
                    num_points=num_points,
                    context_label="Context from Knowledge Base",
                )
                self.add_to_memory(user_input, result)
                gc.collect()
                return result
            print(f"[DEBUG] KB precheck: context empty — falling through")
# ── END KB precheck ───────────────────────────────────────────────────
        
        # ── How-does-X-work fast path ─────────────────────────────────────────
        _how_does_result = _classify_how_does_work(user_input)
        print(f"[DEBUG] _classify_how_does_work result: {_how_does_result}")
        _has_compound = any(re.search(p, user_lower) for p in [
            r'\band\b', r'\balso\b', r'\balong with\b', r'\bplus\b', r'\bas well as\b'
        ])
        if not _how_does_result and not _has_compound:
            _how_does_result = _detect_explanation_query(user_input)
            if _how_does_result:
                print(f"[DEBUG] _detect_explanation_query fallback triggered")
        if _how_does_result and not _has_compound:
            _hw_tool   = _how_does_result["tool"]
            _hw_intent = _how_does_result["intent"]
            _hw_cat    = _how_does_result["category"]
            _hw_subject = _how_does_result.get("subject", user_input)

            # ── LLM classification for unknown subjects ───────────────────────
            # Triggered when neither _classify_how_does_work nor _detect_explanation_query
            # could match the subject to a known category via regex.
            if _hw_intent == "__llm_classify__":
                try:
                    _llm_cat_response = ollama.chat(
                        model=self.ollama_model,
                        messages=[{
                            "role": "user",
                            "content": (
                                f'Classify what kind of knowledge this question requires.\n\n'
                                f'Question: "{user_input}"\n\n'
                                f'Reply with ONLY one of these category names:\n'
                                f'  scientific   — physics, chemistry, biology, astronomy, geology,\n'
                                f'                 natural phenomena, medicine, health, environment\n'
                                f'  technology   — AI, software, hardware, internet, networks,\n'
                                f'                 computing, devices, algorithms, programming concepts\n'
                                f'  financial    — economics, markets, banking, investing, trade,\n'
                                f'                 currencies, tax, insurance\n'
                                f'  historical   — history, civilizations, wars, politics, culture,\n'
                                f'                 geography, social movements\n'
                                f'  sports       — sports rules, gameplay, tournaments, athletes\n'
                                f'  general      — everything else\n\n'
                                f'Reply with ONLY the category name. Nothing else.'
                            )
                        }],
                        options={"temperature": 0.0, "num_predict": 5}
                    )
                    _llm_cat = _llm_cat_response["message"]["content"].strip().lower()
                    _llm_cat = re.sub(r'[^a-z]', '', _llm_cat)
                    print(f"[DEBUG] LLM category for unknown subject: '{_llm_cat}'")

                    _CAT_TO_INTENT = {
                        "scientific":  "scientific",
                        "technology":  "scientific",
                        "financial":   "facts",
                        "historical":  "knowledge",
                        "sports":      "sports_knowledge",
                        "general":     "knowledge",
                    }
                    _hw_intent = _CAT_TO_INTENT.get(_llm_cat, "knowledge")
                    _hw_tool   = "sports_knowledge" if _llm_cat == "sports" else "internet_search"
                    print(f"[DEBUG] LLM classified unknown subject: "
                        f"category={_llm_cat}, intent={_hw_intent}, tool={_hw_tool}")

                except Exception as e:
                    print(f"[DEBUG] LLM classification failed: {e} — defaulting to knowledge")
                    _hw_intent = "knowledge"
                    _hw_tool   = "internet_search"

            print(f"[DEBUG] How-does-work fast path: "
                f"tool={_hw_tool}, intent={_hw_intent}, category={_hw_cat}")

            preprocessed_query = self._preprocess_query(user_input)
            preprocessed_query["intent"] = _hw_intent
            fmt        = preprocessed_query.get("format", "default")
            num_points = preprocessed_query.get("num_points")
            _fmt_map = {
                "brief":      "Keep your response brief and concise.",
                "summary":    "Provide a clear, structured summary in 3-5 bullet points.",
                "detailed":   "Give a detailed and thorough explanation.",
                "definition": "Give only a brief one paragraph definition.",
                "points":     f"Give exactly {num_points} points only." if num_points else "",
                "default":    "",
            }
            length_instruction = _fmt_map.get(fmt, "")

            if _hw_tool == "sports_knowledge":
                result = self._run_sports_knowledge(
                    query=user_input,
                    intent=_hw_intent,
                    preprocessed_query=preprocessed_query,
                    length_instruction=length_instruction,
                    num_points=num_points,
                )
                self.add_to_memory(user_input, result)
                gc.collect()
                return result

            if _hw_tool == "kb":
                kb_docs = self.hybrid_kb_search(user_input, k=RETRIEVAL_TOP_K * 2)
    
                # Check if KB actually has relevant content using the reranker gate
                _kb_conf = self.get_kb_confidence(user_input)
                _kb_relevant = self._should_use_kb(_kb_conf, kb_docs, user_input)
    
                if _kb_relevant:
                    context = self.build_kb_context(kb_docs=kb_docs, original_query=user_input)
                else:
                    context = ""  # force fallback to internet search
    
                if not context or len(context.strip()) < 50:
                    # KB doesn't have it — fall through to internet search
                    raw_snippets = self.internet_search_tool(
                        self._expand_query(user_input),
                        top_k=SEARCH_RESULTS_LIMIT
                    )
                    _kb_fallback_blocks = self.format_tool_output(raw_snippets, "internet_search")
                    context = self.build_context(
                        blocks=_kb_fallback_blocks,
                        original_query=user_input,
                        preprocessed_query=preprocessed_query,
                    )
                result = self._llm_call(
                    user_input=user_input,
                    context=context,
                    intent=_hw_intent,
                    length_instruction=length_instruction,
                    num_points=num_points,
                    context_label="Context",
                )
                self.add_to_memory(user_input, result)
                return result

            # internet_search path
            _tl = self._get_timelimit(user_input, _hw_intent)
            expanded = self._expand_query(user_input)
            raw_snippets  = self.internet_search_tool(user_input, top_k=SEARCH_RESULTS_LIMIT, timelimit=_tl)
            raw_snippets += self.internet_search_tool(expanded, top_k=SEARCH_RESULTS_LIMIT, timelimit=_tl)
            raw_snippets  = list(dict.fromkeys(raw_snippets))
            blocks  = self.format_tool_output(raw_snippets, "internet_search")
            context = self.build_context(
                blocks=blocks,
                original_query=user_input,
                preprocessed_query=preprocessed_query,
            )
            if not context or len(context.strip()) < 50:
                context = "No reliable information found for this query."
            result = self._llm_call(
                user_input=user_input,
                context=context,
                intent=_hw_intent,
                length_instruction=length_instruction,
                num_points=num_points,
                context_label="Context from internet search",
                fmt=fmt,
            )
            self.add_to_memory(user_input, result)
            gc.collect()
            return result
        


        # ── Early routing ─────────────────────────────────────────────────
        _early_tools = self._route_tools(user_input)

        # ── Sports fast-path — live scores only ───────────────────────────
        if _early_tools == ["sports"] or (_early_tools[0] == "sports" and len(_early_tools) == 1):
            _sports_intent = self._classify_intent(user_input)
            _KNOWLEDGE_INTENTS = {"knowledge", "facts", "sports_knowledge", "general", "ownership"}

            _is_how_many = bool(re.search(
                r'\bhow many\b|\bhow much\b|\bcareer\b|\btotal runs\b|'
                r'\bbatting average\b|\bbowling average\b|\bmost runs\b|'
                r'\bmost wickets\b|\bmost goals\b|\bmost centuries\b',
                user_lower
            ))
            if _is_how_many:
                _sports_intent = "knowledge"

            _is_score_query = (
                not self._is_career_stats_query(user_input)
            ) and bool(re.search(
                r'\b(scorecard|latest score|latest result|match result|'
                r'final score|live score|today score|today match|'
                r'last match|last game|recent match|live|'
                r'upcoming|next match|next game|fixture|between)\b',
                user_lower
            ))
            _is_historical = bool(
                _sports_intent == "facts" and
                re.search(r'\b(20\d{2})\b|\b(world cup|tournament|series|championship|ipl|final)\b', user_lower) and
                not re.search(r'\b(last night|yesterday|today|this week|recent|last match|live)\b', user_lower)
            )

            # Redirect sports_knowledge queries away from live sports fast-path
            
            if _is_historical or (_sports_intent in _KNOWLEDGE_INTENTS and not _is_score_query) or _is_how_many:
                print(f"[DEBUG] Sports fast-path: redirecting to sports_knowledge (intent={_sports_intent})")
                preprocessed_query = self._preprocess_query(user_input)
                intent     = preprocessed_query["intent"]
                fmt        = preprocessed_query["format"]
                num_points = preprocessed_query["num_points"]
                _sem_intent = self._semantic_intent_check(user_input)
                if _sem_intent and intent not in ("sports_knowledge", "facts", "weather", "news"):
                    print(f"[DEBUG] Semantic intent override: {intent} → {_sem_intent}")
                    intent = _sem_intent
                    preprocessed_query["intent"] = intent
                _fmt_map = {
                    "brief":      "Keep your response brief and concise.",
                    "summary":    "Provide a clear, structured summary. Cover the main points in 3-5 sentences or bullet points. Do not give a full detailed answer — summarize only.",
                    "detailed":   "Give a detailed and thorough explanation.",
                    "definition": "Give only a brief one paragraph definition. Nothing else.",
                    "points":     f"Give exactly {num_points} points only." if num_points else "", 
                    "default":    "",
                }
                length_instruction = _fmt_map.get(fmt, "")
                result = self._run_sports_knowledge(
                    query=user_input,
                    intent=intent,
                    preprocessed_query=preprocessed_query,
                    length_instruction=length_instruction,
                    num_points=num_points,
                )
                self.add_to_memory(user_input, result)
                gc.collect()
                return result
            elif not _is_historical and (_sports_intent not in _KNOWLEDGE_INTENTS or _is_score_query):
                print(f"[DEBUG] Sports fast-path: live scores, bypassing preprocessing")
                _corrected_sports_q = self._correct_sports_team_name(user_input)
                print(f"[DEBUG] Sports query cleaned: {repr(user_input)} → {repr(_corrected_sports_q)}")
                _sports_result = get_sports_scores(_corrected_sports_q)
                if _sports_result and len(_sports_result.strip()) > 20:
                    self.add_to_memory(user_input, _sports_result)
                    return _sports_result
                fallback = f"No sports data found for: {user_input}"
                self.add_to_memory(user_input, fallback)
                return fallback

        # ── sports_knowledge fast-path ─────────────────────────────────────
        if _early_tools == ["sports_knowledge"] or (
            len(_early_tools) == 1 and _early_tools[0] == "sports_knowledge"
        ):
            preprocessed_query = self._preprocess_query(user_input)
            intent    = preprocessed_query["intent"]
            fmt       = preprocessed_query["format"]
            num_points = preprocessed_query["num_points"]
            _sem_intent = self._semantic_intent_check(user_input)
            if _sem_intent and intent not in ("sports_knowledge", "facts", "weather", "news"):
                print(f"[DEBUG] Semantic intent override (sk fast-path): {intent} → {_sem_intent}")
                intent = _sem_intent
                preprocessed_query["intent"] = intent

            _fmt_map = {
                "brief":      "Keep your response brief and concise.",
                "summary":    "Provide a clear, structured summary. Cover the main points in 3-5 sentences or bullet points. Do not give a full detailed answer — summarize only.",
                "detailed":   "Give a detailed and thorough explanation.",
                "definition": "Give only a brief one paragraph definition. Nothing else.",
                "points":     f"Give exactly {num_points} points only." if num_points else "",
                "default":    "",
            }
            length_instruction = _fmt_map.get(fmt, "")

            # Normalise intent for sports knowledge
            if intent not in ("knowledge", "facts", "sports_knowledge", "scientific"):
                intent = "sports_knowledge"

            print(f"[DEBUG] sports_knowledge fast-path: intent={intent}")
            result = self._run_sports_knowledge(
                query=user_input,
                intent=intent,
                preprocessed_query=preprocessed_query,
                length_instruction=length_instruction,
                num_points=num_points,
            )
            self.add_to_memory(user_input, result)
            gc.collect()
            return result

        # ── Preprocess ────────────────────────────────────────────────────
        preprocessed_query = self._preprocess_query(user_input)
        cleaned_query      = preprocessed_query["cleaned"]
        intent             = preprocessed_query["intent"]
        fmt                = preprocessed_query["format"]
        num_points         = preprocessed_query["num_points"]

        _fmt_map = {
            "brief":      "Keep your response brief and concise.",
            "summary":    "Provide a clear, structured summary. Cover the main points in 3-5 sentences or bullet points. Do not give a full detailed answer — summarize only.",
            "detailed":   "Give a detailed and thorough explanation.",
            "definition": "Give only a brief one paragraph definition. Nothing else.",
            "points":     f"Give exactly {num_points} points only. Nothing more." if num_points else "",
            "default":    "",
        }
        length_instruction = _fmt_map.get(fmt, "")

        tools = _early_tools
        tool  = tools[0]

        # ── Post-preprocess sports_knowledge redirect ──────────────────────
        # Catches cases where early routing was ambiguous and intent clarifies it
        if tool in ("sports", "internet_search") and _is_sports_knowledge_query(user_input, intent):
            print(f"[DEBUG] Post-preprocess redirect: {tool} → sports_knowledge "
                  f"(intent={intent}, query is sports knowledge)")
            tool  = "sports_knowledge"
            tools = ["sports_knowledge"]

        # ── Multi-tool path ────────────────────────────────────────────────
        if len(tools) > 1:
            print(f"[DEBUG] Running {len(tools)} tools in parallel: {tools}")
            sub_questions = self._decompose_query(user_input, tools)

            raw_tool_results = {}
            with ThreadPoolExecutor(max_workers=len(tools)) as executor:
                futures = {
                    executor.submit(
                        self._run_tool_with_subquery,
                        t,
                        sub_questions.get(t, user_input),
                        user_input,
                        preprocessed_query,
                        confirmed,
                    ): t
                    for t in tools
                }
                for future in as_completed(futures):
                    t_name, t_result = future.result()
                    raw_tool_results[t_name] = t_result

            DIRECT_TOOLS = {"system", "github", "weather", "stocks", "crypto"}
            combined_blocks = []
            direct_contexts = {}

            for t_name in tools:
                raw_result = raw_tool_results.get(t_name)
                if t_name in DIRECT_TOOLS:
                    direct_contexts[t_name] = raw_result if isinstance(raw_result, str) else str(raw_result)
                    print(f"[DEBUG] {t_name}: direct answer ({len(str(raw_result).split())} tokens)")
                else:
                    if isinstance(raw_result, list):
                        # sports_knowledge returns a mixed list of [SPORTS KB] + snippet strings
                        if raw_result and isinstance(raw_result[0], dict):
                        # News tool returns list of dicts — use format_tool_output to convert
                            blocks = self.format_tool_output(
                                tool_output=raw_result,
                                tool_name=t_name,
                            )
                        else:
                        # sports_knowledge returns mixed list of [SPORTS KB] + snippet strings
                            blocks = []
                            for item in raw_result:
                                if isinstance(item, str):
                                    blocks.append(item)
                        if t_name == "news":
                            blocks = [
                                re.sub(
                                    r'\[NEWS\]\[(?:TRUSTED|unverified)\](?:\[score:[^\]]+\])?\s*',
                                    '', blk
                                ).strip()
                                for blk in blocks
                            ]
                    else:
                        blocks = self.format_tool_output(
                            tool_output=raw_result,
                            tool_name=t_name,
                        )
                        if t_name == "news":
                            blocks = [
                                re.sub(
                                    r'\[NEWS\]\[(?:TRUSTED|unverified)\](?:\[score:[^\]]+\])?\s*',
                                    '', blk
                                ).strip()
                                for blk in blocks
                            ]
                    combined_blocks.extend(blocks)
                    print(f"[DEBUG] {t_name}: {len(blocks)} blocks added to combined")

            print(f"[DEBUG] Combined blocks total: {len(combined_blocks)} from {len(tools)} tools")

            # Build per-tool contexts separately so reranker stays focused on each tool's sub-question
            _tool_sub_contexts = {}
            for t_name in tools:
                if t_name not in direct_contexts:
                    t_blocks = [b for b in combined_blocks if t_name in tools]
                    # Use the sub-question for reranking, not the full user query
                    _sub_q = sub_questions.get(t_name, user_input)
                    _t_specific_blocks = [
                        b for b in combined_blocks
                        if raw_tool_results.get(t_name) and b in (
                            [b for b in combined_blocks]  # all blocks for this tool
                        )
                    ]
            _tool_sub_contexts = {}
            # Track which blocks belong to which tool
            _tool_block_map = {}
            _block_offset = 0
            for t_name in tools:
                if t_name not in direct_contexts:
                    raw_result = raw_tool_results.get(t_name)
                    if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
                        t_blocks = self.format_tool_output(raw_result, t_name)
                elif isinstance(raw_result, list):
                    t_blocks = [item for item in raw_result if isinstance(item, str)]
                else:
                    t_blocks = self.format_tool_output(raw_result, t_name)
                if t_name == "news":
                    t_blocks = [re.sub(r'\[NEWS\]\[(?:TRUSTED|unverified)\](?:\[score:[^\]]+\])?\s*', '', b).strip() for b in t_blocks]
                _sub_q = sub_questions.get(t_name, user_input)
                _tool_sub_contexts[t_name] = self.build_context(
                    blocks=t_blocks,
                    original_query=_sub_q,   # ← rerank against sub-question, not full query
                    preprocessed_query=preprocessed_query,
                )
                print(f"[DEBUG] {t_name}: per-tool context built ({len(t_blocks)} blocks, sub_q='{_sub_q[:50]}')")

            combined_context = "\n\n".join(_tool_sub_contexts.values()) if _tool_sub_contexts else ""

            if direct_contexts:
                direct_str = "\n\n".join(
                    f"[{t.upper()} DATA]\n{ctx}" for t, ctx in direct_contexts.items()
                )
                combined_context = (direct_str + "\n\n" + combined_context).strip()

            if not combined_context or len(combined_context.strip()) < 20:
                combined_context = "No relevant data retrieved from any tool."

            print(f"[DEBUG] build_context output: {len(combined_context.split())} tokens")

            tool_contexts = {}
            for t_name in tools:
                if t_name in direct_contexts:
                    tool_contexts[t_name] = direct_contexts[t_name]
                elif t_name in _tool_sub_contexts:
                    tool_contexts[t_name] = _tool_sub_contexts[t_name]   # ← per-tool, not combined

            result = self._merge_tool_results(
                user_input=user_input,
                tool_contexts=tool_contexts,
                sub_questions=sub_questions,
            )
            self.add_to_memory(user_input, result)
            return result

        print(f"[DEBUG] Intent: {intent}, Tool: {tool}")

        # ── Single-tool paths ──────────────────────────────────────────────

        if tool == "system":
            result = system_command(user_lower)
            if result:
                self.add_to_memory(user_input, result)
                return result

        if tool == "github":
            result = handle_github(user_input,confirmed=confirmed)
            self.add_to_memory(user_input, result)
            return result

        # ── sports_knowledge single-tool ───────────────────────────────────
        if tool == "sports_knowledge":
            if intent not in ("knowledge", "facts", "sports_knowledge", "scientific"):
                intent = "sports_knowledge"
            result = self._run_sports_knowledge(
                query=user_input,
                intent=intent,
                preprocessed_query=preprocessed_query,
                length_instruction=length_instruction,
                num_points=num_points,
            )
            self.add_to_memory(user_input, result)
            gc.collect()
            return result

        if tool == "sports":
            _pre_sports_intent = intent
            _KNOWLEDGE_INTENTS_PRE = {"knowledge", "facts", "sports_knowledge", "general", "ownership"}

            # Redirect sports → sports_knowledge if knowledge intent confirmed
            if _is_sports_knowledge_query(user_input, _pre_sports_intent):
                print(f"[DEBUG] Sports single-tool: redirecting to sports_knowledge pipeline")
                if _pre_sports_intent not in ("knowledge", "facts", "sports_knowledge", "scientific"):
                    _pre_sports_intent = "sports_knowledge"
                result = self._run_sports_knowledge(
                    query=user_input,
                    intent=_pre_sports_intent,
                    preprocessed_query=preprocessed_query,
                    length_instruction=length_instruction,
                    num_points=num_points,
                )
                self.add_to_memory(user_input, result)
                return result

            # Live/upcoming sports — use get_sports_scores directly
            _corrected_sports_q = self._correct_sports_team_name(user_input)
            sports_raw = get_sports_scores(_corrected_sports_q)
            if sports_raw and len(sports_raw.strip()) > 20:
                print(f"[DEBUG] Sports: returning raw sports_tool output directly")
                self.add_to_memory(user_input, sports_raw)
                return sports_raw

            fallback_msg = f"No sports data found for: {user_input}"
            self.add_to_memory(user_input, fallback_msg)
            return fallback_msg

        # ── Single sub-query tools ─────────────────────────────────────────
        if tool in ("sports",):
            sub_queries = [user_input]
        else:
            sub_queries = self._decompose_single_query(user_input, tool)
        tool_query = sub_queries[0] if sub_queries else user_input

        _, raw_result = self._run_tool(
            tool=tool,
            user_input=user_input,
            original_query=user_input,
            tool_query=tool_query,
            preprocessed_query=preprocessed_query,
            confirmed=confirmed,
        )

        if tool == "weather":
            raw_str = raw_result if isinstance(raw_result, str) else str(raw_result)
            self.add_to_memory(user_input, raw_str)
            return raw_str

        blocks = self.format_tool_output(
            tool_output=raw_result,
            tool_name=tool,
        )
        context = self.build_context(
            blocks=blocks,
            original_query="" if tool == "news" else user_input,
            preprocessed_query=preprocessed_query,
        )

        if not context or len(context.strip()) < 50:
            context = "No reliable information found for this query."

        if tool == "news":
            import re as _re
            packed_news = ""
            total_news_tokens = 0
            reranked_blocks = context.split("\n\n")
            for blk in reranked_blocks:
                clean_blk = _re.sub(
                    r'\[NEWS\]\[(?:TRUSTED|unverified)\](?:\[score:[^\]]+\])?\s*', '', blk
                ).strip()
                if not clean_blk:
                    continue
                tokens = len(clean_blk.split())
                if total_news_tokens + tokens > MAX_TOKENS:
                    break
                packed_news += clean_blk + "\n\n"
                total_news_tokens += tokens
            news_context = packed_news.strip() or "No news data found."
            news_context = re.sub(r'https?://\S+', '', news_context).strip()
            _raw_seen_sources = set()
            for blk in packed_news.strip().split("\n\n"):
                blk = blk.strip()
                if not blk or len(blk) < 50:
                    continue
                # Extract source from "(source, date)" pattern
                _src_m = re.search(r'\(([^,)]+),\s*\d{4}', blk)
                if _src_m:
                    _raw_seen_sources.add(_src_m.group(1).strip().lower())
                else:
                    _raw_seen_sources.add(blk[:40])
            news_block_count = max(len(_raw_seen_sources),1)
            print(f"[DEBUG] news_block_count: {news_block_count}")
            if intent == "news" and preprocessed_query.get("tool_params", {}).get("topic", ""):
                _topic_check = preprocessed_query.get("tool_params", {}).get("topic", "")
            else:
                _topic_check = user_input
            from rag.tools.news_tool import _classify_entertainment_query as _cet
            if _cet(_topic_check, re.sub(r'(\D)(\d+)', r'\1 \2', _topic_check)):
                news_block_count = max(news_block_count, 3)
                print(f"[DEBUG] Entertainment query — news_block_count raised to min 3")
            result = self._llm_call(
                user_input=user_input,
                context=news_context,
                intent="news",
                length_instruction=length_instruction,
                num_points=None,
                context_label="News Context",
                news_article_count=news_block_count,
            )
            _JUNK_LINES = {"no stories found.", "no results.", "no data.", "not available."}

            bullet_lines = [
                line for line in result.split("\n")
                if line.strip().lower().strip('"') not in _JUNK_LINES
            ]
            bullet_lines = _dedup_bullets(bullet_lines)
            bullet_lines = [
                re.sub(r'^\s*\+\s+', '  - ', line) if line.strip().startswith('*') else line
                for line in bullet_lines
            ]
            print(f"[DEBUG] bullet_lines[0]: {repr(bullet_lines[0])}")
            result = "\n".join(bullet_lines)
            result = re.sub(r'\*\*\[([^\]]+)\]\(https?://[^\)]+\)\*\*', r'**\1**', result)
            result = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', result)
            self.add_to_memory(user_input, result)
            return result

        if tool in ("stocks", "crypto"):
            result = context if context and len(context.strip()) > 20 else (raw_result if isinstance(raw_result, str) else context)
            self.add_to_memory(user_input, result)
            return result

        # ── Internet search ────────────────────────────────────────────────
        if tool == "internet_search":
            print(f"[DEBUG] Source: internet_search tool")
            top_k = 10 if intent == "ownership" else SEARCH_RESULTS_LIMIT
            if intent in ("facts", "knowledge"):
                _tl = self._get_timelimit(user_input, intent)
                expanded = self._expand_query(user_input)           # expand once
                year_query = expanded + f" {time.strftime('%Y')}"
                raw_snippets = self.internet_search_tool(user_input, top_k=top_k, timelimit=_tl)
                raw_snippets += self.internet_search_tool(expanded, top_k=top_k, timelimit=_tl)
                raw_snippets += self.internet_search_tool(year_query, top_k=top_k, timelimit=_tl)
                # ── ADD: biography/person queries need extra detail searches ──
                _is_person_query = bool(re.search(
                    r'^\s*(who\s+is|who\s+was|tell\s+me\s+about|biography\s+of|'
                    r'about\s+[A-Z])',
                    user_input, re.IGNORECASE
                ))
                if _is_person_query:
                    # Extract the person's name and search for specific facts
                    _name_match = re.search(
                        r'(?:who\s+is|who\s+was|about|biography\s+of)\s+(.+?)(?:\?|$)',
                        user_input, re.IGNORECASE
                    )
                    if _name_match:
                        _name = _name_match.group(1).strip()
                        raw_snippets += self.internet_search_tool(
                            f"{_name} biography net worth achievements",
                            top_k=top_k, timelimit=_tl
                        )
                        raw_snippets += self.internet_search_tool(
                            f"{_name} career companies founded",
                            top_k=top_k, timelimit=_tl
                        )
            elif intent == "scientific":
                _tl = self._get_timelimit(user_input, intent)
                raw_snippets = self.internet_search_tool(user_input, top_k=top_k, timelimit=_tl)
                raw_snippets += self.internet_search_tool(
                    self._expand_query(user_input), top_k=top_k, timelimit=_tl
                )
                raw_snippets += self.internet_search_tool(
                    self._expand_query(user_input) + f" {time.strftime('%Y')}", top_k=top_k
                )
            else:
                raw_snippets = self.internet_search_tool(
                    self._expand_query(user_input), top_k=top_k
                )

            raw_snippets = list(dict.fromkeys(raw_snippets))
            blocks = self.format_tool_output(
                tool_output=raw_snippets,
                tool_name="internet_search",
            )
            context = self.build_context(
                blocks=blocks,
                original_query="" if tool == "news" else user_input,
                preprocessed_query=preprocessed_query,
            )
            if not context or len(context.strip()) < 50:
                context = "No reliable information found for this query."
            print(f"[DEBUG] internet_search context: {len(context.split())} tokens "
                  f"from {len(blocks)} blocks after reranking")
            result = self._llm_call(
                user_input=user_input,
                context=context,
                intent=intent,
                length_instruction=length_instruction,
                num_points=num_points,
                context_label="Context from internet search",
                fmt=fmt,
            )
            self.add_to_memory(user_input, result)
            gc.collect()
            return result

        # ── KB + LLM fallback ──────────────────────────────────────────────
        kb_confidence = self.get_kb_confidence(user_input)
        print(f"[DEBUG] KB Confidence: {kb_confidence:.3f} (threshold: {SIMILARITY_THRESHOLD})")

        user_input_expanded = self._expand_query(user_input)

        _KB_IRRELEVANT_TOPICS = re.compile(
            r'\b(cloud|aws|azure|gcp|machine learning|artificial intelligence|'
            r'blockchain|cryptocurrency|nft|metaverse|quantum computing|'
            r'cybersecurity|data science|neural network|deep learning|'
            r'internet of things|iot|big data|devops|kubernetes|microservices|'
            r'saas|paas|iaas|virtualization|containerization|serverless)\b',
            re.IGNORECASE
        )

        def _kb_doc_is_relevant(query: str, top_docs: list) -> bool:
            """Return True only if the top KB doc shares meaningful topic words with the query."""
            if not top_docs:
                return False
            query_words = set(
                w.lower() for w in re.findall(r'\b[a-z]{4,}\b', query.lower())
                if w not in {'what', 'does', 'have', 'this', 'that', 'with', 'from',
                            'about', 'tell', 'give', 'show', 'explain', 'describe',
                            'make', 'does', 'will', 'would', 'could', 'should',
                            'how', 'why', 'when', 'where', 'which', 'their', 'there'}
            )
            top_doc_words = set(re.findall(r'\b[a-z]{4,}\b', top_docs[0].lower()))
            overlap = query_words & top_doc_words
            overlap_ratio = len(overlap) / max(len(query_words), 1)
            print(f"[DEBUG] KB relevance check: overlap={overlap}, ratio={overlap_ratio:.2f}")
            return overlap_ratio >= 0.25  # at least 25% of query's content words appear in top doc

        kb_docs_candidate = self.hybrid_kb_search(user_input_expanded, k=RETRIEVAL_TOP_K * 2)
        _kb_relevant = _kb_doc_is_relevant(user_input, kb_docs_candidate)

        # Also block KB for topics that are clearly internet-domain knowledge
        _is_internet_topic = bool(_KB_IRRELEVANT_TOPICS.search(user_input))

        if kb_confidence >= SIMILARITY_THRESHOLD and _kb_relevant and not _is_internet_topic:
            print(f"[DEBUG] Source: KB (confidence={kb_confidence:.3f}, relevant=True)")
            kb_docs = kb_docs_candidate
            print(f"[DEBUG] Top doc: {kb_docs[0][:80] if kb_docs else 'empty'}")
            context = self.build_kb_context(
                kb_docs=kb_docs,
                original_query=user_input,
            )
        else:
            print(f"[DEBUG] Source: Internet search (KB fallback)")
            top_k = 10 if intent == "ownership" else SEARCH_RESULTS_LIMIT

            raw_snippets = self._parallel_retrieve(
                sub_queries, "internet_search", top_k=top_k
            )
            blocks = self.format_tool_output(
                tool_output=raw_snippets,
                tool_name="internet_search",
            )
            context = self.build_context(
                blocks=blocks,
                original_query="" if tool == "news" else user_input,
                preprocessed_query=preprocessed_query,
            )

        if not context or len(context.strip()) < 50:
            context = "No reliable information found for this query."

        result = self._llm_call(
            user_input=user_input,
            context=context,
            intent=intent,
            length_instruction=length_instruction,
            num_points=num_points,
            context_label="Context",
            fmt=fmt,
        )
        self.add_to_memory(user_input, result)
        gc.collect()
        return result


# =============================================================================
# MODULE-LEVEL HELPER
# =============================================================================

_SCORE_KEYWORDS = re.compile(
    r'\b(score|scorecard|latest score|latest result|match result|'
    r'final score|live score|today score|today match|who won|'
    r'last match|last game|recent match|yesterday.{0,10}match|'
    r'playing now|result of)\b'
    r'|\bscore\b(?!d)',
    re.IGNORECASE
)

_NEWS_OVERRIDE = re.compile(
    r'\b(breaking news|latest news|news about|current events|'
    r'what happened|conflict|war|election update|geopolit)\b',
    re.IGNORECASE
)

def _fix_sports_intent(query: str, raw_intent: str) -> str:
    if raw_intent in ("live_sports", "upcoming_sports", "facts", "ownership",
                      "knowledge", "sports_knowledge", "weather", "scientific"):
        return raw_intent

    has_score   = bool(_SCORE_KEYWORDS.search(query))
    has_news    = bool(_NEWS_OVERRIDE.search(query))

    if has_score and not has_news:
        print(f"[DEBUG] _fix_sports_intent: '{raw_intent}' → 'live_sports' (score keyword matched)")
        return "live_sports"

    return raw_intent


if __name__ == "__main__":
    retriever = HybridRetriever()

    print("\n--- Test 5: Normal KB query ---")
    response = retriever.query_agent("How do I initialize a Git repository?")
    print("\nResponse:\n", response)

    print("\n--- Memory Summary ---")
    print(retriever.get_memory_summary())
    