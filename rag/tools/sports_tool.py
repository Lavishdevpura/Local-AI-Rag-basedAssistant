"""
Sports Live Data Tool — RAG-ready Python module
================================================
APIs used (ALL FREE, NO RapidAPI):

  1. Bzzoiro Sports Data  https://sports.bzzoiro.com
  2. football-data.org   https://www.football-data.org
  3. API-Football (direct) https://dashboard.api-football.com
  4. CricAPI             https://cricketdata.org
  5. TheSportsDB          https://www.thesportsdb.com
  6. OpenLigaDB           https://www.openligadb.de
  7. Sportmonks Cricket   https://sportmonks.com/cricket-api  ← NEW

.env file:
  APIFOOTBALL_COM_KEY=your_key       # apifootball.com (1019 leagues, current season)
  CRICAPI_KEY=your_key
  TSDB_API_KEY=your_patreon_key   # optional
  SPORTMONKS_KEY=your_token       ← NEW
"""

import os
import datetime
import requests
from pathlib import Path
from typing import Optional
import re as _re_sports

# ══════════════════════════════════════════════
# Load .env automatically
# ══════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    def _load_env():
        here = Path(__file__).resolve().parent
        for candidate in [here, *here.parents]:
            env_file = candidate / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)
                print(f"  [.env] Loaded → {env_file}")
                return
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env, override=False)
            print(f"  [.env] Loaded → {cwd_env}")
    _load_env()
except ImportError:
    print("  [warn] python-dotenv not installed → pip install python-dotenv")


# ══════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════

def _resolve_date(date_str: str) -> str:
    s = date_str.strip().lower()
    today = datetime.date.today()
    if s in ("today", ""):
        return today.isoformat()
    if s == "yesterday":
        return (today - datetime.timedelta(days=1)).isoformat()
    try:
        datetime.date.fromisoformat(s)
        return s
    except ValueError:
        raise ValueError(f"Bad date '{date_str}'. Use 'today', 'yesterday', or 'YYYY-MM-DD'.")


def _safe_get(url: str, params: dict = None,
              headers: dict = None, timeout: int = 12) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            print(f"  [warn] 429 Too Many Requests: {url[:60]}")
            # Mark TSDB as rate-limited if it's a TSDB URL
            if "thesportsdb" in url:
                _tsdb_mark_rate_limited()
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] {e}")
        return None


def _key(env_var: str) -> str:
    return os.environ.get(env_var, "").strip()


# ══════════════════════════════════════════════
# Cricket detection
# ══════════════════════════════════════════════

CRICKET_KEYWORDS = {
    "cricket", "ipl", "odi", "t20", "t20i", "test match", "test cricket",
    "one day", "twenty20", "bbl", "big bash", "psl", "cpl",
    "caribbean premier", "the hundred", "sa20", "ilt20", "wpl",
    "asia cup", "icc", "ranji", "champions trophy", "wtc",
    "world test championship", "scorecard", "innings", "wicket",
    "batting", "bowling", "run chase",
}
# Keywords that require word-boundary matching (to avoid substring false positives)
# e.g. "test" inside "galatasaray latest" should NOT trigger cricket
_CRICKET_KEYWORDS_EXACT = {"test match", "test cricket", "one day", "twenty20",
                            "big bash", "caribbean premier", "the hundred",
                            "world test championship", "run chase"}
_CRICKET_KEYWORDS_WORD  = {"odi", "t20", "t20i", "ipl", "bbl", "psl", "cpl",
                            "sa20", "ilt20", "wpl", "icc", "ranji", "wtc",
                            "innings", "wicket", "batting", "bowling",
                            "scorecard"}

CRICKET_NATIONS = {
    # Full members
    "england", "india", "australia", "pakistan",
    "new zealand", "south africa", "west indies",
    "sri lanka", "bangladesh", "afghanistan", "zimbabwe",
    "ireland", "kenya",
    # Associate / common in queries
    "netherlands", "scotland", "nepal", "oman", "uae",
    "namibia", "canada", "usa", "uganda", "png",
    "papua new guinea",
}

# Cricket franchise teams → their tournament (for dynamic routing)
CRICKET_FRANCHISE_MAP = {
    # IPL
    "mumbai indians": "IPL",       "mi": "IPL",
    "chennai super kings": "IPL",  "csk": "IPL",
    "royal challengers": "IPL",    "rcb": "IPL",
    "kolkata knight riders": "IPL","kkr": "IPL",
    "delhi capitals": "IPL",       "dc": "IPL",
    "punjab kings": "IPL",         "pbks": "IPL",
    "rajasthan royals": "IPL",     "rr": "IPL",
    "sunrisers hyderabad": "IPL",  "srh": "IPL",
    "lucknow super giants": "IPL", "lsg": "IPL",
    "gujarat titans": "IPL",       "gt": "IPL",
    # PSL
    "karachi kings": "PSL",        "lahore qalandars": "PSL",
    "peshawar zalmi": "PSL",       "quetta gladiators": "PSL",
    "islamabad united": "PSL",     "multan sultans": "PSL",
    # BBL
    "sydney sixers": "BBL",        "sydney thunder": "BBL",
    "melbourne stars": "BBL",      "melbourne renegades": "BBL",
    "brisbane heat": "BBL",        "perth scorchers": "BBL",
    "hobart hurricanes": "BBL",    "adelaide strikers": "BBL",
    # CPL
    "trinbago knight riders": "CPL", "barbados royals": "CPL",
    "guyana amazon warriors": "CPL", "jamaica tallawahs": "CPL",
    "st kitts and nevis patriots": "CPL", "st lucia kings": "CPL",
    # SA20
    "mi cape town": "SA20",        "sunrisers eastern cape": "SA20",
    "pretoria capitals": "SA20",   "paarl royals": "SA20",
    "joburg super kings": "SA20",  "durban super giants": "SA20",
    # The Hundred
    "oval invincibles": "The Hundred",  "london spirit": "The Hundred",
    "trent rockets": "The Hundred",     "manchester originals": "The Hundred",
    "southern brave": "The Hundred",    "welsh fire": "The Hundred",
    "northern superchargers": "The Hundred", "birmingham phoenix": "The Hundred",
}

SCORE_WORDS = {
    "score", "result", "match", "won", "win", "lost",
    "playing", "latest", "live", "series", "test", "odi", "t20",
}

FOOTBALL_OVERRIDE = {
    "football", "soccer", "premier league", "la liga",
    "bundesliga", "serie a", "ligue 1", "champions league",
    "epl", "fifa", "goal", "penalty", "freekick",
}

def _is_cricket(query: str) -> bool:
    import re as _re_c
    q = query.lower()
    # Always check "cricket" first — unambiguous
    if 'cricket' in q:
        return True
    if any(kw in q for kw in FOOTBALL_OVERRIDE):
        return False
    # ALL keywords require word-boundary matching to avoid false positives:
    # "test match" inside "galatasaray latest match" → "test" must start at word boundary
    # "t20" inside some team name → must be standalone
    for kw in CRICKET_KEYWORDS:
        # Build word-boundary pattern for the whole phrase
        pattern = r'\b' + r'\s+'.join(_re_c.escape(w) for w in kw.split()) + r'\b'
        if _re_c.search(pattern, q):
            return True
    if any(kw in q for kw in FOOTBALL_OVERRIDE):
        return False
    # Franchise teams are always cricket — use word boundaries to avoid
    # matching short abbreviations like "mi" inside "inter milan"
    import re as _re_cric
    for franchise in CRICKET_FRANCHISE_MAP:
        if len(franchise) <= 3:
            # Short abbreviations: require word boundary
            if _re_cric.search(r'\b' + _re_cric.escape(franchise) + r'\b', q):
                return True
        else:
            if franchise in q:
                return True
    has_nation = any(nation in q for nation in CRICKET_NATIONS)
    has_score  = any(word in q for word in SCORE_WORDS)
    return has_nation and has_score


# ══════════════════════════════════════════════
# Status helpers — used for prefer_completed sorting
# ══════════════════════════════════════════════

_STATUS_PRIORITY = {
    "full time": 0,
    "match ended": 0,
    "finished": 0,
    "complete": 0,
    "closed": 0,
    "full time (aet)": 0,
    "full time (penalties)": 0,
    "live": 1,
    "1st half": 1,
    "2nd half": 1,
    "half time": 1,
    "extra time": 1,
    "in progress": 1,
    "inprogress": 1,
    "scheduled": 2,
    "not started": 2,
    "to be decided": 2,
    "postponed": 3,
    "cancelled": 3,
    "suspended": 3,
    "interrupted": 3,
}

def _status_rank(status: str) -> int:
    return _STATUS_PRIORITY.get(status.strip().lower(), 2)

def _sort_key_from_str(match_str: str) -> int:
    for line in match_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Status:") or stripped.startswith("   Status:"):
            status_val = stripped.split(":", 1)[1].strip()
            return _status_rank(status_val)
    return 2


# ══════════════════════════════════════════════
# Output formatter
# ══════════════════════════════════════════════

def _winner(t1, t2, s1, s2, status: str) -> str:
    try:
        h = int(str(s1).split("/")[0])
        a = int(str(s2).split("/")[0])
        done = any(w in status.lower() for w in
                   ("won","full time","finished","complete","final","closed","ft","ended","result"))
        if not done:
            return ""
        if h > a:   return f"   🏅 Winner: {t1}"
        if a > h:   return f"   🏅 Winner: {t2}"
        return "   🏅 Result: Draw / Tie"
    except Exception:
        return ""


def _fmt(t1, t2, s1, s2, status, league, season, venue, dt,
         extra: list = None, emoji="🏆") -> str:
    lines = [
        "Most Recent Match:",
        f"{emoji} {t1} vs {t2}",
        f"   {t1} {s1} – {s2} {t2}",
        f"   Status: {status}",
        f"   League/Series: {league}",
        f"   Season: {season}",
        f"   Venue: {venue}",
        f"   Date: {dt}",
    ]
    if extra:
        lines.extend(extra)
    w = _winner(t1, t2, s1, s2, status)
    if w:
        lines.append(w)
    return "\n".join(lines)


def _fmt_prediction(home, away, league, date,
                    prob_home, prob_draw, prob_away,
                    predicted, confidence,
                    prob_over25=None, prob_btts=None,
                    source="") -> str:
    result_map = {"H": f"{home} Win", "D": "Draw", "A": f"{away} Win"}
    predicted_str = result_map.get(predicted, predicted)
    lines = [
        "⚽ Prediction:",
        f"🔮 {home} vs {away}",
        f"   League: {league}",
        f"   Date: {date}",
        f"   🏠 {home} Win: {prob_home}%",
        f"   🤝 Draw:        {prob_draw}%",
        f"   ✈️  {away} Win: {prob_away}%",
        f"   🎯 Predicted:   {predicted_str}",
        f"   📊 Confidence:  {confidence}%",
    ]
    if prob_over25 is not None:
        lines.append(f"   ⚡ Over 2.5 goals: {prob_over25}%")
    if prob_btts is not None:
        lines.append(f"   ⚽ Both Teams Score: {prob_btts}%")
    if source:
        lines.append(f"   🔬 Model: {source}")
    return "\n".join(lines)


SEP = "\n\n" + ("─" * 45) + "\n\n"

_STATUS_MAP = {
    "notstarted": "Not Started", "ns": "Not Started",
    "tbd": "To Be Decided",
    "ft": "Full Time", "finished": "Full Time",
    "complete": "Full Time", "closed": "Full Time",
    "aet": "Full Time (AET)", "pen": "Full Time (Penalties)",
    "ht": "Half Time", "1h": "1st Half", "2h": "2nd Half",
    "et": "Extra Time", "live": "Live",
    "inprogress": "Live", "in_progress": "Live",
    "pst": "Postponed", "postponed": "Postponed",
    "susp": "Suspended", "int": "Interrupted",
    "canc": "Cancelled", "cancelled": "Cancelled",
    "scheduled": "Scheduled", "matchended": "Match Ended",
}

def _clean_status(raw: str) -> str:
    if not raw:
        return "Scheduled"
    key = raw.lower().replace(" ", "").replace("_", "")
    return _STATUS_MAP.get(key, raw.replace("_", " ").title())

# ══════════════════════════════════════════════
# API 1 — ESPN (no key required)
# ══════════════════════════════════════════════

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_HDR  = {"User-Agent": "Mozilla/5.0"}

# slug → display name  (17 working leagues confirmed)
ESPN_SLUG_MAP = {
    "eng.1":            "Premier League",
    "ita.1":            "Serie A",
    "esp.1":            "La Liga",
    "ger.1":            "Bundesliga",
    "fra.1":            "Ligue 1",
    "ned.1":            "Eredivisie",
    "por.1":            "Primeira Liga",
    "tur.1":            "Süper Lig",
    "sco.1":            "Scottish Premiership",
    "eng.2":            "Championship",
    "esp.2":            "Segunda División",
    "arg.1":            "Liga Profesional",
    "bra.1":            "Brasileirão",
    "mex.1":            "Liga MX",
    "usa.1":            "MLS",
    "uefa.champions":   "UEFA Champions League",
    "uefa.europa":      "UEFA Europa League",
    "uefa.europa.conf": "UEFA Conference League",
    "eng.fa":           "FA Cup",
    "eng.league_cup":   "League Cup",
    # International / national team competitions
    "fifa.world":               "FIFA World Cup",
    "fifa.worldq.conmebol":     "CONMEBOL WC Qualifiers",
    "fifa.worldq.uefa":         "UEFA WC Qualifiers",
    "fifa.worldq.concacaf":     "CONCACAF WC Qualifiers",
    "fifa.worldq.afc":          "AFC WC Qualifiers",
    "fifa.worldq.caf":          "CAF WC Qualifiers",
    "concacaf.nations.league":  "CONCACAF Nations League",
    "uefa.nations":             "UEFA Nations League",
    "conmebol.america":         "Copa America",
    "afc.cup":                  "AFC Asian Cup",
    "caf.nations":              "Africa Cup of Nations",
    "concacaf.gold":            "CONCACAF Gold Cup",
    "fifa.friendly":            "International Friendly",
    "uefa.euro":                "UEFA European Championship",
}

# keyword → slugs to search (own league first + European comps)
# ESPN slugs for international/national team competitions
ESPN_INTL_SLUGS = [
    "fifa.worldq.conmebol",   # CONMEBOL World Cup Qualifiers (S America)
    "fifa.worldq.uefa",       # UEFA World Cup Qualifiers (Europe)
    "fifa.worldq.concacaf",   # CONCACAF World Cup Qualifiers
    "fifa.worldq.afc",        # AFC World Cup Qualifiers (Asia)
    "fifa.worldq.caf",        # CAF World Cup Qualifiers (Africa)
    "fifa.worldq.ofc",        # OFC Qualifiers
    "concacaf.nations.league",# CONCACAF Nations League
    "uefa.nations",           # UEFA Nations League
    "conmebol.america",       # Copa America
    "afc.cup",                # AFC Asian Cup
    "caf.nations",            # AFCON
    "concacaf.gold",          # Gold Cup
    "fifa.friendly",          # International Friendlies
    "fifa.world",             # FIFA World Cup
    "uefa.euro",              # UEFA Euros
]

ESPN_HINT_SLUGS = {
    "serie a":        ["ita.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"],
    "italy":          ["ita.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"] + ESPN_INTL_SLUGS,
    "premier league": ["eng.1", "eng.fa", "eng.league_cup", "uefa.champions", "uefa.europa"],
    "england":        ["eng.1", "eng.fa", "eng.league_cup", "uefa.champions", "uefa.europa"] + ESPN_INTL_SLUGS,
    "bundesliga":     ["ger.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"],
    "germany":        ["ger.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"] + ESPN_INTL_SLUGS,
    "la liga":        ["esp.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"],
    "spain":          ["esp.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"] + ESPN_INTL_SLUGS,
    "ligue 1":        ["fra.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"],
    "france":         ["fra.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"] + ESPN_INTL_SLUGS,
    "eredivisie":     ["ned.1", "uefa.europa.conf"],
    "netherlands":    ["ned.1", "uefa.europa.conf"] + ESPN_INTL_SLUGS,
    "champions league": ["uefa.champions"],
    "ucl":            ["uefa.champions"],
    "europa league":  ["uefa.europa"],
    "conference":     ["uefa.europa.conf"],
    "mls":            ["usa.1"],
    "american major league soccer": ["usa.1"],
    "major league soccer": ["usa.1"],
    "scotland":       ["sco.1"] + ESPN_INTL_SLUGS,
    "turkey":         ["tur.1"] + ESPN_INTL_SLUGS,
    "portugal":       ["por.1"] + ESPN_INTL_SLUGS,
    "championship":   ["eng.2"],
    "fa cup":         ["eng.fa"],
    # National team only — pure international slugs
    "brazil":         ESPN_INTL_SLUGS + ["bra.1"],
    "argentina":      ESPN_INTL_SLUGS + ["arg.1"],
    "mexico":         ESPN_INTL_SLUGS + ["mex.1"],
    "international":  ESPN_INTL_SLUGS,
    "world cup":      ["fifa.world", "fifa.worldq.conmebol", "fifa.worldq.uefa",
                       "fifa.worldq.concacaf", "fifa.worldq.afc", "fifa.worldq.caf"],
    "euros":          ["uefa.euro"],  "euro": ["uefa.euro"],
    "nations league": ["uefa.nations", "concacaf.nations.league"],
    "copa america":   ["conmebol.america"],
    "asian cup":      ["afc.cup"],
    "afcon":          ["caf.nations"],
    "gold cup":       ["concacaf.gold"],
    "friendly":       ["fifa.friendly"],
}

DEFAULT_ESPN_SLUGS = ["eng.1", "ita.1", "esp.1", "ger.1", "fra.1",
                      "uefa.champions", "uefa.europa", "ned.1", "por.1",
                      "usa.1", "sco.1", "tur.1", "bra.1", "arg.1", "mex.1"]


def _espn_parse_score(val):
    """ESPN score can be plain string or a $ref dict — extract displayValue."""
    if isinstance(val, dict):
        return val.get("displayValue", "?")
    return str(val) if val is not None else "?"


def _espn_get_matches(query: str, date: str) -> list:
    """Fetch football matches from ESPN for a given date."""
    if _is_cricket(query):
        return []

    q  = query.lower()
    # Find matching slugs
    slugs = next((v for k, v in ESPN_HINT_SLUGS.items() if k in q), None)
    if not slugs:
        # Try to match a single known slug name
        slugs = [s for s, name in ESPN_SLUG_MAP.items() if name.lower() in q or s in q]
    if not slugs:
        slugs = DEFAULT_ESPN_SLUGS

    date_param = date.replace("-", "")  # ESPN uses YYYYMMDD format
    results = []

    for slug in slugs:
        data = _safe_get(f"{ESPN_BASE}/{slug}/scoreboard",
                         params={"dates": date_param},
                         headers=ESPN_HDR, timeout=12)
        events = (data or {}).get("events") or []
        for e in events:
            comp   = (e.get("competitions") or [{}])[0]
            teams  = comp.get("competitors") or []
            if len(teams) < 2:
                continue
            home   = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
            away   = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
            ht     = home.get("team", {}).get("displayName", "?")
            at     = away.get("team", {}).get("displayName", "?")
            hs     = _espn_parse_score(home.get("score"))
            as_    = _espn_parse_score(away.get("score"))
            status = comp.get("status", {}).get("type", {}).get("description", "Scheduled")
            lg     = ESPN_SLUG_MAP.get(slug, slug)
            venue  = (comp.get("venue") or {}).get("fullName", "N/A")
            dt     = (e.get("date") or date)[:16].replace("T", "  ")
            results.append(_fmt(ht, at, hs, as_, status, lg, "", venue, dt))

    return results


def _espn_get_last_match(team_name: str, hint_league: str = "", target_date: str = "") -> list:
    """
    Fetch most recent completed match for a team across all competitions via ESPN.
    Searches league-specific team schedules — no key required.
    """
    import datetime as _dt

    tf = team_name.lower().strip()
    hl = hint_league.lower()

    # National team detection
    _is_natl = _TEAM_LEAGUE_MAP.get(tf) == "International" or "international" in hl
    if _is_natl:
        # For national teams use international slugs — no club league slugs
        slugs = ESPN_INTL_SLUGS
    else:
        # Pick slugs to search based on hint
        slugs = next((v for k, v in ESPN_HINT_SLUGS.items() if k in hl), None) or DEFAULT_ESPN_SLUGS

    # Team name aliases for ESPN display names
    ESPN_ALIASES = {
        "inter milan":     ["internazionale", "inter"],
        "ac milan":        ["ac milan", "milan"],
        "man city":        ["manchester city"],
        "man united":      ["manchester united"],
        "man utd":         ["manchester united"],
        "spurs":           ["tottenham hotspur", "tottenham"],
        "tottenham":       ["tottenham hotspur"],
        "psg":             ["paris saint-germain", "paris sg"],
        "barca":           ["barcelona", "fc barcelona"],
        "atletico madrid": ["atletico de madrid", "atletico madrid"],
        "atletico":        ["atletico"],
        "dortmund":        ["borussia dortmund"],
        "bvb":             ["borussia dortmund"],
        "leverkusen":      ["bayer leverkusen"],
        "gladbach":        ["borussia monchengladbach"],
        "bayern munich":   ["fc bayern munich", "bayern munich"],
        "wolves":          ["wolverhampton wanderers"],
        "inter miami":     ["inter miami cf", "inter miami"],
        "new york city":   ["new york city fc", "nyc fc"],
        "new york rb":     ["new york red bulls"],
        "la galaxy":       ["la galaxy"],
        "portland":        ["portland timbers"],
        "seattle":         ["seattle sounders"],
        "atlanta":         ["atlanta united"],
        "toronto":         ["toronto fc"],
        "cf montreal":     ["cf montréal", "cf montreal"],
    }
    search_names = ESPN_ALIASES.get(tf, [tf])

    # Slugs whose /teams endpoint returns empty — must scan scoreboard by date
    _ESPN_NO_TEAMS_ENDPOINT = {"uefa.champions", "uefa.europa", "uefa.europa.conf",
                                "eng.fa", "eng.league_cup",
                                # All international competition slugs
                                "fifa.world", "fifa.worldq.conmebol", "fifa.worldq.uefa",
                                "fifa.worldq.concacaf", "fifa.worldq.afc", "fifa.worldq.caf",
                                "fifa.worldq.ofc", "concacaf.nations.league", "uefa.nations",
                                "conmebol.america", "afc.cup", "caf.nations",
                                "concacaf.gold", "fifa.friendly", "uefa.euro"}

    all_completed = []

    def _espn_team_in_event(e, snames):
        """Check if any search name matches a competitor in this event."""
        comp  = (e.get("competitions") or [{}])[0]
        teams = comp.get("competitors") or []
        for t in teams:
            name = (t.get("team", {}).get("displayName") or "").lower()
            abbr = (t.get("team", {}).get("abbreviation") or "").lower()
            for sn in snames:
                words = sn.split()
                if (sn == name or sn == abbr or name.startswith(sn)
                        or (len(words) > 1 and all(w in name for w in words))):
                    return True
        return False

    def _espn_event_to_block(e, slug):
        """Convert ESPN event to a match block string."""
        comp   = (e.get("competitions") or [{}])[0]
        teams  = comp.get("competitors") or []
        if len(teams) < 2:
            return None
        home   = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
        away   = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
        ht     = home.get("team", {}).get("displayName", "?")
        at     = away.get("team", {}).get("displayName", "?")
        hs     = _espn_parse_score(home.get("score"))
        as_    = _espn_parse_score(away.get("score"))
        lg     = ESPN_SLUG_MAP.get(slug, slug)
        venue  = (comp.get("venue") or {}).get("fullName", "N/A")
        dt_raw = e.get("date", "")
        dt     = dt_raw[:16].replace("T", "  ")
        return {"block": _fmt(ht, at, hs, as_, "Full Time", lg, "", venue, dt),
                "date":  dt_raw}

    for slug in slugs:
        if slug in _ESPN_NO_TEAMS_ENDPOINT:
            # Scan scoreboard by date for last 90 days — no /teams endpoint available
            print(f"  [espn_last] Scanning /{slug} scoreboard for '{team_name}'")
            for days_back in range(0, 90, 1):
                d = (_dt.date.today() - _dt.timedelta(days=days_back)).strftime("%Y%m%d")
                data = _safe_get(f"{ESPN_BASE}/{slug}/scoreboard",
                                 params={"dates": d}, headers=ESPN_HDR, timeout=12)
                events = (data or {}).get("events") or []
                for e in events:
                    comp = (e.get("competitions") or [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed", False):
                        continue
                    if _espn_team_in_event(e, search_names):
                        block = _espn_event_to_block(e, slug)
                        if block:
                            all_completed.append(block)
                if all_completed:
                    break  # found a match — stop scanning earlier dates
            continue

        # Standard approach: get teams roster → find team_id → get schedule
        teams_data = _safe_get(f"{ESPN_BASE}/{slug}/teams",
                               headers=ESPN_HDR, timeout=12)
        league_teams = ((teams_data or {}).get("sports") or [{}])[0]                       .get("leagues", [{}])[0].get("teams", [])

        team_id = None
        for t in league_teams:
            name = (t.get("team", {}).get("displayName") or "").lower()
            abbr = (t.get("team", {}).get("abbreviation") or "").lower()
            for sn in search_names:
                words = sn.split()
                matched = (
                    sn == name or sn == abbr
                    or name.startswith(sn)
                    or (len(words) > 1 and all(w in name for w in words))
                )
                if matched:
                    team_id = t.get("team", {}).get("id")
                    break
            if team_id:
                break

        if not team_id:
            continue

        print(f"  [espn_last] Found team id={team_id} in /{slug}/teams")

        # Get schedule for this team in this league
        sched = _safe_get(f"{ESPN_BASE}/{slug}/teams/{team_id}/schedule",
                          headers=ESPN_HDR, timeout=12)
        events = (sched or {}).get("events") or []

        for e in events:
            comp = (e.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            # Filter by target date if specified (±1 day tolerance)
            if target_date and target_date not in ("today", "yesterday"):
                e_date = (e.get("date") or "")[:10]
                if e_date != target_date:
                    # Allow ±1 day for timezone differences
                    import datetime as _dtt_e
                    try:
                        _td  = _dtt_e.date.fromisoformat(target_date)
                        _ed  = _dtt_e.date.fromisoformat(e_date)
                        if abs((_ed - _td).days) > 1:
                            continue
                    except ValueError:
                        continue
            block = _espn_event_to_block(e, slug)
            if block:
                all_completed.append(block)

    if not all_completed:
        _msg = f"for '{team_name}'"
        if target_date and target_date not in ("today","yesterday"):
            _msg += f" on {target_date}"
        print(f"  [espn_last] No completed matches found {_msg}")
        return []

    all_completed.sort(key=lambda x: x["date"], reverse=True)
    best = all_completed[0]
    _log_line = best['block'].split('\n')[1] if '\n' in best['block'] else best['block'][:60]
    print(f"  [espn_last] Most recent: {_log_line}")
    return [best["block"]]


# ══════════════════════════════════════════════
# API 2 — apifootball.com  (1,019 leagues, no season block)
# ══════════════════════════════════════════════

APIFOOTBALL_COM_BASE = "https://apiv3.apifootball.com"

# Verified league IDs from apifootball.com
AF_COM_LEAGUE_MAP = {
    "premier league":   152,  "epl": 152,
    "serie a":          207,  "italian": 207,
    "la liga":          302,  "spanish": 302,
    "bundesliga":       175,  "german": 175,
    "ligue 1":          168,  "french": 168,
    "champions league": 3,    "ucl": 3,
    "europa league":    4,    "uel": 4,
    "conference":       683,
    "fa cup":           146,
    "league cup":       147,
    "dfb pokal":        172,
    "copa del rey":     300,
    "coppa italia":     205,
    "coupe de france":  165,
    "mls":              244,
    "brasileirao":      13,
    "liga profesional": 128,
    "mexico":           188,  "liga mx": 188,
    "eredivisie":       61,
    "primeira liga":    182,
    "super lig":        203,
    "scottish":         264,
    "championship":     153,
    # International / national team competitions
    "world cup":        6,    "fifa world cup": 6,
    "euros":            5,    "euro":  5,    "european championship": 5,
    "nations league":   73,   "uefa nations league": 73,
    "copa america":     197,  "conmebol": 197,
    "asian cup":        22,   "afc asian cup": 22,
    "afcon":            36,   "africa cup": 36,  "african cup": 36,
    "gold cup":         23,   "concacaf gold cup": 23,
    "international friendlies": 257, "friendly": 257,
}

# hint → list of league ids to search
# International league IDs for national team competitions
_INTL_IDS = [6, 5, 73, 74, 197, 22, 36, 23, 24, 257]  # WC, Euro, NL, CA, AFC, AFCON, GC, CNL, Friendly

AF_COM_HINT_MAP = {
    "serie a":        [207, 205, 3, 4, 683],
    "italy":          [207, 205, 3, 4, 683] + _INTL_IDS,
    "premier league": [152, 146, 147, 3, 4],
    "england":        [152, 146, 147, 3, 4] + _INTL_IDS,
    "bundesliga":     [175, 172, 3, 4, 683],
    "germany":        [175, 172, 3, 4, 683] + _INTL_IDS,
    "la liga":        [302, 300, 3, 4, 683],
    "spain":          [302, 300, 3, 4, 683] + _INTL_IDS,
    "ligue 1":        [168, 165, 3, 4, 683],
    "france":         [168, 165, 3, 4, 683] + _INTL_IDS,
    "champions league": [3],  "ucl": [3],
    "europa league":  [4],    "conference": [683],
    "mls":            [244],  "american major league soccer": [244],  "major league soccer": [244],
    "brazil":         [13] + _INTL_IDS,
    "argentina":      [128] + _INTL_IDS,
    "mexico":         [188] + _INTL_IDS,
    "netherlands":    [61] + _INTL_IDS,
    "portugal":       [182] + _INTL_IDS,
    "turkey":         [203] + _INTL_IDS,
    "scotland":       [264] + _INTL_IDS,
    # Pure national team searches — only international competitions
    "world cup":      [6],
    "euros":          [5],   "euro": [5],
    "nations league": [73, 74],
    "copa america":   [197],
    "asian cup":      [22],
    "afcon":          [36],
    "international":  _INTL_IDS,
    "friendly":       [257],
}

AF_COM_DEFAULT_IDS = [152, 207, 302, 175, 168, 3, 4, 683, 244, 13, 128, 188, 61, 182, 203, 264]


def _afcom_get(params: dict) -> list:
    key = _key("APIFOOTBALL_COM_KEY")
    if not key:
        return []
    params["APIkey"] = key
    data = _safe_get(APIFOOTBALL_COM_BASE, params=params, timeout=15)
    if isinstance(data, dict) and "error" in data:
        return []
    return data if isinstance(data, list) else []


def _apifootballcom_get_matches(query: str, date: str) -> list:
    """Fetch football matches from apifootball.com for a given date."""
    if _is_cricket(query) or not _key("APIFOOTBALL_COM_KEY"):
        return []

    q         = query.lower()
    league_id = next((v for k, v in AF_COM_LEAGUE_MAP.items() if k in q), None)
    params    = {"action": "get_events", "from": date, "to": date}
    if league_id:
        params["league_id"] = league_id

    events  = _afcom_get(params)
    results = []
    for e in events:
        ht     = e.get("match_hometeam_name", "?")
        at     = e.get("match_awayteam_name", "?")
        hs     = e.get("match_hometeam_score") or "–"
        as_    = e.get("match_awayteam_score") or "–"
        status = e.get("match_status") or "Scheduled"
        lg     = e.get("league_name", query)
        season = e.get("match_season", "")
        venue  = e.get("match_stadium") or "N/A"
        dt     = f"{e.get('match_date','?')}  {e.get('match_time','??:??')}"
        results.append(_fmt(ht, at, hs, as_, status, lg, season, venue, dt))
    return results


# ══════════════════════════════════════════════
# API 4 — CricAPI
# ══════════════════════════════════════════════

CRICAPI_BASE = "https://api.cricapi.com/v1"

# Cricket national teams — these must match exactly, not as substrings
# "India" should not match "India Tigers", "India Captains", "Mumbai Indians"
CRICKET_NATIONAL_TEAMS = {
    "india", "england", "australia", "pakistan", "south africa",
    "new zealand", "west indies", "sri lanka", "bangladesh", "zimbabwe",
    "afghanistan", "ireland", "scotland", "netherlands", "kenya",
    "namibia", "oman", "nepal", "usa", "canada", "uae",
    "united arab emirates", "papua new guinea", "hong kong",
}


def _cricket_team_matches(team_filter: str, team_name: str) -> bool:
    """
    Check if team_filter matches team_name for cricket.
    National teams (India, England etc.) must match exactly or as standalone word,
    NOT as prefix of another team (India Tigers, India Captains, England Lions).
    Franchise/club teams use substring match.
    """
    import re as _re_cm
    tf   = team_filter.lower().strip()
    name = team_name.lower().strip()

    if tf not in CRICKET_NATIONAL_TEAMS:
        # Exact substring match first
        if tf in name:
            return True
        # Franchise abbreviations (csk, rcb, mi, kkr etc.) mapped to full names
        _ABBREV_MAP = {
            "csk": ["chennai super kings", "chennai"],
            "rcb": ["royal challengers", "bangalore", "bengaluru"],
            "mi":  ["mumbai indians", "mumbai"],
            "kkr": ["kolkata knight riders", "kolkata"],
            "srh": ["sunrisers hyderabad", "hyderabad"],
            "rr":  ["rajasthan royals", "rajasthan"],
            "dc":  ["delhi capitals", "delhi"],
            "pbks":["punjab kings", "punjab"],
            "lsg": ["lucknow super giants", "lucknow"],
            "gt":  ["gujarat titans", "gujarat"],
            "bbl": ["big bash"],
            "psl": ["pakistan super league"],
        }
        if tf in _ABBREV_MAP:
            return any(full in name for full in _ABBREV_MAP[tf])
        return False

    # National team exact matching:
    # OK:  "india", "india cricket", "team india"
    # NOT: "india tigers", "india captains", "india a", "india women",
    #      "india legends", "mumbai indians", "south australia"

    # Allowed suffixes after the country name (optional)
    _allowed_suffix = r'(?:\s+(?:cricket|national|team|sr|senior))?'
    # Allowed prefixes before the country name
    _allowed_prefix = r'(?:(?:team|national|senior|sr)\s+)?'

    # Pattern: optional allowed prefix + country + optional allowed suffix + end of string
    _exact = r'^(?:' + _allowed_prefix + r')?' + _re_cm.escape(tf) + _allowed_suffix + r'$'
    if _re_cm.match(_exact, name, _re_cm.IGNORECASE):
        return True

    # Also match if team name contains the country as a standalone token
    # but ONLY if: (a) country appears at start of name, and
    #              (b) next word (if any) is in the allowed suffix set
    _tokens = name.split()
    for i, tok in enumerate(_tokens):
        if tok == tf:
            # Country must NOT be preceded by another country/place word
            # e.g. "South Australia" → "australia" at pos 1, preceded by "south" → no match
            if i > 0:
                prev = _tokens[i-1].lower()
                # If preceded by a directional/descriptor word → it's a regional team, not national
                if prev in {'south', 'north', 'east', 'west', 'central', 'western',
                            'eastern', 'northern', 'southern', 'new', 'old'}:
                    return False
            # Check what follows
            rest = _tokens[i+1:]
            if not rest:
                return True  # "India" alone, or "Team India" after prefix
            next_word = rest[0].lower().rstrip('s')  # strip plural
            if next_word in {'cricket', 'national', 'team', 'sr', 'senior', 'men'}:
                return True
            return False  # Tigers, Captains, A, B, Women, Lions etc. → no match
    return False


# In-memory cache for CricAPI series scan (1 hour TTL)
_CRICAPI_CACHE: dict = {}
_CRICAPI_CACHE_TTL = 3600

# Shared currentMatches cache (5 min TTL) — prevents 3x calls per query
_CURRENT_MATCHES_CACHE: dict = {}
_CURRENT_MATCHES_TTL   = 300

# Daily limit tracking — once exhausted, skip ALL CricAPI calls until midnight UTC
_CRICAPI_DAILY_EXHAUSTED = False   # set True when daily quota hit
_CRICAPI_EXHAUSTED_DATE  = ""      # date string "YYYY-MM-DD" when exhausted


def _cricapi_quota_ok() -> bool:
    """Return True if CricAPI daily quota is still available."""
    import datetime as _dq
    global _CRICAPI_DAILY_EXHAUSTED, _CRICAPI_EXHAUSTED_DATE
    today = _dq.date.today().isoformat()
    if _CRICAPI_DAILY_EXHAUSTED and _CRICAPI_EXHAUSTED_DATE == today:
        print("  [cricapi] Daily quota exhausted — skipping all CricAPI calls today")
        return False
    # New day — reset
    if _CRICAPI_EXHAUSTED_DATE != today:
        _CRICAPI_DAILY_EXHAUSTED  = False
        _CRICAPI_EXHAUSTED_DATE   = today
    return True


def _cricapi_mark_exhausted(reason: str = ""):
    """Mark CricAPI as exhausted for today."""
    import datetime as _dq
    global _CRICAPI_DAILY_EXHAUSTED, _CRICAPI_EXHAUSTED_DATE
    _CRICAPI_DAILY_EXHAUSTED = True
    _CRICAPI_EXHAUSTED_DATE  = _dq.date.today().isoformat()
    print(f"  [cricapi] Marked as daily-exhausted: {reason}")


def _get_current_matches(key: str) -> tuple:
    """
    Fetch currentMatches once, cache for 5 minutes.
    Returns (matches_list, api_is_alive).
    api_is_alive=True means API responded successfully (even if 0 matches).
    """
    import time as _t
    if not _cricapi_quota_ok():
        return [], False
    ts = _CURRENT_MATCHES_CACHE.get("ts", 0)
    if _CURRENT_MATCHES_CACHE.get("status") is not None and (_t.time() - ts) < _CURRENT_MATCHES_TTL:
        cached   = _CURRENT_MATCHES_CACHE.get("data", [])
        is_alive = _CURRENT_MATCHES_CACHE.get("status") == "success"
        print(f"  [cricapi] Using cached currentMatches ({len(cached)} matches)")
        return cached, is_alive
    data    = _safe_get(f"{CRICAPI_BASE}/currentMatches",
                        params={"apikey": key, "offset": 0}, timeout=25)
    status  = (data or {}).get("status", "")
    reason  = (data or {}).get("reason", "")
    matches = (data or {}).get("data") or []
    # Track daily exhaustion
    if status == "failure":
        _lower_reason = reason.lower()
        if any(x in _lower_reason for x in ["exceeded", "limit", "quota", "daily"]):
            _cricapi_mark_exhausted(reason)
        elif "blocked" in _lower_reason:
            print(f"  [cricapi] currentMatches blocked (rate limit): {reason}")
        else:
            print(f"  [cricapi] currentMatches failed: {reason}")
    # Track hits used — warn when approaching limit
    info = (data or {}).get("info") or {}
    hits_used  = info.get("hitsUsed", 0) or 0
    hits_limit = info.get("hitsLimit", 100) or 100
    if hits_used and hits_limit and hits_used >= hits_limit - 5:
        print(f"  [cricapi] WARNING: {hits_used}/{hits_limit} hits used today — nearly exhausted")
    _CURRENT_MATCHES_CACHE["status"] = status
    _CURRENT_MATCHES_CACHE["ts"]     = _t.time()
    if status == "success":
        _CURRENT_MATCHES_CACHE["data"] = matches
        print(f"  [cricapi] Fetched {len(matches)} currentMatches ({hits_used}/{hits_limit} hits today)")
        return matches, True
    return [], False

CRICAPI_SERIES_MAP = {
    "ipl": "indian premier league", "bbl": "big bash",
    "psl": "pakistan super league", "cpl": "caribbean premier",
    "sa20": "sa20", "ilt20": "ilt20", "the hundred": "the hundred",
    "wpl": "women premier", "odi": "odi", "t20i": "t20i",
    "t20": "t20", "test": "test", "champions trophy": "champions trophy",
    "asia cup": "asia cup", "world cup": "world cup",
}

COUNTRY_TEAMS = {
    "india": ["india", "ind"], "australia": ["australia", "aus"],
    "england": ["england", "eng"], "pakistan": ["pakistan", "pak"],
    "new zealand": ["new zealand", "nz"], "south africa": ["south africa", "sa"],
    "west indies": ["west indies", "wi"], "sri lanka": ["sri lanka", "sl"],
    "bangladesh": ["bangladesh", "ban"], "afghanistan": ["afghanistan", "afg"],
}


def _cricapi_filter(matches: list, query: str) -> list:
    q = query.lower()
    series_f  = next((v for k, v in CRICAPI_SERIES_MAP.items() if k in q), None)
    country_f = next((v for k, v in COUNTRY_TEAMS.items() if k in q), None)

    filtered = []
    for m in matches:
        name  = (m.get("name") or "").lower()
        teams = [t.lower() for t in (m.get("teams") or [])]
        if series_f and series_f not in name:
            continue
        if country_f and not any(a in t for a in country_f for t in teams):
            continue
        filtered.append(m)
    return filtered or matches


def _cricapi_get_matches(query: str, date: str) -> list:
    key = _key("CRICAPI_KEY")
    if not key:
        print("  [CricAPI] No CRICAPI_KEY → https://cricketdata.org/signup.aspx")
        return []

    data = _safe_get(f"{CRICAPI_BASE}/matches",
                     params={"apikey": key, "date": date, "offset": 0}, timeout=25)
    if not data or data.get("status") != "success":
        data = _safe_get(f"{CRICAPI_BASE}/currentMatches",
                         params={"apikey": key, "offset": 0}, timeout=25)

    all_m = (data or {}).get("data") or []
    day_m = [m for m in all_m
             if (m.get("date") or m.get("dateTimeGMT") or "")[:10] == date] or all_m
    day_m = _cricapi_filter(day_m, query)

    results = []
    for m in day_m:
        teams  = m.get("teams") or []
        t1     = teams[0] if len(teams) > 0 else "TBA"
        t2     = teams[1] if len(teams) > 1 else "TBA"
        scores = m.get("score") or []
        t1_inn = next((s for s in scores if t1.lower()[:4] in (s.get("inning") or "").lower()), None)
        t2_inn = next((s for s in scores if t2.lower()[:4] in (s.get("inning") or "").lower()), None)
        s1     = f"{t1_inn['r']}/{t1_inn['w']} ({t1_inn['o']} ov)" if t1_inn else "Yet to bat"
        s2     = f"{t2_inn['r']}/{t2_inn['w']} ({t2_inn['o']} ov)" if t2_inn else "Yet to bat"
        status = "Match Ended" if m.get("matchEnded") else ("Live" if m.get("matchStarted") else "Scheduled")
        result = m.get("status") or status
        league = m.get("name") or "Cricket"
        venue  = m.get("venue") or "N/A"
        dt     = (m.get("dateTimeGMT") or date)[:16].replace("T", "  ")

        def _clean_inning(name):
            """CricAPI sometimes returns 'Team1,Team2 Inning 1' — clean it."""
            if "," in name:
                parts = name.split(",")
                # take the part that contains "inning" or the last part
                for p in parts:
                    if "inning" in p.lower():
                        return p.strip()
                return parts[-1].strip()
            return name
        detail = [f"   📊 {_clean_inning(s.get('inning',''))}: {s.get('r','–')}/{s.get('w','–')} ({s.get('o','–')} ov)"
                  for s in scores]

        lines = [
            "Most Recent Match:",
            f"🏏 {t1} vs {t2}",
            f"   {t1}: {s1}",
            f"   {t2}: {s2}",
            f"   Status: {status}",
            f"   Series: {league}",
            f"   Match Type: {(m.get('matchType') or 'N/A').upper()}",
            f"   Season: {date[:4]}",
            f"   Venue: {venue}",
            f"   Date: {dt}",
        ]
        lines.extend(detail)
        if "won" in result.lower() or m.get("matchEnded"):
            lines.append(f"   🏅 Result: {result}")
        results.append("\n".join(lines))

    return results


# ══════════════════════════════════════════════
# API 5 — TheSportsDB
# ══════════════════════════════════════════════

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

TSDB_LEAGUE_MAP = {
    "premier league": "4328", "epl": "4328",
    "la liga": "4335", "bundesliga": "4331", "serie a": "4332",
    "ligue 1": "4334", "champions league": "4480", "europa league": "4481",
    "mls": "4346", "eredivisie": "4337",
    "nba": "4387", "nfl": "4391", "nhl": "4380", "mlb": "4424",
    "cricket": "4507", "ipl": "4507",
    "rugby": "4516", "formula 1": "4370", "f1": "4370",
    "ufc": "4443", "tennis": "4388",
    "nba": "4387", "wnba": "4390",
    "afl": "4434", "a-league": "4356",
    "super rugby": "4515",
    "pro14": "4517",
    "premiership rugby": "4518",
    "nrl": "4435",
}


def _tsdb_key() -> str:
    return _key("TSDB_API_KEY") or "123"

# Track TSDB 429 rate limit — stop scanning when hit
_TSDB_RATE_LIMITED = False
_TSDB_RATE_LIMITED_UNTIL = 0.0

def _tsdb_rate_ok() -> bool:
    import time as _t
    global _TSDB_RATE_LIMITED, _TSDB_RATE_LIMITED_UNTIL
    if _TSDB_RATE_LIMITED and _t.time() < _TSDB_RATE_LIMITED_UNTIL:
        return False
    _TSDB_RATE_LIMITED = False
    return True

def _tsdb_mark_rate_limited():
    import time as _t
    global _TSDB_RATE_LIMITED, _TSDB_RATE_LIMITED_UNTIL
    _TSDB_RATE_LIMITED = True
    _TSDB_RATE_LIMITED_UNTIL = _t.time() + 60  # back off 60 seconds
    print("  [tsdb] Rate limited (429) — backing off 60s")


def _tsdb_get_matches(query: str, date: str) -> list:
    if not _tsdb_rate_ok():
        return []
    q  = query.lower()
    lid = next((v for k, v in TSDB_LEAGUE_MAP.items() if k in q), None)

    if not lid:
        data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/search_all_leagues.php",
                         {"s": query.split()[0]})
        if data and data.get("countrys"):
            for lg in data["countrys"]:
                if any(w.lower() in (lg.get("strLeague") or "").lower() for w in query.split()):
                    lid = lg.get("idLeague")
                    break

    if not lid:
        return []

    data   = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/eventsday.php",
                       {"d": date, "l": lid})
    events = (data or {}).get("events") or []
    results = []

    for e in events:
        home   = e.get("strHomeTeam", "?")
        away   = e.get("strAwayTeam", "?")
        hs     = e.get("intHomeScore") if e.get("intHomeScore") is not None else "–"
        as_    = e.get("intAwayScore") if e.get("intAwayScore") is not None else "–"
        status = _clean_status(e.get("strStatus") or e.get("strProgress") or "Scheduled")
        league = e.get("strLeague", query)
        season = e.get("strSeason", "N/A")
        venue  = e.get("strVenue") or e.get("strStadium") or "N/A"
        dt     = f"{e.get('dateEvent', date)}  {e.get('strTime', '??:??')}"
        results.append(_fmt(home, away, hs, as_, status, league, season, venue, dt))

    return results


# ══════════════════════════════════════════════
# API 6 — OpenLigaDB  (Bundesliga)
# ══════════════════════════════════════════════

def _openliga_get_matches(query: str, date: str) -> list:
    q = query.lower()
    if not any(k in q for k in ["bundesliga", "german", "bl1", "bl2"]):
        return []
    league_short = "bl2" if "2" in q else "bl1"
    data = _safe_get(f"https://api.openligadb.de/getmatchdata/{league_short}/{date[:4]}")
    results = []
    for m in (data or []):
        match_date = (m.get("matchDateTimeUTC") or "")[:10]
        if match_date != date:
            continue
        t1     = m.get("team1", {}).get("teamName", "?")
        t2     = m.get("team2", {}).get("teamName", "?")
        goals  = m.get("matchResults") or []
        final  = next((g for g in goals if g.get("resultTypeID") == 2), goals[-1] if goals else {})
        hs     = final.get("pointsTeam1", "–")
        as_    = final.get("pointsTeam2", "–")
        status = "Full Time" if m.get("matchIsFinished") else "Scheduled"
        lg     = "Bundesliga" if league_short == "bl1" else "2. Bundesliga"
        dt     = match_date + "  " + (m.get("matchDateTimeUTC") or "")[11:16]
        results.append(_fmt(t1, t2, hs, as_, status, lg, date[:4], "N/A", dt))
    return results


# ══════════════════════════════════════════════
# API 7 — Sportmonks Cricket  ← NEW
# ══════════════════════════════════════════════

SPORTMONKS_BASE = "https://cricket.sportmonks.com/api/v2.0"

# Sportmonks fixture status strings that mean "completed"
_SM_COMPLETED_STATUSES = {
    "finished", "completed", "result", "stumps",
    "aban", "cancelled", "no result",
}


def _sportmonks_get_last_match(team_name: str) -> list:
    """
    Fetch the most recent completed cricket match for a team via Sportmonks.
    No 72-hour restriction — returns full historical data.
    Used as fallback when CricAPI returns nothing.
    """
    key = _key("SPORTMONKS_KEY")
    if not key:
        print("  [Sportmonks] No SPORTMONKS_KEY in .env — get free key at https://sportmonks.com/cricket-api")
        return []

    # ── Step 1: Find team ID ──────────────────────────────────────────────
    search_data = _safe_get(
        f"{SPORTMONKS_BASE}/teams",
        params={"api_token": key, "filter[name]": team_name},
        timeout=15
    )
    teams_found = (search_data or {}).get("data") or []

    # If exact match fails, try first word of team name (e.g. "England" from "England cricket")
    if not teams_found:
        short = team_name.split()[0]
        search_data = _safe_get(
            f"{SPORTMONKS_BASE}/teams",
            params={"api_token": key, "filter[name]": short},
            timeout=15
        )
        teams_found = (search_data or {}).get("data") or []

    if not teams_found:
        print(f"  [Sportmonks] No team found for '{team_name}'")
        return []

    team_id      = teams_found[0].get("id")
    team_official = teams_found[0].get("name", team_name)
    print(f"  [Sportmonks] Found team: {team_official} (id={team_id})")

    # ── Step 2: Fetch recent fixtures ─────────────────────────────────────
    # Sportmonks does NOT allow filter[team_id] — must query localteam_id and visitorteam_id separately
    # Also filter to last 90 days only to avoid returning years of historical data
    import datetime as _dt
    import requests as _req
    _today     = _dt.date.today().isoformat()
    _since     = (_dt.date.today() - _dt.timedelta(days=90)).isoformat()

    def _sm_fixtures(filter_key):
        url = (f"{SPORTMONKS_BASE}/fixtures"
               f"?api_token={key}"
               f"&filter[{filter_key}]={team_id}"
               f"&filter[starts_between]={_since},{_today}"
               f"&sort=-starting_at"
               f"&include=runs,venue,localteam,visitorteam,league"
               f"&per_page=10")
        try:
            r = _req.get(url, timeout=20)
            if r.status_code == 200:
                return r.json().get("data") or []
        except Exception:
            pass
        return []

    home_fixtures = _sm_fixtures("localteam_id")
    away_fixtures = _sm_fixtures("visitorteam_id")
    # Merge, deduplicate by fixture id, sort newest first
    seen_ids = set()
    all_fixtures = []
    for f in home_fixtures + away_fixtures:
        if f.get("id") not in seen_ids:
            seen_ids.add(f.get("id"))
            all_fixtures.append(f)
    all_fixtures.sort(key=lambda f: f.get("starting_at") or "", reverse=True)
    fixtures_data = {"data": all_fixtures}
    fixtures = (fixtures_data or {}).get("data") or []
    print(f"  [Sportmonks] {len(fixtures)} fixtures for '{team_name}'")

    results = []
    for f in fixtures:
        raw_status = (f.get("status") or "").lower().strip()
        if raw_status not in _SM_COMPLETED_STATUSES:
            continue

        localteam_id   = f.get("localteam_id")
        visitorteam_id = f.get("visitorteam_id")

        # Prefer included team objects over bare IDs
        local_obj   = f.get("localteam")   or {}
        visitor_obj = f.get("visitorteam") or {}
        local_name   = local_obj.get("name")   or f"Team {localteam_id}"
        visitor_name = visitor_obj.get("name") or f"Team {visitorteam_id}"

        # Build score strings from runs[]
        runs = f.get("runs") or []

        def _score_str(tid):
            innings = [r for r in runs if r.get("team_id") == tid]
            if not innings:
                return "Yet to bat"
            # last innings for that team
            r = innings[-1]
            sc  = r.get("score")   if r.get("score")   is not None else "–"
            wk  = r.get("wickets") if r.get("wickets") is not None else "–"
            ov  = r.get("overs")   if r.get("overs")   is not None else "–"
            return f"{sc}/{wk} ({ov} ov)"

        s1 = _score_str(localteam_id)
        s2 = _score_str(visitorteam_id)

        league_obj = f.get("league") or {}
        league     = league_obj.get("name") or "Cricket"
        venue_obj  = f.get("venue") or {}
        venue      = venue_obj.get("name") or "N/A"
        dt         = (f.get("starting_at") or "")[:16].replace("T", "  ")
        note       = f.get("note") or ""     # e.g. "India won by 6 wickets"
        match_type = (f.get("type") or "N/A").upper()

        lines = [
            "Most Recent Match:",
            f"🏏 {local_name} vs {visitor_name}",
            f"   {local_name}: {s1}",
            f"   {visitor_name}: {s2}",
            f"   Status: Match Ended",
            f"   Series: {league}",
            f"   Match Type: {match_type}",
            f"   Season: {dt[:4]}",
            f"   Venue: {venue}",
            f"   Date: {dt}",
        ]
        if note:
            lines.append(f"   🏅 Result: {note}")

        results.append("\n".join(lines))

    if results:
        # Return only the single most recent completed match
        print(f"  [Sportmonks] Returning most recent of {len(results)} completed match(es) for '{team_name}'")
        return results[:1]
    else:
        print(f"  [Sportmonks] No completed matches found for '{team_name}' in last 90 days")
    return []


def _sportmonks_get_matches_by_date(query: str, date: str) -> list:
    """
    Fetch cricket fixtures for a given date from Sportmonks.
    Used as a fallback in get_matches() for date-specific cricket queries.
    """
    key = _key("SPORTMONKS_KEY")
    if not key:
        return []

    fixtures_data = _safe_get(
        f"{SPORTMONKS_BASE}/fixtures",
        params={
            "api_token": key,
            "filter[starts_between]": f"{date},{date}",
            "include": "runs,venue,localteam,visitorteam,league",
            "per_page": 20,
        },
        timeout=20
    )
    fixtures = (fixtures_data or {}).get("data") or []

    q = query.lower()
    results = []

    for f in fixtures:
        localteam_id   = f.get("localteam_id")
        visitorteam_id = f.get("visitorteam_id")
        local_obj      = f.get("localteam")   or {}
        visitor_obj    = f.get("visitorteam") or {}
        local_name     = local_obj.get("name")   or f"Team {localteam_id}"
        visitor_name   = visitor_obj.get("name") or f"Team {visitorteam_id}"

        # Apply team/series filter from query
        if q and not any(
            term in local_name.lower() or term in visitor_name.lower()
            for term in q.split()
            if len(term) > 3 and term not in
               {"cricket","match","score","result","latest","today","yesterday"}
        ):
            # Only skip if query has meaningful filter words
            meaningful = [w for w in q.split()
                          if len(w) > 3 and w not in
                          {"cricket","match","score","result","latest","today","yesterday","what","show","give"}]
            if meaningful:
                continue

        raw_status = (f.get("status") or "").lower().strip()
        status     = "Match Ended" if raw_status in _SM_COMPLETED_STATUSES else (
                     "Live" if raw_status in {"live", "inprogress", "in_progress"} else "Scheduled")

        runs = f.get("runs") or []

        def _sc(tid):
            inn = [r for r in runs if r.get("team_id") == tid]
            if not inn:
                return "Yet to bat"
            r = inn[-1]
            return f"{r.get('score','–')}/{r.get('wickets','–')} ({r.get('overs','–')} ov)"

        league = (f.get("league") or {}).get("name") or "Cricket"
        venue  = (f.get("venue")  or {}).get("name") or "N/A"
        dt     = (f.get("starting_at") or date)[:16].replace("T", "  ")
        note   = f.get("note") or ""

        lines = [
            "Most Recent Match:",
            f"🏏 {local_name} vs {visitor_name}",
            f"   {local_name}: {_sc(localteam_id)}",
            f"   {visitor_name}: {_sc(visitorteam_id)}",
            f"   Status: {status}",
            f"   Series: {league}",
            f"   Match Type: {(f.get('type') or 'N/A').upper()}",
            f"   Season: {dt[:4]}",
            f"   Venue: {venue}",
            f"   Date: {dt}",
        ]
        if note:
            lines.append(f"   🏅 Result: {note}")

        results.append("\n".join(lines))

    return results


# ══════════════════════════════════════════════
# De-duplicate
# ══════════════════════════════════════════════

def _dedup(results: list) -> list:
    seen: dict = {}
    for r in results:
        lines = r.split("\n")
        key   = lines[1] if len(lines) > 1 else r[:60]
        if key not in seen:
            seen[key] = r
        else:
            if _has_real_score(r) and not _has_real_score(seen[key]):
                seen[key] = r
    return list(seen.values())


# ══════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════

def get_matches(query: str, date: str = "today",
                prefer_completed: bool = False) -> list:
    resolved = _resolve_date(date)
    cricket  = _is_cricket(query)
    print(f"\n🔍 '{query}' on {resolved} {'🏏' if cricket else '⚽'}")

    results = []
    if cricket:
        results.extend(_cricapi_get_matches(query, resolved))
        results.extend(_tsdb_get_matches(query, resolved))
        # Sportmonks fallback for cricket date queries
        if not results or all(not _has_real_score(r) for r in results):
            print(f"  [get_matches] Cricket: trying Sportmonks for date {resolved}")
            results.extend(_sportmonks_get_matches_by_date(query, resolved))
    else:
        results.extend(_espn_get_matches(query, resolved))
        results.extend(_apifootballcom_get_matches(query, resolved))
        results.extend(_tsdb_get_matches(query, resolved))
        results.extend(_openliga_get_matches(query, resolved))

    unique = _dedup(results)

    if not unique:
        return [f"No matches found for '{query}' on {resolved}."]

    if prefer_completed:
        unique.sort(key=_sort_key_from_str)
        completed_or_live = [
            m for m in unique
            if _sort_key_from_str(m) <= 1 and _has_real_score(m)
        ]
        if completed_or_live:
            return completed_or_live

    return unique


def get_matches_text(query: str, date: str = "today",
                     prefer_completed: bool = False) -> str:
    return "\n\n" + SEP.join(get_matches(query, date,
                                          prefer_completed=prefer_completed))


def get_predictions(query: str, date: str = "today") -> str:
    resolved = _resolve_date(date)
    print(f"\n🔮 Predictions for '{query}' on {resolved}")

    results = []
    unique = _dedup(results)
    if not unique:
        return f"No predictions found for '{query}'."
    return "\n\n" + SEP.join(unique)


def get_cricket_live(team_filter: str = "") -> str:
    key = _key("CRICAPI_KEY")
    if not key:
        return "Set CRICAPI_KEY in .env — free at https://cricketdata.org/signup.aspx"

    matches, _ = _get_current_matches(key)

    # Domestic/franchise league keywords — always exclude these for national team queries
    _DOMESTIC_SERIES = {
        "legends league", "ranji", "sheffield shield", "county",
        "domestic", "franchise", "provincial", "inter-provincial",
        "inter state", "club", "regional", "qualifier", "emerging",
        "under-19", "u19", "u-19", "women", "a team", "a tour",
        "legends", "masters", "veteran", "old",
    }

    if team_filter:
        tf = team_filter.lower()
        # Step 1: exact national team match
        filtered = [m for m in matches
                    if any(_cricket_team_matches(tf, t) for t in (m.get("teams") or []))]
        # Step 2: for national teams, also exclude domestic series
        if tf in CRICKET_NATIONAL_TEAMS:
            filtered = [m for m in filtered
                        if not any(kw in (m.get("name") or "").lower()
                                   for kw in _DOMESTIC_SERIES)]
        matches = filtered

    if not matches:
        return f"No live cricket matches found{' for ' + team_filter if team_filter else ''}."

    results = []
    for m in matches:
        teams  = m.get("teams") or []
        t1     = teams[0] if len(teams) > 0 else "TBA"
        t2     = teams[1] if len(teams) > 1 else "TBA"
        scores = m.get("score") or []
        status = "Match Ended" if m.get("matchEnded") else ("Live" if m.get("matchStarted") else "Scheduled")
        result = m.get("status") or status
        venue  = m.get("venue") or "N/A"
        dt     = (m.get("dateTimeGMT") or datetime.date.today().isoformat())[:16].replace("T", "  ")
        lines  = [
            "Most Recent Match:",
            f"🏏 {t1} vs {t2}",
            f"   Status: {status}",
            f"   Series: {m.get('name', 'Cricket')}",
            f"   Match Type: {(m.get('matchType') or 'N/A').upper()}",
            f"   Venue: {venue}",
            f"   Date: {dt}",
        ]
        for s in scores:
            lines.append(f"   📊 {s.get('inning','')}: {s.get('r','–')}/{s.get('w','–')} ({s.get('o','–')} ov)")
        if "won" in result.lower() or m.get("matchEnded"):
            lines.append(f"   🏅 Result: {result}")
        results.append("\n".join(lines))

    return "\n\n" + SEP.join(results)


def validate_keys() -> None:
    print("\n" + "═" * 50)
    print("  API KEY VALIDATION REPORT")
    print("═" * 50)

    checks = [
        ("APIFOOTBALL_COM_KEY", "apifootball.com (1019 leagues, current season)", "https://apifootball.com"),
        ("CRICAPI_KEY",      "CricAPI (100/day free cricket)",       "https://cricketdata.org/signup.aspx"),
        ("SPORTMONKS_KEY",   "Sportmonks Cricket (historical data)", "https://sportmonks.com/cricket-api"),  # ← NEW
        ("TSDB_API_KEY",     "TheSportsDB Patreon (optional)",       "https://www.thesportsdb.com/patreon"),
    ]

    for env_var, label, signup in checks:
        val = _key(env_var)
        if val:
            print(f"  ✅ {label}")
            print(f"     Key: {val[:6]}{'*' * (len(val) - 6)}")
        else:
            if env_var == "TSDB_API_KEY":
                print(f"  ℹ️  {label} — using free key '123' (built-in)")
            else:
                print(f"  ❌ {label}")
                print(f"     Get free key → {signup}")

    print("═" * 50 + "\n")


# ══════════════════════════════════════════════
# RAG INTERFACE
# ══════════════════════════════════════════════

_TEAM_LEAGUE_MAP = {
    # La Liga
    'real madrid': 'La Liga', 'barcelona': 'La Liga',
    'atletico madrid': 'La Liga', 'sevilla': 'La Liga',
    'real sociedad': 'La Liga', 'villarreal': 'La Liga',
    # Premier League
    'manchester united': 'Premier League', 'manchester city': 'Premier League',
    'liverpool': 'Premier League', 'chelsea': 'Premier League',
    'arsenal': 'Premier League', 'tottenham': 'Premier League',
    'newcastle': 'Premier League', 'aston villa': 'Premier League',
    'west ham': 'Premier League', 'brighton': 'Premier League',
    'everton': 'Premier League', 'brentford': 'Premier League',
    # Bundesliga
    'bayern munich': 'Bundesliga', 'borussia dortmund': 'Bundesliga',
    'rb leipzig': 'Bundesliga', 'bayer leverkusen': 'Bundesliga',
    # Serie A
    'juventus': 'Serie A', 'ac milan': 'Serie A',
    'inter milan': 'Serie A', 'napoli': 'Serie A',
    'as roma': 'Serie A', 'lazio': 'Serie A',
    'parma': 'Serie A', 'fiorentina': 'Serie A',
    'atalanta': 'Serie A', 'bologna': 'Serie A',
    'torino': 'Serie A', 'udinese': 'Serie A',
    'genoa': 'Serie A', 'cagliari': 'Serie A',
    'lecce': 'Serie A', 'monza': 'Serie A',
    'empoli': 'Serie A', 'hellas verona': 'Serie A',
    'como': 'Serie A', 'venezia': 'Serie A', 'cremonese': 'Serie A',
    'salernitana': 'Serie A', 'frosinone': 'Serie A', 'pisa': 'Serie A',
    'sassuolo': 'Serie A', 'spezia': 'Serie A',
    # Bundesliga extras
    'gladbach': 'Bundesliga', 'borussia monchengladbach': 'Bundesliga',
    'wolfsburg': 'Bundesliga', 'eintracht frankfurt': 'Bundesliga',
    'freiburg': 'Bundesliga', 'union berlin': 'Bundesliga',
    'hoffenheim': 'Bundesliga', 'mainz': 'Bundesliga',
    'augsburg': 'Bundesliga', 'bochum': 'Bundesliga',
    'heidenheim': 'Bundesliga', 'darmstadt': 'Bundesliga',
    'stuttgart': 'Bundesliga', 'werder bremen': 'Bundesliga',
    # Premier League extras
    'fulham': 'Premier League', 'crystal palace': 'Premier League',
    'wolves': 'Premier League', 'nottingham forest': 'Premier League',
    'bournemouth': 'Premier League', 'luton': 'Premier League',
    'sheffield united': 'Premier League', 'burnley': 'Premier League',
    'leicester': 'Premier League', 'ipswich': 'Premier League',
    'southampton': 'Premier League',
    # La Liga extras
    'real betis': 'La Liga', 'athletic bilbao': 'La Liga',
    'osasuna': 'La Liga', 'getafe': 'La Liga',
    'celta vigo': 'La Liga', 'girona': 'La Liga',
    'mallorca': 'La Liga', 'rayo vallecano': 'La Liga',
    'cadiz': 'La Liga', 'almeria': 'La Liga', 'granada': 'La Liga',
    # Ligue 1 extras
    'rennes': 'Ligue 1', 'strasbourg': 'Ligue 1',
    'nantes': 'Ligue 1', 'reims': 'Ligue 1',
    'brest': 'Ligue 1', 'toulouse': 'Ligue 1',
    # Other leagues
    'inter miami': 'MLS', 'la galaxy': 'MLS',
    'seattle sounders': 'MLS', 'portland timbers': 'MLS',
    'atlanta united': 'MLS', 'toronto fc': 'MLS',
    'flamengo': 'Brasileirao', 'palmeiras': 'Brasileirao',
    'corinthians': 'Brasileirao', 'sao paulo': 'Brasileirao',
    'boca juniors': 'Liga Profesional', 'river plate': 'Liga Profesional',
    # Ligue 1
    'psg': 'Ligue 1', 'paris saint-germain': 'Ligue 1',
    'marseille': 'Ligue 1', 'lyon': 'Ligue 1',
    'monaco': 'Ligue 1', 'lille': 'Ligue 1',
    'nice': 'Ligue 1', 'lens': 'Ligue 1',
    # National football teams (International competitions)
    'england': 'International', 'germany': 'International',
    'france': 'International', 'spain': 'International',
    'italy': 'International', 'portugal': 'International',
    'netherlands': 'International', 'belgium': 'International',
    'croatia': 'International', 'denmark': 'International',
    'sweden': 'International', 'norway': 'International',
    'switzerland': 'International', 'austria': 'International',
    'poland': 'International', 'ukraine': 'International',
    'serbia': 'International', 'turkey': 'International',
    'czech republic': 'International', 'greece': 'International',
    'scotland': 'International', 'wales': 'International',
    'ireland': 'International', 'hungary': 'International',
    'brazil': 'International', 'argentina': 'International',
    'uruguay': 'International', 'colombia': 'International',
    'chile': 'International', 'ecuador': 'International',
    'peru': 'International', 'venezuela': 'International',
    'mexico': 'International', 'usa': 'International',
    'united states': 'International', 'canada': 'International',
    'costa rica': 'International', 'panama': 'International',
    'japan': 'International', 'south korea': 'International',
    'iran': 'International', 'saudi arabia': 'International',
    'australia': 'International', 'china': 'International',
    'india': 'International', 'pakistan': 'International',
    'senegal': 'International', 'nigeria': 'International',
    'ghana': 'International', 'cameroon': 'International',
    'egypt': 'International', 'morocco': 'International',
    'south africa': 'International', 'ivory coast': 'International',
    'tunisia': 'International', 'algeria': 'International',
    # Other football
    'ajax': 'Eredivisie', 'psv': 'Eredivisie',
    'porto': 'Primeira Liga', 'benfica': 'Primeira Liga',
    'celtic': 'Scottish Premier', 'rangers fc': 'Scottish Premier',
    'glasgow rangers': 'Scottish Premier',
    # NBA
    'lakers': 'NBA', 'celtics': 'NBA', 'warriors': 'NBA',
    'bulls': 'NBA', 'heat': 'NBA', 'nets': 'NBA',
    'knicks': 'NBA', 'bucks': 'NBA', 'nuggets': 'NBA',
    'suns': 'NBA', 'clippers': 'NBA', 'mavericks': 'NBA',
    'spurs': 'NBA', 'rockets': 'NBA', 'cavaliers': 'NBA',
    'pistons': 'NBA', 'hawks': 'NBA', 'hornets': 'NBA',
    'pacers': 'NBA', 'magic': 'NBA', 'wizards': 'NBA',
    'raptors': 'NBA', 'sixers': 'NBA', '76ers': 'NBA',
    'jazz': 'NBA', 'thunder': 'NBA', 'pelicans': 'NBA',
    'grizzlies': 'NBA', 'timberwolves': 'NBA', 'blazers': 'NBA',
    'trail blazers': 'NBA', 'kings': 'NBA',
    # NFL
    'chiefs': 'NFL', 'patriots': 'NFL', 'cowboys': 'NFL',
    'eagles': 'NFL', '49ers': 'NFL', 'ravens': 'NFL',
    'bills': 'NFL', 'bengals': 'NFL', 'steelers': 'NFL',
    'broncos': 'NFL', 'raiders': 'NFL', 'chargers': 'NFL',
    'dolphins': 'NFL', 'jets': 'NFL', 'giants': 'NFL',
    'commanders': 'NFL', 'bears': 'NFL', 'lions': 'NFL',
    'packers': 'NFL', 'vikings': 'NFL',
    # NHL
    'maple leafs': 'NHL', 'canadiens': 'NHL', 'bruins': 'NHL',
    'new york rangers': 'NHL', 'penguins': 'NHL', 'blackhawks': 'NHL',
    'red wings': 'NHL', 'oilers': 'NHL', 'flames': 'NHL',
    'canucks': 'NHL', 'capitals': 'NHL', 'lightning': 'NHL',
    'avalanche': 'NHL', 'golden knights': 'NHL',
    # MLB
    'yankees': 'MLB', 'red sox': 'MLB', 'dodgers': 'MLB',
    'cubs': 'MLB', 'mets': 'MLB', 'giants': 'MLB',
    'astros': 'MLB', 'braves': 'MLB', 'cardinals': 'MLB',
    # Cricket tournaments
    'ipl': 'IPL', 'bbl': 'BBL', 'psl': 'PSL', 'cpl': 'CPL',
    # Formula 1
    'red bull': 'Formula 1', 'ferrari': 'Formula 1',
    'mercedes': 'Formula 1', 'mclaren': 'Formula 1',
    'aston martin': 'Formula 1', 'alpine': 'Formula 1',
    'williams': 'Formula 1', 'haas': 'Formula 1',
}

_DATE_SIGNALS = {
    'yesterday': 'yesterday', 'last night': 'yesterday',
    'last week': 'yesterday',
    'today': 'today', 'tonight': 'today', 'now': 'today',
    'live': 'today', 'ongoing': 'today',
}

_LEAGUE_KEYWORDS = {
    'premier league': 'Premier League', 'epl': 'Premier League',
    'la liga': 'La Liga', 'bundesliga': 'Bundesliga',
    'serie a': 'Serie A', 'ligue 1': 'Ligue 1',
    'champions league': 'Champions League', 'ucl': 'Champions League',
    'europa league': 'Europa League', 'uel': 'Europa League',
    'conference league': 'Conference League', 'uecl': 'Conference League',
    # Domestic cups — listed before generic league names so they match first
    'fa cup': 'FA Cup', 'league cup': 'League Cup', 'carabao cup': 'League Cup',
    'copa del rey': 'Copa del Rey',
    'coppa italia': 'Coppa Italia',
    'dfb pokal': 'DFB Pokal', 'dfb-pokal': 'DFB Pokal',
    'coupe de france': 'Coupe de France',
    'mls': 'MLS', 'eredivisie': 'Eredivisie',
    'primeira liga': 'Primeira Liga', 'scottish premiership': 'Scottish Premiership',
    'super lig': 'Süper Lig',
    'nba': 'NBA', 'nfl': 'NFL', 'nhl': 'NHL', 'mlb': 'MLB',
    'formula 1': 'Formula 1', 'f1': 'Formula 1',
    'ufc': 'UFC', 'mma': 'MMA',
    'tennis': 'Tennis', 'atp': 'Tennis', 'wta': 'Tennis',
    'ipl': 'IPL', 'cricket': 'cricket',
    't20i': 'T20I', 't20': 'T20',
    'odi': 'ODI', 'test match': 'Test cricket',
    'psl': 'PSL', 'bbl': 'BBL', 'cpl': 'CPL',
    'rugby': 'Rugby', 'super rugby': 'Super Rugby',
    'nrl': 'NRL', 'afl': 'AFL',
}

_WANTS_RESULT_RE = _re_sports.compile(
    r'\b(latest score|latest result|score of|result of|who won|'
    r'last match|last game|last \d+|last\s+\d+\s+match|'
    r'recent match|yesterday|final score|'
    r'match result|scorecard|what was the score|what is the score|'
    r'current score|live score)\b',
    _re_sports.IGNORECASE
)

_WANTS_UPCOMING_RE = _re_sports.compile(
    r'\b(upcoming|next match|next game|next ipl|next kkr|next csk|next mi|next rcb|'
    r'fixture|fixtures|schedule|when will|when does|will play|coming up|next fixture|'
    r'next\s+\w+\s+match|next\s+\w+\s+game)\b',
    _re_sports.IGNORECASE
)


def _has_real_score(match_str: str) -> bool:
    for line in match_str.split("\n"):
        stripped = line.strip()
        if ":" in stripped and stripped.split(":")[0].strip() in (
            "Status", "League/Series", "Season", "Venue", "Date",
            "Series", "Match Type", "Result"
        ):
            continue
        if _re_sports.search(r'(?<!\d)\d{1,3}\s*[–\-]\s*\d{1,3}(?!\d)', stripped):
            if not _re_sports.search(r'\b\d{4}[–\-]\d{2}\b|\b\d{2}[–\-]\d{2}\b', stripped):
                return True
        if _re_sports.search(r'\b\d+/\d+\b', stripped):
            return True
    return False


def _lookup_cricket_team_dynamic(team_name: str) -> str:
    """
    Dynamically identify if a team name is a cricket team and which tournament.
    Checks CRICKET_FRANCHISE_MAP first, then CricAPI series search.
    Returns tournament name like 'IPL', 'PSL', 'cricket' or None.
    """
    tl = team_name.lower().strip()

    # 1. Check franchise map first — instant lookup
    for franchise, tournament in CRICKET_FRANCHISE_MAP.items():
        if franchise in tl or tl in franchise:
            print(f"  [cricket_dynamic] '{team_name}' → franchise team → {tournament}")
            return tournament

    # 2. CricAPI series search — check if team appears in any series
    key = _key("CRICAPI_KEY")
    if not key:
        return None
    try:
        data = _safe_get(
            f"{CRICAPI_BASE}/series",
            params={"apikey": key, "offset": 0, "search": team_name},
            timeout=15
        )
        series = (data or {}).get("data") or []
        if series:
            # Found series containing this team name — it's a cricket team
            print(f"  [cricket_dynamic] '{team_name}' found in {len(series)} CricAPI series → cricket")
            return "cricket"
    except Exception as e:
        print(f"  [cricket_dynamic] Error: {e}")

    return None


def _lookup_team_dynamic(team_name: str) -> str:
    search_name = team_name.strip().title()

    try:
        data = _safe_get(
            f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
            {"t": search_name}
        )
        teams = (data or {}).get("teams") or []
        if not teams:
            short = _re_sports.sub(r'^(fc|afc|sc|ac|ss|as|rc|vfb|vfl|rb|bv|sv|cf|cd|ud|sd|rcd)\s+', '',
                           team_name.lower()).strip()
            if short != team_name.lower():
                data2 = _safe_get(
                    f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
                    {"t": short.title()}
                )
                teams = (data2 or {}).get("teams") or []

        if not teams:
            print(f"  [dynamic_lookup] No team found for '{team_name}'")
            return None

        team = teams[0]
        league_raw = team.get("strLeague") or ""
        sport_raw  = team.get("strSport") or ""

        TSDB_TO_INTERNAL = {
            "german bundesliga":         "Bundesliga",
            "bundesliga":                "Bundesliga",
            "english premier league":    "Premier League",
            "premier league":            "Premier League",
            "la liga":                   "La Liga",
            "spanish la liga":           "La Liga",
            "serie a":                   "Serie A",
            "italian serie a":           "Serie A",
            "ligue 1":                   "Ligue 1",
            "french ligue 1":            "Ligue 1",
            "champions league":          "Champions League",
            "uefa champions league":     "Champions League",
            "europa league":             "Europa League",
            "mls":                       "MLS",
            "eredivisie":                "Eredivisie",
            "scottish premiership":      "Scottish Premier",
            "primeira liga":             "Primeira Liga",
            "nba":                       "NBA",
            "nfl":                       "NFL",
            "nhl":                       "NHL",
            "mlb":                       "MLB",
            "ipl":                       "IPL",
            "cricket":                   "cricket",
            "formula 1":                 "Formula 1",
            "nrl":                       "NRL",
            "afl":                       "AFL",
            "super rugby":               "Super Rugby",
        }

        league_key = None
        league_lower = league_raw.lower()
        for tsdb_name, internal in sorted(TSDB_TO_INTERNAL.items(), key=lambda x: -len(x[0])):
            if tsdb_name in league_lower:
                league_key = internal
                break

        if not league_key:
            if sport_raw.lower() == "cricket":
                league_key = "cricket"
            elif sport_raw.lower() in ("soccer", "football"):
                league_key = league_raw
            else:
                league_key = sport_raw or league_raw

        if league_key:
            key = team_name.lower().strip()
            _TEAM_LEAGUE_MAP[key] = league_key
            official = (team.get("strTeam") or "").lower().strip()
            if official and official != key:
                _TEAM_LEAGUE_MAP[official] = league_key
            print(f"  [dynamic_lookup] '{team_name}' → '{league_key}' (cached)")
            return league_key

    except Exception as e:
        print(f"  [dynamic_lookup] Error for '{team_name}': {e}")

    return None


def _extract_sport_and_date(query: str) -> tuple:
    q = query.lower().strip()
    date = 'today'
    team_filter = ''

    import calendar as _cal
    date = 'today'
    date_from = None  # for range queries
    date_to   = None

    # Explicit YYYY-MM-DD
    dm = _re_sports.search(r'(\d{4}-\d{2}-\d{2})', q)
    if dm:
        date = dm.group(1)
    else:
        import datetime as _dttx
        MONTHS = {
            'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
            'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
            'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
            'sep':9,'oct':10,'nov':11,'dec':12,
        }
        _month_pat = '|'.join(MONTHS.keys())
        # Specific day: "8 march", "8th march 2026", "march 8", "march 8th 2026"
        _dm = _re_sports.search(
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(' + _month_pat + r')(?:\s+(\d{4}))?\b'
            r'|\b(' + _month_pat + r')\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b',
            q, _re_sports.IGNORECASE
        )
        if _dm:
            g = _dm.groups()
            if g[0]: day_s, mon_s, yr_s = g[0], g[1], g[2]
            else:    mon_s, day_s, yr_s = g[3], g[4], g[5]
            mon_s = (mon_s or '').lower()
            if mon_s in MONTHS:
                yr = int(yr_s) if yr_s else _dttx.date.today().year
                try:
                    date = _dttx.date(yr, MONTHS[mon_s], int(day_s)).isoformat()
                except ValueError:
                    pass
        # Month + year range — only when no specific day AND no between-range was already parsed
        if date == 'today' and not date_from:
            _mrange = _re_sports.search(
                r'\b(' + '|'.join(MONTHS.keys()) + r')\b.*?(\d{4})|\b(\d{4}).*?\b(' + '|'.join(MONTHS.keys()) + r')\b',
                q, _re_sports.IGNORECASE
            )
            if _mrange:
                g = _mrange.groups()
                month_str = (g[0] or g[3] or '').lower()
                year_str  = g[1] or g[2] or ''
                if month_str in MONTHS and year_str:
                    m  = MONTHS[month_str]
                    yr = int(year_str)
                    last_day = _cal.monthrange(yr, m)[1]
                    date_from = f"{yr}-{m:02d}-01"
                    date_to   = f"{yr}-{m:02d}-{last_day:02d}"
                    date      = date_from
        # "between Month1 and Month2 [Year]" — year optional, defaults to current
        # Must run BEFORE month+year range to take priority
        _between = _re_sports.search(
            r'between\s+(\w+)\s+and\s+(\w+)(?:\s+(\d{4}))?', q, _re_sports.IGNORECASE
        )
        if _between:
            import datetime as _dtt
            m1_str = _between.group(1).lower()
            m2_str = _between.group(2).lower()
            yr_str = _between.group(3)
            if m1_str in MONTHS and m2_str in MONTHS:
                yr = int(yr_str) if yr_str else _dtt.date.today().year
                m1 = MONTHS[m1_str]; m2 = MONTHS[m2_str]
                # Handle year wrap: "between november and january" → nov this year, jan next year
                if m2 < m1:
                    yr2 = yr + 1
                else:
                    yr2 = yr
                last_day2 = _cal.monthrange(yr2, m2)[1]
                date_from = f"{yr}-{m1:02d}-01"
                date_to   = f"{yr2}-{m2:02d}-{last_day2:02d}"
                date      = date_from
        if not date_from:
            # "last month" / "last week" etc.
            import datetime as _dtt2
            today2 = _dtt2.date.today()
            if 'last month' in q:
                first_this = today2.replace(day=1)
                last_month_end = first_this - _dtt2.timedelta(days=1)
                date_from = last_month_end.replace(day=1).isoformat()
                date_to   = last_month_end.isoformat()
                date      = date_from
            elif 'last week' in q:
                start = today2 - _dtt2.timedelta(days=today2.weekday() + 7)
                date_from = start.isoformat()
                date_to   = (start + _dtt2.timedelta(days=6)).isoformat()
                date      = date_from
            elif 'this month' in q:
                date_from = today2.replace(day=1).isoformat()
                date_to   = today2.isoformat()
                date      = date_from
            else:
                for phrase, d in sorted(_DATE_SIGNALS.items(), key=lambda x: -len(x[0])):
                    if phrase in q:
                        date = d
                        break

    sport_query = None

    _league_explicit = False  # True only if user explicitly named a league/competition
    import re as _re_lkw
    # Keywords that need word-boundary matching to avoid substring false positives
    # e.g. "test match" inside "latest match", "odi" inside "modi", "t20" inside "t200"
    _REQUIRES_WORD_BOUNDARY = {"test match", "test cricket", "odi", "t20", "t20i",
                                "one day", "big bash", "the hundred"}
    for kw, league in sorted(_LEAGUE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in _REQUIRES_WORD_BOUNDARY:
            # Use word-boundary regex
            _pattern = r'\b' + r'\s+'.join(_re_lkw.escape(w) for w in kw.split()) + r'\b'
            if not _re_lkw.search(_pattern, q, _re_lkw.IGNORECASE):
                continue
        elif kw not in q:
            continue
        sport_query = league
        _league_explicit = True
        if league == 'cricket' or 'cricket' in league.lower():
            for nation in CRICKET_NATIONS:
                if nation in q:
                    team_filter = nation.title()
                    break
        else:
            # Check if a known team is also mentioned alongside the league
            for team in sorted(_TEAM_LEAGUE_MAP.keys(), key=len, reverse=True):
                if team in q:
                    team_filter = team
                    break
        break

    # If cricket detected but no team yet — scan full query for nations/franchises
    if not team_filter and (_is_cricket(query) or (sport_query and (
            'cricket' in (sport_query or '').lower() or
            sport_query in {'T20','T20I','ODI','Test cricket','IPL','BBL','PSL',
                            'CPL','The Hundred','SA20','ILT20','WPL'}))):
        import re as _re_ct
        for _nat in CRICKET_NATIONS:
            if _re_ct.search(r'\b' + _re_ct.escape(_nat) + r'\b', q, _re_ct.IGNORECASE):
                team_filter = _nat.title()
                break
        if not team_filter:
            for _fr in CRICKET_FRANCHISE_MAP:
                if len(_fr) <= 3:
                    if _re_ct.search(r'\b' + _re_ct.escape(_fr) + r'\b', q):
                        team_filter = _fr.title()
                        break
                elif _fr in q:
                    team_filter = _fr.title()
                    break

    pre_filler = _re_sports.compile(
        r'^(give me|show me|who won|what was|what is|what are|latest|get|find|'
        r'score of|result of|upcoming|next|schedule|fixture|'
        r'the|a|an|tell me|fetch|between|of|for)\s+'
    )
    post_filler = _re_sports.compile(
        r'\s+(score|result|match|today|yesterday|last night|live|latest|2\d{3}.*)$'
    )

    if not sport_query and ' vs ' in q:
        q_vs = pre_filler.sub('', q).strip()
        parts = _re_sports.split(r'\s+vs\.?\s+', q_vs, maxsplit=1)
        if len(parts) == 2:
            t1 = post_filler.sub('', parts[0]).strip()
            t2 = post_filler.sub('', parts[1]).strip()
            t1 = pre_filler.sub('', t1).strip()
            if t1 and t2 and len(t1) > 1 and len(t2) > 1:
                sport_query = f'{t1} vs {t2}'

    # Also detect "between team1 and team2" as a VS / H2H query
    if not sport_query:
        _bet = _re_sports.search(
            r'\bbetween\s+(.+?)\s+and\s+([\w\s]+?)(?:\s*[?!.]|$)', q, _re_sports.IGNORECASE
        )
        if _bet:
            t1_raw = _bet.group(1).strip()
            t2_raw = _bet.group(2).strip()
            # Strip trailing filler words
            _trail = _re_sports.compile(
                r'\s+(score|result|match|today|yesterday|live|latest|in|on|at|the|a|an|\?|!)$',
                _re_sports.IGNORECASE
            )
            t1_raw = _trail.sub('', t1_raw).strip().rstrip('?,. ')
            t2_raw = _trail.sub('', t2_raw).strip().rstrip('?,. ')
            # Validate — both should be team-like (not date/schedule words)
            _non_team_words = {'january','february','march','april','may','june',
                               'july','august','september','october','november','december',
                               'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                               'upcoming','next','schedule','fixture','latest','recent','now',
                               'today','tonight','yesterday','live','score','result'}
            if (t1_raw and t2_raw and len(t1_raw) > 1 and len(t2_raw) > 1
                    and t1_raw.lower() not in _non_team_words
                    and t2_raw.lower() not in _non_team_words):
                sport_query = f'{t1_raw} vs {t2_raw}'
                print(f'  [parser] between→VS: {repr(sport_query)}')

    # ── Two-team detection: any phrasing with 2 teams = H2H ─────────────
    # Works with vs, between...and, or just two team names side by side.
    # Strategy: find known teams in the query; if 2 found → H2H.
    # If only 1 known team found but there's leftover text that looks like
    # another team, use that as the second team.
    _SCORE_INTENT = _re_sports.search(
        r'\b(score|result|won|win|goal|match|game|latest|recent|yesterday|today|who)\b',
        q, _re_sports.IGNORECASE
    )
    # Run two-team check even when league already detected (e.g. "arsenal chelsea fa cup")
    if _SCORE_INTENT:
        # Build clean version of query for team scanning
        _h2h_filler = _re_sports.compile(
            r'\b(who|won|what|was|the|is|between|and|vs|latest|recent|'
            r'score|result|match|game|today|yesterday|last\s+night|live|'
            r'give|me|show|tell|fetch|get|find|on|in|at|of|for|a|an|'
            r'upcoming|next|schedule|fixture|fixtures|when|will|coming|'
            r'what\s+is|what\s+are|how|does|do|did)\b',
            _re_sports.IGNORECASE
        )
        _q_clean = _h2h_filler.sub(' ', q)
        _q_clean = _re_sports.sub(r'\s+', ' ', _q_clean).strip()

        # Scan for known teams (longest first to avoid partial matches)
        _found_teams = []
        _used_positions = []
        _q_lower = _q_clean.lower()
        for _team in sorted(_TEAM_LEAGUE_MAP.keys(), key=len, reverse=True):
            _idx = _q_lower.find(_team)
            if _idx >= 0:
                _end = _idx + len(_team)
                _overlaps = any(_s <= _idx < _e or _s < _end <= _e
                                for _s, _e in _used_positions)
                if not _overlaps:
                    _found_teams.append(_team)
                    _used_positions.append((_idx, _end))
                    if len(_found_teams) == 2:
                        break

        # If only 1 known team, check if remaining text has a second team candidate
        if len(_found_teams) == 1:
            _known_team = _found_teams[0]
            # Remove known team from clean query to get remainder
            _remainder = _q_lower.replace(_known_team, ' ').strip()
            _remainder = _re_sports.sub(r'\s+', ' ', _remainder).strip()
            # Also strip league keywords from remainder
            for _lkw in _LEAGUE_KEYWORDS:
                _remainder = _remainder.replace(_lkw, ' ').strip()
            _remainder = _re_sports.sub(r'\s+', ' ', _remainder).strip()
            # Remainder must be 2-25 chars, no digits, not a filler/suffix word
            _bad_words = {'yesterday','today','latest','recent','live','score',
                          'result','match','game','goal','won','who','what',
                          # Upcoming/schedule intent words — never team names
                          'upcoming','next','schedule','fixture','fixtures',
                          'when','will','coming','soon','later',
                          # Club suffixes that are never standalone teams
                          'fc','cf','sc','ac','bc','sk','fk','bk','if',
                          'united','city','town','rovers','wanderers',
                          # League/competition fragments
                          'serie','liga','ligue','bundesliga','premier',
                          'champions','europa','conference','cup','league',
                          # Generic words
                          'club','team','squad','the','a','an','of','is','are',
                          # National team descriptors — never a team name alone
                          'national','football','soccer','international',
                          'men','women','mens','womens','senior','junior',
                          'side','eleven','xi','nt'}
            # Also reject multi-word phrases that are clearly not team names
            _bad_phrases = {
                'national football team', 'national team', 'national side',
                'football team', 'soccer team', 'football squad',
                'international team', 'national football squad',
            }
            if (2 <= len(_remainder) <= 25 and
                    not _re_sports.search(r'\d', _remainder) and
                    _remainder not in _bad_words and
                    _remainder not in _bad_phrases and
                    not all(w in _bad_words for w in _remainder.split()) and
                    len(_remainder.split()) <= 3):
                _found_teams.append(_remainder)

        if len(_found_teams) == 2:
            t1, t2 = _found_teams[0], _found_teams[1]
            # Only use a league hint if:
            #   a) user explicitly named a league in the query, OR
            #   b) both teams are in the SAME league
            # Never lock cross-league pairs to one team's league.
            _t1_league = _TEAM_LEAGUE_MAP.get(t1, "")
            _t2_league = _TEAM_LEAGUE_MAP.get(t2, "")
            if sport_query and ' vs ' not in (sport_query or ''):
                _orig_league = sport_query  # user said explicit league
            elif _t1_league and _t2_league and _t1_league == _t2_league:
                _orig_league = _t1_league  # same league — safe to hint
            else:
                _orig_league = ""  # cross-league or unknown — no hint
            sport_query = f"{t1} vs {t2}"
            if _orig_league and _orig_league not in (t1, t2):
                sport_query = f"{t1} vs {t2} [{_orig_league}]"
            team_filter = ""
            print(f'  [parser] two-team H2H: {repr(t1)} vs {repr(t2)} hint={repr(_orig_league)}')

    if not sport_query:
        for team, league in sorted(_TEAM_LEAGUE_MAP.items(), key=lambda x: -len(x[0])):
            if team in q:
                sport_query = league
                team_filter = team
                # _league_explicit stays False — league inferred from team, not user-specified
                break

    if not sport_query:
        _filler = _re_sports.compile(
            r'\b(give me|show me|what is|what was|what are|latest score of|latest result of|'
            r'score of|score for|result of|result for|latest|recent|current|live|'
            r'who won|tell me|fetch|get|find|upcoming|next|schedule|fixture|fixtures|'
            r'of|for|about|regarding|on|in|at|'
            r'last|the last|most recent|past|previous|'
            r'today|yesterday|last night|last match|last game|now|'
            r'match|matches|game|games|score|scores|result|results|the|a|an)\b'
        )
        candidate = _filler.sub(' ', q).strip()
        candidate = _re_sports.sub(r'\s+', ' ', candidate).strip()
        # Strip standalone numbers (match counts) that leaked into candidate
        # e.g. "the last 3 rcb matches" → candidate="3 rcb" → strip "3" → "rcb"
        candidate = _re_sports.sub(r'\b\d+\b\s*', '', candidate).strip()
        candidate = _re_sports.sub(r'\s+', ' ', candidate).strip()
        if len(candidate) > 2:
            # Check cricket franchise/team first
            cricket_league = _lookup_cricket_team_dynamic(candidate)
            if cricket_league:
                sport_query = cricket_league
                team_filter = candidate
            else:
                # Then check football/other sports via TSDB
                found_league = _lookup_team_dynamic(candidate)
                if found_league:
                    sport_query = found_league
                    team_filter = candidate

    if not sport_query:
        for nation in CRICKET_NATIONS:
            if nation in q:
                sport_query = f'{nation} cricket'
                break

    if not sport_query:
        filler = (r'\b(give me|show me|what is|what was|latest|recent|current|live|'
                  r'score of|score for|result of|result for|the|a|an|'
                  r'today|yesterday|last night|match|game|who won|'
                  r'tell me|fetch|get|find)\b')
        cleaned = _re_sports.sub(filler, '', q).strip()
        sport_query = cleaned if len(cleaned) > 2 else q

    # Extract number of matches requested.
    # Strategy: if the query contains a team + any number + score/match intent,
    # that number is the requested match count — no matter how the question is phrased.
    # EXCEPTION: if a specific date was already parsed, numbers belong to the date not match count.
    n_matches = None
    _score_intent = _re_sports.search(
        r'\b(?:matches?|games?|results?|fixtures?|scores?|scoreline)\b', q, _re_sports.IGNORECASE
    )
    _has_specific_date = (date != 'today' and date != 'yesterday')
    if _score_intent and not _has_specific_date:
        # Build a version of the query with date-adjacent numbers masked
        # e.g. "18 february" → mask "18" so it won't be counted as n_matches
        _MONTH_PAT = (r'(?:january|february|march|april|may|june|july|august|'
                      r'september|october|november|december|'
                      r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)')
        _q_masked = _re_sports.sub(
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+' + _MONTH_PAT +
            r'|' + _MONTH_PAT + r'\s+(\d{1,2})(?:st|nd|rd|th)?\b',
            'DATE', q, flags=_re_sports.IGNORECASE
        )
        # Extract any standalone number 1-20 from the masked query
        _nums = _re_sports.findall(r'\b([1-9]|1[0-9]|20)\b', _q_masked)
        # Exclude years
        _valid = [int(n) for n in _nums
                  if not _re_sports.match(r'20\d\d|19\d\d', n)
                  and int(n) <= 20]
        if _valid:
            n_matches = _valid[0]


    # Extract a year filter if present (e.g. "2023 world cup", "2022 ipl")
    _year_filter = None
    _yr_match = _re_sports.search(r'\b(20[012]\d)\b', q)
    if _yr_match:
        _yr_candidate = int(_yr_match.group(1))
        # Only treat as year filter if it's a plausible sports season year
        # and not already embedded in a YYYY-MM-DD date string
        if 2000 <= _yr_candidate <= 2030 and not _re_sports.search(r'\d{4}-\d{2}-\d{2}', q):
            _year_filter = _yr_candidate

    return sport_query, date, team_filter, n_matches, date_from, date_to, _league_explicit, _year_filter


def _cricapi_get_last_match(team_name: str) -> list:
    key = _key("CRICAPI_KEY")
    if not key:
        return []

    tf = team_name.lower()

    def _build_match_block(m, fallback_date=""):
        teams  = m.get("teams") or []
        t1     = teams[0] if len(teams) > 0 else "TBA"
        t2     = teams[1] if len(teams) > 1 else "TBA"
        scores = m.get("score") or []
        t1_inn = next((s for s in scores if t1.lower()[:4] in (s.get("inning") or "").lower()), None)
        t2_inn = next((s for s in scores if t2.lower()[:4] in (s.get("inning") or "").lower()), None)
        s1 = f"{t1_inn['r']}/{t1_inn['w']} ({t1_inn['o']} ov)" if t1_inn else None
        s2 = f"{t2_inn['r']}/{t2_inn['w']} ({t2_inn['o']} ov)" if t2_inn else None
        result = m.get("status") or "Match Ended"
        venue  = m.get("venue") or "N/A"
        dt     = (m.get("dateTimeGMT") or fallback_date)[:16].replace("T", "  ")

        # Build innings detail lines for all innings (shown when summary unavailable)
        def _clean_inn(name):
            if "," in name:
                parts = name.split(",")
                for p in parts:
                    if "inning" in p.lower():
                        return p.strip()
                return parts[-1].strip()
            return name
        inn_lines = [
            f"   📊 {_clean_inn(s.get('inning','?'))}: {s.get('r','–')}/{s.get('w','–')} ({s.get('o','–')} ov)"
            for s in scores
        ]

        lines = [
            "Most Recent Match:",
            f"🏏 {t1} vs {t2}",
        ]
        # Show per-team summary if available, else show innings detail lines
        if s1 or s2:
            lines.append(f"   {t1}: {s1 or '–'}")
            lines.append(f"   {t2}: {s2 or '–'}")
        lines += [
            f"   Status: Match Ended",
            f"   Series: {m.get('name', 'Cricket')}",
            f"   Match Type: {(m.get('matchType') or 'N/A').upper()}",
            f"   Venue: {m.get('venue', 'N/A')}",
            f"   Date: {dt}",
        ]
        lines += inn_lines
        lines.append(f"   🏅 Result: {result}")
        return "\n".join(lines)

    def _is_completed(m):
        return (m.get("matchEnded") or
                "won" in (m.get("status") or "").lower() or
                "draw" in (m.get("status") or "").lower() or
                "tied" in (m.get("status") or "").lower() or
                "no result" in (m.get("status") or "").lower())

    # ── Step 1: currentMatches (shared cache — not re-fetched if called recently) ──
    matches, _cricapi_alive = _get_current_matches(key)

    # Domestic/franchise series to exclude for national team queries
    _DOMESTIC_EXCL = {
        "legends league", "ranji", "sheffield shield", "county", "domestic",
        "franchise", "provincial", "regional", "club", "qualifier", "emerging",
        "under-19", "u19", "u-19", "women", "a team", "a tour",
        "legends", "masters", "veteran",
    }

    if matches:
        team_m = [m for m in matches if any(_cricket_team_matches(tf, t) for t in (m.get("teams") or []))]
        # For national teams, exclude domestic/franchise series
        if tf in CRICKET_NATIONAL_TEAMS:
            team_m = [m for m in team_m
                      if not any(kw in (m.get("name") or "").lower() for kw in _DOMESTIC_EXCL)]
        completed = [m for m in team_m if _is_completed(m)]
        print(f"  [cricapi_last] currentMatches: {len(matches)} total, "
              f"{len(team_m)} for '{team_name}' (national-only), {len(completed)} completed")
        if completed:
            completed.sort(key=lambda m: m.get("dateTimeGMT") or "", reverse=True)
            print(f"  [cricapi_last] Found via currentMatches: {completed[0].get('name')}")
            return [_build_match_block(completed[0])]
    # _cricapi_alive already set by _get_current_matches return value

    # ── Step 2: Sportmonks — bilateral series (no ICC tournaments on free tier) ──
    # Note: Sportmonks free tier does NOT include ICC events (WC, Champions Trophy).
    # So we run it in parallel with CricAPI series scan and pick the newest result.
    print(f"  [cricapi_last] Trying Sportmonks for bilateral history: '{team_name}'")
    sm_result = _sportmonks_get_last_match(team_name)
    sm_date = ""
    if sm_result:
        # Extract date from result block for comparison
        for line in sm_result[0].split("\n"):
            if line.strip().startswith("Date:"):
                sm_date = line.split("Date:")[-1].strip()[:10]
                break
        print(f"  [cricapi_last] Sportmonks latest: {sm_date}")

    # ── Step 3: CricAPI series scan — catches ICC tournaments Sportmonks misses ──
    # Uses cache + rate-limit delays to avoid being blocked on rapid requests.
    import time as _time

    _cache_key = team_name.lower().strip()
    _now = _time.time()
    all_completed = None

    # Return cached result if fresh (within 1 hour)
    if _cache_key in _CRICAPI_CACHE:
        _ts, _cached = _CRICAPI_CACHE[_cache_key]
        if _now - _ts < _CRICAPI_CACHE_TTL:
            print(f"  [cricapi_last] Using cached result for '{team_name}'")
            all_completed = _cached

    if all_completed is None:
        # Skip series scan if daily quota exhausted or Step 1 blocked
        if not _cricapi_quota_ok() or not _cricapi_alive:
            reason = "daily quota exhausted" if not _cricapi_quota_ok() else "API unavailable"
            print(f"  [cricapi_last] Skipping series scan ({reason}), using Sportmonks")
            if sm_result:
                return sm_result
            return []

        all_series = []
        seen_ids   = set()

        # 1 bilateral search call — skip if quota exhausted
        if not _cricapi_quota_ok():
            print(f"  [cricapi_last] Quota exhausted mid-scan, using Sportmonks")
            if sm_result: return sm_result
            return []
        series_data = _safe_get(f"{CRICAPI_BASE}/series",
                                params={"apikey": key, "offset": 0, "search": team_name},
                                timeout=20)
        if series_data and series_data.get("status") == "success":
            for s in (series_data.get("data") or []):
                if s.get("id") not in seen_ids:
                    seen_ids.add(s.get("id"))
                    all_series.append(s)
        elif series_data and series_data.get("status") == "failure":
            _r = (series_data.get("reason") or "").lower()
            if any(x in _r for x in ["exceeded", "limit", "quota", "daily"]):
                _cricapi_mark_exhausted(_r)
                if sm_result: return sm_result
                return []
        _time.sleep(0.5)

        # ICC tournament searches — covers World Cup, Champions Trophy etc.
        for icc_term in ["t20 world cup", "icc men", "champions trophy"]:
            if not _cricapi_quota_ok():
                break
            icc_data = _safe_get(f"{CRICAPI_BASE}/series",
                                 params={"apikey": key, "offset": 0, "search": icc_term},
                                 timeout=20)
            if icc_data and icc_data.get("status") == "success":
                for s in (icc_data.get("data") or []):
                    if s.get("id") not in seen_ids:
                        seen_ids.add(s.get("id"))
                        all_series.append(s)
            _time.sleep(0.5)

        print(f"  [cricapi_last] CricAPI series scan: {len(all_series)} series")

        exclude = {"lions", "under", "u19", "u-19", "women", "emerging", "shadow", "under-19"}
        national = [s for s in all_series
                    if not any(ex in (s.get("name") or "").lower() for ex in exclude)]

        # Priority order:
        # 1. ICC World Cup / Champions Trophy / major ICC events (most likely to be recent)
        # 2. Other international series for the team
        # 3. Domestic/franchise tournaments
        _icc_keywords = {"world cup", "champions trophy", "world twenty20", "icc men",
                         "icc women", "icc t20", "world test championship"}
        icc_series   = [s for s in national
                        if any(kw in (s.get("name") or "").lower() for kw in _icc_keywords)]
        other_series = [s for s in national if s not in icc_series]
        search_order = icc_series + other_series + [s for s in all_series if s not in national]

        print(f"  [cricapi_last] Prioritised: {len(icc_series)} ICC, {len(other_series)} bilateral")

        all_completed = []
        # Scan top 8 series (ICC events bumped to front so they're always checked)
        for s in search_order[:8]:
            sid = s.get("id")
            if not sid:
                continue
            if not _cricapi_quota_ok():
                break
            s_info = _safe_get(f"{CRICAPI_BASE}/series_info",
                               params={"apikey": key, "id": sid}, timeout=20)
            all_m = ((s_info or {}).get("data") or {}).get("matchList") or []
            team_m = [m for m in all_m if any(_cricket_team_matches(tf, t) for t in (m.get("teams") or []))]
            completed = [m for m in team_m if _is_completed(m)]
            for m in completed:
                m["_series_name"] = s.get("name", "")
            all_completed.extend(completed)
            _time.sleep(0.3)

        # Only cache non-empty results — don't cache API failures/blocks
        if all_completed:
            _CRICAPI_CACHE[_cache_key] = (_now, list(all_completed))
            print(f"  [cricapi_last] Cached {len(all_completed)} matches for '{team_name}' (1hr)")
        else:
            print(f"  [cricapi_last] Not caching empty result (API may be blocked)")

        if all_completed:
            all_completed.sort(key=lambda m: m.get("dateTimeGMT") or "", reverse=True)
            best = all_completed[0]
            cric_date = (best.get("dateTimeGMT") or "")[:10]
            print(f"  [cricapi_last] CricAPI series scan found: {best.get('name')} ({cric_date})")

            # Compare with Sportmonks result — return whichever is newer
            if sm_result and sm_date >= cric_date:
                print(f"  [cricapi_last] Sportmonks is newer ({sm_date} >= {cric_date}) — using Sportmonks")
                return sm_result

            # CricAPI is newer (or Sportmonks had nothing) — fetch full scorecard
            mid = best.get("id")
            if mid:
                m_info = _safe_get(f"{CRICAPI_BASE}/match_info",
                                   params={"apikey": key, "id": mid}, timeout=20)
                full_m = (m_info or {}).get("data")
                if full_m:
                    best = full_m

            print(f"  [cricapi_last] Using CricAPI result ({cric_date})")
            return [_build_match_block(best)]

    # CricAPI found nothing — fall back to Sportmonks if it had something
    if sm_result:
        print(f"  [cricapi_last] CricAPI empty — using Sportmonks result ({sm_date})")
        return sm_result

    print(f"  [cricapi_last] No completed match found for '{team_name}'")
    return []


def _tsdb_get_last_match(team_name: str, sport: str = "") -> list:
    # Use longer timeout for national teams (TSDB can be slow)
    _tsdb_timeout = 20
    data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
                     {"t": team_name.strip().title()}, timeout=_tsdb_timeout)
    if data is None:
        # Retry once on timeout
        import time as _t; _t.sleep(1)
        data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
                         {"t": team_name.strip().title()}, timeout=_tsdb_timeout)
    teams_found = (data or {}).get("teams") or []
    if not teams_found:
        short = _re_sports.sub(
            r'^(fc|afc|sc|ac|ss|as|rc|vfb|vfl|rb|bv|sv|cf|cd|ud|sd|rcd)\s+',
            '', team_name.lower()
        ).strip()
        if short != team_name.lower():
            data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
                             {"t": short.title()})
            teams_found = (data or {}).get("teams") or []

    if not teams_found:
        print(f"  [tsdb_last] No team found for '{team_name}'")
        return []

    # For national teams: always filter to Soccer/Football 
    # (TSDB may return cricket "India" before football "India")
    _is_natl_tsdb = _TEAM_LEAGUE_MAP.get(team_name.lower().strip()) == "International"
    _sport_filter = sport or ("Soccer" if _is_natl_tsdb else "")
    
    if _sport_filter:
        sport_l = _sport_filter.lower()
        matched = [t for t in teams_found
                   if sport_l in (t.get("strSport") or "").lower()
                   or sport_l in (t.get("strLeague") or "").lower()
                   or "football" in (t.get("strSport") or "").lower()]
        if matched:
            teams_found = matched
            print(f"  [tsdb_last] Sport filter '{_sport_filter}' → {teams_found[0].get('strTeam')} ({teams_found[0].get('strSport')})")
        else:
            print(f"  [tsdb_last] No '{_sport_filter}' entry for '{team_name}', using first result")
    elif sport:
        print(f"  [tsdb_last] No sport filter for '{team_name}', using first result")

    team_id = teams_found[0].get("idTeam")
    if not team_id:
        return []

    events_data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/eventslast.php",
                            {"id": team_id}, timeout=20)
    events = (events_data or {}).get("results") or []
    print(f"  [tsdb_last] '{team_name}' → {len(events)} recent events from TSDB")

    results = []
    for e in events:
        home_t     = e.get("strHomeTeam", "?")
        away_t     = e.get("strAwayTeam", "?")
        home_score = e.get("intHomeScore") if e.get("intHomeScore") is not None else "–"
        away_score = e.get("intAwayScore") if e.get("intAwayScore") is not None else "–"
        status     = _clean_status(e.get("strStatus") or e.get("strProgress") or "Full Time")
        league     = e.get("strLeague", "Unknown")
        season     = e.get("strSeason", "N/A")
        venue      = e.get("strVenue") or e.get("strStadium") or "N/A"
        dt         = f"{e.get('dateEvent', '?')}  {e.get('strTime', '??:??')}"
        results.append(_fmt(home_t, away_t, home_score, away_score,
                            status, league, season, venue, dt))

    return results[:1]  # TSDB eventslast.php is already newest-first, return only latest



def _apifootball_get_last_match(team_name: str, hint_league: str = "", target_date: str = "", max_results: int = 1) -> list:
    """
    Fetch most recent completed match for a football team across all competitions.
    Priority order:
      1. football-data.org  — top 13 leagues, cleanest data, most-searched teams
      2. ESPN               — 20+ leagues, no key needed
      3. apifootball.com    — 1,019 leagues, for obscure teams/leagues
    Cricket is handled separately — this function is football only.
    """
    import time as _time

    # ── 1. football-data.org (top 13 leagues) ─────────────────
    FOOTBALLDATA_BASE = "https://api.football-data.org/v4"
    fd_key = _key("FOOTBALLDATA_KEY")
    if fd_key:
        HINT_MAP = {
            "premier league": "PL",  "epl": "PL",         "england": "PL",
            "serie a":        "SA",  "italy": "SA",        "italian": "SA",
            "la liga":        "PD",  "spain": "PD",        "spanish": "PD",
            "bundesliga":     "BL1", "germany": "BL1",     "german": "BL1",
            "ligue 1":        "FL1", "france": "FL1",      "french": "FL1",
            "champions league": "CL","ucl": "CL",
            "eredivisie":     "DED", "netherlands": "DED",
            "championship":   "ELC",
            "primeira liga":  "PPL", "portugal": "PPL",
            "brasileirao":    "BSA", "brazil": "BSA",
            "copa libertadores": "CLI",
            "euros": "EC",           "world cup": "WC",
        }
        ALL_CODES = ["PL","SA","PD","BL1","FL1","CL","ELC","DED","PPL","BSA","CLI","EC","WC"]
        hint_code = next((v for k, v in HINT_MAP.items() if k in hint_league.lower()), None)
        fd_order  = ([hint_code] + [c for c in ALL_CODES if c != hint_code]) if hint_code else ALL_CODES

        FD_ALIASES = {
            "bayern munich":   "fc bayern münchen",   "munich": "münchen",
            "atletico madrid": "atlético de madrid",  "atletico": "atlético",
            "ac milan":        "milan",                "inter milan": "internazionale",
            "psg":             "paris saint-germain",  "paris sg": "paris saint-germain",
            "man united":      "manchester united",    "man utd": "manchester united",
            "man city":        "manchester city",      "spurs": "tottenham hotspur",
            "tottenham":       "tottenham hotspur",    "ajax": "afc ajax",
            "porto":           "fc porto",             "benfica": "sl benfica",
            "barca":           "fc barcelona",         "barcelona": "fc barcelona",
            "real madrid":     "real madrid cf",       "juventus": "juventus fc",
            "napoli":          "ssc napoli",            "roma": "as roma",
            "lazio":           "ss lazio",             "dortmund": "borussia dortmund",
            "bvb":             "borussia dortmund",    "leverkusen": "bayer 04 leverkusen",
            "gladbach":        "borussia mönchengladbach",
            "flamengo":        "cr flamengo",          "palmeiras": "se palmeiras",
            "sao paulo":       "são paulo fc",
        }
        tf_raw = team_name.lower().strip()
        tf_fd  = FD_ALIASES.get(tf_raw, tf_raw)

        def _fd_matches(t):
            name  = (t.get("name")      or "").lower()
            short = (t.get("shortName") or "").lower()
            tla   = (t.get("tla")       or "").lower()
            for q in {tf_fd, tf_raw}:
                words = q.split()
                # Always accept exact matches
                if q == name or q == short or q == tla:
                    return True
                if len(words) == 1:
                    w = words[0]
                    # Single word: must be the START of name or exact short/tla
                    # Prevents "barcelona" matching "rcd espanyol de barcelona"
                    if name.startswith(w) or short == w or tla == w:
                        return True
                    # Also accept if the alias resolves to a full name that matches
                    if tf_fd != tf_raw and (tf_fd == name or tf_fd == short):
                        return True
                else:
                    # Multi-word: all words must appear in name or short
                    if (all(w in name  for w in words) or
                            all(w in short for w in words)):
                        return True
            return False

        team_id = None
        for code in fd_order:
            data = _safe_get(f"{FOOTBALLDATA_BASE}/competitions/{code}/teams",
                             headers={"X-Auth-Token": fd_key}, timeout=15)
            if data is None:
                print(f"  [fd_last] Rate limit at /{code} — stopping")
                break
            for t in (data or {}).get("teams") or []:
                if _fd_matches(t):
                    team_id = t.get("id")
                    print(f"  [fd_last] Matched '{t.get('name')}' (id={team_id}) via /{code}")
                    break
            if team_id:
                break
            _time.sleep(0.15)

        if team_id:
            # Fetch more matches when filtering by competition
            fetch_limit = max(30, max_results)  # always fetch enough for H2H scanning
            _fd_params  = {"status": "FINISHED", "limit": fetch_limit}
            if target_date and target_date not in ("today", "yesterday"):
                # Fetch matches around the target date (±3 days window)
                import datetime as _dtt_fd
                try:
                    _td = _dtt_fd.date.fromisoformat(target_date)
                    _fd_params["dateFrom"] = (_td - _dtt_fd.timedelta(days=1)).isoformat()
                    _fd_params["dateTo"]   = (_td + _dtt_fd.timedelta(days=1)).isoformat()
                    del _fd_params["status"]  # date range doesn't need status filter
                    del _fd_params["limit"]
                except ValueError:
                    pass
            md = _safe_get(f"{FOOTBALLDATA_BASE}/teams/{team_id}/matches",
                           params=_fd_params,
                           headers={"X-Auth-Token": fd_key}, timeout=15)
            matches = sorted((md or {}).get("matches") or [],
                             key=lambda m: m.get("utcDate") or "", reverse=True)

            # If hint_league specifies a competition, filter to that comp only
            # football-data.org uses official names (e.g. "Primera Division" not "La Liga")
            if hint_league and matches:
                hl_lower = hint_league.lower()
                # Map common user-facing names to football-data.org official names
                FD_NAME_MAP = {
                    "la liga":          ["primera division", "la liga"],
                    "serie a":          ["serie a"],
                    "bundesliga":       ["bundesliga"],
                    "premier league":   ["premier league"],
                    "ligue 1":          ["ligue 1"],
                    "champions league": ["uefa champions league", "champions league"],
                    "europa league":    ["uefa europa league", "europa league"],
                    "conference":       ["uefa conference league", "conference league"],
                    "fa cup":           ["fa cup"],
                    "league cup":       ["efl cup", "league cup", "carabao cup"],
                    "copa del rey":     ["copa del rey"],
                    "coppa italia":     ["coppa italia"],
                    "dfb pokal":        ["dfb-pokal", "dfb pokal"],
                    "brasileirao":      ["série a", "serie a"],
                }
                # Get all acceptable competition name variants
                acceptable = FD_NAME_MAP.get(hl_lower, [hl_lower])
                def _comp_matches(m):
                    comp_name = (m.get("competition") or {}).get("name", "").lower()
                    return any(a in comp_name or comp_name in a for a in acceptable)

                comp_filtered = [m for m in matches if _comp_matches(m)]
                if comp_filtered:
                    matches = comp_filtered
                    print(f"  [fd_last] Filtered to {len(matches)} {hint_league} matches")
                else:
                    print(f"  [fd_last] No {hint_league} matches found for '{team_name}' — trying all competitions")
                    # Don't clear matches — fall through to return most recent overall

            if matches:
                _fd_results = []
                for m in matches:
                    home_t = ((m.get("homeTeam") or {}).get("shortName")
                              or (m.get("homeTeam") or {}).get("name", "?"))
                    away_t = ((m.get("awayTeam") or {}).get("shortName")
                              or (m.get("awayTeam") or {}).get("name", "?"))
                    sc     = (m.get("score") or {}).get("fullTime") or {}
                    hs     = sc.get("home") if sc.get("home") is not None else "–"
                    as_    = sc.get("away") if sc.get("away") is not None else "–"
                    comp   = (m.get("competition") or {}).get("name", "Unknown")
                    season = str((m.get("season") or {}).get("startDate", "")[:4])
                    venue  = m.get("venue") or "N/A"
                    dt     = (m.get("utcDate") or "")[:16].replace("T", "  ")
                    _fd_results.append(_fmt(home_t, away_t, hs, as_, "Full Time", comp, season, venue, dt))
                print(f"  [fd_last] Most recent: {((matches[0].get('homeTeam') or {}).get('shortName') or '?')} vs {((matches[0].get('awayTeam') or {}).get('shortName') or '?')} ({(matches[0].get('competition') or {}).get('name','?')}, {(matches[0].get('utcDate') or '')[:16]})")
                return _fd_results[:max_results]

        print(f"  [fd_last] '{team_name}' not in football-data.org top 13 leagues — trying ESPN")

    # ── 2. ESPN (20+ leagues, no key) ─────────────────────────
    espn_result = _espn_get_last_match(team_name, hint_league, target_date=target_date)
    if espn_result:
        return espn_result

    # ── 3. apifootball.com (1,019 leagues) ────────────────────
    import datetime
    if _key("APIFOOTBALL_COM_KEY"):
        hl         = hint_league.lower()
        tf         = team_name.lower().strip()

        # National team / International routing
        _is_intl_team = _TEAM_LEAGUE_MAP.get(tf) == "International"
        _is_intl_hint = "international" in hl or any(k in hl for k in [
            "world cup", "euro", "nations league", "copa america",
            "asian cup", "afcon", "gold cup", "friendly"
        ])
        if _is_intl_team or _is_intl_hint:
            league_ids = _INTL_IDS
            print(f"  [af_last] National team detected — searching international competitions")
        else:
            league_ids = next((v for k, v in AF_COM_HINT_MAP.items() if k in hl), AF_COM_DEFAULT_IDS)

        # If hint is a specific competition (not just a country/league), use only that comp
        SPECIFIC_COMPS = {
            "champions league": [3],
            "europa league":    [4],
            "conference":       [683],
            "fa cup":           [146],
            "league cup":       [147],
            "copa del rey":     [300],
            "coppa italia":     [205],
            "dfb pokal":        [172],
            "coupe de france":  [165],
            "world cup":        [6],
            "euros":            [5],   "euro 20": [5],
            "nations league":   [73, 74],
            "copa america":     [197],
            "asian cup":        [22],
            "afcon":            [36],
            "gold cup":         [23],
            "friendly":         [257],
        }
        for comp_kw, comp_ids in SPECIFIC_COMPS.items():
            if comp_kw in hl:
                league_ids = comp_ids
                print(f"  [af_last] Specific competition filter: {comp_kw} → league_ids={comp_ids}")
                break

        AF_ALIASES = {
            "inter milan":     ["inter", "internazionale"],
            "ac milan":        ["ac milan", "milan"],
            "man city":        ["manchester city"],    "man united": ["manchester united"],
            "man utd":         ["manchester united"],  "spurs": ["tottenham"],
            "tottenham":       ["tottenham hotspur", "tottenham"],
            "psg":             ["paris saint-germain", "psg"],
            "barca":           ["barcelona"],          "atletico madrid": ["atletico madrid"],
            "dortmund":        ["borussia dortmund"],  "bvb": ["borussia dortmund"],
            "leverkusen":      ["bayer leverkusen"],   "gladbach": ["b. monchengladbach"],
            "bayern munich":   ["fc bayern munich", "bayern munich"],
            "inter miami":     ["inter miami cf", "inter miami"],
        }
        search_names = AF_ALIASES.get(tf, [tf])

        def _team_in_event(e):
            home = (e.get("match_hometeam_name") or "").lower()
            away = (e.get("match_awayteam_name") or "").lower()
            for sn in search_names:
                words = sn.split()
                for side in (home, away):
                    if (sn == side
                            or side.startswith(sn)
                            or (len(words) > 1 and all(w in side for w in words))):
                        return True
            return False

        # Use target_date window if specified, else last 60 days
        if target_date and target_date not in ("today", "yesterday"):
            import datetime as _dtt_af
            try:
                _td_af = _dtt_af.date.fromisoformat(target_date)
                today  = (_td_af + _dtt_af.timedelta(days=1)).isoformat()
                since  = (_td_af - _dtt_af.timedelta(days=1)).isoformat()
            except ValueError:
                today = datetime.date.today().isoformat()
                since = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
        else:
            today = datetime.date.today().isoformat()
            since = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
        all_matches = []

        # National teams: search by team name directly (no league_id needed)
        # apifootball.com supports match_hometeam_name / match_awayteam_name params
        if _is_intl_team or _is_intl_hint:
            print(f"  [af_last] National team — searching by name across all leagues ({since} to {today})")
            _search_names = list(AF_ALIASES.get(tf, [tf]))
            # Also try title-case (Argentina, England, Germany etc.)
            _search_names += [n.title() for n in _search_names if n.title() not in _search_names]
            for _sn in _search_names[:3]:
                for _side in ["match_hometeam_name", "match_awayteam_name"]:
                    _evs = _afcom_get({"action": "get_events", "from": since,
                                       "to": today, _side: _sn})
                    _fin = [e for e in _evs
                            if _team_in_event(e) and e.get("match_status") in (
                                "Finished", "FT", "Match Finished", "After ET",
                                "After Pen", "AET", "PEN", "finished")]
                    all_matches.extend(_fin)
            # Deduplicate by match_id
            _seen_ids = set()
            _deduped = []
            for e in all_matches:
                mid = e.get("match_id") or f"{e.get('match_date')}{e.get('match_hometeam_name')}"
                if mid not in _seen_ids:
                    _seen_ids.add(mid)
                    _deduped.append(e)
            all_matches = _deduped
        else:
            print(f"  [af_last] Searching {len(league_ids)} leagues for '{team_name}' ({since} to {today})")
            for lid in league_ids:
                events   = _afcom_get({"action": "get_events", "from": since,
                                       "to": today, "league_id": lid})
                finished = [e for e in events
                            if _team_in_event(e) and e.get("match_status") in (
                                "Finished", "FT", "Match Finished", "After ET",
                                "After Pen", "AET", "PEN", "finished")]
                all_matches.extend(finished)

        if all_matches:
            all_matches.sort(key=lambda e: e.get("match_date") or "", reverse=True)
            _results_out = []
            for _m in all_matches[:max(max_results, 1)]:
                ht     = _m.get("match_hometeam_name", "?")
                at     = _m.get("match_awayteam_name", "?")
                hs     = _m.get("match_hometeam_score", "–") or "–"
                as_    = _m.get("match_awayteam_score", "–") or "–"
                comp   = _m.get("league_name", "Unknown")
                season = _m.get("match_season", "")
                venue  = _m.get("match_stadium") or "N/A"
                dt     = f"{_m.get('match_date','?')}  {_m.get('match_time','??:??')}"
                _results_out.append(_fmt(ht, at, hs, as_, "Full Time", comp, season, venue, dt))
            best = all_matches[0]
            print(f"  [af_last] Found: {best.get('match_hometeam_name')} {best.get('match_hometeam_score')}-{best.get('match_awayteam_score')} {best.get('match_awayteam_name')} ({best.get('league_name')}, {best.get('match_date')})")
            return _results_out

        print(f"  [af_last] No matches found for '{team_name}'")

    return []

def _tsdb_search_by_teams(team1: str, team2: str, date: str) -> list:
    results = []
    seen_keys = set()

    for team_name in [team1, team2]:
        data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/searchteams.php",
                         {"t": team_name})
        teams_found = (data or {}).get("teams") or []
        if not teams_found:
            continue

        team_id = teams_found[0].get("idTeam")
        if not team_id:
            continue

        events_data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/eventslast.php",
                                {"id": team_id})
        events = (events_data or {}).get("results") or []

        next_data = _safe_get(f"{TSDB_BASE}/{_tsdb_key()}/eventsnext.php",
                              {"id": team_id})
        events += (next_data or {}).get("events") or []

        other_team = team2 if team_name == team1 else team1

        for e in events:
            home = (e.get("strHomeTeam") or "").lower()
            away = (e.get("strAwayTeam") or "").lower()
            # Strict match: the OTHER team must appear in home OR away
            # Use full team name match OR all words match on a SINGLE side
            # Do NOT split words across home+away (avoids "madrid" matching "atletico madrid")
            _ot = other_team.lower()
            _ot_words = [w for w in _ot.split() if len(w) > 3]
            _in_home = (_ot in home) or (_ot_words and all(w in home for w in _ot_words))
            _in_away = (_ot in away) or (_ot_words and all(w in away for w in _ot_words))
            if not (_in_home or _in_away):
                continue

            key = f"{e.get('strHomeTeam')} vs {e.get('strAwayTeam')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            home_t     = e.get("strHomeTeam", "?")
            away_t     = e.get("strAwayTeam", "?")
            home_score = e.get("intHomeScore") if e.get("intHomeScore") is not None else "–"
            away_score = e.get("intAwayScore") if e.get("intAwayScore") is not None else "–"
            status     = _clean_status(e.get("strStatus") or e.get("strProgress") or "Scheduled")
            league     = e.get("strLeague", "Unknown Competition")
            season     = e.get("strSeason", "N/A")
            venue      = e.get("strVenue") or e.get("strStadium") or "N/A"
            dt         = f"{e.get('dateEvent', date)}  {e.get('strTime', '??:??')}"
            results.append(_fmt(home_t, away_t, home_score, away_score,
                                status, league, season, venue, dt))
    return results


def _get_matches_in_range(team_name: str, date_from: str, date_to: str,
                           league_hint: str = "") -> list:
    """
    Return all completed matches for a team between date_from and date_to.
    Optionally filtered to a specific league.
    Uses football-data.org (primary) → apifootball.com (fallback).
    """
    import time as _tm

    results = []
    comp_kw  = league_hint.lower().strip()

    # ── football-data.org ─────────────────────────────────────
    fd_key = _key("FOOTBALLDATA_KEY")
    if fd_key:
        FOOTBALLDATA_BASE = "https://api.football-data.org/v4"
        HINT_MAP = {
            "premier league":"PL","serie a":"SA","la liga":"PD","bundesliga":"BL1",
            "ligue 1":"FL1","champions league":"CL","ucl":"CL","eredivisie":"DED",
            "championship":"ELC","primeira liga":"PPL","brasileirao":"BSA",
        }
        ALL_CODES = ["PL","SA","PD","BL1","FL1","CL","ELC","DED","PPL","BSA","CLI","EC","WC"]
        hint_code = next((v for k, v in HINT_MAP.items() if k in comp_kw), None)
        fd_order  = ([hint_code] + [c for c in ALL_CODES if c != hint_code]) if hint_code else ALL_CODES

        FD_ALIASES = {
            "bayern munich":"fc bayern münchen","inter milan":"internazionale",
            "ac milan":"milan","man city":"manchester city",
            "man united":"manchester united","man utd":"manchester united",
            "psg":"paris saint-germain","barca":"fc barcelona",
            "barcelona":"fc barcelona","real madrid":"real madrid cf",
            "atletico madrid":"atlético de madrid","spurs":"tottenham hotspur",
            "tottenham":"tottenham hotspur","dortmund":"borussia dortmund",
        }
        tf_raw = team_name.lower().strip()
        tf_fd  = FD_ALIASES.get(tf_raw, tf_raw)

        def _ok(t):
            name  = (t.get("name") or "").lower()
            short = (t.get("shortName") or "").lower()
            tla   = (t.get("tla") or "").lower()
            for q in {tf_fd, tf_raw}:
                words = q.split()
                if q == name or q == short or q == tla: return True
                if len(words)==1 and (name.startswith(q) or short==q or tla==q): return True
                if len(words)>1 and (all(w in name for w in words) or all(w in short for w in words)): return True
            return False

        team_id = None
        for code in fd_order:
            data = _safe_get(f"{FOOTBALLDATA_BASE}/competitions/{code}/teams",
                             headers={"X-Auth-Token": fd_key}, timeout=15)
            if data is None: break
            for t in (data or {}).get("teams") or []:
                if _ok(t):
                    team_id = t.get("id")
                    break
            if team_id: break
            _tm.sleep(0.15)

        if team_id:
            md = _safe_get(f"{FOOTBALLDATA_BASE}/teams/{team_id}/matches",
                           params={"status":"FINISHED","dateFrom":date_from,"dateTo":date_to,"limit":50},
                           headers={"X-Auth-Token": fd_key}, timeout=15)
            matches = sorted((md or {}).get("matches") or [],
                             key=lambda m: m.get("utcDate") or "", reverse=True)
            if comp_kw:
                matches = [m for m in matches
                           if comp_kw in (m.get("competition") or {}).get("name","").lower()
                           or (m.get("competition") or {}).get("name","").lower() in comp_kw]
            for m in matches:
                home_t = ((m.get("homeTeam") or {}).get("shortName") or
                          (m.get("homeTeam") or {}).get("name","?"))
                away_t = ((m.get("awayTeam") or {}).get("shortName") or
                          (m.get("awayTeam") or {}).get("name","?"))
                sc     = (m.get("score") or {}).get("fullTime") or {}
                hs     = sc.get("home") if sc.get("home") is not None else "–"
                as_    = sc.get("away") if sc.get("away") is not None else "–"
                comp   = (m.get("competition") or {}).get("name","Unknown")
                season = str((m.get("season") or {}).get("startDate","")[:4])
                venue  = m.get("venue") or "N/A"
                dt     = (m.get("utcDate") or "")[:16].replace("T","  ")
                results.append(_fmt(home_t, away_t, hs, as_, "Full Time", comp, season, venue, dt))

    # ── apifootball.com fallback ──────────────────────────────
    if not results and _key("APIFOOTBALL_COM_KEY"):
        hl         = comp_kw
        league_ids = next((v for k, v in AF_COM_HINT_MAP.items() if k in hl), AF_COM_DEFAULT_IDS)
        SPECIFIC   = {
            "champions league":[3],"europa league":[4],"conference":[683],
            "fa cup":[146],"league cup":[147],"copa del rey":[300],
            "coppa italia":[205],"dfb pokal":[172],
        }
        for kw, ids in SPECIFIC.items():
            if kw in hl: league_ids = ids; break

        tf = team_name.lower().strip()
        AF_AL = {
            "inter milan":["inter","internazionale"],"ac milan":["ac milan","milan"],
            "man city":["manchester city"],"man united":["manchester united"],
            "psg":["paris saint-germain"],"barca":["barcelona"],
            "atletico madrid":["atletico madrid"],"dortmund":["borussia dortmund"],
            "bayern munich":["fc bayern munich","bayern munich"],
        }
        search_names = AF_AL.get(tf, [tf])

        def _af_ok(e):
            home = (e.get("match_hometeam_name") or "").lower()
            away = (e.get("match_awayteam_name") or "").lower()
            for sn in search_names:
                words = sn.split()
                for side in (home, away):
                    if (sn==side or side.startswith(sn) or
                            (len(words)>1 and all(w in side for w in words))): return True
            return False

        for lid in league_ids:
            evs = _afcom_get({"action":"get_events","from":date_from,"to":date_to,"league_id":lid})
            for e in [ev for ev in evs if _af_ok(ev) and ev.get("match_status")=="Finished"]:
                ht  = e.get("match_hometeam_name","?")
                at  = e.get("match_awayteam_name","?")
                hs  = e.get("match_hometeam_score","–") or "–"
                as_ = e.get("match_awayteam_score","–") or "–"
                lg  = e.get("league_name","Unknown")
                dt  = f"{e.get('match_date','?')}  {e.get('match_time','??:??')}"
                results.append(_fmt(ht, at, hs, as_, "Full Time", lg,
                                    e.get("match_season",""), e.get("match_stadium","N/A"), dt))

    # Sort newest first, deduplicate
    seen, unique = set(), []
    for r in results:
        if r not in seen:
            seen.add(r); unique.append(r)
    return unique


def _get_team_recent_matches(team_name: str, n: int, league_hint: str = "") -> list:
    """
    Return the last N completed matches for a team across all competitions.
    Optionally filter to a specific league/competition via league_hint.
    Sources: football-data.org → ESPN → apifootball.com
    """
    import datetime as _dt, time as _tm

    results = []
    comp_filter_kw = league_hint.lower().strip()

    # Detect national team for special routing
    _is_natl_team = (_TEAM_LEAGUE_MAP.get(team_name.lower().strip()) == "International"
                     or "international" in comp_filter_kw)

    # ── football-data.org ─────────────────────────────────────
    fd_key = _key("FOOTBALLDATA_KEY")
    if fd_key and not results:
        FOOTBALLDATA_BASE = "https://api.football-data.org/v4"
        HINT_MAP = {
            "premier league": "PL",  "epl": "PL",
            "serie a": "SA",         "la liga": "PD",
            "bundesliga": "BL1",     "ligue 1": "FL1",
            "champions league": "CL","ucl": "CL",
            "eredivisie": "DED",     "championship": "ELC",
            "primeira liga": "PPL",  "brasileirao": "BSA",
            "copa libertadores": "CLI",
            "world cup": "WC",       "euros": "EC",  "euro": "EC",
            "international": "WC",
        }
        # National teams: search WC + EC first, then all
        ALL_CODES = ["PL","SA","PD","BL1","FL1","CL","ELC","DED","PPL","BSA","CLI","EC","WC"]
        INTL_CODES = ["WC","EC","CLI"]  # World Cup, Euros, Copa Libertadores
        hint_code = next((v for k, v in HINT_MAP.items() if k in comp_filter_kw), None)
        if _is_natl_team:
            fd_order = INTL_CODES + [c for c in ALL_CODES if c not in INTL_CODES]
        else:
            fd_order  = ([hint_code] + [c for c in ALL_CODES if c != hint_code]) if hint_code else ALL_CODES

        FD_ALIASES = {
            "bayern munich": "fc bayern münchen", "inter milan": "internazionale",
            "ac milan": "milan", "man city": "manchester city",
            "man united": "manchester united", "man utd": "manchester united",
            "psg": "paris saint-germain", "barca": "fc barcelona",
            "barcelona": "fc barcelona", "real madrid": "real madrid cf",
            "atletico madrid": "atlético de madrid", "spurs": "tottenham hotspur",
            "tottenham": "tottenham hotspur", "dortmund": "borussia dortmund",
            "leverkusen": "bayer 04 leverkusen",
        }
        tf_raw = team_name.lower().strip()
        tf_fd  = FD_ALIASES.get(tf_raw, tf_raw)

        def _fd_name_ok(t):
            name  = (t.get("name") or "").lower()
            short = (t.get("shortName") or "").lower()
            tla   = (t.get("tla") or "").lower()
            for q in {tf_fd, tf_raw}:
                words = q.split()
                if q == name or q == short or q == tla: return True
                if len(words)==1 and (name.startswith(q) or short==q or tla==q): return True
                if len(words)>1 and (all(w in name for w in words) or all(w in short for w in words)): return True
            return False

        team_id = None
        for code in fd_order:
            data = _safe_get(f"{FOOTBALLDATA_BASE}/competitions/{code}/teams",
                             headers={"X-Auth-Token": fd_key}, timeout=15)
            if data is None: break
            for t in (data or {}).get("teams") or []:
                if _fd_name_ok(t):
                    team_id = t.get("id")
                    print(f"  [recent] fd matched '{t.get('name')}' (id={team_id}) via /{code}")
                    break
            if team_id: break
            _tm.sleep(0.15)

        if team_id:
            limit = max(n * 3, 30) if comp_filter_kw else max(n, 10)
            md = _safe_get(f"{FOOTBALLDATA_BASE}/teams/{team_id}/matches",
                           params={"status": "FINISHED", "limit": limit},
                           headers={"X-Auth-Token": fd_key}, timeout=15)
            matches = sorted((md or {}).get("matches") or [],
                             key=lambda m: m.get("utcDate") or "", reverse=True)
            if comp_filter_kw:
                matches = [m for m in matches
                           if comp_filter_kw in (m.get("competition") or {}).get("name","").lower()
                           or (m.get("competition") or {}).get("name","").lower() in comp_filter_kw]
            for m in matches[:n]:
                home_t = ((m.get("homeTeam") or {}).get("shortName") or
                          (m.get("homeTeam") or {}).get("name","?"))
                away_t = ((m.get("awayTeam") or {}).get("shortName") or
                          (m.get("awayTeam") or {}).get("name","?"))
                sc     = (m.get("score") or {}).get("fullTime") or {}
                hs     = sc.get("home") if sc.get("home") is not None else "–"
                as_    = sc.get("away") if sc.get("away") is not None else "–"
                comp   = (m.get("competition") or {}).get("name","Unknown")
                season = str((m.get("season") or {}).get("startDate","")[:4])
                venue  = m.get("venue") or "N/A"
                dt     = (m.get("utcDate") or "")[:16].replace("T","  ")
                results.append(_fmt(home_t, away_t, hs, as_, "Full Time", comp, season, venue, dt))

    # ── ESPN fallback ─────────────────────────────────────────
    if len(results) < n:
        needed = n - len(results)
        slugs  = next((v for k, v in ESPN_HINT_SLUGS.items() if k in comp_filter_kw), None) or DEFAULT_ESPN_SLUGS
        ESPN_ALIASES = {
            "inter milan": ["internazionale","inter"], "ac milan": ["ac milan","milan"],
            "man city": ["manchester city"], "man united": ["manchester united"],
            "man utd": ["manchester united"], "spurs": ["tottenham hotspur","tottenham"],
            "tottenham": ["tottenham hotspur"], "psg": ["paris saint-germain"],
            "barca": ["barcelona"], "atletico madrid": ["atletico de madrid"],
            "dortmund": ["borussia dortmund"], "bvb": ["borussia dortmund"],
            "leverkusen": ["bayer leverkusen"], "bayern munich": ["fc bayern munich","bayern munich"],
        }
        tf = team_name.lower().strip()
        search_names = ESPN_ALIASES.get(tf, [tf])

        for slug in slugs:
            if len(results) >= n: break
            teams_data = _safe_get(f"{ESPN_BASE}/{slug}/teams", headers=ESPN_HDR, timeout=12)
            league_teams = ((teams_data or {}).get("sports") or [{}])[0]                           .get("leagues",[{}])[0].get("teams",[])
            team_id = None
            for t in league_teams:
                name = (t.get("team",{}).get("displayName") or "").lower()
                abbr = (t.get("team",{}).get("abbreviation") or "").lower()
                for sn in search_names:
                    words = sn.split()
                    if (sn==name or sn==abbr or name.startswith(sn) or
                            (len(words)>1 and all(w in name for w in words))):
                        team_id = t.get("team",{}).get("id"); break
                if team_id: break
            if not team_id: continue
            sched = _safe_get(f"{ESPN_BASE}/{slug}/teams/{team_id}/schedule",
                              headers=ESPN_HDR, timeout=12)
            events = sorted(
                [e for e in (sched or {}).get("events") or []
                 if (e.get("competitions") or [{}])[0].get("status",{}).get("type",{}).get("completed",False)],
                key=lambda e: e.get("date",""), reverse=True
            )
            for e in events:
                if len(results) >= n: break
                comp = (e.get("competitions") or [{}])[0]
                teams = comp.get("competitors") or []
                if len(teams) < 2: continue
                home = next((t for t in teams if t.get("homeAway")=="home"), teams[0])
                away = next((t for t in teams if t.get("homeAway")=="away"), teams[1])
                ht   = home.get("team",{}).get("displayName","?")
                at   = away.get("team",{}).get("displayName","?")
                hs   = _espn_parse_score(home.get("score"))
                as_  = _espn_parse_score(away.get("score"))
                lg   = ESPN_SLUG_MAP.get(slug, slug)
                if comp_filter_kw and comp_filter_kw not in lg.lower(): continue
                venue = (comp.get("venue") or {}).get("fullName","N/A")
                dt    = e.get("date","")[:16].replace("T","  ")
                results.append(_fmt(ht, at, hs, as_, "Full Time", lg, "", venue, dt))

    # ── apifootball.com fallback ──────────────────────────────
    if len(results) < n and _key("APIFOOTBALL_COM_KEY"):
        needed = n - len(results)
        hl     = comp_filter_kw
        if _is_natl_team or "international" in hl:
            league_ids = _INTL_IDS
        else:
            league_ids = next((v for k, v in AF_COM_HINT_MAP.items() if k in hl), AF_COM_DEFAULT_IDS)
        SPECIFIC = {
            "champions league":[3], "europa league":[4], "conference":[683],
            "fa cup":[146], "league cup":[147], "copa del rey":[300],
            "coppa italia":[205], "dfb pokal":[172],
            "world cup":[6], "euros":[5], "euro":[5],
            "nations league":[73,74], "copa america":[197],
            "asian cup":[22], "afcon":[36], "gold cup":[23], "friendly":[257],
        }
        for kw, ids in SPECIFIC.items():
            if kw in hl: league_ids = ids; break

        tf = team_name.lower().strip()
        AF_ALIASES = {
            "inter milan":["inter","internazionale"], "ac milan":["ac milan","milan"],
            "man city":["manchester city"], "man united":["manchester united"],
            "man utd":["manchester united"], "psg":["paris saint-germain"],
            "barca":["barcelona"], "atletico madrid":["atletico madrid"],
            "dortmund":["borussia dortmund"], "bvb":["borussia dortmund"],
            "leverkusen":["bayer leverkusen"], "bayern munich":["fc bayern munich","bayern munich"],
        }
        search_names = AF_ALIASES.get(tf, [tf])

        def _af_team_ok(e):
            home = (e.get("match_hometeam_name") or "").lower()
            away = (e.get("match_awayteam_name") or "").lower()
            for sn in search_names:
                words = sn.split()
                for side in (home, away):
                    if (sn==side or side.startswith(sn) or
                            (len(words)>1 and all(w in side for w in words))): return True
            return False

        today = _dt.date.today().isoformat()
        since = (_dt.date.today() - _dt.timedelta(days=180)).isoformat()
        all_matches = []
        for lid in league_ids:
            evs = _afcom_get({"action":"get_events","from":since,"to":today,"league_id":lid})
            finished = [e for e in evs if _af_team_ok(e) and e.get("match_status")=="Finished"]
            all_matches.extend(finished)
        all_matches.sort(key=lambda e: e.get("match_date") or "", reverse=True)
        seen = set(results)
        for e in all_matches:
            if len(results) >= n: break
            ht  = e.get("match_hometeam_name","?")
            at  = e.get("match_awayteam_name","?")
            hs  = e.get("match_hometeam_score","–") or "–"
            as_ = e.get("match_awayteam_score","–") or "–"
            lg  = e.get("league_name","Unknown")
            dt  = f"{e.get('match_date','?')}  {e.get('match_time','??:??')}"
            blk = _fmt(ht, at, hs, as_, "Full Time", lg, e.get("match_season",""), e.get("match_stadium","N/A"), dt)
            if blk not in seen:
                results.append(blk)
                seen.add(blk)

    return results[:n]


# ── Cricket filter helpers ─────────────────────────────────────────────────

_CRICKET_DOMESTIC_EXCL = {
    "legends league", "ranji", "sheffield shield", "county", "domestic",
    "franchise", "provincial", "regional", "club", "qualifier", "emerging",
    "under-19", "u19", "u-19", "women", "a team", "a tour",
    "legends", "masters", "veteran",
}

def _cricket_is_national_match(m: dict, team: str) -> bool:
    """Return True only if match involves the actual national team (not A/Women/Legends)."""
    tf = team.lower()
    if tf not in CRICKET_NATIONAL_TEAMS:
        return True  # franchise teams — no exclusion needed
    series = (m.get("name") or "").lower()
    if any(kw in series for kw in _CRICKET_DOMESTIC_EXCL):
        return False
    return any(_cricket_team_matches(tf, t) for t in (m.get("teams") or []))


def _cricapi_get_matches_by_date(team_name: str, target_date: str) -> list:
    """
    Search CricAPI currentMatches + series scan for a specific date (±1 day).
    Returns formatted match blocks.
    """
    import datetime as _dtt, time as _tm
    key = _key("CRICAPI_KEY")
    if not key or not _cricapi_quota_ok():
        return []

    tf = team_name.lower().strip()
    try:
        td  = _dtt.date.fromisoformat(target_date)
        d_from = (td - _dtt.timedelta(days=1)).isoformat()
        d_to   = (td + _dtt.timedelta(days=1)).isoformat()
    except ValueError:
        return []

    results = []

    # Step 1: currentMatches window
    matches, _ = _get_current_matches(key)
    for m in matches:
        m_date = (m.get("dateTimeGMT") or "")[:10]
        if d_from <= m_date <= d_to and _cricket_is_national_match(m, team_name):
            results.append(m)

    if results:
        return [_cricapi_fmt_match(m) for m in results]

    # Step 2: Sportmonks bilateral history filtered by date
    sm_all = _sportmonks_get_all_matches(team_name, limit=50)
    for block in sm_all:
        for line in block.split("\n"):
            if line.strip().startswith("Date:"):
                m_date = line.split("Date:")[-1].strip()[:10]
                if d_from <= m_date <= d_to:
                    results.append(block)
                break

    return results


def _cricapi_fmt_match(m: dict) -> str:
    """Format a CricAPI match dict into a display block (reuses _build_match_block logic)."""
    teams  = m.get("teams") or []
    t1     = teams[0] if teams else "TBA"
    t2     = teams[1] if len(teams) > 1 else "TBA"
    scores = m.get("score") or []
    status = "Match Ended" if m.get("matchEnded") else ("Live" if m.get("matchStarted") else "Scheduled")
    result = m.get("status") or status
    venue  = m.get("venue") or "N/A"
    dt     = (m.get("dateTimeGMT") or "")[:16].replace("T", "  ")
    lines  = [
        "Most Recent Match:",
        f"🏏 {t1} vs {t2}",
        f"   Status: {status}",
        f"   Series: {m.get('name', 'Cricket')}",
        f"   Match Type: {(m.get('matchType') or 'N/A').upper()}",
        f"   Venue: {venue}",
        f"   Date: {dt}",
    ]
    for s in scores:
        inn = s.get("inning", "")
        lines.append(f"   📊 {inn}: {s.get('r','–')}/{s.get('w','–')} ({s.get('o','–')} ov)")
    if "won" in result.lower() or m.get("matchEnded"):
        lines.append(f"   🏅 Result: {result}")
    return "\n".join(lines)


def _sportmonks_get_all_matches(team_name: str, limit: int = 20) -> list:
    """
    Fetch all recent Sportmonks matches for a team, returning formatted blocks.
    Used for n_matches and date-range cricket queries.
    """
    key = _key("SPORTMONKS_KEY")
    if not key:
        return []

    import time as _tm
    SM_BASE = "https://cricket.sportmonks.com/api/v2.0"
    params  = {"api_token": key, "filter[name]": team_name, "per_page": 5}
    team_data = _safe_get(f"{SM_BASE}/teams", params=params, timeout=15)
    teams_list = ((team_data or {}).get("data") or [])
    team_id = None
    for t in teams_list:
        if team_name.lower() in (t.get("name") or "").lower():
            team_id = t.get("id")
            break
    if not team_id:
        return []

    import datetime as _dtt
    since = (_dtt.date.today() - _dtt.timedelta(days=365)).isoformat()
    today = _dtt.date.today().isoformat()

    fix_params = {
        "api_token": key,
        "filter[starts_between]": f"{since},{today}",
        "filter[localteam_id]":   team_id,
        "sort": "-starting_at",
        "per_page": limit,
        "include": "localteam,visitorteam,scoreboards,league",
    }
    fix_data = _safe_get(f"{SM_BASE}/fixtures", params=fix_params, timeout=20)
    fixtures = (fix_data or {}).get("data") or []

    # Also visitor side
    fix_params2 = dict(fix_params)
    fix_params2["filter[localteam_id]"]   = ""
    fix_params2["filter[visitorteam_id]"] = team_id
    fix_data2   = _safe_get(f"{SM_BASE}/fixtures", params=fix_params2, timeout=20)
    fixtures   += (fix_data2 or {}).get("data") or []

    # Sort newest first, remove duplicates
    seen, unique = set(), []
    fixtures.sort(key=lambda f: f.get("starting_at") or "", reverse=True)
    for f in fixtures:
        if f.get("id") not in seen:
            seen.add(f.get("id"))
            unique.append(f)

    def _sm_name(val, fallback="?"):
        if not val: return fallback
        if isinstance(val, dict):
            inner = val.get("data") or val
            return (inner.get("name") if isinstance(inner, dict) else fallback) or fallback
        return fallback

    def _sm_venue_name(val):
        if not val: return "N/A"
        if isinstance(val, dict):
            inner = val.get("data") or val
            return (inner.get("name") if isinstance(inner, dict) else "N/A") or "N/A"
        return "N/A"

    results = []
    for f in unique[:limit]:
        lt = _sm_name(f.get("localteam"), "?")
        vt = _sm_name(f.get("visitorteam"), "?")
        lg = _sm_name(f.get("league"), "Cricket")
        status_raw = (f.get("status") or "").lower()
        ended = status_raw in ("finished", "completed", "result", "stumps")
        status = "Match Ended" if ended else status_raw.title()
        dt = (f.get("starting_at") or "")[:16].replace("T", "  ")

        _sbs_raw = f.get("scoreboards") or []
        sbs = _sbs_raw.get("data") if isinstance(_sbs_raw, dict) else _sbs_raw
        sbs = sbs or []
        score_lines = []
        for sb in sbs:
            if not isinstance(sb, dict):
                continue
            inn_label = f"{sb.get('team_id','?')} Inning {sb.get('innings',1)}"
            runs = sb.get("total") or "–"
            wkts = sb.get("wickets") or "–"
            ovrs = sb.get("overs") or "–"
            score_lines.append(f"   📊 {inn_label}: {runs}/{wkts} ({ovrs} ov)")

        lines = [
            "Most Recent Match:",
            f"🏏 {lt} vs {vt}",
            f"   Status: {status}",
            f"   Series: {lg}",
            f"   Match Type: {(f.get('type') or 'N/A').upper()}",
            f"   Venue: {_sm_venue_name(f.get('venue'))}",
            f"   Date: {dt}",
        ] + score_lines
        winner = f.get("winner_team_id")
        if winner:
            wname = lt if f.get("localteam_id") == winner else vt
            lines.append(f"   🏅 Winner: {wname}")
        results.append("\n".join(lines))

    return results


# ══════════════════════════════════════════════════════════════
# UPCOMING MATCHES
# ══════════════════════════════════════════════════════════════

def _get_upcoming_football(team_name: str = "", league_hint: str = "",
                            n: int = 5, team2: str = "") -> list:
    """
    Return upcoming/scheduled football fixtures.
    Sources: football-data.org → ESPN → apifootball.com
    """
    import datetime as _dt, time as _tm

    results = []
    comp_kw = league_hint.lower().strip()
    today   = _dt.date.today().isoformat()
    future  = (_dt.date.today() + _dt.timedelta(days=60)).isoformat()

    # ── football-data.org ─────────────────────────────────────
    fd_key = _key("FOOTBALLDATA_KEY")
    if fd_key:
        FOOTBALLDATA_BASE = "https://api.football-data.org/v4"
        HINT_MAP = {
            "premier league":"PL","serie a":"SA","la liga":"PD","bundesliga":"BL1",
            "ligue 1":"FL1","champions league":"CL","ucl":"CL","eredivisie":"DED",
            "championship":"ELC","primeira liga":"PPL","brasileirao":"BSA",
            "europa league":"EL","conference":"ECOL",
            "fa cup":"FAC","copa del rey":"CDR","coppa italia":"CI",
        }
        ALL_CODES = ["PL","SA","PD","BL1","FL1","CL","EL","ECOL","ELC","DED","PPL","BSA","FAC","CDR","CI","EC","WC"]
        hint_code = next((v for k, v in HINT_MAP.items() if k in comp_kw), None)

        FD_ALIASES = {
            "bayern munich":"fc bayern münchen","inter milan":"internazionale",
            "ac milan":"milan","man city":"manchester city",
            "man united":"manchester united","man utd":"manchester united",
            "psg":"paris saint-germain","barca":"fc barcelona",
            "barcelona":"fc barcelona","real madrid":"real madrid cf",
            "atletico madrid":"atlético de madrid","spurs":"tottenham hotspur",
            "tottenham":"tottenham hotspur","dortmund":"borussia dortmund",
            "leverkusen":"bayer 04 leverkusen",
        }

        if team_name:
            tf_raw = team_name.lower().strip()
            tf_fd  = FD_ALIASES.get(tf_raw, tf_raw)
            fd_order = ([hint_code] + [c for c in ALL_CODES if c != hint_code]) if hint_code else ALL_CODES

            def _fd_ok(t):
                name  = (t.get("name") or "").lower()
                short = (t.get("shortName") or "").lower()
                tla   = (t.get("tla") or "").lower()
                for q in {tf_fd, tf_raw}:
                    words = q.split()
                    if q == name or q == short or q == tla: return True
                    if len(words)==1 and (name.startswith(q) or short==q or tla==q): return True
                    if len(words)>1 and all(w in name for w in words): return True
                return False

            team_id = None
            for code in fd_order:
                data = _safe_get(f"{FOOTBALLDATA_BASE}/competitions/{code}/teams",
                                 headers={"X-Auth-Token": fd_key}, timeout=15)
                if data is None: break
                for t in (data or {}).get("teams") or []:
                    if _fd_ok(t):
                        team_id = t.get("id"); break
                if team_id: break
                _tm.sleep(0.15)

            if team_id:
                md = _safe_get(f"{FOOTBALLDATA_BASE}/teams/{team_id}/matches",
                               params={"status":"SCHEDULED","dateFrom":today,"dateTo":future,"limit":n*2},
                               headers={"X-Auth-Token": fd_key}, timeout=15)
                matches = sorted((md or {}).get("matches") or [],
                                 key=lambda m: m.get("utcDate") or "")
                if comp_kw:
                    FD_NAME_MAP = {
                        "la liga":["primera division","la liga"],
                        "champions league":["uefa champions league","champions league"],
                        "europa league":["uefa europa league","europa league"],
                        "premier league":["premier league"],"serie a":["serie a"],
                        "bundesliga":["bundesliga"],"ligue 1":["ligue 1"],
                    }
                    acceptable = FD_NAME_MAP.get(comp_kw, [comp_kw])
                    matches = [m for m in matches
                               if any(a in (m.get("competition") or {}).get("name","").lower()
                                      for a in acceptable)]
                for m in matches[:n]:
                    ht   = ((m.get("homeTeam") or {}).get("shortName") or
                            (m.get("homeTeam") or {}).get("name","?"))
                    at   = ((m.get("awayTeam") or {}).get("shortName") or
                            (m.get("awayTeam") or {}).get("name","?"))
                    comp = (m.get("competition") or {}).get("name","Unknown")
                    dt   = (m.get("utcDate") or "")[:16].replace("T","  ")
                    venue = m.get("venue") or "N/A"
                    # VS filter
                    if team2 and team2.lower() not in ht.lower() and team2.lower() not in at.lower():
                        continue
                    results.append(_fmt_upcoming(ht, at, comp, dt, venue))

        elif league_hint:
            # League-wide upcoming — no team filter
            code = hint_code
            if code:
                data = _safe_get(f"{FOOTBALLDATA_BASE}/competitions/{code}/matches",
                                 params={"status":"SCHEDULED","dateFrom":today,"dateTo":future},
                                 headers={"X-Auth-Token": fd_key}, timeout=15)
                matches = sorted((data or {}).get("matches") or [],
                                 key=lambda m: m.get("utcDate") or "")
                for m in matches[:n]:
                    ht   = ((m.get("homeTeam") or {}).get("shortName") or
                            (m.get("homeTeam") or {}).get("name","?"))
                    at   = ((m.get("awayTeam") or {}).get("shortName") or
                            (m.get("awayTeam") or {}).get("name","?"))
                    comp = (m.get("competition") or {}).get("name", league_hint)
                    dt   = (m.get("utcDate") or "")[:16].replace("T","  ")
                    venue = m.get("venue") or "N/A"
                    results.append(_fmt_upcoming(ht, at, comp, dt, venue))

    # ── ESPN fallback ─────────────────────────────────────────
    if not results and team_name:
        tf  = team_name.lower().strip()
        slugs = next((v for k, v in ESPN_HINT_SLUGS.items() if k in comp_kw), None) or DEFAULT_ESPN_SLUGS
        ESPN_AL = {
            "inter milan":["internazionale","inter"],"ac milan":["ac milan","milan"],
            "man city":["manchester city"],"man united":["manchester united"],
            "psg":["paris saint-germain"],"barca":["barcelona"],
            "atletico madrid":["atletico de madrid"],"dortmund":["borussia dortmund"],
            "spurs":["tottenham hotspur","tottenham"],"tottenham":["tottenham hotspur"],
            "leverkusen":["bayer leverkusen"],"bayern munich":["fc bayern munich","bayern munich"],
        }
        search_names = ESPN_AL.get(tf, [tf])
        for slug in slugs:
            if len(results) >= n: break
            td = _safe_get(f"{ESPN_BASE}/{slug}/teams", headers=ESPN_HDR, timeout=12)
            lt = ((td or {}).get("sports") or [{}])[0].get("leagues",[{}])[0].get("teams",[])
            team_id = None
            for t in lt:
                name = (t.get("team",{}).get("displayName") or "").lower()
                abbr = (t.get("team",{}).get("abbreviation") or "").lower()
                for sn in search_names:
                    words = sn.split()
                    if (sn==name or sn==abbr or name.startswith(sn) or
                            (len(words)>1 and all(w in name for w in words))):
                        team_id = t.get("team",{}).get("id"); break
                if team_id: break
            if not team_id: continue
            sched = _safe_get(f"{ESPN_BASE}/{slug}/teams/{team_id}/schedule",
                              headers=ESPN_HDR, timeout=12)
            events = sorted(
                [e for e in (sched or {}).get("events") or []
                 if not (e.get("competitions") or [{}])[0].get("status",{}).get("type",{}).get("completed",False)
                 and (e.get("date","")) >= today],
                key=lambda e: e.get("date","")
            )
            for e in events[:n]:
                comp2 = (e.get("competitions") or [{}])[0]
                teams = comp2.get("competitors") or []
                if len(teams) < 2: continue
                home = next((t for t in teams if t.get("homeAway")=="home"), teams[0])
                away = next((t for t in teams if t.get("homeAway")=="away"), teams[1])
                ht   = home.get("team",{}).get("displayName","?")
                at   = away.get("team",{}).get("displayName","?")
                if team2 and team2.lower() not in ht.lower() and team2.lower() not in at.lower():
                    continue
                lg   = ESPN_SLUG_MAP.get(slug, slug)
                dt   = e.get("date","")[:16].replace("T","  ")
                venue = (comp2.get("venue") or {}).get("fullName","N/A")
                results.append(_fmt_upcoming(ht, at, lg, dt, venue))

    # ── apifootball.com fallback ──────────────────────────────
    if not results and _key("APIFOOTBALL_COM_KEY"):
        tf = (team_name or "").lower().strip()
        league_ids = next((v for k, v in AF_COM_HINT_MAP.items() if k in comp_kw), AF_COM_DEFAULT_IDS)
        AF_AL = {
            "inter milan":["inter","internazionale"],"ac milan":["ac milan","milan"],
            "man city":["manchester city"],"man united":["manchester united"],
            "psg":["paris saint-germain"],"barca":["barcelona"],
            "atletico madrid":["atletico madrid"],"dortmund":["borussia dortmund"],
            "spurs":["tottenham hotspur","tottenham"],"tottenham":["tottenham hotspur"],
            "leverkusen":["bayer leverkusen"],"bayern munich":["fc bayern munich","bayern munich"],
        }
        search_names = AF_AL.get(tf, [tf]) if tf else []

        def _af_match(e):
            if not search_names: return True
            home = (e.get("match_hometeam_name") or "").lower()
            away = (e.get("match_awayteam_name") or "").lower()
            return any(sn in home or sn in away for sn in search_names)

        all_upcoming = []
        for lid in league_ids:
            evs = _afcom_get({"action":"get_events","from":today,"to":future,"league_id":lid})
            scheduled = [e for e in evs if e.get("match_status") in ("", "Not Started")
                         and _af_match(e)]
            all_upcoming.extend(scheduled)
        all_upcoming.sort(key=lambda e: e.get("match_date") or "")
        for e in all_upcoming[:n]:
            ht  = e.get("match_hometeam_name","?")
            at  = e.get("match_awayteam_name","?")
            lg  = e.get("league_name","Unknown")
            dt  = f"{e.get('match_date','?')}  {e.get('match_time','??:??')}"
            venue = e.get("match_stadium") or "N/A"
            if team2 and team2.lower() not in ht.lower() and team2.lower() not in at.lower():
                continue
            results.append(_fmt_upcoming(ht, at, lg, dt, venue))

    return results[:n]


def _get_upcoming_cricket(team_name: str = "", league_hint: str = "",
                           n: int = 5, team2: str = "") -> list:
    """
    Return upcoming cricket fixtures.
    Sources:
      1. CricAPI series_info matchList (best for IPL/franchise — full schedule)
      2. CricAPI currentMatches scheduled (matches within ~72hr window)
      3. Sportmonks (bilateral international series)
    """
    import datetime as _dt, time as _up_tm
    results = []
    today  = _dt.date.today().isoformat()
    future = (_dt.date.today() + _dt.timedelta(days=90)).isoformat()

    lh = (league_hint or "").lower().strip()
    tf = (team_name or "").lower().strip()

    # Abbreviation → full name fragments for series name matching
    _LH_EXPAND = {
        "ipl":  ["indian premier league", "ipl"],
        "bbl":  ["big bash league", "big bash", "bbl"],
        "psl":  ["pakistan super league", "psl"],
        "cpl":  ["caribbean premier league", "caribbean premier", "cpl"],
        "sa20": ["sa20"],
    }
    _lh_frags = _LH_EXPAND.get(lh, [lh] if lh else [])

    # IPL franchise abbrev → team name fragments for match filtering
    _IPL_ABBREV = {
        "csk":  ["chennai", "super kings"],
        "mi":   ["mumbai", "indians"],
        "rcb":  ["royal challengers", "bangalore", "bengaluru"],
        "kkr":  ["kolkata", "knight riders"],
        "srh":  ["sunrisers", "hyderabad"],
        "rr":   ["rajasthan", "royals"],
        "dc":   ["delhi", "capitals"],
        "pbks": ["punjab", "kings"],
        "lsg":  ["lucknow", "super giants"],
        "gt":   ["gujarat", "titans"],
    }
    _tf_frags = _IPL_ABBREV.get(tf, [tf] if tf else [])

    def _series_matches_league(series_name: str) -> bool:
        if not _lh_frags: return True
        return any(frag in series_name.lower() for frag in _lh_frags)

    def _match_has_team(teams_list: list) -> bool:
        if not _tf_frags: return True
        joined = " ".join(teams_list).lower()
        return any(frag in joined for frag in _tf_frags)

    def _match_has_team2(teams_list: list) -> bool:
        if not team2: return True
        return team2.lower() in " ".join(teams_list).lower()

    # ── SOURCE 1: CricAPI series_info matchList ───────────────
    # This gives the FULL schedule (past + future) — best for IPL/franchise
    cric_key = _key("CRICAPI_KEY")
    if cric_key and _cricapi_quota_ok():
        # Determine which series to search
        _search_terms = []
        if lh in _LH_EXPAND:
            _search_terms.append(_LH_EXPAND[lh][0])  # e.g. "indian premier league"
        elif tf:
            # franchise team without explicit league — use team name
            _search_terms.append(team_name or tf)
        else:
            _search_terms.append("indian premier league")  # default fallback

        _seen_series = set()
        for _term in _search_terms:
            if not _cricapi_quota_ok(): break
            _sd = _safe_get(f"{CRICAPI_BASE}/series",
                            params={"apikey": cric_key, "offset": 0, "search": _term},
                            timeout=20)
            if (_sd or {}).get("status") == "failure":
                _r = (_sd.get("reason") or "").lower()
                if any(x in _r for x in ["exceeded","limit","quota","daily"]):
                    _cricapi_mark_exhausted(_r)
                    break
            for _s in ((_sd or {}).get("data") or []):
                sid = _s.get("id")
                sname = (_s.get("name") or "")
                if sid in _seen_series: continue
                if not _series_matches_league(sname): continue
                _seen_series.add(sid)
                if not _cricapi_quota_ok(): break
                _si = _safe_get(f"{CRICAPI_BASE}/series_info",
                                params={"apikey": cric_key, "id": sid}, timeout=20)
                _matches = ((_si or {}).get("data") or {}).get("matchList") or []
                for m in _matches:
                    m_date = (m.get("dateTimeGMT") or "")[:10]
                    # Only future matches
                    if m_date < today: continue
                    if m_date > future: continue
                    teams = m.get("teams") or []
                    if not _match_has_team(teams): continue
                    if not _match_has_team2(teams): continue
                    t1 = teams[0] if teams else "TBA"
                    t2n = teams[1] if len(teams) > 1 else "TBA"
                    dt  = (m.get("dateTimeGMT") or "")[:16].replace("T", "  ")
                    venue = m.get("venue") or "N/A"
                    blk = _fmt_upcoming(t1, t2n, sname, dt, venue, emoji="🏏")
                    if blk not in results:
                        results.append(blk)
                _up_tm.sleep(0.3)
                if len(results) >= n: break
            if len(results) >= n: break

        # Sort by date ascending (soonest first)
        results.sort(key=lambda b: b.split("Date:")[-1].strip()[:16] if "Date:" in b else "")

    # ── SOURCE 2: CricAPI currentMatches (scheduled, within ~72hr) ──
    if cric_key and _cricapi_quota_ok() and len(results) < n:
        matches, _ = _get_current_matches(cric_key)
        scheduled = [m for m in matches
                     if not m.get("matchStarted") and not m.get("matchEnded")]
        for m in scheduled:
            teams = m.get("teams") or []
            series = m.get("name", "Cricket")
            if not _series_matches_league(series): continue
            if not _match_has_team(teams): continue
            if not _match_has_team2(teams): continue
            t1 = teams[0] if teams else "TBA"
            t2n = teams[1] if len(teams) > 1 else "TBA"
            dt    = (m.get("dateTimeGMT") or "")[:16].replace("T", "  ")
            venue = m.get("venue") or "N/A"
            blk = _fmt_upcoming(t1, t2n, series, dt, venue, emoji="🏏")
            if blk not in results:
                results.append(blk)

    # ── SOURCE 3: Sportmonks (bilateral international) ────────
    sm_key = _key("SPORTMONKS_KEY")
    if sm_key and team_name and len(results) < n:
        SM_BASE = "https://cricket.sportmonks.com/api/v2.0"
        team_data = _safe_get(f"{SM_BASE}/teams",
                              params={"api_token": sm_key, "filter[name]": team_name,
                                      "per_page": 5}, timeout=15)
        team_id = None
        for t in ((team_data or {}).get("data") or []):
            if tf in (t.get("name") or "").lower():
                team_id = t.get("id"); break
        if team_id:
            for fk in ["localteam_id", "visitorteam_id"]:
                fd = _safe_get(f"{SM_BASE}/fixtures",
                               params={"api_token": sm_key,
                                       f"filter[{fk}]": team_id,
                                       "filter[starts_between]": f"{today},{future}",
                                       "sort": "starting_at",
                                       "per_page": n * 2,
                                       "include": "localteam,visitorteam,league,venue"},
                               timeout=20)
                for f in ((fd or {}).get("data") or []):
                    if (f.get("status") or "").lower() not in ("ns","not started","scheduled","upcoming",""):
                        continue
                    lt  = _sm_name(f.get("localteam"), "?")
                    vt  = _sm_name(f.get("visitorteam"), "?")
                    lg  = _sm_name(f.get("league"), "Cricket")
                    dt  = (f.get("starting_at") or "")[:16].replace("T", "  ")
                    vn  = _sm_venue_name(f.get("venue"))
                    if not _series_matches_league(lg): continue
                    if team2 and team2.lower() not in lt.lower() and team2.lower() not in vt.lower():
                        continue
                    blk = _fmt_upcoming(lt, vt, lg, dt, vn, emoji="🏏")
                    if blk not in results:
                        results.append(blk)

    return results[:n]


def _fmt_upcoming(t1: str, t2: str, league: str, dt: str,
                  venue: str = "N/A", emoji: str = "🏆") -> str:
    """Format an upcoming match for display."""
    return "\n".join([
        "Upcoming Match:",
        f"{emoji} {t1} vs {t2}",
        f"   League/Tournament: {league}",
        f"   Venue: {venue}",
        f"   Date: {dt}",
    ])


def _sm_name(val, fallback="?"):
    if not val: return fallback
    if isinstance(val, dict):
        inner = val.get("data") or val
        return (inner.get("name") if isinstance(inner, dict) else fallback) or fallback
    return fallback


def _sm_venue_name(val):
    if not val: return "N/A"
    if isinstance(val, dict):
        inner = val.get("data") or val
        return (inner.get("name") if isinstance(inner, dict) else "N/A") or "N/A"
    return "N/A"


def get_sports_scores(query: str) -> str:
    sport_query, date, team_filter, n_matches, date_from, date_to, _league_explicit, _year_filter = _extract_sport_and_date(query)
    # ── Sanitise team_filter: strip pure tournament/league keywords ─────
    # e.g. "ipl score" → team_filter='ipl' is wrong; ipl is a tournament, not a team
    _TOURNAMENT_ONLY_WORDS = {
        'ipl', 'bbl', 'psl', 'cpl', 'sa20', 'ilt20', 'wpl', 'the hundred',
        'nba', 'nfl', 'nhl', 'mlb', 'mls', 'wnba', 'nrl', 'afl',
        'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1',
        'champions league', 'europa league', 'cricket', 'football', 'soccer',
    }
    if team_filter and team_filter.lower().strip() in _TOURNAMENT_ONLY_WORDS:
        team_filter = ''

    print(f'  [sports_tool] parsed → query={repr(sport_query)}, date={repr(date)}, team={repr(team_filter)}')

    q_lower = query.lower()

    wants_result   = bool(_WANTS_RESULT_RE.search(query))
    wants_upcoming = bool(_WANTS_UPCOMING_RE.search(query))

    if not wants_upcoming and _re_sports.search(r'\bscore\b', q_lower):
        wants_result = True

    prefer_completed = wants_result and not wants_upcoming

    # "last N matches", "last match", "recent matches", "vs" H2H queries
    # all imply completed results even without explicit score/result words
    if not wants_upcoming and not prefer_completed:
        _implies_completed = _re_sports.search(
            r'\b(last\s+\d+|last\s+match|last\s+game|recent\s+match|'
            r'most\s+recent|previous\s+match|h2h|head.to.head|'
            r'vs|played|defeated|beat|won\s+against|lost\s+to)\b',
            q_lower, _re_sports.IGNORECASE
        )
        if _implies_completed or n_matches:
            prefer_completed = True

    # explicit_date = True whenever date was parsed from the query (not just defaulted to today)
    # explicit_date = True only when user pinned a SPECIFIC past/future date
    # "today" / "tonight" / "now" mean "current info" not "restrict to today only"
    _explicit_date = bool(
        (date != 'today' and date != datetime.date.today().isoformat()) or
        _re_sports.search(r'\d{4}-\d{2}-\d{2}', query) or
        _re_sports.search(r'\b(yesterday|last night)\b', q_lower)
    )

    print(f'  [sports_tool] prefer_completed={prefer_completed}, explicit_date={_explicit_date}')

    # ── UPCOMING MATCHES ──────────────────────────────────────────────────
    if wants_upcoming:
        _n_up    = n_matches or 5
        _is_cric = _is_cricket(query)

        # Extract VS team2 for upcoming H2H
        _up_team2 = ""
        if team_filter and ' vs ' in (sport_query or ''):
            import re as _re_up
            _bracket = _re_up.search(r'\[(.+?)\]', sport_query)
            _sq_up   = _re_up.sub(r'\s*\[.+?\]', '', sport_query).strip()
            _parts   = _sq_up.split(' vs ', 1)
            if len(_parts) == 2:
                _up_team2 = _parts[1].strip()
                team_filter = _parts[0].strip()
        elif ' vs ' not in (sport_query or '') and team_filter:
            # Scan query for a second team
            if _is_cric:
                _tf_low_up = team_filter.lower()
                for _nat_up in CRICKET_NATIONS:
                    if _nat_up != _tf_low_up and _nat_up in q_lower:
                        _up_team2 = _nat_up.title(); break
            else:
                for _tm_up in sorted(_TEAM_LEAGUE_MAP.keys(), key=len, reverse=True):
                    if _tm_up != team_filter.lower() and _tm_up in q_lower:
                        _up_team2 = _tm_up; break

        # Only use as competition filter if user explicitly named a non-generic league
        # "cricket" is a sport, not a competition name — never use as filter
        _GENERIC_SPORTS = {"cricket", "football", "soccer", "basketball", "rugby"}
        _league_up = (sport_query
                      if _league_explicit and (sport_query or "").lower() not in _GENERIC_SPORTS
                      else "")

        print(f'  [sports_tool] UPCOMING: team={repr(team_filter)} team2={repr(_up_team2)} ' +
              f'league={repr(_league_up)} n={_n_up} cricket={_is_cric}')

        if _is_cric:
            # For IPL/franchise queries, ensure league_hint is set even if _league_explicit=False
            _ipl_teams_up = {
                "csk","mi","rcb","kkr","srh","rr","dc","pbks","lsg","gt",
                "mumbai indians","chennai super kings","royal challengers",
                "kolkata knight riders","sunrisers hyderabad","rajasthan royals",
                "delhi capitals","punjab kings","lucknow super giants","gujarat titans",
            }
            _up_lh = _league_up
            if not _up_lh:
                # Detect tournament from query directly (before _tour_hint is set in cricket block)
                _q_lh = q_lower
                if "ipl" in _q_lh or "indian premier" in _q_lh or team_filter.lower() in _ipl_teams_up:
                    _up_lh = "IPL"
                elif "psl" in _q_lh or "pakistan super" in _q_lh:
                    _up_lh = "PSL"
                elif "bbl" in _q_lh or "big bash" in _q_lh:
                    _up_lh = "BBL"
                elif "cpl" in _q_lh or "caribbean premier" in _q_lh:
                    _up_lh = "CPL"
                elif "sa20" in _q_lh:
                    _up_lh = "SA20"
            up_results = _get_upcoming_cricket(
                team_name=team_filter, league_hint=_up_lh,
                n=_n_up, team2=_up_team2
            )
            if up_results:
                header = "Upcoming cricket"
                if team_filter: header += f" — {team_filter}"
                if _up_team2:   header += f" vs {_up_team2}"
                if _league_up:  header += f" ({_league_up})"
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(up_results)
            return f"No upcoming cricket matches found{(' for ' + team_filter) if team_filter else ''}."
        else:
            up_results = _get_upcoming_football(
                team_name=team_filter, league_hint=_league_up or (sport_query or ""),
                n=_n_up, team2=_up_team2
            )
            if up_results:
                header = "Upcoming matches"
                if team_filter: header += f" — {team_filter.title()}"
                if _up_team2:   header += f" vs {_up_team2.title()}"
                if _league_up:  header += f" ({_league_up})"
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(up_results)
            return f"No upcoming matches found{(' for ' + team_filter.title()) if team_filter else ''}."

    # ── Date range request (e.g. "Barcelona matches in January 2026") ─────
    if date_from and team_filter and not _is_cricket(query):
        print(f'  [sports_tool] Date range: {team_filter} from {date_from} to {date_to or date_from}')
        range_results = _get_matches_in_range(
            team_filter, date_from, date_to or date_from, sport_query or ""
        )
        if range_results:
            label_to   = date_to or date_from
            header     = f"{team_filter.title()} matches"
            if sport_query and sport_query.lower() not in ("cricket",):
                header += f" — {sport_query}"
            header += f" ({date_from} to {label_to})"
            return f"\n{header}\n\n" + ("\n\n" + SEP).join(range_results)
        return (f"No matches found for '{team_filter.title()}'"
                f" between {date_from} and {date_to or date_from}"
                f"{(' in ' + sport_query) if sport_query else '.'}.")

    # ── Multiple matches request (e.g. "last 5 Barcelona matches") ────────
    # Skip if a specific date was requested — date takes priority over match count
    if n_matches and team_filter and not _is_cricket(query) and not _explicit_date:
        print(f'  [sports_tool] Recent {n_matches} matches for {repr(team_filter)} in {repr(sport_query)}')
        recent = _get_team_recent_matches(team_filter, n_matches, sport_query or "")
        if recent:
            header = f"Last {n_matches} matches for {team_filter.title()}"
            if sport_query and sport_query not in ("cricket",):
                header += f" — {sport_query}"
            return f"\n{header}\n\n" + ("\n\n" + SEP).join(recent)
        return f"No recent matches found for '{team_filter.title()}'{(' in ' + sport_query) if sport_query else '.'}."

    # ── H2H request (e.g. "Real Madrid vs Man United Champions League") ────
    # Already handled by VS path below, but ensure league filter is passed through

    # ── Cricket path ──────────────────────────────────────────────────────
    is_cricket_q = _is_cricket(sport_query) or _is_cricket(query)
    if is_cricket_q:
        # ── Extract team(s) from query ─────────────────────────────────
        import re as _re_f
        if not team_filter:
            for franchise, tournament in CRICKET_FRANCHISE_MAP.items():
                if len(franchise) <= 3:
                    if _re_f.search(r'\b' + _re_f.escape(franchise) + r'\b', q_lower):
                        team_filter = franchise.title(); break
                elif franchise in q_lower:
                    team_filter = franchise.title(); break
            if not team_filter:
                for nation in CRICKET_NATIONS:
                    if _re_f.search(r'\b' + _re_f.escape(nation) + r'\b', query.lower()):
                        team_filter = nation.title(); break

        # ── Detect second team for H2H ─────────────────────────────────
        _cric_team2 = ""
        if team_filter:
            _tf_low = team_filter.lower()
            for nation in CRICKET_NATIONS:
                if nation != _tf_low and nation in query.lower():
                    _cric_team2 = nation.title()
                    break

        # ── Extract tournament hint (odi, t20i, test, ipl etc.) ────────
        _tour_hint = ""
        for _th in ["t20 world cup", "champions trophy", "ipl", "odi", "t20i",
                    "test", "world cup", "asia cup", "bbl", "psl"]:
            if _th in q_lower:
                _tour_hint = _th
                break

        cricket_q = sport_query if _is_cricket(sport_query) else (
            f'{team_filter} cricket' if team_filter else 'cricket'
        )

        print(f'  [sports_tool] Cricket: team={repr(team_filter)} team2={repr(_cric_team2)} ' +
              f'tour={repr(_tour_hint)} date={repr(date)} n={n_matches} explicit={_explicit_date} year={_year_filter}')

        # ── YEAR filter — "india 2023 world cup", "ipl 2022" ──────────
        if _year_filter and team_filter and not date_from and not _explicit_date:
            _yr_from = f"{_year_filter}-01-01"
            _yr_to   = f"{_year_filter}-12-31"
            print(f'  [sports_tool] Cricket year filter: {_year_filter}')
            ranged_yr = []

            # CricAPI series scan filtered by year
            _cric_key_yr = _key("CRICAPI_KEY")
            if _cric_key_yr and _cricapi_quota_ok():
                import time as _tm_yr
                _tf_yr = team_filter.lower().strip()
                _all_series_yr = []
                _seen_yr = set()
                for _term_yr in [team_filter, "t20 world cup", "icc men",
                                  "champions trophy", "odi world cup"]:
                    if not _cricapi_quota_ok(): break
                    _sd_yr = _safe_get(f"{CRICAPI_BASE}/series",
                                       params={"apikey": _cric_key_yr, "offset": 0,
                                               "search": _term_yr}, timeout=20)
                    if (_sd_yr or {}).get("status") == "success":
                        for s in (_sd_yr.get("data") or []):
                            # Filter by year in series name or startDate
                            s_name = (s.get("name") or "").lower()
                            s_date = s.get("startDate") or s.get("date") or ""
                            if str(_year_filter) in s_name or str(_year_filter) in s_date:
                                if s.get("id") not in _seen_yr:
                                    _seen_yr.add(s.get("id"))
                                    _all_series_yr.append(s)
                    elif (_sd_yr or {}).get("status") == "failure":
                        _r_yr = (_sd_yr.get("reason") or "").lower()
                        if any(x in _r_yr for x in ["exceeded","limit","quota","daily"]):
                            _cricapi_mark_exhausted(_r_yr); break
                    _tm_yr.sleep(0.4)

                # Also check 1hr cache
                _cache_key_yr = team_filter.lower().strip()
                if _cache_key_yr in _CRICAPI_CACHE:
                    _ts_yr, _cached_yr = _CRICAPI_CACHE[_cache_key_yr]
                    import time as _t_yr
                    if _t_yr.time() - _ts_yr < _CRICAPI_CACHE_TTL:
                        for m in _cached_yr:
                            m_date = (m.get("dateTimeGMT") or "")[:10]
                            if _yr_from <= m_date <= _yr_to:
                                ranged_yr.append(_cricapi_fmt_match(m))

                # Scan series found
                _icc_kw_yr = {"world cup","champions trophy","icc men","icc t20","odi world cup"}
                _icc_yr = [s for s in _all_series_yr if any(k in (s.get("name") or "").lower() for k in _icc_kw_yr)]
                _bil_yr = [s for s in _all_series_yr if s not in _icc_yr]
                for s_yr in (_icc_yr + _bil_yr)[:8]:
                    if not _cricapi_quota_ok(): break
                    _si_yr = _safe_get(f"{CRICAPI_BASE}/series_info",
                                       params={"apikey": _cric_key_yr, "id": s_yr.get("id")},
                                       timeout=20)
                    for m in ((_si_yr or {}).get("data") or {}).get("matchList") or []:
                        m_date = (m.get("dateTimeGMT") or "")[:10]
                        if (_yr_from <= m_date <= _yr_to and
                                any(_cricket_team_matches(_tf_yr, t) for t in (m.get("teams") or []))):
                            ranged_yr.append(_cricapi_fmt_match(m))
                    _tm_yr.sleep(0.3)

            # Sportmonks fallback for year range
            if not ranged_yr:
                import datetime as _dtt_yr2
                _sm_all_yr = _sportmonks_get_all_matches(team_filter, limit=100)
                for block in _sm_all_yr:
                    for line in block.split("\n"):
                        if line.strip().startswith("Date:"):
                            m_date = line.split("Date:")[-1].strip()[:10]
                            if _yr_from <= m_date <= _yr_to:
                                ranged_yr.append(block)
                            break

            if ranged_yr:
                header = f"{team_filter} cricket — {_year_filter}"
                if _tour_hint: header += f" {_tour_hint.upper()}"
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(ranged_yr)
            return f"No {_year_filter} cricket matches found for {team_filter}{(' — ' + _tour_hint.upper()) if _tour_hint else ''}."

        # ── DATE RANGE filter ──────────────────────────────────────────
        if date_from and team_filter:
            d_to_use = date_to or date_from
            print(f'  [sports_tool] Cricket date range: {date_from} to {d_to_use}')
            ranged = []

            # 1. CricAPI cached series scan — covers ICC tournaments
            _tf_rng = team_filter.lower().strip()
            _cache_key_rng = _tf_rng
            _cached_rng = None
            import time as _tm_rng
            _now_rng = _tm_rng.time()
            if _cache_key_rng in _CRICAPI_CACHE:
                _ts_rng, _cd_rng = _CRICAPI_CACHE[_cache_key_rng]
                if _now_rng - _ts_rng < _CRICAPI_CACHE_TTL:
                    _cached_rng = _cd_rng
            if _cached_rng:
                for m in _cached_rng:
                    m_date = (m.get("dateTimeGMT") or "")[:10]
                    if date_from <= m_date <= d_to_use:
                        ranged.append(_cricapi_fmt_match(m))

            # 2. Sportmonks — bilateral series
            sm_all = _sportmonks_get_all_matches(team_filter, limit=50)
            for block in sm_all:
                for line in block.split("\n"):
                    if line.strip().startswith("Date:"):
                        m_date = line.split("Date:")[-1].strip()[:10]
                        if date_from <= m_date <= d_to_use:
                            ranged.append(block)
                        break

            if ranged:
                header = f"{team_filter} cricket matches ({date_from} to {d_to_use})"
                if _tour_hint: header += f" — {_tour_hint.upper()}"
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(ranged)
            return f"No cricket matches found for {team_filter} between {date_from} and {d_to_use}."

        # ── EXPLICIT DATE filter ───────────────────────────────────────
        if _explicit_date and team_filter and date not in ("today", "yesterday"):
            print(f'  [sports_tool] Cricket on date: {date}')
            import datetime as _dtt_cd
            try:
                _td     = _dtt_cd.date.fromisoformat(date)
                _d_from = (_td - _dtt_cd.timedelta(days=1)).isoformat()
                _d_to   = (_td + _dtt_cd.timedelta(days=1)).isoformat()
            except ValueError:
                _d_from = _d_to = date
            dated = []

            # 1. Try CricAPI series scan first — covers ICC tournaments
            _cric_key = _key("CRICAPI_KEY")
            if _cric_key and _cricapi_quota_ok():
                import time as _tm_cd
                # Get cached series for this team or fetch fresh
                _cache_key_cd = team_filter.lower().strip()
                _now_cd = _tm_cd.time()
                _cached_series = None
                if _cache_key_cd in _CRICAPI_CACHE:
                    _ts_cd, _cached_cd = _CRICAPI_CACHE[_cache_key_cd]
                    if _now_cd - _ts_cd < _CRICAPI_CACHE_TTL:
                        _cached_series = _cached_cd

                if _cached_series is None:
                    # Run a focused series scan
                    _all_series_cd = []
                    _seen_ids_cd   = set()
                    # Add "ipl" to search when tour_hint is ipl
                    _series_search_terms = [team_filter, "t20 world cup", "icc men", "champions trophy"]
                    if _tour_hint == "ipl":
                        _series_search_terms = ["indian premier league", "ipl"] + _series_search_terms
                    for _icc_term in _series_search_terms:
                        if not _cricapi_quota_ok(): break
                        _sd = _safe_get(f"{CRICAPI_BASE}/series",
                                        params={"apikey": _cric_key, "offset": 0, "search": _icc_term},
                                        timeout=20)
                        if (_sd or {}).get("status") == "success":
                            for s in (_sd.get("data") or []):
                                if s.get("id") not in _seen_ids_cd:
                                    _seen_ids_cd.add(s.get("id"))
                                    _all_series_cd.append(s)
                        elif (_sd or {}).get("status") == "failure":
                            _r = (_sd.get("reason") or "").lower()
                            if any(x in _r for x in ["exceeded","limit","quota","daily"]):
                                _cricapi_mark_exhausted(_r); break
                        _tm_cd.sleep(0.4)

                    # Prioritise ICC series
                    # Prioritise IPL when that's what was asked for
                    if _tour_hint == "ipl":
                        _icc_kw = {"indian premier league", "ipl"}
                    else:
                        _icc_kw = {"world cup","champions trophy","icc men","icc t20"}
                    _icc_s  = [s for s in _all_series_cd if any(k in (s.get("name") or "").lower() for k in _icc_kw)]
                    _bil_s  = [s for s in _all_series_cd if s not in _icc_s]
                    _ordered = _icc_s + _bil_s

                    _tf_cd = team_filter.lower()
                    _all_m_cd = []
                    for s in _ordered[:8]:
                        if not _cricapi_quota_ok(): break
                        _si = _safe_get(f"{CRICAPI_BASE}/series_info",
                                        params={"apikey": _cric_key, "id": s.get("id")}, timeout=20)
                        for m in ((_si or {}).get("data") or {}).get("matchList") or []:
                            if any(_cricket_team_matches(_tf_cd, t) for t in (m.get("teams") or [])):
                                _all_m_cd.append(m)
                        _tm_cd.sleep(0.3)

                    if _all_m_cd:
                        _CRICAPI_CACHE[_cache_key_cd] = (_now_cd, list(_all_m_cd))
                    _cached_series = _all_m_cd or []

                # Filter cached matches by date
                for m in (_cached_series or []):
                    m_date = (m.get("dateTimeGMT") or "")[:10]
                    if _d_from <= m_date <= _d_to:
                        # Fetch full scorecard
                        mid = m.get("id")
                        full_m = m
                        if mid and _cricapi_quota_ok():
                            _mi = _safe_get(f"{CRICAPI_BASE}/match_info",
                                           params={"apikey": _cric_key, "id": mid}, timeout=20)
                            full_m = (_mi or {}).get("data") or m
                        dated.append(_cricapi_fmt_match(full_m))

            # 2. Sportmonks fallback — bilateral series only
            if not dated:
                sm_all = _sportmonks_get_all_matches(team_filter, limit=50)
                for block in sm_all:
                    for line in block.split("\n"):
                        if line.strip().startswith("Date:"):
                            m_date = line.split("Date:")[-1].strip()[:10]
                            if _d_from <= m_date <= _d_to:
                                dated.append(block)
                            break

            if dated:
                return "\n\n" + ("\n\n" + SEP).join(dated)
            return f"No cricket match found for {team_filter} on {date}."

        # ── N_MATCHES filter ───────────────────────────────────────────
        # Tournament-level n_matches without team: "last 3 ipl matches", "last 5 t20 matches"
        if n_matches and not team_filter and not _explicit_date and _tour_hint and _tour_hint != "ipl":
            print(f'  [sports_tool] Cricket: last {n_matches} {_tour_hint} matches (no team)')
            _cric_key_nm = _key("CRICAPI_KEY")
            if _cric_key_nm and _cricapi_quota_ok():
                _all_nm, _ = _get_current_matches(_cric_key_nm)
                _tour_nm = [m for m in _all_nm
                            if _tour_hint in (m.get("name") or "").lower()
                            and (m.get("matchEnded") or "won" in (m.get("status") or "").lower())]
                _tour_nm.sort(key=lambda m: m.get("dateTimeGMT") or "", reverse=True)
                if _tour_nm:
                    header = f"Last {n_matches} {_tour_hint.upper()} matches"
                    return f"\n{header}\n\n" + ("\n\n" + SEP).join(
                        [_cricapi_fmt_match(m) for m in _tour_nm[:n_matches]])

        if n_matches and team_filter and not _explicit_date:
            print(f'  [sports_tool] Cricket: last {n_matches} matches for {team_filter}')
            # Try CricAPI first (covers ICC tournaments) then Sportmonks (bilateral)
            _nm_results = []
            _cric_nm = _cricapi_get_last_match(team_filter)
            if _cric_nm:
                _nm_results.extend(_cric_nm)
            sm_all = _sportmonks_get_all_matches(team_filter, limit=max(n_matches * 3, 30))
            for blk in sm_all:
                if blk not in _nm_results:
                    _nm_results.append(blk)
            if _nm_results:
                header = f"Last {n_matches} cricket matches for {team_filter}"
                if _tour_hint: header += f" — {_tour_hint.upper()}"
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(_nm_results[:n_matches])
            return f"No recent cricket matches found for '{team_filter}'."

        # ── H2H filter ────────────────────────────────────────────────
        if team_filter and _cric_team2:
            print(f'  [sports_tool] Cricket H2H: {team_filter} vs {_cric_team2}')
            sm_all = _sportmonks_get_all_matches(team_filter, limit=50)
            h2h = []
            t2l = _cric_team2.lower()
            for block in sm_all:
                if t2l in block.lower():
                    h2h.append(block)
            if h2h:
                header = f"{team_filter} vs {_cric_team2} cricket"
                if _tour_hint: header += f" — {_tour_hint.upper()}"
                _cric_h2h_limit = n_matches or 1
                return f"\n{header}\n\n" + ("\n\n" + SEP).join(h2h[:_cric_h2h_limit])
            return f"No cricket H2H matches found for {team_filter} vs {_cric_team2}."

        # ── LATEST SCORE (default) ─────────────────────────────────────
        if prefer_completed and not _explicit_date:

            # Identify IPL / franchise-tournament queries FIRST
            _IPL_TEAMS = {
                "mumbai indians", "mi", "chennai super kings", "csk",
                "royal challengers", "rcb", "kolkata knight riders", "kkr",
                "delhi capitals", "dc", "punjab kings", "pbks",
                "rajasthan royals", "rr", "sunrisers hyderabad", "srh",
                "lucknow super giants", "lsg", "gujarat titans", "gt",
            }
            _is_ipl_query = (
                _tour_hint == "ipl" or
                "ipl" in q_lower or
                "indian premier" in q_lower or
                team_filter.lower() in _IPL_TEAMS or
                any(t in q_lower for t in _IPL_TEAMS)
            )

            # Generic live check — SKIP for IPL queries so PSL/Women's matches
            # don't swallow the response before we reach the IPL-specific path
            if not _is_ipl_query:
                live = get_cricket_live(team_filter)
                if 'No live' not in live:
                    return live

            if _is_ipl_query:
                print(f'  [sports_tool] IPL query detected — fetching IPL matches')
                _ipl_key = _key("CRICAPI_KEY")
                if _ipl_key and _cricapi_quota_ok():
                    _all_m, _alive = _get_current_matches(_ipl_key)
                    # Filter to IPL matches: series name contains "indian premier"
                    _ipl_matches = [
                        m for m in _all_m
                        if "indian premier" in (m.get("name") or "").lower() or
                           "ipl" in (m.get("name") or "").lower()
                    ]
                    if team_filter:
                        _tf_low = team_filter.lower().strip()
                        # Map abbreviations → full name fragments for CricAPI matching
                        _IPL_ABBREV_MAP = {
                            "csk": ["chennai", "super kings"],
                            "mi":  ["mumbai", "indians"],
                            "rcb": ["royal challengers", "bangalore", "bengaluru"],
                            "kkr": ["kolkata", "knight riders"],
                            "srh": ["sunrisers", "hyderabad"],
                            "rr":  ["rajasthan", "royals"],
                            "dc":  ["delhi", "capitals"],
                            "pbks":["punjab", "kings"],
                            "lsg": ["lucknow", "super giants"],
                            "gt":  ["gujarat", "titans"],
                        }
                        _search_frags = _IPL_ABBREV_MAP.get(_tf_low, [_tf_low])
                        _ipl_matches = [
                            m for m in _ipl_matches
                            if any(
                                frag in " ".join(m.get("teams") or []).lower()
                                for frag in _search_frags
                            )
                        ]
                    if _ipl_matches:
                        _ipl_matches.sort(
                            key=lambda m: m.get("dateTimeGMT") or "", reverse=True
                        )
                        # Return n_matches if specified, else just the single most recent
                        _limit = n_matches or 1
                        results = []
                        for m in _ipl_matches[:_limit]:
                            results.append(_cricapi_fmt_match(m))
                        if results:
                            _label = "Latest Match" if _limit == 1 else f"Latest {_limit} Matches"
                            return f"\n\nIPL — {_label}\n\n" + ("\n\n" + SEP).join(results)

                    # No live IPL matches — search CricAPI series for most recent IPL match
                    print(f'  [sports_tool] No live IPL — searching CricAPI series for recent IPL')
                    import time as _ipl_tm

                    # Build team name fragments for filtering series matches
                    _IPL_ABBREV_MAP2 = {
                        "csk": ["chennai", "super kings"],
                        "mi":  ["mumbai", "indians"],
                        "rcb": ["royal challengers", "bangalore", "bengaluru"],
                        "kkr": ["kolkata", "knight riders"],
                        "srh": ["sunrisers", "hyderabad"],
                        "rr":  ["rajasthan", "royals"],
                        "dc":  ["delhi", "capitals"],
                        "pbks":["punjab", "kings"],
                        "lsg": ["lucknow", "super giants"],
                        "gt":  ["gujarat", "titans"],
                        "mumbai indians": ["mumbai", "indians"],
                        "chennai super kings": ["chennai", "super kings"],
                        "royal challengers": ["royal challengers"],
                        "kolkata knight riders": ["kolkata", "knight riders"],
                        "sunrisers hyderabad": ["sunrisers", "hyderabad"],
                        "rajasthan royals": ["rajasthan", "royals"],
                        "delhi capitals": ["delhi", "capitals"],
                        "punjab kings": ["punjab", "kings"],
                        "lucknow super giants": ["lucknow", "super giants"],
                        "gujarat titans": ["gujarat", "titans"],
                    }
                    _tf_frags = _IPL_ABBREV_MAP2.get(team_filter.lower(), [team_filter.lower()]) if team_filter else []

                    # Also detect second IPL team for H2H
                    _ipl_team2_frags = []
                    if not team_filter:
                        # Check if two IPL teams mentioned in query
                        import re as _re_ipl
                        _found_ipl_teams = []
                        for _abbr in ["csk","mi","rcb","kkr","srh","rr","dc","pbks","lsg","gt"]:
                            if _re_ipl.search(r'\b' + _abbr + r'\b', q_lower):
                                _found_ipl_teams.append(_abbr)
                        if len(_found_ipl_teams) == 2:
                            _tf_frags = _IPL_ABBREV_MAP2.get(_found_ipl_teams[0], [_found_ipl_teams[0]])
                            _ipl_team2_frags = _IPL_ABBREV_MAP2.get(_found_ipl_teams[1], [_found_ipl_teams[1]])

                    def _team_in_match(m, frags):
                        if not frags: return True
                        teams_str = " ".join(m.get("teams") or []).lower()
                        return any(f in teams_str for f in frags)

                    _ipl_series_data = _safe_get(
                        f"{CRICAPI_BASE}/series",
                        params={"apikey": _ipl_key, "offset": 0, "search": "indian premier league"},
                        timeout=20
                    )
                    if (_ipl_series_data or {}).get("status") == "success":
                        _ipl_series = sorted(
                            (_ipl_series_data.get("data") or []),
                            key=lambda s: s.get("startDate") or s.get("date") or "",
                            reverse=True
                        )
                        # Collect completed matches across ALL top-3 IPL series first,
                        # then pick the globally newest — avoids returning IPL 2024 when 2026 exists
                        _all_ipl_completed = []  # list of (match_dict, series_name)
                        for _s in _ipl_series[:3]:
                            if not _cricapi_quota_ok(): break
                            _si = _safe_get(
                                f"{CRICAPI_BASE}/series_info",
                                params={"apikey": _ipl_key, "id": _s.get("id")},
                                timeout=20
                            )
                            _matches = ((_si or {}).get("data") or {}).get("matchList") or []
                            _completed = [
                                m for m in _matches
                                if (m.get("matchEnded") or
                                    "won" in (m.get("status") or "").lower())
                            ]
                            # Apply team filter
                            if _tf_frags:
                                _filtered = [m for m in _completed if _team_in_match(m, _tf_frags)]
                                if _ipl_team2_frags:
                                    _filtered = [m for m in _filtered if _team_in_match(m, _ipl_team2_frags)]
                                if _filtered:
                                    _completed = _filtered
                            for _mc in _completed:
                                _all_ipl_completed.append((_mc, _s.get("name", "IPL")))
                            _ipl_tm.sleep(0.3)

                        if _all_ipl_completed:
                            # Sort all completed matches globally by date — pick newest
                            _all_ipl_completed.sort(
                                key=lambda x: x[0].get("dateTimeGMT") or "", reverse=True
                            )
                            _limit = n_matches or 1
                            _results_ipl = []
                            _used_series_name = _all_ipl_completed[0][1]
                            for _best, _ in _all_ipl_completed[:_limit]:
                                _mid = _best.get("id")
                                if _mid and _cricapi_quota_ok():
                                    _mi = _safe_get(
                                        f"{CRICAPI_BASE}/match_info",
                                        params={"apikey": _ipl_key, "id": _mid},
                                        timeout=20
                                    )
                                    _best = (_mi or {}).get("data") or _best
                                _results_ipl.append(_cricapi_fmt_match(_best))
                                _ipl_tm.sleep(0.2)

                            _header = _used_series_name
                            if team_filter: _header += f" — {team_filter.upper()}"
                            if _ipl_team2_frags: _header += f" vs {_ipl_team2_frags[0].title()}"
                            _label = "Most Recent Match" if _limit == 1 else f"Last {_limit} Matches"
                            return f"\n\n{_header} — {_label}\n\n" + ("\n\n" + SEP).join(_results_ipl)
                    elif (_ipl_series_data or {}).get("status") == "failure":
                        _r_ipl = (_ipl_series_data.get("reason") or "").lower()
                        if any(x in _r_ipl for x in ["exceeded","limit","quota","daily"]):
                            _cricapi_mark_exhausted(_r_ipl)

                    # Final fallback — TSDB IPL (capped at 30 days to avoid 429s)
                    import datetime as _dt_ipl, time as _ipl_tsdb_tm
                    for _d_back in range(0, 30):
                        if not _tsdb_rate_ok():
                            print("  [sports_tool] TSDB rate-limited — stopping IPL scan")
                            break
                        _scan = (_dt_ipl.date.today() - _dt_ipl.timedelta(days=_d_back)).isoformat()
                        _tsdb_ipl = _tsdb_get_matches("ipl", _scan)
                        if _tsdb_ipl is None:  # 429 returned None
                            break
                        _done_ipl = [m for m in _tsdb_ipl if _sort_key_from_str(m) == 0 and _has_real_score(m)]
                        if _tf_frags:
                            _done_ipl = [m for m in _done_ipl if any(f in m.lower() for f in _tf_frags)]
                        if _done_ipl:
                            _header_tsdb = "IPL"
                            if team_filter: _header_tsdb += f" — {team_filter.upper()}"
                            return f"\n\n{_header_tsdb} — Most Recent Match\n\n" + _done_ipl[0]
                        _ipl_tsdb_tm.sleep(0.2)  # gentle rate limiting
                    _no_result_msg = "No IPL matches found"
                    if team_filter: _no_result_msg += f" for {team_filter.upper()}"
                    return _no_result_msg + ". The IPL season may not have started yet."

            if team_filter:
                print(f'  [sports_tool] Cricket: CricAPI+Sportmonks recent matches for {repr(team_filter)}')
                cric_result = _cricapi_get_last_match(team_filter)
                if cric_result:
                    return "\n\n" + SEP.join(cric_result)

                # Fallback: TSDB scan
                for days_back in range(0, 30):
                    scan_date = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()
                    tsdb_only = _tsdb_get_matches(cricket_q, scan_date)
                    tsdb_only = [m for m in tsdb_only if team_filter.lower() in m.lower()]
                    completed = [m for m in tsdb_only if _sort_key_from_str(m) == 0 and _has_real_score(m)]
                    if completed:
                        return "\n\n" + SEP.join(completed[:1])

                return f"No completed cricket matches found for '{team_filter}'."
            else:
                # No team specified — return top 5 most recent international matches from CricAPI
                print(f'  [sports_tool] Cricket: no team specified — fetching recent international matches')
                key = _key("CRICAPI_KEY")
                if key:
                    matches, _ = _get_current_matches(key)
                    # Filter to TRUE international matches only
                    # Two conditions must BOTH be true:
                    # 1. Both teams are full international nations (not provinces/clubs)
                    # 2. Series name doesn't contain domestic/qualifier keywords
                    FULL_MEMBERS = {
                        "england", "india", "australia", "pakistan",
                        "new zealand", "south africa", "west indies",
                        "sri lanka", "bangladesh", "afghanistan", "zimbabwe",
                        "ireland", "netherlands", "scotland", "nepal",
                        "oman", "uae", "namibia", "kenya", "canada", "usa",
                        "papua new guinea", "uganda",
                    }
                    # Series name keywords that indicate domestic/non-international
                    DOMESTIC_KEYWORDS = {
                        "provincial", "ranji", "sheffield shield", "division",
                        "emerging", "legends league", "qualifier", "sub regional",
                        "domestic", "county", "inter-provincial", "franchise",
                        "cup challenge", "one-day challenge", "inter state",
                        "under", "u19", "u-19", "women", "a tour", "lions",
                        "primavera", "regional", "club",
                    }

                    def _is_true_intl(m):
                        teams  = " ".join(m.get("teams") or []).lower()
                        series = (m.get("name") or "").lower()
                        # Both teams must be full member nations
                        team_list = [t.lower() for t in (m.get("teams") or [])]
                        both_intl = sum(
                            1 for t in team_list
                            if any(n in t for n in FULL_MEMBERS)
                        ) >= 2
                        # Series must not be domestic
                        is_domestic = any(kw in series for kw in DOMESTIC_KEYWORDS)
                        return both_intl and not is_domestic

                    def _is_done(m):
                        return (m.get("matchEnded") or
                                "won" in (m.get("status") or "").lower() or
                                "draw" in (m.get("status") or "").lower() or
                                "no result" in (m.get("status") or "").lower())

                    intl = [m for m in matches if _is_true_intl(m)]
                    completed = [m for m in intl if _is_done(m)]
                    if completed:
                        completed.sort(key=lambda m: m.get("dateTimeGMT") or "", reverse=True)
                        results = []
                        for m in completed[:5]:
                            teams  = m.get("teams") or []
                            t1     = teams[0] if teams else "TBA"
                            t2     = teams[1] if len(teams) > 1 else "TBA"
                            scores = m.get("score") or []
                            innings_lines = []
                            for inn in scores:
                                inn_name = inn.get("inning", "?")
                                r = inn.get("r", "?"); w = inn.get("w", "?"); o = inn.get("o", "?")
                                innings_lines.append(f"   📊 {inn_name}: {r}/{w} ({o} ov)")
                            dt = (m.get("dateTimeGMT") or "")[:16].replace("T", "  ")
                            block = "\n".join([
                                "Most Recent Match:",
                                f"🏏 {t1} vs {t2}",
                                f"   Status: Match Ended",
                                f"   Series: {m.get('name','')}",
                                f"   Match Type: {(m.get('matchType') or 'N/A').upper()}",
                                f"   Venue: {m.get('venue','N/A')}",
                                f"   Date: {dt}",
                            ] + innings_lines + [f"   🏅 Result: {m.get('status','')}"])
                            results.append(block)
                        if results:
                            return "\n\n" + SEP.join(results)

                return f"No recent international cricket matches found."
        else:
            if not _explicit_date and date == 'today':
                live = get_cricket_live(team_filter)
                if 'No live' not in live:
                    return live
            return get_matches_text(cricket_q, date,
                                    prefer_completed=prefer_completed)

    # ── VS / H2H query ────────────────────────────────────────────────────
    # Handles: "Real Madrid vs Man United", "Real Madrid vs Man United Champions League"
    if ' vs ' in sport_query:
        # Extract bracketed league hint if present: "team1 vs team2 [FA Cup]"
        import re as _re_vs
        _bracket = _re_vs.search(r'\[(.+?)\]', sport_query)
        _league_in_bracket = _bracket.group(1) if _bracket else ""
        _sq_clean = _re_vs.sub(r'\s*\[.+?\]', '', sport_query).strip()
        parts  = _sq_clean.split(' vs ', 1)
        team1  = parts[0].strip()
        team2  = parts[1].strip() if len(parts) > 1 else ""
        # Normalise common typos/splits BEFORE league hint detection
        _VS_SQUISH = {
            "new castle": "newcastle", "man united": "manchester united",
            "man city": "manchester city", "man utd": "manchester united",
            "paris sg": "paris saint-germain", "psg": "paris saint-germain",
            "spurs": "tottenham", "barca": "barcelona",
            "bayern": "bayern munich", "bayer munich": "bayern munich",
            "baryern": "bayern munich", "bayren": "bayern munich",
            "bayer munchen": "bayern munich", "fc bayern": "bayern munich",
            "atletico": "atletico madrid", "atleti": "atletico madrid",
            "inter": "inter milan", "internazionale": "inter milan",
            "ac milan": "ac milan", "milan": "ac milan",
            "dortmund": "borussia dortmund", "bvb": "borussia dortmund",
            "leverkusen": "bayer 04 leverkusen",
            "ajax": "ajax", "psv": "psv",
            "benfica": "benfica", "porto": "porto",
        }
        team1 = _VS_SQUISH.get(team1.lower(), team1)
        team2 = _VS_SQUISH.get(team2.lower(), team2)

        # Fuzzy correction for typos: if team not in _TEAM_LEAGUE_MAP,
        # find closest known team name using difflib
        def _fuzzy_correct_team(name: str) -> str:
            nl = name.lower().strip()
            if nl in _TEAM_LEAGUE_MAP:
                return name  # already known
            if nl in _VS_SQUISH:
                return _VS_SQUISH[nl]
            # Try difflib closest match against known teams
            import difflib as _dl
            known = list(_TEAM_LEAGUE_MAP.keys())
            matches = _dl.get_close_matches(nl, known, n=1, cutoff=0.72)
            if matches:
                corrected = matches[0]
                print(f'  [fuzzy] Corrected "{name}" → "{corrected}"')
                return corrected
            return name

        team1 = _fuzzy_correct_team(team1)
        team2 = _fuzzy_correct_team(team2)
        comp_hint = _league_in_bracket or sport_query  # prefer explicit league hint
        print(f'  [sports_tool] H2H search: {repr(team1)} vs {repr(team2)} in {repr(comp_hint)}')

        # Try apifootball.com H2H first — best cross-competition coverage
        if _key("APIFOOTBALL_COM_KEY") and prefer_completed:
            import datetime as _dtt_h2h
            hl         = comp_hint.lower()
            # National team H2H → search international competitions only
            _h2h_t1_intl = _TEAM_LEAGUE_MAP.get(team1.lower()) == "International"
            _h2h_t2_intl = _TEAM_LEAGUE_MAP.get(team2.lower()) == "International"
            if _h2h_t1_intl or _h2h_t2_intl:
                league_ids = _INTL_IDS
                print(f'  [h2h] National teams — searching international competitions')
            else:
                league_ids = next((v for k, v in AF_COM_HINT_MAP.items() if k in hl), AF_COM_DEFAULT_IDS)
            SPECIFIC   = {"champions league":[3],"europa league":[4],"conference":[683],
                          "fa cup":[146],"league cup":[147],"copa del rey":[300],
                          "coppa italia":[205],"dfb pokal":[172],
                          "world cup":[6],"euros":[5],"nations league":[73,74],
                          "copa america":[197],"asian cup":[22],"afcon":[36],
                          "gold cup":[23],"friendly":[257]}
            for kw, ids in SPECIFIC.items():
                if kw in hl: league_ids = ids; break

            t1l = team1.lower(); t2l = team2.lower()
            since_h2h = (_dtt_h2h.date.today() - _dtt_h2h.timedelta(days=365)).isoformat()
            today_h2h = _dtt_h2h.date.today().isoformat()
            h2h_results = []

            # For cross-league H2H: search UCL + Europa + cups where teams can meet
            _t1l_league = _TEAM_LEAGUE_MAP.get(t1l, "")
            _t2l_league = _TEAM_LEAGUE_MAP.get(t2l, "")
            _cross_league = _t1l_league and _t2l_league and _t1l_league != _t2l_league
            if _cross_league:
                # Cross-league teams can only meet in European cups or international cups
                # Search UCL(3), Europa(4), Conference(683), Club World Cup(1), Super Cup(531)
                _search_ids = [3, 4, 683, 1, 531]
                print(f'  [h2h] Cross-league: {_t1l_league} vs {_t2l_league} — searching UCL/Europa')
            else:
                _search_ids = league_ids

            for lid in _search_ids:
                evs = _afcom_get({"action":"get_events","from":since_h2h,
                                  "to":today_h2h,"league_id":lid})
                def _side_matches(team_str: str, side: str) -> bool:
                    """Check if team_str matches a side (home/away name)."""
                    if team_str in side:
                        return True
                    # Multi-word: all significant words must appear
                    words = [w for w in team_str.split() if len(w) > 2]
                    if words and all(w in side for w in words):
                        return True
                    return False

                _FINISHED_STATUSES = {"Finished", "FT", "Match Finished", "After ET",
                                      "After Pen", "AET", "PEN", "finished", "ft"}
                for e in evs:
                    _ms = (e.get("match_status") or "").strip()
                    if _ms not in _FINISHED_STATUSES:
                        continue
                    home = (e.get("match_hometeam_name") or "").lower()
                    away = (e.get("match_awayteam_name") or "").lower()
                    # t1 must be in home OR away; t2 must be in the OTHER side
                    t1_home = _side_matches(t1l, home)
                    t1_away = _side_matches(t1l, away)
                    t2_home = _side_matches(t2l, home)
                    t2_away = _side_matches(t2l, away)
                    # Valid H2H: t1 on one side, t2 on the other side
                    t1_in = t1_home or t1_away
                    t2_in = t2_home or t2_away
                    # Both teams must appear AND be on opposite sides (not same team matched twice)
                    if not (t1_in and t2_in): continue
                    if t1_home and t2_home: continue  # both matched home — false positive
                    if t1_away and t2_away: continue  # both matched away — false positive
                    if True:
                        ht  = e.get("match_hometeam_name","?")
                        at  = e.get("match_awayteam_name","?")
                        hs  = e.get("match_hometeam_score","–") or "–"
                        as_ = e.get("match_awayteam_score","–") or "–"
                        lg  = e.get("league_name","Unknown")
                        dt  = (e.get("match_date","?") + "  " + e.get("match_time","??:??"))
                        h2h_results.append(_fmt(ht, at, hs, as_, "Full Time", lg,
                                                e.get("match_season",""),
                                                e.get("match_stadium","N/A"), dt))
            if h2h_results:
                h2h_results.sort(reverse=True)
                sep = "\n\n" + ("─" * 45) + "\n\n"
                header = f"{team1.title()} vs {team2.title()}"
                if any(k in hl for k in ["champions league","ucl","europa","fa cup",
                                          "league cup","copa","coppa","pokal"]):
                    header += f" — {sport_query.split(' vs ')[0].strip()} competitions"
                _h2h_limit = n_matches or 1
                return f"\n{header}\n\n" + sep.join(h2h_results[:_h2h_limit])

        # Fallback: TSDB
        vs_date = date if _explicit_date else 'today'
        vs_results = _tsdb_search_by_teams(team1, team2, vs_date)

        if prefer_completed and not _explicit_date and not vs_results:
            vs_results = _tsdb_search_by_teams(team1, team2, 'yesterday')

        if vs_results:
            if prefer_completed:
                vs_results.sort(key=_sort_key_from_str)
                completed = [m for m in vs_results if _sort_key_from_str(m) <= 1]
                if completed:
                    vs_results = completed
                else:
                    # No completed H2H in TSDB — skip and try per-team fallback
                    vs_results = []
            if vs_results:
                sep = "\n\n" + ("─" * 45) + "\n\n"
                _vs_limit = n_matches or 1
                return "\n\n" + sep.join(vs_results[:_vs_limit])

        # Fallback: fetch recent matches for each team independently, no league restriction
        # This handles cross-league H2H (e.g. Barcelona vs Newcastle)
        print(f'  [sports_tool] VS fallback: per-team lookup strictly filtered to both teams')
        import datetime as _dtt_fb, time as _tm_fb

        # Normalise team names: remove extra spaces, fix common typos
        def _normalise_team(t: str) -> str:
            import re as _re_n, difflib as _dl_n
            t = _re_n.sub(r'\s+', ' ', t.strip().lower())
            _SQUISH = {
                "new castle": "newcastle", "man united": "manchester united",
                "man city": "manchester city", "man utd": "manchester united",
                "paris sg": "paris saint-germain", "psg": "paris saint-germain",
                "spurs": "tottenham", "barca": "barcelona",
                "baryern munich": "bayern munich", "bayren munich": "bayern munich",
                "bayer munchen": "bayern munich", "baryern": "bayern munich",
            }
            t = _SQUISH.get(t, t)
            # Fuzzy correct unknown teams
            if t not in _TEAM_LEAGUE_MAP:
                _close = _dl_n.get_close_matches(t, list(_TEAM_LEAGUE_MAP.keys()), n=1, cutoff=0.72)
                if _close:
                    print(f'  [fuzzy_norm] Corrected "{t}" → "{_close[0]}"')
                    t = _close[0]
            return t

        team1_norm = _normalise_team(team1)
        team2_norm = _normalise_team(team2)
        _t1_words = [w for w in team1_norm.split() if len(w) > 3]
        _t2_words = [w for w in team2_norm.split() if len(w) > 3]

        def _block_has_both(block: str) -> bool:
            """
            Check that the match block contains both teams.
            Uses the team header line (line 1) for matching.
            Accepts: full name, any significant word (>3 chars), or first word of name.
            Handles short names like "Atleti" for "Atletico Madrid".
            """
            lines = block.split("\n")
            # Line 1 is the team matchup: "🏆 Team A vs Team B"
            team_line = lines[1].lower() if len(lines) > 1 else block.lower()

            # Known short-name / alias map for _team_found
            _SHORTNAME_MAP = {
                "barcelona":           ["barca", "barça", "fcb", "fc barcelona"],
                "atletico madrid":     ["atleti", "atletico", "atm", "atlético"],
                "real madrid":         ["real madrid cf", "rmcf"],
                "manchester united":   ["man utd", "man united", "mufc"],
                "manchester city":     ["man city", "mcfc"],
                "tottenham":           ["spurs", "thfc"],
                "paris saint-germain": ["psg", "paris sg", "paris"],
                "ac milan":            ["milan", "ac milan"],
                "inter milan":         ["inter", "internazionale"],
                "juventus":            ["juve"],
                "borussia dortmund":   ["bvb", "dortmund"],
                "bayer 04 leverkusen": ["leverkusen", "bayer leverkusen"],
                "newcastle":           ["newcastle united", "nufc"],
                "chelsea":             ["che"],
                "arsenal":             ["afc", "arsenal fc"],
                "liverpool":           ["lfc"],
                # Teams whose fd.org shortName drops the city
                "bayern munich":       ["bayern", "fc bayern", "fc bayern münchen", "münchen"],
                "rb leipzig":          ["leipzig", "rasenballsport"],
                "borussia monchengladbach": ["gladbach", "mönchengladbach"],
                "eintracht frankfurt": ["frankfurt", "eintracht"],
                "aston villa":         ["villa"],
                "west ham":            ["west ham united"],
                "nottingham forest":   ["forest", "nott'm forest"],
                "real sociedad":       ["sociedad", "real sociedad cf"],
                "atletico bilbao":     ["bilbao", "athletic bilbao", "athletic"],
                "porto":               ["fc porto"],
                "benfica":             ["sl benfica"],
                "ajax":                ["afc ajax"],
                "psv":                 ["psv eindhoven"],
                "celtic":              ["celtic fc"],
            }

            def _team_found(tname: str) -> bool:
                """
                Check if tname appears in team_line.
                Handles short/abbreviated names (Barça, Atleti, etc.)
                Uses: exact full-name > alias list > 4-char prefix on FIRST word only
                      when the full search name is multi-word (avoids "real" matching "real sociedad").
                """
                tl = tname.lower()
                # 1. Exact full-name match
                if tl in team_line:
                    return True
                # 2. Known aliases/short names (explicit map, no ambiguity)
                for alias in _SHORTNAME_MAP.get(tl, []):
                    if alias in team_line:
                        return True
                # 3. Prefix match — ONLY for single-word teams or when ALL words present
                words = [w for w in tl.split() if w]
                if not words:
                    return False
                # Tokenise the team_line once for prefix matching
                tokens = [
                    t.strip().lower()
                    for t in team_line.replace("🏆", "").replace(" vs ", " ").split()
                    if t.strip()
                ]
                if len(words) > 1:
                    # Tier A: ALL words present (most precise)
                    if all(w in team_line for w in words):
                        return True
                    # Tier B: FIRST significant word (len>=5) prefix-matches a token
                    # Handles fd shortNames like "Bayern" for "Bayern Munich",
                    # "Atleti" for "Atletico Madrid", etc.
                    sig = [w for w in words if len(w) >= 5]
                    if sig:
                        first_sig = sig[0]
                        for token in tokens:
                            if len(token) >= 4 and token[:4] == first_sig[:4]:
                                return True
                    return False
                # Single-word team: 4-char prefix match against tokens
                first = words[0]
                for token in tokens:
                    if len(token) >= 4 and len(first) >= 4 and token[:4] == first[:4]:
                        return True
                return False

            return _team_found(team1_norm) and _team_found(team2_norm)

        # Cross-league H2H (e.g. Barcelona vs Newcastle, Real Madrid vs Liverpool):
        # Search each team with their OWN league hint, collect all recent matches,
        # then filter blocks that contain BOTH teams (e.g. in a cup/friendly/UCL).
        _t1_league = _TEAM_LEAGUE_MAP.get(team1_norm, "")
        _t2_league = _TEAM_LEAGUE_MAP.get(team2_norm, "")
        _same_league = bool(_t1_league and _t2_league and _t1_league == _t2_league)

        print(f'  [sports_tool] H2H fallback: t1={team1_norm}({_t1_league}) t2={team2_norm}({_t2_league}) same={_same_league}')

        # Search each team across ALL competitions, fetch many recent matches
        # so we can find cross-league H2H (e.g. UCL, cups, friendlies)
        _t1_blocks = _apifootball_get_last_match(team1_norm, hint_league="", max_results=20)
        _t2_blocks = _apifootball_get_last_match(team2_norm, hint_league="", max_results=20)
        _all_blocks = _t1_blocks + _t2_blocks

        # Deduplicate and filter to blocks containing BOTH teams
        _seen_blk_keys = set()
        _strict = []
        for m in _all_blocks:
            _blk_key = m.split("\n")[1] if "\n" in m else m[:60]
            if _blk_key in _seen_blk_keys:
                continue
            _seen_blk_keys.add(_blk_key)
            if _block_has_both(m):
                _strict.append(m)

        if _strict:
            # Found a real H2H match (e.g. UCL, cup, friendly)
            _strict.sort(key=lambda b: b.split("Date:")[-1].strip()[:16] if "Date:" in b else "", reverse=True)
            sep = "\n\n" + ("─" * 45) + "\n\n"
            _strict_limit = n_matches or 1
            return f"\n{team1.title()} vs {team2.title()}\n\n" + sep.join(_strict[:_strict_limit])

        # No completed H2H found — check for upcoming match between them
        _upcoming_note = ""
        if _key("APIFOOTBALL_COM_KEY"):
            import datetime as _dtt_up_h2h
            _today_up = _dtt_up_h2h.date.today().isoformat()
            _future_up = (_dtt_up_h2h.date.today() + _dtt_up_h2h.timedelta(days=30)).isoformat()
            for _up_lid in [3, 4, 683]:  # UCL, Europa, Conference
                _up_evs = _afcom_get({"action": "get_events", "from": _today_up,
                                      "to": _future_up, "league_id": _up_lid})
                for _ue in _up_evs:
                    _uh = (_ue.get("match_hometeam_name") or "").lower()
                    _ua = (_ue.get("match_awayteam_name") or "").lower()
                    _t1_in = team1_norm in _uh or team1_norm in _ua or any(w in _uh or w in _ua for w in team1_norm.split() if len(w) > 3)
                    _t2_in = team2_norm in _uh or team2_norm in _ua or any(w in _uh or w in _ua for w in team2_norm.split() if len(w) > 3)
                    if _t1_in and _t2_in:
                        _up_dt = f"{_ue.get('match_date','?')}  {_ue.get('match_time','??:??')}"
                        _up_lg = _ue.get("league_name", "European Cup")
                        _upcoming_note = f"\n🔜 Upcoming: {_ue.get('match_hometeam_name')} vs {_ue.get('match_awayteam_name')} — {_up_lg} on {_up_dt}\n"
                        break
                if _upcoming_note: break

        # Show each team's latest match separately
        sep = "\n\n" + ("─" * 45) + "\n\n"
        _parts = []
        if _t1_blocks:
            _parts.append(f"Latest — {team1.title()}\n\n" + _t1_blocks[0])
        if _t2_blocks:
            _parts.append(f"Latest — {team2.title()}\n\n" + _t2_blocks[0])
        if _parts:
            _note = (f"\n⚠️  No recent completed {team1.title()} vs {team2.title()} match found. "
                     f"Showing each team's latest result:{_upcoming_note}\n")
            return _note + sep.join(_parts)

        return f"No recent match found between {team1.title()} and {team2.title()}."

    # ── League-wide / single-team league search ───────────────────────────
    if prefer_completed:
        # If team_filter is empty but sport_query looks like a team name
        # (not a known league keyword), extract it as the team
        if not team_filter and sport_query:
            _known_leagues = set(_LEAGUE_KEYWORDS.values()) | set(_LEAGUE_KEYWORDS.keys())
            _sq_low = sport_query.lower().strip()
            # sport_query is a team name if it's not a known league and not a VS query
            if (' vs ' not in _sq_low and
                    _sq_low not in {l.lower() for l in _known_leagues} and
                    not _is_cricket(sport_query)):
                # Strip trailing filler words that leaked in
                import re as _re_tf
                _cleaned = _re_tf.sub(
                    r'\s*(score|result|match|game|latest|recent|today|yesterday|now)\s*$',
                    '', _sq_low, flags=_re_tf.IGNORECASE
                ).strip()
                if len(_cleaned) > 2:
                    team_filter = _cleaned
                    print(f'  [sports_tool] Inferred team_filter={repr(team_filter)} from sport_query')

        if team_filter:
            # Check if user also specified a competition (e.g. "juventus champions league score")
            # In that case, sport_query is the competition — filter results to that comp only
            comp_keywords = {
                "champions league": ["champions league", "ucl", "uefa champions"],
                "europa league":    ["europa league", "uel", "europa"],
                "conference":       ["conference league", "uecl"],
                "fa cup":           ["fa cup"],
                "league cup":       ["league cup", "carabao"],
                "copa del rey":     ["copa del rey"],
                "coppa italia":     ["coppa italia"],
                "dfb pokal":        ["dfb-pokal", "dfb pokal"],
                "coupe de france":  ["coupe de france"],
                "serie a":          ["serie a"],
                "premier league":   ["premier league"],
                "la liga":          ["la liga", "primera division"],
                "bundesliga":       ["bundesliga"],
                "ligue 1":          ["ligue 1"],
                "mls":              ["mls", "major league soccer"],
            }
            sq_lower = sport_query.lower()
            comp_filter = None
            for comp, keywords in comp_keywords.items():
                if any(k in sq_lower for k in keywords):
                    comp_filter = keywords  # list of strings to match in league name
                    break

            print(f'  [sports_tool] ESPN + apifootball.com lookup for: {repr(team_filter)} in {repr(sport_query)}')
            # National team: force international competition search
            _tf_league = _TEAM_LEAGUE_MAP.get(team_filter.lower(), "")
            _is_national = _tf_league == "International"
            _af_limit = n_matches or 1

            if _is_national:
                # National teams: TSDB first (most reliable for intl matches), then apifootball
                print(f'  [sports_tool] National team — trying TSDB first')
                tsdb_natl = _tsdb_get_last_match(team_filter)
                scored_tsdb = [m for m in tsdb_natl if _has_real_score(m)]
                if scored_tsdb:
                    if _af_limit > 1:
                        # Need more matches — also search TSDB for recent events
                        all_natl = scored_tsdb
                        af_natl = _apifootball_get_last_match(team_filter,
                                      hint_league="International",
                                      max_results=_af_limit)
                        for blk in af_natl:
                            if blk not in all_natl:
                                all_natl.append(blk)
                        return "\n\n" + SEP.join(all_natl[:_af_limit])
                    return "\n\n" + SEP.join(scored_tsdb[:1])
                # TSDB found nothing — try apifootball with name search
                _hint = "International"
            elif _league_explicit:
                _hint = sport_query
            else:
                _hint = ""

            print(f'  [sports_tool] league_explicit={_league_explicit}, hint={repr(_hint)}')
            af_results = _apifootball_get_last_match(team_filter, hint_league=_hint,
                                                     target_date=date if _explicit_date else "",
                                                     max_results=_af_limit)
            scored_af = [m for m in af_results if _has_real_score(m)]

            if scored_af:
                if _af_limit > 1:
                    af_results_multi = _get_team_recent_matches(team_filter, _af_limit, _hint)
                    if af_results_multi:
                        return "\n\n" + SEP.join(af_results_multi[:_af_limit])
                return "\n\n" + SEP.join(scored_af[:1])

            # Nothing found — if a specific competition was requested, say so
            if comp_filter:
                return f"No recent {sport_query} match found for '{team_filter.title()}'. They may not currently be in this competition."

            # Fallback to TSDB (for club teams not already tried)
            if not _is_national:
                print(f'  [sports_tool] Falling back to TSDB for: {repr(team_filter)}')
                tsdb_results = _tsdb_get_last_match(team_filter)
                scored = [m for m in tsdb_results if _has_real_score(m)]
                if scored:
                    return "\n\n" + SEP.join(scored[:1])

        if _explicit_date:
            results = get_matches(sport_query, date, prefer_completed=True)
            if team_filter:
                results = [m for m in results if team_filter in m.lower()]
            return "\n\n" + SEP.join(results) if results else f"No completed matches found for '{team_filter or sport_query}' on {date}."

        max_days = 30  # cap at 30 days to avoid 429 rate limits on TSDB
        import time as _scan_tm
        for days_back in range(0, max_days):
            if not _tsdb_rate_ok():
                print("  [sports_tool] TSDB rate-limited — stopping scan")
                break
            scan_date = (
                datetime.date.today() - datetime.timedelta(days=days_back)
            ).isoformat()
            print(f'  [sports_tool] Scanning {scan_date} for completed matches...')
            scan_results = get_matches(sport_query, scan_date, prefer_completed=True)

            if team_filter:
                scan_results = [m for m in scan_results if team_filter in m.lower()]

            has_completed = any(
                _sort_key_from_str(m) == 0
                for m in scan_results
                if not m.startswith("No matches")
            )
            if has_completed:
                print(f'  [sports_tool] Found completed matches on {scan_date}')
                return "\n\n" + SEP.join(scan_results)
            _scan_tm.sleep(0.1)  # gentle rate limiting

        return f"No completed matches found for '{sport_query}'."

    return get_matches_text(sport_query, date, prefer_completed=False)


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "validate":
        validate_keys()
    elif cmd == "predict":
        query = sys.argv[2] if len(sys.argv) > 2 else "Premier League"
        date  = sys.argv[3] if len(sys.argv) > 3 else "today"
        print(get_predictions(query, date))
    elif cmd == "live cricket":
        team = sys.argv[2] if len(sys.argv) > 2 else ""
        print(get_cricket_live(team))
    else:
        query = sys.argv[1]
        date  = sys.argv[2] if len(sys.argv) > 2 else "today"
        if query.lower().startswith("live cricket"):
            team = query[12:].strip()
            print(get_cricket_live(team))
        else:
            print(get_matches_text(query, date))