# system_commands/system_tools.py

import subprocess
import webbrowser
import os
import re
import threading
import json
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import ollama
import requests


def find_app(app_name: str):
    """
    Search for any app in /Applications and ~/Applications.
    Returns app name if found and opened, None otherwise.
    """
    search_paths = [
        "/Applications",
        os.path.expanduser("~/Applications"),
        "/System/Applications",
        "/System/Applications/Utilities"
    ]

    app_name_clean = app_name.strip().title()

    for path in search_paths:
        if not os.path.exists(path):
            continue
        for app in os.listdir(path):
            if app.endswith(".app"):
                app_lower = app.replace(".app", "").lower()
                if app_name_clean.lower() in app_lower or \
                   app_lower in app_name_clean.lower():
                    full_path = os.path.join(path, app)
                    subprocess.run(["open", full_path])
                    return app.replace(".app", "")

    return None


def open_url(url: str) -> str:
    """Open any URL in default browser."""
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}..."


def open_in_chrome(query: str) -> str:
    """
    Fallback — open app website or search in Chrome
    when app is not installed.
    """
    if "." in query:
        url = query if query.startswith("http") else f"https://{query}"
    else:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    try:
        subprocess.run(["open", "-a", "Google Chrome", url])
        return f"Opening {query} in Chrome..."
    except Exception:
        webbrowser.open(url)
        return f"Opening {query} in browser..."


def _notify(title: str, message: str):
    """Send a macOS system notification via osascript."""
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}" sound name "Ping"'
    ])


def _set_reminder_reminders_app(title: str, due_datetime) -> bool:
    """
    Create a reminder in macOS Reminders.app via AppleScript.
    Returns True on success.
    """
    subprocess.run(['osascript', '-e', 'tell application "Reminders" to launch'],
                   capture_output=True)
    date_str = due_datetime.strftime('%A, %d %B %Y at %I:%M:%S %p')

    # Try default list first
    script = (
        f'tell application "Reminders"\n'
        f'  set newReminder to make new reminder with properties {{'
        f'name:"{title}", '
        f'due date:date "{date_str}", '
        f'remind me date:date "{date_str}"}}\n'
        f'end tell'
    )
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode == 0:
        return True

    # Fallback: target the first available list explicitly
    script2 = (
        f'tell application "Reminders"\n'
        f'  set theList to first list\n'
        f'  make new reminder at end of reminders of theList with properties {{'
        f'name:"{title}", '
        f'due date:date "{date_str}", '
        f'remind me date:date "{date_str}"}}\n'
        f'end tell'
    )
    result2 = subprocess.run(['osascript', '-e', script2], capture_output=True)
    return result2.returncode == 0


def _set_reminder_app(title: str, due_datetime) -> bool:
    """
    Create a persistent Calendar event with a sound alarm via AppleScript.
    Uses Calendar app instead of Reminders — louder, rings like an alarm.
    This survives server restarts — the OS owns it.
    due_datetime: a datetime object for when to fire.
    Returns True if successful.
    """
    subprocess.run(['osascript', '-e', 'tell application "Calendar" to launch'], 
               capture_output=True)
    
    date_str     = due_datetime.strftime('%A, %d %B %Y at %I:%M:%S %p')
    end_datetime = due_datetime + timedelta(minutes=1)
    end_str      = end_datetime.strftime('%A, %d %B %Y at %I:%M:%S %p')

    script = (
        f'tell application "Calendar"\n'
        f'  tell calendar "Home"\n'
        f'    set newEvent to make new event with properties {{'
        f'summary:"{title}", '
        f'start date:date "{date_str}", '
        f'end date:date "{end_str}"}}\n'
        f'    tell newEvent\n'
        f'      make new sound alarm with properties {{trigger interval:-1}}\n'
        f'    end tell\n'
        f'  end tell\n'
        f'end tell'
    )
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode != 0:
        script2 = (
            f'tell application "Calendar"\n'
            f'  set theCal to first calendar\n'
            f'  tell theCal\n'
            f'    set newEvent to make new event with properties {{'
            f'summary:"{title}", '
            f'start date:date "{date_str}", '
            f'end date:date "{end_str}"}}\n'
            f'    tell newEvent\n'
            f'      make new sound alarm with properties {{trigger interval:-1}}\n'
            f'    end tell\n'
            f'  end tell\n'
            f'end tell'
        )
        result2 = subprocess.run(['osascript', '-e', script2], capture_output=True)
        return result2.returncode == 0
    return True


def _open_file_in_app(file_path: str, app_name: str = None) -> str:
    """
    Open a specific file, optionally in a given app.
    Expands ~ and resolves relative paths.
    """
    file_path = os.path.expanduser(file_path.strip())

    if not os.path.isabs(file_path):
        search_dirs = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~"),
        ]
        for d in search_dirs:
            candidate = os.path.join(d, file_path)
            if os.path.exists(candidate):
                file_path = candidate
                break

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    if app_name:
        APP_ALIASES = {
            "vscode":    "Visual Studio Code",
            "vs code":   "Visual Studio Code",
            "code":      "Visual Studio Code",
            "sublime":   "Sublime Text",
            "atom":      "Atom",
            "pycharm":   "PyCharm",
            "xcode":     "Xcode",
            "notepad":   "TextEdit",
            "textedit":  "TextEdit",
            "excel":     "Microsoft Excel",
            "word":      "Microsoft Word",
            "powerpoint":"Microsoft PowerPoint",
            "numbers":   "Numbers",
            "pages":     "Pages",
            "keynote":   "Keynote",
        }
        resolved = APP_ALIASES.get(app_name.lower().strip(), app_name.title())
        result = subprocess.run(
            ["open", "-a", resolved, file_path],
            capture_output=True
        )
        if result.returncode == 0:
            return f"Opening {os.path.basename(file_path)} in {resolved}..."
        else:
            subprocess.run(["open", file_path])
            return f"Opening {os.path.basename(file_path)} (could not find {resolved}, used default app)..."
    else:
        subprocess.run(["open", file_path])
        return f"Opening {os.path.basename(file_path)}..."


def _parse_time_str(time_str: str):
    """
    Parse a time string like '5 minutes', '2 hours', '30 seconds',
    '10:30', '10:30 AM', '22:00'.
    Returns (seconds_from_now, display_label) or (None, None) if unparseable.
    """
    time_str = time_str.strip().lower()

    if re.match(r'^noon$', time_str):
        time_str = '12:00 pm'
    elif re.match(r'^midnight$', time_str):
        time_str = '12:00 am'
    else:
        ampm_suffix = ''
        ampm_m = re.search(r'\s*(am|pm)\s*$', time_str)
        if ampm_m:
            ampm_suffix = ' ' + ampm_m.group(1)
            time_str = time_str[:ampm_m.start()].strip()
        if re.match(r'^\d{3,4}$', time_str):
            time_str = time_str[:-2] + ':' + time_str[-2:] + ampm_suffix
        elif re.match(r'^\d{1,2}\s+\d{2}$', time_str):
            time_str = re.sub(r'^(\d{1,2})\s+(\d{2})$', r'\1:\2', time_str) + ampm_suffix
        elif re.match(r'^\d{1,2}\.\d{2}$', time_str):
            time_str = time_str.replace('.', ':') + ampm_suffix
        else:
            time_str = time_str + ampm_suffix

    total_seconds = 0
    found = False

    hour_match = re.search(r'(\d+)\s*h(?:our|r)?s?', time_str)
    min_match  = re.search(r'(\d+)\s*m(?:in(?:ute)?)?s?', time_str)
    sec_match  = re.search(r'(\d+)\s*s(?:ec(?:ond)?)?s?', time_str)

    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        found = True
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        found = True
    if sec_match:
        total_seconds += int(sec_match.group(1))
        found = True

    if found and total_seconds > 0:
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        parts = []
        if h: parts.append(f"{h}h")
        if m: parts.append(f"{m}m")
        if s: parts.append(f"{s}s")
        return total_seconds, " ".join(parts)

    clock_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', time_str)
    if clock_match:
        hour   = int(clock_match.group(1))
        minute = int(clock_match.group(2))
        ampm   = clock_match.group(3)

        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0

        now    = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        diff_secs = int((target - now).total_seconds())
        label     = target.strftime("%I:%M %p")
        return diff_secs, label

    hour_only = re.search(r'^(\d{1,2})\s*(am|pm)$', time_str.strip())
    if hour_only:
        hour = int(hour_only.group(1))
        ampm = hour_only.group(2)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        now    = datetime.now()
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        diff_secs = int((target - now).total_seconds())
        label     = target.strftime("%I:%M %p")
        return diff_secs, label

    return None, None


def _parse_event_date(date_str: str) -> datetime:
    """
    Parse a natural language date string into a datetime.
    """
    date_str = date_str.strip().lower()
    today    = datetime.now()

    if date_str in ('today', 'now'):
        return today
    if date_str == 'tomorrow':
        return today + timedelta(days=1)
    if date_str == 'yesterday':
        return today - timedelta(days=1)

    weekdays = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for i, day in enumerate(weekdays):
        if day in date_str:
            days_ahead = (i - today.weekday()) % 7
            if days_ahead == 0 and 'next' in date_str:
                days_ahead = 7
            elif days_ahead == 0:
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    months = {
        'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,
        'apr':4,'april':4,'may':5,'jun':6,'june':6,'jul':7,'july':7,
        'aug':8,'august':8,'sep':9,'september':9,'oct':10,'october':10,
        'nov':11,'november':11,'dec':12,'december':12
    }

    formats = [
        '%d %B %Y', '%B %d %Y', '%d %b %Y', '%b %d %Y',
        '%d %B',    '%B %d',    '%d %b',    '%b %d',
        '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d',
    ]
    clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str).strip()

    for fmt in formats:
        for candidate in [clean, date_str]:
            try:
                parsed = datetime.strptime(candidate, fmt)
                if parsed.year == 1900:
                    parsed = parsed.replace(year=today.year)
                    if parsed < today:
                        parsed = parsed.replace(year=today.year + 1)
                return parsed
            except ValueError:
                continue

    day_m   = re.search(r'\b(\d{1,2})', clean)
    month_m = None
    for name, num in months.items():
        if name in clean:
            month_m = num
            break

    if day_m and month_m:
        day  = int(day_m.group(1))
        year = today.year
        try:
            result = datetime(year, month_m, day)
            if result < today:
                result = result.replace(year=year + 1)
            return result
        except ValueError:
            pass

    return None


def _handle_event(command_lower: str) -> str:
    """
    Handle calendar event / date marking commands.
    """
    cmd = command_lower

    time_match = re.search(r'\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', cmd)
    event_time = time_match.group(1).strip() if time_match else None
    if time_match:
        cmd = cmd[:time_match.start()].strip()

    m = re.search(
        r'(?:mark|save|remember)\s+(.+?)\s+(?:for|as)\s+(.+)',
        cmd
    )
    if m:
        date_part  = m.group(1).strip()
        title_part = m.group(2).strip()
        event_date = _parse_event_date(date_part)
        if event_date:
            success = _create_calendar_event(title_part.title(), event_date, event_time)
            if success:
                return f"📅 '{title_part.title()}' added to Calendar on {event_date.strftime('%d %B %Y')}{'  at ' + event_time if event_time else ''} — you'll get a reminder."
            return "Failed to create Calendar event. Make sure Calendar access is allowed in System Settings."

    m = re.search(
        r'(?:add|create|schedule|set)\s+(?:event\s+|a\s+)?(?:an?\s+)?(.+?)\s+(?:on|for)\s+(.+)',
        cmd
    )
    if m:
        title_part = m.group(1).strip()
        date_part  = m.group(2).strip()
        event_date = _parse_event_date(date_part)
        if event_date:
            success = _create_calendar_event(title_part.title(), event_date, event_time)
            if success:
                return f"📅 '{title_part.title()}' added to Calendar on {event_date.strftime('%d %B %Y')}{'  at ' + event_time if event_time else ''} — you'll get a reminder."
            return "Failed to create Calendar event. Make sure Calendar access is allowed in System Settings."

    m = re.search(
        r'(?:add|mark|save)\s+(?:the\s+)?birthday\s+(?:of\s+)?([\w\s]+?)\s+(?:on|for)\s+(.+)',
        cmd
    )
    if m:
        name       = m.group(1).strip().title()
        date_part  = m.group(2).strip()
        event_date = _parse_event_date(date_part)
        if event_date:
            title   = f"{name}'s Birthday 🎂"
            success = _create_calendar_event(title, event_date, event_time)
            if success:
                return f"📅 '{title}' added to Calendar on {event_date.strftime('%d %B %Y')} — you'll get a reminder."
            return "Failed to create Calendar event."

    return None


def _set_timer_or_alarm(seconds: int, label: str, kind: str = "Timer") -> str:
    """
    Set a timer or alarm.
    """
    due = datetime.now() + timedelta(seconds=seconds)

    if kind == "Alarm" or seconds > 600:
        success = _set_reminder_app(f"⏰ {kind}: {label}", due)
        if success:
            return "calendar"

    threading.Timer(
        seconds,
        _notify,
        args=[f"⏰ {kind} Done", f"{label} is up!"]
    ).start()
    return "thread"


# ── Location aliases ─────────────────────────────────────────────────────────
LOCATION_ALIASES = {
    "downloads":  "~/Downloads",
    "desktop":    "~/Desktop",
    "documents":  "~/Documents",
    "home":       "~",
    "pictures":   "~/Pictures",
    "music":      "~/Music",
    "movies":     "~/Movies",
    "tmp":        "/tmp",
    "temp":       "/tmp",
}

# ── File type templates ─────────────────────────────────────────────────────
FILE_TEMPLATES = {
    ".py":    "# Python file\n\n",
    ".java":  "public class {classname} {{\n    public static void main(String[] args) {{\n        // TODO\n    }}\n}}\n",
    ".js":    "// JavaScript file\n\n",
    ".ts":    "// TypeScript file\n\n",
    ".html":  "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\"/>\n  <title>{name}</title>\n</head>\n<body>\n\n</body>\n</html>\n",
    ".css":   "/* CSS file */\n\n",
    ".cpp":   "#include <iostream>\nusing namespace std;\n\nint main() {{\n    return 0;\n}}\n",
    ".c":     "#include <stdio.h>\n\nint main() {{\n    return 0;\n}}\n",
    ".rb":    "# Ruby file\n\n",
    ".sh":    "#!/bin/bash\n\n",
    ".md":    "# {name}\n\n",
    ".txt":   "",
    ".json":  "{{}}\n",
    ".yaml":  "# YAML file\n",
    ".yml":   "# YAML file\n",
    ".csv":   "",
    ".env":   "# Environment variables\n",
    ".toml":  "# TOML config\n",
    ".sql":   "-- SQL file\n\n",
    ".r":     "# R script\n\n",
    ".swift": "// Swift\nimport Foundation\n\n",
    ".kt":    "// Kotlin\nfun main() {{\n    // TODO\n}}\n",
    ".go":    "package main\n\nimport \"fmt\"\n\nfunc main() {{\n    fmt.Println(\"Hello\")\n}}\n",
    ".rs":    "fn main() {{\n    println!(\"Hello\");\n}}\n",
}

LANG_ALIASES = {
    "python":      ".py",
    "java":        ".java",
    "javascript":  ".js",
    "js":          ".js",
    "typescript":  ".ts",
    "ts":          ".ts",
    "html":        ".html",
    "css":         ".css",
    "cpp":         ".cpp",
    "c++":         ".cpp",
    "c":           ".c",
    "ruby":        ".rb",
    "bash":        ".sh",
    "shell":       ".sh",
    "markdown":    ".md",
    "text":        ".txt",
    "json":        ".json",
    "yaml":        ".yaml",
    "sql":         ".sql",
    "r":           ".r",
    "swift":       ".swift",
    "kotlin":      ".kt",
    "go":          ".go",
    "rust":        ".rs",
}


def _resolve_location(loc: str) -> str:
    """Resolve a location alias or path to an absolute path."""
    loc_lower = loc.strip().lower()
    for alias, path in LOCATION_ALIASES.items():
        if alias in loc_lower:
            return os.path.expanduser(path)
    expanded = os.path.expanduser(loc.strip())
    if os.path.isdir(expanded):
        return expanded
    expanded_cap = os.path.expanduser(loc.strip().capitalize())
    if os.path.isdir(expanded_cap):
        return expanded_cap
    return os.path.expanduser('~/Desktop')


def _resolve_file(filename: str, hint_dir: str = None) -> str:
    """Find a file by name. Searches hint_dir first, then common locations."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    search_dirs = []
    if hint_dir and os.path.isdir(hint_dir):
        search_dirs.append(hint_dir)

    search_dirs += [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Pictures"),
        os.path.expanduser("~/Music"),
        os.path.expanduser("~/Movies"),
        os.path.expanduser("~"),
    ]

    filename_lower = filename.lower()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
        try:
            for entry in os.listdir(d):
                if entry.lower() == filename_lower:
                    return os.path.join(d, entry)
        except PermissionError:
            continue
        try:
            for subdir in os.listdir(d):
                subpath = os.path.join(d, subdir)
                if not os.path.isdir(subpath):
                    continue
                try:
                    for entry in os.listdir(subpath):
                        if entry.lower() == filename_lower:
                            return os.path.join(subpath, entry)
                except PermissionError:
                    continue
        except PermissionError:
            continue

    return None


def _move_file(command_lower: str) -> str:
    """Use LLM to understand move intent."""
    import shutil
    try:
        response = ollama.chat(
            model="mistral:7b-instruct-q4_0",
            messages=[{
                "role": "user",
                "content": (
                    f"Extract move intent from this command.\n"
                    f"Command: \"{command_lower}\"\n"
                    f"Reply with ONLY a JSON object:\n"
                    f"  name: the file or folder name to move\n"
                    f"  from: source location (downloads/desktop/documents/home), null if not mentioned\n"
                    f"  to: destination location (downloads/desktop/documents/home)\n"
                    f"Examples:\n"
                    f"  \"move main.py from downloads to documents\" → {{\"name\": \"main.py\", \"from\": \"downloads\", \"to\": \"documents\"}}\n"
                    f"  \"move main.py to desktop\" → {{\"name\": \"main.py\", \"from\": null, \"to\": \"desktop\"}}\n"
                    f"Only the JSON. No explanation."
                )
            }],
            options={"temperature": 0.0, "num_predict": 60}
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        params   = json.loads(raw)
        name     = (params.get("name") or "").strip()
        from_loc = (params.get("from") or "").strip().lower()
        to_loc   = (params.get("to") or "").strip().lower()
        print(f"[DEBUG] Move intent: name={name}, from={from_loc}, to={to_loc}")
    except Exception as e:
        print(f"[DEBUG] LLM move parse failed: {e}, using regex fallback")
        return _move_file_fallback(command_lower)

    if not name or not to_loc:
        return None

    to_dir = _resolve_location(to_loc)

    if from_loc:
        from_dir = _resolve_location(from_loc)
        src = os.path.join(from_dir, name)
        if not os.path.exists(src):
            src = _resolve_file(name, from_dir)
    else:
        src = _resolve_file(name)

    if not src or not os.path.exists(src):
        return f"File or folder '{name}' not found{(' in ' + from_loc) if from_loc else ''}."

    if not os.path.isdir(to_dir):
        return f"Destination '{to_loc}' not found."

    dst = os.path.join(to_dir, os.path.basename(src))
    import shutil
    shutil.move(src, dst)
    from_label = from_loc or os.path.dirname(src).split("/")[-1]
    return f"📦 Moved '{name}' from {from_label} → {to_loc}."


def _move_file_fallback(cmd: str) -> str:
    """Regex fallback for _move_file when LLM unavailable."""
    import shutil
    m = re.search(
        r'move\s+(?:file\s+|folder\s+)?"?([\w.\- ]+?)"?\s+from\s+(?:the\s+)?([\w~/]+)\s+to\s+(?:the\s+)?([\w~/]+)',
        cmd
    )
    if m:
        name, from_alias, to_alias = m.group(1).strip(), m.group(2), m.group(3)
        from_dir = _resolve_location(from_alias)
        to_dir   = _resolve_location(to_alias)
        src = os.path.join(from_dir, name)
        if not os.path.exists(src): src = _resolve_file(name, from_dir)
        if src and os.path.exists(src):
            shutil.move(src, os.path.join(to_dir, os.path.basename(src)))
            return f"📦 Moved '{name}' from {from_alias} → {to_alias}."
    m = re.search(r'move\s+(?:file\s+|folder\s+)?"?([\w.\- ]+?)"?\s+to\s+(?:the\s+)?([\w~/]+)', cmd)
    if m:
        name, to_alias = m.group(1).strip(), m.group(2)
        to_dir = _resolve_location(to_alias)
        src = _resolve_file(name)
        if src:
            import shutil
            shutil.move(src, os.path.join(to_dir, os.path.basename(src)))
            return f"📦 Moved '{name}' → {to_alias}."
    return None


def _delete(command_lower: str) -> str:
    """Use LLM to understand what the user wants to delete."""
    import shutil

    try:
        response = ollama.chat(
            model="mistral:7b-instruct-q4_0",
            messages=[{
                "role": "user",
                "content": (
                    f"Extract delete intent from this command.\n"
                    f"Command: \"{command_lower}\"\n"
                    f"Reply with ONLY a JSON object:\n"
                    f"  type: \"file\" | \"folder\" | \"app\"\n"
                    f"  name: exact name to delete (no extra words)\n"
                    f"  location: location if mentioned (downloads/desktop/documents/home), else null\n"
                    f"Only the JSON. No explanation."
                )
            }],
            options={"temperature": 0.0, "num_predict": 60}
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        params = json.loads(raw)
        dtype    = params.get("type", "folder")
        name     = params.get("name", "").strip()
        location = params.get("location")
        import re as _re
        name = _re.sub(r'^name[_\s]+', '', name, flags=_re.IGNORECASE).strip()
        print(f"[DEBUG] Delete intent: type={dtype}, name={name}, location={location}")
    except Exception as e:
        print(f"[DEBUG] LLM delete parse failed: {e}, falling back to regex")
        dtype, name, location = _delete_fallback_parse(command_lower)

    if not name:
        return None

    if dtype == "app":
        search_paths = ["/Applications", os.path.expanduser("~/Applications"), "/System/Applications"]
        for path in search_paths:
            if not os.path.exists(path):
                continue
            for entry in os.listdir(path):
                if entry.endswith(".app"):
                    entry_name = entry.replace(".app", "").lower()
                    if name.lower() in entry_name or entry_name in name.lower():
                        full_path = os.path.join(path, entry)
                        subprocess.run(["osascript", "-e",
                            f'tell application "Finder" to delete POSIX file "{full_path}"'])
                        return f"🗑 Moved '{entry.replace('.app','')}' to Trash."
        return f"App '{name}' not found in /Applications."

    if dtype == "folder":
        search_dirs = (
            [_resolve_location(location)]
            if location else
            [
                os.path.expanduser("~/Desktop"),
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/Documents"),
                os.path.expanduser("~"),
            ]
        )
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                if entry.lower() == name.lower() and os.path.isdir(os.path.join(d, entry)):
                    full_path = os.path.join(d, entry)
                    import shutil
                    shutil.rmtree(full_path)
                    return f"🗑 Deleted folder '{entry}' from {d.split('/')[-1]}."
        loc_label = location or "common folders"
        return f"Folder '{name}' not found in {loc_label}."

    hint_dir = _resolve_location(location) if location else None
    src = _resolve_file(name, hint_dir)
    if not src:
        return f"File '{name}' not found."
    if os.path.isdir(src):
        import shutil
        shutil.rmtree(src)
    else:
        os.remove(src)
    return f"🗑 Deleted '{name}' from {os.path.dirname(src).split('/')[-1]}."


def _delete_fallback_parse(cmd: str):
    """Regex fallback for _delete when LLM is unavailable."""
    app_m = re.search(
        r'(?:uninstall|delete\s+app|remove\s+app|delete\s+application|remove\s+application)\s+"?([\w\s\-.]+?)"?\s*$',
        cmd
    )
    if app_m:
        return "app", app_m.group(1).strip(), None

    folder_m = (
        re.search(
            r'(?:delete|remove|trash)\s+(?:the\s+)?folder\s+"?([\w\-.\s]+?)"?(?:\s+(?:from|in|on)\s+(?:the\s+)?([\w~/]+))?\s*$',
            cmd
        ) or
        re.search(
            r'(?:delete|remove|trash)\s+(?:the\s+)?"?([\w\-.\s]+?)"?\s+folder(?:\s+(?:from|in|on)\s+(?:the\s+)?([\w~/]+))?\s*$',
            cmd
        )
    )
    if folder_m:
        return "folder", folder_m.group(1).strip(), folder_m.group(2)

    file_m = re.search(
        r'(?:delete|remove|trash)\s+(?:the\s+)?(?:file\s+)?"?([\w\-.]+\.[\w]+)"?(?:\s+(?:from|in|on)\s+(?:the\s+)?([\w~/]+))?',
        cmd
    )
    if file_m:
        return "file", file_m.group(1).strip(), file_m.group(2)

    return "folder", "", None


def _create_calendar_event(title: str, event_date: datetime, event_time: str = None) -> bool:
    """Create a named Calendar event on a specific date."""
    if event_time:
        try:
            t = event_time.strip().lower()
            t = re.sub(r'(\d)(am|pm)', r'\1 \2', t)
            for fmt in ['%I:%M %p', '%I %p', '%H:%M']:
                try:
                    parsed = datetime.strptime(t, fmt)
                    start_dt = event_date.replace(
                        hour=parsed.hour, minute=parsed.minute, second=0
                    )
                    break
                except ValueError:
                    continue
            else:
                start_dt = event_date.replace(hour=9, minute=0, second=0)
        except Exception:
            start_dt = event_date.replace(hour=9, minute=0, second=0)
    else:
        start_dt = event_date.replace(hour=9, minute=0, second=0)

    end_dt   = start_dt + timedelta(hours=1)
    date_str = start_dt.strftime('%A, %d %B %Y at %I:%M:%S %p')
    end_str  = end_dt.strftime('%A, %d %B %Y at %I:%M:%S %p')

    script = (
        f'tell application "Calendar"\n'
        f'  tell calendar "Home"\n'
        f'    set newEvent to make new event with properties {{'
        f'summary:"{title}", '
        f'start date:date "{date_str}", '
        f'end date:date "{end_str}"}}\n'
        f'    tell newEvent\n'
        f'      make new sound alarm with properties {{trigger interval:-1}}\n'
        f'    end tell\n'
        f'  end tell\n'
        f'end tell'
    )
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode != 0:
        script2 = (
            f'tell application "Calendar"\n'
            f'  set theCal to first calendar\n'
            f'  tell theCal\n'
            f'    set newEvent to make new event with properties {{'
            f'summary:"{title}", '
            f'start date:date "{date_str}", '
            f'end date:date "{end_str}"}}\n'
            f'    tell newEvent\n'
            f'      make new sound alarm with properties {{trigger interval:-1}}\n'
            f'    end tell\n'
            f'  end tell\n'
            f'end tell'
        )
        result2 = subprocess.run(['osascript', '-e', script2], capture_output=True)
        return result2.returncode == 0
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# REMINDER HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def _set_reminder_in_reminders_app(title: str, due_datetime) -> bool:
    """
    Create a reminder in macOS Reminders.app via AppleScript.
    Returns True on success.
    """
    date_str = due_datetime.strftime('%A, %d %B %Y at %I:%M:%S %p')

    script = (
        f'tell application "Reminders"\n'
        f'  set newReminder to make new reminder with properties {{'
        f'name:"{title}", '
        f'due date:date "{date_str}", '
        f'remind me date:date "{date_str}"}}\n'
        f'end tell'
    )
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    if result.returncode == 0:
        return True

    script2 = (
        f'tell application "Reminders"\n'
        f'  set theList to first list\n'
        f'  make new reminder at end of reminders of theList with properties {{'
        f'name:"{title}", '
        f'due date:date "{date_str}", '
        f'remind me date:date "{date_str}"}}\n'
        f'end tell'
    )
    result2 = subprocess.run(['osascript', '-e', script2], capture_output=True)
    return result2.returncode == 0


def _handle_reminder(command_lower: str) -> str | None:
    """
    Parse and set a reminder from natural language.
    Priority: Reminders.app → Calendar.app → threading.Timer
    """
    c = command_lower.strip()

    if not re.search(r'\b(remind|reminder)\b', c):
        return None

    task    = None
    seconds = None
    label   = None

    clock_pat = re.search(
        r'\bat\s+'
        r'(\d{1,2}(?:[: ]\d{2})?\s*(?:am|pm)?)',
        c
    )
    if clock_pat:
        time_raw = clock_pat.group(1).strip()
        seconds, label = _parse_time_str(time_raw)

        if seconds is not None:
            remainder = c[:clock_pat.start()] + c[clock_pat.end():]
            task = re.sub(
                r'^\s*(?:set\s+(?:a\s+)?)?remind(?:er|me)?\s*'
                r'(?:me\s+)?(?:to\s+|for\s+)?',
                '', remainder, flags=re.IGNORECASE
            ).strip()
            task = re.sub(r'\s+', ' ', task).strip(" ,.-")

            if not task:
                task = "reminder"

    if seconds is None:
        m = re.search(
            r'\bremind\s+me\s+(?:to\s+)?(.+?)\s+in\s+(\d[\w\s]+)'
            r'|\breminder\s+(?:to\s+|for\s+)?(.+?)\s+in\s+(\d[\w\s]+)',
            c
        )
        if m:
            task_raw = (m.group(1) or m.group(3) or "").strip()
            time_raw = (m.group(2) or m.group(4) or "").strip()
            seconds, label = _parse_time_str(time_raw)
            if seconds:
                task = task_raw

        if seconds is None:
            m = re.search(
                r'\bremind\s+me\s+in\s+(\d[\w\s]+?)\s+to\s+(.+)',
                c
            )
            if m:
                time_raw = m.group(1).strip()
                task_raw = m.group(2).strip()
                seconds, label = _parse_time_str(time_raw)
                if seconds:
                    task = task_raw

        if seconds is None:
            m = re.search(
                r'\b(?:set\s+(?:a\s+)?)?reminder\s+(?:to\s+|for\s+)?(.+?)\s+in\s+(\d[\w\s]+)',
                c
            )
            if m:
                task_raw = m.group(1).strip()
                time_raw = m.group(2).strip()
                seconds, label = _parse_time_str(time_raw)
                if seconds:
                    task = task_raw

    if seconds is not None and task is not None:
        due   = datetime.now() + timedelta(seconds=seconds)
        title = f"🔔 {task.capitalize()}"

        if _set_reminder_in_reminders_app(title, due):
            return f"🔔 Reminder set: '{task}' at {label} — saved to Reminders app."

        if _set_reminder_app(title, due):
            return (
                f"🔔 Reminder set: '{task}' at {label} — saved to Calendar app "
                f"(Reminders app wasn't accessible)."
            )

        threading.Timer(seconds, _notify, args=["🔔 Reminder", task.capitalize()]).start()
        return (
            f"🔔 Reminder set: '{task}' in {label}. "
            f"⚠️ Note: only works while the server is running — "
            f"couldn't reach Reminders app or Calendar app."
        )

    return (
        "Sorry, I couldn't understand the time in your reminder. "
        "Try formats like: 'at 10:30', 'at 10 30', 'at 5pm', or 'in 30 minutes'."
    )


def _find_named_folder(folder_name: str) -> str | None:
    """
    Search for a folder by name across common user directories.
    Returns the absolute path if found, None otherwise.
    """
    folder_name_clean = folder_name.strip().lower().replace(" ", "_")

    search_roots = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Pictures"),
        os.path.expanduser("~/Music"),
        os.path.expanduser("~/Movies"),
        os.path.expanduser("~"),
    ]

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        try:
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full) and entry.lower().replace(" ", "_") == folder_name_clean:
                    return full
        except PermissionError:
            continue

        try:
            for sub in os.listdir(root):
                sub_path = os.path.join(root, sub)
                if not os.path.isdir(sub_path):
                    continue
                try:
                    for entry in os.listdir(sub_path):
                        full = os.path.join(sub_path, entry)
                        if os.path.isdir(full) and entry.lower().replace(" ", "_") == folder_name_clean:
                            return full
                except PermissionError:
                    continue
        except PermissionError:
            continue

    return None


def _create_file(command_lower: str, original: str) -> str:
    """Use LLM to understand what the user wants to create — file or folder."""
    try:
        response = ollama.chat(
            model="mistral:7b-instruct-q4_0",
            messages=[{
                "role": "user",
                "content": (
                    f"Extract create intent from this command.\n"
                    f"Command: \"{command_lower}\"\n"
                    f"Reply with ONLY a JSON object:\n"
                    f"  type: \"file\" | \"folder\"\n"
                    f"  name: the file or folder name (without extension)\n"
                    f"  language: programming language or file type if file, else null\n"
                    f"  extension: file extension if explicitly given, else null\n"
                    f"  location: target folder location if mentioned, else null\n"
                    f"Only the JSON. No explanation."
                )
            }],
            options={"temperature": 0.0, "num_predict": 80}
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        params = json.loads(raw)
        ctype    = params.get("type", "folder")
        name     = (params.get("name") or "").strip()
        language = (params.get("language") or "").strip().lower()
        extension= (params.get("extension") or "").strip().lower()
        location = (params.get("location") or "").strip().lower()
        import re as _re
        name = _re.sub(r'^name[_\s]+', '', name, flags=_re.IGNORECASE).strip()
        print(f"[DEBUG] Create intent: type={ctype}, name={name}, lang={language}, ext={extension}, loc={location}")
    except Exception as e:
        print(f"[DEBUG] LLM create parse failed: {e}, using regex fallback")
        return _create_file_fallback(command_lower, original)

    if not name:
        return None

    _BOGUS = {"in","on","at","the","a","an","mac","macos","terminal","finder",
              "system","computer","desktop","here","this","that","windows","linux"}
    if name.lower() in _BOGUS:
        return None

    if ctype == "folder":
        folder_name = name.replace(" ", "_")
        loc_path    = _resolve_location(location) if location else os.path.expanduser("~/Desktop")
        folder_path = os.path.join(loc_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        loc_label = location or "Desktop"
        return f"📁 Folder '{folder_name}' created in {loc_label.title()}."

    TEMPLATES = {
        "python":     ("py",   "# {name}.py\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n"),
        "java":       ("java", "public class {name} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello World\");\n    }}\n}}\n"),
        "javascript": ("js",   "// {name}.js\n\nfunction main() {{\n    console.log(\"Hello World\");\n}}\n\nmain();\n"),
        "typescript": ("ts",   "// {name}.ts\n\nfunction main(): void {{\n    console.log(\"Hello World\");\n}}\n\nmain();\n"),
        "html":       ("html", "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>{name}</title>\n</head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>\n"),
        "css":        ("css",  "/* {name}.css */\n\nbody {{\n    margin: 0;\n    padding: 0;\n    font-family: sans-serif;\n}}\n"),
        "bash":       ("sh",   "#!/bin/bash\n# {name}.sh\n\necho \"Hello World\"\n"),
        "ruby":       ("rb",   "# {name}.rb\n\nputs \"Hello World\"\n"),
        "go":         ("go",   "package main\n\nimport \"fmt\"\n\nfunc main() {{\n    fmt.Println(\"Hello World\")\n}}\n"),
        "rust":       ("rs",   "fn main() {{\n    println!(\"Hello World!\");\n}}\n"),
        "cpp":        ("cpp",  "#include <iostream>\n\nint main() {{\n    std::cout << \"Hello World\" << std::endl;\n    return 0;\n}}\n"),
        "markdown":   ("md",   "# {name}\n\n## Overview\n\n"),
        "json":       ("json", "{{\n    \"name\": \"{name}\"\n}}\n"),
        "yaml":       ("yaml", "name: {name}\n"),
        "sql":        ("sql",  "-- {name}.sql\n\nSELECT * FROM table_name;\n"),
        "text":       ("txt",  ""),
        "r":          ("r",    "# {name}.r\n\ncat(\"Hello World\\n\")\n"),
    }

    if extension and not extension.startswith("."):
        extension = "." + extension

    if not extension and language and language in TEMPLATES:
        ext_str, template = TEMPLATES[language]
        extension = "." + ext_str
    elif extension:
        ext_to_lang = {".py":"python",".js":"javascript",".ts":"typescript",
                       ".java":"java",".html":"html",".css":"css",".sh":"bash",
                       ".rb":"ruby",".go":"go",".rs":"rust",".cpp":"cpp",
                       ".md":"markdown",".json":"json",".yaml":"yaml",".yml":"yaml",
                       ".sql":"sql",".txt":"text",".r":"r"}
        lang_key = ext_to_lang.get(extension, "text")
        _, template = TEMPLATES.get(lang_key, ("txt", ""))
    else:
        extension = ".txt"
        template  = ""

    filename = name.replace(" ", "_") + extension

    loc_path = None
    loc_label = location or "Desktop"

    if location:
        resolved = _resolve_location(location)
        if resolved != os.path.expanduser("~/Desktop") or "desktop" in location:
            loc_path = resolved
        else:
            found = _find_named_folder(location)
            if found:
                print(f"[DEBUG] Named folder found: {found}")
                loc_path  = found
                loc_label = os.path.basename(found)
            else:
                new_folder = os.path.join(os.path.expanduser("~/Desktop"), location.replace(" ", "_"))
                os.makedirs(new_folder, exist_ok=True)
                loc_path  = new_folder
                loc_label = location
                print(f"[DEBUG] Named folder not found — created it at {new_folder}")

    if loc_path is None:
        loc_path = os.path.expanduser("~/Desktop")

    file_path    = os.path.join(loc_path, filename)
    file_content = template.format(name=name.replace(" ", "_")) if template else ""
    with open(file_path, "w") as f:
        f.write(file_content)

    return f"✅ Created '{filename}' in '{loc_label}'  ({loc_path})."


def _create_file_fallback(command_lower: str, original: str) -> str:
    """Regex fallback for _create_file when LLM is unavailable."""
    cmd = command_lower

    folder_m = (
        re.search(
            r'(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?folder\s+'
            r'(?:called\s+|named\s+|with\s+name\s+|with\s+the\s+name\s+)?'
            r'"?([\w\-. ]+?)"?\s*(?:\b(?:in|on|at|inside)\b|$)',
            cmd
        ) or
        re.search(
            r'(?:create|make)\s+(?:a\s+)?(?:new\s+)?"?([\w\-. ]+?)"?\s+folder'
            r'(?:\s+(?:in|on|at|inside)\s+(?:the\s+)?[\w]+)?\s*$',
            cmd
        )
    )
    loc_m = re.search(r'\b(?:in|on|at|inside)\s+(?:the\s+)?([\w~/. ]+?)(?:\s*$|\s+(?:folder|directory))', cmd)

    if folder_m:
        folder_name = folder_m.group(1).strip().replace(" ", "_")
        loc_alias   = loc_m.group(1).strip() if loc_m else "desktop"
        location    = _resolve_location(loc_alias)
        os.makedirs(os.path.join(location, folder_name), exist_ok=True)
        return f"📁 Folder '{folder_name}' created in {loc_alias.title()}."

    file_m = re.search(
        r'(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?'
        r'(python|java|javascript|js|typescript|ts|html|css|cpp|ruby|bash|go|rust'
        r'|markdown|json|yaml|sql|txt|text|r)\s+'
        r'(?:file\s+)?(?:called|named|with\s+name\s+)?'
        r'"?([\w\-. ]+?)"?'
        r'(?:\s+(?:in|inside|into|at)\s+(?:the\s+)?(.+?))?$',
        cmd
    )
    if file_m:
        language  = file_m.group(1).strip().lower()
        file_name = file_m.group(2).strip()
        loc_str   = (file_m.group(3) or "").strip().lower()

        EXT_MAP = {
            "python":"py","java":"java","javascript":"js","js":"js",
            "typescript":"ts","ts":"ts","html":"html","css":"css","cpp":"cpp",
            "ruby":"rb","bash":"sh","go":"go","rust":"rs","markdown":"md",
            "json":"json","yaml":"yaml","sql":"sql","txt":"txt","text":"txt","r":"r",
        }
        ext      = "." + EXT_MAP.get(language, "txt")
        filename = file_name.replace(" ", "_") + ext

        loc_path  = None
        loc_label = loc_str or "Desktop"

        if loc_str:
            resolved = _resolve_location(loc_str)
            if resolved != os.path.expanduser("~/Desktop") or "desktop" in loc_str:
                loc_path = resolved
            else:
                found = _find_named_folder(loc_str)
                if found:
                    loc_path  = found
                    loc_label = os.path.basename(found)
                else:
                    new_folder = os.path.join(os.path.expanduser("~/Desktop"), loc_str.replace(" ", "_"))
                    os.makedirs(new_folder, exist_ok=True)
                    loc_path  = new_folder
                    loc_label = loc_str

        if loc_path is None:
            loc_path = os.path.expanduser("~/Desktop")

        open(os.path.join(loc_path, filename), "w").close()
        return f"✅ Created '{filename}' in '{loc_label}'  ({loc_path})."

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE  ← NEW
# ═══════════════════════════════════════════════════════════════════════════════

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

def _play_youtube(query: str, intent: str = "play") -> str:
    """
    Handle YouTube requests based on intent:
      - intent='play'   → find top video and open it directly
      - intent='search' → open YouTube search results page
      - intent='open'   → just open YouTube homepage
    """
    query = query.strip()

    # ── Just open YouTube homepage ────────────────────────────────────
    if intent == "open" or not query:
        webbrowser.open("https://www.youtube.com")
        return "▶️ Opening YouTube..."

    # ── Search results page (let user choose) ─────────────────────────
    if intent == "search":
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        webbrowser.open(url)
        return f"🔍 Searching YouTube for: '{query}'"

    # ── Play: find top result via API and open directly ───────────────
    if YOUTUBE_API_KEY:
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part":       "snippet",
                    "q":          query,
                    "type":       "video",
                    "maxResults": 1,
                    "key":        YOUTUBE_API_KEY,
                },
                timeout=6,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    video_id    = items[0]["id"]["videoId"]
                    video_title = items[0]["snippet"]["title"]
                    url         = f"https://www.youtube.com/watch?v={video_id}"
                    webbrowser.open(url)
                    return f"▶️ Playing: {video_title}\n🔗 {url}"
            print(f"[DEBUG] YouTube API error: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"[DEBUG] YouTube API failed: {e}")

    # ── Fallback: open search page if API fails ───────────────────────
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    webbrowser.open(url)
    return f"▶️ Opening YouTube search for: '{query}'\n🔗 {url}"


def system_command(command: str) -> str:
    """
    Intent-based dispatcher.
    Returns result string or None if command not recognised.
    """
    command_lower = command.lower().strip()
    c = command_lower

    # ── Websites dict ─────────────────────────────────────────────────────────
    websites = {
        "youtube":       "https://www.youtube.com",
        "google":        "https://www.google.com",
        "gmail":         "https://www.gmail.com",
        "github":        "https://www.github.com",
        "netflix":       "https://www.netflix.com",
        "twitter":       "https://www.twitter.com",
        "instagram":     "https://www.instagram.com",
        "linkedin":      "https://www.linkedin.com",
        "whatsapp web":  "https://web.whatsapp.com",
        "chatgpt":       "https://chat.openai.com",
        "amazon":        "https://www.amazon.in",
        "flipkart":      "https://www.flipkart.com",
        "reddit":        "https://www.reddit.com",
        "stackoverflow": "https://www.stackoverflow.com",
    }

    # ── Folders dict ──────────────────────────────────────────────────────────
    folders = {
        "downloads": "~/Downloads",
        "desktop":   "~/Desktop",
        "documents": "~/Documents",
        "home":      "~",
        "pictures":  "~/Pictures",
        "music":     "~/Music",
        "movies":    "~/Movies",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # TIME & DATE
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(
        r"\b(what'?s?\s+the\s+time|what\s+is\s+the\s+time|current\s+time"
        r"|show\s+time|tell\s+(me\s+)?the\s+time|time\s+now"
        r"|what\s+time\s+is\s+it)\b",
        c
    ):
        now = datetime.now()
        return f"🕐 Current time: {now.strftime('%I:%M %p')} ({now.strftime('%A, %d %B %Y')})"

    if re.search(
        r"\b(what'?s?\s+the\s+date|what\s+is\s+the\s+date|current\s+date"
        r"|show\s+date|today'?s?\s+date|what\s+is\s+today"
        r"|what\s+day\s+is\s+(it|today)|tell\s+(me\s+)?the\s+date)\b",
        c
    ):
        now = datetime.now()
        return f"📅 Today is {now.strftime('%A, %d %B %Y')}"

    # ══════════════════════════════════════════════════════════════════════════
    # REMINDER
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r'\b(remind|reminder)\b', c):
        result = _handle_reminder(c)
        if result:
            return result

    # ══════════════════════════════════════════════════════════════════════════
    # TIMER
    # ══════════════════════════════════════════════════════════════════════════
    timer_match = re.search(
        r"\b(?:set\s+(?:a\s+)?|start\s+(?:a\s+)?)?timer\s+(?:for\s+)?(.+)"
        r"|\b(\d+)\s*(?:min(?:ute)?s?|hour?s?|sec(?:ond)?s?)\s+timer\b",
        c
    )
    if timer_match:
        time_str = (timer_match.group(1) or timer_match.group(0)).strip()
        seconds, label = _parse_time_str(time_str)
        if seconds:
            method = _set_timer_or_alarm(seconds, label, kind="Timer")
            if method == "calendar":
                return f"⏱ Timer set for {label} — added to Calendar app. It will ring when done."
            return f"⏱ Timer set for {label}. You'll get a notification when it's done."
        return f"Couldn't understand the time '{time_str}'. Try: 'timer 5 minutes' or 'timer 1 hour 30 minutes'."

    # ══════════════════════════════════════════════════════════════════════════
    # ALARM
    # ══════════════════════════════════════════════════════════════════════════
    alarm_match = re.search(
        r"\b(?:set\s+(?:an?\s+)?)?alarm\s+(?:for\s+|at\s+|in\s+)?(.+)"
        r"|\bwake\s+me\s+up\s+at\s+(.+)",
        c
    )
    if alarm_match:
        time_str = (alarm_match.group(1) or alarm_match.group(2) or "").strip()
        seconds, label = _parse_time_str(time_str)
        if seconds:
            method = _set_timer_or_alarm(seconds, label, kind="Alarm")
            if method == "calendar":
                return f"⏰ Alarm set for {label} — added to Calendar app. It will ring even if the server is closed."
            return f"⏰ Alarm set for {label}. Note: only works while the server is running."
        return f"Couldn't understand the time '{time_str}'. Try: 'alarm at 7:30 AM' or 'alarm in 2 hours'."

    # ══════════════════════════════════════════════════════════════════════════
    # EMPTY TRASH
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(empty\s+(the\s+)?trash(\s+bin)?|clear\s+(the\s+)?trash)\b", c):
        subprocess.run(["osascript", "-e", 'tell application "Finder" to empty trash'])
        return "🗑 Trash emptied."

    # ══════════════════════════════════════════════════════════════════════════
    # DELETE / REMOVE / UNINSTALL
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(delete|remove|uninstall|trash)\b", c):
        result = _delete(c)
        if result:
            return result

    # ══════════════════════════════════════════════════════════════════════════
    # CALENDAR EVENT / DATE MARKING
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(
        r"\b(mark\s+.+\s+(for|as)|add\s+(event|birthday|meeting|appointment)"
        r"|create\s+event|schedule\s+.+\s+on|save\s+date|remember\s+this\s+date)\b",
        c
    ):
        result = _handle_event(c)
        if result:
            return result

    # ══════════════════════════════════════════════════════════════════════════
    # MOVE FILE / FOLDER
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\bmove\s+\S+", c):
        result = _move_file(c)
        if result:
            return result

    # ══════════════════════════════════════════════════════════════════════════
    # CREATE FILE / FOLDER
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(
        r"\b(create|make|new)\b.{0,20}\b(file|folder|directory|dir"
        r"|python|java|javascript|js|typescript|ts|html|css|cpp|ruby|bash|go|rust"
        r"|markdown|json|yaml|sql|txt|text)\b"
        r"|\b(create|make|new)\s+(?:a\s+)?(?:new\s+)?folder\b",
        c
    ):
        result = _create_file(c, command)
        if result:
            return result

    # ══════════════════════════════════════════════════════════════════════════
    # OPEN FILE IN APP
    # ══════════════════════════════════════════════════════════════════════════
    file_in_app = re.search(
        r"\bopen\s+(.+?)\s+in\s+(vs\s?code|code|vscode|sublime|pycharm|xcode"
        r"|word|excel|notepad|textedit|numbers|pages|keynote|powerpoint|atom)",
        c
    )
    if file_in_app:
        return _open_file_in_app(file_in_app.group(1).strip(), file_in_app.group(2).strip())

    file_only = re.search(
        r"\bopen\s+([\w\-. /~]+\.(?:py|js|ts|txt|pdf|csv|json|yaml|yml|md"
        r"|html|css|cpp|c|java|rb|sh|env|toml|ini|log|docx|xlsx|pptx|png|jpg|jpeg|gif|mp4|mp3|zip))\b",
        c
    )
    if file_only:
        return _open_file_in_app(file_only.group(1).strip())

    # ══════════════════════════════════════════════════════════════════════════
    # OPEN URL
    # ══════════════════════════════════════════════════════════════════════════
    url_match = re.search(
        r"(?:open|go\s+to|visit|browse)\s+"
        r"((?:https?://)?[\w.-]+\.(?:com|in|org|net|io|co|ai|edu)(?:/\S*)?)",
        c
    )
    if url_match:
        return open_url(url_match.group(1))

    # ══════════════════════════════════════════════════════════════════════════
    # PLAY YOUTUBE VIDEO  ← NEW
    # ══════════════════════════════════════════════════════════════════════════
    # Intent: open youtube only
    if re.search(r'^open\s+youtube\s*$', c):
        return _play_youtube("", intent="open")

    # Intent: search (user wants to browse results)
    yt_search = re.search(
        r'\bopen\s+youtube\s+and\s+(?:search|look\s+up|search\s+for)\s+(?:for\s+)?(.+)'
        r'|\bsearch\s+(?:for\s+)?(.+?)\s+(?:on|in)\s+youtube\b'
        r'|\byoutube\s+search\s+(?:for\s+)?(.+)',
        c
    )
    if yt_search:
        query = next((g for g in yt_search.groups() if g), "").strip()
        query = re.sub(r'\s+(?:on|in)\s+youtube\s*$', '', query, flags=re.IGNORECASE).strip()
        if query:
            return _play_youtube(query, intent="search")

    # Intent: play (user wants to watch something specific)
    yt_play = re.search(
        r'\b(?:play|put\s+on)\s+(?:on\s+youtube\s+)?(.+?)\s+(?:on\s+youtube|in\s+youtube)\b'
        r'|\b(?:play|open)\s+youtube\s+(?:video\s+(?:of\s+|for\s+)?)?(.+)'
        r'|\bopen\s+youtube\s+and\s+(?:play|watch|find)\s+(?:for\s+)?(.+)'
        r'|\bwatch\s+(.+?)\s+(?:on|in)\s+youtube\b',
        c
    )
    if yt_play:
        query = next((g for g in yt_play.groups() if g), "").strip()
        query = re.sub(r'\s+(?:on|in)\s+youtube\s*$', '', query, flags=re.IGNORECASE).strip()
        if query:
            return _play_youtube(query, intent="play")

    # ══════════════════════════════════════════════════════════════════════════
    # VOLUME
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(increase\s+volume|volume\s+up|turn\s+up\s+(the\s+)?volume|louder)\b", c):
        subprocess.run(["osascript", "-e",
            "set volume output volume (output volume of (get volume settings) + 10)"])
        return "🔊 Volume increased."

    if re.search(r"\b(decrease\s+volume|volume\s+down|turn\s+down\s+(the\s+)?volume|lower\s+(the\s+)?volume|quieter)\b", c):
        subprocess.run(["osascript", "-e",
            "set volume output volume (output volume of (get volume settings) - 10)"])
        return "🔉 Volume decreased."

    if re.search(r"\bunmute\b", c):
        subprocess.run(["osascript", "-e", "set volume output muted false"])
        return "🔊 Unmuted."

    if re.search(r"\b(mute|silence(\s+the\s+volume)?)\b", c):
        subprocess.run(["osascript", "-e", "set volume output muted true"])
        return "🔇 Muted."

    # ══════════════════════════════════════════════════════════════════════════
    # BRIGHTNESS
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(increase\s+brightness|brightness\s+up|brighter|make\s+(the\s+)?screen\s+brighter)\b", c):
        subprocess.run(["osascript", "-e",
            'tell application "System Events" to key code 144'])
        return "☀️ Brightness increased."

    if re.search(r"\b(decrease\s+brightness|brightness\s+down|dim(mer)?|make\s+(the\s+)?screen\s+dimmer|lower\s+(the\s+)?brightness)\b", c):
        subprocess.run(["osascript", "-e",
            'tell application "System Events" to key code 145'])
        return "🌑 Brightness decreased."

    # ══════════════════════════════════════════════════════════════════════════
    # SCREENSHOT
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(take\s+(a\s+)?screenshot|screenshot|capture\s+(the\s+)?screen|screen\s+capture|grab\s+(a\s+)?screenshot)\b", c):
        path = os.path.expanduser("~/Desktop/screenshot.png")
        subprocess.run(["screencapture", path])
        return "📸 Screenshot saved to Desktop."

    # ══════════════════════════════════════════════════════════════════════════
    # LOCK SCREEN
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(lock\s+(screen|the\s+(screen|computer|mac|laptop|device)|my\s+(screen|computer|mac|laptop)|computer|mac|device))\b", c):
        subprocess.run(["osascript", "-e",
            'tell application "System Events" to keystroke "q" using {command down, control down}'])
        return "🔒 Screen locked."

    # ══════════════════════════════════════════════════════════════════════════
    # SHUTDOWN / RESTART / SLEEP
    # ══════════════════════════════════════════════════════════════════════════
    if re.search(r"\b(shutdown|shut\s+down|power\s+off|turn\s+off\s+(the\s+)?(mac|computer|laptop))\b", c):
        subprocess.run(["osascript", "-e", 'tell application "System Events" to shut down'])
        return "Shutting down..."

    if re.search(r"\b(restart|reboot|restart\s+(the\s+)?(mac|computer))\b", c):
        subprocess.run(["osascript", "-e", 'tell application "System Events" to restart'])
        return "Restarting..."

    if re.search(r"\b(sleep|hibernate|put\s+(the\s+)?(mac|computer|laptop)\s+to\s+sleep|go\s+to\s+sleep)\b", c):
        subprocess.run(["osascript", "-e", 'tell application "System Events" to sleep'])
        return "Going to sleep..."

    # ══════════════════════════════════════════════════════════════════════════
    # OPEN APP / WEBSITE / FOLDER — catch-all
    # ══════════════════════════════════════════════════════════════════════════
    open_match = re.search(r"\b(open|launch|start|run)\s+(.+)", c)
    if open_match:
        target = open_match.group(2).strip()

        found = find_app(target)
        if found:
            return f"Opening {found}..."

        for name, url in websites.items():
            if name in target:
                return open_url(url)

        for name, path in folders.items():
            if name in target:
                full = os.path.expanduser(path)
                subprocess.run(["open", full])
                return f"📂 Opened {name} folder."

        result = subprocess.run(["open", "-a", target.title()], capture_output=True)
        if result.returncode == 0:
            return f"Opening {target.title()}..."

        return f"Couldn't find '{target}'. Make sure the app is installed."

    return None


if __name__ == "__main__":
    tests = [
        "remind me to drink water in 30 minutes",
        "set a reminder at 10:30 to drink water",
        "remind me at 5pm to leave for gym",
        "reminder at 9:00 am to take medicine",
        "set reminder at 10:30 am for meeting",
        "remind me at 7:30 to wake up",
        "remind me in 10 minutes to take medicine",
        "set a reminder for call John in 1 hour 30 minutes",
    ]
    for t in tests:
        print(f"INPUT : {t}")
        result = _handle_reminder(t.lower())
        print(f"OUTPUT: {result}\n")