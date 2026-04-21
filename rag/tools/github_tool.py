# rag/tools/github_tool.py

import os
import re
import json
import base64
from pathlib import Path

import ollama
import requests
from dotenv import load_dotenv
from config.settings import (
    DOCUMENTS_DIR,
    MAX_FILE_SIZE_MB,
    ALLOWED_GIT_FOLDER,
    LLM_MODEL,
)

load_dotenv()

GITHUB_API = "https://api.github.com"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".txt", ".md",
    ".csv", ".html", ".css", ".sh", ".env", ".toml", ".ini", ".xml",
    ".java", ".cpp", ".c", ".go", ".rs", ".rb", ".php", ".sql", ".ipynb",
}

_github_session = {
    "username": os.getenv("GITHUB_USERNAME", ""),
    "token":    os.getenv("GITHUB_TOKEN", ""),
}

_CONFIRM_PREFIX = "CONFIRM_REQUIRED::"


# =========================================================
# SECTION 1 — Auth helpers
# =========================================================

def _get_session() -> dict:
    if not _github_session["username"] or not _github_session["token"]:
        raise ValueError(
            "GitHub credentials not found.\n"
            "Add these two lines to your .env file:\n\n"
            "  GITHUB_USERNAME=your_username\n"
            "  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx\n\n"
            "Get a token at: github.com -> Settings -> Developer settings -> Personal access tokens\n"
            "Make sure it has 'repo' and 'delete_repo' scope."
        )
    return _github_session


def _headers() -> dict:
    sess = _get_session()
    h = {"Accept": "application/vnd.github+json"}
    if sess["token"]:
        h["Authorization"] = f"token {sess['token']}"
    return h


def reset_github_session() -> str:
    load_dotenv(override=True)
    _github_session["username"] = os.getenv("GITHUB_USERNAME", "")
    _github_session["token"]    = os.getenv("GITHUB_TOKEN", "")
    if _github_session["username"]:
        return f"GitHub session reloaded from .env (user: {_github_session['username']})."
    return (
        "No credentials found in .env.\n"
        "Add GITHUB_USERNAME and GITHUB_TOKEN to your .env file."
    )


# =========================================================
# SECTION 2 — LLM-based intent + entity classifier
# =========================================================

def _classify_github_query(query: str) -> dict:
    """
    Use the local LLM to understand what the user wants to do on GitHub
    and extract relevant entities (repo name, filename, branch, etc.).
    Falls back to regex if LLM call fails.
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f'Classify this GitHub-related user query and extract entities.\n\n'
                    f'Query: "{query}"\n\n'
                    f'Return ONLY valid JSON, nothing else:\n'
                    f'{{\n'
                    f'  "intent": "<intent>",\n'
                    f'  "repo_name": "<repo name or null>",\n'
                    f'  "filename": "<filename with extension or null>",\n'
                    f'  "branch": "<branch name or null>",\n'
                    f'  "private": <true or false>\n'
                    f'}}\n\n'
                    f'INTENT OPTIONS (pick exactly one):\n'
                    f'  delete_repo       - user wants to delete/remove/erase/destroy a repository\n'
                    f'  create_repo       - user wants to create/make/build/start/new/setup a repository\n'
                    f'  upload_file       - user wants to upload/push/send/add a file to GitHub\n'
                    f'  list_repos        - user wants to see/list/show/view their repositories\n'
                    f'  repo_info         - user wants details/info about a specific repo\n'
                    f'  list_branches     - user wants to see branches of a repo\n'
                    f'  list_issues       - user wants to see issues/bugs of a repo\n'
                    f'  list_prs          - user wants to see pull requests of a repo\n'
                    f'  list_contributors - user wants to see contributors of a repo\n'
                    f'  show_profile      - user wants to see their GitHub profile\n'
                    f'  git_help          - user wants git commands, cheatsheet, or help\n'
                    f'  switch_account    - user wants to switch/logout/reset GitHub account\n'
                    f'  unknown           - cannot determine intent\n\n'
                    f'ENTITY EXTRACTION RULES:\n'
                    f'- repo_name: The repository name the user is referring to.\n'
                    f'  Extract the ACTUAL NAME even if phrased unusually.\n'
                    f'  Examples:\n'
                    f'  "Can you please delete the lavish named repo" -> "lavish"\n'
                    f'  "remove the repo called my-project" -> "my-project"\n'
                    f'  "create a new repo named TodoApp" -> "TodoApp"\n'
                    f'  "please can you delete the repo which is named as lavish" -> "lavish"\n'
                    f'  "I want to get rid of my DataVault repository" -> "DataVault"\n'
                    f'  "spin up a new project repository called API-Gateway" -> "API-Gateway"\n'
                    f'  If truly no name mentioned: null\n'
                    f'- filename: Only if user mentions a file with extension. null otherwise.\n'
                    f'- branch: Branch name if mentioned. null otherwise.\n'
                    f'- private: true only if user says "private" or "secret".\n\n'
                    f'EXAMPLES:\n'
                    f'  "Can you please delete the lavish named repo from my github"\n'
                    f'  -> {{"intent":"delete_repo","repo_name":"lavish","filename":null,"branch":null,"private":false}}\n\n'
                    f'  "please create a new private repository called DataVault"\n'
                    f'  -> {{"intent":"create_repo","repo_name":"DataVault","filename":null,"branch":null,"private":true}}\n\n'
                    f'  "I want to get rid of my old-website repo"\n'
                    f'  -> {{"intent":"delete_repo","repo_name":"old-website","filename":null,"branch":null,"private":false}}\n\n'
                    f'  "show me all my github repos"\n'
                    f'  -> {{"intent":"list_repos","repo_name":null,"filename":null,"branch":null,"private":false}}\n\n'
                    f'  "what branches does my web-app repo have"\n'
                    f'  -> {{"intent":"list_branches","repo_name":"web-app","filename":null,"branch":null,"private":false}}\n\n'
                    f'Only the JSON. Nothing else.'
                )
            }],
            options={"temperature": 0.0, "num_predict": 120}
        )

        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)

        valid_intents = {
            "delete_repo", "create_repo", "upload_file", "list_repos",
            "repo_info", "list_branches", "list_issues", "list_prs",
            "list_contributors", "show_profile", "git_help",
            "switch_account", "unknown"
        }
        intent = str(parsed.get("intent", "unknown")).lower().strip()
        if intent not in valid_intents:
            intent = "unknown"

        def _clean(val):
            if not val:
                return None
            s = str(val).strip('\'".,;:!? ')
            if s.lower() in ("null", "none", "n/a", "repo", "repository", "github", ""):
                return None
            return s

        result = {
            "intent":    intent,
            "repo_name": _clean(parsed.get("repo_name")),
            "filename":  _clean(parsed.get("filename")),
            "branch":    _clean(parsed.get("branch")),
            "private":   bool(parsed.get("private", False)),
            "raw_query": query,
        }
        print(f"[GitHub] LLM classified: {result}")
        return result

    except Exception as e:
        print(f"[GitHub] LLM classification failed: {e} — using regex fallback")
        return _classify_github_query_fallback(query)


def _classify_github_query_fallback(query: str) -> dict:
    """Regex fallback when LLM is unavailable."""
    q = query.lower()
    intent = "unknown"

    if re.search(
        r'\b(delete|remove|destroy|erase|wipe|get rid of|drop)\b.{0,60}\b(repo|repository)\b'
        r'|\b(repo|repository)\b.{0,60}\b(delete|remove|destroy|erase|drop)\b',
        q
    ):
        intent = "delete_repo"
    elif re.search(
        r'\b(create|make|build|start|initialise|initialize|new|setup|set up|spin up|generate)\b'
        r'.{0,40}\b(repo|repository)\b'
        r'|\b(repo|repository)\b.{0,40}\b(create|make|build|new)\b',
        q
    ):
        intent = "create_repo"
    elif re.search(
        r'\b(upload|push|send|add|commit|deploy)\b.{0,40}\b(file|to github|to repo|to my repo)\b'
        r'|push\s+\S+\.(py|js|ts|json|txt|md)',
        q
    ):
        intent = "upload_file"
    elif re.search(
        r'\b(list|show|see|view|display|what are|get|fetch)\b.{0,30}\b(repos|repositories)\b'
        r'|\bmy\s+(github\s+)?repos?\b',
        q
    ):
        intent = "list_repos"
    elif re.search(r'\b(profile|about me|my github)\b', q):
        intent = "show_profile"
    elif re.search(r'\b(branch|branches)\b', q):
        intent = "list_branches"
    elif re.search(r'\b(issue|issues|bug|bugs)\b', q):
        intent = "list_issues"
    elif re.search(r'\b(pull request|pr|prs)\b', q):
        intent = "list_prs"
    elif re.search(r'\b(contributor|contributors)\b', q):
        intent = "list_contributors"
    elif re.search(r'\b(repo info|details of|about the repo)\b', q):
        intent = "repo_info"
    elif re.search(r'\b(git help|git commands|cheatsheet|git guide)\b', q):
        intent = "git_help"
    elif re.search(r'\b(switch account|logout|change account|reset github)\b', q):
        intent = "switch_account"

    result = {
        "intent":    intent,
        "repo_name": _extract_repo_from_query(query) or None,
        "filename":  _extract_filename_from_query(query) or None,
        "branch":    None,
        "private":   bool(re.search(r'\b(private|secret)\b', q)),
        "raw_query": query,
    }
    print(f"[GitHub] Fallback classified: {result}")
    return result


# =========================================================
# SECTION 3 — File search on local system
# =========================================================

def _find_file_on_system(filename: str) -> str | None:
    filename_lower = filename.lower()
    has_extension  = "." in filename
    fallback_exts  = [".py", ".js", ".ts", ".sh", ".txt", ".md", ".json", ".yaml", ".yml"]
    candidates     = [filename_lower]
    if not has_extension:
        candidates += [filename_lower + ext for ext in fallback_exts]

    search_roots = [
        Path(DOCUMENTS_DIR),
        Path.home(),
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home() / "Downloads",
        Path.cwd(),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() in candidates:
                return str(path)
    return None


def _validate_file(file_path: str) -> str | None:
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"File type '{ext}' is not allowed for upload.\n"
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"File is {size_mb:.1f} MB, exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    return None


# =========================================================
# SECTION 4 — GitHub API action functions
# =========================================================

def _delete_repo(repo_name: str, confirmed: bool = False) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    uname = sess["username"]
    hdrs  = _headers()

    if not repo_name:
        return (
            "I could not determine which repository to delete.\n"
            "Please mention the repo name clearly.\n"
            "Example: 'Delete my repository named Lavish'"
        )

    check = requests.get(
        f"{GITHUB_API}/repos/{uname}/{repo_name}",
        headers=hdrs, timeout=10
    )
    if check.status_code == 404:
        return (
            f"Repository '{repo_name}' not found under your account '{uname}'.\n"
            f"Use 'list my repos' to see all your repositories."
        )
    if check.status_code != 200:
        return f"Could not verify repository (HTTP {check.status_code})."

    if not confirmed:
        return (
            f"{_CONFIRM_PREFIX}delete_repo::{uname}/{repo_name}::"
            f"Are you sure you want to permanently delete '{uname}/{repo_name}'?"
        )

    r = requests.delete(
        f"{GITHUB_API}/repos/{uname}/{repo_name}",
        headers=hdrs, timeout=10
    )
    if r.status_code == 204:
        return f"Repository '{uname}/{repo_name}' has been permanently deleted."

    try:
        reason = r.json().get("message", "Unknown error")
    except Exception:
        reason = r.text or "Unknown error"

    return (
        f"Failed to delete repository (HTTP {r.status_code}).\n"
        f"Reason: {reason}\n\n"
        f"Make sure your token has the 'delete_repo' scope enabled.\n"
        f"Go to: github.com -> Settings -> Developer settings -> Personal access tokens"
    )


def _create_repo(repo_name: str, private: bool = False) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    uname = sess["username"]
    hdrs  = _headers()

    if not repo_name:
        return (
            "I could not determine a name for the repository.\n"
            "Please mention the repo name clearly.\n"
            "Example: 'Create a new repo named MyProject'"
        )

    check = requests.get(
        f"{GITHUB_API}/repos/{uname}/{repo_name}",
        headers=hdrs, timeout=10
    )
    if check.status_code == 200:
        existing = check.json()
        return (
            f"Repository '{repo_name}' already exists!\n\n"
            f"  URL        : {existing['html_url']}\n"
            f"  Visibility : {'Private' if existing['private'] else 'Public'}\n"
            f"  Branch     : {existing['default_branch']}"
        )

    payload = {"name": repo_name, "private": private, "auto_init": True}
    r = requests.post(f"{GITHUB_API}/user/repos", json=payload, headers=hdrs, timeout=10)

    if r.status_code == 201:
        repo = r.json()
        vis  = "Private" if repo["private"] else "Public"
        return (
            f"Repository '{repo_name}' created successfully!\n\n"
            f"  Owner        : {uname}\n"
            f"  Visibility   : {vis}\n"
            f"  URL          : {repo['html_url']}\n"
            f"  Clone HTTPS  : git clone {repo['clone_url']}\n"
            f"  Clone SSH    : git clone {repo['ssh_url']}"
        )

    try:
        error  = r.json().get("message", r.text)
        errors = r.json().get("errors", [])
        detail = f" - {errors[0]['message']}" if errors else ""
    except Exception:
        error, detail = r.text, ""
    return f"Failed to create repository (HTTP {r.status_code}): {error}{detail}"


def _upload_file(filename: str, repo_name: str, branch: str = "main") -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not filename:
        return "No filename found. Example: 'upload main.py to my repo'"
    if not repo_name:
        return "No repository name found. Example: 'upload main.py to my-repo'"

    file_path = _find_file_on_system(filename)
    if not file_path:
        return (
            f"File '{filename}' not found on your system.\n"
            f"Searched in: {DOCUMENTS_DIR}, home, Desktop, Documents, Downloads, cwd."
        )

    error = _validate_file(file_path)
    if error:
        return f"Upload blocked: {error}"

    uname = sess["username"]
    hdrs  = _headers()

    r_check = requests.get(f"{GITHUB_API}/repos/{uname}/{repo_name}", headers=hdrs, timeout=10)
    if r_check.status_code == 404:
        rc = requests.post(
            f"{GITHUB_API}/user/repos",
            json={"name": repo_name, "private": False, "auto_init": True},
            headers=hdrs, timeout=10
        )
        if rc.status_code != 201:
            return f"Repository '{repo_name}' not found and could not be created."

    try:
        with open(file_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    sha = None
    sha_check = requests.get(
        f"{GITHUB_API}/repos/{uname}/{repo_name}/contents/{filename}",
        headers=hdrs, params={"ref": branch}, timeout=10,
    )
    if sha_check.status_code == 200:
        sha = sha_check.json().get("sha")

    payload = {"message": "Upload via RAG assistant", "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    r = requests.put(
        f"{GITHUB_API}/repos/{uname}/{repo_name}/contents/{filename}",
        json=payload, headers=hdrs, timeout=15,
    )

    if r.status_code in (200, 201):
        action   = "updated" if sha else "uploaded"
        file_url = r.json().get("content", {}).get("html_url", "")
        size_mb  = Path(file_path).stat().st_size / (1024 * 1024)
        return (
            f"File '{filename}' successfully {action} to GitHub!\n\n"
            f"  Repository : {uname}/{repo_name}\n"
            f"  Branch     : {branch}\n"
            f"  Size       : {size_mb:.2f} MB\n"
            f"  URL        : {file_url}"
        )

    return f"Upload failed (HTTP {r.status_code}): {r.json().get('message', r.text)}"


def _list_repos() -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(
        f"{GITHUB_API}/user/repos",
        headers=hdrs, params={"per_page": 30, "sort": "updated"}, timeout=10,
    )
    if r.status_code != 200:
        return f"Failed to fetch repos (HTTP {r.status_code})."

    repos = r.json()
    if not repos:
        return f"No repositories found for '{uname}'."

    lines = [f"## Repositories - {uname}\n"]
    for repo in repos:
        vis   = "Private" if repo["private"] else "Public"
        lang  = repo.get("language") or "-"
        stars = repo.get("stargazers_count", 0)
        desc  = repo.get("description") or "No description"
        lines.append(
            f"- **{repo['name']}** [{vis}] {stars} stars | {lang}\n"
            f"  {desc}\n"
            f"  {repo['html_url']}"
        )
    return "\n".join(lines)


def _show_profile() -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(f"{GITHUB_API}/users/{uname}", headers=hdrs, timeout=10)
    if r.status_code != 200:
        return f"Could not fetch profile (HTTP {r.status_code})."
    u = r.json()
    return (
        f"## GitHub Profile - {u.get('login')}\n"
        f"- **Name**         : {u.get('name') or '-'}\n"
        f"- **Bio**          : {u.get('bio') or '-'}\n"
        f"- **Location**     : {u.get('location') or '-'}\n"
        f"- **Public repos** : {u.get('public_repos', 0)}\n"
        f"- **Followers**    : {u.get('followers', 0)}  |  "
        f"**Following**: {u.get('following', 0)}\n"
        f"- **Profile URL**  : {u.get('html_url')}"
    )


def _repo_info(repo_name: str) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not repo_name:
        return "Please specify a repository name."

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(f"{GITHUB_API}/repos/{uname}/{repo_name}", headers=hdrs, timeout=10)
    if r.status_code != 200:
        return f"Repository '{repo_name}' not found."
    d = r.json()
    return (
        f"## {d['full_name']}\n"
        f"- **Description**   : {d.get('description') or '-'}\n"
        f"- **Language**      : {d.get('language') or '-'}\n"
        f"- **Stars**         : {d.get('stargazers_count', 0)}\n"
        f"- **Forks**         : {d.get('forks_count', 0)}\n"
        f"- **Open Issues**   : {d.get('open_issues_count', 0)}\n"
        f"- **Default Branch**: {d.get('default_branch')}\n"
        f"- **Visibility**    : {'Private' if d['private'] else 'Public'}\n"
        f"- **URL**           : {d['html_url']}"
    )


def _list_branches(repo_name: str) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not repo_name:
        return "Please specify a repository name."

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(f"{GITHUB_API}/repos/{uname}/{repo_name}/branches", headers=hdrs, timeout=10)
    if r.status_code != 200:
        return f"Could not fetch branches for '{repo_name}'."
    branches = r.json()
    if not branches:
        return f"No branches found in '{repo_name}'."
    lines = [f"## Branches - {uname}/{repo_name}"]
    for b in branches:
        protected = " (protected)" if b.get("protected") else ""
        lines.append(f"- **{b['name']}**{protected}")
    return "\n".join(lines)


def _list_issues(repo_name: str) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not repo_name:
        return "Please specify a repository name."

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(
        f"{GITHUB_API}/repos/{uname}/{repo_name}/issues",
        headers=hdrs, params={"state": "open", "per_page": 20}, timeout=10,
    )
    if r.status_code != 200:
        return f"Could not fetch issues for '{repo_name}'."
    issues = [i for i in r.json() if "pull_request" not in i]
    if not issues:
        return f"No open issues in '{repo_name}'."
    lines = [f"## Open Issues - {uname}/{repo_name}"]
    for issue in issues:
        labels    = ", ".join(l["name"] for l in issue.get("labels", []))
        label_str = f" [{labels}]" if labels else ""
        lines.append(f"- **#{issue['number']}** {issue['title']}{label_str} (by {issue['user']['login']})")
    return "\n".join(lines)


def _list_pull_requests(repo_name: str) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not repo_name:
        return "Please specify a repository name."

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(
        f"{GITHUB_API}/repos/{uname}/{repo_name}/pulls",
        headers=hdrs, params={"state": "open", "per_page": 20}, timeout=10,
    )
    if r.status_code != 200:
        return f"Could not fetch pull requests for '{repo_name}'."
    prs = r.json()
    if not prs:
        return f"No open pull requests in '{repo_name}'."
    lines = [f"## Open Pull Requests - {uname}/{repo_name}"]
    for pr in prs:
        lines.append(
            f"- **#{pr['number']}** {pr['title']} "
            f"(`{pr['head']['ref']}` -> `{pr['base']['ref']}`) "
            f"by {pr['user']['login']}"
        )
    return "\n".join(lines)


def _list_contributors(repo_name: str) -> str:
    try:
        sess = _get_session()
    except ValueError as e:
        return f"GitHub setup required:\n{e}"

    if not repo_name:
        return "Please specify a repository name."

    uname = sess["username"]
    hdrs  = _headers()
    r = requests.get(f"{GITHUB_API}/repos/{uname}/{repo_name}/contributors", headers=hdrs, timeout=10)
    if r.status_code != 200:
        return f"Could not fetch contributors for '{repo_name}'."
    contributors = r.json()
    if not contributors:
        return f"No contributors found for '{repo_name}'."
    lines = [f"## Contributors - {uname}/{repo_name}"]
    for c in contributors:
        lines.append(f"- **{c['login']}** - {c['contributions']} commits")
    return "\n".join(lines)


def _git_help() -> str:
    return """## Common Git Commands

### Setup
- `git config --global user.name "Name"`    - Set your name
- `git config --global user.email "email"`  - Set your email
- `git config --list`                        - View all config

### Starting a Repo
- `git init`                                 - Initialize new local repo
- `git clone <url>`                          - Clone a remote repo
- `git remote add origin <url>`              - Link local repo to GitHub
- `git remote -v`                            - View remote URLs

### Daily Workflow
- `git status`                               - Check changed files
- `git add .`                                - Stage all changes
- `git add <file>`                           - Stage a specific file
- `git commit -m "message"`                  - Commit staged changes
- `git push origin <branch>`                 - Push to remote branch
- `git pull origin <branch>`                 - Pull latest from remote

### Branching
- `git branch`                               - List local branches
- `git checkout -b <branch>`                 - Create and switch in one step
- `git merge <branch>`                       - Merge branch into current
- `git branch -d <branch>`                   - Delete local branch
- `git push origin --delete <branch>`        - Delete remote branch

### Undoing Things
- `git restore <file>`                       - Discard unstaged changes
- `git revert <commit>`                      - Revert a commit (safe)
- `git reset --soft HEAD~1`                  - Undo last commit, keep changes
- `git reset --hard HEAD~1`                  - Undo last commit, discard changes

### History
- `git log --oneline --graph`                - Visual commit history
- `git diff`                                 - Show unstaged changes
- `git blame <file>`                         - Show who changed each line

### Stashing
- `git stash`                                - Stash current changes
- `git stash pop`                            - Restore last stash
"""


# =========================================================
# SECTION 5 — Regex extraction helpers (used by fallback)
# =========================================================

def _extract_filename_from_query(query: str) -> str:
    match = re.search(
        r'\b([\w\-]+\.(py|js|ts|json|yaml|yml|txt|md|csv|html|css|sh|env|'
        r'toml|ini|xml|java|cpp|c|go|rs|rb|php|sql|ipynb))\b',
        query, re.IGNORECASE
    )
    return match.group(1) if match else ""


def _extract_repo_from_query(query: str) -> str:
    """Regex-based repo name extractor used only by the fallback classifier."""
    q = query.strip()

    _BLACKLIST = {
        "repo", "repository", "github", "my", "the", "a", "an",
        "all", "this", "that", "file", "folder", "please", "now",
        "delete", "remove", "create", "make", "list", "show",
        "from", "into", "onto", "with", "for", "of", "in", "to",
        "it", "its", "me", "us", "you", "your", "our", "their",
        "new", "old", "named", "called", "just", "can", "could",
        "would", "should", "will", "want", "need", "also",
        "which", "that", "is", "are", "as",
    }

    patterns = [
        r'\b(?:delete|remove)\s+([\w][\w\-\.]*)\s+from\b',
        r'\bfrom\s+(?:my\s+)?(?:github\s+)?(?:repo|repository)\s+(?:please\s+)?(?:delete|remove)\s+([\w][\w\-\.]*)\b',
        r'\bfrom\s+(?:my\s+)?(?:github\s+)?(?:repo|repository)\s+([\w][\w\-\.]*)\b',
        r'\b(?:repo|repository)\s+(?:named\s+|called\s+)([\w][\w\-\.]*)\b',
        r'\ba\s+(?:new\s+)?repo(?:sitory)?\s+(?:named\s+|called\s+)([\w][\w\-\.]*)\b',
        r'\bnamed\s+as\s+([\w][\w\-\.]*)\b',
        r'\b([\w][\w\-\.]+)\s+(?:repo|repository)\b',
        r'\b(?:delete|remove)\s+(?:my\s+)?(?:the\s+)?([\w][\w\-\.]*)\b',
        r'\b(?:repo|repository)\s+([\w][\w\-\.]*)\b',
        r'\b(?:in|for|of)\s+([\w][\w\-\.]*)\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m and m.group(1).lower() not in _BLACKLIST:
            return m.group(1)

    return ""


# =========================================================
# SECTION 6 — Main dispatcher (LLM-powered, no hardcoded keywords)
# =========================================================

def handle_github(query: str, confirmed: bool = False) -> str:
    """
    Main entry point called from retriever.py.
    Uses LLM to understand intent dynamically.
    confirmed=True skips the confirmation step for destructive actions.
    """
    classified = _classify_github_query(query)

    intent    = classified["intent"]
    repo_name = classified["repo_name"]
    filename  = classified["filename"]
    branch    = classified["branch"] or "main"
    private   = classified["private"]

    if intent == "delete_repo":
        return _delete_repo(repo_name, confirmed=confirmed)

    if intent == "create_repo":
        return _create_repo(repo_name, private=private)

    if intent == "upload_file":
        return _upload_file(filename, repo_name, branch=branch)

    if intent == "list_repos":
        return _list_repos()

    if intent == "show_profile":
        return _show_profile()

    if intent == "repo_info":
        return _repo_info(repo_name)

    if intent == "list_branches":
        return _list_branches(repo_name)

    if intent == "list_issues":
        return _list_issues(repo_name)

    if intent == "list_prs":
        return _list_pull_requests(repo_name)

    if intent == "list_contributors":
        return _list_contributors(repo_name)

    if intent == "git_help":
        return _git_help()

    if intent == "switch_account":
        return reset_github_session()

    # Unknown - default to git help
    print(f"[GitHub] Unknown intent '{intent}' - returning git help")
    return _git_help()