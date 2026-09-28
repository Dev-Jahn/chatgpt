#!/usr/bin/env python3
"""Send one neutral prompt through a logged-in ChatGPT browser and harvest Markdown.

Exit codes: 0 success · 1 generic failure · 2 model verification failed (nothing sent)
· 3 response timeout · 5 ChatGPT rate limit (submit blocked, or harvest blocked after
the prompt was sent — the message then carries the conversation URL) · 64 usage error.
The bash wrapper adds 4 for its own lock/slot timeouts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Sequence, TextIO

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # Unit tests and --help do not require Playwright.
    sync_playwright = None


CHATGPT_URL = "https://chatgpt.com/"
# Composer and conversation DOM, measured live 2026-09-26 and rechecked 2026-09-28: the composer is a
# ProseMirror editor in form[data-chatgpt-composer] (placement 'home' or 'thread'); a conversation
# renders one [data-turn-key] wrapper per exchange holding a user unit, an assistant unit (both keyed
# by data-chatgpt-search-unit-key, e.g. 'fallback-turn-0:0:user' / 'fallback-turn-0:2:assistant'
# since 2026-09-28, 'turn-0:…' before; ids in data-chatgpt-search-message-ids) and, once the answer
# is done, the assistant's .turn-action-controls bar. The page loads only the newest turns of a long
# thread, so turns are told apart by id, not by position or count. Pre-2026-09 selectors stay listed
# after the current ones; the first selector that matches wins. A run harvests only its own exchange:
# the [data-turn-key] wrapper keyed by its own user turn's id (see own_exchange).
COMPOSER_FORM_SELECTOR = "form[data-chatgpt-composer]"
INPUT_SELECTORS = [
    f'{COMPOSER_FORM_SELECTOR} [contenteditable="true"][role="textbox"]',
    "#prompt-textarea",
    'div[contenteditable="true"]',
]
# Three hidden file inputs sit in the composer; the first two accept only images/videos.
FILE_INPUT_SELECTOR = 'input[type="file"]:not([accept]), input[type="file"][accept=""]'
USER_MSG_SELECTORS = [
    '[data-chatgpt-search-unit-key$=":user"]',
    '[data-message-author-role="user"]',
    'article[data-turn="user"]',
]
ASSISTANT_MSG_SELECTORS = [
    '[data-chatgpt-search-unit-key$=":assistant"]',
    '[data-message-author-role="assistant"]',
    'article[data-turn="assistant"]',
]
MESSAGE_IDS_ATTR = "data-chatgpt-search-message-ids"  # space-separated; was data-message-id
# Measured 2026-09-28 on 13 exchanges of two scratch threads: the wrapper's data-turn-key equals its
# user unit's message id, and it holds that user unit, exactly one assistant unit and the assistant's
# action bar (outside both units).
EXCHANGE_KEY_ATTR = "data-turn-key"
MESSAGE_ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")
# A run's own user turn must carry the start of its prompt, compared on letters and digits only (the
# bubble may render Markdown or show an attachment chip before the text).
PROMPT_KEY_CHARS = 48
ASSISTANT_MARKDOWN_SELECTORS = ['[data-markdown-text-style="assistant-message"]', ".markdown"]
TURN_ACTIONS_SELECTOR = ".turn-action-controls"
# The assistant bar's copy action (measured 복사); the user unit's own bar labels its copy 메시지 복사.
COPY_BTN_SELECTORS = [
    'button[aria-label="복사"]',
    'button[aria-label="Copy"]',
    'button[data-testid="copy-turn-action-button"]',
    'button[data-testid*="copy"]',
]
# Measured 2026-09-28 with a prompt in flight: the composer's submit slot turns into a plain button
# labelled 중지 (Stop) until the answer is done. The old test ids follow.
STREAMING_BTN_SELECTORS = [
    f'{COMPOSER_FORM_SELECTOR} button[aria-label*="중지"]',
    f'{COMPOSER_FORM_SELECTOR} button[aria-label*="stop" i]',
    'button[data-testid="stop-button"]',
    'button[aria-label="Stop streaming"]',
    'button[data-testid*="stop"]',
]
# How long a freshly opened conversation is watched for that stop button before a follow-up is
# typed (see reply_in_progress).
REPLY_IN_PROGRESS_POLL_SECS = 3
SEND_BTN_SELECTORS = [
    f'{COMPOSER_FORM_SELECTOR} button[type="submit"]',
    'button[data-testid="send-button"]',
    'button[data-testid="composer-send-button"]',
    'button[aria-label*="send" i]',
    'button[aria-label*="보내기" i]',
    'button[aria-label*="프롬프트 보내기" i]',
]
LOGIN_WALL_SELECTORS = [
    'button[data-testid="login-button"]',
    'a[href*="auth/login"]',
    'button:has-text("로그인")',
    'button:has-text("Log in")',
]
# Composer model picker — Chat mode, measured live 2026-09-26. The pill (composer button targeting
# 'reasoning') opens a menu whose [data-model-picker-view] swaps two panels: 'simple' shows the model
# row (the view toggle, labelled e.g. "6 Pro") above a five-tick power slider; 'advanced' shows the
# model list (최신/Latest plus explicit versions) and disables the slider. The selected model is the
# aria-checked entry of that list (present in the DOM in both views); the [data-explicit-model]
# flag stays "false" with an explicit model selected, so it is not read. The pin is the label that
# model row shows at max effort with the Latest model — "6 Pro" is GPT-6 Pro today (the closed pill
# reads just "Pro" since 2026-09, so it no longer carries the version); below Pro, Latest is labelled
# by effort alone.
REQUIRED_PRO_LABEL = "6 Pro"
LATEST_MODEL_RE = re.compile(r"^(최신|latest|auto)$", re.I)
EFFORT_LEVELS = ("instant", "medium", "high", "extra high", "pro")  # slider ticks 0..4
PILL_SELECTOR = f'{COMPOSER_FORM_SELECTOR} button[aria-haspopup="menu"][data-composer-navigation-target="reasoning"]'
# Chat/Work switch in the page header: two aria-pressed buttons in a role=group. The buttons carry
# no attribute naming the mode, so their (so far untranslated) text decides.
MODE_BUTTON_SELECTOR = '[role="group"] > button[aria-pressed]'
CHAT_MODE_RE = re.compile(r"^(chat|채팅)$", re.I)
WORK_MODE_RE = re.compile(r"^(work|작업)$", re.I)
PICKER_SELECTOR = '[role="menu"] [data-model-picker-view]'
MODEL_TOGGLE_SELECTOR = f'{PICKER_SELECTOR} [role="menuitem"][data-model-picker-view-toggle]'
MODEL_RADIO_SELECTOR = f'{PICKER_SELECTOR} [role="menuitemradio"]'
EFFORT_TICK_SELECTOR = f"{PICKER_SELECTOR} [data-model-picker-power-slider] span[data-selected]"
QUOTA_HINTS = [
    "usage limit",
    "reached your limit",
    "limit reached",
    "you've hit",
    "reached the current usage cap",
    "try again later",
    "upgrade to",
    "사용량 한도",
    "한도에 도달",
    "사용 한도",
    "요금제를 업그레이드",
]
CONV_URL_RE = re.compile(r"/c/[0-9a-f]{8}[0-9a-f-]{4,}", re.I)
# A thread handle is the first 8 characters of a conversation id; any longer prefix works too.
THREAD_HANDLE_RE = re.compile(r"^[0-9a-f]{8}[0-9a-f-]*$", re.I)
THREAD_LEDGER_CAP = 500
STABLE_SECS = 4
# How long the harvest tolerates the run's own turn missing from the page before it gives up on it.
OWN_TURN_GRACE_SECS = 30
STATUS_INTERVAL = 15
# Project grouping (ported from insane-review's pack_and_ask.py). The /g/g-p- URL mark and
# the sidebar/aria heuristics below are the early 2026-08 ChatGPT DOM — revalidate on drift.
PROJECT_URL_MARK = "/g/g-p-"
NEW_PROJECT_RE = r"새 프로젝트|New project|新規プロジェクト|プロジェクトを追加|Add project|Create project"
CREATE_SUBMIT_RE = r"프로젝트 만들기|Create project|プロジェクトを作成|^Create$|^作成$|^만들기$"
# Captured live 2026-08-13: the access-throttle dialog ChatGPT raises after rapid requests
# ("요청이 너무 많습니다 … 몇 분 후 다시 시도해 주세요", one dismiss button).
RATE_LIMIT_MODAL_SELECTOR = '[data-testid="modal-conversation-history-rate-limit"]'


class UsageError(Exception):
    pass


class ModelVerificationError(Exception):
    pass


class ResponseTimeoutError(Exception):
    pass


class SentUnknownLocationError(Exception):
    pass


class RateLimitedError(Exception):
    pass


class ForeignReplyError(Exception):
    """The answer shown under this run's prompt is not that prompt's answer."""


class Reply(NamedTuple):
    body: str
    conversation_url: str


class UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def log(message: str) -> None:
    print(f"chatgpt: {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = UsageParser(prog="chatgpt", description=__doc__)
    parser.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
    parser.add_argument("-f", "--file", type=Path, help="read the prompt from a UTF-8 file")
    parser.add_argument(
        "--effort",
        default="pro",
        help="reasoning effort: instant, medium, high, extra high, pro (default: pro)",
    )
    parser.add_argument("--attach", type=Path, help="attach one file")
    parser.add_argument("--max-wait", type=int, default=7200, metavar="SEC")
    parser.add_argument("--out", type=Path, help="response path")
    parser.add_argument("--quiet", action="store_true", help="print only the response path")
    parser.add_argument(
        "--project",
        help="ChatGPT project to group the chat under (default: '<folder> · <hash8>' from the cwd)",
    )
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="start a plain chat instead of grouping under a project",
    )
    parser.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="follow up in the most recent thread started from this folder "
        "(its context is retained; project grouping does not apply)",
    )
    parser.add_argument(
        "--resume",
        metavar="THREAD",
        help="follow up in a specific thread: the 8-hex handle a previous reply's trailer printed, "
        "or a chatgpt.com conversation URL for a chat this tool did not start",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if (args.prompt is None) == (args.file is None):
        raise UsageError("provide exactly one prompt source: PROMPT, '-', or -f FILE")
    if args.max_wait <= 0:
        raise UsageError("--max-wait must be greater than zero")
    args.effort = EFFORT_LEVELS[effort_index(args.effort)]
    if args.project is not None and args.no_project:
        raise UsageError("choose either --project or --no-project, not both")
    if args.project is not None and not args.project.strip():
        raise UsageError("--project must not be empty")
    if args.continue_last and args.resume is not None:
        raise UsageError("choose either --continue or --resume, not both")
    if (args.continue_last or args.resume is not None) and (args.project is not None or args.no_project):
        raise UsageError(
            "--continue/--resume reopen an existing thread; --project/--no-project do not apply"
        )
    return args


def read_prompt(args: argparse.Namespace, stdin: TextIO) -> str:
    if args.file is not None:
        try:
            prompt = args.file.expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"cannot read prompt file {args.file}: {exc}") from exc
    elif args.prompt == "-":
        prompt = stdin.read()
    else:
        prompt = args.prompt or ""
    if not prompt.strip():
        raise UsageError("prompt is empty")
    return prompt


def response_path(given: Path | None) -> Path:
    if given is not None:
        return given.expanduser().resolve()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    return (Path.home() / ".chatgpt" / "out" / f"{stamp}.md").resolve()


# --- thread ledger ---------------------------------------------------------------------
# Every thread this tool opens is remembered in <state dir>/threads.json, so a follow-up names
# it by an 8-hex handle — or not at all: the most recent thread from this folder — instead of
# the calling agent carrying a 100+ character URL across turns. Resolution runs before any
# browser work: an unknown handle is a usage error with nothing opened.

HANDLE_HINT = "a handle is the 8-hex `Thread` id printed at the end of a previous reply"


def state_dir() -> Path:
    return Path(os.environ.get("CHATGPT_STATE_DIR", str(Path.home() / ".chatgpt")))


def thread_ledger_path() -> Path:
    return state_dir() / "threads.json"


def conversation_id(url: str) -> str:
    """The id in a …/c/<id> conversation URL ('' when the URL carries none)."""
    match = CONV_URL_RE.search(url)
    return match.group(0)[len("/c/") :] if match else ""


def thread_handle(conversation_id: str) -> str:
    return conversation_id[:8].lower()


def folder_hash() -> str:
    return hashlib.sha256(str(Path.cwd().resolve()).encode("utf-8")).hexdigest()[:8]


def prompt_excerpt(prompt: str, limit: int = 120) -> str:
    text = " ".join(prompt.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_threads(path: Path) -> list[dict]:
    """The ledger's entries, oldest first. A missing file is empty; a corrupt one is reported and
    read as empty (the next successful send rewrites it)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except ValueError as exc:
        log(f"thread ledger {path} is corrupt ({exc}); treating it as empty")
        return []
    return [
        entry
        for entry in data
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("url"), str)
    ]


def record_thread(url: str, prompt: str, path: Path | None = None) -> str:
    """Append a new thread, or refresh last_used on a known one; returns its handle. Called after
    the conversation URL is bound and before the submit lock is released — that lock already
    serialises concurrent runs, so the ledger needs no lock of its own."""
    path = path or thread_ledger_path()
    cid = conversation_id(url)
    if not cid:
        raise ValueError(f"no conversation id in {url!r}")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    entries = load_threads(path)
    for entry in entries:
        if entry["id"].lower() == cid.lower():
            entry["last_used"] = now
            break
    else:
        entries.append(
            {
                "id": cid,
                "url": url,
                "cwd": str(Path.cwd().resolve()),
                "started": now,
                "last_used": now,
                "prompt": prompt_excerpt(prompt),
            }
        )
    atomic_write(path, json.dumps(entries[-THREAD_LEDGER_CAP:], ensure_ascii=False, indent=2) + "\n")
    return thread_handle(cid)


def latest_thread_here(entries: list[dict]) -> dict:
    cwd = str(Path.cwd().resolve())
    mine = [entry for entry in entries if entry.get("cwd") == cwd]
    if not mine:
        raise UsageError(
            "no thread has been started from this folder yet; run without --continue to start one"
        )
    return max(mine, key=lambda entry: entry.get("last_used", ""))


def thread_by_handle(handle: str, entries: list[dict]) -> dict:
    needle = handle.lower()
    hits = [entry for entry in entries if entry["id"].lower().startswith(needle)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise UsageError(
            f"thread handle {handle!r} is ambiguous ({len(hits)} threads match); give more of the id"
        )
    if needle == folder_hash():
        raise UsageError(f"{handle!r} is this folder's project hash, not a thread handle; {HANDLE_HINT}")
    raise UsageError(
        f"unknown thread {handle!r}; {HANDLE_HINT} "
        "(pass the full conversation URL to resume a chat this tool did not start)"
    )


def resolve_thread(args: argparse.Namespace) -> str | None:
    """The conversation URL a follow-up lands in (None for a fresh chat), decided from the ledger
    before any browser work so a bad handle costs nothing but exit 64."""
    if args.resume is not None:
        value = args.resume.strip().rstrip("/")
        if CONV_URL_RE.search(value):
            log("resuming by URL (not in the thread ledger)")
            return value if re.match(r"https?://", value, re.I) else f"https://{value}"
        if not THREAD_HANDLE_RE.match(value):
            raise UsageError(
                "--resume expects a thread handle or a chatgpt.com conversation URL, "
                f"got {args.resume!r}; {HANDLE_HINT}"
            )
        entry = thread_by_handle(value, load_threads(thread_ledger_path()))
    elif args.continue_last:
        entry = latest_thread_here(load_threads(thread_ledger_path()))
    else:
        return None
    started = entry.get("started", "")[:16].replace("T", " ")
    handle, excerpt = thread_handle(entry["id"]), entry.get("prompt", "")
    log(f'continuing thread {handle} · started {started} · "{excerpt}"')
    return entry["url"]


def thread_trailer(url: str) -> str:
    """Printed after every reply so the calling agent learns that it can follow up in-thread."""
    handle = thread_handle(conversation_id(url))
    return (
        "---\n"
        f"Thread {handle} · {url}\n"
        "Follow-up on this topic? Add `--continue` to the next `chatgpt` call to keep going in this "
        f"thread (the most recent one from this folder), or `--resume {handle}` to pick it explicitly. "
        "Omit both to start a fresh chat."
    )


# --- conversation lock -----------------------------------------------------------------
# One bridge run per conversation on this machine. Measured 2026-09-28: a follow-up that opened the
# conversation within ~5-10 s of another run's send saw neither that turn nor the stop button, so its
# send forked a sibling branch and the server mixed the two exchanges (it returned the other run's
# answer). The wrapper's submit lock is released ~5 s after a send, so the conversation needs a lock
# of its own that lasts until the reply is harvested. flock is dropped by the kernel when the process
# dies; the small lock files stay (deleting a flock file races with the next opener).

CONVERSATION_BUSY = "another bridge run is waiting for a reply in this conversation; nothing sent"


def conversation_lock_path(conversation_url: str) -> Path:
    cid = conversation_id(conversation_url)
    if not cid:
        raise ValueError(f"no conversation id in {conversation_url!r}")
    return state_dir() / "conversations" / f"{cid.lower()}.lock"


def lock_conversation(conversation_url: str) -> int | None:
    """Take this conversation's lock without waiting: the held fd, or None when another run has it."""
    path = conversation_lock_path(conversation_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()} since={datetime.now().astimezone().isoformat(timespec='seconds')}\n".encode())
    return fd


def conversation_lock_holder(conversation_url: str) -> str:
    try:
        return conversation_lock_path(conversation_url).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def unlock_conversation(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _q(page, selectors):
    for selector in selectors:
        try:
            node = page.query_selector(selector)
        except Exception:
            continue
        if node is not None:
            return node
    return None


def _qa(page, selectors):
    for selector in selectors:
        try:
            nodes = page.query_selector_all(selector)
        except Exception:
            continue
        if nodes:
            return nodes
    return []


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


MESSAGE_IDS_JS = """els => els.flatMap(e =>
  (e.getAttribute('%s') || e.getAttribute('data-message-id') || '').split(/\\s+/).filter(Boolean))""" % MESSAGE_IDS_ATTR


def node_message_ids(node) -> set[str]:
    raw = node.get_attribute(MESSAGE_IDS_ATTR) or node.get_attribute("data-message-id") or ""
    return set(raw.split())


def message_ids(page) -> set[str]:
    try:
        return set(
            page.eval_on_selector_all(f"[{MESSAGE_IDS_ATTR}], [data-message-id]", MESSAGE_IDS_JS)
        )
    except Exception:
        return set()


def current_url(page) -> str:
    try:
        return page.evaluate("() => location.href") or ""
    except Exception:
        return page.url or ""


def _guard_dialogs(context, page=None) -> None:
    def dismiss(dialog):
        try:
            dialog.dismiss()
        except Exception:
            pass

    def attach(target):
        try:
            target.on("dialog", dismiss)
        except Exception:
            pass

    for existing in context.pages:
        attach(existing)
    context.on("page", attach)
    if page is not None:
        attach(page)


def cdp_browser_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=4) as result:
            info = json.loads(result.read().decode("utf-8"))
        name = str(info.get("Browser", ""))
        return any(part in name for part in ("Chrome", "Chromium", "HeadlessChrome", "Edg", "Comet"))
    except Exception:
        return False


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def pick_context(browser):
    if not browser.contexts:
        return None
    for context in browser.contexts:
        try:
            cookies = context.cookies(CHATGPT_URL)
            if any(str(cookie.get("name", "")).startswith("__Secure-next-auth") for cookie in cookies):
                return context
        except Exception:
            continue
    for context in browser.contexts:
        try:
            if context.cookies(CHATGPT_URL):
                return context
        except Exception:
            continue
    return browser.contexts[0]


def find_input(page):
    return _q(page, INPUT_SELECTORS)


def login_state(page, wait_secs: int = 15) -> str:
    deadline = time.monotonic() + wait_secs
    while True:
        for selector in LOGIN_WALL_SELECTORS:
            try:
                item = page.query_selector(selector)
                if item and item.is_visible():
                    return "no"
            except Exception:
                continue
        try:
            if find_input(page) is not None and (
                page.query_selector(PILL_SELECTOR)
                or page.query_selector(FILE_INPUT_SELECTOR)
            ):
                return "ok"
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return "unknown"
        time.sleep(0.5)


# --- project grouping (ported from insane-review's pack_and_ask.py) --------------------
# Chats land inside a per-folder ChatGPT project instead of piling up in the root chat
# list. The project home carries the composer, the file input and the model pill, so once
# the page sits there the rest of the ask flow is unchanged. Every helper here swallows
# its exceptions into None/False: a project is a convenience, never a reason to abort.


def default_project_name() -> str:
    """'<folder> · <hash8>' — the path hash keeps two same-named folders (/a/api, /b/api)
    from merging into one remote project, since remote lookup matches display name only."""
    return f"{Path.cwd().name} · {folder_hash()}"


def project_cache_path() -> Path:
    return state_dir() / "projects.json"


def project_cache_key(name: str) -> str:
    # Absolute path in the key: same-named folders do not share a cache row, and neither
    # do two --project names used from one folder.
    return f"{Path.cwd().resolve()}::{name}"


def _load_project_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_project_cache(path: Path, cache: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def project_home_ok(page, url: str) -> bool:
    """A cached project URL may be stale (project deleted). Alive means: the home loads,
    still looks like a project URL, and carries a composer."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        return PROJECT_URL_MARK in current_url(page) and find_input(page) is not None
    except Exception:
        return False


def find_project_url(page, name: str) -> str | None:
    """Recover the home URL of the sidebar project whose display text is exactly `name`.

    Language-independent: rows are matched on their visible text, not on localized aria
    labels; within the row, the option button's aria-label contains the project name, so
    the button WITHOUT the name is the navigation one. The sidebar list is virtualized —
    poll while scrolling its containers so a project below the fold is not missed (and
    then wrongly re-created)."""
    try:
        for _ in range(12):
            clicked = page.evaluate(
                """(nm) => {
                    const lis = [...document.querySelectorAll('nav li, aside li, li')];
                    for (const li of lis) {
                        const first = ((li.innerText || '').trim().split('\\n')[0] || '').trim();
                        const btns = [...li.querySelectorAll('button[aria-label]')];
                        if (first === nm && btns.length) {
                            const home = btns.find(b => !((b.getAttribute('aria-label') || '').includes(nm))) || btns[0];
                            home.click();
                            return true;
                        }
                    }
                    return false;
                }""",
                name,
            )
            if clicked:
                try:
                    page.wait_for_url(f"**{PROJECT_URL_MARK}**", wait_until="commit", timeout=8000)
                except Exception:
                    pass
                time.sleep(1.2)
                url = current_url(page)
                return url if PROJECT_URL_MARK in url else None
            page.evaluate(
                """() => { for (const el of document.querySelectorAll('nav *, aside *')) {
                    if (el.scrollHeight > el.clientHeight + 20) el.scrollTop = el.scrollHeight; } }"""
            )
            time.sleep(0.5)
    except Exception:
        return None
    return None


def create_project(page, name: str) -> str | None:
    """Create the project through the sidebar modal and return its home URL. The name
    field must be filled (not typed) for the submit button to enable; submit falls back
    to Enter when the localized button text does not match."""
    try:
        opened = page.evaluate(
            """(re) => { const rx = new RegExp(re, 'i');
                const b = [...document.querySelectorAll('button[aria-label]')]
                    .find(x => rx.test(x.getAttribute('aria-label') || ''));
                if (b) { b.click(); return true; } return false; }""",
            NEW_PROJECT_RE,
        )
    except Exception:
        return None
    if not opened:
        return None  # no new-project affordance (unsupported plan or unmatched locale)
    try:
        name_input = page.locator('input[type="text"]:visible').last
        name_input.wait_for(state="visible", timeout=8000)
        name_input.click()
        name_input.fill(name)
        time.sleep(0.4)
        submitted = page.evaluate(
            """(re) => { const rx = new RegExp(re, 'i');
                const btns = [...document.querySelectorAll('button')]
                    .filter(b => !b.disabled && rx.test((b.innerText || '').trim()));
                if (btns.length) { btns[btns.length - 1].click(); return true; } return false; }""",
            CREATE_SUBMIT_RE,
        )
        if not submitted:
            name_input.press("Enter")
        page.wait_for_url(f"**{PROJECT_URL_MARK}**", wait_until="commit", timeout=15000)
        time.sleep(2)
        url = current_url(page)
        return url if PROJECT_URL_MARK in url else None
    except Exception:
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
        return None


def ensure_project(page, name: str, cache_key: str, cache_path: Path) -> str | None:
    """Project home URL via cache → sidebar lookup → creation; None on any failure so the
    caller can fall back to a plain chat. The cache keeps routine runs off the sidebar
    heuristics entirely: one goto against the remembered URL and a liveness check."""
    try:
        cache = _load_project_cache(cache_path)
        cached = cache.get(cache_key)
        if cached and project_home_ok(page, cached):
            return cached
        page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        url = find_project_url(page, name)
        if not url:
            page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)
            url = create_project(page, name)
        if url:
            cache[cache_key] = url
            _save_project_cache(cache_path, cache)
        return url
    except Exception:
        return None


def enter_project(page, name: str) -> bool:
    """Land the page on the named project's home with a live composer; False falls back
    to a plain chat (the caller re-navigates)."""
    url = ensure_project(page, name, project_cache_key(name), project_cache_path())
    if not url:
        return False
    try:
        page.goto(url, wait_until="load", timeout=60000)
    except Exception:
        return False
    time.sleep(2)
    for _ in range(10):
        if find_input(page) is not None:
            return True
        time.sleep(1)
    return False


# --- follow-ups in an existing conversation --------------------------------------------


def conversation_key(url: str) -> str:
    match = CONV_URL_RE.search(url)
    return match.group(0) if match else ""


def dialog_text(page) -> str:
    try:
        for surface in page.query_selector_all('[role="dialog"]'):
            text = normalize(surface.inner_text())
            if text:
                return text[:160]
    except Exception:
        pass
    return ""


def open_conversation(page, url: str) -> str:
    """Land on an existing conversation and return the URL the page settles on (ChatGPT rewrites
    a bare /c/<id> to its project form). A deleted or foreign conversation bounces to the home
    page behind an access dialog (measured 2026-09-05), so a page that never shows the id with
    a composer is reported here, before anything is typed."""
    key = conversation_key(url)
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    for _ in range(15):
        time.sleep(1)
        landed = current_url(page)
        if key.casefold() in landed.casefold() and find_input(page) is not None:
            return landed
    dialog = dialog_text(page)
    raise RuntimeError(
        f"conversation {key} is not reachable; the page settled on {current_url(page)}"
        + (f" ({dialog})" if dialog else "")
    )


# --- rate-limit modal ------------------------------------------------------------------


def rate_limit_modal(page) -> str | None:
    """The visible throttle dialog's text, or None. Kept separate from detect_quota: the
    quota hints scan dialog text for plan-limit wording, while this modal is a distinct
    access block with its own testid that can sit over an otherwise healthy page."""
    try:
        node = page.query_selector(RATE_LIMIT_MODAL_SELECTOR)
        if node is not None and node.is_visible():
            return normalize(node.inner_text())[:200]
    except Exception:
        pass
    return None


def dismiss_rate_limit_modal(page) -> None:
    """Click the modal's last button — language-agnostic on purpose (the one observed
    button reads '알겠습니다' in a Korean UI and would read differently elsewhere)."""
    try:
        node = page.query_selector(RATE_LIMIT_MODAL_SELECTOR)
        if node is None:
            return
        buttons = node.query_selector_all("button")
        if buttons:
            buttons[-1].click()
            time.sleep(0.5)
    except Exception:
        pass


def raise_if_rate_limited(page, context_note: str) -> None:
    limited = rate_limit_modal(page)
    if limited:
        dismiss_rate_limit_modal(page)
        raise RateLimitedError(f"{context_note}: {limited}")


# --- zero-target CDP recovery ----------------------------------------------------------


def ensure_page_target(port: int) -> None:
    """A stack Chrome whose windows were all closed by the user keeps answering
    /json/version but lists zero page targets, and connect_over_cdp then fails with
    "Browser context management is not supported" (measured 2026-08-13). Opening one tab
    over plain HTTP restores a connectable browser; if the recovery itself fails, the
    normal connect error is the one worth seeing."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=4) as result:
            targets = json.loads(result.read().decode("utf-8"))
        if any(target.get("type") == "page" for target in targets):
            return
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/new?url={CHATGPT_URL}", method="PUT"
        )
        with urllib.request.urlopen(request, timeout=6):
            pass
        log("no page target on CDP; opened one")
        time.sleep(2)
    except Exception:
        pass


# --- model picker ----------------------------------------------------------------------
# Chat mode only: Work mode has its own six-tick picker whose top tier (Ultra) is a multi-turn
# agentic mode, not the single long Pro answer this bridge exists for. Everything here fails
# closed with ModelVerificationError (exit 2, nothing sent) the moment the UI disagrees.


def effort_index(effort: str) -> int:
    name = normalize(effort.replace("-", " ").replace("_", " ")).casefold()
    if name not in EFFORT_LEVELS:
        raise UsageError(f"--effort must be one of: {', '.join(EFFORT_LEVELS)} (got {effort!r})")
    return EFFORT_LEVELS.index(name)


MODE_STATE_JS = """(sel) => [...document.querySelectorAll(sel)]
  .map(b => [(b.innerText || '').trim(), b.getAttribute('aria-pressed') === 'true'])"""


def composer_mode(page) -> str:
    """'chat' | 'work' | 'none' — 'none' when the page has no Chat/Work switch at all."""
    try:
        buttons = page.evaluate(MODE_STATE_JS, MODE_BUTTON_SELECTOR) or []
    except Exception:
        buttons = []
    modes = [
        ("chat" if CHAT_MODE_RE.match(text) else "work", pressed)
        for text, pressed in buttons
        if CHAT_MODE_RE.match(text) or WORK_MODE_RE.match(text)
    ]
    if not modes:
        return "none"
    return "chat" if ("chat", True) in modes else "work"


def ensure_chat_mode(page) -> None:
    """Flip a Work-mode composer back to Chat. Chat and Work keep separate model settings, so
    this never disturbs the Chat selection. A page without the switch is left alone: the
    picker checks that follow reject a Work-shaped menu anyway."""
    mode = composer_mode(page)
    if mode != "work":
        return
    log("composer is in Work mode; switching to Chat")
    for button in page.query_selector_all(MODE_BUTTON_SELECTOR):
        try:
            if not CHAT_MODE_RE.match(normalize(button.inner_text())) or not button.is_visible():
                continue
            try:
                button.click(timeout=5000)
            except Exception:
                button.dispatch_event("click")
            break
        except Exception:
            continue
    for _ in range(10):
        time.sleep(0.5)
        if composer_mode(page) == "chat":
            return
    raise ModelVerificationError("composer is in Work mode and could not be switched to Chat")


def open_picker(page) -> bool:
    for item in page.query_selector_all(PILL_SELECTOR):
        try:
            if not item.is_visible():
                continue
            try:
                item.click(timeout=5000)
            except Exception:
                item.dispatch_event("click")
            time.sleep(1.2)
            return True
        except Exception:
            continue
    return False


def close_menu(page) -> None:
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    time.sleep(0.3)


PICKER_STATE_JS = """(sel) => {
  const picker = document.querySelector(sel);
  if (!picker) return null;
  const text = el => (el ? (el.innerText || el.textContent || '') : '').replace(/\\s+/g, ' ').trim();
  const num = el => (el === null || el === undefined) ? null : Number(el);
  const toggle = picker.querySelector('[role="menuitem"][data-model-picker-view-toggle]');
  const effort = toggle && toggle.querySelector('[data-maximum]');
  const slider = picker.querySelector('[data-model-picker-power-slider] [role="slider"]');
  const sliderItem = picker.querySelector('[role="menuitem"][data-reasoning-slider]');
  const radios = [...picker.querySelectorAll('[role="menuitemradio"]')]
    .map(r => [text(r), r.getAttribute('aria-checked') === 'true']);
  const checked = radios.filter(r => r[1]);
  return {
    view: picker.getAttribute('data-model-picker-view'),
    model: checked.length === 1 ? checked[0][0] : null,
    label: text(toggle),
    max_effort: effort ? effort.getAttribute('data-maximum') === 'true' : null,
    value_now: slider ? num(slider.getAttribute('aria-valuenow')) : null,
    value_max: slider ? num(slider.getAttribute('aria-valuemax')) : null,
    slider_disabled: sliderItem ? sliderItem.getAttribute('aria-disabled') === 'true' : null,
    radios,
  };
}"""


def read_picker_state(page) -> dict | None:
    """One snapshot of the open picker, or None when no picker is on the page."""
    try:
        return page.evaluate(PICKER_STATE_JS, PICKER_SELECTOR)
    except Exception:
        return None


def latest_selected(state: dict) -> bool:
    """True only when exactly one model entry is checked and it is Latest."""
    return bool(state["model"]) and LATEST_MODEL_RE.match(state["model"]) is not None


def choose_latest_model(page) -> None:
    """Switch the picker to its model list and pick the Latest entry; confirm Latest is the checked
    entry. Picking returns the menu to its simple view, but callers still close and reopen it
    before touching the slider (it only answers on an untouched menu)."""
    state = read_picker_state(page)
    if state is None or state["view"] != "advanced":
        toggle = _q(page, [MODEL_TOGGLE_SELECTOR])
        if toggle is None:
            raise ModelVerificationError("model list toggle not found in the picker")
        try:
            toggle.click(timeout=5000)
        except Exception:
            toggle.dispatch_event("click")
        time.sleep(1.2)
    radios = page.query_selector_all(MODEL_RADIO_SELECTOR)
    seen = []
    for radio in radios:
        text = normalize(radio.inner_text())
        seen.append(text)
        if not LATEST_MODEL_RE.match(text):
            continue
        try:
            radio.click(timeout=5000)
        except Exception:
            radio.dispatch_event("click")
        time.sleep(1.3)
        state = read_picker_state(page)
        if state is None or not latest_selected(state):
            raise ModelVerificationError(
                f"clicked {text!r} but the picker reports model {state and state['model']!r}"
            )
        return
    raise ModelVerificationError(f"no Latest entry in the model list; the menu offers {seen!r}")


def choose_effort_tick(page, target: int) -> dict:
    """Click the wanted tick (arrow keys and thumb drags do not move this slider) and confirm
    aria-valuenow. Must run on a freshly opened menu."""
    ticks = page.query_selector_all(EFFORT_TICK_SELECTOR)
    if len(ticks) <= target:
        raise ModelVerificationError(
            f"effort slider has {len(ticks)} ticks; tick {target} ({EFFORT_LEVELS[target]}) is unavailable"
        )
    box = ticks[target].bounding_box()
    if not box:
        raise ModelVerificationError("effort slider tick has no geometry")
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    time.sleep(1.3)
    state = read_picker_state(page)
    if state is None or state["value_now"] != target:
        raise ModelVerificationError(
            f"effort tick {target} ({EFFORT_LEVELS[target]}) did not take; "
            f"slider reports {state and state['value_now']!r}"
        )
    return state


def _reopen_picker(page) -> dict:
    close_menu(page)
    if not open_picker(page):
        raise ModelVerificationError("model menu could not be reopened")
    state = read_picker_state(page)
    if state is None:
        raise ModelVerificationError("model picker disappeared after reopening the menu")
    return state


def select_model(page, effort: str) -> str:
    """Land the composer on Chat mode · Latest model · the wanted effort tick, verified.

    The verdict always comes from the open picker: Latest (no explicit model) at the wanted tick,
    and at Pro the model row must read exactly REQUIRED_PRO_LABEL. Below Pro the UI shows no
    model version, so the check there is Latest + tick index."""
    target = effort_index(effort)
    wanted = EFFORT_LEVELS[target]
    try:
        page.wait_for_selector(PILL_SELECTOR, timeout=20000)
    except Exception:
        pass
    try:
        return _select_model(page, target, wanted)
    except ModelVerificationError:
        close_menu(page)
        raise


def _select_model(page, target: int, wanted: str) -> str:
    ensure_chat_mode(page)

    if not open_picker(page):
        raise ModelVerificationError("model menu could not be opened")
    state = read_picker_state(page)
    if state is None:
        raise ModelVerificationError("model picker not found after opening the menu")
    if state["value_max"] != len(EFFORT_LEVELS) - 1:
        raise ModelVerificationError(
            f"unexpected effort slider (max tick {state['value_max']!r}, label {state['label']!r}); "
            "is the composer in Work mode?"
        )
    changed = False
    if state["model"] is None:
        raise ModelVerificationError(
            f"cannot tell the selected model; the model list reads {state['radios']!r}"
        )
    if not latest_selected(state):
        log(f"explicit model selected ({state['model']!r}, {state['label']!r}); switching to Latest")
        choose_latest_model(page)
        state = _reopen_picker(page)  # the slider only answers on an untouched menu
        changed = True
    if state["value_now"] != target:
        log(f"effort tick {state['value_now']} → {target} ({wanted})")
        choose_effort_tick(page, target)
        changed = True
    final = _reopen_picker(page) if changed else state
    close_menu(page)

    if not latest_selected(final) or final["value_now"] != target:
        raise ModelVerificationError(
            f"final check failed: expected Latest at tick {target} ({wanted}), picker reported "
            f"model={final['model']!r} tick={final['value_now']!r}"
        )
    if wanted == "pro":
        if final["label"] != REQUIRED_PRO_LABEL:
            raise ModelVerificationError(
                f"required label {REQUIRED_PRO_LABEL!r}, the picker's model row reads {final['label']!r}"
            )
        log(f"model verified: {REQUIRED_PRO_LABEL} (Latest · Pro)")
        return f"Latest ({REQUIRED_PRO_LABEL})"
    log(
        f"model verified: Latest; effort {wanted} (tick {target}, label {final['label']!r}); "
        "Latest shows no model version below Pro"
    )
    return f"Latest ({wanted})"


def attach_file(page, path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise UsageError(f"attachment is not a file: {path}")
    file_input = page.query_selector(FILE_INPUT_SELECTOR)
    if file_input is None:
        raise RuntimeError("file input is unavailable")
    file_input.set_input_files(str(path))
    log(f"uploading attachment: {path.name}")
    stem = path.stem[:14]
    composer = page.locator(COMPOSER_FORM_SELECTOR).first
    for _ in range(40):
        try:
            if composer.get_by_text(stem, exact=False).count() > 0:
                time.sleep(1.5)
                log(f"attachment verified: {path.name}")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"attachment chip did not appear: {path.name}")


# The composer element: the first INPUT_SELECTORS entry present (passed in as `sels`).
COMPOSER_EL_JS = "sels => sels.map(s => document.querySelector(s)).find(Boolean)"


def put_text(page, prompt: str) -> None:
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.evaluate(
        f"""sels => {{ const el = ({COMPOSER_EL_JS})(sels);
            if (el) {{ el.scrollIntoView({{block: 'center'}}); el.focus(); }} }}""",
        INPUT_SELECTORS,
    )
    try:
        page.keyboard.insert_text(prompt)
    except Exception:
        page.keyboard.type(prompt)
    time.sleep(0.5)


def composer_text(page) -> str:
    return page.evaluate(
        f"""sels => {{ const el = ({COMPOSER_EL_JS})(sels);
            return el ? (el.innerText || el.textContent || '') : ''; }}""",
        INPUT_SELECTORS,
    ) or ""


def composer_has_prompt(page, prompt: str) -> bool:
    wanted = normalize(prompt)
    got = normalize(composer_text(page))
    return bool(wanted) and got == wanted


def clear_composer(page) -> None:
    page.evaluate(f"sels => {{ const el = ({COMPOSER_EL_JS})(sels); if (el) el.focus(); }}", INPUT_SELECTORS)
    page.keyboard.press("ControlOrMeta+a")  # plain Control+a only moves the caret on macOS
    page.keyboard.press("Backspace")


def click_send(page) -> None:
    for _ in range(15):
        for selector in SEND_BTN_SELECTORS:
            try:
                button = page.query_selector(selector)
                if button and button.is_visible() and button.is_enabled():
                    button.click()
                    log("prompt sent")
                    return
            except Exception:
                continue
        time.sleep(1)
    raise RuntimeError("send button never became enabled")


def release_submit_lock() -> None:
    raw_fd = os.environ.pop("CHATGPT_SUBMIT_LOCK_FD", None)
    lock_info = os.environ.pop("CHATGPT_SUBMIT_LOCK_INFO", None)
    if raw_fd is None:
        return
    if lock_info:
        try:
            Path(lock_info).unlink()
        except FileNotFoundError:
            pass
    fd = int(raw_fd)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    log("submit lock released")


def text_key(text: str | None) -> str:
    """Letters and digits only, casefolded: the same for a prompt and its rendered bubble."""
    return re.sub(r"[\W_]+", "", text or "").casefold()


def own_user_turn(page, base_ids: set[str], prompt: str) -> str | None:
    """The message id of the user turn this run sent: a user unit the page did not show before the
    send (by id: a long thread's page shows only its newest turns, fetched with `num_turns=10`, so a
    count proves nothing) whose text carries the start of this run's prompt. Another run's turn that
    shows up on this page is new too, hence the text check. None until such a turn is on the page;
    two such turns cannot be told apart, which fails the run."""
    key = text_key(prompt)[:PROMPT_KEY_CHARS]
    found = []
    for node in _qa(page, USER_MSG_SELECTORS):
        try:
            fresh = node_message_ids(node) - base_ids
            if fresh and key in text_key(node.inner_text()):
                found.append(fresh)
        except Exception:
            continue
    if not found:
        return None
    if len(found) > 1 or len(found[0]) > 1:
        raise RuntimeError("more than one new user turn carries this prompt; cannot tell which one this run sent")
    return next(iter(found[0]))


def _poll(check: Callable[[], object], secs: float):
    """The first truthy value `check` returns within `secs`, else None."""
    deadline = time.monotonic() + secs
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(1)
    return None


def confirm_sent_and_capture(
    page, base_ids: set[str], prompt: str, bound_url: str | None = None
) -> tuple[str, str]:
    """Wait for this run's user turn (own_user_turn), then return the conversation URL and the turn's
    message id, which binds the harvest to this run's own exchange. A turn the live page never shows
    gets one reload of the conversation (the bound one, or the /c/<id> a fresh chat has flipped to),
    and the turn must then be there. The URL alone proves nothing: a follow-up page carries it from
    the start."""
    turn_id = _poll(lambda: own_user_turn(page, base_ids, prompt), 45)
    if turn_id is None:
        url = bound_url or (current_url(page) if CONV_URL_RE.search(current_url(page)) else None)
        if url is None:
            raise RuntimeError("no new user turn appeared after send")
        log("no new user turn on the live page; reloading the conversation to look for it")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        turn_id = _poll(lambda: own_user_turn(page, base_ids, prompt), 30)
        if turn_id is None:
            raise RuntimeError("no new user turn appeared after send (nor after reloading the conversation)")
        log("new user turn found after reload")
    if bound_url is not None:
        return bound_url, turn_id
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        url = current_url(page)
        if CONV_URL_RE.search(url):
            return url, turn_id
        time.sleep(1)
    raise SentUnknownLocationError("prompt was sent but the conversation URL was not captured")


def is_streaming(page) -> bool:
    return _q(page, STREAMING_BTN_SELECTORS) is not None


def reply_in_progress(page, secs: float = REPLY_IN_PROGRESS_POLL_SECS) -> bool:
    """True when the open conversation is still generating an earlier reply. A prompt sent then
    does not start a new turn: it joins the running exchange (measured 2026-09-28), where it can be
    neither confirmed nor harvested. Measured 2026-09-28 on freshly opened tabs of a conversation
    whose Pro reply was running: the composer's stop button (중지) was there within 0.1 s of the
    composer (27 of 27 opens from 10 s after the send on); a finished conversation never showed it
    (2 of 2, 30 s each). The short poll is margin for a slower load. Blind spot: a tab that loads
    the conversation within ~5-10 s of another tab's send shows neither that turn nor the stop
    button, and never catches up without a reload."""
    return bool(_poll(lambda: is_streaming(page), secs))


def detect_quota(page) -> str | None:
    try:
        surfaces = page.query_selector_all('[role="dialog"], [role="alert"]')
    except Exception:
        return None
    for surface in surfaces:
        try:
            text = normalize(surface.inner_text())
        except Exception:
            continue
        lowered = text.casefold()
        if any(hint.casefold() in lowered for hint in QUOTA_HINTS):
            return text[:200]
    return None


MARKDOWN_SERIALIZER = r"""
(root) => {
  const clean = s => (s || '').replace(/\u00a0/g, ' ');
  const inline = node => {
    if (node.nodeType === Node.TEXT_NODE) return clean(node.nodeValue);
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const tag = node.tagName.toLowerCase();
    if (tag === 'br') return '\n';
    if (tag === 'code' && node.parentElement?.tagName.toLowerCase() !== 'pre')
      return '`' + clean(node.textContent).replace(/`/g, '\\`') + '`';
    const body = Array.from(node.childNodes).map(inline).join('');
    if (tag === 'strong' || tag === 'b') return '**' + body + '**';
    if (tag === 'em' || tag === 'i') return '*' + body + '*';
    if (tag === 'del' || tag === 's') return '~~' + body + '~~';
    if (tag === 'a') return '[' + body + '](' + (node.getAttribute('href') || '') + ')';
    if (tag === 'img') return '![' + (node.getAttribute('alt') || '') + '](' + (node.getAttribute('src') || '') + ')';
    return body;
  };
  const block = (node, depth = 0) => {
    if (node.nodeType === Node.TEXT_NODE) return clean(node.nodeValue);
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const tag = node.tagName.toLowerCase();
    if (/^h[1-6]$/.test(tag)) return '#'.repeat(Number(tag[1])) + ' ' + inline(node).trim() + '\n\n';
    if (tag === 'pre') {
      const code = node.querySelector('code');
      const value = clean((code || node).textContent).replace(/\n$/, '');
      const language = ((code?.className || '').match(/language-([\w+-]+)/) || [,''])[1];
      const ticks = '`'.repeat(Math.max(3, ...((value.match(/`+/g) || []).map(x => x.length + 1))));
      return ticks + language + '\n' + value + '\n' + ticks + '\n\n';
    }
    if (tag === 'ul' || tag === 'ol') {
      let n = 1;
      return Array.from(node.children).filter(x => x.tagName.toLowerCase() === 'li').map(li => {
        const prefix = tag === 'ol' ? `${n++}. ` : '- ';
        const own = Array.from(li.childNodes).filter(x => !(x.nodeType === 1 && ['ul','ol'].includes(x.tagName.toLowerCase()))).map(block).join('').trim();
        const nested = Array.from(li.children).filter(x => ['ul','ol'].includes(x.tagName.toLowerCase())).map(x => block(x, depth + 1).trimEnd().split('\n').map(line => '  ' + line).join('\n')).join('\n');
        return prefix + own + (nested ? '\n' + nested : '');
      }).join('\n') + '\n\n';
    }
    if (tag === 'blockquote') return blockChildren(node).trim().split('\n').map(x => '> ' + x).join('\n') + '\n\n';
    if (tag === 'hr') return '---\n\n';
    if (tag === 'table') {
      const rows = Array.from(node.querySelectorAll('tr')).map(tr => Array.from(tr.querySelectorAll(':scope > th, :scope > td')).map(cell => inline(cell).trim().replace(/\|/g, '\\|')));
      if (!rows.length) return '';
      const width = Math.max(...rows.map(r => r.length));
      const render = row => '| ' + Array.from({length: width}, (_, i) => row[i] || '').join(' | ') + ' |';
      return render(rows[0]) + '\n' + render(Array(width).fill('---')) + '\n' + rows.slice(1).map(render).join('\n') + '\n\n';
    }
    if (tag === 'p') return inline(node).trim() + '\n\n';
    if (tag === 'br') return '\n';
    return blockChildren(node);
  };
  const blockChildren = node => Array.from(node.childNodes).map(child => block(child)).join('');
  return blockChildren(root).replace(/\n[ \t]+\n/g, '\n\n').replace(/\n{3,}/g, '\n\n').trim();
}
"""


def assistant_markdown(node) -> str:
    try:
        markdown = _q(node, ASSISTANT_MARKDOWN_SELECTORS)
        return (markdown or node).evaluate(MARKDOWN_SERIALIZER) or ""
    except Exception:
        try:
            return node.inner_text() or ""
        except Exception:
            return ""


def own_exchange(page, turn_id: str):
    """This run's exchange wrapper: [data-turn-key] keyed by its own user turn's id. Nothing outside
    it is ever harvested."""
    if not MESSAGE_ID_RE.match(turn_id):
        raise RuntimeError(f"unexpected message id {turn_id!r}")
    return page.query_selector(f'[{EXCHANGE_KEY_ATTR}="{turn_id}"]')


def exchange_answer(exchange):
    """The exchange's assistant unit (the last one, should there ever be more), or None."""
    nodes = _qa(exchange, ASSISTANT_MSG_SELECTORS)
    return nodes[-1] if nodes else None


def user_turn_on_page(page, turn_id: str) -> bool:
    return page.query_selector(f'[{MESSAGE_IDS_ATTR}~="{turn_id}"]') is not None


# The exchange is done once its assistant action bar (not the user unit's own bar, whose copy is
# labelled 메시지 복사) carries the copy action.
EXCHANGE_DONE_JS = """w => [...w.querySelectorAll(%s)]
  .filter(bar => !bar.closest(%s))
  .some(bar => bar.querySelector(%s) !== null)""" % tuple(
    json.dumps(value)
    for value in (TURN_ACTIONS_SELECTOR, ", ".join(USER_MSG_SELECTORS), ", ".join(COPY_BTN_SELECTORS))
)


def exchange_complete(page, exchange) -> bool:
    """Nothing streams and this exchange's own assistant bar shows the copy action. Earlier
    exchanges keep their bars, so nothing outside the exchange counts."""
    if is_streaming(page):
        return False
    try:
        return bool(exchange.evaluate(EXCHANGE_DONE_JS))
    except Exception:
        return False


def wait_for_response(
    page,
    conversation_url: str,
    turn_id: str,
    deadline: float,
) -> tuple[str, set[str]]:
    """The finished answer of this run's own exchange (see own_exchange) and the assistant message ids
    it was read from. A turn that stays off the page for OWN_TURN_GRACE_SECS fails the harvest: some
    other exchange's answer is never a substitute."""
    key = conversation_key(conversation_url)
    stable_since = None
    missing_since = None
    previous = ""
    last_status = -STATUS_INTERVAL
    log(f"waiting for response (up to {max(0, int(deadline - time.monotonic()))}s)")
    while time.monotonic() < deadline:
        if key not in current_url(page):
            log("conversation drift detected; returning to the bound URL")
            page.goto(conversation_url, wait_until="domcontentloaded", timeout=60000)
            stable_since = None
            time.sleep(2)
            continue
        remaining = int(deadline - time.monotonic())
        if remaining // STATUS_INTERVAL != last_status:
            status = "generating" if is_streaming(page) else "checking completion"
            log(f"response {status}; {remaining}s remaining")
            last_status = remaining // STATUS_INTERVAL
        exchange = own_exchange(page, turn_id)
        if exchange is None:
            if user_turn_on_page(page, turn_id):
                raise RuntimeError(f"this run's turn {turn_id} is on the page outside an exchange wrapper")
            missing_since = missing_since or time.monotonic()
            if time.monotonic() - missing_since >= OWN_TURN_GRACE_SECS:
                raise RuntimeError(f"this run's turn {turn_id} is no longer on the page")
        else:
            missing_since = None
        node = exchange_answer(exchange) if exchange is not None else None
        if node is None or not exchange_complete(page, exchange):
            # The prompt is already in flight here: a throttle now blocks only the
            # harvest, so the error must carry the conversation URL for a later pickup.
            raise_if_rate_limited(
                page,
                f"rate limited while harvesting; the prompt was sent ({conversation_url})",
            )
            quota = detect_quota(page)
            if quota:
                raise RuntimeError(f"ChatGPT usage limit: {quota}")
            stable_since = None
            time.sleep(2)
            continue
        current = assistant_markdown(node).strip()
        if not current:
            stable_since = None
            time.sleep(2)
            continue
        if normalize(current) != normalize(previous):
            previous = current
            stable_since = time.monotonic()
            time.sleep(1)
            continue
        if stable_since is not None and time.monotonic() - stable_since >= STABLE_SECS:
            log(f"response received: {len(current)} characters")
            return current, node_message_ids(node)
        time.sleep(1)
    raise ResponseTimeoutError("response wait timed out")


# --- the server's record ---------------------------------------------------------------
# The conversation as the server keeps it, read with the page's own session token and reduced to
# each message's parent, role, status, turn_exchange_id and end_turn, the text of user messages, and
# whether a user message has versions (another prompt answering the same message).
# Measured 2026-09-28:
# - Every message of an exchange carries its user turn's turn_exchange_id; the answer the page shows
#   is the exchange's last message, role assistant, channel 'final', end_turn true (45 messages of
#   two scratch threads, and the 1,716-message agentic work thread 6ab7a0f7, whose answers sit after
#   up to 417 tool calls, tool outputs, commentary, hidden sub-agent notes and plan updates, all
#   'finished_successfully' under the same exchange id; the page lists only the final message).
# - A prompt sent into a running reply becomes the child of that reply's latest message and joins
#   its exchange (same turn_exchange_id); the earlier prompt is left without an answer.
# - While a Pro reply runs, the conversation's async_status is 3 and the leaf (current_node) is the
#   user turn or a 'thoughts' message, every message already 'finished_successfully'; once the reply
#   ends or is stopped, async_status is null and the leaf has end_turn true.
# - Two endpoints. /backend-api/conversation/<id> returns the whole tree (13.7 MB on the work
#   thread). The page itself reads /backend-api/conversations/<id>?num_turns=10&include_has_versions=true:
#   the newest turns of the current branch as a list in branch order (a user prompt and its reply
#   count as two turns: num_turns=2 is the last exchange, 205 messages and 2.7 MB on the work
#   thread; num_turns=10 was 16.8 MB, more than the whole tree), with the same role, status, end_turn, channel, recipient, turn_exchange_id and text,
#   async_status and current_node, page_info.has_previous_page, and has_versions on a user message
#   whose parent has another child (the sibling prompts of a race thread). It lists no parent of a
#   user message, and the first reply message's metadata.parent_id names a message that is not in
#   the tree, so a message's parent is the one listed before it. Other branches are not listed.
# - Read every ~3 s for a minute, the whole-tree endpoint answered 429 ("Too many requests") for
#   minutes after, on every conversation of the account, and the page then could not load the
#   conversation either. So each check reads the last exchange only, widens to the whole tree only
#   when this run's messages are not in it, and waits out a 429 (Retry-After, else exponential
#   backoff) instead of reading again.
SERVER_CONVERSATION_JS = """async ({cid, turns}) => {
  try {
    const session = await fetch('/api/auth/session');
    if (!session.ok) return {error: 'HTTP ' + session.status, status: session.status, retry_after: session.headers.get('retry-after')};
    const token = (await session.json()).accessToken;
    const url = turns ? '/backend-api/conversations/' + cid + '?num_turns=' + turns + '&include_has_versions=true'
                      : '/backend-api/conversation/' + cid;
    const response = await fetch(url, {headers: {Authorization: 'Bearer ' + token}});
    if (!response.ok) return {error: 'HTTP ' + response.status, status: response.status, retry_after: response.headers.get('retry-after')};
    const conversation = await response.json();
    const text = message => ((message.content || {}).parts || []).filter(p => typeof p === 'string').join('\\n');
    const reduce = (message, parent) => {
      const role = message ? message.author.role : null, metadata = (message && message.metadata) || {};
      return {
        parent: parent || null,
        role,
        status: message ? message.status || null : null,
        exchange: metadata.turn_exchange_id || null,
        end_turn: message ? message.end_turn === true : false,
        versions: metadata.has_versions === true,
        text: role === 'user' ? text(message) : '',
      };
    };
    const messages = {};
    if (turns) {
      let previous = null;
      for (const message of conversation.messages || []) {
        messages[message.id] = reduce(message, previous);
        previous = message.id;
      }
    } else {
      for (const [id, node] of Object.entries(conversation.mapping || {})) messages[id] = reduce(node.message, node.parent);
    }
    return {
      current_node: conversation.current_node || null,
      async_status: conversation.async_status ?? null,
      complete: turns ? !(conversation.page_info || {}).has_previous_page : true,
      messages,
    };
  } catch (error) {
    return {error: 'network: ' + String(error).slice(0, 200)};
  }
}"""
LAST_EXCHANGE_TURNS = 2
# Waits between reads: Retry-After when the server sends it, otherwise exponential backoff with
# jitter from SERVER_BACKOFF_FIRST doubling to SERVER_BACKOFF_MAX seconds, for as long as the check's
# patience: the harvest's (a finished answer is at stake) and the pre-send guard's (a refused send
# costs nothing; nothing is typed).
SERVER_BACKOFF_FIRST = 5
SERVER_BACKOFF_MAX = 120
HARVEST_PATIENCE = 20 * 60
GUARD_PATIENCE = 2 * 60
# A record that reads fine but lacks this run's messages or leaves them unfinished is waited for about
# 3 minutes: the page already showed the answer finished, and a mixed race answer stays unfinished.
RECORD_PATIENCE = 3 * 60


class ServerReadError(RuntimeError):
    """One read of the server's record failed (HTTP status, network, page)."""

    def __init__(self, problem: str, retry_after: float | None = None):
        super().__init__(problem)
        self.retry_after = retry_after


class ServerUnreadableError(RuntimeError):
    """The harvest check could not read the server's record within its patience."""


class UnverifiedReplyError(RuntimeError):
    """The answer was harvested from the page but the server's record could not be read to check it."""

    def __init__(self, message: str, body: str, conversation_url: str):
        super().__init__(message)
        self.body, self.conversation_url = body, conversation_url


def retry_after_secs(value) -> float | None:
    """A Retry-After header: delta seconds or an HTTP date."""
    if value in (None, ""):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        return max(0.0, parsedate_to_datetime(str(value)).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def server_conversation(page, conversation_url: str, turns: int | None = LAST_EXCHANGE_TURNS) -> dict:
    """One read of the server's record (SERVER_CONVERSATION_JS): the newest `turns` turns of the
    current branch, or the whole tree when `turns` is None. ServerReadError when it fails."""
    try:
        found = page.evaluate(SERVER_CONVERSATION_JS, {"cid": conversation_id(conversation_url), "turns": turns}) or {}
    except Exception as exc:
        raise ServerReadError(str(exc).splitlines()[0] if str(exc) else type(exc).__name__) from None
    if found.get("error") or not isinstance(found.get("messages"), dict):
        raise ServerReadError(found.get("error") or "no messages in the server's answer", retry_after_secs(found.get("retry_after")))
    return found


def server_wait(attempt: int, deadline: float, problem: str, retry_after: float | None = None) -> bool:
    """Sleep before the next read, or False when that would pass the deadline. One stderr line per wait."""
    if retry_after is not None:
        wait = max(1.0, retry_after)
    else:
        wait = min(SERVER_BACKOFF_MAX, SERVER_BACKOFF_FIRST * 2**attempt) * random.uniform(0.75, 1.0)
    if time.monotonic() + wait > deadline:
        return False
    reason = "server busy (HTTP 429)" if problem == "HTTP 429" else f"server record: {problem}"
    log(f"{reason}; retrying in {round(wait)} s")
    time.sleep(wait)
    return True


def prompt_of(messages: dict, message_id: str) -> str | None:
    """The user turn a message answers: its nearest user ancestor on the server's tree."""
    node = messages.get(message_id)
    for _ in range(len(messages)):
        if node is None or node["parent"] is None:
            return None
        parent = node["parent"]
        node = messages.get(parent)
        if node is not None and node["role"] == "user":
            return parent
    return None


def server_reply_running(conversation: dict) -> bool:
    """The server is still generating the reply at the conversation's leaf: async_status is set and
    the leaf is not an answer that ended its turn. Only the leaf counts: an earlier prompt left
    without an answer further up does not block."""
    leaf = conversation["messages"].get(conversation.get("current_node") or "")
    ended = leaf is not None and leaf["role"] == "assistant" and leaf["end_turn"]
    return conversation.get("async_status") is not None and not ended


REPLY_RUNNING = "a reply is still in progress in this conversation; nothing sent"


def refuse_if_reply_running(page, conversation_url: str) -> None:
    """Before a follow-up is typed: the composer's stop button (reply_in_progress) is missing from a
    page that shows a Pro reply as 'Thinking' after a reload, so the server's record decides too —
    its last exchange, which always holds the leaf. A record that cannot be read within
    GUARD_PATIENCE refuses the send as well."""
    deadline = time.monotonic() + GUARD_PATIENCE
    attempt = 0
    while True:
        try:
            conversation = server_conversation(page, conversation_url)
        except ServerReadError as exc:
            if server_wait(attempt, deadline, str(exc), exc.retry_after):
                attempt += 1
                continue
            raise RuntimeError(f"could not read the conversation's state on the server ({exc}); nothing sent") from None
        if server_reply_running(conversation):
            raise RuntimeError(REPLY_RUNNING)
        return


def verify_reply_exchange(page, conversation_url: str, turn_id: str, reply_ids: set[str], prompt: str) -> None:
    """Refuse an answer the server does not file as this run's own. The page ties the answer to this
    run's turn, but measured 2026-09-28 (two races on a scratch thread) that is not enough: a prompt
    sent from a page that did not yet show another run's fresh turn replies to the same message as
    that turn (a sibling branch), both generations run at once, and the server mixes them — the
    other run's answer streamed into this run's wrapper, the server filed one run's final text under
    the other's exchange, and the answer that streamed was left 'in_progress' with no text. So this
    run's turn must have no sibling prompt, and the answer must come from this run's exchange
    (turn_exchange_id), follow this run's turn with no other prompt in between (a prompt sent into
    a running reply shares its exchange), be finished there and end the turn.

    Reads the last exchange (2.7 MB on the work thread); when this run's messages are not all in it
    (a later turn, another branch, or a record still being written), the whole tree (13.7 MB; the
    newest ten turns were 16.8 MB there), which later tries keep reading. When reads
    keep failing for HARVEST_PATIENCE, ServerUnreadableError; when the record stays incomplete or
    unfinished for RECORD_PATIENCE, RuntimeError."""
    if not reply_ids:
        raise RuntimeError(f"the reply carries no message id, so it cannot be checked against this run's turn; see {conversation_url}")
    ids = [turn_id, *sorted(reply_ids)]
    start = time.monotonic()
    turns: int | None = LAST_EXCHANGE_TURNS
    attempt, problem, unreadable, unfinished_seen = 0, "", False, False
    while True:
        retry_after = None
        try:
            conversation = server_conversation(page, conversation_url, turns)
        except ServerReadError as exc:
            problem, retry_after, unreadable = str(exc), exc.retry_after, True
        else:
            messages = conversation["messages"]
            missing = [message_id for message_id in ids if not (messages.get(message_id) or {}).get("role")]
            if missing and turns is not None:
                turns = None
                continue  # the whole tree at the same moment, not a retry
            unreadable = False
            if missing:
                problem = f"not on the server yet: {', '.join(missing)}"
            else:
                _refuse_foreign(messages, turn_id, sorted(reply_ids), prompt, conversation_url)
                unfinished = [message_id for message_id in ids if messages[message_id]["status"] != "finished_successfully"]
                if not unfinished and any(messages[message_id]["end_turn"] for message_id in reply_ids):
                    return
                problem = f"not finished on the server: {', '.join(unfinished or sorted(reply_ids))}"
                unfinished_seen = True
        deadline = start + (HARVEST_PATIENCE if unreadable else RECORD_PATIENCE)
        if not server_wait(attempt, deadline, problem, retry_after):
            break
        attempt += 1
    message = f"could not confirm the reply as this run's own on the server ({problem})"
    if unreadable and not unfinished_seen:  # an answer once seen unfinished is one of the race shapes
        raise ServerUnreadableError(message)
    raise RuntimeError(f"{message}; nothing returned — read it at {conversation_url}")


def _refuse_foreign(messages: dict, turn_id: str, reply_ids: list[str], prompt: str, conversation_url: str) -> None:
    user = messages[turn_id]
    if user["role"] != "user" or text_key(prompt)[:PROMPT_KEY_CHARS] not in text_key(user["text"]):
        raise ForeignReplyError(f"the server's message {turn_id} is not this run's prompt; see {conversation_url}")
    if user.get("versions") or any(
        message_id != turn_id and message["parent"] == user["parent"] and message["role"] == "user"
        for message_id, message in messages.items()
    ):
        raise ForeignReplyError(
            "another prompt was sent in reply to the same message as this run's (a sibling branch), and the "
            f"server mixes such answers; nothing returned — see {conversation_url}"
        )
    exchange = user["exchange"]
    foreign = [
        message_id
        for message_id in reply_ids
        if messages[message_id]["role"] != "assistant" or not exchange or messages[message_id]["exchange"] != exchange
    ]
    if foreign:
        raise ForeignReplyError(
            "the answer shown under this run's prompt belongs to another prompt's exchange "
            f"(messages {', '.join(foreign)}); nothing returned — see {conversation_url}"
        )
    later = [message_id for message_id in reply_ids if prompt_of(messages, message_id) != turn_id]
    if later:
        raise ForeignReplyError(
            "the answer shown under this run's prompt answers another prompt sent after it into the same exchange "
            f"(messages {', '.join(later)}); nothing returned — see {conversation_url}"
        )


def ask(
    prompt: str,
    *,
    effort: str,
    attach: Path | None,
    max_wait: int,
    project: str | None = None,
    conversation: str | None = None,
) -> Reply:
    """One prompt, one reply. The conversation lock (lock_conversation) is held from before anything
    is typed — a follow-up takes it up front, a fresh chat as soon as its URL is known and before the
    submit lock is released — until the reply is harvested or the run gives up."""
    held: list[int] = []
    if conversation:
        fd = lock_conversation(conversation)
        if fd is None:
            holder = conversation_lock_holder(conversation)
            raise RuntimeError(CONVERSATION_BUSY + (f" (held by {holder})" if holder else ""))
        held.append(fd)
    try:
        return _ask(
            prompt, effort=effort, attach=attach, max_wait=max_wait, project=project,
            conversation=conversation, held=held,
        )
    finally:
        for fd in held:
            unlock_conversation(fd)


def _ask(
    prompt: str,
    *,
    effort: str,
    attach: Path | None,
    max_wait: int,
    project: str | None,
    conversation: str | None,
    held: list[int],
) -> Reply:
    if sync_playwright is None:
        raise RuntimeError(
            "Python package 'playwright' is required "
            "(python3 -m pip install playwright; no browser download needed — "
            "this only attaches to an already-running Chrome over CDP)"
        )
    port = int(os.environ.get("CHATGPT_CDP_PORT", "9222"))
    if not port_open(port) or not cdp_browser_ok(port):
        raise RuntimeError(f"CDP {port} is not a supported Chromium browser")
    ensure_page_target(port)

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        context = pick_context(browser)
        if context is None:
            raise RuntimeError("no browser context is available")
        page = context.new_page()
        _guard_dialogs(context, page)
        try:
            page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=60000)
            state = login_state(page)
            if state != "ok":
                detail = "login wall detected" if state == "no" else "composer not detected"
                raise RuntimeError(f"ChatGPT session unavailable: {detail}")
            raise_if_rate_limited(page, "rate limited before submit; nothing sent")

            bound_url = None
            if conversation:
                bound_url = open_conversation(page, conversation)
                if reply_in_progress(page):
                    raise RuntimeError(REPLY_RUNNING)
                log(f"following up in conversation: {bound_url}")
            elif project:
                if enter_project(page, project):
                    log(f"chat grouped under project {project!r}: {current_url(page)}")
                else:
                    # A project is a convenience: any failure to secure one falls back to
                    # a plain chat with a working composer, never an abort.
                    log(f"project {project!r} unavailable; continuing in a plain chat")
                    page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=60000)
                    if login_state(page) != "ok":
                        raise RuntimeError("ChatGPT session unavailable after project fallback")

            select_model(page, effort)
            if attach is not None:
                attach_file(page, attach)

            base_ids = message_ids(page)
            if bound_url is not None:
                refuse_if_reply_running(page, bound_url)

            put_text(page, prompt)
            if not composer_has_prompt(page, prompt):
                clear_composer(page)
                put_text(page, prompt)
                if not composer_has_prompt(page, prompt):
                    raise RuntimeError("prompt did not enter the composer intact")
            raise_if_rate_limited(page, "rate limited before submit; nothing sent")
            click_send(page)
            conversation_url, turn_id = confirm_sent_and_capture(page, base_ids, prompt, bound_url)
            log(f"thread bound: {thread_handle(conversation_id(conversation_url))} ({conversation_url}); turn {turn_id}")
            if not held:  # a fresh chat: locked before the submit lock lets a --continue in
                fd = lock_conversation(conversation_url)
                if fd is None:
                    raise RuntimeError(f"the new conversation is already locked by another run; see {conversation_url}")
                held.append(fd)
            try:
                record_thread(conversation_url, prompt)
            except Exception as exc:  # bookkeeping — the reply, not the ledger, is the deliverable
                log(f"thread ledger not updated: {exc}")
            release_submit_lock()
            deadline = time.monotonic() + max_wait

            for attempt in range(2):
                try:
                    body, reply_ids = wait_for_response(page, conversation_url, turn_id, deadline)
                    break
                except (ResponseTimeoutError, RateLimitedError):
                    raise  # neither gets better by reloading the page right away
                except Exception as exc:
                    if attempt == 1 or time.monotonic() >= deadline:
                        raise
                    log(f"harvest interrupted; retrying the same conversation: {exc}")
                    try:
                        page.close()
                    except Exception:
                        pass
                    page = context.new_page()
                    _guard_dialogs(context, page)
                    page.goto(conversation_url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(2)
            try:
                verify_reply_exchange(page, conversation_url, turn_id, reply_ids, prompt)
            except ServerUnreadableError as exc:
                raise UnverifiedReplyError(str(exc), body, conversation_url) from None
            log(f"reply checked on the server: {', '.join(sorted(reply_ids))} answer turn {turn_id}")
            return Reply(body, conversation_url)
        finally:
            try:
                page.close()
            except Exception:
                pass


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    ask_fn: Callable[..., Reply] = ask,
) -> int:
    try:
        args = parse_args(argv)
        prompt = read_prompt(args, stdin or sys.stdin)
        if args.attach is not None and not args.attach.expanduser().is_file():
            raise UsageError(f"attachment is not a file: {args.attach}")
        output = response_path(args.out)
        conversation = resolve_thread(args)
        project = None
        if not args.no_project and conversation is None:
            project = args.project or default_project_name()
        reply = ask_fn(
            prompt,
            effort=args.effort,
            attach=args.attach,
            max_wait=args.max_wait,
            project=project,
            conversation=conversation,
        )
        body = reply.body.strip()
        if not body:
            raise RuntimeError("harvested response is empty")
        atomic_write(output, body + "\n")  # the saved file is the pure response
        print(output if args.quiet else body)
        print()
        print(thread_trailer(reply.conversation_url))
        return 0
    except UsageError as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 64
    except ModelVerificationError as exc:
        print(f"chatgpt: model verification failed; prompt not sent: {exc}", file=sys.stderr)
        return 2
    except ResponseTimeoutError as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 3
    except RateLimitedError as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 5
    except UnverifiedReplyError as exc:
        # Explicitly marked, never printed as the answer: the page showed it under this run's turn,
        # but the server's record, which catches the race shapes, could not be read.
        atomic_write(
            output,
            f"UNVERIFIED: not confirmed as the answer to this run's prompt ({exc}); check it at {exc.conversation_url}\n\n"
            f"{exc.body.strip()}\n",
        )
        print(
            f"chatgpt: {exc}; the answer the page showed is saved, marked UNVERIFIED, at {output} — "
            f"check it at {exc.conversation_url}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("chatgpt: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
