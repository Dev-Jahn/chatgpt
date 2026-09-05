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
INPUT_SELECTORS = ["#prompt-textarea", 'div[contenteditable="true"]']
FILE_INPUT_SELECTOR = 'input[type="file"]'
USER_MSG_SELECTORS = ['[data-message-author-role="user"]', 'article[data-turn="user"]']
ASSISTANT_MSG_SELECTORS = [
    '[data-message-author-role="assistant"]',
    'article[data-turn="assistant"]',
]
COPY_BTN_SELECTORS = [
    'button[data-testid="copy-turn-action-button"]',
    'button[aria-label="Copy"]',
    'button[data-testid*="copy"]',
]
STREAMING_BTN_SELECTORS = [
    'button[data-testid="stop-button"]',
    'button[aria-label="Stop streaming"]',
    'button[data-testid*="stop"]',
]
SEND_BTN_SELECTORS = [
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
# Composer model picker — Chat mode, measured live 2026-09-05 (DOM notes: .hippo/briefs/dom-facts.md on
# dev). The pill opens a two-level menu: a model list (최신/Latest plus explicit versions) behind a toggle and a
# five-tick effort slider. The pin is the label the CLOSED pill shows at max effort with the Latest
# model — "6 Pro" is GPT-6 Pro today; below Pro, Latest is labelled by effort alone (an explicit
# model keeps its version prefix, e.g. "5.6 High").
REQUIRED_PRO_LABEL = "6 Pro"
LATEST_MODEL_RE = re.compile(r"^(최신|latest|auto)$", re.I)
EFFORT_LEVELS = ("instant", "medium", "high", "extra high", "pro")  # slider ticks 0..4
PILL_SELECTOR = 'button.__composer-pill[aria-haspopup="menu"]'
MODE_RADIO_SELECTOR = '[role="radio"][data-tpp-toggle-value]'
CHAT_MODE_RADIO_SELECTOR = '[role="radio"][data-tpp-toggle-value="chatgpt"]'
PICKER_SELECTOR = '[data-testid="composer-intelligence-picker-content"]'
MODEL_TOGGLE_SELECTOR = f'{PICKER_SELECTOR} [role="menuitem"][aria-expanded]'
MODEL_RADIO_SELECTOR = f'{PICKER_SELECTOR} [role="menuitemradio"]'
EFFORT_TICK_SELECTOR = "[data-model-reasoning-effort-slider] span[data-selected]"
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
CONVERSATION_ID_RE = re.compile(r"^[0-9a-f]{8}[0-9a-f-]{4,}$", re.I)
STABLE_SECS = 4
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
        dest="conversation",
        metavar="CONVERSATION",
        help="follow up inside an existing conversation (its chatgpt.com URL or bare id); "
        "the thread's context is retained and project grouping does not apply",
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
    if args.conversation is not None:
        if args.project is not None or args.no_project:
            raise UsageError(
                "--continue reopens an existing conversation; --project/--no-project do not apply"
            )
        args.conversation = conversation_url(args.conversation)
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


def conversation_url(text: str) -> str:
    """Normalize --continue: a chatgpt.com conversation URL is kept as given (a project chat lives
    under /g/g-p-…/c/<id>; ChatGPT resolves the bare /c/<id> form to it as well), a bare id
    becomes https://chatgpt.com/c/<id>."""
    value = text.strip().rstrip("/")
    if CONVERSATION_ID_RE.match(value):
        return f"{CHATGPT_URL}c/{value}"
    if CONV_URL_RE.search(value):
        return value if re.match(r"https?://", value, re.I) else f"https://{value}"
    raise UsageError(
        f"--continue expects a chatgpt.com conversation URL (…/c/<id>) or the bare id, got {text!r}"
    )


def conversation_trailer(url: str) -> str:
    """Printed after every reply so the calling agent learns that it can follow up in-thread."""
    return (
        "---\n"
        f"Conversation: {url}\n"
        "To continue this thread with a follow-up (its context is retained), pass "
        f"`--continue {url}` on the next `chatgpt` call. Omit it to start a fresh chat."
    )


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


def count_nodes(page, selectors) -> int:
    if isinstance(selectors, str):
        selectors = [selectors]
    for selector in selectors:
        try:
            nodes = page.query_selector_all(selector)
        except Exception:
            continue
        if nodes:
            return len(nodes)
    return 0


def count_nodes_strict(page, selectors) -> int:
    last_error = None
    for _ in range(3):
        clean_zero = True
        for selector in selectors:
            try:
                nodes = page.query_selector_all(selector)
            except Exception as exc:
                clean_zero = False
                last_error = exc
                continue
            if nodes:
                return len(nodes)
        if clean_zero:
            return 0
        time.sleep(0.3)
    raise RuntimeError(f"could not snapshot message counts: {last_error}")


def message_ids(page) -> set[str]:
    try:
        return set(
            page.eval_on_selector_all(
                "[data-message-id]",
                "els => els.map(e => e.getAttribute('data-message-id')).filter(Boolean)",
            )
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
                page.query_selector("button.__composer-pill")
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
    digest = hashlib.sha256(str(Path.cwd().resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{Path.cwd().name} · {digest}"


def project_cache_path() -> Path:
    return Path(os.environ.get("CHATGPT_STATE_DIR", str(Path.home() / ".chatgpt"))) / "projects.json"


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


def read_pill_label(page) -> str:
    """The closed pill's text is the current selection ('6 Pro', 'High', …). While the menu is
    open it shows a placeholder, so read it only with the menu closed."""
    node = _q(page, [PILL_SELECTOR])
    if node is None:
        return ""
    try:
        return normalize(node.inner_text())
    except Exception:
        return ""


def composer_mode(page) -> str:
    """'chat' | 'work' | 'none' — 'none' when the page has no Chat/Work toggle at all."""
    try:
        radios = page.evaluate(
            """() => [...document.querySelectorAll('[role="radio"][data-tpp-toggle-value]')]
                .map(r => [r.getAttribute('data-tpp-toggle-value'), r.getAttribute('aria-checked') === 'true'])"""
        )
    except Exception:
        radios = []
    if not radios:
        return "none"
    return "chat" if any(value == "chatgpt" and checked for value, checked in radios) else "work"


def ensure_chat_mode(page) -> None:
    """Flip a Work-mode composer back to Chat. Chat and Work keep separate model settings, so
    this never disturbs the Chat selection. A page without the toggle is left alone: the
    picker checks that follow reject a Work-shaped menu anyway."""
    mode = composer_mode(page)
    if mode != "work":
        return
    log("composer is in Work mode; switching to Chat")
    for radio in page.query_selector_all(CHAT_MODE_RADIO_SELECTOR):
        try:
            if not radio.is_visible():
                continue
            try:
                radio.click(timeout=5000)
            except Exception:
                radio.dispatch_event("click")
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


PICKER_STATE_JS = """() => {
  const picker = document.querySelector('[data-testid="composer-intelligence-picker-content"]');
  if (!picker) return null;
  const text = el => (el ? (el.innerText || '') : '').replace(/\\s+/g, ' ').trim();
  const num = el => (el === null || el === undefined) ? null : Number(el);
  const controls = picker.querySelector('[data-explicit-model]');
  const toggle = picker.querySelector('[role="menuitem"][aria-expanded]');
  const effort = toggle && toggle.querySelector('[data-max-effort]');
  const slider = picker.querySelector('[data-model-reasoning-effort-slider] [role="slider"]');
  const sliderItem = picker.querySelector('[role="menuitem"][aria-keyshortcuts]');
  const view = picker.querySelector('[data-view]');
  return {
    view: view ? view.getAttribute('data-view') : null,
    explicit_model: controls ? controls.getAttribute('data-explicit-model') : null,
    label: text(toggle),
    max_effort: effort ? effort.getAttribute('data-max-effort') === 'true' : null,
    value_now: slider ? num(slider.getAttribute('aria-valuenow')) : null,
    value_max: slider ? num(slider.getAttribute('aria-valuemax')) : null,
    slider_disabled: sliderItem ? sliderItem.getAttribute('aria-disabled') === 'true' : null,
    radios: [...picker.querySelectorAll('[role="menuitemradio"]')]
      .map(r => [text(r), r.getAttribute('aria-checked') === 'true']),
  };
}"""


def read_picker_state(page) -> dict | None:
    """One snapshot of the open picker, or None when no picker is on the page."""
    try:
        return page.evaluate(PICKER_STATE_JS)
    except Exception:
        return None


def choose_latest_model(page) -> None:
    """Expand the model list and pick the Latest entry; confirm the picker no longer reports an
    explicit model. The slider item is disabled while the list is expanded, so callers close and
    reopen the menu before touching the slider."""
    toggle = _q(page, [MODEL_TOGGLE_SELECTOR])
    if toggle is None:
        raise ModelVerificationError("model list toggle not found in the picker")
    if toggle.get_attribute("aria-expanded") != "true":
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
        if state is None or state["explicit_model"] != "false":
            raise ModelVerificationError(
                f"clicked {text!r} but the picker still reports an explicit model "
                f"({state and state['label']!r})"
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

    At Pro the closed pill must read exactly REQUIRED_PRO_LABEL. Below Pro the UI shows no
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

    pill = read_pill_label(page)
    if wanted == "pro" and pill == REQUIRED_PRO_LABEL:
        log(f"model verified: pill already {REQUIRED_PRO_LABEL!r} (Latest · Pro)")
        return f"Latest ({REQUIRED_PRO_LABEL})"

    if not open_picker(page):
        raise ModelVerificationError("model menu could not be opened")
    state = read_picker_state(page)
    if state is None:
        raise ModelVerificationError(f"model picker not found after opening the menu; pill read {pill!r}")
    if state["value_max"] != len(EFFORT_LEVELS) - 1:
        raise ModelVerificationError(
            f"unexpected effort slider (max tick {state['value_max']!r}, label {state['label']!r}); "
            "is the composer in Work mode?"
        )
    if state["explicit_model"] != "false":
        log(f"explicit model selected ({state['label']!r}); switching to Latest")
        choose_latest_model(page)
        state = _reopen_picker(page)  # the slider only answers on an untouched menu
    if state["value_now"] != target:
        log(f"effort tick {state['value_now']} → {target} ({wanted})")
        choose_effort_tick(page, target)
    close_menu(page)

    pill = read_pill_label(page)
    if wanted == "pro":
        if pill != REQUIRED_PRO_LABEL:
            raise ModelVerificationError(f"required label {REQUIRED_PRO_LABEL!r}, pill reads {pill!r}")
        log(f"model verified: {REQUIRED_PRO_LABEL} (Latest · Pro)")
        return f"Latest ({REQUIRED_PRO_LABEL})"

    after = _reopen_picker(page)
    close_menu(page)
    if after["explicit_model"] != "false" or after["value_now"] != target:
        raise ModelVerificationError(
            f"final check failed: expected Latest at tick {target} ({wanted}), picker reported "
            f"model={'Latest' if after['explicit_model'] == 'false' else after['label']!r} "
            f"tick={after['value_now']!r}"
        )
    log(f"model verified: Latest; effort {wanted} (tick {target}, pill {pill!r}); Latest shows no model version below Pro")
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
    composer = page.locator(
        "form:has(#prompt-textarea), [role='presentation']:has(#prompt-textarea)"
    ).first
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


def put_text(page, prompt: str) -> None:
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.evaluate(
        """() => { const el = document.querySelector('#prompt-textarea')
            || document.querySelector('div[contenteditable="true"]');
            if (el) { el.scrollIntoView({block: 'center'}); el.focus(); } }"""
    )
    try:
        page.keyboard.insert_text(prompt)
    except Exception:
        page.keyboard.type(prompt)
    time.sleep(0.5)


def composer_text(page) -> str:
    return page.evaluate(
        """() => { const el = document.querySelector('#prompt-textarea')
            || document.querySelector('div[contenteditable="true"]');
            return el ? (el.innerText || el.textContent || '') : ''; }"""
    ) or ""


def composer_has_prompt(page, prompt: str) -> bool:
    wanted = normalize(prompt)
    got = normalize(composer_text(page))
    return bool(wanted) and got == wanted


def clear_composer(page) -> None:
    page.evaluate(
        """() => { const el = document.querySelector('#prompt-textarea')
            || document.querySelector('div[contenteditable="true"]');
            if (el) el.focus(); }"""
    )
    page.keyboard.press("Control+a")
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


def confirm_sent_and_capture(page, base_user: int, bound_url: str | None = None) -> str:
    """Wait for the send to register, then return the conversation URL. In a fresh chat the URL
    flipping to /c/<id> is as good as a new user turn; when following up (bound_url) the page
    already carries that URL, so only the new user turn counts."""
    deadline = time.monotonic() + 45
    sent = False
    while time.monotonic() < deadline:
        if count_nodes(page, USER_MSG_SELECTORS) > base_user:
            sent = True
            break
        if bound_url is None and CONV_URL_RE.search(current_url(page)):
            sent = True
            break
        time.sleep(1)
    if not sent:
        raise RuntimeError("no new user turn appeared after send")
    if bound_url is not None:
        log(f"conversation bound: {bound_url}")
        return bound_url
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        url = current_url(page)
        if CONV_URL_RE.search(url):
            log(f"conversation bound: {url}")
            return url
        time.sleep(1)
    raise SentUnknownLocationError("prompt was sent but the conversation URL was not captured")


def is_streaming(page) -> bool:
    return _q(page, STREAMING_BTN_SELECTORS) is not None


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


def fresh_assistant_node(page, base_ids: set[str], base_assistant: int):
    nodes = _qa(page, ASSISTANT_MSG_SELECTORS)
    fresh = []
    for index, node in enumerate(nodes):
        try:
            message_id = node.get_attribute("data-message-id") or ""
        except Exception:
            continue
        if (message_id and message_id not in base_ids) or (not message_id and index >= base_assistant):
            fresh.append(node)
    return fresh[-1] if fresh else None


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
        markdown = node.query_selector(".markdown")
        return (markdown or node).evaluate(MARKDOWN_SERIALIZER) or ""
    except Exception:
        try:
            return node.inner_text() or ""
        except Exception:
            return ""


# Climb from a message to its own turn container: the first ancestor that holds a copy action.
# Should that ancestor hold other messages too, the copy action belongs to an earlier turn.
TURN_HAS_COPY_JS = """el => {
  const copy = %s;
  let node = el;
  for (let hop = 0; hop < 8 && node; hop++) {
    if (node.querySelector(copy)) return node.querySelectorAll('[data-message-author-role]').length <= 1;
    node = node.parentElement;
  }
  return false;
}""" % json.dumps(", ".join(COPY_BTN_SELECTORS))


def turn_complete(page, node) -> bool:
    """The fresh assistant turn is done once nothing streams and its own turn carries the copy
    action. A page-wide copy count would be satisfied too early: earlier turns keep their
    buttons (a follow-up inherits all of them) and the user's own turn carries one as well."""
    if is_streaming(page):
        return False
    try:
        return bool(node.evaluate(TURN_HAS_COPY_JS))
    except Exception:
        return False


def wait_for_response(
    page,
    conversation_url: str,
    base_ids: set[str],
    base_assistant: int,
    deadline: float,
) -> str:
    key = conversation_key(conversation_url)
    stable_since = None
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
        node = fresh_assistant_node(page, base_ids, base_assistant)
        if node is None or not turn_complete(page, node):
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
            return current
        time.sleep(1)
    raise ResponseTimeoutError("response wait timed out")


def ask(
    prompt: str,
    *,
    effort: str,
    attach: Path | None,
    max_wait: int,
    project: str | None = None,
    conversation: str | None = None,
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

            base_user = count_nodes_strict(page, USER_MSG_SELECTORS)
            base_assistant = count_nodes_strict(page, ASSISTANT_MSG_SELECTORS)
            base_ids = message_ids(page)

            put_text(page, prompt)
            if not composer_has_prompt(page, prompt):
                clear_composer(page)
                put_text(page, prompt)
                if not composer_has_prompt(page, prompt):
                    raise RuntimeError("prompt did not enter the composer intact")
            raise_if_rate_limited(page, "rate limited before submit; nothing sent")
            click_send(page)
            conversation_url = confirm_sent_and_capture(page, base_user, bound_url)
            release_submit_lock()
            deadline = time.monotonic() + max_wait

            for attempt in range(2):
                try:
                    body = wait_for_response(
                        page,
                        conversation_url,
                        base_ids,
                        base_assistant,
                        deadline,
                    )
                    return Reply(body, conversation_url)
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
            raise RuntimeError("unreachable harvest state")
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
        project = None
        if not args.no_project and args.conversation is None:
            project = args.project or default_project_name()
        reply = ask_fn(
            prompt,
            effort=args.effort,
            attach=args.attach,
            max_wait=args.max_wait,
            project=project,
            conversation=args.conversation,
        )
        body = reply.body.strip()
        if not body:
            raise RuntimeError("harvested response is empty")
        atomic_write(output, body + "\n")  # the saved file is the pure response
        print(output if args.quiet else body)
        print()
        print(conversation_trailer(reply.conversation_url))
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
    except KeyboardInterrupt:
        print("chatgpt: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
