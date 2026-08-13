#!/usr/bin/env python3
"""Send one neutral prompt through a logged-in ChatGPT browser and harvest Markdown."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, TextIO

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # Unit tests and --help do not require Playwright.
    sync_playwright = None


REQUIRED_MODEL = "GPT-5.6 Sol"
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
MODEL_SWITCHER_SELECTORS = [
    'button.__composer-pill[aria-haspopup="menu"]',
    'button[data-testid="model-switcher-dropdown-button"]',
    'button[aria-label*="model" i]',
]
MENU_ITEM_SELECTOR = '[role="menuitem"], [role="menuitemradio"], [role="option"]'
EFFORT_ITEM_SELECTORS = ['[role="menuitemradio"]', '[role="menuitem"]', '[role="option"]']
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
MODEL_RE = re.compile(r"GPT|gpt|o\d|Claude|Gemini")
CONV_URL_RE = re.compile(r"/c/[0-9a-f]{8}[0-9a-f-]{4,}", re.I)
STABLE_SECS = 4
STATUS_INTERVAL = 15


class UsageError(Exception):
    pass


class ModelVerificationError(Exception):
    pass


class ResponseTimeoutError(Exception):
    pass


class SentUnknownLocationError(Exception):
    pass


class UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def log(message: str) -> None:
    print(f"chatgpt: {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = UsageParser(prog="chatgpt", description=__doc__)
    parser.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
    parser.add_argument("-f", "--file", type=Path, help="read the prompt from a UTF-8 file")
    parser.add_argument("--effort", default="pro", help="reasoning effort (default: pro)")
    parser.add_argument("--attach", type=Path, help="attach one file")
    parser.add_argument("--max-wait", type=int, default=7200, metavar="SEC")
    parser.add_argument("--out", type=Path, help="response path")
    parser.add_argument("--quiet", action="store_true", help="print only the response path")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if (args.prompt is None) == (args.file is None):
        raise UsageError("provide exactly one prompt source: PROMPT, '-', or -f FILE")
    if args.max_wait <= 0:
        raise UsageError("--max-wait must be greater than zero")
    if not args.effort.strip():
        raise UsageError("--effort must not be empty")
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


def read_model_pills(page) -> list[str]:
    values = []
    for item in page.query_selector_all("button.__composer-pill"):
        try:
            value = normalize(item.inner_text())
            if value:
                values.append(value)
        except Exception:
            continue
    return values


def open_switcher(page) -> bool:
    for selector in MODEL_SWITCHER_SELECTORS:
        try:
            items = page.query_selector_all(selector)
            for item in items:
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


def model_name_from_text(text: str) -> str | None:
    for line in text.splitlines():
        value = normalize(line)
        if value and MODEL_RE.search(value):
            return value[:80]
    return None


def read_menu_state(page) -> dict:
    state = {"model": None, "model_source": None, "models": [], "effort": None, "items": []}
    try:
        items = page.query_selector_all(MENU_ITEM_SELECTOR)
    except Exception:
        return state
    for item in items:
        try:
            text = (item.inner_text() or "").strip()
            role = item.get_attribute("role")
            checked = item.get_attribute("aria-checked") == "true" or item.get_attribute("aria-selected") == "true"
        except Exception:
            continue
        if text:
            state["items"].append(text)
        name = model_name_from_text(text)
        if name:
            if name not in state["models"]:
                state["models"].append(name)
            if checked and state["model"] is None:
                state["model"] = name
                state["model_source"] = "checked"
            continue
        if role == "menuitemradio" and checked and text:
            state["effort"] = normalize(text)
        if item.get_attribute("aria-haspopup") == "menu":
            lines = [normalize(line) for line in text.splitlines() if normalize(line)]
            if len(lines) >= 2:
                state["effort"] = lines[-1]
    if state["model"] is None and len(state["models"]) == 1:
        state["model"] = state["models"][0]
        state["model_source"] = "single"
    return state


def exact_model(model: str | None) -> bool:
    return normalize(model).casefold() == REQUIRED_MODEL.casefold()


def close_menu(page) -> None:
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    time.sleep(0.3)


def collect_effort_items(page):
    result = []
    seen = set()
    for selector in EFFORT_ITEM_SELECTORS:
        try:
            items = page.query_selector_all(selector)
        except Exception:
            continue
        for item in items:
            marker = id(item)
            if marker not in seen:
                result.append(item)
                seen.add(marker)
    return result


def open_effort_submenu(page) -> bool:
    try:
        triggers = page.query_selector_all('[role="menuitem"][aria-haspopup="menu"]')
    except Exception:
        return False
    for trigger in triggers:
        try:
            text = trigger.inner_text() or ""
            if MODEL_RE.search(text):
                continue
            trigger.dispatch_event("click")
            time.sleep(1.2)
            return True
        except Exception:
            continue
    return False


def choose_effort(page, effort: str) -> bool:
    wanted = normalize(effort).casefold()
    candidates = collect_effort_items(page)
    has_radio = any(
        item.get_attribute("role") == "menuitemradio"
        and not MODEL_RE.search(item.inner_text() or "")
        for item in candidates
    )
    if not has_radio:
        open_effort_submenu(page)
        candidates = collect_effort_items(page)
    for exact in (True, False):
        for item in candidates:
            try:
                text = normalize(item.inner_text())
                folded = text.casefold()
                matches = folded == wanted if exact else wanted in folded
                if not text or MODEL_RE.search(text) or not matches:
                    continue
                try:
                    item.click(timeout=5000)
                except Exception:
                    item.dispatch_event("click")
                time.sleep(1.5)
                return True
            except Exception:
                continue
    return False


def select_model(page, effort: str) -> str:
    """Select effort and fail closed unless the menu says exactly GPT-5.6 Sol."""
    try:
        page.wait_for_selector("button.__composer-pill", timeout=20000)
    except Exception:
        pass
    wanted = normalize(effort)
    pills = read_model_pills(page)
    effort_ready = any(value.casefold() == wanted.casefold() for value in pills)

    if not open_switcher(page):
        raise ModelVerificationError("model menu could not be opened")
    state = read_menu_state(page)
    if not exact_model(state["model"]):
        close_menu(page)
        raise ModelVerificationError(
            f"required model {REQUIRED_MODEL!r}, menu reported {state['model']!r}"
        )

    if effort_ready:
        close_menu(page)
        log(f"model verified: {REQUIRED_MODEL}; effort pill already {wanted}")
        return f"{REQUIRED_MODEL} ({wanted})"

    if not choose_effort(page, wanted):
        close_menu(page)
        raise ModelVerificationError(f"reasoning effort {wanted!r} could not be selected")
    close_menu(page)

    pills = read_model_pills(page)
    if not any(value.casefold() == wanted.casefold() for value in pills):
        raise ModelVerificationError(
            f"reasoning effort verification failed: expected {wanted!r}, pills={pills!r}"
        )
    if not open_switcher(page):
        raise ModelVerificationError("model menu could not be reopened for final verification")
    after = read_menu_state(page)
    close_menu(page)
    if not exact_model(after["model"]):
        raise ModelVerificationError(
            f"required model {REQUIRED_MODEL!r}, final menu reported {after['model']!r}"
        )
    log(f"model verified: {REQUIRED_MODEL}; effort selected: {wanted}")
    return f"{REQUIRED_MODEL} ({wanted})"


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


def confirm_sent_and_capture(page, base_user: int) -> str:
    deadline = time.monotonic() + 45
    sent = False
    while time.monotonic() < deadline:
        url = current_url(page)
        if count_nodes(page, USER_MSG_SELECTORS) > base_user or CONV_URL_RE.search(url):
            sent = True
            break
        time.sleep(1)
    if not sent:
        raise RuntimeError("no new user turn appeared after send")
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


def turn_complete(page, base_assistant: int) -> bool:
    if is_streaming(page):
        return False
    return (
        count_nodes(page, ASSISTANT_MSG_SELECTORS) > base_assistant
        and count_nodes(page, COPY_BTN_SELECTORS) > 0
    )


def wait_for_response(
    page,
    conversation_url: str,
    base_ids: set[str],
    base_assistant: int,
    deadline: float,
) -> str:
    match = CONV_URL_RE.search(conversation_url)
    conversation_key = match.group(0) if match else ""
    stable_since = None
    previous = ""
    last_status = -STATUS_INTERVAL
    log(f"waiting for response (up to {max(0, int(deadline - time.monotonic()))}s)")
    while time.monotonic() < deadline:
        if conversation_key not in current_url(page):
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
        if node is None or not turn_complete(page, base_assistant):
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


def ask(prompt: str, *, effort: str, attach: Path | None, max_wait: int) -> str:
    if sync_playwright is None:
        raise RuntimeError(
            "Python package 'playwright' is required "
            "(python3 -m pip install playwright; no browser download needed — "
            "this only attaches to an already-running Chrome over CDP)"
        )
    port = int(os.environ.get("CHATGPT_CDP_PORT", "9222"))
    if not port_open(port) or not cdp_browser_ok(port):
        raise RuntimeError(f"CDP {port} is not a supported Chromium browser")

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
            click_send(page)
            conversation_url = confirm_sent_and_capture(page, base_user)
            release_submit_lock()
            deadline = time.monotonic() + max_wait

            for attempt in range(2):
                try:
                    return wait_for_response(
                        page,
                        conversation_url,
                        base_ids,
                        base_assistant,
                        deadline,
                    )
                except ResponseTimeoutError:
                    raise
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
    ask_fn: Callable[..., str] = ask,
) -> int:
    try:
        args = parse_args(argv)
        prompt = read_prompt(args, stdin or sys.stdin)
        if args.attach is not None and not args.attach.expanduser().is_file():
            raise UsageError(f"attachment is not a file: {args.attach}")
        output = response_path(args.out)
        response = ask_fn(
            prompt,
            effort=args.effort,
            attach=args.attach,
            max_wait=args.max_wait,
        ).strip()
        if not response:
            raise RuntimeError("harvested response is empty")
        atomic_write(output, response + "\n")
        if args.quiet:
            print(output)
        else:
            print(response)
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
    except KeyboardInterrupt:
        print("chatgpt: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"chatgpt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
