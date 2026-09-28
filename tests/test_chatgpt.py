from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import ask_core  # noqa: E402

CONV = "https://chatgpt.com/g/g-p-6a9b861a83d48191ba6b3bd85b197802/c/6a9b8621-0250-83ee-93ed-50f50ee5d7bd"
TRAILER = ask_core.thread_trailer(CONV)


def answer(*_args, **_kwargs):
    return ask_core.Reply("answer", CONV)


class ParsingTests(unittest.TestCase):
    def test_positional_defaults(self):
        args = ask_core.parse_args(["hello"])
        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.effort, "pro")
        self.assertEqual(args.max_wait, 7200)
        self.assertIsNone(args.file)

    def test_file_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "prompt.md"
            prompt_file.write_text("from file", encoding="utf-8")
            args = ask_core.parse_args(["-f", str(prompt_file)])
            self.assertEqual(ask_core.read_prompt(args, io.StringIO("ignored")), "from file")

    def test_stdin_prompt(self):
        args = ask_core.parse_args(["-"])
        self.assertEqual(ask_core.read_prompt(args, io.StringIO("from stdin\n")), "from stdin\n")

    def test_rejects_ambiguous_sources(self):
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["text", "-f", "prompt.md"])
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args([])

    def test_rejects_nonpositive_wait(self):
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["--max-wait", "0", "text"])


class ExitCodeTests(unittest.TestCase):
    def run_main(self, argv, ask_fn):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ask_core.main(argv, stdin=io.StringIO(), ask_fn=ask_fn)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_model_mismatch_is_exit_2_and_writes_nothing(self):
        def mismatch(*_args, **_kwargs):
            raise ask_core.ModelVerificationError("mock mismatch")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            code, stdout, stderr = self.run_main(["--out", str(output), "hello"], mismatch)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("prompt not sent", stderr)
            self.assertFalse(output.exists())

    def test_timeout_is_exit_3(self):
        def timeout(*_args, **_kwargs):
            raise ask_core.ResponseTimeoutError("mock timeout")

        code, stdout, stderr = self.run_main(["hello"], timeout)
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("mock timeout", stderr)

    def test_success_prints_body_and_saves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            code, stdout, stderr = self.run_main(["--out", str(output), "hello"], answer)
            self.assertEqual(code, 0)
            self.assertEqual(stdout, f"answer\n\n{TRAILER}\n")
            self.assertEqual(stderr, "")
            self.assertEqual(output.read_text(encoding="utf-8"), "answer\n")

    def test_quiet_prints_path_then_trailer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            code, stdout, _ = self.run_main(["--quiet", "--out", str(output), "hello"], answer)
            self.assertEqual(code, 0)
            self.assertEqual(stdout, f"{output.resolve()}\n\n{TRAILER}\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "answer\n")


class ComposerTests(unittest.TestCase):
    def test_composer_requires_exact_prompt(self):
        page = mock.Mock()
        page.evaluate.return_value = "wanted prompt"
        self.assertTrue(ask_core.composer_has_prompt(page, "wanted prompt"))

        page.evaluate.return_value = "stale draft wanted prompt"
        self.assertFalse(ask_core.composer_has_prompt(page, "wanted prompt"))


class ResponseCompletionTests(unittest.TestCase):
    def test_exchange_is_complete_only_when_its_own_bar_shows_the_copy_action(self):
        """Older exchanges (and the user's own bar) keep copy buttons, so completion is decided
        on this run's exchange wrapper, never on a page-wide count."""
        exchange = mock.Mock()
        with mock.patch.object(ask_core, "is_streaming", return_value=False):
            exchange.evaluate.return_value = True
            self.assertTrue(ask_core.exchange_complete(mock.Mock(), exchange))
            exchange.evaluate.return_value = False
            self.assertFalse(ask_core.exchange_complete(mock.Mock(), exchange))
            self.assertEqual(exchange.evaluate.call_args[0][0], ask_core.EXCHANGE_DONE_JS)
        with mock.patch.object(ask_core, "is_streaming", return_value=True):
            exchange.evaluate.return_value = True
            self.assertFalse(ask_core.exchange_complete(mock.Mock(), exchange))


class EffortArgTests(unittest.TestCase):
    def test_effort_names_normalize_to_slider_levels(self):
        self.assertEqual(ask_core.parse_args(["--effort", "Extra-High", "hi"]).effort, "extra high")
        self.assertEqual(ask_core.parse_args(["--effort", "PRO", "hi"]).effort, "pro")
        self.assertEqual(ask_core.effort_index("extra_high"), 3)

    def test_unknown_effort_is_a_usage_error(self):
        for bad in ("ultra", "", "  ", "5"):
            with self.assertRaises(ask_core.UsageError):
                ask_core.parse_args(["--effort", bad, "hi"])


PROMPT = "GUARD-B2: reply with exactly the word PONG-B2 and nothing else."
OTHER_PROMPT = "GUARD-A2: Write a numbered list from 1 to 1500; on each line the English word."


def conversation(*rows, async_status=None, current=None):
    """The server's record as SERVER_CONVERSATION_JS returns it. Rows are (id, parent, role,
    exchange, overrides); the leaf is the last row unless `current` names another."""
    messages = {"root": {"parent": None, "role": None, "status": None, "exchange": None, "end_turn": False, "text": ""}}
    for message_id, parent, role, exchange, *overrides in rows:
        messages[message_id] = {
            "parent": parent, "role": role, "status": "finished_successfully", "exchange": exchange,
            "end_turn": False, "text": "", **(overrides[0] if overrides else {}),
        }
    return {"current_node": current or rows[-1][0], "async_status": async_status, "messages": messages}


@contextlib.contextmanager
def server_clock():
    """time.sleep advances time.monotonic; yields the sleeps and the log mock."""
    now, sleeps = [0.0], []

    def sleep(secs):
        sleeps.append(secs)
        now[0] += secs

    with mock.patch.object(ask_core.time, "sleep", side_effect=sleep), mock.patch.object(
        ask_core.time, "monotonic", side_effect=lambda: now[0]
    ), mock.patch.object(ask_core, "log") as log:
        yield sleeps, log


BUSY = {"error": "HTTP 429", "status": 429, "retry_after": None}


def window(server, first, complete=False):
    """The light endpoint's view of `server`: the current branch from message `first` on, listed
    in branch order (parent = the message listed before), no other branch."""
    path, node = [], server["current_node"]
    while node:
        path.append(node)
        node = server["messages"][node]["parent"]
    path = path[::-1][path[::-1].index(first):]
    messages, previous = {}, None
    for message_id in path:
        messages[message_id] = {**server["messages"][message_id], "parent": previous}
        previous = message_id
    return {**server, "complete": complete, "messages": messages}


FINAL = {"end_turn": True}
# Synthetic trees shaped like the records measured 2026-09-28 (see ask_core's server section).
EARLIER = [
    ("e-turn", "root", "user", "x-e", {"text": "earlier question"}),
    ("e-final", "e-turn", "assistant", "x-e", FINAL),
]
PLAIN = EARLIER + [
    ("b2-turn", "e-final", "user", "x-b2", {"text": PROMPT}),
    ("b2-thoughts", "b2-turn", "assistant", "x-b2"),
    ("b2-recap", "b2-thoughts", "assistant", "x-b2"),
    ("b2-answer", "b2-recap", "assistant", "x-b2", FINAL),
]
FINISHED = PLAIN
RUNNING = EARLIER + [  # async_status 3: the leaf is the turn, then a 'thoughts' message
    ("r-turn", "e-final", "user", "x-r", {"text": "next"}),
    ("r-thoughts", "r-turn", "assistant", "x-r"),
]
STOPPED = EARLIER + [  # stopped by hand: async_status null, the recap ends the turn
    ("s-turn", "e-final", "user", "x-s", {"text": "next"}),
    ("s-thoughts", "s-turn", "assistant", "x-s"),
    ("s-commentary", "s-thoughts", "assistant", "x-s"),
    ("s-recap", "s-commentary", "assistant", "x-s", FINAL),
]
AGENTIC = EARLIER + [
    ("a-turn", "e-final", "user", "x-a", {"text": "Round 3: run the agents"}),
    ("a-thoughts", "a-turn", "assistant", "x-a"),
    ("a-exec", "a-thoughts", "assistant", "x-a"),  # recipient container.exec, code
    ("a-output", "a-exec", "tool", "x-a"),  # execution_output
    ("a-plan", "a-output", "assistant", "x-a"),  # turn_plan.update_turn_plan, commentary
    ("a-call", "a-plan", "assistant", "x-a"),  # api_tool.call_tool
    ("a-tool", "a-call", "tool", "x-a"),  # code, commentary
    ("a-sub", "a-tool", "assistant", "x-a"),  # SubAgentActivityThreadItem.completed, hidden
    ("a-note", "a-sub", "assistant", "x-a"),  # text, commentary
    ("a-recap", "a-note", "assistant", "x-a"),  # reasoning_recap
    ("a-final", "a-recap", "assistant", "x-a", FINAL),  # text, final
]
JOINED = EARLIER + [
    ("j-turn", "e-final", "user", "x-j", {"text": "Round 2"}),
    ("j-thoughts", "j-turn", "assistant", "x-j"),
    ("j-recap", "j-thoughts", "assistant", "x-j"),
    ("j-resend", "j-recap", "user", "x-j", {"text": "Round 2 again"}),
    ("j-exec", "j-resend", "assistant", "x-j"),
    ("j-output", "j-exec", "tool", "x-j"),
    ("j-final", "j-output", "assistant", "x-j", FINAL),
]


class FakeUnit:
    def __init__(self, message_id, text="earlier question"):
        self.message_id, self.text = message_id, text

    def get_attribute(self, name):
        return self.message_id if name == ask_core.MESSAGE_IDS_ATTR else None

    def inner_text(self):
        return self.text


class FakeThread:
    """The 2026-09-28 conversation page as confirm_sent_and_capture sees it: user units keyed
    '…:user' with ids in data-chatgpt-search-message-ids. The page may keep only the newest
    `window` turns mounted, so a new turn can push the oldest out. `send_after` polls after the
    click the sent turn appears on the live page (None: never); `on_reload` makes it appear once
    the conversation is reloaded."""

    def __init__(self, turns, *, send_after, window=None, on_reload=False, with_ids=True, also=()):
        self.with_ids, self.window = with_ids, window
        self.users = [self.unit(f"old-{n}") for n in range(turns)]
        self.send_after, self.on_reload, self.polls = send_after, on_reload, 0
        self.also = list(also)  # (id, text) of other runs' turns that land with this one
        self.goto = mock.Mock(side_effect=self.reload)

    def unit(self, message_id, text="earlier question"):
        return FakeUnit(message_id if self.with_ids else None, text)

    def ids(self):
        return {unit.message_id for unit in self.users if unit.message_id}

    def land(self):
        self.users.append(self.unit("sent", PROMPT))
        self.users.extend(self.unit(message_id, text) for message_id, text in self.also)
        if self.window:
            self.users = self.users[-self.window :]

    def tick(self):
        self.polls += 1
        if self.polls == self.send_after:
            self.land()

    def reload(self, *_args, **_kwargs):
        if self.on_reload:
            self.land()

    def query_selector_all(self, selector):
        return list(self.users) if selector == ask_core.USER_MSG_SELECTORS[0] else []


class FakeNode:
    TICK_X0 = 100
    TICK_STEP = 50

    def __init__(self, page, kind, index=0, text=""):
        self.page, self.kind, self.index, self.text = page, kind, index, text

    def inner_text(self):
        if self.kind == "pill":  # the closed pill no longer carries the model version
            return "추론 수준" if self.page.menu_open else self.page.label().split()[-1]
        return self.text

    def get_attribute(self, _name):
        return None

    def is_visible(self):
        return True

    def click(self, timeout=None):
        self.page.act(self)

    def dispatch_event(self, _name):
        self.page.act(self)

    def bounding_box(self):
        return {"x": self.TICK_X0 + self.TICK_STEP * self.index, "y": 10, "width": 10, "height": 10}


class FakePicker:
    """Just enough of the 2026-09-26 Chat-mode composer to drive select_model: a Chat/Work switch,
    a pill that opens the menu, and a menu whose model row (the view toggle) names the selection,
    with the model list behind that toggle (view 'advanced') and a tick slider that ignores clicks
    while the list is showing (measured)."""

    LABELS = ["Instant", "중간", "High", "매우 높음", "Pro"]

    def __init__(
        self, *, mode="chat", explicit=False, value=4, value_max=4, radios=None, picker=True,
        pro_label="6 Pro",
    ):
        self.pro_label = pro_label
        self.mode = mode
        self.explicit = explicit
        self.value = value
        self.value_max = value_max
        self.radio_names = radios or ["최신", "GPT-5.6 Sol", "GPT-5.5"]
        self.picker = picker
        self.menu_open = False
        self.expanded = False
        self.menu_opens = 0
        self.tick_clicks = []
        self.mouse = mock.Mock()
        self.mouse.click.side_effect = self._mouse_click
        self.keyboard = mock.Mock()
        self.keyboard.press.side_effect = self._press

    def label(self):
        if self.value == self.value_max:
            return "5.6 Pro" if self.explicit else self.pro_label
        return self.LABELS[self.value]

    def wait_for_selector(self, *_args, **_kwargs):
        return None

    def evaluate(self, js, *_args):
        if js == ask_core.MODE_STATE_JS:
            if self.mode is None:
                return []
            return [["Chat", self.mode == "chat"], ["Work", self.mode == "work"]]
        if js == ask_core.PICKER_STATE_JS:
            if not (self.menu_open and self.picker):
                return None
            checked = "GPT-5.6 Sol" if self.explicit else "최신"
            return {
                "view": "advanced" if self.expanded else "simple",
                "model": checked if checked in self.radio_names else None,
                "label": self.label(),
                "max_effort": self.value == self.value_max,
                "value_now": self.value,
                "value_max": self.value_max,
                "slider_disabled": self.expanded,
                "radios": [[name, name == checked] for name in self.radio_names],
            }
        raise AssertionError(f"unexpected evaluate: {js[:60]}")

    def query_selector(self, selector):
        nodes = self.query_selector_all(selector)
        return nodes[0] if nodes else None

    def query_selector_all(self, selector):
        if selector == ask_core.PILL_SELECTOR:
            return [FakeNode(self, "pill")]
        if selector == ask_core.MODE_BUTTON_SELECTOR:
            if not self.mode:
                return []
            return [FakeNode(self, "work", text="Work"), FakeNode(self, "chat", text="Chat")]
        if not self.menu_open:
            return []
        if selector == ask_core.MODEL_TOGGLE_SELECTOR:
            return [FakeNode(self, "toggle")]
        if selector == ask_core.MODEL_RADIO_SELECTOR:
            return [FakeNode(self, "radio", i, name) for i, name in enumerate(self.radio_names)]
        if selector == ask_core.EFFORT_TICK_SELECTOR:
            return [FakeNode(self, "tick", i) for i in range(self.value_max + 1)]
        return []

    def _press(self, key):
        if key == "Escape":
            self.menu_open = False
            self.expanded = False

    def _mouse_click(self, x, _y):
        index = int((x - FakeNode.TICK_X0) // FakeNode.TICK_STEP)
        self.tick_clicks.append((index, self.expanded))
        if self.menu_open and not self.expanded:
            self.value = index

    def act(self, node):
        if node.kind == "pill":
            self.menu_open = True
            self.expanded = False
            self.menu_opens += 1
        elif node.kind in ("chat", "work"):
            self.mode = node.kind
        elif node.kind == "toggle":
            self.expanded = True
        elif node.kind == "radio":
            self.explicit = not ask_core.LATEST_MODEL_RE.match(node.text)
            self.expanded = False


class ModelPickerTests(unittest.TestCase):
    def select(self, page, effort="pro"):
        stderr = io.StringIO()
        with mock.patch.object(ask_core.time, "sleep"), contextlib.redirect_stderr(stderr):
            result = ask_core.select_model(page, effort)
        return result, stderr.getvalue()

    def test_pro_already_selected_is_verified_in_one_menu_open(self):
        """The closed pill reads only 'Pro' now, so even an untouched run opens the menu once and
        reads the model row there; nothing is clicked."""
        page = FakePicker()
        self.assertEqual(FakeNode(page, "pill").inner_text(), "Pro")
        result, log = self.select(page)
        self.assertEqual(result, "Latest (6 Pro)")
        self.assertEqual(page.menu_opens, 1)
        self.assertEqual(page.tick_clicks, [])
        self.assertFalse(page.menu_open)
        self.assertIn("model verified: 6 Pro", log)

    def test_latest_pro_under_another_label_fails_closed(self):
        """Latest at the top tick that no longer resolves to 6 Pro must not pass as Pro."""
        page = FakePicker(pro_label="7 Pro")
        with self.assertRaises(ask_core.ModelVerificationError) as caught:
            self.select(page)
        self.assertIn("'7 Pro'", str(caught.exception))
        self.assertFalse(page.menu_open)

    def test_explicit_model_is_switched_to_latest_before_the_tick_moves(self):
        page = FakePicker(explicit=True, value=2)
        result, _ = self.select(page)
        self.assertEqual(result, "Latest (6 Pro)")
        self.assertFalse(page.explicit)
        self.assertEqual(page.value, 4)
        # exactly one tick click, on a reopened menu whose model list is collapsed
        self.assertEqual(page.tick_clicks, [(4, False)])
        self.assertFalse(page.menu_open)

    def test_work_mode_is_flipped_to_chat_first(self):
        page = FakePicker(mode="work", value=3)
        result, log = self.select(page)
        self.assertEqual(page.mode, "chat")
        self.assertEqual(result, "Latest (6 Pro)")
        self.assertIn("Work mode", log)

    def test_lower_effort_verifies_latest_and_tick(self):
        page = FakePicker()
        result, _ = self.select(page, "high")
        self.assertEqual(result, "Latest (high)")
        self.assertEqual(page.value, 2)
        self.assertFalse(page.menu_open)

    def test_back_to_pro_from_a_lower_tick(self):
        page = FakePicker(value=2)
        result, _ = self.select(page)
        self.assertEqual(result, "Latest (6 Pro)")
        self.assertEqual(page.tick_clicks, [(4, False)])

    def test_changed_pro_label_fails_closed(self):
        page = FakePicker()
        with mock.patch.object(ask_core, "REQUIRED_PRO_LABEL", "7 Pro"):
            with self.assertRaises(ask_core.ModelVerificationError) as caught:
                self.select(page)
        self.assertIn("'6 Pro'", str(caught.exception))
        self.assertFalse(page.menu_open)

    def test_work_shaped_picker_is_refused(self):
        page = FakePicker(mode=None, value=3, value_max=5)
        with self.assertRaises(ask_core.ModelVerificationError) as caught:
            self.select(page)
        self.assertIn("Work mode", str(caught.exception))
        self.assertEqual(page.tick_clicks, [])

    def test_missing_picker_fails_closed(self):
        page = FakePicker(value=3, picker=False)
        with self.assertRaises(ask_core.ModelVerificationError):
            self.select(page)

    def test_model_list_without_a_checked_entry_fails_closed(self):
        """The model is read from the list's checked entry (the old explicit-model flag stays
        'false' with an explicit model since 2026-09); no checked entry means no verdict."""
        page = FakePicker(radios=["GPT-5.6 Sol", "GPT-5.5"])
        with self.assertRaises(ask_core.ModelVerificationError) as caught:
            self.select(page)
        self.assertIn("cannot tell the selected model", str(caught.exception))
        self.assertEqual(page.tick_clicks, [])
        self.assertFalse(page.menu_open)

    def test_menu_without_latest_entry_fails_closed(self):
        page = FakePicker(explicit=True, radios=["GPT-5.6 Sol", "GPT-5.5"])
        with self.assertRaises(ask_core.ModelVerificationError) as caught:
            self.select(page)
        self.assertIn("GPT-5.5", str(caught.exception))
        self.assertEqual(page.tick_clicks, [])


@contextlib.contextmanager
def thread_state():
    """A private ledger (CHATGPT_STATE_DIR) and a private cwd, so no test touches ~/.chatgpt."""
    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "state"
        here = Path(directory) / "here"
        here.mkdir()
        old = os.getcwd()
        os.chdir(here)
        try:
            with mock.patch.dict(os.environ, {"CHATGPT_STATE_DIR": str(state)}):
                yield state / "threads.json"
        finally:
            os.chdir(old)


def thread_entry(cid, cwd, last_used, started="2026-09-05T14:02:31+09:00", prompt="first question"):
    return {
        "id": cid,
        "url": f"https://chatgpt.com/c/{cid}",
        "cwd": cwd,
        "started": started,
        "last_used": last_used,
        "prompt": prompt,
    }


def write_ledger(path, entries):
    ask_core.atomic_write(path, json.dumps(entries))


class FollowUpArgTests(unittest.TestCase):
    def test_continue_is_a_flag_that_leaves_the_prompt_alone(self):
        args = ask_core.parse_args(["--continue", "hello"])
        self.assertEqual(args.prompt, "hello")
        self.assertTrue(args.continue_last)
        self.assertIsNone(args.resume)
        plain = ask_core.parse_args(["hello"])
        self.assertFalse(plain.continue_last)
        self.assertIsNone(plain.resume)

    def test_resume_takes_a_value(self):
        args = ask_core.parse_args(["--resume", "6a9b8621", "hello"])
        self.assertEqual(args.resume, "6a9b8621")
        self.assertEqual(args.prompt, "hello")

    def test_follow_up_flags_exclude_each_other_and_the_project_flags(self):
        for argv in (
            ["--continue", "--resume", "6a9b8621", "hi"],
            ["--continue", "--project", "Docs", "hi"],
            ["--continue", "--no-project", "hi"],
            ["--resume", "6a9b8621", "--project", "Docs", "hi"],
            ["--resume", "6a9b8621", "--no-project", "hi"],
        ):
            with self.assertRaises(ask_core.UsageError, msg=argv):
                ask_core.parse_args(argv)


class ThreadLedgerTests(unittest.TestCase):
    ID = "6a9b8621-0250-83ee-93ed-50f50ee5d7bd"
    OTHER = "0c4b4501-1111-4222-8333-444455556666"

    def resolve(self, *argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            url = ask_core.resolve_thread(ask_core.parse_args([*argv, "hi"]))
        return url, stderr.getvalue()

    def test_new_thread_is_appended_and_a_follow_up_only_refreshes_last_used(self):
        with thread_state() as ledger:
            self.assertEqual(ask_core.record_thread(CONV, "  first\n question  "), "6a9b8621")
            [first] = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(first["id"], self.ID)
            self.assertEqual(first["url"], CONV)
            self.assertEqual(first["cwd"], str(Path.cwd().resolve()))
            self.assertEqual(first["prompt"], "first question")
            self.assertEqual(first["started"], first["last_used"])

            first["started"] = first["last_used"] = "2020-01-01T00:00:00+00:00"
            write_ledger(ledger, [first])
            ask_core.record_thread(CONV, "a follow-up")
            [again] = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(again["started"], "2020-01-01T00:00:00+00:00")
            self.assertEqual(again["prompt"], "first question")
            self.assertNotEqual(again["last_used"], "2020-01-01T00:00:00+00:00")

            ask_core.record_thread(f"https://chatgpt.com/c/{self.OTHER}", "another topic")
            self.assertEqual([e["id"] for e in json.loads(ledger.read_text(encoding="utf-8"))], [self.ID, self.OTHER])

    def test_ledger_keeps_only_the_newest_entries(self):
        with thread_state() as ledger, mock.patch.object(ask_core, "THREAD_LEDGER_CAP", 2):
            for prefix in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
                ask_core.record_thread(f"https://chatgpt.com/c/{prefix}-0000-4000-8000-000000000000", "q")
            ids = [e["id"][:8] for e in json.loads(ledger.read_text(encoding="utf-8"))]
            self.assertEqual(ids, ["bbbbbbbb", "cccccccc"])

    def test_missing_corrupt_or_odd_ledger_reads_as_empty(self):
        with thread_state() as ledger:
            self.assertEqual(ask_core.load_threads(ledger), [])
            ledger.parent.mkdir(parents=True)
            for junk in ("{not json", '{"id": "an object, not a list"}', '[{"url": "no id"}, 3, "x"]'):
                ledger.write_text(junk, encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(ask_core.load_threads(ledger), [], junk)

    def test_continue_picks_the_most_recent_thread_from_this_folder_only(self):
        with thread_state() as ledger:
            with self.assertRaises(ask_core.UsageError) as caught:
                self.resolve("--continue")
            self.assertIn("no thread has been started from this folder", str(caught.exception))
            here = str(Path.cwd().resolve())
            write_ledger(
                ledger,
                [
                    thread_entry(self.ID, here, "2026-09-05T15:00:00+09:00", prompt="the one to continue"),
                    thread_entry("11111111-2222-4333-8444-555555555555", here, "2026-09-05T14:00:00+09:00"),
                    thread_entry(self.OTHER, "/somewhere/else", "2026-09-07T09:00:00+09:00"),
                ],
            )
            url, stderr = self.resolve("--continue")
            self.assertEqual(url, f"https://chatgpt.com/c/{self.ID}")
            self.assertIn(
                'continuing thread 6a9b8621 · started 2026-09-05 14:02 · "the one to continue"', stderr
            )

    def test_resume_matches_exactly_one_thread_by_prefix_from_any_folder(self):
        with thread_state() as ledger:
            write_ledger(
                ledger,
                [
                    thread_entry(self.ID, "/somewhere/else", "2026-09-05T15:00:00+09:00"),
                    thread_entry(self.OTHER, str(Path.cwd().resolve()), "2026-09-07T09:00:00+09:00"),
                ],
            )
            for handle in ("6a9b8621", "6A9B8621", self.ID[:13], self.ID):
                url, stderr = self.resolve("--resume", handle)
                self.assertEqual(url, f"https://chatgpt.com/c/{self.ID}", handle)
                self.assertIn("continuing thread 6a9b8621", stderr)

    def test_resume_rejects_unknown_ambiguous_project_hash_and_garbage(self):
        with thread_state() as ledger:
            here = str(Path.cwd().resolve())
            twin = "6a9b8621-ffff-4fff-8fff-ffffffffffff"
            write_ledger(ledger, [thread_entry(self.ID, here, "1"), thread_entry(twin, here, "2")])
            cases = {
                "6a9b8621": "ambiguous",
                "deadbeef": "previous reply",
                ask_core.folder_hash(): "project hash",
                "not-a-thread": "expects a thread handle",
                "6a9b862": "expects a thread handle",  # seven characters is not a handle
                "": "expects a thread handle",
            }
            for target, message in cases.items():
                with self.assertRaises(ask_core.UsageError, msg=target) as caught:
                    self.resolve("--resume", target)
                self.assertIn(message, str(caught.exception), target)

    def test_resume_by_url_needs_no_ledger(self):
        with thread_state():
            url, stderr = self.resolve("--resume", CONV)
            self.assertEqual(url, CONV)
            self.assertIn("resuming by URL", stderr)
            url, _ = self.resolve("--resume", f"chatgpt.com/c/{self.ID}/")
            self.assertEqual(url, f"https://chatgpt.com/c/{self.ID}")

    def test_main_hands_the_resolved_thread_to_ask_fn_without_a_project(self):
        seen = {}

        def capture(_prompt, **kwargs):
            seen.update(kwargs)
            return ask_core.Reply("answer", CONV)

        with thread_state() as ledger:
            out = str(ledger.parent / "a.md")
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = ask_core.main(["--out", out, "--continue", "hello"], stdin=io.StringIO(), ask_fn=capture)
                self.assertEqual(code, 64)
                self.assertEqual(seen, {})
                write_ledger(ledger, [thread_entry(self.ID, str(Path.cwd().resolve()), "1")])
                code = ask_core.main(["--out", out, "--continue", "hello"], stdin=io.StringIO(), ask_fn=capture)
                self.assertEqual(code, 0)
                self.assertEqual(seen["conversation"], f"https://chatgpt.com/c/{self.ID}")
                self.assertIsNone(seen["project"])
                seen.clear()
                code = ask_core.main(["--out", out, "--resume", "6a9b8621", "hello"], stdin=io.StringIO(), ask_fn=capture)
                self.assertEqual(code, 0)
                self.assertEqual(seen["conversation"], f"https://chatgpt.com/c/{self.ID}")
            self.assertIn("no thread has been started from this folder", stderr.getvalue())


class ContinuationFlowTests(unittest.TestCase):
    def test_trailer_names_the_thread_and_both_options(self):
        self.assertEqual(
            TRAILER.splitlines(),
            [
                "---",
                f"Thread 6a9b8621 · {CONV}",
                "Follow-up on this topic? Add `--continue` to the next `chatgpt` call to keep going in this "
                "thread (the most recent one from this folder), or `--resume 6a9b8621` to pick it explicitly. "
                "Omit both to start a fresh chat.",
            ],
        )

    def confirm(self, thread, bound_url=CONV):
        """confirm_sent_and_capture against a FakeThread, on a clock that steps 10 s per poll."""
        import itertools

        base_ids = thread.ids()
        clock = itertools.count(0, 10)
        with mock.patch.object(ask_core, "current_url", return_value=CONV), mock.patch.object(
            ask_core.time, "sleep", side_effect=lambda _s: thread.tick()
        ), mock.patch.object(ask_core.time, "monotonic", side_effect=lambda: next(clock)), mock.patch.object(
            ask_core, "log"
        ):
            return ask_core.confirm_sent_and_capture(thread, base_ids, PROMPT, bound_url)

    def test_follow_up_is_confirmed_by_a_new_turn_id_while_the_count_stays_flat(self):
        """The 2026-09-27 regression: the prompt landed, but the long thread's page never showed
        more user turns than before, so a count-based check exited 1 after a successful send."""
        thread = FakeThread(5, send_after=2, window=5)
        self.assertEqual(self.confirm(thread), (CONV, "sent"))
        self.assertEqual(len(thread.users), 5)
        thread.goto.assert_not_called()

    def test_follow_up_reloads_once_when_the_live_page_never_shows_the_turn(self):
        thread = FakeThread(5, send_after=None, on_reload=True)
        self.assertEqual(self.confirm(thread), (CONV, "sent"))
        thread.goto.assert_called_once()
        self.assertEqual(thread.goto.call_args[0][0], CONV)

    def test_follow_up_never_takes_the_bound_url_as_proof_of_sending(self):
        """In a fresh chat the URL flipping to /c/<id> proves the send; a follow-up page carries
        that URL from the start, so the same shortcut would report a send that never happened —
        and a reload that still shows no new turn fails too."""
        thread = FakeThread(1, send_after=None, on_reload=False)
        with self.assertRaises(RuntimeError) as caught:
            self.confirm(thread)
        self.assertIn("no new user turn", str(caught.exception))
        thread.goto.assert_called_once()
        # A fresh chat whose turn the live page misses is looked for once on its /c/<id> page.
        thread = FakeThread(0, send_after=None, on_reload=True)
        self.assertEqual(self.confirm(thread, bound_url=None), (CONV, "sent"))
        self.assertEqual(thread.goto.call_args[0][0], CONV)

    def test_another_runs_new_turn_is_never_taken_for_this_runs_turn(self):
        """The 2026-09-28 race: a reload shows the other run's turn as new too (its id was not on
        this page before the send). Only the turn carrying this run's prompt is this run's turn."""
        thread = FakeThread(3, send_after=None, on_reload=True, also=[("a2-turn", OTHER_PROMPT)])
        self.assertEqual(self.confirm(thread), (CONV, "sent"))
        alone = FakeThread(3, send_after=1, also=[("a2-turn", OTHER_PROMPT)])
        alone.land = lambda: alone.users.append(alone.unit("a2-turn", OTHER_PROMPT))
        with self.assertRaises(RuntimeError):
            self.confirm(alone)

    def test_turns_without_ids_are_never_taken_for_this_runs_turn(self):
        """Without an id nothing ties the answer to this run's turn, so the run fails instead."""
        with self.assertRaises(RuntimeError):
            self.confirm(FakeThread(1, send_after=1, with_ids=False))

    def test_the_turn_text_is_matched_on_letters_and_digits(self):
        page = FakeThread(0, send_after=None)
        page.users = [FakeUnit("u1", "report.pdf\nPDF\n# Round 1\n\n**Hi** there, `x_y`")]
        self.assertEqual(ask_core.own_user_turn(page, set(), "# Round 1\n\nHi there, x_y"), "u1")
        self.assertIsNone(ask_core.own_user_turn(page, {"u1"}, "# Round 1\n\nHi there, x_y"))
        self.assertIsNone(ask_core.own_user_turn(page, set(), "Round 2"))

    def test_open_conversation_returns_the_url_the_page_settles_on(self):
        page = mock.Mock()
        page.evaluate.return_value = CONV  # a bare /c/<id> request is rewritten to the project form
        page.query_selector.return_value = mock.Mock()  # composer present
        with mock.patch.object(ask_core.time, "sleep"):
            landed = ask_core.open_conversation(page, "https://chatgpt.com/c/6a9b8621-0250-83ee-93ed-50f50ee5d7bd")
        self.assertEqual(landed, CONV)
        page.goto.assert_called_once()

    def test_bounced_conversation_is_reported_before_anything_is_typed(self):
        page = mock.Mock()
        page.evaluate.return_value = "https://chatgpt.com/"  # bounced home
        dialog = mock.Mock()
        dialog.inner_text.return_value = "이 대화에 접근할 수 없습니다."
        page.query_selector_all.return_value = [dialog]
        page.query_selector.return_value = mock.Mock()
        with mock.patch.object(ask_core.time, "sleep"):
            with self.assertRaises(RuntimeError) as caught:
                ask_core.open_conversation(page, "https://chatgpt.com/c/00000000-0000-4000-8000-000000000000")
        self.assertIn("not reachable", str(caught.exception))
        self.assertIn("접근할 수 없습니다", str(caught.exception))
        page.keyboard.insert_text.assert_not_called()

    def follow_up(self, *, streaming, server=None):
        """ask() following up in CONV on a page whose composer does or does not show the stop
        button and whose server record is `server` (default: finished); put_text stops the run
        where typing would begin."""
        import itertools

        page = mock.Mock()
        stop = ask_core.STREAMING_BTN_SELECTORS[0]
        page.query_selector.side_effect = lambda sel: mock.Mock() if streaming and sel == stop else None
        answers = server if callable(server) else (lambda: server or conversation(*FINISHED))
        page.evaluate.side_effect = lambda js, *_: answers() if js == ask_core.SERVER_CONVERSATION_JS else CONV
        playwright = mock.MagicMock()
        with thread_state(), mock.patch.object(ask_core, "sync_playwright", return_value=playwright), mock.patch.multiple(
            ask_core,
            port_open=mock.Mock(return_value=True),
            cdp_browser_ok=mock.Mock(return_value=True),
            ensure_page_target=mock.Mock(),
            pick_context=mock.Mock(return_value=mock.Mock(new_page=mock.Mock(return_value=page))),
            _guard_dialogs=mock.Mock(),
            login_state=mock.Mock(return_value="ok"),
            raise_if_rate_limited=mock.Mock(),
            open_conversation=mock.Mock(return_value=CONV),
            select_model=mock.Mock(),
            message_ids=mock.Mock(return_value=set()),
            put_text=mock.Mock(side_effect=RuntimeError("reached typing")),
            click_send=mock.Mock(),
        ), server_clock() as (self.sleeps, _):
            with self.assertRaises(RuntimeError) as caught:
                ask_core.ask("hi", effort="pro", attach=None, max_wait=60, conversation=CONV)
            return str(caught.exception), ask_core.put_text, ask_core.click_send

    def test_follow_up_into_a_reply_in_progress_sends_nothing(self):
        """2026-09-28: a prompt sent while the conversation's previous reply was still running
        joined that exchange instead of starting a turn, so it could not be confirmed or harvested."""
        message, put_text, click_send = self.follow_up(streaming=True)
        self.assertEqual(message, "a reply is still in progress in this conversation; nothing sent")
        put_text.assert_not_called()
        click_send.assert_not_called()

    def test_follow_up_into_a_finished_conversation_goes_on_to_type(self):
        message, _, _ = self.follow_up(streaming=False)
        self.assertEqual(message, "reached typing")

    def test_follow_up_is_refused_when_the_server_is_still_writing_the_leaf_reply(self):
        """A page reloaded during a Pro reply can show 'Thinking' without the stop button; measured
        2026-09-28 on a scratch thread, the server then reports async_status 3 with the user turn or
        a 'thoughts' message as the leaf."""
        for leaf in (RUNNING, RUNNING[:3]):
            message, put_text, click_send = self.follow_up(streaming=False, server=conversation(*leaf, async_status=3))
            self.assertEqual(message, ask_core.REPLY_RUNNING)
            put_text.assert_not_called()
            click_send.assert_not_called()

    def test_only_the_leaf_exchange_can_hold_a_follow_up_back(self):
        """Measured 2026-09-28: a finished or stopped reply leaves async_status null and a leaf that
        ends its turn; an earlier prompt that never got an answer (the work thread's 22:20 send)
        sits further up and does not count, and neither does a flag left on an ended leaf."""
        free = {
            "finished": conversation(*FINISHED),
            "stopped": conversation(*STOPPED),
            "unanswered prompt further up": conversation(*JOINED),
            "flag on an ended leaf": conversation(*FINISHED, async_status=3),
        }
        for name, server in free.items():
            self.assertFalse(ask_core.server_reply_running(server), name)
            self.assertEqual(self.follow_up(streaming=False, server=server)[0], "reached typing", name)

    def test_follow_up_is_refused_when_the_server_record_cannot_be_read(self):
        """A 429 before sending is waited out for about 2 minutes; after that nothing is sent."""
        message, put_text, click_send = self.follow_up(streaming=False, server=BUSY)
        self.assertIn("could not read the conversation's state on the server (HTTP 429); nothing sent", message)
        put_text.assert_not_called()
        click_send.assert_not_called()
        self.assertTrue(ask_core.GUARD_PATIENCE - ask_core.SERVER_BACKOFF_MAX <= sum(self.sleeps) <= ask_core.GUARD_PATIENCE)
        answers = iter([BUSY, {**BUSY, "retry_after": "20"}])
        message, _, _ = self.follow_up(streaming=False, server=lambda: next(answers, conversation(*FINISHED)))
        self.assertEqual(message, "reached typing")
        self.assertEqual(self.sleeps[-1], 20.0)  # Retry-After
        self.assertTrue(3.75 <= self.sleeps[-2] <= 5, self.sleeps)


class FakeAnswer:
    """An assistant unit: message ids, and its Markdown as the serializer would return it."""

    def __init__(self, message_id, text):
        self.message_id, self.text = message_id, text

    def get_attribute(self, name):
        return self.message_id if name == ask_core.MESSAGE_IDS_ATTR else None

    def query_selector(self, _selector):
        return None  # no separate Markdown root: the unit itself is serialized

    def evaluate(self, _js):
        return self.text


class FakeExchange:
    def __init__(self, answer=None, done=False):
        self.answer, self.done = answer, done

    def query_selector_all(self, selector):
        return [self.answer] if self.answer and selector == ask_core.ASSISTANT_MSG_SELECTORS[0] else []

    def evaluate(self, js):
        assert js == ask_core.EXCHANGE_DONE_JS
        return self.done


class FakeConversation:
    """A conversation page as the harvest sees it: exchange wrappers keyed by user-turn id, in page
    order; `stray` user turns sit outside any wrapper."""

    def __init__(self, exchanges, stray=()):
        self.exchanges, self.stray = exchanges, set(stray)

    def query_selector(self, selector):
        import re

        wrapper = re.fullmatch(r'\[data-turn-key="(.+)"\]', selector)
        if wrapper:
            return self.exchanges.get(wrapper.group(1))
        unit = re.fullmatch(r'\[%s~="(.+)"\]' % ask_core.MESSAGE_IDS_ATTR, selector)
        if unit and (unit.group(1) in self.exchanges or unit.group(1) in self.stray):
            return object()
        return None


class HarvestBindingTests(unittest.TestCase):
    """2026-09-28, run b2: queued behind a2's submit lock, b2 opened the conversation seconds after
    a2's send, on a page that showed neither a2's turn nor the stop button. Its send forked a sibling
    branch, a2's answer appeared on b2's page with ids b2 had never seen, and b2 returned it as its
    own with rc 0."""

    def harvest(self, page, *, on_tick=lambda n: None, wait=600):
        import itertools

        clock, ticks = [0.0], itertools.count(1)

        def sleep(secs):
            clock[0] += secs
            on_tick(next(ticks))

        with mock.patch.multiple(
            ask_core,
            current_url=mock.Mock(return_value=CONV),
            is_streaming=mock.Mock(return_value=False),
            raise_if_rate_limited=mock.Mock(),
            detect_quota=mock.Mock(return_value=None),
            log=mock.Mock(),
        ), mock.patch.object(ask_core.time, "sleep", side_effect=sleep), mock.patch.object(
            ask_core.time, "monotonic", side_effect=lambda: clock[0]
        ):
            return ask_core.wait_for_response(page, CONV, "b2-turn", wait)

    def b2_page(self, own_done):
        return FakeConversation(
            {
                "b-turn": FakeExchange(FakeAnswer("b-answer", "PONG-B"), done=True),
                "b2-turn": FakeExchange(FakeAnswer("b2-answer", "PONG-B2"), done=own_done),
                "a2-turn": FakeExchange(FakeAnswer("a2-answer", "1. one\n2. two"), done=True),
            }
        )

    def test_only_this_runs_exchange_is_harvested(self):
        """Another exchange's finished answer, fresh ids and all, is never taken, not even while
        this run's own answer is still being written."""
        page = self.b2_page(own_done=False)

        def finish(tick):
            if tick == 20:
                page.exchanges["b2-turn"].done = True

        self.assertEqual(self.harvest(page, on_tick=finish), ("PONG-B2", {"b2-answer"}))

    def test_a_missing_own_turn_fails_the_harvest_instead_of_taking_another_answer(self):
        page = self.b2_page(own_done=True)
        del page.exchanges["b2-turn"]
        with self.assertRaises(RuntimeError) as caught:
            self.harvest(page)
        self.assertIn("b2-turn is no longer on the page", str(caught.exception))

    def test_own_turn_outside_an_exchange_wrapper_fails_at_once(self):
        page = FakeConversation({"a2-turn": FakeExchange(FakeAnswer("a2-answer", "list"), done=True)}, stray={"b2-turn"})
        with self.assertRaises(RuntimeError) as caught:
            self.harvest(page)
        self.assertIn("outside an exchange wrapper", str(caught.exception))

    def verify(self, server, reply_ids=("b2-answer",), turn="b2-turn", prompt=PROMPT):
        page = mock.Mock()
        page.evaluate.side_effect = server if callable(server) else (lambda *_: server)
        with server_clock() as (self.sleeps, self.log):
            ask_core.verify_reply_exchange(page, CONV, turn, set(reply_ids), prompt)
        return page

    @staticmethod
    def server(turn=None, answer=None, extra=()):
        """b2's plain exchange (turn, thoughts, recap, final) after one earlier exchange, as
        SERVER_CONVERSATION_JS returns it; `turn`/`answer` override fields of b2's turn and answer."""
        server = conversation(*PLAIN, *extra)
        server["messages"]["b2-turn"].update(turn or {})
        server["messages"]["b2-answer"].update(answer or {})
        return server

    def test_server_check_accepts_the_finished_answer_of_this_runs_exchange(self):
        page = self.verify(self.server())
        js, arg = page.evaluate.call_args[0]
        self.assertEqual(js, ask_core.SERVER_CONVERSATION_JS)
        self.assertEqual(arg, {"cid": "6a9b8621-0250-83ee-93ed-50f50ee5d7bd", "turns": 2})  # the last exchange only
        page.evaluate.assert_called_once()

    def test_server_check_accepts_an_agentic_answer(self):
        """The work thread 6ab7a0f7 (1,716 messages, measured 2026-09-28): an answer is the last of
        up to 417 messages under the turn's exchange id — code sent to container.exec or
        api_tool.call_tool, tool execution_output and code messages, commentary text, sub-agent
        notes hidden from the page, turn_plan updates, thoughts and a reasoning recap — and the page
        lists only the final message."""
        self.verify(conversation(*AGENTIC), reply_ids=("a-final",), turn="a-turn", prompt="Round 3: run the agents")

    def test_server_check_binds_a_prompt_sent_into_a_running_reply_to_its_own_answer(self):
        """The work thread's 22:20 send got no answer: a resend two minutes later became the child of
        its reasoning recap and joined its exchange. The final answer belongs to the resend only."""
        self.verify(conversation(*JOINED), reply_ids=("j-final",), turn="j-resend", prompt="Round 2 again")
        with self.assertRaises(ask_core.ForeignReplyError) as caught:
            self.verify(conversation(*JOINED), reply_ids=("j-final",), turn="j-turn", prompt="Round 2")
        self.assertIn("answers another prompt sent after it into the same exchange", str(caught.exception))

    def test_server_check_refuses_every_shape_of_the_measured_race(self):
        """Measured 2026-09-28 (b2, and again with the lock bypassed): b2's prompt forked a sibling of
        a2's turn, a2's answer streamed into b2's wrapper under a2's exchange id and was left
        'in_progress' on the server; a2 itself saw its own answer under its own exchange id, but the
        server had filed b2's final text there."""
        a2 = [("a2-turn", "e-final", "user", "x-a2", {"text": OTHER_PROMPT})]
        shapes = {
            "another prompt's exchange": self.server(answer={"exchange": "x-a2"}),
            "sibling branch": self.server(extra=a2),
            "not this run's prompt": self.server(turn={"text": OTHER_PROMPT}),
            "another prompt's exchange ": self.server(turn={"exchange": None}, answer={"exchange": None}),
        }
        for expected, server in shapes.items():
            with self.assertRaises(ask_core.ForeignReplyError, msg=expected) as caught:
                self.verify(server)
            self.assertIn(expected.strip(), str(caught.exception))
        # The unit that showed a2's list streaming lists that message and b2's final one.
        streamed = ("a2-stream", "b2-recap", "assistant", "x-b2", {"status": "in_progress"})
        mixed = self.server(extra=[streamed])
        mixed["messages"]["b2-answer"]["parent"] = "a2-stream"
        for server, reply_ids in (
            (self.server(answer={"status": "in_progress", "end_turn": False}), ("b2-answer",)),
            (mixed, ("a2-stream", "b2-answer")),
            (self.server(answer={"end_turn": False}), ("b2-answer",)),  # an answer that has not ended its turn
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.verify(server, reply_ids=reply_ids)
            self.assertNotIsInstance(caught.exception, ask_core.ForeignReplyError)
            self.assertIn("not finished on the server", str(caught.exception))

    def test_server_check_waits_for_a_late_record_but_fails_loudly_when_it_cannot_run(self):
        missing = self.server()
        del missing["messages"]["b2-answer"]
        late = iter([self.server(answer={"status": "in_progress", "end_turn": False}), missing, {"error": "HTTP 429"}, self.server()])
        self.verify(lambda *_: next(late))
        for server in (
            {"error": "HTTP 401"},
            missing,
            mock.Mock(side_effect=RuntimeError("Target page has been closed")),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.verify(server)
            self.assertIn("could not confirm the reply", str(caught.exception))
        with self.assertRaises(RuntimeError):
            self.verify({}, reply_ids=())

    def test_server_check_honours_retry_after_then_backs_off_with_jitter(self):
        """Measured 2026-09-28: after a minute of reads every ~3 s the endpoint answered 429 for
        minutes. A 429 is waited out, never read through."""
        answers = iter([{**BUSY, "retry_after": "37"}, self.server()])
        self.verify(lambda *_: next(answers))
        self.assertEqual(self.sleeps, [37.0])
        self.log.assert_called_once_with("server busy (HTTP 429); retrying in 37 s")
        answers = iter([BUSY, BUSY, BUSY, self.server()])
        self.verify(lambda *_: next(answers))
        self.assertEqual(len(self.sleeps), 3)
        for sleep, step in zip(self.sleeps, (5, 10, 20)):
            self.assertTrue(0.75 * step <= sleep <= step, (sleep, step))

    def test_harvest_check_waits_about_20_minutes_of_429_then_reports_it_unreadable(self):
        page = mock.Mock()
        page.evaluate.return_value = BUSY
        with server_clock() as (sleeps, log):
            with self.assertRaises(ask_core.ServerUnreadableError) as caught:
                ask_core.verify_reply_exchange(page, CONV, "b2-turn", {"b2-answer"}, PROMPT)
        self.assertIn("HTTP 429", str(caught.exception))
        self.assertLessEqual(max(sleeps), ask_core.SERVER_BACKOFF_MAX)
        self.assertTrue(ask_core.HARVEST_PATIENCE - ask_core.SERVER_BACKOFF_MAX <= sum(sleeps) <= ask_core.HARVEST_PATIENCE, sum(sleeps))
        self.assertEqual(page.evaluate.call_count, len(sleeps) + 1)  # one read per wait, none in between
        self.assertEqual(log.call_count, len(sleeps))
        # A Retry-After beyond the patience left ends the wait at once.
        page.evaluate.return_value = {**BUSY, "retry_after": "3600"}
        with server_clock() as (sleeps, _):
            with self.assertRaises(ask_core.ServerUnreadableError):
                ask_core.verify_reply_exchange(page, CONV, "b2-turn", {"b2-answer"}, PROMPT)
        self.assertEqual(sleeps, [])

    def test_an_answer_once_seen_unfinished_is_never_left_unverified(self):
        """The mixed race answer stays 'in_progress'; a 429 after that must not turn it into an
        UNVERIFIED answer on disk."""
        answers = iter([self.server(answer={"status": "in_progress", "end_turn": False})])
        with self.assertRaises(RuntimeError) as caught:
            self.verify(lambda *_: next(answers, BUSY))
        self.assertNotIsInstance(caught.exception, ask_core.ServerUnreadableError)
        self.assertIn("nothing returned", str(caught.exception))

    def light(self, full, first, complete=False):
        """A server that answers a light read with window(full, first) and a whole-tree read with
        `full`; records each read's turns."""
        reads = []

        def evaluate(_js, arg):
            reads.append(arg["turns"])
            return full if arg["turns"] is None else window(full, first, complete)

        return evaluate, reads

    def test_light_read_falls_back_to_one_whole_tree_read_when_this_runs_messages_are_not_in_it(self):
        later = [("l-turn", "b2-answer", "user", "x-l", {"text": "later"}), ("l-final", "l-turn", "assistant", "x-l", FINAL)]
        # This run's exchange is the last one: one light read.
        evaluate, reads = self.light(self.server(), "b2-turn")
        self.verify(evaluate)
        self.assertEqual(reads, [2])
        # Another turn came after it, or another branch is current: one read of the whole tree.
        for complete in (False, True):
            evaluate, reads = self.light(self.server(extra=later), "l-turn", complete)
            self.verify(evaluate)
            self.assertEqual(reads, [2, None])
        # A record still being written is read again as a whole tree, after a wait.
        late = self.server()
        del late["messages"]["b2-answer"]
        late["current_node"] = "b2-recap"
        answers = iter([window(late, "b2-turn"), late, self.server()])
        reads = []
        self.verify(lambda _js, arg: reads.append(arg["turns"]) or next(answers))
        self.assertEqual((reads, len(self.sleeps)), ([2, None, None], 1))

    def test_light_read_refuses_the_race_shapes_as_the_whole_tree_does(self):
        """The light endpoint lists only the current branch but marks a prompt that has a sibling
        with has_versions (measured on the race thread 6ab9ff2c); a streamed message from the other
        branch is not listed, so the whole tree decides."""
        sibling = window(self.server(), "e-turn", complete=True)
        sibling["messages"]["b2-turn"]["versions"] = True
        with self.assertRaises(ask_core.ForeignReplyError) as caught:
            self.verify(sibling)
        self.assertIn("sibling branch", str(caught.exception))
        foreign = window(self.server(answer={"exchange": "x-a2"}), "b2-turn")
        with self.assertRaises(ask_core.ForeignReplyError):
            self.verify(foreign)
        joined = window(conversation(*JOINED), "j-turn")
        with self.assertRaises(ask_core.ForeignReplyError) as caught:
            self.verify(joined, reply_ids=("j-final",), turn="j-turn", prompt="Round 2")
        self.assertIn("answers another prompt", str(caught.exception))
        streamed = ("a2-stream", "e-final", "assistant", "x-a2", {"status": "in_progress"})
        mixed = self.server(extra=[streamed])
        mixed["current_node"] = "b2-answer"
        evaluate, reads = self.light(mixed, "b2-turn", complete=False)
        with self.assertRaises(ask_core.ForeignReplyError) as caught:
            self.verify(evaluate, reply_ids=("a2-stream", "b2-answer"))
        self.assertIn("belongs to another prompt's exchange", str(caught.exception))
        self.assertEqual(reads, [2, None])

    def test_an_unreadable_record_at_harvest_saves_the_answer_marked_unverified(self):
        """20 minutes of 429 after a finished answer: nothing on stdout, exit 1, and the page's
        answer on disk under an UNVERIFIED first line, its path and the thread URL in the error."""
        page = mock.Mock()
        page.evaluate.side_effect = lambda js, *_: BUSY if js == ask_core.SERVER_CONVERSATION_JS else CONV
        with thread_state(), mock.patch.object(ask_core, "sync_playwright", return_value=mock.MagicMock()), mock.patch.multiple(
            ask_core,
            port_open=mock.Mock(return_value=True),
            cdp_browser_ok=mock.Mock(return_value=True),
            ensure_page_target=mock.Mock(),
            pick_context=mock.Mock(return_value=mock.Mock(new_page=mock.Mock(return_value=page))),
            _guard_dialogs=mock.Mock(),
            login_state=mock.Mock(return_value="ok"),
            raise_if_rate_limited=mock.Mock(),
            enter_project=mock.Mock(return_value=True),
            select_model=mock.Mock(),
            message_ids=mock.Mock(return_value=set()),
            put_text=mock.Mock(),
            composer_has_prompt=mock.Mock(return_value=True),
            click_send=mock.Mock(),
            confirm_sent_and_capture=mock.Mock(return_value=(CONV, "b2-turn")),
            release_submit_lock=mock.Mock(),
            current_url=mock.Mock(return_value=CONV),
            wait_for_response=mock.Mock(return_value=("PONG-B2", {"b2-answer"})),
        ), server_clock() as (sleeps, log), tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = ask_core.main(["--out", str(output), PROMPT], stdin=io.StringIO())
            self.assertEqual((code, stdout.getvalue()), (1, ""))
            saved = output.read_text(encoding="utf-8").splitlines()
            self.assertTrue(saved[0].startswith("UNVERIFIED: "), saved[0])
            self.assertIn(CONV, saved[0])
            self.assertEqual(saved[1:], ["", "PONG-B2"])
            self.assertIn(f"marked UNVERIFIED, at {output.resolve()}", stderr.getvalue())
            self.assertIn(f"check it at {CONV}", stderr.getvalue())
            self.assertGreaterEqual(sum(sleeps), ask_core.HARVEST_PATIENCE - ask_core.SERVER_BACKOFF_MAX)
            self.assertIn("server busy (HTTP 429); retrying in", log.call_args_list[-1].args[0])

    def test_a_foreign_reply_is_exit_1_with_nothing_printed(self):
        def foreign(*_args, **_kwargs):
            raise ask_core.ForeignReplyError("the answer shown under this run's prompt belongs to another prompt's exchange")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = ask_core.main(["--out", str(output), "hello"], stdin=io.StringIO(), ask_fn=foreign)
            self.assertEqual((code, stdout.getvalue()), (1, ""))
            self.assertIn("another prompt's exchange", stderr.getvalue())
            self.assertFalse(output.exists())


class ConversationLockTests(unittest.TestCase):
    def free(self, url=CONV):
        fd = ask_core.lock_conversation(url)
        if fd is None:
            return False
        ask_core.unlock_conversation(fd)
        return True

    def test_a_second_follow_up_is_refused_before_anything_opens(self):
        with thread_state():
            fd = ask_core.lock_conversation(CONV)
            try:
                with mock.patch.object(ask_core, "sync_playwright") as playwright, mock.patch.object(
                    ask_core, "port_open"
                ) as port_open:
                    with self.assertRaises(RuntimeError) as caught:
                        ask_core.ask("hi", effort="pro", attach=None, max_wait=60, conversation=CONV)
                self.assertTrue(str(caught.exception).startswith(ask_core.CONVERSATION_BUSY))
                self.assertIn(f"pid={os.getpid()}", str(caught.exception))
                playwright.assert_not_called()
                port_open.assert_not_called()
            finally:
                ask_core.unlock_conversation(fd)
            self.assertTrue(self.free())

    def test_the_lock_is_released_on_success_failure_and_exceptions(self):
        other = "https://chatgpt.com/c/0c4b4501-1111-4222-8333-444455556666"
        outcomes = [
            mock.Mock(return_value=ask_core.Reply("answer", CONV)),
            mock.Mock(side_effect=ask_core.ResponseTimeoutError("timed out")),
            mock.Mock(side_effect=KeyboardInterrupt),
        ]
        with thread_state():
            for inner in outcomes:
                with mock.patch.object(ask_core, "_ask", inner):
                    try:
                        ask_core.ask("hi", effort="pro", attach=None, max_wait=60, conversation=CONV)
                    except (ask_core.ResponseTimeoutError, KeyboardInterrupt):
                        pass
                self.assertTrue(self.free(), inner)

            def fresh_chat_then_crash(*_args, held, **_kwargs):
                held.append(ask_core.lock_conversation(other))
                self.assertFalse(self.free(other))
                raise RuntimeError("page crashed")

            with mock.patch.object(ask_core, "_ask", fresh_chat_then_crash):
                with self.assertRaises(RuntimeError):
                    ask_core.ask("hi", effort="pro", attach=None, max_wait=60, project="p")
            self.assertTrue(self.free(other))

    def test_a_fresh_chat_is_locked_before_the_submit_lock_is_released(self):
        seen = {}

        def release_submit_lock():
            seen["free at release"] = self.free()
            with self.assertRaises(RuntimeError) as caught:  # a --continue queued behind the submit lock
                ask_core.ask("next", effort="pro", attach=None, max_wait=60, conversation=CONV)
            seen["queued follow-up"] = str(caught.exception)

        page = mock.Mock()
        playwright = mock.MagicMock()
        with thread_state(), mock.patch.object(ask_core, "sync_playwright", return_value=playwright), mock.patch.multiple(
            ask_core,
            port_open=mock.Mock(return_value=True),
            cdp_browser_ok=mock.Mock(return_value=True),
            ensure_page_target=mock.Mock(),
            pick_context=mock.Mock(return_value=mock.Mock(new_page=mock.Mock(return_value=page))),
            _guard_dialogs=mock.Mock(),
            login_state=mock.Mock(return_value="ok"),
            raise_if_rate_limited=mock.Mock(),
            enter_project=mock.Mock(return_value=True),
            select_model=mock.Mock(),
            message_ids=mock.Mock(return_value=set()),
            put_text=mock.Mock(),
            composer_has_prompt=mock.Mock(return_value=True),
            click_send=mock.Mock(),
            confirm_sent_and_capture=mock.Mock(return_value=(CONV, "u1")),
            record_thread=mock.Mock(),
            release_submit_lock=release_submit_lock,
            wait_for_response=mock.Mock(return_value=("answer", {"a1"})),
            verify_reply_exchange=mock.Mock(),
            current_url=mock.Mock(return_value=CONV),
            log=mock.Mock(),
        ):
            reply = ask_core.ask("hi", effort="pro", attach=None, max_wait=60, project="p")
            self.assertEqual(reply, ask_core.Reply("answer", CONV))
            self.assertTrue(self.free())
        self.assertEqual(seen["free at release"], False)
        self.assertTrue(seen["queued follow-up"].startswith(ask_core.CONVERSATION_BUSY))


@contextlib.contextmanager
def held_locks(paths):
    fds = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fds.append(fd)
        yield
    finally:
        for fd in fds:
            os.close(fd)


class SubmitLockTests(unittest.TestCase):
    def test_release_submit_lock_unlocks_fd_and_removes_info(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "run.lock"
            info_path = Path(f"{lock_path}.info")
            info_path.write_text("owner\n", encoding="utf-8")
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = {
                "CHATGPT_SUBMIT_LOCK_FD": str(fd),
                "CHATGPT_SUBMIT_LOCK_INFO": str(info_path),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ask_core.release_submit_lock()

            contender = os.open(lock_path, os.O_WRONLY)
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(contender)
            self.assertFalse(info_path.exists())
            self.assertNotIn("CHATGPT_SUBMIT_LOCK_FD", os.environ)


class WrapperLockTests(unittest.TestCase):
    def wrapper_env(self, home: Path, **updates):
        env = os.environ.copy()
        env.pop("CHATGPT_MAX_PARALLEL", None)
        env["HOME"] = str(home)
        env.update(updates)
        return env

    def run_wrapper(self, home: Path, **updates):
        return subprocess.run(
            [str(ROOT / "bin" / "chatgpt"), "hello"],
            env=self.wrapper_env(home, **updates),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    def test_default_three_slots_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            slots = home / ".chatgpt" / "slots"
            paths = [slots / f"slot{index}.lock" for index in range(3)]
            with held_locks(paths):
                result = self.run_wrapper(home, CHATGPT_LOCK_WAIT="0")
            self.assertEqual(result.returncode, 4)
            self.assertIn("all 3 run slots are busy", result.stderr)

    def test_max_parallel_env_changes_slot_count(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            slot = home / ".chatgpt" / "slots" / "slot0.lock"
            with held_locks([slot]):
                result = self.run_wrapper(
                    home,
                    CHATGPT_LOCK_WAIT="0",
                    CHATGPT_MAX_PARALLEL="1",
                )
            self.assertEqual(result.returncode, 4)
            self.assertIn("all 1 run slots are busy", result.stderr)

    def test_slot_timeout_is_exit_4(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            slot = home / ".chatgpt" / "slots" / "slot0.lock"
            started = time.monotonic()
            with held_locks([slot]):
                result = self.run_wrapper(
                    home,
                    CHATGPT_LOCK_WAIT="1",
                    CHATGPT_MAX_PARALLEL="1",
                )
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 4)
            self.assertGreaterEqual(elapsed, 0.5)
            self.assertIn("slot wait timed out after 1s", result.stderr)

    def test_slot_is_released_when_run_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            state = home / ".chatgpt"
            run_lock = state / "run.lock"
            slot = state / "slots" / "slot0.lock"
            with held_locks([run_lock]):
                result = self.run_wrapper(
                    home,
                    CHATGPT_LOCK_WAIT="0",
                    CHATGPT_MAX_PARALLEL="1",
                )
            self.assertEqual(result.returncode, 4)
            self.assertIn("another submit holds the lock", result.stderr)
            with held_locks([slot]):
                pass
            self.assertFalse(Path(f"{slot}.info").exists())

    def test_submit_lock_releases_while_slot_stays_held(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            home = Path(directory)
            state = home / ".chatgpt"
            state.mkdir()
            (state / "stack.env").write_text("VNC_DISPLAY=:2\n", encoding="utf-8")
            marker = state / "submit-released"
            fake_bin = home / "bin"
            fake_bin.mkdir()
            scripts = {
                "curl": "#!/bin/sh\nprintf '{\"Browser\":\"Chrome\"}\\n'\n",
                "ss": "#!/bin/sh\nprintf 'LISTEN 0 128 127.0.0.1:6080 0.0.0.0:*\\n'\n",
                "python3": (
                    "#!/bin/sh\n"
                    "[ \"$1\" = -c ] && exit 0\n"
                    "rm -f \"$CHATGPT_SUBMIT_LOCK_INFO\"\n"
                    "flock -u \"$CHATGPT_SUBMIT_LOCK_FD\" || exit 70\n"
                    "touch \"$HOME/.chatgpt/submit-released\"\n"
                    "sleep 2\n"
                    "printf 'mock answer\\n'\n"
                ),
            }
            for name, body in scripts.items():
                path = fake_bin / name
                path.write_text(body, encoding="utf-8")
                path.chmod(0o755)
            env = self.wrapper_env(
                home,
                CHATGPT_MAX_PARALLEL="1",
                PATH=f"{fake_bin}:{os.environ['PATH']}",
            )
            process = subprocess.Popen(
                [str(ROOT / "bin" / "chatgpt"), "hello"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not marker.exists():
                    stdout, stderr = process.communicate(timeout=5)
                    self.fail(
                        f"mock core did not release submit lock: "
                        f"rc={process.returncode} stdout={stdout!r} stderr={stderr!r}"
                    )

                run_fd = os.open(state / "run.lock", os.O_WRONLY)
                slot_fd = os.open(state / "slots" / "slot0.lock", os.O_WRONLY)
                try:
                    fcntl.flock(run_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(run_fd)
                    os.close(slot_fd)
                stdout, stderr = process.communicate(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stdout, "mock answer\n")


class DarwinPathTests(unittest.TestCase):
    """The wrapper's Darwin branch must complete without any VNC/noVNC machinery."""

    def test_darwin_reuses_cdp_and_skips_vnc(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            fake_bin.mkdir()
            scripts = {
                "uname": "#!/bin/sh\nprintf 'Darwin\\n'\n",
                "curl": "#!/bin/sh\nprintf '{\"Browser\":\"Chrome\"}\\n'\n",
                # lsof must not be consulted when CDP is already up; make it fail
                # loudly if it is.
                "lsof": "#!/bin/sh\nexit 66\n",
                "python3": "#!/bin/sh\nprintf 'darwin answer\\n'\n",
            }
            for name, body in scripts.items():
                path = fake_bin / name
                path.write_text(body, encoding="utf-8")
                path.chmod(0o755)
            env = os.environ.copy()
            env.pop("CHATGPT_MAX_PARALLEL", None)
            env["HOME"] = str(home)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            result = subprocess.run(
                [str(ROOT / "bin" / "chatgpt"), "hello"],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "darwin answer\n")
            self.assertIn("reusing Chrome CDP", result.stderr)
            self.assertNotIn("VNC", result.stderr)
            state = (home / ".chatgpt" / "stack.env").read_text(encoding="utf-8")
            self.assertNotIn("VNC_DISPLAY", state)

    def test_darwin_missing_chrome_fails_with_path(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            fake_bin.mkdir()
            scripts = {
                "uname": "#!/bin/sh\nprintf 'Darwin\\n'\n",
                # CDP down -> the wrapper must try to start Chrome and fail on
                # the missing binary. CHATGPT_CHROME_BIN points into the sandbox:
                # on a real Mac the app-bundle binary exists, and without the
                # override this test launches a live Chrome (measured: fresh
                # profile under the fake HOME, macOS Keychain prompt included).
                "curl": "#!/bin/sh\nexit 1\n",
                "lsof": "#!/bin/sh\nexit 1\n",
            }
            for name, body in scripts.items():
                path = fake_bin / name
                path.write_text(body, encoding="utf-8")
                path.chmod(0o755)
            env = os.environ.copy()
            env.pop("CHATGPT_MAX_PARALLEL", None)
            env["HOME"] = str(home)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["CHATGPT_CHROME_BIN"] = str(home / "Google Chrome.app" / "absent")
            result = subprocess.run(
                [str(ROOT / "bin" / "chatgpt"), "hello"],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Chrome binary is unavailable", result.stderr)
            self.assertIn("Google Chrome.app", result.stderr)
            self.assertNotIn("VNC", result.stderr)


class ProjectNameTests(unittest.TestCase):
    def test_default_name_is_folder_dot_hash8_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            spot = Path(directory) / "api"
            spot.mkdir()
            old = os.getcwd()
            os.chdir(spot)
            try:
                first = ask_core.default_project_name()
                second = ask_core.default_project_name()
            finally:
                os.chdir(old)
            self.assertEqual(first, second)
            folder, _, digest = first.rpartition(" · ")
            self.assertEqual(folder, "api")
            self.assertRegex(digest, r"^[0-9a-f]{8}$")

    def test_same_folder_name_different_path_gets_different_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            names = []
            for parent in ("a", "b"):
                spot = Path(directory) / parent / "api"
                spot.mkdir(parents=True)
                old = os.getcwd()
                os.chdir(spot)
                try:
                    names.append(ask_core.default_project_name())
                finally:
                    os.chdir(old)
            self.assertNotEqual(names[0], names[1])

    def test_cache_roundtrip_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projects.json"
            ask_core._save_project_cache(path, {"key": "https://chatgpt.com/g/g-p-x/project"})
            self.assertEqual(
                ask_core._load_project_cache(path),
                {"key": "https://chatgpt.com/g/g-p-x/project"},
            )
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(ask_core._load_project_cache(path), {})
            self.assertEqual(ask_core._load_project_cache(path / "absent"), {})

    def test_parse_args_project_flags(self):
        self.assertEqual(ask_core.parse_args(["--project", "Docs", "hi"]).project, "Docs")
        self.assertTrue(ask_core.parse_args(["--no-project", "hi"]).no_project)
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["--project", "Docs", "--no-project", "hi"])
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["--project", "  ", "hi"])

    def test_main_resolves_project_for_ask_fn(self):
        seen = {}

        def capture(_prompt, **kwargs):
            seen.update(kwargs)
            return ask_core.Reply("answer", CONV)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / "a.md")
            self.assertEqual(ask_core.main(["--out", out, "hello"], stdin=io.StringIO(), ask_fn=capture), 0)
            self.assertEqual(seen["project"], ask_core.default_project_name())
            self.assertIsNone(seen["conversation"])
            ask_core.main(["--out", out, "--project", "Docs", "hello"], stdin=io.StringIO(), ask_fn=capture)
            self.assertEqual(seen["project"], "Docs")
            ask_core.main(["--out", out, "--no-project", "hello"], stdin=io.StringIO(), ask_fn=capture)
            self.assertIsNone(seen["project"])


class RateLimitModalTests(unittest.TestCase):
    def modal_page(self, visible=True, text="요청이 너무 많습니다 몇 분 후 다시 시도해 주세요"):
        node = mock.Mock()
        node.is_visible.return_value = visible
        node.inner_text.return_value = text
        button = mock.Mock()
        node.query_selector_all.return_value = [button]
        page = mock.Mock()
        page.query_selector.return_value = node
        return page, node, button

    def test_visible_modal_is_reported_and_dismiss_clicks_last_button(self):
        page, _node, button = self.modal_page()
        self.assertIn("요청이 너무 많습니다", ask_core.rate_limit_modal(page))
        with mock.patch.object(ask_core.time, "sleep"):
            ask_core.dismiss_rate_limit_modal(page)
        button.click.assert_called_once()

    def test_hidden_or_absent_modal_is_none(self):
        page, _node, _button = self.modal_page(visible=False)
        self.assertIsNone(ask_core.rate_limit_modal(page))
        page.query_selector.return_value = None
        self.assertIsNone(ask_core.rate_limit_modal(page))

    def test_raise_if_rate_limited_dismisses_then_raises(self):
        page, _node, button = self.modal_page()
        with mock.patch.object(ask_core.time, "sleep"):
            with self.assertRaises(ask_core.RateLimitedError) as caught:
                ask_core.raise_if_rate_limited(page, "before submit")
        self.assertIn("before submit", str(caught.exception))
        button.click.assert_called_once()

    def test_rate_limited_error_is_exit_5(self):
        def limited(*_args, **_kwargs):
            raise ask_core.RateLimitedError("mock limit")

        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = ask_core.main(["hello"], stdin=io.StringIO(), ask_fn=limited)
        self.assertEqual(code, 5)
        self.assertIn("mock limit", stderr.getvalue())


class EnsurePageTargetTests(unittest.TestCase):
    def fake_urlopen(self, targets, calls):
        import contextlib as _ctx

        def opener(request, timeout=0):
            url = request if isinstance(request, str) else request.full_url
            method = "GET" if isinstance(request, str) else request.get_method()
            calls.append((method, url))
            reply = mock.Mock()
            reply.read.return_value = json.dumps(targets).encode("utf-8")
            return _ctx.nullcontext(reply)

        return opener

    def test_existing_page_target_means_no_new_tab(self):
        calls = []
        with mock.patch.object(
            ask_core.urllib.request, "urlopen",
            side_effect=self.fake_urlopen([{"type": "page", "url": "about:blank"}], calls),
        ):
            ask_core.ensure_page_target(9222)
        self.assertEqual(len(calls), 1)
        self.assertIn("/json/list", calls[0][1])

    def test_zero_targets_opens_one(self):
        calls = []
        with mock.patch.object(
            ask_core.urllib.request, "urlopen",
            side_effect=self.fake_urlopen([], calls),
        ), mock.patch.object(ask_core.time, "sleep"):
            ask_core.ensure_page_target(9222)
        self.assertEqual(len(calls), 2)
        method, url = calls[1]
        self.assertEqual(method, "PUT")
        self.assertIn("/json/new", url)


class SpawnUnlockedTests(unittest.TestCase):
    def test_daemon_releases_both_locks(self):
        """A spawned daemon must not keep either flock alive once the wrapper's own
        fds are gone. Regression: `without_locks … &` (0.2.0) forked an outer shell
        that skipped the closes and held both locks for the daemon's whole life —
        the daemon's own fd table looked clean, so the assertion must be on lock
        re-acquisition, not on the daemon's fds."""
        wrapper = (ROOT / "bin" / "chatgpt").read_text(encoding="utf-8")
        import re

        helper = re.search(r"^spawn_unlocked\(\) \{.*?^\}", wrapper, re.S | re.M)
        self.assertIsNotNone(helper, "spawn_unlocked() not found in bin/chatgpt")
        with tempfile.TemporaryDirectory() as directory:
            script = "\n".join(
                [
                    "set -u",
                    f'D="{directory}"',
                    helper.group(0),
                    'exec {SLOT_FD}>"$D/slot.lock"; flock -n "$SLOT_FD" || exit 90',
                    'exec {RUN_FD}>"$D/run.lock"; flock -n "$RUN_FD" || exit 91',
                    "spawn_unlocked sleep 2 > /dev/null 2>&1",
                    "sleep 0.5",
                    "exec {SLOT_FD}>&- {RUN_FD}>&-",
                    'exec {S2}>"$D/slot.lock"; flock -n "$S2" || { echo leaked-slot; exit 92; }',
                    'exec {R2}>"$D/run.lock"; flock -n "$R2" || { echo leaked-run; exit 93; }',
                    "echo clean",
                ]
            )
            result = subprocess.run(
                ["bash", "-c", script], text=True, capture_output=True, timeout=10, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("clean", result.stdout)

    def test_no_call_site_backgrounds_the_helper(self):
        """The helper backgrounds internally; a trailing `&` at a call site would
        re-create the outer-shell fork the 0.2.1 fix removed."""
        wrapper = (ROOT / "bin" / "chatgpt").read_text(encoding="utf-8")
        for line in wrapper.replace("\\\n", " ").splitlines():
            if "spawn_unlocked " in line and not line.lstrip().startswith("#"):
                self.assertFalse(line.rstrip().endswith("&"), line)



def _write_stubs(directory: Path, scripts: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        path = directory / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


def _linux_stack_stubs(home: Path) -> dict[str, str]:
    """Stubs for the Linux branch of bin/chatgpt. Every daemon the wrapper spawns drops a
    marker under $MARK; `ss`/`curl` answer from those markers, so the wrapper only sees a
    port as open once the matching stub actually ran."""
    mark = home / "mark"
    mark.mkdir()
    return {
        "uname": "#!/bin/sh\nprintf 'Linux\\n'\n",
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": (
            "#!/bin/sh\n"
            f'[ -e "{mark}/chrome" ] || exit 1\n'
            "printf '{\"Browser\":\"Chrome/1.0\"}\\n'\n"
        ),
        "ss": (
            "#!/bin/sh\n"
            f'for f in "{mark}"/port*; do [ -e "$f" ] || continue; '
            'printf "LISTEN 0 128 127.0.0.1:${f##*port} 0.0.0.0:*\\n"; done\n'
        ),
        # Receives ":N" first; a real vncserver would listen on 5900+N.
        "vncserver": (
            "#!/bin/sh\n"
            f'n="${{1#:}}"; touch "{mark}/port$((5900 + n))"; touch "{mark}/vnc-args-$*"\n'
        ),
        "chrome": f"#!/bin/sh\ntouch \"{mark}/chrome\"\nprintf '%s\\n' \"$*\" > \"{mark}/chrome-args\"\n",
        # `vglrun -d egl <cmd…>`: record the VirtualGL args, then run the command.
        "vglrun": f"#!/bin/sh\nprintf '%s\\n' \"$*\" > \"{mark}/vglrun-args\"\n[ \"$1\" = -d ] && shift 2\nexec \"$@\"\n",
        "websockify": f"#!/bin/sh\ntouch \"{mark}/port6080\"\n",
        "nvidia-smi": "#!/bin/sh\nexit 0\n",
        # `-c` is the wrapper's own `import playwright` probe, not an ask_core run.
        "python3": f"#!/bin/sh\n[ \"$1\" = -c ] && exit 0\ntouch \"{mark}/python3\"\nprintf 'linux answer\\n'\n",
    }


def _run_wrapper(home: Path, fake_bin: Path, argv=("hello",), **updates):
    env = os.environ.copy()
    env.pop("CHATGPT_MAX_PARALLEL", None)
    env.pop("CHATGPT_PYTHON", None)
    env.pop("CHATGPT_STACK_ONLY", None)
    env["HOME"] = str(home)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update(updates)
    return subprocess.run(
        [str(ROOT / "bin" / "chatgpt"), *argv],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


class LinuxStackPathTests(unittest.TestCase):
    """The Linux branch must bring up VNC, Chrome and noVNC through spawn_unlocked.
    Regression: 0.2.1 renamed the helper but start_vnc still called `without_locks`, so
    on a fresh Linux host the VNC server was never launched ("VNC startup failed")."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.fake_bin = self.home / "bin"
        _write_stubs(self.fake_bin, _linux_stack_stubs(self.home))
        (self.home / ".chatgpt" / "browser-profile").mkdir(parents=True)
        self.overrides = {
            "CHATGPT_VNCSERVER": str(self.fake_bin / "vncserver"),
            "CHATGPT_CHROME_BIN": str(self.fake_bin / "chrome"),
            "CHATGPT_VGLRUN": "",  # no VirtualGL: the software-WebGL path
            "CHATGPT_SUBMIT_GAP_MIN": "0",
            "CHATGPT_SUBMIT_GAP_MAX": "0",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_host_starts_vnc_chrome_novnc_then_answers(self):
        result = _run_wrapper(self.home, self.fake_bin, **self.overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "linux answer\n")
        self.assertIn("started VNC :", result.stderr)
        self.assertIn("started Chrome CDP 9222", result.stderr)
        self.assertIn("started noVNC 6080", result.stderr)
        state = (self.home / ".chatgpt" / "stack.env").read_text(encoding="utf-8")
        self.assertRegex(state, r"(?m)^VNC_DISPLAY=:\d+$")
        self.assertRegex(state, r"(?m)^CHROME_GPU=software$")
        vnc_args = [p.name for p in (self.home / "mark").glob("vnc-args-*")]
        self.assertEqual(len(vnc_args), 1, vnc_args)
        self.assertIn("-securitytypes otp", vnc_args[0])
        self.assertIn("-wm openbox", vnc_args[0])
        chrome_args = (self.home / "mark" / "chrome-args").read_text(encoding="utf-8")
        # Software WebGL: OpenAI's sentinel check hangs without any WebGL on a GPU-less Xvnc.
        self.assertIn("--enable-unsafe-swiftshader", chrome_args)
        self.assertIn("--use-angle=swiftshader", chrome_args)
        self.assertIn(f"--user-data-dir={self.home}/.chatgpt/browser-profile", chrome_args)
        self.assertIn("software WebGL", result.stderr)
        self.assertFalse((self.home / "mark" / "vglrun-args").exists())

    def test_virtualgl_present_puts_chrome_on_the_gpu(self):
        """With VirtualGL available Chrome runs under `vglrun -d egl` with the GPU sandbox
        off (it blocks VirtualGL's X connection) and no SwiftShader flags — the software
        renderer is what OpenAI's bot check rejected during login."""
        overrides = {**self.overrides, "CHATGPT_VGLRUN": str(self.fake_bin / "vglrun")}
        result = _run_wrapper(self.home, self.fake_bin, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "linux answer\n")
        self.assertIn("GPU via VirtualGL", result.stderr)
        vgl_args = (self.home / "mark" / "vglrun-args").read_text(encoding="utf-8")
        self.assertTrue(vgl_args.startswith("-d egl "), vgl_args)
        chrome_args = (self.home / "mark" / "chrome-args").read_text(encoding="utf-8")
        self.assertIn("--ignore-gpu-blocklist", chrome_args)
        self.assertIn("--disable-gpu-sandbox", chrome_args)
        self.assertNotIn("swiftshader", chrome_args)
        state = (self.home / ".chatgpt" / "stack.env").read_text(encoding="utf-8")
        self.assertRegex(state, r"(?m)^CHROME_GPU=virtualgl$")

    def test_virtualgl_without_working_nvidia_smi_uses_software(self):
        _write_stubs(self.fake_bin, {"nvidia-smi": "#!/bin/sh\nexit 1\n"})
        overrides = {**self.overrides, "CHATGPT_VGLRUN": str(self.fake_bin / "vglrun")}
        result = _run_wrapper(self.home, self.fake_bin, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("software WebGL", result.stderr)
        chrome_args = (self.home / "mark" / "chrome-args").read_text(encoding="utf-8")
        self.assertIn("--enable-unsafe-swiftshader", chrome_args)
        self.assertFalse((self.home / "mark" / "vglrun-args").exists())
        state = (self.home / ".chatgpt" / "stack.env").read_text(encoding="utf-8")
        self.assertRegex(state, r"(?m)^CHROME_GPU=software$")

    def test_stack_only_exits_before_submit(self):
        result = _run_wrapper(self.home, self.fake_bin, argv=(), CHATGPT_STACK_ONLY="1", **self.overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("stack ready", result.stderr)
        self.assertIn("started noVNC 6080", result.stderr)
        self.assertFalse((self.home / "mark" / "python3").exists(), "ask_core must not run")
        self.assertFalse((self.home / ".chatgpt" / "last_submit").exists(), "limiter stamp must not move")

    def test_no_reference_to_the_old_helper_name(self):
        wrapper = (ROOT / "bin" / "chatgpt").read_text(encoding="utf-8")
        for line in wrapper.splitlines():
            if line.lstrip().startswith("#"):
                continue
            self.assertNotIn("without_locks", line, line)


WSL_UNAME = "#!/bin/sh\ncase \"$1\" in -m) printf 'x86_64\\n' ;; -r) printf '6.18.33.1-microsoft-standard-WSL2\\n' ;; *) printf 'Linux\\n' ;; esac\n"


def _fake_wslg_socket(directory: Path) -> str:
    """A real Unix socket file standing in for WSLg's /mnt/wslg/.X11-unix/X0."""
    path = directory / "X0"
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(path))
    return str(path)


class WslgPathTests(unittest.TestCase):
    """WSLg has a real desktop but mounts /tmp/.X11-unix read-only, so the VNC path cannot
    start there (measured: `vncserver` died within ~2s). Chrome must run directly on WSLg's
    :0 with Mesa's d3d12 driver — plain Chrome under WSLg had no WebGL at all."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.fake_bin = self.home / "bin"
        self.mark = self.home / "mark"
        stubs = _linux_stack_stubs(self.home)
        _write_stubs(self.fake_bin, {
            **stubs,
            "uname": WSL_UNAME,
            "chrome": stubs["chrome"] + (
                f"printf 'DISPLAY=%s GALLIUM_DRIVER=%s ADAPTER=%s\\n' \"$DISPLAY\" \"$GALLIUM_DRIVER\" "
                f"\"$MESA_D3D12_DEFAULT_ADAPTER_NAME\" > \"{self.mark}/chrome-env\"\n"
            ),
        })
        self.overrides = {
            "CHATGPT_VNCSERVER": str(self.fake_bin / "vncserver"),
            "CHATGPT_CHROME_BIN": str(self.fake_bin / "chrome"),
            "CHATGPT_VGLRUN": str(self.fake_bin / "vglrun"),
            "CHATGPT_SUBMIT_GAP_MIN": "0",
            "CHATGPT_SUBMIT_GAP_MAX": "0",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_wslg(self, **updates):
        env = {**self.overrides, "CHATGPT_WSLG_SOCKET": _fake_wslg_socket(self.home), **updates}
        with mock.patch.dict(os.environ):
            os.environ.pop("DISPLAY", None)
            os.environ.pop("CHATGPT_D3D12_ADAPTER", None)
            return _run_wrapper(self.home, self.fake_bin, **env)

    def test_wslg_runs_chrome_on_the_desktop_with_d3d12_and_no_vnc(self):
        result = self.run_wslg()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "linux answer\n")
        self.assertIn("GPU via WSLg D3D12, adapter NVIDIA", result.stderr)
        # First run: the profile is created and the user is pointed at the Windows desktop.
        self.assertIn("Windows desktop", result.stderr)
        self.assertTrue((self.home / ".chatgpt" / "browser-profile").is_dir())
        chrome_env = (self.mark / "chrome-env").read_text(encoding="utf-8")
        self.assertEqual(chrome_env, "DISPLAY=:0 GALLIUM_DRIVER=d3d12 ADAPTER=NVIDIA\n")
        chrome_args = (self.mark / "chrome-args").read_text(encoding="utf-8")
        self.assertNotIn("swiftshader", chrome_args)
        self.assertEqual(list(self.mark.glob("vnc-args-*")), [])
        self.assertFalse((self.mark / "port6080").exists(), "websockify must not run")
        self.assertFalse((self.mark / "vglrun-args").exists(), "VirtualGL must not run")
        self.assertNotIn("VNC", result.stderr)
        state = (self.home / ".chatgpt" / "stack.env").read_text(encoding="utf-8")
        self.assertRegex(state, r"(?m)^CHROME_GPU=d3d12$")
        self.assertNotIn("VNC_DISPLAY", state)
        self.assertNotIn("NOVNC_PORT", state)

    def test_empty_adapter_override_lets_mesa_pick(self):
        result = self.run_wslg(CHATGPT_D3D12_ADAPTER="")
        self.assertEqual(result.returncode, 0, result.stderr)
        chrome_env = (self.mark / "chrome-env").read_text(encoding="utf-8")
        self.assertEqual(chrome_env, "DISPLAY=:0 GALLIUM_DRIVER=d3d12 ADAPTER=\n")

    def test_wsl_without_wslg_keeps_the_vnc_path(self):
        (self.home / ".chatgpt" / "browser-profile").mkdir(parents=True)
        result = self.run_wslg(CHATGPT_WSLG_SOCKET=str(self.home / "absent"), CHATGPT_VGLRUN="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("started VNC :", result.stderr)
        self.assertIn("started noVNC 6080", result.stderr)
        self.assertNotIn("D3D12", result.stderr)
        chrome_env = (self.mark / "chrome-env").read_text(encoding="utf-8")
        self.assertRegex(chrome_env, r"^DISPLAY=:\d+ GALLIUM_DRIVER= ADAPTER=\n$")


class PythonResolutionTests(unittest.TestCase):
    """ask_core runs under the first interpreter that can import playwright: CHATGPT_PYTHON,
    then python3, then the venv `uv tool install playwright` creates. Uses the Darwin
    branch (CDP already up) so no stack machinery is involved."""

    def darwin_stubs(self):
        return {
            "uname": "#!/bin/sh\nprintf 'Darwin\\n'\n",
            "curl": "#!/bin/sh\nprintf '{\"Browser\":\"Chrome\"}\\n'\n",
            "lsof": "#!/bin/sh\nexit 66\n",
        }

    def test_falls_back_to_uv_tool_venv_when_python3_lacks_playwright(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            uv_tools = home / "uvtools"
            _write_stubs(fake_bin, {
                **self.darwin_stubs(),
                # `python3 -c 'import playwright'` fails; anything else would be a bug.
                "python3": "#!/bin/sh\n[ \"$1\" = -c ] && exit 1\nprintf 'system python ran\\n'\nexit 99\n",
                "uv": f"#!/bin/sh\n[ \"$1 $2\" = 'tool dir' ] && printf '{uv_tools}\\n'\n",
            })
            _write_stubs(uv_tools / "playwright" / "bin", {
                "python": "#!/bin/sh\n[ \"$1\" = -c ] && exit 0\nprintf 'uv venv answer\\n'\n",
            })
            result = _run_wrapper(home, fake_bin)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "uv venv answer\n")

    def test_finds_uv_in_default_local_bin_when_not_on_path(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            uv_tools = home / "uvtools"
            _write_stubs(fake_bin, {
                **self.darwin_stubs(),
                "python3": "#!/bin/sh\n[ \"$1\" = -c ] && exit 1\nexit 99\n",
            })
            _write_stubs(home / ".local" / "bin", {
                "uv": f"#!/bin/sh\n[ \"$1 $2\" = 'tool dir' ] && printf '{uv_tools}\\n'\n",
            })
            _write_stubs(uv_tools / "playwright" / "bin", {
                "python": "#!/bin/sh\n[ \"$1\" = -c ] && exit 0\nprintf 'local uv answer\\n'\n",
            })
            path = f"{fake_bin}:/opt/homebrew/bin:/usr/bin:/bin"
            result = _run_wrapper(home, fake_bin, PATH=path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "local uv answer\n")

    def test_explicit_chatgpt_python_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {
                **self.darwin_stubs(),
                "python3": "#!/bin/sh\nprintf 'system python answer\\n'\n",
                "mypython": "#!/bin/sh\nprintf 'explicit answer\\n'\n",
            })
            result = _run_wrapper(home, fake_bin, CHATGPT_PYTHON=str(fake_bin / "mypython"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "explicit answer\n")

    def test_help_uses_resolved_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {"mypython": "#!/bin/sh\nprintf 'help from %s\\n' \"$2\"\n"})
            result = _run_wrapper(home, fake_bin, argv=("--help",), CHATGPT_PYTHON=str(fake_bin / "mypython"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "help from --help\n")


class SetupScriptTests(unittest.TestCase):
    """bin/chatgpt-setup: platform gate and the --check report. Install paths need root
    and the network, so they are exercised by hand on a fresh host, not here."""

    SETUP = ROOT / "bin" / "chatgpt-setup"

    def run_setup(self, home: Path, fake_bin: Path, *argv, **updates):
        env = os.environ.copy()
        for key in ("CHATGPT_PYTHON", "CHATGPT_PROFILE", "CHATGPT_CHROME_BIN", "CHATGPT_VNCSERVER", "CHATGPT_VGLRUN"):
            env.pop(key, None)
        env["HOME"] = str(home)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env.update(updates)
        return subprocess.run(
            [str(self.SETUP), *argv], env=env, text=True, capture_output=True, timeout=30, check=False
        )

    def linux_stubs(self, dpkg_status: str, gpu: bool = True) -> dict[str, str]:
        return {
            "uname": "#!/bin/sh\ncase \"$1\" in -m) printf 'x86_64\\n' ;; *) printf 'Linux\\n' ;; esac\n",
            "id": "#!/bin/sh\ncase \"$1\" in -u) printf '1000\\n' ;; -un) printf 'tester\\n' ;; esac\n",
            "dpkg-query": f"#!/bin/sh\nprintf '{dpkg_status}'\n",
            "flock": "#!/bin/sh\n", "curl": "#!/bin/sh\n", "ss": "#!/bin/sh\n", "gpg": "#!/bin/sh\n",
            # `nvidia-smi -L` succeeding is how the script decides a GPU is present.
            "nvidia-smi": "#!/bin/sh\nexit 0\n" if gpu else "#!/bin/sh\nexit 1\n",
        }

    def os_release(self, home: Path, text: str) -> str:
        path = home / "os-release"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_non_debian_platform_is_exit_2(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {"uname": "#!/bin/sh\ncase \"$1\" in -m) printf 'arm64\\n' ;; *) printf 'Darwin\\n' ;; esac\n"})
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=str(home / "absent"))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Darwin/arm64", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_id_like_debian_passes_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, self.linux_stubs(""))
            release = self.os_release(home, 'ID=linuxmint\nID_LIKE="ubuntu debian"\n')
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                                    CHATGPT_VNCSERVER=str(home / "absent"), CHATGPT_CHROME_BIN=str(home / "absent"),
                                    CHATGPT_PROFILE=str(home / "absent"))
            self.assertNotEqual(result.returncode, 2, result.stderr)
            self.assertIn("dependency", result.stdout)

    def test_check_passes_without_uv_when_python3_has_playwright(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {
                **self.linux_stubs("install ok installed"),
                "python3": "#!/bin/sh\nexit 0\n",
                "vncserver": "#!/bin/sh\n",
                "chrome": "#!/bin/sh\n",
                "vglrun": "#!/bin/sh\n",
            })
            profile = home / "profile"
            profile.mkdir()
            release = self.os_release(home, "ID=ubuntu\nID_LIKE=debian\n")
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                                    CHATGPT_VNCSERVER=str(fake_bin / "vncserver"),
                                    CHATGPT_CHROME_BIN=str(fake_bin / "chrome"),
                                    CHATGPT_VGLRUN=str(fake_bin / "vglrun"),
                                    CHATGPT_PROFILE=str(profile),
                                    PATH=f"{fake_bin}:/opt/homebrew/bin:/usr/bin:/bin")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("MISSING", result.stdout)
            self.assertIn("all dependencies present", result.stderr)
            rows = [line.split()[0] for line in result.stdout.splitlines()[1:]]
            self.assertEqual(rows, ["bash>=5", "flock", "curl", "ss", "gpg", "python3", "openbox", "websockify",
                                    "novnc", "fonts-cjk", "turbovnc", "chrome", "playwright", "profile-dir",
                                    "virtualgl"])

    def test_explicit_chatgpt_python_probe_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {
                **self.linux_stubs("install ok installed", gpu=False),
                "python3": "#!/bin/sh\nexit 0\n",
                "mypython": "#!/bin/sh\nexit 1\n",
                "vncserver": "#!/bin/sh\n",
                "chrome": "#!/bin/sh\n",
            })
            profile = home / "profile"
            profile.mkdir()
            release = self.os_release(home, "ID=ubuntu\n")
            result = self.run_setup(
                home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                CHATGPT_PYTHON=str(fake_bin / "mypython"),
                CHATGPT_VNCSERVER=str(fake_bin / "vncserver"),
                CHATGPT_CHROME_BIN=str(fake_bin / "chrome"), CHATGPT_PROFILE=str(profile),
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertRegex(result.stdout, r"(?m)^playwright\s+MISSING\b")

    def test_root_is_refused_with_one_line_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {"id": "#!/bin/sh\nprintf '0\\n'\n"})
            result = self.run_setup(home, fake_bin, "--check")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
            self.assertIn("unprivileged user that owns the browser profile", result.stderr)
            self.assertIn("apt installs use sudo", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_login_creates_profile_before_dependency_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {
                **self.linux_stubs("install ok installed", gpu=False),
                "python3": "#!/bin/sh\nexit 0\n",
                "chrome": "#!/bin/sh\n",
            })
            profile = home / "new-profile"
            release = self.os_release(home, "ID=debian\n")
            result = self.run_setup(
                home, fake_bin, "--login", CHATGPT_OS_RELEASE=release,
                CHATGPT_VNCSERVER=str(home / "missing-vncserver"),
                CHATGPT_CHROME_BIN=str(fake_bin / "chrome"), CHATGPT_PROFILE=str(profile),
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(profile.is_dir())
            self.assertNotRegex(result.stdout, r"(?m)^profile-dir\s+MISSING\b")
            self.assertIn("missing: turbovnc", result.stderr)

    def test_login_restarts_cdp_when_recorded_gpu_mode_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            setup_stubs = self.linux_stubs("install ok installed", gpu=True)
            stack_stubs = _linux_stack_stubs(home)
            mark = home / "mark"
            _write_stubs(fake_bin, {
                **setup_stubs,
                **stack_stubs,
                "uname": setup_stubs["uname"],
                "id": setup_stubs["id"],
                "pkill": (
                    f"#!/bin/sh\nprintf '%s\\n' \"$*\" > \"{mark}/pkill-args\"\n"
                    f"rm -f \"{mark}/chrome\"\n"
                ),
                "vncpasswd": "#!/bin/sh\nprintf 'One-time password: otp123\\n'\n",
            })
            profile = home / "profile"
            profile.mkdir()
            state_dir = home / ".chatgpt"
            state_dir.mkdir()
            (state_dir / "stack.env").write_text(
                "VNC_DISPLAY=:2\nNOVNC_PORT=6080\nCDP_PORT=9222\nCHROME_GPU=software\n",
                encoding="utf-8",
            )
            (mark / "chrome").touch()
            release = self.os_release(home, "ID=ubuntu\n")
            result = self.run_setup(
                home, fake_bin, "--login", CHATGPT_OS_RELEASE=release,
                CHATGPT_VNCSERVER=str(fake_bin / "vncserver"),
                CHATGPT_CHROME_BIN=str(fake_bin / "chrome"),
                CHATGPT_VGLRUN=str(fake_bin / "vglrun"), CHATGPT_PROFILE=str(profile),
                CHATGPT_SUBMIT_GAP_MIN="0", CHATGPT_SUBMIT_GAP_MAX="0",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Chrome GPU mode changed (software -> virtualgl)", result.stderr)
            pkill_args = (mark / "pkill-args").read_text(encoding="utf-8")
            self.assertIn(f"user-data-dir={profile}", pkill_args)
            state = (state_dir / "stack.env").read_text(encoding="utf-8")
            self.assertRegex(state, r"(?m)^CHROME_GPU=virtualgl$")

    def test_virtualgl_row_only_on_nvidia_hosts_and_not_when_opted_out(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {**self.linux_stubs("", gpu=False), "python3": "#!/bin/sh\nexit 1\n"})
            release = self.os_release(home, "ID=debian\n")
            common = dict(CHATGPT_OS_RELEASE=release, CHATGPT_VNCSERVER=str(home / "absent"),
                          CHATGPT_CHROME_BIN=str(home / "absent"), CHATGPT_PROFILE=str(home / "absent"))
            result = self.run_setup(home, fake_bin, "--check", **common)
            self.assertNotIn("virtualgl ", result.stdout)
            self.assertIn("virtualgl: skipped (no NVIDIA GPU", result.stderr)
            # GPU present but explicitly opted out.
            _write_stubs(fake_bin, {"nvidia-smi": "#!/bin/sh\nexit 0\n"})
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_VGLRUN="", **common)
            self.assertNotIn("virtualgl ", result.stdout)
            self.assertIn("virtualgl: skipped (CHATGPT_VGLRUN is empty", result.stderr)

    def test_check_on_wslg_drops_the_vnc_stack_and_virtualgl(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {**self.linux_stubs(""), "uname": WSL_UNAME, "python3": "#!/bin/sh\nexit 1\n"})
            release = self.os_release(home, "ID=ubuntu\n")
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                                    CHATGPT_WSLG_SOCKET=_fake_wslg_socket(home),
                                    CHATGPT_VNCSERVER=str(home / "absent"), CHATGPT_CHROME_BIN=str(home / "absent"),
                                    CHATGPT_PROFILE=str(home / "absent"))
            rows = [line.split()[0] for line in result.stdout.splitlines()[1:]]
            self.assertEqual(rows, ["bash>=5", "flock", "curl", "ss", "gpg", "python3", "fonts-cjk", "chrome",
                                    "playwright", "profile-dir", "mesa-d3d12"])
            self.assertIn("virtualgl: skipped (WSLg", result.stderr)

    def test_check_lists_missing_items_and_exits_1(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            uv_tools = home / "uvtools"
            uv_tools.mkdir()
            _write_stubs(fake_bin, {
                **self.linux_stubs(""),
                "python3": "#!/bin/sh\nexit 1\n",
                "uv": f"#!/bin/sh\n[ \"$1 $2\" = 'tool dir' ] && printf '{uv_tools}\\n'\n",
            })
            release = self.os_release(home, "ID=debian\n")
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                                    CHATGPT_VNCSERVER=str(home / "absent"), CHATGPT_CHROME_BIN=str(home / "absent"),
                                    CHATGPT_VGLRUN=str(home / "absent"), CHATGPT_PROFILE=str(home / "absent"))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            missing = {line.split()[0] for line in result.stdout.splitlines() if " MISSING " in line}
            self.assertEqual(missing, {"openbox", "websockify", "novnc", "fonts-cjk", "turbovnc", "chrome",
                                       "playwright", "profile-dir", "virtualgl"})
            self.assertIn("missing: openbox websockify novnc fonts-cjk turbovnc chrome playwright profile-dir virtualgl",
                          result.stderr)
            self.assertNotRegex(result.stdout, r"(?m)^uv\s+")

    def test_check_never_touches_sudo_or_the_network(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            fake_bin = home / "bin"
            _write_stubs(fake_bin, {
                **self.linux_stubs(""),
                "sudo": "#!/bin/sh\necho SUDO-CALLED >&2; exit 77\n",
                "apt-get": "#!/bin/sh\necho APT-CALLED >&2; exit 77\n",
                "curl": "#!/bin/sh\necho CURL-CALLED >&2; exit 77\n",
                "python3": "#!/bin/sh\nexit 1\n",
            })
            release = self.os_release(home, "ID=debian\n")
            result = self.run_setup(home, fake_bin, "--check", CHATGPT_OS_RELEASE=release,
                                    CHATGPT_VNCSERVER=str(home / "absent"), CHATGPT_CHROME_BIN=str(home / "absent"),
                                    CHATGPT_PROFILE=str(home / "absent"))
            self.assertEqual(result.returncode, 1)
            for token in ("SUDO-CALLED", "APT-CALLED", "CURL-CALLED"):
                self.assertNotIn(token, result.stderr)

    def test_unknown_option_is_usage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            result = self.run_setup(home, home / "nobin", "--bogus")
            self.assertEqual(result.returncode, 64)
            self.assertIn("unknown option", result.stderr)

if __name__ == "__main__":
    unittest.main()


def test_rate_limiter_env_defaults():
    """The submit limiter block exists with the expected defaults and stamp file."""
    src = open(BIN).read() if 'BIN' in globals() else open(__file__.replace('tests/test_chatgpt.py','bin/chatgpt')).read()
    assert 'CHATGPT_SUBMIT_GAP_MIN:-8' in src
    assert 'CHATGPT_SUBMIT_GAP_MAX:-20' in src
    assert 'last_submit' in src
