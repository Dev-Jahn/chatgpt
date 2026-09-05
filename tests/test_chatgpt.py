from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
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
TRAILER = ask_core.conversation_trailer(CONV)


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
    def test_turn_is_complete_only_when_its_own_turn_shows_the_copy_action(self):
        """Older turns (and the user's own turn) keep copy buttons, so completion is decided
        on the fresh assistant node's turn, never on a page-wide count."""
        node = mock.Mock()
        with mock.patch.object(ask_core, "is_streaming", return_value=False):
            node.evaluate.return_value = True
            self.assertTrue(ask_core.turn_complete(mock.Mock(), node))
            node.evaluate.return_value = False
            self.assertFalse(ask_core.turn_complete(mock.Mock(), node))
            self.assertEqual(node.evaluate.call_args[0][0], ask_core.TURN_HAS_COPY_JS)
        with mock.patch.object(ask_core, "is_streaming", return_value=True):
            node.evaluate.return_value = True
            self.assertFalse(ask_core.turn_complete(mock.Mock(), node))


class EffortArgTests(unittest.TestCase):
    def test_effort_names_normalize_to_slider_levels(self):
        self.assertEqual(ask_core.parse_args(["--effort", "Extra-High", "hi"]).effort, "extra high")
        self.assertEqual(ask_core.parse_args(["--effort", "PRO", "hi"]).effort, "pro")
        self.assertEqual(ask_core.effort_index("extra_high"), 3)

    def test_unknown_effort_is_a_usage_error(self):
        for bad in ("ultra", "", "  ", "5"):
            with self.assertRaises(ask_core.UsageError):
                ask_core.parse_args(["--effort", bad, "hi"])


class FakeNode:
    TICK_X0 = 100
    TICK_STEP = 50

    def __init__(self, page, kind, index=0, text=""):
        self.page, self.kind, self.index, self.text = page, kind, index, text

    def inner_text(self):
        if self.kind == "pill":
            return "추론 수준" if self.page.menu_open else self.page.label().replace(" ", "\n")
        return self.text

    def get_attribute(self, name):
        if self.kind == "toggle" and name == "aria-expanded":
            return "true" if self.page.expanded else "false"
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
    """Just enough of the 2026-09 Chat-mode composer to drive select_model: a Chat/Work toggle,
    a pill whose closed text is the current selection, and a menu with a model list behind a
    toggle plus a tick slider that ignores clicks while the list is expanded (measured)."""

    LABELS = ["Instant", "중간", "High", "매우 높음", "Pro"]

    def __init__(self, *, mode="chat", explicit=False, value=4, value_max=4, radios=None, picker=True):
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
            return "5.6 Pro" if self.explicit else "6 Pro"
        return self.LABELS[self.value]

    def wait_for_selector(self, *_args, **_kwargs):
        return None

    def evaluate(self, js, *_args):
        if "data-tpp-toggle-value" in js:
            if self.mode is None:
                return []
            return [["chatgpt", self.mode == "chat"], ["work", self.mode == "work"]]
        if "composer-intelligence-picker-content" in js:
            if not (self.menu_open and self.picker):
                return None
            checked = "GPT-5.6 Sol" if self.explicit else "최신"
            return {
                "view": "advanced" if self.expanded else "simple",
                "explicit_model": "true" if self.explicit else "false",
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
        if selector == ask_core.CHAT_MODE_RADIO_SELECTOR:
            return [FakeNode(self, "chat")] if self.mode else []
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
        elif node.kind == "chat":
            self.mode = "chat"
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

    def test_pill_already_at_pro_skips_the_menu(self):
        page = FakePicker()
        result, log = self.select(page)
        self.assertEqual(result, "Latest (6 Pro)")
        self.assertEqual(page.menu_opens, 0)
        self.assertIn("already", log)

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

    def test_menu_without_latest_entry_fails_closed(self):
        page = FakePicker(explicit=True, radios=["GPT-5.6 Sol", "GPT-5.5"])
        with self.assertRaises(ask_core.ModelVerificationError) as caught:
            self.select(page)
        self.assertIn("GPT-5.5", str(caught.exception))
        self.assertEqual(page.tick_clicks, [])


class ContinueArgTests(unittest.TestCase):
    ID = "6a9b8621-0250-83ee-93ed-50f50ee5d7bd"

    def test_conversation_forms_normalize_to_a_url(self):
        self.assertEqual(ask_core.parse_args(["--continue", CONV, "hi"]).conversation, CONV)
        self.assertEqual(
            ask_core.parse_args(["--continue", self.ID, "hi"]).conversation,
            f"https://chatgpt.com/c/{self.ID}",
        )
        self.assertEqual(
            ask_core.parse_args(["--continue", f"chatgpt.com/c/{self.ID}/", "hi"]).conversation,
            f"https://chatgpt.com/c/{self.ID}",
        )
        self.assertIsNone(ask_core.parse_args(["hi"]).conversation)

    def test_garbage_and_project_flags_are_usage_errors(self):
        for bad in ("not-a-conversation", "https://chatgpt.com/", "", "  "):
            with self.assertRaises(ask_core.UsageError):
                ask_core.parse_args(["--continue", bad, "hi"])
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["--continue", self.ID, "--project", "Docs", "hi"])
        with self.assertRaises(ask_core.UsageError):
            ask_core.parse_args(["--continue", self.ID, "--no-project", "hi"])

    def test_main_passes_the_conversation_and_no_project(self):
        seen = {}

        def capture(_prompt, **kwargs):
            seen.update(kwargs)
            return ask_core.Reply("answer", CONV)

        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / "a.md")
            code = ask_core.main(
                ["--out", out, "--continue", self.ID, "hello"], stdin=io.StringIO(), ask_fn=capture
            )
        self.assertEqual(code, 0)
        self.assertEqual(seen["conversation"], f"https://chatgpt.com/c/{self.ID}")
        self.assertIsNone(seen["project"])


class ContinuationFlowTests(unittest.TestCase):
    def test_trailer_names_the_conversation_and_the_option(self):
        self.assertEqual(
            TRAILER.splitlines(),
            [
                "---",
                f"Conversation: {CONV}",
                "To continue this thread with a follow-up (its context is retained), pass "
                f"`--continue {CONV}` on the next `chatgpt` call. Omit it to start a fresh chat.",
            ],
        )

    def test_follow_up_send_is_confirmed_by_a_new_user_turn_only(self):
        counts = iter([1, 1, 2])
        with mock.patch.object(ask_core, "count_nodes", side_effect=lambda *_: next(counts)), mock.patch.object(
            ask_core, "current_url", return_value=CONV
        ), mock.patch.object(ask_core.time, "sleep"):
            self.assertEqual(ask_core.confirm_sent_and_capture(mock.Mock(), 1, CONV), CONV)

    def test_follow_up_never_takes_the_bound_url_as_proof_of_sending(self):
        """In a fresh chat the URL flipping to /c/<id> proves the send; a follow-up page carries
        that URL from the start, so the same shortcut would report a send that never happened."""
        import itertools

        clock = itertools.count(0, 10)
        with mock.patch.object(ask_core, "count_nodes", return_value=1), mock.patch.object(
            ask_core, "current_url", return_value=CONV
        ), mock.patch.object(ask_core.time, "sleep"), mock.patch.object(
            ask_core.time, "monotonic", side_effect=lambda: next(clock)
        ):
            with self.assertRaises(RuntimeError):
                ask_core.confirm_sent_and_capture(mock.Mock(), 1, CONV)
        with mock.patch.object(ask_core, "count_nodes", return_value=0), mock.patch.object(
            ask_core, "current_url", return_value=CONV
        ), mock.patch.object(ask_core.time, "sleep"):
            self.assertEqual(ask_core.confirm_sent_and_capture(mock.Mock(), 0), CONV)

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
