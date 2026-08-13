from __future__ import annotations

import contextlib
import fcntl
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import ask_core  # noqa: E402


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
            code, stdout, stderr = self.run_main(
                ["--out", str(output), "hello"], lambda *_args, **_kwargs: "answer"
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout, "answer\n")
            self.assertEqual(stderr, "")
            self.assertEqual(output.read_text(encoding="utf-8"), "answer\n")

    def test_quiet_prints_only_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.md"
            code, stdout, _ = self.run_main(
                ["--quiet", "--out", str(output), "hello"],
                lambda *_args, **_kwargs: "answer",
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout, f"{output.resolve()}\n")


class WrapperLockTests(unittest.TestCase):
    def test_lock_contention_is_exit_4(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            state = home / ".chatgpt"
            state.mkdir()
            lock_path = state / "run.lock"
            with lock_path.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                env = os.environ.copy()
                env["HOME"] = str(home)
                env["CHATGPT_LOCK_WAIT"] = "0"
                result = subprocess.run(
                    [str(ROOT / "bin" / "chatgpt"), "hello"],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            self.assertEqual(result.returncode, 4)
            self.assertEqual(result.stdout, "")
            self.assertIn("holds the lock", result.stderr)


if __name__ == "__main__":
    unittest.main()

