# chatgpt

A neutral Claude Code plugin that forwards a prompt to a logged-in subscription
ChatGPT Pro browser session and returns the response as Markdown. No review
framing, no templates, no repomix packing, no connector manipulation — just a
bridge. The model is pinned and verified as `GPT-5.6 Sol` (fail-closed: nothing
is sent on a mismatch), and the default reasoning effort is `Pro`.

Intended use: offloading ultra-hard analysis / design / research / verification
tasks that take tens of minutes or more of Pro-tier reasoning.

## Install

```bash
claude plugin marketplace add Dev-Jahn/jahns-cc-marketplace
claude plugin install chatgpt@jahns-cc-marketplace
```

Once the plugin is installed, its `bin/` directory is automatically added to
`PATH` in Claude Code sessions — just call `chatgpt`. No symlinks needed.

On a fresh Linux x86_64 host with apt (Debian/Ubuntu), `chatgpt-setup` installs
everything listed under [Environment](#environment) and then walks you through
the one manual step, signing in:

```bash
chatgpt-setup --check    # report what is present / missing; no sudo, no network
chatgpt-setup            # install the missing pieces (apt via sudo, uv, Chrome .deb)
chatgpt-setup --login    # start VNC + Chrome + noVNC and print a URL + one-time password
```

`--login` prints an `ssh -L` tunnel line and a `http://localhost:6080/vnc.html?…`
URL; sign in to chatgpt.com in that Chrome window once and the profile keeps the
session. Other platforms exit 2 — install by hand from the list below.

## Usage

```bash
chatgpt "your question"
chatgpt -f prompt.md
echo "your question" | chatgpt -
chatgpt --effort pro --attach context.pdf --max-wait 7200 --out answer.md "question"
```

- stdout carries the response body only; progress and diagnostics go to
  stderr. `--quiet` prints just the saved-file path. Responses are also saved
  under `~/.chatgpt/out/<timestamp>.md`.
- Up to a few runs execute concurrently; excess waits on a lock for up to
  `CHATGPT_LOCK_WAIT` seconds (default 3600).
- Exit codes: `0` success, `2` model verification failed (nothing sent),
  `3` response timeout, `4` lock timeout.
- Connectors (GitHub, Drive, …) already authenticated in the ChatGPT account
  are used by simply asking for them in the prompt (e.g. "use the GitHub
  connector to inspect repo X"); the tool never packs or attaches anything
  unless you pass `--attach`.

## Environment

- Linux, Bash ≥ 5, `flock`, `curl`, `ss`
- `python3` with the `playwright` package — `uv tool install playwright` is
  enough (the launcher finds that venv on its own when the system `python3`
  cannot import it; `CHATGPT_PYTHON` forces a specific interpreter). No
  `playwright install` — the tool attaches to Chrome over CDP.
- `/usr/bin/google-chrome` (`CHATGPT_CHROME_BIN`), TurboVNC
  (`/opt/TurboVNC/bin/vncserver`, `CHATGPT_VNCSERVER`), `openbox`,
  `websockify`, noVNC assets under `/usr/share/novnc`
- a logged-in Chrome profile at `~/.chatgpt/browser-profile` (override with `CHATGPT_PROFILE`)

If a Chrome CDP stack is already alive on port 9222 it is reused. Otherwise a
free VNC display is picked and Chrome CDP + noVNC are started automatically;
stack state and logs live under `~/.chatgpt/`. The tool never modifies login
or connector authentication state. `CHATGPT_STACK_ONLY=1 chatgpt` brings the
stack up (or reuses it) and exits without submitting — that is what
`chatgpt-setup --login` builds on.

## Development

Development happens on the `dev` branch, which carries the `tests/` directory.
`main` is the release branch (tests stripped); every push to `main` runs a
workflow that pins the new sha/version into
[`Dev-Jahn/jahns-cc-marketplace`](https://github.com/Dev-Jahn/jahns-cc-marketplace).
