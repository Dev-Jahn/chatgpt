# chatgpt

A neutral Claude Code plugin that forwards a prompt to a logged-in subscription
ChatGPT Pro browser session and returns the response as Markdown. No review
framing, no templates, no repomix packing, no connector manipulation — just a
bridge. The composer is driven in **Chat** mode with the **Latest** model
(`최신`) at **Pro** reasoning effort, and the run is fail-closed: the model pill
must read exactly `6 Pro` (GPT-6 Pro, as of 2026-09) before anything is sent,
otherwise the run stops with exit 2. A composer left in Work mode or on an
explicit model is switched back first; Work mode's top tier (Ultra, a
multi-turn agentic mode) is deliberately not used.

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
chatgpt-setup            # install the missing pieces (apt via sudo, Chrome .deb; uv when needed)
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
chatgpt --continue "follow-up question"          # most recent thread from this folder
chatgpt --resume 6a9b8621 "follow-up question"   # a specific thread, by its handle
```

- stdout carries the response body, then a blank line and a short trailer
  (`---`, `Thread <handle> · <url>`, and one sentence on how to follow up);
  progress and diagnostics go to stderr. `--quiet` replaces the body with the
  saved-file path but keeps the trailer. The saved file (`--out`, default
  `~/.chatgpt/out/<timestamp>.md`) holds the body only.
- Follow-ups keep the thread's context: `--continue` reopens the most recent
  thread started from this folder; `--resume <handle>` picks one by the 8-hex
  handle a previous trailer printed (a chatgpt.com conversation URL is accepted
  too, for a chat this tool did not start). Threads are remembered in
  `~/.chatgpt/threads.json` (`CHATGPT_STATE_DIR` relocates it); a handle that
  is not in it — or matches more than one thread — is exit 64 with nothing
  opened. Project grouping does not apply to a follow-up (`--project`/
  `--no-project` are rejected alongside it). A deleted or foreign thread is
  reported before anything is typed (exit 1).
- Up to a few runs execute concurrently; excess waits on a lock for up to
  `CHATGPT_LOCK_WAIT` seconds (default 3600).
- `--effort` takes one of the five slider positions `instant`, `medium`,
  `high`, `extra high`, `pro` (default `pro`). Below `pro` the ChatGPT UI labels
  the Latest model by effort alone (no version), so those runs are verified as
  Latest + slider position only.
- Exit codes: `0` success, `2` model verification failed (nothing sent),
  `3` response timeout, `4` lock timeout, `5` ChatGPT rate limit.
- Connectors (GitHub, Drive, …) already authenticated in the ChatGPT account
  are used by simply asking for them in the prompt (e.g. "use the GitHub
  connector to inspect repo X"); the tool never packs or attaches anything
  unless you pass `--attach`.

## Environment

- Linux, Bash ≥ 5, `flock`, `curl`, `ss`
- `python3` with the `playwright` package — `uv tool install playwright` is
  enough (uv is only an installer, not a runtime dependency; the launcher finds
  that venv, including uv's default `~/.local/bin` install, when system `python3`
  cannot import it; `CHATGPT_PYTHON` forces a specific interpreter). No
  `playwright install` — the tool attaches to Chrome over CDP.
- `/usr/bin/google-chrome` (`CHATGPT_CHROME_BIN`), TurboVNC
  (`/opt/TurboVNC/bin/vncserver`, `CHATGPT_VNCSERVER`), `openbox`,
  `websockify`, noVNC assets under `/usr/share/novnc`
- on a host with a working `nvidia-smi -L`: VirtualGL (`/opt/VirtualGL/bin/vglrun`,
  `CHATGPT_VGLRUN`; set it empty to opt out). Chrome then renders WebGL on the
  GPU through VirtualGL's EGL back end — no X server on the GPU needed. Without
  it Chrome falls back to software WebGL, which OpenAI's bot check treated as a
  headless browser: the login hung after the password step until the GPU path
  was used.
- a logged-in Chrome profile at `~/.chatgpt/browser-profile` (override with `CHATGPT_PROFILE`)

If a Chrome CDP stack is already alive on port 9222 it is reused. Otherwise a
free VNC display is picked and Chrome CDP + noVNC are started automatically
(on Linux under `vglrun -d egl` when VirtualGL is present and `nvidia-smi -L`
succeeds, else with software
WebGL via `--enable-unsafe-swiftshader`; the startup line says which);
stack state and logs live under `~/.chatgpt/`. The tool never modifies login
or connector authentication state. `CHATGPT_STACK_ONLY=1 chatgpt` brings the
stack up (or reuses it) and exits without submitting — that is what
`chatgpt-setup --login` builds on.

## Development

Development happens on the `dev` branch, which carries the `tests/` directory.
`main` is the release branch (tests stripped); every push to `main` runs a
workflow that pins the new sha/version into
[`Dev-Jahn/jahns-cc-marketplace`](https://github.com/Dev-Jahn/jahns-cc-marketplace).
