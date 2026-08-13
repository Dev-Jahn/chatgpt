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
- Runs are serialized by an exclusive lock (`~/.chatgpt/run.lock`); a second
  invocation waits up to `CHATGPT_LOCK_WAIT` seconds (default 3600).
- Exit codes: `0` success, `2` model verification failed (nothing sent),
  `3` response timeout, `4` lock timeout.
- Connectors (GitHub, Drive, …) already authenticated in the ChatGPT account
  are used by simply asking for them in the prompt (e.g. "use the GitHub
  connector to inspect repo X"); the tool never packs or attaches anything
  unless you pass `--attach`.

## Environment

- Linux, Bash, `flock`, `curl`, `ss`
- system `python3` with the `playwright` package
- `/usr/bin/google-chrome`, TurboVNC (`/opt/TurboVNC/bin/vncserver`),
  `openbox`, `websockify`, noVNC assets
- a logged-in Chrome profile at `~/.chatgpt/browser-profile` (override with `CHATGPT_PROFILE`)

If a Chrome CDP stack is already alive on port 9222 it is reused. Otherwise a
free VNC display is picked and Chrome CDP + noVNC are started automatically;
stack state and logs live under `~/.chatgpt/`. The tool never modifies login
or connector authentication state.

## Development

Development happens on the `dev` branch, which carries the `tests/` directory.
`main` is the release branch (tests stripped); every push to `main` runs a
workflow that pins the new sha/version into
[`Dev-Jahn/jahns-cc-marketplace`](https://github.com/Dev-Jahn/jahns-cc-marketplace).
