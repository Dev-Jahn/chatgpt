---
name: chatgpt
description: Delegate ultra-hard analysis/design/research/verification tasks (tens of minutes+) to subscription ChatGPT Pro (GPT-6 Pro — Chat mode, Latest model at Pro effort, verified before sending). Just pass a prompt — no packing or templates; ask for authenticated connectors (GitHub etc.) directly in the prompt.
---

The plugin's `bin/` is on PATH — call `chatgpt` directly. If it fails on a
missing dependency or a login page, `chatgpt-setup --check` shows what is
missing, `chatgpt-setup` installs it (Linux x86_64 / Debian-family apt only),
and `chatgpt-setup --login` starts the browser stack and prints a noVNC URL +
one-time password for the user to sign in with.

```bash
chatgpt "your question"
chatgpt -f prompt.md
echo "your question" | chatgpt -
```

Long-running by nature — invoke with `run_in_background`. Up to a few runs
execute concurrently; excess waits on a lock. stdout is the response body
followed by a `Conversation: <url>` trailer (see Follow-ups).
The model is fixed (Chat mode · Latest · Pro, shown as `6 Pro`); `--effort
instant|medium|high|extra high` lowers the reasoning effort when a Pro-length
wait is not warranted. Exit codes: 0 = success,
2 = model verification failed (nothing sent), 3 = response timeout,
4 = lock timeout, 5 = ChatGPT rate limit (if the prompt was already sent, the
error carries the conversation URL for a later manual pickup).

Chats are grouped under a per-folder ChatGPT project named `<folder> · <hash8>`
(hash of the cwd, so same-named folders stay separate) instead of piling up in
the root chat list. `--project NAME` picks an explicit project; `--no-project`
opts out. Any project failure falls back to a plain chat — never an abort.

Follow-ups: every reply ends with a trailer naming its conversation URL. When
the next request builds on that answer (same topic, more detail, a correction),
pass `--continue <that url>` so the prompt lands in the same thread with its
context retained; otherwise omit it and a fresh chat is started.
