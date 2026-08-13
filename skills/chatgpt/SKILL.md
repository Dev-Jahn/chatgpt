---
name: chatgpt
description: Delegate ultra-hard analysis/design/research/verification tasks (tens of minutes+) to subscription ChatGPT Pro (GPT-5.6 Sol, pinned). Just pass a prompt — no packing or templates; ask for authenticated connectors (GitHub etc.) directly in the prompt.
---

The plugin's `bin/` is on PATH — call `chatgpt` directly.

```bash
chatgpt "your question"
chatgpt -f prompt.md
echo "your question" | chatgpt -
```

Long-running by nature — invoke with `run_in_background`. Up to a few runs
execute concurrently; excess waits on a lock. stdout is the response body only. Exit codes: 0 = success,
2 = model verification failed (nothing sent), 3 = response timeout,
4 = lock timeout, 5 = ChatGPT rate limit (if the prompt was already sent, the
error carries the conversation URL for a later manual pickup).

Chats are grouped under a per-folder ChatGPT project named `<folder> · <hash8>`
(hash of the cwd, so same-named folders stay separate) instead of piling up in
the root chat list. `--project NAME` picks an explicit project; `--no-project`
opts out. Any project failure falls back to a plain chat — never an abort.
