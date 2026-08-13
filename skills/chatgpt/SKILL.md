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
4 = lock timeout.
