---
name: image-gen
description: Generate or edit images with the subscription ChatGPT's image tool (effort medium by default — pass `--effort LEVEL` to change it). Saves the images and prints their paths; a reference image can be attached for edits.
argument-hint: "[--effort instant|medium|high|extra high|pro] <image prompt>"
---

The plugin's `bin/` is on PATH — call `chatgpt --image` directly (setup and
sign-in problems: see `chatgpt:pro`, the same `chatgpt-setup` applies).

```bash
chatgpt --image "a watercolor fox reading a newspaper in a Seoul cafe"
chatgpt --image -f prompt.md
echo "an isometric icon of a GPU rack" | chatgpt --image -
```

The effort defaults to `medium`. When the skill's arguments include
`--effort LEVEL` (`instant`, `medium`, `high`, `extra high`, `pro`), pass it
through unchanged; everything else in the arguments is the image prompt.
Before sending, the run verifies the composer is on Chat with the GPT-6 entry
checked at that effort (the row reads `Pro` at `pro`) and the image tool
switched on — otherwise (a list without GPT-6 included) it stops with exit 2
and nothing is sent. ChatGPT then makes the image on its own image model, not
GPT-6 (at `medium` the server recorded thinking effort `standard`).

- `--attach ref.png` attaches a reference image to edit or restyle.
- `--continue` edits the previous image in its thread ("make the sky darker");
  `--resume <handle>` picks another thread by the handle from its trailer.
- `--out PATH` names the first image (an existing directory gets a timestamped
  name inside it); the suffix always follows the image's real type. Further
  images get `-2`, `-3`, … Default: `~/.chatgpt/out/<timestamp>.png`.

Invoke with `run_in_background`; a generation takes a minute or more. stdout is
one saved image path per line, then any text ChatGPT wrote with the image
(`--quiet` drops it), then the `Thread <handle> · <url>` trailer. Exit codes:
0 = success, 1 = failure — including a reply without an image (the error quotes
ChatGPT's text, e.g. a question back: answer it with `--continue`),
2 = model or image-tool verification failed (nothing sent), 3 = response
timeout, 4 = lock timeout, 5 = ChatGPT rate limit.
