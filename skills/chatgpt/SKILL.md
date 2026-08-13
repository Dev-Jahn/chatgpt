---
name: chatgpt
description: 수십 분+ 초고난도 분석/설계/조사/검증을 구독 ChatGPT Pro(GPT-5.6 Sol)에 위임. 프롬프트만 넘기면 됨 — 패킹/템플릿 불요, connector는 프롬프트에서 지시.
---

```bash
"${CLAUDE_PLUGIN_ROOT}/bin/chatgpt" "질문 텍스트"
"${CLAUDE_PLUGIN_ROOT}/bin/chatgpt" -f prompt.md
echo "질문 텍스트" | "${CLAUDE_PLUGIN_ROOT}/bin/chatgpt" -
```

장시간 작업이므로 `run_in_background`로 호출한다. 실행은 lock으로 직렬화된다. 종료 코드는 0=성공, 2=모델 검증 실패, 3=timeout, 4=lock timeout이다.

