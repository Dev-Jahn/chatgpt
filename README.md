# chatgpt

Claude Code에서 로그인된 구독 ChatGPT Pro 브라우저로 프롬프트를 중립 전달하고 응답 Markdown을 회수하는 독립 플러그인이다. 리뷰 프레이밍, 템플릿, repomix 패킹, connector 개입은 하지 않는다. 모델은 `GPT-5.6 Sol`로 고정 검증되며 기본 추론 강도는 `Pro`다.

## 설치

```bash
claude plugin marketplace add ~/workspace/chatgpt
claude plugin install chatgpt@chatgpt-local
```

직접 CLI로 자주 쓸 경우 선택적으로 링크한다.

```bash
ln -s ~/workspace/chatgpt/bin/chatgpt ~/.local/bin/chatgpt
```

## 사용

```bash
chatgpt "질문 텍스트"
chatgpt -f prompt.md
echo "질문 텍스트" | chatgpt -
chatgpt --effort pro --attach context.pdf --max-wait 7200 --out answer.md "질문"
```

stdout에는 응답 본문만 출력되고 진행·진단 로그는 stderr로 간다. `--quiet`은 stdout에 저장 경로만 출력한다. 기본 응답 경로는 `~/.chatgpt/out/<timestamp>.md`다. 배타 lock은 `~/.chatgpt/run.lock`이며 `CHATGPT_LOCK_WAIT`(기본 3600초) 동안 기다린다.

종료 코드는 0=회수 성공, 2=`GPT-5.6 Sol` 검증 실패(전송하지 않음), 3=응답 timeout, 4=lock timeout이다.

## 환경

- Linux, Bash, `flock`, `curl`, `ss`
- system `python3`와 Python 패키지 `playwright`
- `/usr/bin/google-chrome`, TurboVNC (`/opt/TurboVNC/bin/vncserver`), `openbox`, `websockify`, noVNC assets
- 기존 로그인 자산 `~/.insane-review/browser-profile`

CDP 9222의 기존 insane-review 스택이 살아 있으면 그대로 재사용한다. 아니면 빈 VNC display를 골라 Chrome CDP와 noVNC를 자동 기동하고, 스택 상태와 로그만 `~/.chatgpt/`에 기록한다. 로그인이나 connector 인증은 도구가 변경하지 않는다.
