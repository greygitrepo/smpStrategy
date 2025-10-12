#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
PID_FILE="$PROJECT_DIR/bot.pid"
MAIN_CMD="python3 $PROJECT_DIR/src/run/main.py"

kill_safely () {
  local pid="$1"
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "🛑 종료 시도: $pid"
    kill "$pid" || true
    sleep 1
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "⚠️ 정상 종료 실패, 강제 종료"
      kill -9 "$pid" || true
    fi
  fi
}

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  kill_safely "$PID"
  rm -f "$PID_FILE"
  echo "✅ Bot 중지 완료."
else
  # PID 파일이 없어도 커맨드로 탐색해서 중지
  if pgrep -f "$MAIN_CMD" > /dev/null 2>&1; then
    for p in $(pgrep -f "$MAIN_CMD"); do
      kill_safely "$p"
    done
    echo "✅ Bot 중지 완료. (PID 파일 없이 탐색)"
  else
    echo "❌ 실행 중인 Bot를 찾지 못했습니다."
  fi
fi
