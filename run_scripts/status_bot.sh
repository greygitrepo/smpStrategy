#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
PID_FILE="$PROJECT_DIR/bot.pid"
LOG_DIR="$PROJECT_DIR/logs"
MAIN_CMD="python3 $PROJECT_DIR/src/run/main.py"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ Bot 실행 중 (PID: $PID)"
    exit 0
  else
    # PID 파일은 있으나 죽어있음 → 보조 체크(커맨드로 탐색)
    if pgrep -f "$MAIN_CMD" > /dev/null 2>&1; then
      REAL_PID=$(pgrep -f "$MAIN_CMD" | head -n1)
      echo "$REAL_PID" > "$PID_FILE"
      echo "✅ Bot 실행 중 (PID 업데이트: $REAL_PID)"
      exit 0
    fi
    echo "⚠️ bot.pid는 있지만 프로세스가 없습니다. (비정상 종료)"
    exit 1
  fi
else
  # PID 파일이 없을 때도 커맨드로 탐색
  if pgrep -f "$MAIN_CMD" > /dev/null 2>&1; then
    REAL_PID=$(pgrep -f "$MAIN_CMD" | head -n1)
    echo "$REAL_PID" > "$PID_FILE"
    echo "✅ Bot 실행 중 (PID 복구: $REAL_PID)"
    exit 0
  fi
  echo "❌ Bot가 실행 중이 아닙니다."
  exit 1
fi
