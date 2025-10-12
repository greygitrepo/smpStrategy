#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
PID_FILE="$PROJECT_DIR/bot.pid"
LOG_DIR="$PROJECT_DIR/logs"
MAIN_CMD="python3 $PROJECT_DIR/src/run/main.py"

mkdir -p "$LOG_DIR"

# 이미 실행 중이면 중복 실행 방지
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
  echo "⚠️ 이미 실행 중입니다. PID: $(cat "$PID_FILE")"
  exit 0
fi

# 백그라운드 실행
nohup $MAIN_CMD > "$LOG_DIR/bot.log" 2>&1 &

echo $! > "$PID_FILE"

echo "✅ Bot 실행됨. PID: $(cat "$PID_FILE")"
echo "📄 로그: $LOG_DIR/bot.log"
