#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
PID_FILE="$PROJECT_DIR/bot.pid"
LOG_DIR="$PROJECT_DIR/logs"
TELEGRAM_PID_FILE="$PROJECT_DIR/telegram_bot.pid"
MAIN_CMD="python3 $PROJECT_DIR/src/run/main.py"
TELEGRAM_CMD="python3 $PROJECT_DIR/telegram_bot/bot.py"
TELEGRAM_LOG="$LOG_DIR/telegram_bot.log"

mkdir -p "$LOG_DIR"

# 이미 실행 중이면 중복 실행 방지
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
  echo "⚠️ 이미 실행 중입니다. PID: $(cat "$PID_FILE")"
  exit 0
fi

# 기존 PID 파일이 있지만 프로세스가 없다면 정리
if [ -f "$PID_FILE" ] && ! ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
  rm -f "$PID_FILE"
fi

# 백그라운드 실행
nohup $MAIN_CMD > "$LOG_DIR/bot.log" 2>&1 &

echo $! > "$PID_FILE"

echo "✅ Bot 실행됨. PID: $(cat "$PID_FILE")"
echo "📄 로그: $LOG_DIR/bot.log"

start_telegram_bot () {
  if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
    echo "⚠️ TELEGRAM_BOT_TOKEN이 설정되지 않아 Telegram bot을 시작하지 않습니다."
    return
  fi

  if [ -f "$TELEGRAM_PID_FILE" ] && ps -p "$(cat "$TELEGRAM_PID_FILE")" > /dev/null 2>&1; then
    echo "ℹ️ Telegram bot이 이미 실행 중입니다. PID: $(cat "$TELEGRAM_PID_FILE")"
    return
  fi

  if [ -f "$TELEGRAM_PID_FILE" ] && ! ps -p "$(cat "$TELEGRAM_PID_FILE")" > /dev/null 2>&1; then
    rm -f "$TELEGRAM_PID_FILE"
  fi

  local existing=""
  local first_pid=""
  existing=$(pgrep -f "$TELEGRAM_CMD" || true)
  if [ -n "$existing" ]; then
    first_pid=$(echo "$existing" | head -n1)
    echo "$first_pid" > "$TELEGRAM_PID_FILE"
    echo "ℹ️ Telegram bot 프로세스가 이미 실행 중입니다. PID: $first_pid"
    return
  fi

  nohup $TELEGRAM_CMD > "$TELEGRAM_LOG" 2>&1 &
  echo $! > "$TELEGRAM_PID_FILE"
  echo "🤖 Telegram bot 실행됨. PID: $(cat "$TELEGRAM_PID_FILE")"
  echo "📄 Telegram 로그: $TELEGRAM_LOG"
}

start_telegram_bot
