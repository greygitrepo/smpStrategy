#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
PID_FILE="$PROJECT_DIR/bot.pid"
LOG_DIR="$PROJECT_DIR/logs"
BOT_LOG="$LOG_DIR/bot.log"
TELEGRAM_PID_FILE="$PROJECT_DIR/telegram_bot.pid"
MAIN_CMD="python3 $PROJECT_DIR/src/run/main.py"
TELEGRAM_CMD="python3 $PROJECT_DIR/telegram_bot/bot.py"
TELEGRAM_LOG="$LOG_DIR/telegram_bot.log"

BOT_CRON_SCHEDULE="0 0 * * *"
BOT_CRON_CMD="truncate -s 0 $BOT_LOG"
TELEGRAM_CRON_SCHEDULE="5 0 * * *"
TELEGRAM_CRON_CMD="truncate -s 0 $TELEGRAM_LOG"

ensure_log_truncate_cron () {
  if ! command -v crontab >/dev/null 2>&1; then
    echo "⚠️ crontab 명령을 찾을 수 없어 로그 정리 스케줄을 설정하지 못했습니다."
    return
  fi

  local cron_tmp
  local updated=0
  cron_tmp=$(mktemp)
  if ! crontab -l > "$cron_tmp" 2>/dev/null; then
    > "$cron_tmp"
  fi

  if ! grep -F -- "$BOT_CRON_CMD" "$cron_tmp" > /dev/null; then
    if [ -s "$cron_tmp" ]; then
      echo "" >> "$cron_tmp"
    fi
    echo "# smpStrategy log trim (bot.log)" >> "$cron_tmp"
    echo "$BOT_CRON_SCHEDULE $BOT_CRON_CMD" >> "$cron_tmp"
    updated=1
  fi

  if ! grep -F -- "$TELEGRAM_CRON_CMD" "$cron_tmp" > /dev/null; then
    if [ -s "$cron_tmp" ]; then
      echo "" >> "$cron_tmp"
    fi
    echo "# smpStrategy log trim (telegram_bot.log)" >> "$cron_tmp"
    echo "$TELEGRAM_CRON_SCHEDULE $TELEGRAM_CRON_CMD" >> "$cron_tmp"
    updated=1
  fi

  if [ "$updated" -eq 1 ]; then
    crontab "$cron_tmp"
    echo "🕒 로그 정리 cron job을 추가했습니다."
  else
    echo "ℹ️ 로그 정리 cron job이 이미 설정되어 있습니다."
  fi

  rm -f "$cron_tmp"
}

mkdir -p "$LOG_DIR"
ensure_log_truncate_cron

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
nohup $MAIN_CMD > "$BOT_LOG" 2>&1 &

echo $! > "$PID_FILE"

echo "✅ Bot 실행됨. PID: $(cat "$PID_FILE")"
echo "📄 로그: $BOT_LOG"

start_telegram_bot () {
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
