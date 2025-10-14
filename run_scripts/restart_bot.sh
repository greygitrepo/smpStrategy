#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/smpStrategy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$PROJECT_DIR/logs"
LEGACY_LOG_DIR="$PROJECT_DIR/log"
EXCLUSIONS_FILE="$PROJECT_DIR/config/newListingStrategy.exclusions.json"

echo "♻️ Bot 재시작을 준비합니다."

if ! "$SCRIPT_DIR/stop_bot.sh"; then
  echo "⚠️ Bot 중지 과정에서 문제가 있었지만 계속 진행합니다."
fi

cleanup_dir () {
  local target_dir="$1"
  if [ -d "$target_dir" ]; then
    echo "🧹 디렉터리 삭제: $target_dir"
    rm -rf "$target_dir"
  else
    echo "ℹ️ 디렉터리 없음: $target_dir"
  fi
}

cleanup_dir "$LOGS_DIR"
cleanup_dir "$LEGACY_LOG_DIR"

if [ -f "$EXCLUSIONS_FILE" ]; then
  echo "🗑️ 파일 삭제: $EXCLUSIONS_FILE"
  rm -f "$EXCLUSIONS_FILE"
else
  echo "ℹ️ 파일 없음: $EXCLUSIONS_FILE"
fi

"$SCRIPT_DIR/run_bot.sh"

echo "✅ Bot 재시작 완료."
