#!/bin/bash

if [ -f bot.pid ]; then
    PID=$(cat bot.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Bot 실행 중 (PID: $PID)"
    else
        echo "⚠️ bot.pid는 있지만 프로세스가 없습니다. (비정상 종료)"
    fi
else
    echo "❌ Bot가 실행 중이 아닙니다."
fi
