#!/bin/bash

if [ -f bot.pid ]; then
    PID=$(cat bot.pid)
    echo "🛑 PID $PID 종료 시도..."
    kill $PID 2>/dev/null

    # 혹시 안 죽었으면 강제 종료
    sleep 1
    if ps -p $PID > /dev/null; then
        echo "⚠️ 정상 종료 안 됨. 강제 종료합니다."
        kill -9 $PID
    fi

    rm -f bot.pid
    echo "✅ Bot 중지 완료."
else
    echo "❌ bot.pid 파일이 없습니다. 이미 꺼져있을 수 있음."
fi
