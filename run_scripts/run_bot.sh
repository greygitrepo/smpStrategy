#!/bin/bash

cd ~/smpStrategy

mkdir -p logs

nohup python3 src/run/main.py > logs/bot.log 2>&1 &

echo $! > bot.pid
echo "✅ Bot 실행됨. PID: $(cat bot.pid)"
echo "📄 로그: logs/bot.log"
