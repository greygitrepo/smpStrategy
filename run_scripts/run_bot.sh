#!/bin/bash

# 프로젝트 디렉토리로 이동
cd ~/smpStrategy

# 가상환경 활성화
source venv/bin/activate

# 로그 디렉토리 없으면 생성
mkdir -p logs

# nohup으로 실행 (백그라운드)
nohup python3 src/run/main.py > logs/bot.log 2>&1 &

# PID 저장
echo $! > bot.pid

echo "✅ Bot 실행됨. PID: $(cat bot.pid)"
echo "📄 로그: logs/bot.log"
