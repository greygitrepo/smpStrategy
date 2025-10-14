# smpStrategy

- Run `python -m src.config.account_config` to create or update `config/accountConfig.ini` with your Bybit API credentials.
- Use `python -m src.run.main` (or run `src/run/main.py` via IDE F5) as the central entry point for the project.
- Configure symbol discovery via `config/symbolStrategy.ini` (copy from `config/sampleSymbolStrategy.ini`) to choose between discovery strategies and limits.
- Control wallet polling via `config/wallet.ini` (copy from `config/sampleWallet.ini`) to adjust refresh intervals.
- Once saved, the REST (`BybitV5Client`) and WebSocket (`BybitPrivateWS`) clients automatically consume the config file when explicit credentials are not provided.

## Telegram Control

- Install requirements with `python -m pip install -r requirements.txt`.
- Set `TELEGRAM_BOT_TOKEN` (and optional `TELEGRAM_ALLOWED_CHAT_IDS="123456,987654"`) before starting the bot.
- Launch the bot with `python telegram_bot/bot.py` to enable `/run`, `/stop`, `/status`, `/restart`, and `/performance` commands.
- `/performance` reads `analytics/performance_snapshot.json` and sends a formatted summary of total and window metrics.
- `run_scripts/run_bot.sh`와 `run_scripts/stop_bot.sh`은 환경 변수에 토큰이 설정되어 있을 때 Telegram bot을 자동으로 시작/중지합니다.
