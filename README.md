# smpStrategy

- Run `python -m src.config.account_config` to create or update `config/accountConfig.ini` with your Bybit API credentials.
- Use `python -m src.run.main` (or run `src/run/main.py` via IDE F5) as the central entry point for the project.
- Once saved, the REST (`BybitV5Client`) and WebSocket (`BybitPrivateWS`) clients automatically consume the config file when explicit credentials are not provided.
