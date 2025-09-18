# smpStrategy

- Run `python -m src.config.account_config` to create or update `config/accountConfig.ini` with your Bybit API credentials.
- Once saved, the REST (`BybitV5Client`) and WebSocket (`BybitPrivateWS`) clients automatically consume the config file when explicit credentials are not provided.
