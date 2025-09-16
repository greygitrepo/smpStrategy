# smpStrategy

- Run `python -m src.exchange.account_config` to create or update `accountConfig.ini` with your Bybit API credentials.
- Once saved, the REST (`BybitV5Client`) and WebSocket (`BybitPrivateWS`) clients automatically consume the config file when explicit credentials are not provided.
