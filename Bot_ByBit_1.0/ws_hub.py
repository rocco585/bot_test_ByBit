import threading
from typing import Dict, Optional

from pybit.unified_trading import WebSocket

class WsHub:
    def __init__(self, api_key: str, api_secret: str, testnet: bool, symbols: list[str]):
        self.symbols = symbols
        self._lock = threading.Lock()
        self._prices: Dict[str, float] = {}

        self.ws_public = WebSocket(testnet=testnet, channel_type="linear", ping_interval=20, ping_timeout=10)
        # private подключаем, но не критично
        self.ws_private = WebSocket(
            testnet=testnet, channel_type="private",
            api_key=api_key, api_secret=api_secret
        )
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True

        for s in self.symbols:
            self.ws_public.ticker_stream(symbol=s, callback=self._on_ticker)

        # позиции можно добавить позже; часто зависит от версии pybit
        # try:
        #     self.ws_private.position_stream(callback=self._on_position)
        # except Exception:
        #     pass

    def stop(self):
        for ws in (self.ws_public, self.ws_private):
            try:
                ws.exit()
            except Exception:
                pass
        self._started = False

    def _on_ticker(self, msg: dict):
        try:
            data = msg.get("data") or {}
            sym = data.get("symbol")
            lp = data.get("lastPrice")
            if sym and lp is not None:
                with self._lock:
                    self._prices[sym] = float(lp)
        except Exception:
            pass

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)
