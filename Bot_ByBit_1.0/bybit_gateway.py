from dataclasses import dataclass
from typing import Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd


@dataclass
class BybitConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # для private (позже)


class BybitGateway:
    """
    Используем как market-data шлюз:
    - Свечи берём с Bybit mainnet public (реальные графики)
    - Инструменты (tickSize/qtyStep/minQty) тоже берём из Bybit public
    - Добавлены retries + пул соединений
    """

    def __init__(self, cfg: BybitConfig):
        self.cfg = cfg
        self._cache: Dict[str, Any] = {}

        self._http = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self._http.mount("https://", adapter)
        self._http.mount("http://", adapter)

    # ---------- helpers ----------

    @staticmethod
    def _bybit_base(use_mainnet_public: bool) -> str:
        # для public-данных: mainnet = реальные свечи, testnet = тестовые/могут быть пустыми
        return "https://api.bybit.com" if use_mainnet_public else "https://api-testnet.bybit.com"

    @staticmethod
    def _df_from_bybit_kline(rows: List[List[Any]]) -> pd.DataFrame:
        """
        Bybit v5 market/kline list:
        [
          [startTime, open, high, low, close, volume, turnover],
          ...
        ]
        """
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume", "turnover"])
        df["ts_ms"] = df["ts_ms"].astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype("float64")

        df = df.sort_values("ts_ms").reset_index(drop=True)
        return df[["ts_ms", "open", "high", "low", "close", "volume"]]

    # ---------- market data ----------

    def get_kline_df(
        self,
        symbol: str,
        interval: str,
        limit: int = 300,
        use_mainnet_public: bool = True,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Получить свечи.
        interval: "5","15","60","240"
        """
        base = self._bybit_base(use_mainnet_public=use_mainnet_public)
        url = f"{base}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval),
            "limit": int(min(max(limit, 1), 1000))
        }

        r = self._http.get(url, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()

        if j.get("retCode") != 0:
            raise RuntimeError(f"Bybit kline retCode={j.get('retCode')} retMsg={j.get('retMsg')}")

        rows = j.get("result", {}).get("list", [])
        return self._df_from_bybit_kline(rows)

    def get_instruments_info(self, symbol: str, category: str = "linear", use_mainnet_public: bool = True) -> dict:
        """
        Возвращает правила инструмента:
        qty_step, min_qty, price_step
        """
        cache_key = f"inst:{use_mainnet_public}:{category}:{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        base = self._bybit_base(use_mainnet_public=use_mainnet_public)
        url = f"{base}/v5/market/instruments-info"
        params = {"category": category, "symbol": symbol}

        r = self._http.get(url, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()

        if j.get("retCode") != 0:
            raise RuntimeError(f"Bybit instruments-info retCode={j.get('retCode')} retMsg={j.get('retMsg')}")

        lst = j.get("result", {}).get("list", [])
        if not lst:
            raise RuntimeError(f"Bybit instruments-info empty list for {symbol}")

        info = lst[0]
        lot = info.get("lotSizeFilter", {}) or {}
        price = info.get("priceFilter", {}) or {}

        rules = {
            "symbol": symbol,
            "qty_step": float(lot.get("qtyStep", "0.001")),
            "min_qty": float(lot.get("minOrderQty", lot.get("minTradingQty", "0.001"))),
            "price_step": float(price.get("tickSize", "0.01")),
        }

        self._cache[cache_key] = rules
        return rules
