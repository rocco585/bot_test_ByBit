import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

class EmaRsiStrategy:
    def __init__(self, ema_fast=20, ema_slow=50, ema_trend=200, rsi_period=14):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period

    def signal(self, candles: pd.DataFrame) -> str:
        close = candles["close"].astype(float)
        ef = ema(close, self.ema_fast)
        es = ema(close, self.ema_slow)
        et = ema(close, self.ema_trend)
        rv = rsi(close, self.rsi_period)

        if len(close) < self.ema_trend + 5:
            return "HOLD"

        trend_up = close.iloc[-1] > et.iloc[-1]
        trend_dn = close.iloc[-1] < et.iloc[-1]

        cross_up = ef.iloc[-2] <= es.iloc[-2] and ef.iloc[-1] > es.iloc[-1]
        cross_dn = ef.iloc[-2] >= es.iloc[-2] and ef.iloc[-1] < es.iloc[-1]

        if trend_up and cross_up and rv.iloc[-1] > 50:
            return "BUY"
        if trend_dn and cross_dn and rv.iloc[-1] < 50:
            return "SELL"
        return "HOLD"
