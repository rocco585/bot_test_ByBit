import pandas as pd

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

def sl_tp_from_atr(side: str, entry: float, atr_value: float, k_sl: float = 1.5, rr: float = 1.8):
    if entry <= 0 or atr_value <= 0:
        return (0.0, 0.0)

    sl_dist = atr_value * k_sl
    tp_dist = sl_dist * rr

    if side == "Buy":
        return (entry - sl_dist, entry + tp_dist)
    else:
        return (entry + sl_dist, entry - tp_dist)
