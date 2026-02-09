from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict

def _day_key_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

@dataclass
class TradeGuards:
    cooldown_sec: int = 180
    max_trades_per_day: int = 10
    max_loss_per_day_usdt: float = 50.0
    enable_loss_guard: bool = True

    last_trade_ts: Dict[str, float] = field(default_factory=dict)
    day_key: str = field(default_factory=_day_key_utc)
    trades_today: int = 0
    realised_pnl_today: float = 0.0

    def reset_if_new_day(self):
        dk = _day_key_utc()
        if dk != self.day_key:
            self.day_key = dk
            self.trades_today = 0
            self.realised_pnl_today = 0.0
            self.last_trade_ts = {}

    def can_open(self, symbol: str, now_ts: float):
        self.reset_if_new_day()

        if self.trades_today >= self.max_trades_per_day:
            return False, "Max trades/day reached"

        lt = self.last_trade_ts.get(symbol)
        if lt is not None and (now_ts - lt) < self.cooldown_sec:
            left = int(self.cooldown_sec - (now_ts - lt))
            return False, f"Cooldown active ({left}s left)"

        if self.enable_loss_guard and self.realised_pnl_today <= -abs(self.max_loss_per_day_usdt):
            return False, "Daily loss limit hit (STOP DAY)"

        return True, "OK"

    def on_open(self, symbol: str, now_ts: float):
        self.reset_if_new_day()
        self.trades_today += 1
        self.last_trade_ts[symbol] = now_ts

    def on_realised_pnl(self, pnl: float):
        self.reset_if_new_day()
        self.realised_pnl_today += float(pnl or 0.0)
