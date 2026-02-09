import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class PaperPosition:
    in_pos: bool = False
    trade_id: str = ""
    side: str = ""
    leverage: int = 10
    qty: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    margin_used: float = 0.0


@dataclass
class PendingOrder:
    order_id: str
    symbol: str
    side: str
    leverage: int
    qty: float
    price: float
    sl: float
    tp: float
    created_ts_ms: int


class PaperBroker:
    def __init__(self, starting_equity: float = 1000.0, taker_fee_rate: float = 0.0006):
        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)
        self.taker_fee_rate = float(taker_fee_rate)

        self.pos: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, List[PendingOrder]] = {}

        self.equity_curve: List[Dict[str, float]] = []  # ts_ms, equity

    # ===== helpers =====

    def snapshot_equity(self):
        self.equity_curve.append({
            "ts_ms": int(time.time() * 1000),
            "equity": float(self.equity)
        })

    def available_equity(self) -> float:
        used = sum(p.margin_used for p in self.pos.values() if p.in_pos)
        return self.equity - used

    def _fee(self, qty: float, price: float) -> float:
        return abs(qty * price) * self.taker_fee_rate

    def _id(self) -> str:
        return uuid.uuid4().hex[:12]

    # ===== views =====

    def get_position_view(self, symbol: str, last_price: float) -> dict:
        p = self.pos.get(symbol, PaperPosition())

        upl = 0.0
        if p.in_pos:
            upl = (last_price - p.entry_price) * p.qty if p.side == "Buy" else (p.entry_price - last_price) * p.qty

        return {
            "symbol": symbol,
            "in_pos": p.in_pos,
            "side": p.side,
            "leverage": p.leverage,
            "qty": p.qty,
            "entry": p.entry_price,
            "sl": p.sl,
            "tp": p.tp,
            "upl": upl,
            "equity": self.equity,
            "available": self.available_equity()
        }

    # ===== orders =====

    def place_limit_entry(self, symbol: str, side: str, leverage: int, qty: float, limit_price: float, sl: float, tp: float):
        o = PendingOrder(
            order_id=self._id(),
            symbol=symbol,
            side=side,
            leverage=leverage,
            qty=qty,
            price=limit_price,
            sl=sl,
            tp=tp,
            created_ts_ms=int(time.time() * 1000)
        )
        self.orders.setdefault(symbol, []).append(o)
        return o

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        ids = []
        if symbol:
            ids = [o.order_id for o in self.orders.get(symbol, [])]
            self.orders[symbol] = []
        else:
            for lst in self.orders.values():
                ids.extend(o.order_id for o in lst)
            self.orders = {}
        return ids

    def try_fill_orders(self, symbol: str, last_price: float) -> Optional[Dict[str, Any]]:
        if symbol not in self.orders:
            return None

        if self.pos.get(symbol, PaperPosition()).in_pos:
            self.orders[symbol] = []
            return None

        for i, o in enumerate(self.orders[symbol]):
            ok = (last_price <= o.price) if o.side == "Buy" else (last_price >= o.price)
            if ok:
                self.orders[symbol].pop(i)
                return self.open_market(symbol, o.side, o.leverage, o.qty, o.price, o.sl, o.tp, "LIMIT_FILLED")
        return None

    # ===== positions =====

    def open_market(self, symbol, side, leverage, qty, price, sl, tp, reason):
        notional = qty * price
        margin = notional / leverage
        if self.available_equity() < margin:
            raise RuntimeError("Недостаточно маржи")

        fee = self._fee(qty, price)
        self.equity -= fee

        trade_id = self._id()
        self.pos[symbol] = PaperPosition(
            True, trade_id, side, leverage, qty, price, sl, tp, margin
        )

        self.snapshot_equity()

        return {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "leverage": leverage,
            "qty": qty,
            "notional": notional,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "open_ts_ms": int(time.time() * 1000),
            "fee_usdt": fee
        }

    def close_market(self, symbol, price, reason):
        p = self.pos[symbol]
        pnl = (price - p.entry_price) * p.qty if p.side == "Buy" else (p.entry_price - price) * p.qty
        fee = self._fee(p.qty, price)

        self.equity += pnl
        self.equity -= fee
        self.pos[symbol] = PaperPosition()

        self.snapshot_equity()

        return {
            "trade_id": p.trade_id,
            "close_ts_ms": int(time.time() * 1000),
            "close_price": price,
            "pnl_usdt": pnl,
            "fee_usdt": fee,
            "close_reason": reason
        }

    def check_sl_tp(self, symbol, last_price):
        p = self.pos.get(symbol, PaperPosition())
        if not p.in_pos:
            return None

        if p.side == "Buy":
            if last_price <= p.sl:
                return self.close_market(symbol, p.sl, "SL")
            if last_price >= p.tp:
                return self.close_market(symbol, p.tp, "TP")
        else:
            if last_price >= p.sl:
                return self.close_market(symbol, p.sl, "SL")
            if last_price <= p.tp:
                return self.close_market(symbol, p.tp, "TP")
        return None
