from dataclasses import dataclass
from typing import Optional

@dataclass
class InstrumentRules:
    symbol: str
    qty_step: float
    min_qty: float
    price_tick: float
    min_notional: float

def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

class InstrumentsInfo:
    def __init__(self, gateway):
        self.gateway = gateway

    def get_rules(self, symbol: str) -> Optional[InstrumentRules]:
        info = self.gateway.get_instruments_info(symbol)
        if not info:
            return None

        lot = info.get("lotSizeFilter", {}) or {}
        price = info.get("priceFilter", {}) or {}

        qty_step = _to_float(lot.get("qtyStep") or 0.001, 0.001)
        min_qty = _to_float(lot.get("minOrderQty") or 0.0, 0.0)
        tick = _to_float(price.get("tickSize") or 0.01, 0.01)

        min_notional = _to_float(lot.get("minOrderAmt") or lot.get("minNotional") or 0.0, 0.0)

        return InstrumentRules(
            symbol=symbol,
            qty_step=max(qty_step, 1e-12),
            min_qty=max(min_qty, 0.0),
            price_tick=max(tick, 1e-12),
            min_notional=max(min_notional, 0.0),
        )
