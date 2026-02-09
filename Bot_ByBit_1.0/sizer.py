import math
from typing import Optional
from instruments_info import InstrumentRules

def floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def qty_from_usdt(usdt: float, price: float, rules: Optional[InstrumentRules]) -> float:
    if price <= 0 or usdt <= 0:
        return 0.0

    qty = usdt / price

    if rules:
        qty = floor_to_step(qty, rules.qty_step)
        if rules.min_qty > 0 and qty < rules.min_qty:
            return 0.0
        if rules.min_notional > 0 and qty * price < rules.min_notional:
            return 0.0

    return float(qty) if qty > 0 else 0.0
