import time
import uuid
import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List, Tuple

import pandas as pd

from bybit_gateway import BybitGateway, BybitConfig
from trade_store import TradeStore
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("BOT")


# =========================
# Helpers: rounding
# =========================

def round_step_floor(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return math.floor(float(x) / float(step)) * float(step)


def round_price(x: float, tick: float) -> float:
    # Для лимиток безопаснее "вниз" по шагу
    return round_step_floor(float(x), float(tick))


def round_qty(x: float, step: float) -> float:
    return round_step_floor(float(x), float(step))


# =========================
# Indicators
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def strategy_signal(df: pd.DataFrame) -> str:
    """
    Консервативная стратегия:
      - EMA fast/slow
      - RSI фильтр
    Возвращает: BUY / SELL / HOLD
    """
    if df is None or df.empty or len(df) < 120:
        return "HOLD"

    close = df["close"]
    fast = ema(close, 20)
    slow = ema(close, 50)
    r = rsi(close, 14)

    # последние значения
    f0, f1 = fast.iloc[-2], fast.iloc[-1]
    s0, s1 = slow.iloc[-2], slow.iloc[-1]
    r1 = float(r.iloc[-1])

    cross_up = (f0 <= s0) and (f1 > s1)
    cross_dn = (f0 >= s0) and (f1 < s1)

    # фильтр по RSI (можно потом подкрутить)
    if cross_up and r1 < 70:
        return "BUY"
    if cross_dn and r1 > 30:
        return "SELL"
    return "HOLD"


def sl_tp_from_atr(side: str, entry: float, atr_val: float, k_sl: float = 1.5, rr: float = 1.8) -> Tuple[float, float]:
    """
    side: "Buy" / "Sell"
    """
    entry = float(entry)
    atr_val = float(max(atr_val, 1e-9))
    sl_dist = k_sl * atr_val
    tp_dist = rr * sl_dist

    if side == "Buy":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    return float(sl), float(tp)


def qty_from_usdt(per_trade_usdt: float, price: float, leverage: int) -> float:
    """
    Линейный контракт: notional = qty * price
    Мы задаём сколько USDT выделяем как маржу на сделку (per_trade_usdt),
    значит notional примерно per_trade_usdt * leverage.
    """
    price = float(max(price, 1e-9))
    notional = float(per_trade_usdt) * int(leverage)
    return notional / price


# =========================
# Instruments info wrapper
# =========================

class InstrumentsInfo:
    def __init__(self, gateway: BybitGateway):
        self.gateway = gateway
        self._rules_cache: Dict[str, Dict[str, float]] = {}

    def get_rules(self, symbol: str) -> Dict[str, float]:
        """
        Ожидаем от gateway.get_instruments_info:
            {"qty_step":..., "min_qty":..., "price_step":...}
        """
        if symbol in self._rules_cache:
            return self._rules_cache[symbol]

        info = self.gateway.get_instruments_info(symbol=symbol, category="linear", use_mainnet_public=True)
        rules = {
            "qty_step": float(info.get("qty_step", 0.001)),
            "min_qty": float(info.get("min_qty", 0.001)),
            "price_step": float(info.get("price_step", 0.01)),
        }
        self._rules_cache[symbol] = rules
        return rules


# =========================
# Paper exchange (LIMIT)
# =========================

@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str          # "Buy" / "Sell"
    leverage: int
    qty: float
    price: float       # LIMIT price
    sl: Optional[float]
    tp: Optional[float]
    created_ts_ms: int
    status: str = "OPEN"    # OPEN/FILLED/CANCELED
    filled_ts_ms: Optional[int] = None


@dataclass
class PaperPosition:
    trade_id: str
    symbol: str
    side: str          # "Buy"/"Sell"
    leverage: int
    qty: float
    entry_price: float
    sl: Optional[float]
    tp: Optional[float]
    open_ts_ms: int
    fee_usdt: float = 0.0


class PaperBroker:
    """
    Очень простой paper-движок:
      - держит отложенные LIMIT ордера
      - при достижении цены -> исполняет и открывает позицию
      - при достижении TP/SL -> закрывает позицию
      - equity считается как: старт + realized pnl
    """

    def __init__(self, starting_equity: float, session_usdt: float):
        self.starting_equity = float(starting_equity)
        self.session_usdt = float(session_usdt)
        self.realized_pnl = 0.0
        self.open_orders: Dict[str, PaperOrder] = {}   # order_id -> order
        self.position_by_symbol: Dict[str, PaperPosition] = {}  # symbol -> position
        self.used_margin = 0.0  # суммарно выделенная "маржа" на открытые позиции (по per_trade_usdt)

    def equity(self) -> float:
        return self.starting_equity + self.realized_pnl

    def available(self) -> float:
        return max(0.0, self.session_usdt - self.used_margin)

    def place_limit_entry(self, symbol: str, side: str, leverage: int, qty: float, limit_price: float,
                          sl: Optional[float], tp: Optional[float], per_trade_usdt: float) -> PaperOrder:
        if symbol in self.position_by_symbol:
            raise RuntimeError(f"Position already open for {symbol}")

        if self.available() < float(per_trade_usdt):
            raise RuntimeError(f"Not enough session USDT: avail={self.available():.2f} need={per_trade_usdt:.2f}")

        oid = uuid.uuid4().hex[:12]
        od = PaperOrder(
            order_id=oid,
            symbol=symbol,
            side=side,
            leverage=int(leverage),
            qty=float(qty),
            price=float(limit_price),
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            created_ts_ms=int(time.time() * 1000),
            status="OPEN"
        )
        self.open_orders[oid] = od
        return od

    def cancel_all(self):
        for od in self.open_orders.values():
            od.status = "CANCELED"

    def on_price(self, symbol: str, last_price: float, ts_ms: int, per_trade_usdt: float) -> Dict[str, Any]:
        """
        Обрабатывает:
          - исполнение лимиток
          - закрытие позиции по TP/SL
        Возвращает события для TradeStore/UI
        """
        events: Dict[str, Any] = {"filled_orders": [], "closed_trades": [], "opened_trades": []}
        lp = float(last_price)

        # 1) fill orders
        for oid, od in list(self.open_orders.items()):
            if od.symbol != symbol or od.status != "OPEN":
                continue

            # Buy limit: исполняется если last_price <= limit_price
            # Sell limit: исполняется если last_price >= limit_price
            can_fill = (od.side == "Buy" and lp <= od.price) or (od.side == "Sell" and lp >= od.price)
            if not can_fill:
                continue

            # если позиция уже появилась (вдруг), пропускаем
            if symbol in self.position_by_symbol:
                od.status = "CANCELED"
                continue

            od.status = "FILLED"
            od.filled_ts_ms = int(ts_ms)
            events["filled_orders"].append(od)

            # open position
            tid = uuid.uuid4().hex[:12]
            pos = PaperPosition(
                trade_id=tid,
                symbol=od.symbol,
                side=od.side,
                leverage=od.leverage,
                qty=od.qty,
                entry_price=od.price,  # считаем, что исполнилось ровно по лимитной
                sl=od.sl,
                tp=od.tp,
                open_ts_ms=int(ts_ms),
                fee_usdt=0.0
            )
            self.position_by_symbol[symbol] = pos
            self.used_margin += float(per_trade_usdt)
            events["opened_trades"].append(pos)

        # 2) manage open position TP/SL
        pos = self.position_by_symbol.get(symbol)
        if pos:
            close_reason = None
            close_price = None

            if pos.side == "Buy":
                if pos.tp is not None and lp >= float(pos.tp):
                    close_reason = "TP"
                    close_price = float(pos.tp)
                elif pos.sl is not None and lp <= float(pos.sl):
                    close_reason = "SL"
                    close_price = float(pos.sl)
            else:
                if pos.tp is not None and lp <= float(pos.tp):
                    close_reason = "TP"
                    close_price = float(pos.tp)
                elif pos.sl is not None and lp >= float(pos.sl):
                    close_reason = "SL"
                    close_price = float(pos.sl)

            if close_reason and close_price is not None:
                pnl = self._pnl_usdt(pos, close_price)
                self.realized_pnl += pnl
                self.used_margin = max(0.0, self.used_margin - float(per_trade_usdt))
                events["closed_trades"].append((pos, int(ts_ms), float(close_price), float(pnl), close_reason))
                del self.position_by_symbol[symbol]

        return events

    @staticmethod
    def _pnl_usdt(pos: PaperPosition, close_price: float) -> float:
        """
        PnL в USDT по линейке:
          pnl = (close - entry) * qty   для Buy
          pnl = (entry - close) * qty   для Sell
        """
        cp = float(close_price)
        if pos.side == "Buy":
            return (cp - pos.entry_price) * pos.qty
        else:
            return (pos.entry_price - cp) * pos.qty


# =========================
# Settings + Engine
# =========================

@dataclass
class BotSettings:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # для будущего live
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "15"  # "5","15","60","240"
    leverage: int = 10

    session_usdt: float = 1000.0   # сколько USDT выделяем на сессию (paper)
    per_trade_usdt: float = 50.0   # маржа на одну сделку

    limit_offset_pct: float = 0.05  # насколько отступаем от last_price для лимитки (%)

    poll_sec: float = 10.0
    safe_stop_errors: int = 20

    run_id: str = ""


class BotEngine:
    def __init__(self, settings: BotSettings, store: TradeStore, on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.settings = settings
        self.store = store
        self.on_event = on_event

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        cfg = BybitConfig(api_key=settings.api_key, api_secret=settings.api_secret, testnet=settings.testnet)
        self.gateway = BybitGateway(cfg)
        self.instruments = InstrumentsInfo(self.gateway)

        self.paper = PaperBroker(starting_equity=settings.session_usdt, session_usdt=settings.session_usdt)
        self._equity_points: List[Dict[str, float]] = []

        self._errors_total = 0
        self._errors_by_symbol: Dict[str, int] = {}

    # ---------- events ----------
    def emit(self, ev: Dict[str, Any]):
        # 1) в консоль
        t = ev.get("type", "EVENT")
        sym = ev.get("symbol", "")
        msg = ev.get("msg", "")
        if t == "ERROR":
            logger.error("%s | %s | %s", t, sym, msg)
        elif t in ("WARN", "WARNING"):
            logger.warning("%s | %s | %s", t, sym, msg)
        else:
            logger.info("%s | %s | %s", t, sym, msg)

        # 2) в SQLite (как было)
        try:
            self.store.add_event(
                run_id=self.settings.run_id,
                type_=t,
                symbol=sym,
                message=msg,
                payload=ev
            )
        except Exception:
            pass

        # 3) callback в UI (как было)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    # ---------- lifecycle ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self.settings.run_id = uuid.uuid4().hex[:12]
        self._equity_points = []

        self.emit({"type": "BOT", "msg": f"Запущен paper-режим (LIMIT) на реальных свечах. run_id={self.settings.run_id}"})

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.emit({"type": "BOT", "msg": "Остановка..."})
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

        # сохранить equity curve
        try:
            self.store.save_equity_curve(self.settings.run_id, self._equity_points)
        except Exception:
            pass

        self.emit({"type": "BOT", "msg": "Остановлен"})

    # ---------- main loop ----------
    def _run_loop(self):
        while not self._stop.is_set():
            for sym in self.settings.symbols:
                if self._stop.is_set():
                    break

                try:
                    self._process_symbol(sym)
                    self._errors_by_symbol[sym] = 0
                except Exception as e:
                    self._errors_total += 1
                    self._errors_by_symbol[sym] = self._errors_by_symbol.get(sym, 0) + 1
                    self.emit({"type": "ERROR", "symbol": sym, "msg": str(e)})

                    if self._errors_by_symbol[sym] >= self.settings.safe_stop_errors:
                        self.emit({"type": "BOT", "msg": f"Safe-stop: слишком много ошибок по {sym}. Останавливаю."})
                        self._stop.set()
                        break

            time.sleep(float(self.settings.poll_sec))

    def _process_symbol(self, symbol: str):
        # 1) свечи (реальные) — mainnet public
        df = self.gateway.get_kline_df(symbol=symbol, interval=str(self.settings.timeframe), limit=300, use_mainnet_public=True, category="linear")
        if df is None or df.empty or len(df) < 120:
            self.emit({"type": "WARN", "symbol": symbol, "msg": "Недостаточно свечей для сигнала"})
            return

        last_price = float(df["close"].iloc[-1])
        ts_ms = int(df["ts_ms"].iloc[-1])

        # 2) signal
        sig = strategy_signal(df)
        self.emit({"type": "SIGNAL", "symbol": symbol, "msg": sig})

        # 3) rules
        rules = self.instruments.get_rules(symbol)
        qty_step = float(rules.get("qty_step", 0.001))
        min_qty = float(rules.get("min_qty", 0.001))
        tick = float(rules.get("price_step", 0.01))

        # 4) обновить paper по цене: fill лимиток / закрытие по TP/SL
        evs = self.paper.on_price(symbol, last_price, ts_ms, per_trade_usdt=float(self.settings.per_trade_usdt))

        # order filled -> store order status + open trade record
        for od in evs.get("filled_orders", []):
            try:
                self.store.update_order_filled(od.order_id, od.filled_ts_ms or ts_ms, note="filled (paper)")
            except Exception:
                pass
            self.emit({"type": "ORDER", "symbol": symbol, "msg": f"Ордер исполнен: {od.side} qty={od.qty} @ {od.price}"})

        for pos in evs.get("opened_trades", []):
            # записываем открытие сделки
            self.store.upsert_trade_open(self.settings.run_id, {
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "leverage": pos.leverage,
                "side": pos.side,
                "qty": pos.qty,
                "notional": float(pos.qty) * float(pos.entry_price),
                "entry_price": pos.entry_price,
                "sl": pos.sl,
                "tp": pos.tp,
                "open_ts_ms": pos.open_ts_ms,
                "fee_usdt": pos.fee_usdt
            })
            self.emit({"type": "TRADE_OPEN", "symbol": symbol, "msg": f"Открыта позиция: {pos.side} qty={pos.qty} @ {pos.entry_price}"})

        for (pos, close_ts, close_price, pnl, reason) in evs.get("closed_trades", []):
            self.store.update_trade_close(pos.trade_id, close_ts, close_price, pnl_usdt=pnl, fee_usdt=0.0, close_reason=reason)
            self.emit({"type": "TRADE_CLOSE", "symbol": symbol, "msg": f"Закрыто: {reason} pnl={pnl:.4f} USDT @ {close_price}"})

        # 5) equity point
        eq = float(self.paper.equity())
        av = float(self.paper.available())
        self._equity_points.append({"ts_ms": ts_ms, "equity": eq})
        self.emit({"type": "POSITION_VIEW", "symbol": symbol, "msg": f"equity={eq:.2f} avail={av:.2f} upl=0.0000"})

        # 6) если позиции нет и нет открытого ордера на символ — можно поставить лимитку по сигналу
        if symbol in self.paper.position_by_symbol:
            return

        # если уже есть открытая лимитка по символу — не дублируем
        for od in self.paper.open_orders.values():
            if od.symbol == symbol and od.status == "OPEN":
                return

        if sig not in ("BUY", "SELL"):
            return

        side = "Buy" if sig == "BUY" else "Sell"

        # ATR
        atr_val = float(atr(df, period=14).iloc[-1])

        # qty -> from USDT+leverage -> round by qty_step
        raw_qty = qty_from_usdt(float(self.settings.per_trade_usdt), last_price, int(self.settings.leverage))
        qty = round_qty(raw_qty, qty_step)

        if qty <= 0 or qty < min_qty:
            self.emit({"type": "WARN", "symbol": symbol,
                       "msg": f"Слишком маленький объём для {symbol}: qty={qty} < min={min_qty}. Увеличь USDT на сделку."})
            return

        # SL/TP -> round by tick
        sl, tp = sl_tp_from_atr(side, last_price, atr_val, k_sl=1.5, rr=1.8)
        sl = round_price(sl, tick)
        tp = round_price(tp, tick)

        # limit price offset -> round by tick
        off = float(self.settings.limit_offset_pct) / 100.0
        raw_limit = last_price * (1.0 - off) if side == "Buy" else last_price * (1.0 + off)
        limit_price = round_price(raw_limit, tick)

        # place
        od = self.paper.place_limit_entry(
            symbol=symbol,
            side=side,
            leverage=int(self.settings.leverage),
            qty=float(qty),
            limit_price=float(limit_price),
            sl=float(sl),
            tp=float(tp),
            per_trade_usdt=float(self.settings.per_trade_usdt)
        )

        # store order row
        self.store.upsert_order_open(self.settings.run_id, {
            "order_id": od.order_id,
            "symbol": od.symbol,
            "leverage": od.leverage,
            "side": od.side,
            "qty": od.qty,
            "order_type": "LIMIT_ENTRY",
            "price": od.price,
            "sl": od.sl,
            "tp": od.tp,
            "created_ts_ms": od.created_ts_ms,
            "note": "paper limit entry"
        })

        self.emit({
            "type": "ORDER",
            "symbol": symbol,
            "msg": f"Лимитный вход: {side} qty={qty} @ {limit_price} (SL={sl}, TP={tp})"
        })
