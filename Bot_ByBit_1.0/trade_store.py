import sqlite3
import json
import time
from typing import Optional, Any, Dict, List, Tuple


class TradeStore:
    def __init__(self, path: str = "paper_trades.sqlite"):
        self.path = path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init(self):
        with self._conn() as con:
            # ===== events =====
            con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                symbol TEXT,
                type TEXT NOT NULL,
                message TEXT,
                payload_json TEXT
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, ts_ms)")

            # ===== trades =====
            # Одна сделка = одна строка: open+close.
            con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                leverage INTEGER NOT NULL,
                side TEXT NOT NULL,                 -- Buy/Sell
                qty REAL NOT NULL,
                notional REAL NOT NULL,

                entry_price REAL NOT NULL,
                sl REAL,
                tp REAL,

                open_ts_ms INTEGER NOT NULL,
                close_ts_ms INTEGER,
                close_price REAL,

                pnl_usdt REAL DEFAULT 0.0,
                fee_usdt REAL DEFAULT 0.0,

                status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN/CLOSED
                close_reason TEXT
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id, open_ts_ms)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_trades_run_status ON trades(run_id, status)")

            # ===== orders =====
            # LIMIT входы
            con.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                leverage INTEGER NOT NULL,
                side TEXT NOT NULL,                 -- Buy/Sell
                qty REAL NOT NULL,

                order_type TEXT NOT NULL,           -- LIMIT_ENTRY
                price REAL NOT NULL,                -- лимитная цена

                sl REAL,
                tp REAL,

                created_ts_ms INTEGER NOT NULL,
                filled_ts_ms INTEGER,
                canceled_ts_ms INTEGER,

                status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN/FILLED/CANCELED
                note TEXT
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id, created_ts_ms)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_orders_run_status ON orders(run_id, status)")

            # ===== equity =====
            # Кривая депозита (точки)
            con.execute("""
            CREATE TABLE IF NOT EXISTS equity (
                ts_ms INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                equity REAL NOT NULL
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_equity_run ON equity(run_id, ts_ms)")

    # ======================
    # EVENTS
    # ======================

    def add_event(self, run_id: str, type_: str, symbol: Optional[str], message: str, payload: Optional[Dict[str, Any]] = None):
        ts = int(time.time() * 1000)
        with self._conn() as con:
            con.execute(
                "INSERT INTO events(ts_ms, run_id, symbol, type, message, payload_json) VALUES(?,?,?,?,?,?)",
                (ts, run_id, symbol, type_ or "EVENT", message or "", json.dumps(payload or {}, ensure_ascii=False))
            )

    # ======================
    # TRADES
    # ======================

    def upsert_trade_open(self, run_id: str, trade: Dict[str, Any]):
        """
        trade must contain:
        trade_id, symbol, leverage, side, qty, notional, entry_price, sl, tp, open_ts_ms, fee_usdt
        """
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO trades(
                    trade_id, run_id, symbol, leverage, side, qty, notional,
                    entry_price, sl, tp,
                    open_ts_ms, close_ts_ms, close_price,
                    pnl_usdt, fee_usdt, status, close_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade["trade_id"], run_id, trade["symbol"],
                int(trade["leverage"]), trade["side"],
                float(trade["qty"]), float(trade["notional"]),
                float(trade["entry_price"]),
                float(trade["sl"]) if trade.get("sl") is not None else None,
                float(trade["tp"]) if trade.get("tp") is not None else None,
                int(trade["open_ts_ms"]),
                None, None,
                0.0,
                float(trade.get("fee_usdt", 0.0)),
                "OPEN",
                None
            ))

    def update_trade_close(self, trade_id: str, close_ts_ms: int, close_price: float, pnl_usdt: float, fee_usdt: float, close_reason: str):
        with self._conn() as con:
            con.execute("""
                UPDATE trades
                SET close_ts_ms=?,
                    close_price=?,
                    pnl_usdt=?,
                    fee_usdt=fee_usdt+?,
                    status='CLOSED',
                    close_reason=?
                WHERE trade_id=?
            """, (int(close_ts_ms), float(close_price), float(pnl_usdt), float(fee_usdt), close_reason or "", trade_id))

    def get_open_trades(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            cur = con.execute("""
                SELECT trade_id, symbol, leverage, side, qty, notional, entry_price, sl, tp, open_ts_ms, fee_usdt
                FROM trades
                WHERE run_id=? AND status='OPEN'
                ORDER BY open_ts_ms DESC
            """, (run_id,))
            rows = cur.fetchall()

        out = []
        for r in rows:
            out.append({
                "trade_id": r[0],
                "symbol": r[1],
                "leverage": r[2],
                "side": r[3],
                "qty": r[4],
                "notional": r[5],
                "entry_price": r[6],
                "sl": r[7],
                "tp": r[8],
                "open_ts_ms": r[9],
                "fee_usdt": r[10],
            })
        return out

    def get_closed_trades(self, run_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        with self._conn() as con:
            cur = con.execute("""
                SELECT trade_id, symbol, leverage, side, qty, notional, entry_price, sl, tp,
                       open_ts_ms, close_ts_ms, close_price, pnl_usdt, fee_usdt, close_reason
                FROM trades
                WHERE run_id=? AND status='CLOSED'
                ORDER BY close_ts_ms DESC
                LIMIT ?
            """, (run_id, limit))
            rows = cur.fetchall()

        out = []
        for r in rows:
            out.append({
                "trade_id": r[0],
                "symbol": r[1],
                "leverage": r[2],
                "side": r[3],
                "qty": r[4],
                "notional": r[5],
                "entry_price": r[6],
                "sl": r[7],
                "tp": r[8],
                "open_ts_ms": r[9],
                "close_ts_ms": r[10],
                "close_price": r[11],
                "pnl_usdt": r[12],
                "fee_usdt": r[13],
                "close_reason": r[14],
            })
        return out

    # ======================
    # ORDERS
    # ======================

    def upsert_order_open(self, run_id: str, order: Dict[str, Any]):
        """
        order must contain:
        order_id, symbol, leverage, side, qty, order_type, price, sl, tp, created_ts_ms, note
        """
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO orders(
                    order_id, run_id, symbol, leverage, side, qty, order_type,
                    price, sl, tp, created_ts_ms, filled_ts_ms, canceled_ts_ms,
                    status, note
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                order["order_id"], run_id, order["symbol"],
                int(order["leverage"]), order["side"], float(order["qty"]),
                order["order_type"], float(order["price"]),
                float(order["sl"]) if order.get("sl") is not None else None,
                float(order["tp"]) if order.get("tp") is not None else None,
                int(order["created_ts_ms"]),
                None, None,
                "OPEN",
                order.get("note", "")
            ))

    def update_order_filled(self, order_id: str, filled_ts_ms: int, note: str = ""):
        with self._conn() as con:
            con.execute("""
                UPDATE orders
                SET filled_ts_ms=?,
                    status='FILLED',
                    note=?
                WHERE order_id=?
            """, (int(filled_ts_ms), note or "", order_id))

    def update_order_canceled(self, order_id: str, canceled_ts_ms: int, note: str = ""):
        with self._conn() as con:
            con.execute("""
                UPDATE orders
                SET canceled_ts_ms=?,
                    status='CANCELED',
                    note=?
                WHERE order_id=?
            """, (int(canceled_ts_ms), note or "", order_id))

    def get_open_orders(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            cur = con.execute("""
                SELECT order_id, symbol, leverage, side, qty, order_type, price, sl, tp,
                       created_ts_ms, note
                FROM orders
                WHERE run_id=? AND status='OPEN'
                ORDER BY created_ts_ms DESC
            """, (run_id,))
            rows = cur.fetchall()

        out = []
        for r in rows:
            out.append({
                "order_id": r[0],
                "symbol": r[1],
                "leverage": r[2],
                "side": r[3],
                "qty": r[4],
                "order_type": r[5],
                "price": r[6],
                "sl": r[7],
                "tp": r[8],
                "created_ts_ms": r[9],
                "note": r[10],
            })
        return out

    # ======================
    # EQUITY CURVE
    # ======================

    def save_equity_curve(self, run_id: str, curve: List[Dict[str, float]]):
        """
        curve = [{"ts_ms":..., "equity":...}, ...]
        """
        if not curve:
            return
        with self._conn() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS equity (
                ts_ms INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                equity REAL NOT NULL
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_equity_run ON equity(run_id, ts_ms)")

            for p in curve:
                con.execute(
                    "INSERT INTO equity(ts_ms, run_id, equity) VALUES(?,?,?)",
                    (int(p["ts_ms"]), run_id, float(p["equity"]))
                )

    def load_equity_curve(self, run_id: str) -> List[Tuple[int, float]]:
        with self._conn() as con:
            # гарантируем, что таблица существует
            con.execute("""
            CREATE TABLE IF NOT EXISTS equity (
                ts_ms INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                equity REAL NOT NULL
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_equity_run ON equity(run_id, ts_ms)")

            cur = con.execute(
                "SELECT ts_ms, equity FROM equity WHERE run_id=? ORDER BY ts_ms",
                (run_id,)
            )
            return cur.fetchall()
