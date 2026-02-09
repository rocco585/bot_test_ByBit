import sqlite3
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from bot_engine import BotEngine, BotSettings
from trade_store import TradeStore


DB_PATH = "paper_trades.sqlite"

st.set_page_config(page_title="Bybit Paper Bot", layout="wide")
st.title("🤖 Bybit Paper Trading Bot (LIMIT)")
st.caption("Paper-торговля на реальных свечах (mainnet public) + сохранение в SQLite")


# =========================
# Session state init
# =========================
if "engine" not in st.session_state:
    st.session_state.engine = None
if "running" not in st.session_state:
    st.session_state.running = False
if "run_id" not in st.session_state:
    st.session_state.run_id = None


# =========================
# SQLite helpers
# =========================
def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def get_columns(con: sqlite3.Connection, table: str) -> List[str]:
    try:
        cur = con.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        return [r["name"] for r in rows]
    except Exception:
        return []


def pick_first(existing: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in existing:
            return c
    return None


def safe_select(con: sqlite3.Connection, sql: str, params: Tuple = ()) -> pd.DataFrame:
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def ms_to_dt(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, unit="ms")
    except Exception:
        return series


# =========================
# UI: Sidebar settings
# =========================
st.sidebar.header("⚙️ Настройки")

symbols_all = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "DASHUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]

symbols: List[str] = st.sidebar.multiselect(
    "Торговые пары",
    options=symbols_all,
    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
)

timeframe = st.sidebar.selectbox(
    "Таймфрейм",
    options=["5", "15", "60", "240"],
    index=1,
    help="5м / 15м / 1ч / 4ч"
)

leverage = st.sidebar.slider("Плечо", min_value=1, max_value=20, value=5)

session_usdt = st.sidebar.number_input(
    "Капитал на сессию (USDT)",
    min_value=10.0,
    value=1000.0,
    step=50.0
)

per_trade_usdt = st.sidebar.number_input(
    "USDT на одну сделку",
    min_value=5.0,
    value=50.0,
    step=5.0
)

limit_offset_pct = st.sidebar.number_input(
    "Отступ для LIMIT (%)",
    min_value=0.01,
    value=0.10,
    step=0.05
)

poll_sec = st.sidebar.number_input(
    "Интервал обновления (сек)",
    min_value=5,
    value=10,
    step=1
)


# =========================
# Engine builder
# =========================
store = TradeStore(DB_PATH)


def build_engine() -> BotEngine:
    settings = BotSettings(
        symbols=symbols,
        timeframe=timeframe,
        leverage=leverage,
        session_usdt=session_usdt,
        per_trade_usdt=per_trade_usdt,
        limit_offset_pct=limit_offset_pct,
        poll_sec=poll_sec,
        run_id=""
    )
    return BotEngine(settings=settings, store=store, on_event=None)


# =========================
# Controls
# =========================
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    if st.button("▶️ Старт", width="stretch"):
        if not symbols:
            st.warning("Выберите хотя бы одну торговую пару")
        else:
            st.session_state.engine = build_engine()
            st.session_state.engine.start()
            st.session_state.running = True
            st.session_state.run_id = st.session_state.engine.settings.run_id
            st.success(f"Бот запущен. run_id={st.session_state.run_id}")

with c2:
    if st.button("⏹ Стоп", width="stretch"):
        if st.session_state.engine:
            st.session_state.engine.stop()
        st.session_state.running = False
        st.success("Бот остановлен")

with c3:
    st.info(f"Статус: {'🟢 работает' if st.session_state.running else '🔴 остановлен'}")


st.divider()


# =========================
# Data loaders (SQL-driven)
# =========================
run_id = st.session_state.run_id

with db_connect() as con:
    # --- trades ---
    trades_df = pd.DataFrame()
    if run_id and table_exists(con, "trades"):
        cols = get_columns(con, "trades")

        run_col = pick_first(cols, ["run_id"])
        status_col = pick_first(cols, ["status", "state"])

        # возможные названия колонок времени/цен/PNL
        open_ts_col = pick_first(cols, ["open_ts_ms", "opened_ts_ms", "open_time_ms", "opened_at_ms"])
        close_ts_col = pick_first(cols, ["close_ts_ms", "closed_ts_ms", "close_time_ms", "closed_at_ms"])
        entry_col = pick_first(cols, ["entry_price", "open_price", "price_in"])
        exit_col = pick_first(cols, ["exit_price", "close_price", "price_out"])
        pnl_col = pick_first(cols, ["pnl_usdt", "pnl", "realized_pnl_usdt", "profit_usdt"])

        # если есть status — берём открытые/закрытые отдельно, иначе показываем всё
        if run_col:
            if status_col and status_col in cols:
                open_q = f"SELECT * FROM trades WHERE {run_col}=? AND {status_col} IN ('OPEN','open','Open') ORDER BY {open_ts_col or 'rowid'} DESC"
                closed_q = f"SELECT * FROM trades WHERE {run_col}=? AND {status_col} IN ('CLOSED','closed','Closed') ORDER BY {close_ts_col or 'rowid'} DESC"
                open_trades_df = safe_select(con, open_q, (run_id,))
                closed_trades_df = safe_select(con, closed_q, (run_id,))
            else:
                open_trades_df = safe_select(con, f"SELECT * FROM trades WHERE {run_col}=? ORDER BY {open_ts_col or 'rowid'} DESC", (run_id,))
                closed_trades_df = pd.DataFrame()
        else:
            open_trades_df = pd.DataFrame()
            closed_trades_df = pd.DataFrame()

        # приведём “витринные” колонки, если они есть
        def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            show = pd.DataFrame()
            # базовые поля
            for k in ["symbol", "side", "leverage", "qty"]:
                if k in df.columns:
                    show[k] = df[k]
            # цены/TP/SL
            for k in ["entry_price", "sl", "tp"]:
                if k in df.columns:
                    show[k] = df[k]
            if entry_col and entry_col in df.columns and "entry_price" not in show.columns:
                show["entry_price"] = df[entry_col]
            if exit_col and exit_col in df.columns:
                show["exit_price"] = df[exit_col]
            if pnl_col and pnl_col in df.columns:
                show["pnl_usdt"] = df[pnl_col]
            # времена
            if open_ts_col and open_ts_col in df.columns:
                show["open_time"] = ms_to_dt(df[open_ts_col])
            if close_ts_col and close_ts_col in df.columns:
                show["close_time"] = ms_to_dt(df[close_ts_col])
            # TP/SL если назывались иначе
            if "sl" not in show.columns and "sl" in df.columns:
                show["sl"] = df["sl"]
            if "tp" not in show.columns and "tp" in df.columns:
                show["tp"] = df["tp"]
            # статус если есть
            if status_col and status_col in df.columns:
                show["status"] = df[status_col]
            return show

        open_trades_view = normalize_trades(open_trades_df)
        closed_trades_view = normalize_trades(closed_trades_df)

    else:
        open_trades_view = pd.DataFrame()
        closed_trades_view = pd.DataFrame()

    # --- orders ---
    if run_id and table_exists(con, "orders"):
        cols = get_columns(con, "orders")
        run_col = pick_first(cols, ["run_id"])
        status_col = pick_first(cols, ["status", "state"])
        ts_col = pick_first(cols, ["created_ts_ms", "created_at_ms", "ts_ms", "time_ms"])

        if run_col:
            if status_col:
                orders_df = safe_select(
                    con,
                    f"SELECT * FROM orders WHERE {run_col}=? AND {status_col} IN ('OPEN','open','Open') ORDER BY {ts_col or 'rowid'} DESC",
                    (run_id,)
                )
            else:
                orders_df = safe_select(con, f"SELECT * FROM orders WHERE {run_col}=? ORDER BY {ts_col or 'rowid'} DESC", (run_id,))
        else:
            orders_df = pd.DataFrame()

        # витрина
        if not orders_df.empty:
            orders_view = pd.DataFrame()
            for k in ["symbol", "side", "leverage", "qty", "price", "sl", "tp", "order_type"]:
                if k in orders_df.columns:
                    orders_view[k] = orders_df[k]
            if ts_col and ts_col in orders_df.columns:
                orders_view["created_time"] = ms_to_dt(orders_df[ts_col])
            if status_col and status_col in orders_df.columns:
                orders_view["status"] = orders_df[status_col]
        else:
            orders_view = pd.DataFrame()
    else:
        orders_view = pd.DataFrame()

    # --- equity ---
    if run_id and table_exists(con, "equity"):
        cols = get_columns(con, "equity")
        run_col = pick_first(cols, ["run_id"])
        ts_col = pick_first(cols, ["ts_ms", "time_ms", "timestamp_ms"])
        eq_col = pick_first(cols, ["equity", "value", "balance"])

        if run_col and ts_col and eq_col:
            eq_df = safe_select(con, f"SELECT {ts_col} as ts_ms, {eq_col} as equity FROM equity WHERE {run_col}=? ORDER BY {ts_col}", (run_id,))
        else:
            eq_df = pd.DataFrame()
    else:
        eq_df = pd.DataFrame()


# =========================
# UI tables
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("📌 Отложенные ордера (OPEN)")
    if orders_view.empty:
        st.caption("Нет данных (возможно ещё не было сигналов/ордеров).")
    else:
        st.dataframe(orders_view, width="stretch", height=300)

with right:
    st.subheader("📈 Открытые сделки")
    if open_trades_view.empty:
        st.caption("Открытых сделок нет.")
    else:
        st.dataframe(open_trades_view, width="stretch", height=300)

st.subheader("✅ Закрытые сделки")
if closed_trades_view.empty:
    st.caption("Закрытых сделок нет.")
else:
    # раскраска pnl, если есть
    df = closed_trades_view.copy()
    if "pnl_usdt" in df.columns:
        try:
            sty = df.style.applymap(
                lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else ("color: red" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["pnl_usdt"]
            )
            st.dataframe(sty, width="stretch", height=350)
        except Exception:
            st.dataframe(df, width="stretch", height=350)
    else:
        st.dataframe(df, width="stretch", height=350)


st.subheader("💰 Equity")
if eq_df.empty:
    st.caption("Нет данных по equity (появится после сохранения точек или остановки).")
else:
    eq_df["time"] = pd.to_datetime(eq_df["ts_ms"], unit="ms")
    st.line_chart(eq_df.set_index("time")["equity"])


st.divider()
st.caption("Если таблицы пустые — поставь TF=5 и дай боту поработать 5–15 минут: стратегия может давать HOLD долго.")
