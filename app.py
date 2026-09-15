#!/usr/bin/env python3
"""
Production Autonomous Stock Options Bot Web Server (FastAPI)
Deploys Quad-Confirmation + Alligator Golden Zone Stock Bot on Render free tier.

Features:
- FastAPI responds to Render & UptimeRobot health checks immediately
- Autonomous daily trading loop in background thread (08:50 AM to 15:35 IST)
- Real-time Telegram alerting on entries, exits, near-misses, and daily EOD summary
- Strict Capital Management: Rs 1,00,000 (Rs 1.0 Lakh) Total Capital
"""

import os
import sys
import threading
import traceback
import time
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from main_stocks import StockOptionsBot, STOCKS, TOTAL_CAPITAL, MAX_CONCURRENT_POSITIONS, PER_STOCK_CAPITAL_LIMIT, DAILY_LOSS_LIMIT, now_ist, PAPER_TRADING
from auto_login import KiteAutoLogin, load_credentials

# Global state
bot_instance = None
bot_thread = None
bot_logs = []

bot_status = {
    "status": "initialized",
    "paper_trading": PAPER_TRADING,
    "started_at": None,
    "last_health_check": None,
    "authenticated": False,
    "error": None,
    "market_status": None,
    "capital": {
        "total_capital": TOTAL_CAPITAL,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "per_stock_limit": PER_STOCK_CAPITAL_LIMIT,
        "daily_loss_limit": DAILY_LOSS_LIMIT
    },
    "trading_day": None,
    "days_run": 0
}


def add_log(message: str):
    """Add log message with timestamp."""
    timestamp = now_ist().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    bot_logs.append(log_entry)
    if len(bot_logs) > 250:
        bot_logs.pop(0)


def run_single_trading_day() -> bool:
    """Run a single trading day session."""
    global bot_instance, bot_status

    today = now_ist().strftime("%Y-%m-%d")
    bot_status["trading_day"] = today
    bot_status["days_run"] += 1

    add_log(f"Starting Stock trading day: {today} (Day #{bot_status['days_run']})")

    now = now_ist()
    login_time = now.replace(hour=8, minute=50, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # Weekend check
    if now.weekday() >= 5:
        bot_status["status"] = "sleeping"
        bot_status["market_status"] = "Weekend"
        add_log("Weekend - Market closed")
        return True

    # After hours check — clear any stale error status
    if now > market_close:
        bot_status["status"] = "sleeping"
        bot_status["market_status"] = "After Hours"
        add_log("After market hours - waiting for tomorrow")
        return True

    # Wait for login time (8:50 AM)
    if now < login_time:
        mins = int((login_time - now).total_seconds() / 60)
        add_log(f"Waiting {mins} mins until 08:50 AM login time...")
        bot_status["status"] = "waiting_for_login_time"
        while now_ist() < login_time:
            time.sleep(60)

    # Fresh Authentication
    add_log("Performing automated Kite Connect login...")
    bot_status["status"] = "authenticating"

    bot_instance = StockOptionsBot()
    if not bot_instance.authenticate():
        add_log("Kite authentication failed!")
        bot_status["status"] = "error"
        bot_status["error"] = "Auth failed"
        return False

    bot_status["authenticated"] = True
    bot_status["status"] = "initializing"
    bot_instance.is_running = True  # Enable WebSocket reconnect logic

    # Wait for market open
    now = now_ist()
    if now < market_open:
        mins = int((market_open - now).total_seconds() / 60)
        add_log(f"Waiting {mins} mins for market open at 09:15 AM...")
        while now_ist() < market_open:
            time.sleep(30)

    # Start Trading Session — everything below is inside try/except so
    # any failure (metadata, historical, WebSocket, etc.) gets reported to Telegram
    add_log("Starting Stock Options trading session...")
    bot_status["status"] = "running"
    bot_status["market_status"] = "Market Open"

    eod_summary_sent = False
    try:
        # Load metadata and historical data INSIDE try so failures reach Telegram
        bot_instance.load_market_metadata()
        bot_instance.fetch_historical_and_gz()

        if bot_instance.telegram:
            bot_instance.telegram.notify_bot_start(list(STOCKS.keys()), capital_tracker=bot_instance.capital_tracker)

        bot_instance.start_live_feed()

        heartbeat_sent = False
        pcr_last_updated = None

        # Market close time: 15:30 IST. Give 5 extra mins buffer.
        session_end = now_ist().replace(hour=15, minute=35, second=0, microsecond=0)

        while now_ist() < session_end:
            now = now_ist()

            # 1. Update PCR every 15 mins (only during market hours)
            if bot_instance.is_market_open():
                if pcr_last_updated is None or (now - pcr_last_updated).total_seconds() >= 900:
                    for sym in STOCKS.keys():
                        bot_instance.pcr_tracker.update_stock_pcr(sym, bot_instance.nfo_df)
                    pcr_last_updated = now

            # 2. Mid-Day Heartbeat at 12:00 PM IST (fires once)
            if not heartbeat_sent and now.hour == 12 and now.minute >= 0:
                if bot_instance.telegram:
                    status_dict = {s: t.get_diagnostics() for s, t in bot_instance.traders.items()}
                    total_ticks = sum(t.tick_count for t in bot_instance.traders.values())
                    active_pos = sum(1 for t in bot_instance.traders.values() if t.position is not None)
                    bot_instance.telegram.notify_midday_heartbeat(status_dict, total_ticks, active_pos)
                heartbeat_sent = True

            # 3. Tick-Starvation Watchdog (two cases handled):
            #    A) Had ticks before, now silent for >5 mins → restart
            #    B) Never got any tick, and market has been open for >10 mins → restart
            #    C) on_noreconnect set _needs_restart flag → restart from main thread
            if bot_instance.is_market_open():
                last_tick = getattr(bot_instance, "_last_tick_time", None)
                feed_start = getattr(bot_instance, "_feed_start_time", None)

                needs_restart = getattr(bot_instance, "_needs_restart", False)
                restart_reason = ""

                if needs_restart:
                    restart_reason = "WebSocket exhausted reconnect attempts (flag set by on_noreconnect)"
                elif last_tick is not None and (now - last_tick).total_seconds() > 300:
                    needs_restart = True
                    restart_reason = "no ticks for 5+ minutes (feed dropped)"
                elif last_tick is None and feed_start is not None and (now - feed_start).total_seconds() > 300:
                    needs_restart = True
                    restart_reason = "no ticks received in first 5 mins (feed never connected)"

                if needs_restart:
                    add_log(f"Watchdog triggered: {restart_reason} — restarting WebSocket...")
                    try:
                        if hasattr(bot_instance, "telegram") and bot_instance.telegram:
                            bot_instance.telegram.send_message(
                                f"⚠️ <b>WebSocket Watchdog Triggered</b>\n\n"
                                f"<b>Reason:</b> {restart_reason}\n"
                                f"Restarting WebSocket feed..."
                            )
                    except Exception as tg_err:
                        add_log(f"Telegram watchdog alert failed: {tg_err}")
                    try:
                        bot_instance._restart_ticker()
                        add_log("WebSocket feed restarted by watchdog.")
                    except Exception as wd_err:
                        add_log(f"Watchdog restart failed: {wd_err}")

            # 4. Time-based forced square-off at 15:20 IST
            #    Safety net — ensures positions close even if candle-driven square-off
            #    fails (30-min stocks, missing ticks, etc.)
            if now.hour == 15 and now.minute >= 20 and now.minute < 25:
                for sym, trader in bot_instance.traders.items():
                    if trader.position is not None:
                        add_log(f"⏰ Force-closing open position for {sym} at 15:20 (time-based safety)")
                        try:
                            trader._close_position("EOD_FORCE_CLOSE")
                        except Exception as sq_err:
                            add_log(f"Force square-off error for {sym}: {sq_err}")

            # 5. EOD Summary at 15:31 IST (fires once, inside loop — survives loop exit)
            if not eod_summary_sent and now.hour == 15 and now.minute >= 31:
                add_log("15:31 IST — generating EOD summary...")
                try:
                    bot_instance.generate_report()
                    eod_summary_sent = True
                except Exception as eod_err:
                    add_log(f"EOD summary error: {eod_err}")

            time.sleep(5)

        add_log("Market session window closed.")
        
        # Ticker cleanup — prevent thread leaks and overnight restart loops
        bot_instance.is_running = False
        try:
            if bot_instance.ticker:
                bot_instance.ticker.close()
                add_log("WebSocket ticker closed cleanly.")
        except Exception as tc_err:
            add_log(f"Ticker cleanup warning: {tc_err}")
        
        if not eod_summary_sent:
            add_log("Sending delayed EOD summary...")
            bot_instance.generate_report()
        bot_status["status"] = "day_complete"
        bot_status["market_status"] = "Market Closed"
        return True

    except Exception as e:
        add_log(f"Session error: {e}")
        traceback.print_exc()
        # Notify via Telegram if possible so user knows something failed
        try:
            if bot_instance and bot_instance.telegram:
                bot_instance.telegram.send_message(
                    f"<b>Stock Bot Session Error</b>\n\n"
                    f"<code>{str(e)[:300]}</code>\n\n"
                    f"<i>Bot will retry in 15 minutes.</i>"
                )
        except Exception:
            pass
        # Clean up ticker on session error to prevent thread leaks
        try:
            if bot_instance:
                bot_instance.is_running = False
                if bot_instance.ticker:
                    bot_instance.ticker.close()
                    add_log("WebSocket ticker closed after session error.")
        except Exception as cleanup_err:
            add_log(f"Error during ticker cleanup: {cleanup_err}")

        # Try EOD summary only if market closed and summary wasn't sent yet
        try:
            now = now_ist()
            if (now.hour > 15 or (now.hour == 15 and now.minute >= 30)) and bot_instance and not eod_summary_sent:
                bot_instance.generate_report()
                eod_summary_sent = True
        except Exception:
            pass
        bot_status["status"] = "error"
        bot_status["error"] = str(e)
        return False


def run_trading_bot():
    """Main daemon loop running day after day."""
    add_log("Stock Options Bot Daemon started.")

    while True:
        try:
            success = run_single_trading_day()
            if not success:
                add_log("Session failed. Retrying in 15 minutes...")
                bot_status["status"] = "error_retry"
                time.sleep(900)
                bot_status["status"] = "sleeping"
                bot_status["error"] = None
                continue

            add_log("Session complete. Sleeping until 08:45 AM tomorrow...")
            bot_status["status"] = "sleeping"
            next_morning = (now_ist() + timedelta(days=1)).replace(hour=8, minute=45, second=0)
            while now_ist() < next_morning:
                time.sleep(300)

        except Exception as e:
            add_log(f"Daemon loop error: {e}")
            traceback.print_exc()
            bot_status["status"] = "sleeping"
            time.sleep(60)


def start_bot_thread():
    """Start bot in background thread."""
    global bot_thread
    if bot_thread is not None and bot_thread.is_alive():
        return
    bot_thread = threading.Thread(target=run_trading_bot, name="StockBotDaemon", daemon=True)
    bot_thread.start()
    add_log("Stock bot background daemon thread launched.")


# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    add_log("FastAPI initializing...")
    start_bot_thread()
    yield
    add_log("FastAPI shutting down...")


app = FastAPI(
    title="Stock Options Momentum Bot",
    description="Quad-Confirmation + Alligator Golden Zone - Autonomous Daily Runner",
    version="3.2.0",
    lifespan=lifespan
)


@app.get("/", response_class=PlainTextResponse)
@app.head("/")
async def root():
    bot_status["last_health_check"] = now_ist().isoformat()
    st = bot_status.get("status", "unknown")
    day = bot_status.get("trading_day", "N/A")
    cap = bot_instance.capital_tracker.session_capital if bot_instance and hasattr(bot_instance, "capital_tracker") else TOTAL_CAPITAL
    text = f"Stock Bot: {st} | Day: {day} | Capital: Rs {cap:,.0f}"
    status_code = 503 if st in ("error", "error_retry") else 200
    return PlainTextResponse(content=text, status_code=status_code)


@app.get("/ping", response_class=PlainTextResponse)
@app.head("/ping")
async def ping():
    return "pong"


@app.get("/status")
async def status():
    global bot_instance, bot_status
    res = {
        "bot": bot_status.copy(),
        "current_time": now_ist().isoformat(),
        "market_open": bot_instance.is_market_open() if bot_instance else False,
        "stocks": list(STOCKS.keys())
    }
    if bot_instance and hasattr(bot_instance, 'traders'):
        res["positions"] = {
            s: {"has_position": t.position is not None, "contract": t.position.tradingsymbol if t.position else None, "entry_ltp": t.position.entry_price if t.position else None, "sl": t.position.sl if t.position else None}
            for s, t in bot_instance.traders.items()
        }
        res["diagnostics"] = {s: t.get_diagnostics() for s, t in bot_instance.traders.items()}
    return res


@app.get("/diagnostics")
async def diagnostics_endpoint():
    global bot_instance
    if not bot_instance or not hasattr(bot_instance, 'traders'):
        return {"status": "bot_not_initialized", "diagnostics": {}}
    return {
        "timestamp": now_ist().isoformat(),
        "market_open": bot_instance.is_market_open(),
        "diagnostics": {s: t.get_diagnostics() for s, t in bot_instance.traders.items()}
    }


@app.get("/logs")
async def logs():
    return {"logs": bot_logs[-50:], "count": len(bot_logs)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    add_log(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
