#!/usr/bin/env python3
"""
Production Autonomous Stock Options Bot Web Server (FastAPI)
Deploys Quad-Confirmation + Alligator Golden Zone Stock Bot on Render free tier.

Features:
- FastAPI responds to Render & UptimeRobot health checks immediately
- Autonomous daily trading loop in background thread (08:50 AM to 15:30 PM IST)
- Built-in keepalive self-pinger to prevent 15-minute Render free-tier sleep
- Real-time Telegram alerting on entries, exits, near-misses, and daily EOD summary
- Strict Capital Management: ₹1,00,000 (₹1.0 Lakh) Total Capital
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


def keepalive_pinger():
    """Background thread that pings RENDER_EXTERNAL_URL every 8 minutes to prevent Render free-tier sleep."""
    import requests
    render_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("SELF_PING_URL")
    if not render_url:
        add_log("ℹ️ No RENDER_EXTERNAL_URL detected; use UptimeRobot for external pinging if on Render free tier.")
        return
        
    ping_url = render_url.rstrip("/") + "/ping"
    add_log(f"⏰ Render Keep-Alive self-pinger active: pinging {ping_url} every 8 minutes...")
    
    while True:
        try:
            time.sleep(480)  # 8 minutes
            now = now_ist()
            if now.weekday() < 5 and (8 <= now.hour < 16):
                r = requests.get(ping_url, timeout=10)
                if r.status_code == 200:
                    add_log("💓 Keep-alive self-ping sent to Render router (container awake)")
        except Exception as e:
            add_log(f"⚠️ Keep-alive ping warning: {e}")


def run_single_trading_day() -> bool:
    """Run a single trading day session."""
    global bot_instance, bot_status
    
    today = now_ist().strftime("%Y-%m-%d")
    bot_status["trading_day"] = today
    bot_status["days_run"] += 1
    
    add_log(f"📅 Starting Stock trading day: {today} (Day #{bot_status['days_run']})")
    
    now = now_ist()
    login_time = now.replace(hour=8, minute=50, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Weekend check
    if now.weekday() >= 5:
        bot_status["market_status"] = "Weekend"
        add_log("📅 Weekend - Market closed")
        return True
        
    # After hours check
    if now > market_close:
        bot_status["market_status"] = "After Hours"
        add_log("📅 After market hours - waiting for tomorrow")
        return True
        
    # Wait for login time (8:50 AM)
    if now < login_time:
        mins = int((login_time - now).total_seconds() / 60)
        add_log(f"⏰ Waiting {mins} mins until 08:50 AM login time...")
        bot_status["status"] = "waiting_for_login_time"
        while now_ist() < login_time:
            time.sleep(60)
            
    # Fresh Authentication
    add_log("🔐 Performing automated Kite Connect login...")
    bot_status["status"] = "authenticating"
    
    bot_instance = StockOptionsBot()
    if not bot_instance.authenticate():
        add_log("❌ Kite authentication failed!")
        bot_status["status"] = "error"
        bot_status["error"] = "Auth failed"
        return False
        
    bot_status["authenticated"] = True
    bot_status["status"] = "waiting_for_market"
    
    # Load metadata (tokens, lot sizes, strike steps)
    bot_instance.load_market_metadata()
    bot_instance.fetch_historical_and_gz()
    
    # Wait for market open
    now = now_ist()
    if now < market_open:
        mins = int((market_open - now).total_seconds() / 60)
        add_log(f"⏳ Waiting {mins} mins for market open at 09:15 AM...")
        while now_ist() < market_open:
            time.sleep(30)
            
    # Start Trading Session
    add_log("📊 Starting Stock Options trading session...")
    bot_status["status"] = "running"
    bot_status["market_status"] = "Market Open"
    
    try:
        if bot_instance.telegram:
            bot_instance.telegram.notify_bot_start(list(STOCKS.keys()))
            
        bot_instance.start_live_feed()
        
        heartbeat_sent = False
        pcr_last_updated = None
        
        while bot_instance.is_running and bot_instance.is_market_open():
            now = now_ist()
            # 1. Update PCR every 15 mins
            if pcr_last_updated is None or (now - pcr_last_updated).total_seconds() >= 900:
                for sym in STOCKS.keys():
                    bot_instance.pcr_tracker.update_stock_pcr(sym, bot_instance.nfo_df)
                pcr_last_updated = now
                
            # 2. Mid-Day Heartbeat at 12:00 PM IST
            if not heartbeat_sent and now.hour == 12 and now.minute >= 0:
                if bot_instance.telegram:
                    status_dict = {s: t.get_diagnostics() for s, t in bot_instance.traders.items()}
                    total_ticks = sum(t.tick_count for t in bot_instance.traders.values())
                    active_pos = sum(1 for t in bot_instance.traders.values() if t.position is not None)
                    bot_instance.telegram.notify_midday_heartbeat(status_dict, total_ticks, active_pos)
                heartbeat_sent = True
                
            time.sleep(5)
            
        add_log("🏁 Market closed. Concluding session...")
        bot_instance.generate_report()
        bot_status["status"] = "day_complete"
        bot_status["market_status"] = "Market Closed"
        return True
        
    except Exception as e:
        add_log(f"❌ Session error: {e}")
        traceback.print_exc()
        bot_status["status"] = "error"
        bot_status["error"] = str(e)
        return False


def run_trading_bot():
    """Main daemon loop running day after day."""
    add_log("🚀 Stock Options Bot Daemon started.")
    
    while True:
        try:
            success = run_single_trading_day()
            if not success:
                add_log("⚠️ Session failed. Retrying in 15 minutes...")
                time.sleep(900)
                continue
                
            add_log("💤 Session complete. Sleeping until 08:45 AM tomorrow...")
            next_morning = (now_ist() + timedelta(days=1)).replace(hour=8, minute=45, second=0)
            while now_ist() < next_morning:
                if now_ist().weekday() >= 5:
                    break
                time.sleep(300)
                
        except Exception as e:
            add_log(f"❌ Daemon loop error: {e}")
            time.sleep(60)


def start_bot_thread():
    """Start bot in background thread."""
    global bot_thread
    if bot_thread is not None and bot_thread.is_alive():
        return
    bot_thread = threading.Thread(target=run_trading_bot, name="StockBotDaemon", daemon=True)
    bot_thread.start()
    add_log("✅ Stock bot background daemon thread launched.")


# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    add_log("🌐 FastAPI initializing...")
    start_bot_thread()
    pinger = threading.Thread(target=keepalive_pinger, name="KeepAlivePinger", daemon=True)
    pinger.start()
    yield
    add_log("🛑 FastAPI shutting down...")


app = FastAPI(
    title="Stock Options Momentum Bot",
    description="Quad-Confirmation + Alligator Golden Zone - Autonomous Daily Runner",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/", response_class=PlainTextResponse)
@app.head("/")
async def root():
    bot_status["last_health_check"] = now_ist().isoformat()
    st = bot_status.get("status", "unknown")
    day = bot_status.get("trading_day", "N/A")
    return f"Stock Bot: {st} | Day: {day} | Capital: ₹{TOTAL_CAPITAL:,.0f}"


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
    add_log(f"🌐 Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
