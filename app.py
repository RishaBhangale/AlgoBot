#!/usr/bin/env python3
"""
Supertrend Bot Web Server Wrapper (FastAPI)
Disguises the trading bot as a web service for Render free tier.

Key features:
- FastAPI responds to health checks IMMEDIATELY
- Trading bot runs in background thread
- DAILY LOOP: Re-authenticates each morning with fresh Kite token
"""
import threading
import os
import traceback
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

# Import the trading bot and auto-login
from main import SupertrendBot, SECURITIES, now_ist, KITE_AVAILABLE
from auto_login import KiteAutoLogin, load_credentials, SELENIUM_AVAILABLE

# Global state
bot_instance = None
bot_thread = None
bot_logs = []

bot_status = {
    "status": "initialized",
    "started_at": None,
    "last_health_check": None,
    "authenticated": False,
    "kite_user": None,
    "error": None,
    "market_status": None,
    "candles_loaded": {},
    "current_trend": {},
    "trading_day": None,
    "days_run": 0
}


def add_log(message: str):
    """Add log message with timestamp (immediately flushed)."""
    timestamp = now_ist().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    bot_logs.append(log_entry)
    if len(bot_logs) > 200:
        bot_logs.pop(0)


def run_single_trading_day(creds: dict) -> bool:
    """
    Run a single trading day session.
    Returns True if successful, False if error.
    """
    global bot_instance, bot_status
    
    today = now_ist().strftime("%Y-%m-%d")
    bot_status["trading_day"] = today
    bot_status["days_run"] += 1
    
    add_log(f"📅 Starting trading day: {today} (Day #{bot_status['days_run']})")
    
    # Calculate market times
    now = now_ist()
    login_time = now.replace(hour=8, minute=45, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Weekend check
    if now.weekday() >= 5:
        bot_status["market_status"] = "Weekend"
        add_log("📅 Weekend - Market closed")
        return True  # Not an error, just skip
    
    # After hours check
    if now > market_close:
        bot_status["market_status"] = "After Hours"
        add_log("📅 After market hours - waiting for tomorrow")
        return True
    
    # Wait for login time (8:45 AM)
    if now < login_time:
        mins_to_wait = int((login_time - now).total_seconds() / 60)
        add_log(f"⏰ Waiting {mins_to_wait} mins until 8:45 AM login time...")
        bot_status["status"] = "waiting_for_login_time"
        
        while now_ist() < login_time:
            time.sleep(60)
    
    # FRESH AUTHENTICATION EVERY DAY
    add_log("🔐 Starting Kite auto-login (fresh token)...")
    bot_status["status"] = "authenticating"
    bot_status["authenticated"] = False
    
    try:
        auto_login = KiteAutoLogin(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            user_id=creds["user_id"],
            password=creds["password"],
            totp_secret=creds["totp_secret"],
            headless=True
        )
        
        access_token = auto_login.login()
        
        if not access_token:
            bot_status["status"] = "error"
            bot_status["error"] = "Auto-login failed"
            add_log("❌ Auto-login failed!")
            return False
        
        bot_status["authenticated"] = True
        bot_status["kite_user"] = creds["user_id"]
        add_log(f"✅ Authenticated successfully!")
        
    except Exception as e:
        add_log(f"❌ Auth error: {e}")
        bot_status["status"] = "error"
        bot_status["error"] = str(e)
        return False
    
    # Wait for market open
    now = now_ist()
    if now < market_open:
        mins_to_wait = int((market_open - now).total_seconds() / 60)
        add_log(f"⏳ Waiting {mins_to_wait} mins for market open at 9:15 AM...")
        bot_status["status"] = "waiting_for_market"
        
        while now_ist() < market_open:
            time.sleep(30)
    
    # Start trading session
    add_log("📊 Starting trading session...")
    bot_status["status"] = "running"
    bot_status["market_status"] = "Market Open"
    
    try:
        # Create fresh bot instance each day
        bot_instance = SupertrendBot()
        bot_instance.kite = auto_login.kite
        bot_instance.is_running = True
        
        # Telegram notification
        if bot_instance.telegram:
            bot_instance.telegram.notify_bot_start(list(SECURITIES.keys()))
            add_log("📱 Telegram notification sent!")
        
        # Fetch historical data
        add_log("📊 Fetching historical data...")
        bot_instance.fetch_historical()
        
        for symbol, trader in bot_instance.traders.items():
            bot_status["candles_loaded"][symbol] = len(trader.candles)
            add_log(f"   {symbol}: {len(trader.candles)} candles loaded")
        
        # Start live feed
        add_log("🔴 Starting live data feed...")
        bot_instance.start_live_feed()
        
        add_log("✅ Bot is now active and trading!")
        
        # Run until market close
        while bot_instance.is_running and bot_instance.is_market_open():
            for symbol, trader in bot_instance.traders.items():
                if trader.current_trend != 0:
                    trend = "BULLISH" if trader.current_trend == 1 else "BEARISH"
                    bot_status["current_trend"][symbol] = trend
            time.sleep(5)
        
        # End of day
        add_log("📈 Market closed. Generating report...")
        bot_instance.generate_report()
        
        if bot_instance.ticker:
            bot_instance.ticker.close()
        
        add_log("✅ Trading session complete!")
        bot_status["status"] = "session_complete"
        bot_status["market_status"] = "Market Closed"
        
        return True
        
    except Exception as e:
        add_log(f"❌ Trading error: {e}")
        traceback.print_exc()
        bot_status["error"] = str(e)
        return False


def run_trading_bot():
    """
    MAIN BOT LOOP - Runs continuously, re-authenticating each trading day.
    """
    global bot_status
    
    # Small delay to let FastAPI start
    time.sleep(2)
    
    add_log("🚀 Trading bot started - DAILY LOOP MODE")
    bot_status["started_at"] = now_ist().isoformat()
    
    # Load credentials once
    add_log("📋 Loading credentials...")
    creds = load_credentials()
    
    if not creds["api_key"] or not creds["api_secret"]:
        bot_status["status"] = "error"
        bot_status["error"] = "Missing KITE_API_KEY or KITE_API_SECRET"
        add_log("❌ Missing API credentials!")
        return
    
    if not creds["user_id"] or not creds["password"]:
        bot_status["status"] = "error"
        bot_status["error"] = "Missing KITE_USER_ID or KITE_PASSWORD"
        add_log("❌ Missing login credentials!")
        return
    
    add_log(f"✅ Credentials loaded for user: {creds['user_id']}")
    
    if not SELENIUM_AVAILABLE:
        bot_status["status"] = "error"
        bot_status["error"] = "Selenium not available"
        add_log("❌ Selenium not installed!")
        return
    
    # INFINITE DAILY LOOP
    while True:
        try:
            now = now_ist()
            
            # Skip weekends
            if now.weekday() >= 5:
                add_log(f"📅 Weekend ({now.strftime('%A')}) - sleeping until Monday...")
                bot_status["status"] = "weekend_sleep"
                
                # Sleep until Monday 8:30 AM
                while now_ist().weekday() >= 5:
                    time.sleep(300)  # Check every 5 mins
                continue
            
            # Skip if after market hours (wait for next day)
            market_close = now.replace(hour=15, minute=30, second=0)
            if now > market_close:
                next_login = (now + timedelta(days=1)).replace(hour=8, minute=30, second=0)
                wait_hours = (next_login - now).total_seconds() / 3600
                add_log(f"📅 Market closed. Next session in {wait_hours:.1f} hours...")
                bot_status["status"] = "waiting_for_next_day"
                
                # Sleep until 8:30 AM next day
                while now_ist() < next_login:
                    time.sleep(300)
                continue
            
            # Run today's trading session
            success = run_single_trading_day(creds)
            
            if not success:
                add_log("⚠️ Session failed - retrying in 30 minutes...")
                time.sleep(1800)  # Wait 30 mins before retry
                continue
            
            # Wait for next trading day
            add_log("💤 Session complete. Waiting for next trading day...")
            bot_status["status"] = "day_complete"
            
            # Sleep until next day 8:30 AM
            next_login = (now_ist() + timedelta(days=1)).replace(hour=8, minute=30, second=0)
            while now_ist() < next_login:
                # Skip weekends
                if now_ist().weekday() >= 5:
                    break
                time.sleep(300)
                
        except Exception as e:
            add_log(f"❌ Loop error: {e}")
            traceback.print_exc()
            time.sleep(600)  # Wait 10 mins on error


def start_bot_thread():
    """Start the trading bot in a background thread."""
    global bot_thread
    
    if bot_thread is not None and bot_thread.is_alive():
        add_log("⚠️ Bot thread already running")
        return
    
    bot_thread = threading.Thread(target=run_trading_bot, name="TradingBot")
    bot_thread.daemon = True
    bot_thread.start()
    add_log("✅ Trading bot thread launched (daily loop mode)")


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    add_log("🌐 FastAPI starting up...")
    start_bot_thread()
    yield
    add_log("🛑 FastAPI shutting down...")


app = FastAPI(
    title="Supertrend Trading Bot",
    description="NIFTY & BANKNIFTY Options Trading Bot - Daily Auto-Login",
    version="2.0.0",
    lifespan=lifespan
)


# --- HEALTH CHECK ROUTES ---

@app.get("/", response_class=PlainTextResponse)
@app.head("/")
async def health_check():
    """Main health check endpoint."""
    bot_status["last_health_check"] = now_ist().isoformat()
    status = bot_status.get("status", "unknown")
    day = bot_status.get("trading_day", "N/A")
    return f"Bot status: {status} | Day: {day}"


@app.get("/ping", response_class=PlainTextResponse)
@app.head("/ping")
async def ping():
    """Simple ping for UptimeRobot."""
    return "pong"


@app.get("/favicon.ico")
async def favicon():
    """Empty favicon."""
    return ""


@app.get("/status")
async def detailed_status():
    """Detailed bot status."""
    global bot_instance, bot_status
    
    status = {
        "bot": bot_status.copy(),
        "current_time": now_ist().isoformat(),
        "market_open": bot_instance.is_market_open() if bot_instance else False,
        "securities": list(SECURITIES.keys())
    }
    
    if bot_instance and hasattr(bot_instance, 'traders'):
        status["positions"] = {}
        status["trades_today"] = {}
        
        for symbol, trader in bot_instance.traders.items():
            status["positions"][symbol] = {
                "has_position": trader.position is not None,
                "option_type": trader.position.option_type if trader.position else None,
                "entry_price": trader.position.entry_price if trader.position else None,
                "candles": len(trader.candles),
                "trend": "BULLISH" if trader.current_trend == 1 else "BEARISH" if trader.current_trend == -1 else "NONE"
            }
            status["trades_today"][symbol] = len(trader.trades)
    
    return status


@app.get("/logs")
async def get_logs():
    """Get recent bot logs."""
    return {"logs": bot_logs[-50:], "count": len(bot_logs)}


@app.get("/logs/all")
async def get_all_logs():
    """Get all bot logs."""
    return {"logs": bot_logs, "count": len(bot_logs)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    add_log(f"🌐 Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
