#!/usr/bin/env python3
"""
Supertrend Bot Web Server Wrapper (FastAPI)
Disguises the trading bot as a web service for Render free tier.

Key feature: FastAPI responds to health checks IMMEDIATELY,
while the trading bot authenticates and runs in a background thread.
"""
import threading
import os
import traceback
import time
from datetime import datetime
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
    "current_trend": {}
}


def add_log(message: str):
    """Add log message with timestamp (immediately flushed)."""
    timestamp = now_ist().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    bot_logs.append(log_entry)
    if len(bot_logs) > 100:
        bot_logs.pop(0)


def run_trading_bot():
    """Run the trading bot in background thread with auto-login."""
    global bot_instance, bot_status
    
    # Small delay to let FastAPI fully start first
    time.sleep(2)
    
    add_log("🚀 Trading bot thread started...")
    bot_status["status"] = "starting"
    bot_status["started_at"] = now_ist().isoformat()
    
    try:
        # Step 1: Load credentials
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
        add_log(f"   TOTP: {'Configured' if creds['totp_secret'] else 'Not configured'}")
        
        # Step 2: Check if we should wait for market hours
        now = now_ist()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now.weekday() >= 5:
            bot_status["market_status"] = "Weekend"
            add_log("📅 Weekend - Market closed. Bot will wait.")
        elif now < market_open.replace(hour=8, minute=45):
            bot_status["market_status"] = "Pre-market (too early)"
            mins_to_wait = int((market_open.replace(hour=8, minute=45) - now).total_seconds() / 60)
            add_log(f"⏰ Too early ({now.strftime('%H:%M')}). Login will happen at 8:45 AM.")
            add_log(f"   Waiting {mins_to_wait} minutes...")
            bot_status["status"] = "waiting_for_login_time"
            
            # Wait until 8:45 AM to login
            while now_ist() < market_open.replace(hour=8, minute=45):
                time.sleep(60)
                if not bot_instance:
                    return
            add_log("⏰ 8:45 AM - Starting login process...")
        elif now > market_close:
            bot_status["market_status"] = "After hours"
            add_log("📅 After market hours. Bot will wait for tomorrow.")
            bot_status["status"] = "waiting_for_tomorrow"
            return
        else:
            bot_status["market_status"] = "Market hours"
        
        # Step 3: Auto-login with Selenium
        add_log("🔐 Starting Kite auto-login...")
        bot_status["status"] = "authenticating"
        
        if not SELENIUM_AVAILABLE:
            bot_status["status"] = "error"
            bot_status["error"] = "Selenium not available"
            add_log("❌ Selenium not installed!")
            return
        
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
            bot_status["error"] = "Auto-login failed - check credentials"
            add_log("❌ Auto-login failed!")
            return
        
        bot_status["authenticated"] = True
        bot_status["kite_user"] = creds["user_id"]
        add_log(f"✅ Authenticated successfully!")
        
        # Step 4: Wait for market if needed
        now = now_ist()
        if now < market_open:
            mins_to_wait = int((market_open - now).total_seconds() / 60)
            add_log(f"⏳ Waiting {mins_to_wait} mins for market open at 9:15 AM...")
            bot_status["status"] = "waiting_for_market"
            
            while now_ist() < market_open:
                time.sleep(30)
        
        # Step 5: Start trading bot
        add_log("📊 Starting trading bot...")
        bot_status["status"] = "running"
        
        bot_instance = SupertrendBot()
        bot_instance.kite = auto_login.kite
        bot_instance.is_running = True
        
        # Notify via Telegram
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
        
        add_log("📈 Market closed. Generating report...")
        bot_instance.generate_report()
        
        if bot_instance.ticker:
            bot_instance.ticker.close()
        
        add_log("✅ Trading session complete!")
        bot_status["status"] = "session_complete"
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        bot_status["status"] = "error"
        bot_status["error"] = error_msg
        add_log(f"❌ Bot error: {error_msg}")
        traceback.print_exc()


def start_bot_thread():
    """Start the trading bot in a background thread."""
    global bot_thread
    
    if bot_thread is not None and bot_thread.is_alive():
        add_log("⚠️ Bot thread already running")
        return
    
    bot_thread = threading.Thread(target=run_trading_bot, name="TradingBot")
    bot_thread.daemon = True
    bot_thread.start()
    add_log("✅ Trading bot thread launched")


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    add_log("🌐 FastAPI starting up...")
    # Start bot in background - doesn't block FastAPI
    start_bot_thread()
    yield
    add_log("🛑 FastAPI shutting down...")


app = FastAPI(
    title="Supertrend Trading Bot",
    description="NIFTY & BANKNIFTY Options Trading Bot",
    version="1.0.0",
    lifespan=lifespan
)


# --- HEALTH CHECK ROUTES (respond immediately) ---

@app.get("/", response_class=PlainTextResponse)
@app.head("/")
async def health_check():
    """Main health check - responds immediately."""
    bot_status["last_health_check"] = now_ist().isoformat()
    status = bot_status.get("status", "unknown")
    return f"Bot status: {status}"


@app.get("/ping", response_class=PlainTextResponse)
@app.head("/ping")
async def ping():
    """Simple ping - for UptimeRobot."""
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
