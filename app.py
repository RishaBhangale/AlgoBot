#!/usr/bin/env python3
"""
Supertrend Bot Web Server Wrapper (FastAPI)
Disguises the trading bot as a web service for Render free tier.

The FastAPI server handles health checks while the actual trading bot
runs in a background thread.

Usage:
    Local: uvicorn app:app --host 0.0.0.0 --port 5000
    Render: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import threading
import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Import the trading bot
from main import SupertrendBot, SECURITIES, now_ist

# Global state
bot_instance = None
bot_thread = None
bot_status = {
    "status": "initialized",
    "started_at": None,
    "last_health_check": None,
    "trades_today": 0,
    "total_pnl": 0
}


def run_trading_bot():
    """Run the trading bot in background thread."""
    global bot_instance, bot_status
    
    print("🚀 Trading bot thread started...")
    bot_status["status"] = "running"
    bot_status["started_at"] = now_ist().isoformat()
    
    try:
        bot_instance = SupertrendBot()
        bot_instance.run()
    except Exception as e:
        print(f"❌ Bot error: {e}")
        bot_status["status"] = f"error: {str(e)}"
    finally:
        bot_status["status"] = "stopped"


def start_bot_thread():
    """Start the trading bot in a background thread."""
    global bot_thread
    
    if bot_thread is not None and bot_thread.is_alive():
        print("⚠️ Bot thread already running")
        return
    
    bot_thread = threading.Thread(target=run_trading_bot, name="TradingBot")
    bot_thread.daemon = True
    bot_thread.start()
    print("✅ Trading bot thread started")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    start_bot_thread()
    yield
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(
    title="Supertrend Trading Bot",
    description="NIFTY & BANKNIFTY Options Trading Bot",
    version="1.0.0",
    lifespan=lifespan
)


# --- HEALTH CHECK ROUTES ---

@app.get("/", response_class=PlainTextResponse)
@app.head("/")
async def health_check():
    """Main health check endpoint for UptimeRobot."""
    bot_status["last_health_check"] = now_ist().isoformat()
    return "Bot is running!"


@app.get("/ping", response_class=PlainTextResponse)
@app.head("/ping")
async def ping():
    """Simple ping endpoint."""
    return "pong"


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to avoid 404s."""
    return ""


@app.get("/status")
async def detailed_status():
    """Detailed bot status."""
    global bot_instance, bot_status
    
    status = {
        "status": bot_status["status"],
        "started_at": bot_status["started_at"],
        "last_health_check": bot_status["last_health_check"],
        "current_time": now_ist().isoformat(),
        "securities": list(SECURITIES.keys())
    }
    
    # Get trade info if bot is running
    if bot_instance and hasattr(bot_instance, 'traders'):
        status["positions"] = {}
        status["trades_today"] = {}
        
        for symbol, trader in bot_instance.traders.items():
            status["positions"][symbol] = {
                "has_position": trader.position is not None,
                "option_type": trader.position.option_type if trader.position else None,
                "entry_price": trader.position.entry_price if trader.position else None
            }
            status["trades_today"][symbol] = len(trader.trades)
    
    return status


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
