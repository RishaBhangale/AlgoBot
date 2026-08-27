#!/usr/bin/env python3
"""
Quad-Confirmation + Alligator Golden Zone Stock Options Bot Runner
Combines fast headless auto-login with autonomous daily scheduling.

Usage:
    python3 run_bot.py

This script:
1. Validates or refreshes Kite Connect access token automatically at 08:50 AM IST.
2. Automatically launches the Quad-Confirmation + Alligator Golden Zone Stock Algo at 09:15 AM IST.
3. Trades RELIANCE, ICICIBANK, SBIN, AXISBANK, LT through market close (15:30 IST).
4. Sends real-time trade alerts (entry/exit/SL) and lot size tags to Telegram.
5. Auto squares off open positions at 15:15 IST.
6. Generates full EOD P&L report, saves JSON logs in logs/, and pings daily summary to Telegram.
7. Automatically sleeps and repeats for the next trading day.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from auto_login import KiteAutoLogin, load_credentials, log, now_ist
from main_stocks import StockOptionsBot, STOCKS, KITE_AVAILABLE

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None


def ensure_fresh_token() -> bool:
    """Ensure a valid access token exists, refreshing headlessly if necessary."""
    creds = load_credentials()
    if not creds.get("api_key") or not creds.get("api_secret"):
        log("❌ Missing API credentials in api_key.txt or .env")
        return False
        
    auto_login = KiteAutoLogin(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        user_id=creds.get("user_id"),
        password=creds.get("password"),
        totp_secret=creds.get("totp_secret"),
        headless=True
    )
    
    # Try saved token first
    saved_token = auto_login.get_saved_token()
    if saved_token:
        try:
            kite = KiteConnect(api_key=creds["api_key"])
            kite.set_access_token(saved_token)
            profile = kite.profile()
            log(f"✅ Reusing valid saved token. Logged in as: {profile.get('user_name', 'N/A')}")
            return True
        except Exception:
            log("⚠️ Saved token expired, generating fresh login...")
            
    # Fresh login
    token = auto_login.login()
    if token:
        log("✅ Fresh access token generated successfully.")
        return True
    else:
        log("❌ Failed to obtain access token.")
        return False


def wait_until_market_open():
    """Wait until 09:14 AM IST before starting the live trading session."""
    now = now_ist()
    
    # If today is weekend (Saturday=5, Sunday=6), advance to Monday 09:14
    target = now.replace(hour=9, minute=14, second=0, microsecond=0)
    
    if now >= target:
        if now.hour >= 15 and now.minute >= 30:
            # Market closed for the day, target next morning
            target += timedelta(days=1)
            target = target.replace(hour=9, minute=14, second=0, microsecond=0)
        else:
            # Market is open or about to open (between 09:14 and 15:30)
            return
            
    while target.weekday() >= 5:
        target += timedelta(days=1)
        
    wait_secs = (target - now).total_seconds()
    if wait_secs > 0:
        hrs = int(wait_secs // 3600)
        mins = int((wait_secs % 3600) // 60)
        log(f"⏳ Market opens at 09:15 AM. Waiting {hrs}h {mins}m until {target.strftime('%Y-%m-%d %H:%M')} IST...")
        
        while wait_secs > 0:
            sleep_chunk = min(300, wait_secs)
            time.sleep(sleep_chunk)
            now = now_ist()
            wait_secs = (target - now).total_seconds()


def main():
    print("\n" + "=" * 70)
    print("🚀 AUTONOMOUS STOCK OPTIONS TRADING ENGINE")
    print("   Strategy: Quad-Confirmation + Alligator Golden Zone Overlay")
    print("   Securities: " + ", ".join(STOCKS.keys()))
    print("   Mode: Automated Daily Runner with Real-Time Telegram Alerts")
    print("=" * 70 + "\n")
    
    while True:
        try:
            now = now_ist()
            # If weekend, wait until Monday morning
            if now.weekday() >= 5:
                wait_until_market_open()
                continue
                
            # If market is already closed today, wait until tomorrow
            if now.hour >= 15 and now.minute >= 30:
                wait_until_market_open()
                continue
                
            # Ensure fresh authentication
            log("🔐 Checking Zerodha Kite Connect session...")
            if not ensure_fresh_token():
                log("⚠️ Authentication failed. Retrying in 60 seconds...")
                time.sleep(60)
                continue
                
            # Wait until 09:14 AM if early morning
            wait_until_market_open()
            
            # Start Stock Bot
            log("🏁 Starting StockOptionsBot session for today...")
            bot = StockOptionsBot()
            bot.run()
            
            # After market close (15:30), sleep until tomorrow morning
            log("🏁 Daily session concluded. Standing by for next market session...")
            wait_until_market_open()
            
        except KeyboardInterrupt:
            log("⏹️ Bot runner stopped by user.")
            break
        except Exception as e:
            log(f"❌ Unexpected error in runner loop: {e}")
            log("🔄 Restarting runner in 30 seconds...")
            time.sleep(30)


if __name__ == "__main__":
    main()
