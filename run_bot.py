#!/usr/bin/env python3
"""
Supertrend Bot Runner with Auto-Login
Combines auto-login with the trading bot for fully automated operation.

Usage:
    python3 run_bot.py

This script:
1. Attempts to use saved token if valid
2. If no valid token, performs auto-login via Selenium
3. Starts the Supertrend trading bot
4. Waits for market open and trades
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from auto_login import KiteAutoLogin, load_credentials, log, now_ist
from main import SupertrendBot, SECURITIES, KITE_AVAILABLE

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None


def get_access_token() -> str:
    """Get access token via auto-login or saved token."""
    creds = load_credentials()
    
    if not creds["api_key"] or not creds["api_secret"]:
        log("❌ Missing API key/secret in api_key.txt")
        return None
    
    auto_login = KiteAutoLogin(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        user_id=creds["user_id"],
        password=creds["password"],
        totp_secret=creds["totp_secret"],
        headless=True  # Run headless on server
    )
    
    # Try saved token first
    saved_token = auto_login.get_saved_token()
    if saved_token:
        log("✅ Using saved access token")
        return saved_token
    
    # Need fresh login
    if not creds["user_id"] or not creds["password"]:
        log("❌ No valid token and no credentials for auto-login")
        log("   Add credentials to .env file")
        return None
    
    log("🔄 Performing auto-login...")
    return auto_login.login()


class AutoLoginBot(SupertrendBot):
    """Supertrend bot with auto-login capability."""
    
    def authenticate(self) -> bool:
        """Override to use auto-login."""
        if not KITE_AVAILABLE:
            log("❌ kiteconnect not installed")
            return False
        
        try:
            # Load API credentials
            api_file = Path(__file__).parent / "api_key.txt"
            lines = api_file.read_text().strip().split("\n")
            api_key = lines[0].strip()
            api_secret = lines[1].strip()
            
            self.kite = KiteConnect(api_key=api_key)
            
            # Get access token via auto-login
            access_token = get_access_token()
            
            if not access_token:
                log("❌ Failed to get access token")
                return False
            
            self.kite.set_access_token(access_token)
            
            # Verify token works
            try:
                profile = self.kite.profile()
                log(f"✅ Logged in as: {profile.get('user_name', 'N/A')}")
                return True
            except Exception as e:
                log(f"❌ Token invalid: {e}")
                # Try fresh login
                creds = load_credentials()
                auto_login = KiteAutoLogin(
                    api_key=api_key,
                    api_secret=api_secret,
                    user_id=creds["user_id"],
                    password=creds["password"],
                    totp_secret=creds["totp_secret"],
                    headless=True
                )
                access_token = auto_login.login()
                
                if access_token:
                    self.kite.set_access_token(access_token)
                    return True
                return False
                
        except Exception as e:
            log(f"❌ Authentication failed: {e}")
            return False


def wait_until_login_time():
    """Wait until appropriate time to login (before market opens)."""
    now = now_ist()
    
    # Target login time: 9:00 AM IST (15 minutes before market open)
    login_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    
    if now >= login_time:
        # Already past login time today
        if now.hour >= 15 and now.minute >= 30:
            # Market closed, wait for tomorrow
            login_time += timedelta(days=1)
        else:
            # Market is open or about to open, login now
            return
    
    # Skip weekends
    while login_time.weekday() >= 5:
        login_time += timedelta(days=1)
    
    wait_seconds = (login_time - now).total_seconds()
    
    if wait_seconds > 0:
        hours = int(wait_seconds // 3600)
        mins = int((wait_seconds % 3600) // 60)
        log(f"⏳ Waiting for login time: {hours}h {mins}m")
        
        while wait_seconds > 0:
            time.sleep(min(300, wait_seconds))  # Sleep max 5 minutes at a time
            now = now_ist()
            wait_seconds = (login_time - now).total_seconds()


def main():
    print("\n" + "=" * 60)
    print("🚀 SUPERTREND BOT WITH AUTO-LOGIN")
    print("=" * 60)
    print("Mode: Fully Automated")
    print("Login: Selenium Auto-Login")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Wait for appropriate login time
            wait_until_login_time()
            
            # Run bot
            bot = AutoLoginBot()
            bot.run()
            
            # After market close, wait for next day
            now = now_ist()
            next_day = now.replace(hour=8, minute=55, second=0) + timedelta(days=1)
            
            # Skip weekends
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            
            wait_seconds = (next_day - now).total_seconds()
            hours = int(wait_seconds // 3600)
            mins = int((wait_seconds % 3600) // 60)
            
            log(f"📅 Next session in: {hours}h {mins}m")
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            log("⏹️ Bot stopped by user")
            break
        except Exception as e:
            log(f"❌ Error: {e}")
            log("🔄 Restarting in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main()
