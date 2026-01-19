#!/usr/bin/env python3
"""
Supertrend Trading Bot for NIFTY & BANKNIFTY
Uses ATR-based Supertrend indicator for trend detection.

Configuration:
- ATR Period: 10
- ATR Multiplier: 3.0
- Timeframe: 5 minutes
- NIFTY: Target +20%, SL -5%
- BANKNIFTY: Target +10%, SL -5%

Usage:
    python3 main.py
"""
import time
import signal
import sys
import webbrowser
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from threading import Event, Lock
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Import Telegram notifier
try:
    from telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramNotifier = None

try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

CORPUS = 100000
TIMEFRAME_MINUTES = 5

# Supertrend Parameters
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0

# Securities configuration
SECURITIES = {
    "NIFTY": {
        "name": "NIFTY 50",
        "instrument_token": 256265,
        "lot_size": 50,
        "strike_interval": 50,
        "option_prefix": "NIFTY",
        "target_pct": 20,
        "sl_pct": 5,
    },
    "BANKNIFTY": {
        "name": "BANK NIFTY",
        "instrument_token": 260105,
        "lot_size": 25,
        "strike_interval": 100,
        "option_prefix": "BANKNIFTY",
        "target_pct": 10,
        "sl_pct": 5,
    },
}


def now_ist():
    if IST:
        return datetime.now(IST)
    return datetime.now()


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Supertrend indicator.
    Returns DataFrame with 'supertrend', 'trend', and 'signal' columns.
    """
    df = df.copy()
    
    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # HL2
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # Basic bands
    df['basic_upper'] = df['hl2'] - (multiplier * df['atr'])
    df['basic_lower'] = df['hl2'] + (multiplier * df['atr'])
    
    # Initialize
    df['upper_band'] = 0.0
    df['lower_band'] = 0.0
    df['trend'] = 1
    df['supertrend'] = 0.0
    
    # Calculate iteratively
    for i in range(period, len(df)):
        # Upper band
        if df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'upper_band'] = max(df['basic_upper'].iloc[i], df['upper_band'].iloc[i-1])
        else:
            df.loc[df.index[i], 'upper_band'] = df['basic_upper'].iloc[i]
        
        # Lower band
        if df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'lower_band'] = min(df['basic_lower'].iloc[i], df['lower_band'].iloc[i-1])
        else:
            df.loc[df.index[i], 'lower_band'] = df['basic_lower'].iloc[i]
        
        # Trend
        if df['trend'].iloc[i-1] == -1 and df['close'].iloc[i] > df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = 1
        elif df['trend'].iloc[i-1] == 1 and df['close'].iloc[i] < df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
        
        # Supertrend value
        if df['trend'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
    
    # Signals
    df['prev_trend'] = df['trend'].shift(1)
    df['signal'] = None
    df.loc[(df['trend'] == 1) & (df['prev_trend'] == -1), 'signal'] = 'BUY'
    df.loc[(df['trend'] == -1) & (df['prev_trend'] == 1), 'signal'] = 'SELL'
    
    return df


class Position:
    """Trading position."""
    def __init__(self, security: str, option_type: str, strike: int,
                 entry_price: float, target: float, sl: float, 
                 quantity: int, entry_time: datetime, spot_at_entry: float):
        self.security = security
        self.option_type = option_type
        self.strike = strike
        self.entry_price = entry_price
        self.target = target
        self.sl = sl
        self.quantity = quantity
        self.entry_time = entry_time
        self.spot_at_entry = spot_at_entry
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
    
    def close(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_time = now_ist()
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.quantity


class SupertrendTrader:
    """Handles Supertrend trading for one security."""
    
    def __init__(self, symbol: str, config: Dict, logger, telegram: 'TelegramNotifier' = None):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.telegram = telegram
        
        self.candles: List[Dict] = []
        self.current_candle: Optional[Dict] = None
        self.last_candle_time: Optional[datetime] = None
        
        self.current_trend: int = 0
        self.supertrend_value: float = 0
        
        self.position: Optional[Position] = None
        self.trades: List[Position] = []
        self.signals: List[Dict] = []
        
        self.lock = Lock()
    
    def process_tick(self, ltp: float, tick_time: datetime):
        """Process incoming tick and build candles."""
        with self.lock:
            candle_minute = (tick_time.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
            candle_start = tick_time.replace(minute=candle_minute, second=0, microsecond=0)
            
            if self.last_candle_time and candle_start > self.last_candle_time:
                if self.current_candle:
                    self._on_candle_close(self.current_candle)
                
                self.current_candle = {
                    "timestamp": candle_start,
                    "open": ltp, "high": ltp, "low": ltp, "close": ltp
                }
            elif self.current_candle is None:
                self.current_candle = {
                    "timestamp": candle_start,
                    "open": ltp, "high": ltp, "low": ltp, "close": ltp
                }
            else:
                self.current_candle["high"] = max(self.current_candle["high"], ltp)
                self.current_candle["low"] = min(self.current_candle["low"], ltp)
                self.current_candle["close"] = ltp
            
            self.last_candle_time = candle_start
            
            if self.position:
                self._check_exit(ltp)
    
    def _on_candle_close(self, candle: Dict):
        """Handle candle close."""
        self.candles.append(candle)
        if len(self.candles) > 100:
            self.candles = self.candles[-100:]
        
        if len(self.candles) < ATR_PERIOD + 2:
            return
        
        df = pd.DataFrame(self.candles)
        df = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
        
        latest = df.iloc[-1]
        self.current_trend = int(latest['trend'])
        self.supertrend_value = latest['supertrend']
        signal = latest['signal']
        
        trend_str = "🟢 BULLISH" if self.current_trend == 1 else "🔴 BEARISH"
        self.logger(f"[{self.symbol}] {candle['timestamp'].strftime('%H:%M')} | C:{candle['close']:.2f} | ST:{self.supertrend_value:.2f} | {trend_str}")
        
        if signal:
            self._execute_signal(signal, candle)
    
    def _execute_signal(self, signal: str, candle: Dict):
        """Execute Supertrend signal."""
        spot = candle["close"]
        new_type = "CE" if signal == "BUY" else "PE"
        
        # Position reversal
        if self.position:
            if (self.position.option_type == "CE" and signal == "SELL") or \
               (self.position.option_type == "PE" and signal == "BUY"):
                
                move = spot - self.position.spot_at_entry
                delta = 0.5 if self.position.option_type == "CE" else -0.5
                exit_price = self.position.entry_price + (move * delta)
                
                self.position.close(exit_price, "REVERSAL")
                self.trades.append(self.position)
                
                emoji = "✅" if self.position.pnl > 0 else "🛑"
                self.logger(f"[{self.symbol}] {emoji} REVERSAL | P&L: ₹{self.position.pnl:.2f}")
                print(f"\n🔄 [{self.symbol}] REVERSAL | P&L: ₹{self.position.pnl:.2f}")
                
                # Telegram notification for exit
                if self.telegram:
                    self.telegram.notify_trade_exit(
                        self.symbol, self.position.option_type, self.position.strike,
                        self.position.entry_price, exit_price, self.position.pnl, "REVERSAL"
                    )
                
                self.position = None
            else:
                return
        
        # New position
        strike = round(spot / self.config["strike_interval"]) * self.config["strike_interval"]
        itm = max(0, spot - strike) if new_type == "CE" else max(0, strike - spot)
        entry = itm + spot * 0.003 + 20
        
        target = entry * (1 + self.config["target_pct"] / 100)
        sl = entry * (1 - self.config["sl_pct"] / 100)
        
        self.position = Position(
            security=self.symbol,
            option_type=new_type,
            strike=strike,
            entry_price=entry,
            target=target,
            sl=sl,
            quantity=self.config["lot_size"],
            entry_time=now_ist(),
            spot_at_entry=spot
        )
        
        self.signals.append({
            "time": str(candle["timestamp"]),
            "signal": signal,
            "spot": spot,
            "supertrend": self.supertrend_value
        })
        
        emoji = "🟢" if signal == "BUY" else "🔴"
        self.logger(f"[{self.symbol}] {emoji} {signal} | {new_type} {strike} @ ₹{entry:.2f}")
        
        print(f"\n{'='*50}")
        print(f"{emoji} [{self.symbol}] SUPERTREND {signal} - {new_type}")
        print(f"   Strike: {strike} | Entry: ₹{entry:.2f}")
        print(f"   Target: ₹{target:.2f} | SL: ₹{sl:.2f}")
        print(f"{'='*50}\n")
        
        # Telegram notification for entry
        if self.telegram:
            self.telegram.notify_trade_entry(
                self.symbol, new_type, strike, entry, target, sl,
                self.config["lot_size"], signal
            )
    
    def _check_exit(self, current_ltp: float):
        """Check for SL/Target."""
        if not self.position:
            return
        
        move = current_ltp - self.position.spot_at_entry
        delta = 0.5 if self.position.option_type == "CE" else -0.5
        opt_price = self.position.entry_price + (move * delta)
        
        exit_price = None
        reason = None
        
        if opt_price >= self.position.target:
            exit_price = self.position.target
            reason = "TARGET"
        elif opt_price <= self.position.sl:
            exit_price = self.position.sl
            reason = "SL"
        
        if exit_price:
            self.position.close(exit_price, reason)
            self.trades.append(self.position)
            
            emoji = "✅" if self.position.pnl > 0 else "🛑"
            self.logger(f"[{self.symbol}] {emoji} {reason} | P&L: ₹{self.position.pnl:.2f}")
            print(f"\n{emoji} [{self.symbol}] {reason} | P&L: ₹{self.position.pnl:.2f}\n")
            
            # Telegram notification for exit
            if self.telegram:
                self.telegram.notify_trade_exit(
                    self.symbol, self.position.option_type, self.position.strike,
                    self.position.entry_price, exit_price, self.position.pnl, reason
                )
            
            self.position = None


class SupertrendBot:
    """Main Supertrend trading bot."""
    
    def __init__(self):
        self.is_running = False
        self.stop_event = Event()
        
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        
        # Initialize Telegram
        self.telegram = TelegramNotifier() if TELEGRAM_AVAILABLE else None
        
        self.traders = {
            symbol: SupertrendTrader(symbol, config, self._log, self.telegram)
            for symbol, config in SECURITIES.items()
        }
        
        self.log_file = LOG_DIR / f"supertrend_{now_ist().strftime('%Y%m%d')}.log"
    
    def _log(self, message: str):
        timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {message}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
    
    def _load_credentials(self) -> tuple:
        """Load API credentials from env vars or file."""
        api_key = os.environ.get("KITE_API_KEY", "")
        api_secret = os.environ.get("KITE_API_SECRET", "")
        
        if not api_key or not api_secret:
            api_file = BASE_DIR / "api_key.txt"
            if api_file.exists():
                lines = api_file.read_text().strip().split("\n")
                api_key = lines[0].strip()
                api_secret = lines[1].strip() if len(lines) > 1 else ""
        
        return api_key, api_secret
    
    def authenticate(self) -> bool:
        if not KITE_AVAILABLE:
            print("❌ Run: pip install kiteconnect")
            return False
        
        try:
            api_key, api_secret = self._load_credentials()
            self.kite = KiteConnect(api_key=api_key)
            
            login_url = self.kite.login_url()
            print("\n" + "=" * 60)
            print("🔐 KITE CONNECT LOGIN")
            print("=" * 60)
            print(f"URL: {login_url}")
            
            try:
                webbrowser.open(login_url)
            except:
                pass
            
            request_token = input("\nEnter request_token: ").strip()
            if not request_token:
                return False
            
            data = self.kite.generate_session(request_token, api_secret)
            self.kite.set_access_token(data["access_token"])
            
            self._log(f"✅ Login: {data.get('user_name', 'N/A')}")
            return True
            
        except Exception as e:
            self._log(f"❌ Auth failed: {e}")
            return False
    
    def fetch_historical(self):
        for symbol, trader in self.traders.items():
            try:
                config = SECURITIES[symbol]
                to_date = now_ist()
                from_date = to_date - timedelta(hours=4)
                
                data = self.kite.historical_data(
                    instrument_token=config["instrument_token"],
                    from_date=from_date,
                    to_date=to_date,
                    interval="5minute"
                )
                
                for candle in data[-50:]:
                    trader.candles.append({
                        "timestamp": candle["date"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"]
                    })
                
                self._log(f"Loaded {len(trader.candles)} candles for {symbol}")
                
            except Exception as e:
                self._log(f"Error fetching {symbol}: {e}")
    
    def start_live_feed(self):
        api_key, _ = self._load_credentials()
        self.ticker = KiteTicker(api_key=api_key, access_token=self.kite.access_token)
        
        tokens = [config["instrument_token"] for config in SECURITIES.values()]
        token_to_symbol = {config["instrument_token"]: symbol for symbol, config in SECURITIES.items()}
        
        def on_connect(ws, response):
            self._log(f"Connected: {list(SECURITIES.keys())}")
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
        
        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick.get("instrument_token")
                ltp = tick.get("last_price")
                tick_time = tick.get("timestamp") or now_ist()
                
                if token in token_to_symbol and ltp:
                    self.traders[token_to_symbol[token]].process_tick(ltp, tick_time)
        
        def on_close(ws, code, reason):
            self._log(f"Disconnected: {code}")
        
        self.ticker.on_connect = on_connect
        self.ticker.on_ticks = on_ticks
        self.ticker.on_close = on_close
        
        self.ticker.connect(threaded=True)
    
    def is_market_open(self) -> bool:
        now = now_ist()
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        return market_open <= now <= market_close
    
    def wait_for_market(self):
        while not self.is_market_open() and self.is_running:
            now = now_ist()
            next_open = now.replace(hour=9, minute=15, second=0)
            if now >= next_open:
                next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            
            wait = next_open - now
            hours = int(wait.total_seconds() // 3600)
            mins = int((wait.total_seconds() % 3600) // 60)
            
            print(f"⏳ Market opens in: {hours}h {mins}m")
            self.stop_event.wait(min(300, wait.total_seconds()))
    
    def generate_report(self):
        today = now_ist().strftime("%Y-%m-%d")
        total_pnl = 0
        securities_data = {}
        
        print(f"\n{'='*60}")
        print(f"📊 DAILY REPORT - {today}")
        print(f"{'='*60}")
        
        for symbol, trader in self.traders.items():
            pnl = sum(t.pnl for t in trader.trades)
            total_pnl += pnl
            wins = len([t for t in trader.trades if t.pnl > 0])
            losses = len([t for t in trader.trades if t.pnl < 0])
            print(f"{symbol}: {len(trader.trades)} trades | W:{wins} L:{losses} | P&L: ₹{pnl:,.2f}")
            
            securities_data[symbol] = {
                "trades": len(trader.trades),
                "pnl": round(pnl, 2),
                "wins": wins,
                "losses": losses
            }
        
        print(f"\n📈 TOTAL P&L: ₹{total_pnl:,.2f}")
        print(f"{'='*60}\n")
        
        # Save report
        report = {"date": today, "total_pnl": round(total_pnl, 2), "securities": securities_data}
        
        report_path = LOG_DIR / f"report_{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Send Telegram daily summary
        if self.telegram:
            self.telegram.notify_daily_summary(today, securities_data, total_pnl)
    
    def run(self):
        self.is_running = True
        
        def signal_handler(sig, frame):
            print("\n⏹️ Stopping...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\n" + "=" * 60)
        print("🚀 SUPERTREND TRADING BOT")
        print("=" * 60)
        print(f"Indicator: Supertrend (ATR:{ATR_PERIOD}, Mult:{ATR_MULTIPLIER})")
        print(f"Timeframe: {TIMEFRAME_MINUTES}min | Corpus: ₹{CORPUS:,}")
        print(f"NIFTY: +{SECURITIES['NIFTY']['target_pct']}%/-{SECURITIES['NIFTY']['sl_pct']}%")
        print(f"BANKNIFTY: +{SECURITIES['BANKNIFTY']['target_pct']}%/-{SECURITIES['BANKNIFTY']['sl_pct']}%")
        print("=" * 60 + "\n")
        
        # Skip authenticate() if kite is already set (from app.py auto-login)
        if self.kite is None:
            if not self.authenticate():
                return
        
        if not self.is_market_open():
            self.wait_for_market()
        
        if not self.is_running:
            return
        
        self._log("Starting trading session...")
        
        # Telegram notification for bot start
        if self.telegram:
            self.telegram.notify_bot_start(list(SECURITIES.keys()))
        
        self.fetch_historical()
        self.start_live_feed()
        
        while self.is_running and self.is_market_open():
            time.sleep(1)
        
        self._log("Session ended.")
        self.generate_report()
        
        if self.ticker:
            self.ticker.close()
    
    def stop(self):
        self.is_running = False
        self.stop_event.set()
        
        for trader in self.traders.values():
            if trader.position:
                trader.position.close(trader.position.entry_price, "MANUAL_STOP")
                trader.trades.append(trader.position)
                trader.position = None


def main():
    bot = SupertrendBot()
    bot.run()


if __name__ == "__main__":
    main()
