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
import os
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
# Target: 20%, SL: 10% for all
TARGET_PCT = 20
SL_PCT = 10

SECURITIES = {
    "NIFTY": {
        "name": "NIFTY 50",
        "instrument_token": 256265,
        "lot_size": 50,
        "strike_interval": 50,
        "option_prefix": "NIFTY",
        "target_pct": TARGET_PCT,
        "sl_pct": SL_PCT,
    },
    "BANKNIFTY": {
        "name": "BANK NIFTY",
        "instrument_token": 260105,
        "lot_size": 25,
        "strike_interval": 100,
        "option_prefix": "BANKNIFTY",
        "target_pct": TARGET_PCT,
        "sl_pct": SL_PCT,
    },
}


def now_ist():
    if IST:
        return datetime.now(IST)
    return datetime.now()


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Supertrend indicator - MATCHES TRADINGVIEW PINESCRIPT EXACTLY.
    
    TradingView naming:
    - up = hl2 - (mult * atr) = SUPPORT line (shown when bullish)
    - dn = hl2 + (mult * atr) = RESISTANCE line (shown when bearish)
    
    Trend switching:
    - Bearish to Bullish: close > dn (crosses above resistance)
    - Bullish to Bearish: close < up (crosses below support)
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
    
    # ATR (using SMA like TradingView's atr() function)
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    
    # HL2 (source)
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # Basic bands (TradingView naming)
    # up = support line, dn = resistance line
    df['basic_up'] = df['hl2'] - (multiplier * df['atr'])  # Support
    df['basic_dn'] = df['hl2'] + (multiplier * df['atr'])  # Resistance
    
    # Initialize final bands
    df['up'] = df['basic_up']  # Support (plotted when bullish)
    df['dn'] = df['basic_dn']  # Resistance (plotted when bearish)
    df['trend'] = 1
    df['supertrend'] = 0.0
    
    # Calculate iteratively (matching PineScript logic exactly)
    for i in range(1, len(df)):
        # Up band (support) - only moves UP, never down
        # up := close[1] > up1 ? max(up, up1) : up
        if df['close'].iloc[i-1] > df['up'].iloc[i-1]:
            df.loc[df.index[i], 'up'] = max(df['basic_up'].iloc[i], df['up'].iloc[i-1])
        else:
            df.loc[df.index[i], 'up'] = df['basic_up'].iloc[i]
        
        # Down band (resistance) - only moves DOWN, never up
        # dn := close[1] < dn1 ? min(dn, dn1) : dn
        if df['close'].iloc[i-1] < df['dn'].iloc[i-1]:
            df.loc[df.index[i], 'dn'] = min(df['basic_dn'].iloc[i], df['dn'].iloc[i-1])
        else:
            df.loc[df.index[i], 'dn'] = df['basic_dn'].iloc[i]
        
        # Trend switching (EXACTLY like TradingView)
        # trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        prev_trend = df['trend'].iloc[i-1]
        if prev_trend == -1 and df['close'].iloc[i] > df['dn'].iloc[i-1]:
            # Bearish to Bullish: close crosses ABOVE resistance (dn)
            df.loc[df.index[i], 'trend'] = 1
        elif prev_trend == 1 and df['close'].iloc[i] < df['up'].iloc[i-1]:
            # Bullish to Bearish: close crosses BELOW support (up)
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = prev_trend
        
        # Supertrend value (the line shown on chart)
        # Bullish: show support line (up)
        # Bearish: show resistance line (dn)
        if df['trend'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['up'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['dn'].iloc[i]
    
    # Rename for compatibility with rest of code
    df['upper_band'] = df['dn']  # Resistance
    df['lower_band'] = df['up']  # Support
    
    # Signals (trend change)
    df['prev_trend'] = df['trend'].shift(1)
    df['signal'] = None
    df.loc[(df['trend'] == 1) & (df['prev_trend'] == -1), 'signal'] = 'BUY'
    df.loc[(df['trend'] == -1) & (df['prev_trend'] == 1), 'signal'] = 'SELL'
    
    return df


class Position:
    """Trading position with hybrid trailing stop-loss."""
    def __init__(self, security: str, option_type: str, strike: int,
                 entry_price: float, target: float, sl: float, 
                 quantity: int, entry_time: datetime, spot_at_entry: float):
        self.security = security
        self.option_type = option_type
        self.strike = strike
        self.entry_price = entry_price
        self.target = target
        self.initial_sl = sl  # Original SL (never go back below this)
        self.sl = sl  # Current trailing SL
        self.quantity = quantity
        self.entry_time = entry_time
        self.spot_at_entry = spot_at_entry
        self.peak_spot = spot_at_entry  # Track highest (CE) or lowest (PE) spot
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
    
    def update_trailing_sl(self, current_spot: float, supertrend: float, atr: float) -> bool:
        """
        Update trailing SL using hybrid approach (Supertrend + ATR).
        Returns True if SL was updated.
        """
        # Update peak spot
        if self.option_type == "CE":
            self.peak_spot = max(self.peak_spot, current_spot)
        else:
            self.peak_spot = min(self.peak_spot, current_spot)
        
        # Calculate hybrid trailing SL
        new_sl = self._calculate_hybrid_sl(current_spot, supertrend, atr)
        
        # Only move SL in favorable direction (never backward)
        if self.option_type == "CE":
            # For CE: SL should only move UP
            if new_sl > self.sl:
                old_sl = self.sl
                self.sl = new_sl
                return True
        else:
            # For PE: SL should only move DOWN (higher is tighter for PE)
            # But since we track option premium, higher SL is still tighter
            if new_sl > self.sl:
                old_sl = self.sl
                self.sl = new_sl
                return True
        
        return False
    
    def _calculate_hybrid_sl(self, current_spot: float, supertrend: float, atr: float) -> float:
        """
        Hybrid trailing SL: Use TIGHTER of Supertrend-based or ATR-based.
        """
        # Calculate estimated option price based on spot movement
        delta = 0.5 if self.option_type == "CE" else -0.5
        spot_move = current_spot - self.spot_at_entry
        current_opt_price = self.entry_price + (spot_move * delta)
        
        # Option A: Supertrend-based SL (with small buffer)
        if self.option_type == "CE":
            # For CE: If spot drops to supertrend, exit
            st_spot_sl = supertrend - 0.5  # Small buffer
            st_move = st_spot_sl - self.spot_at_entry
            sl_option_a = max(0.05, self.entry_price + (st_move * delta))
        else:
            # For PE: If spot rises to supertrend, exit
            st_spot_sl = supertrend + 0.5
            st_move = st_spot_sl - self.spot_at_entry
            sl_option_a = max(0.05, self.entry_price + (st_move * delta))
        
        # Option B: ATR-based SL (2x ATR from peak)
        if self.option_type == "CE":
            atr_spot_sl = self.peak_spot - (atr * 2.0)
            atr_move = atr_spot_sl - self.spot_at_entry
            sl_option_b = max(0.05, self.entry_price + (atr_move * delta))
        else:
            atr_spot_sl = self.peak_spot + (atr * 2.0)
            atr_move = atr_spot_sl - self.spot_at_entry
            sl_option_b = max(0.05, self.entry_price + (atr_move * delta))
        
        # Use the HIGHER (more protective) SL
        new_sl = max(sl_option_a, sl_option_b)
        
        # Never go below initial SL
        return max(new_sl, self.initial_sl)
    
    def close(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_time = now_ist()
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.quantity


class SupertrendTrader:
    """
    Handles Supertrend trading for one security.
    
    Rules:
    - Enter position immediately based on current trend (BULLISH → CE, BEARISH → PE)
    - Target: 20%, SL: 10%
    - Hold till expiry (not daily)
    - Close only on: signal change, target hit, or SL hit
    """
    
    def __init__(self, symbol: str, config: Dict, logger, telegram: 'TelegramNotifier' = None):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.telegram = telegram
        
        self.candles: List[Dict] = []
        self.current_candle: Optional[Dict] = None
        self.last_candle_time: Optional[datetime] = None
        
        self.current_trend: int = 0  # 1 = BULLISH, -1 = BEARISH
        self.supertrend_value: float = 0
        self.current_atr: float = 0  # Track ATR for trailing SL
        self.initial_position_taken: bool = False  # Track if we've entered initial position
        
        self.position: Optional[Position] = None
        self.trades: List[Position] = []
        self.signals: List[Dict] = []
        
        self.tick_count: int = 0  # Track ticks received
        self.candle_count: int = 0  # Track candles processed
        
        self.lock = Lock()
    
    def process_tick(self, ltp: float, tick_time: datetime):
        """Process incoming tick and build candles."""
        with self.lock:
            self.tick_count += 1
            
            # Log every 100th tick to show activity
            if self.tick_count % 100 == 0:
                print(f"[TICK] [{self.symbol}] #{self.tick_count} LTP: {ltp:.2f}, Candles: {len(self.candles)}", flush=True)
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
            
            # Check for SL/Target on every tick (real-time monitoring)
            if self.position:
                self._check_exit(ltp)
    
    def _on_candle_close(self, candle: Dict):
        """Handle candle close - check trend, update trailing SL, and take action."""
        print(f"\n[DEBUG] [{self.symbol}] _on_candle_close called: {candle['timestamp']}", flush=True)
        
        self.candles.append(candle)
        if len(self.candles) > 100:
            self.candles = self.candles[-100:]
        
        print(f"[DEBUG] [{self.symbol}] Total candles: {len(self.candles)}, Need: {ATR_PERIOD + 2}", flush=True)
        
        if len(self.candles) < ATR_PERIOD + 2:
            print(f"[DEBUG] [{self.symbol}] Not enough candles yet, waiting...", flush=True)
            return
        
        df = pd.DataFrame(self.candles)
        df = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
        
        latest = df.iloc[-1]
        prev_trend = self.current_trend
        self.current_trend = int(latest['trend'])
        self.supertrend_value = latest['supertrend']
        self.current_atr = latest['atr']  # Save ATR for trailing SL
        
        trend_str = "🟢 BULLISH" if self.current_trend == 1 else "🔴 BEARISH"
        self.logger(f"[{self.symbol}] {candle['timestamp'].strftime('%H:%M')} | C:{candle['close']:.2f} | ST:{self.supertrend_value:.2f} | ATR:{self.current_atr:.2f} | {trend_str}")
        
        # UPDATE TRAILING SL on every candle close if we have a position
        if self.position:
            old_sl = self.position.sl
            sl_updated = self.position.update_trailing_sl(
                current_spot=candle['close'],
                supertrend=self.supertrend_value,
                atr=self.current_atr
            )
            if sl_updated:
                print(f"📈 [{self.symbol}] TRAILING SL UPDATED: ₹{old_sl:.2f} → ₹{self.position.sl:.2f}", flush=True)
                self.logger(f"[{self.symbol}] 📈 Trailing SL: ₹{old_sl:.2f} → ₹{self.position.sl:.2f}")
                
                # Notify via Telegram about SL update
                if self.telegram:
                    self.telegram.send_message(
                        f"📈 *{self.symbol} Trailing SL Updated*\n"
                        f"Old SL: ₹{old_sl:.2f}\n"
                        f"New SL: ₹{self.position.sl:.2f}\n"
                        f"Peak: ₹{self.position.peak_spot:.2f}"
                    )
        
        print(f"[DEBUG] [{self.symbol}] Trend: {self.current_trend}, Prev: {prev_trend}, Position: {self.position is not None}, InitialTaken: {self.initial_position_taken}", flush=True)
        
        # IMMEDIATE ENTRY: If no position yet, enter based on current trend
        if not self.initial_position_taken and self.current_trend != 0:
            signal = "BUY" if self.current_trend == 1 else "SELL"
            print(f"[DEBUG] [{self.symbol}] 🎯 TRIGGERING INITIAL ENTRY: {signal}", flush=True)
            self.logger(f"[{self.symbol}] 🎯 Initial entry based on current trend: {signal}")
            self._enter_position(signal, candle)
            self.initial_position_taken = True
            return
        
        # SIGNAL CHANGE: Close existing position and enter new one
        if prev_trend != 0 and self.current_trend != prev_trend:
            new_signal = "BUY" if self.current_trend == 1 else "SELL"
            print(f"\n🔄🔄🔄 [{self.symbol}] SIGNAL CHANGE DETECTED! 🔄🔄🔄", flush=True)
            print(f"    Previous: {'BULLISH' if prev_trend == 1 else 'BEARISH'} -> New: {'BULLISH' if self.current_trend == 1 else 'BEARISH'}", flush=True)
            self.logger(f"[{self.symbol}] 🔄 Trend changed! New signal: {new_signal}")
            
            # Close existing position if any
            if self.position:
                print(f"[DEBUG] [{self.symbol}] Closing existing position on signal change", flush=True)
                self._close_position(candle["close"], "SIGNAL_CHANGE")
            else:
                print(f"[DEBUG] [{self.symbol}] No existing position to close", flush=True)
            
            # Enter new position
            print(f"[DEBUG] [{self.symbol}] Entering new position: {new_signal}", flush=True)
            self._enter_position(new_signal, candle)
        else:
            # Log why signal change didn't trigger
            if prev_trend == 0:
                print(f"[DEBUG] [{self.symbol}] No signal change: prev_trend was 0", flush=True)
            elif self.current_trend == prev_trend:
                pass  # Same trend, no action needed (normal case)
    
    def _enter_position(self, signal: str, candle: Dict):
        """Enter a new position based on signal."""
        spot = candle["close"]
        option_type = "CE" if signal == "BUY" else "PE"
        
        # Calculate strike (ATM)
        strike = round(spot / self.config["strike_interval"]) * self.config["strike_interval"]
        
        # Estimate option premium (ITM value + time value)
        itm = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
        entry_price = itm + spot * 0.003 + 20  # Approximate premium
        
        # Target 20%, SL 10%
        target = entry_price * (1 + self.config["target_pct"] / 100)
        sl = entry_price * (1 - self.config["sl_pct"] / 100)
        
        self.position = Position(
            security=self.symbol,
            option_type=option_type,
            strike=strike,
            entry_price=entry_price,
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
        self.logger(f"[{self.symbol}] {emoji} {signal} | {option_type} {strike} @ ₹{entry_price:.2f}")
        
        print(f"\n{'='*50}", flush=True)
        print(f"{emoji} [{self.symbol}] NEW POSITION - {option_type}", flush=True)
        print(f"   Signal: {signal} (Trend: {'BULLISH' if signal == 'BUY' else 'BEARISH'})", flush=True)
        print(f"   Strike: {strike} | Entry: ₹{entry_price:.2f}", flush=True)
        print(f"   Target: ₹{target:.2f} (+{self.config['target_pct']}%)", flush=True)
        print(f"   SL: ₹{sl:.2f} (-{self.config['sl_pct']}%)", flush=True)
        print(f"   Qty: {self.config['lot_size']} | Hold: Till Expiry", flush=True)
        print(f"{'='*50}\n", flush=True)
        
        # Telegram notification
        if self.telegram:
            self.telegram.notify_trade_entry(
                self.symbol, option_type, strike, entry_price, target, sl,
                self.config["lot_size"], signal
            )
    
    def _close_position(self, current_spot: float, reason: str):
        """Close the current position."""
        if not self.position:
            return
        
        # Calculate exit price based on spot movement
        move = current_spot - self.position.spot_at_entry
        delta = 0.5 if self.position.option_type == "CE" else -0.5
        exit_price = max(0.05, self.position.entry_price + (move * delta))
        
        self.position.close(exit_price, reason)
        self.trades.append(self.position)
        
        emoji = "✅" if self.position.pnl > 0 else "🛑"
        self.logger(f"[{self.symbol}] {emoji} {reason} | P&L: ₹{self.position.pnl:.2f}")
        
        print(f"\n{emoji} [{self.symbol}] POSITION CLOSED", flush=True)
        print(f"   Reason: {reason}", flush=True)
        print(f"   Entry: ₹{self.position.entry_price:.2f} → Exit: ₹{exit_price:.2f}", flush=True)
        print(f"   P&L: ₹{self.position.pnl:,.2f}\n", flush=True)
        
        # Telegram notification
        if self.telegram:
            self.telegram.notify_trade_exit(
                self.symbol, self.position.option_type, self.position.strike,
                self.position.entry_price, exit_price, self.position.pnl, reason
            )
        
        self.position = None
    
    def _check_exit(self, current_ltp: float):
        """Check for SL/Target exit on every tick."""
        if not self.position:
            return
        
        # Estimate current option price based on spot movement
        move = current_ltp - self.position.spot_at_entry
        delta = 0.5 if self.position.option_type == "CE" else -0.5
        estimated_opt_price = self.position.entry_price + (move * delta)
        
        # Check target (20% profit)
        if estimated_opt_price >= self.position.target:
            self._close_position(current_ltp, "TARGET_HIT")
            return
        
        # Check SL (10% loss)
        if estimated_opt_price <= self.position.sl:
            self._close_position(current_ltp, "SL_HIT")
            return


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
        """Fetch historical candle data for all securities."""
        if not self.kite:
            print("❌ fetch_historical: Kite not initialized!", flush=True)
            self._log("❌ fetch_historical: Kite not initialized!")
            return
        
        for symbol, trader in self.traders.items():
            try:
                config = SECURITIES[symbol]
                to_date = now_ist()
                # Look back 5 days to ensure we get enough candles (covers weekends + holidays)
                from_date = to_date - timedelta(days=5)
                
                print(f"📊 Fetching {symbol} from {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}...", flush=True)
                
                data = self.kite.historical_data(
                    instrument_token=config["instrument_token"],
                    from_date=from_date,
                    to_date=to_date,
                    interval="5minute"
                )
                
                print(f"   Got {len(data)} candles from API", flush=True)
                
                for candle in data[-50:]:
                    trader.candles.append({
                        "timestamp": candle["date"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"]
                    })
                
                self._log(f"Loaded {len(trader.candles)} candles for {symbol}")
                print(f"✅ {symbol}: {len(trader.candles)} candles loaded", flush=True)
                
                # CALCULATE INITIAL SUPERTREND AND TAKE POSITION IMMEDIATELY
                if len(trader.candles) >= ATR_PERIOD + 2:
                    df = pd.DataFrame(trader.candles)
                    df = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
                    
                    latest = df.iloc[-1]
                    trader.current_trend = int(latest['trend'])
                    trader.supertrend_value = latest['supertrend']
                    trader.current_atr = latest['atr']  # Save ATR for trailing SL
                    
                    trend_str = "🟢 BULLISH" if trader.current_trend == 1 else "🔴 BEARISH"
                    print(f"📊 [{symbol}] Initial Supertrend: {trader.supertrend_value:.2f} | ATR: {trader.current_atr:.2f} | Trend: {trend_str}", flush=True)
                    self._log(f"[{symbol}] Initial trend: {trend_str} | ATR: {trader.current_atr:.2f}")
                    
                    # TAKE INITIAL POSITION NOW (don't wait for live candle)
                    if not trader.initial_position_taken and trader.current_trend != 0:
                        signal = "BUY" if trader.current_trend == 1 else "SELL"
                        print(f"🎯 [{symbol}] TAKING INITIAL POSITION: {signal}", flush=True)
                        self._log(f"[{symbol}] 🎯 Initial entry: {signal}")
                        
                        # Use last candle data for entry
                        last_candle = trader.candles[-1]
                        trader._enter_position(signal, last_candle)
                        trader.initial_position_taken = True
                else:
                    print(f"⚠️ [{symbol}] Not enough candles for Supertrend ({len(trader.candles)} < {ATR_PERIOD + 2})", flush=True)
                
            except Exception as e:
                error_msg = f"Error fetching {symbol}: {type(e).__name__}: {e}"
                self._log(error_msg)
                print(f"❌ {error_msg}", flush=True)
    
    def start_live_feed(self):
        """Start WebSocket for live price feed."""
        api_key, _ = self._load_credentials()
        
        if not self.kite or not self.kite.access_token:
            print("❌ start_live_feed: No access token!", flush=True)
            self._log("❌ start_live_feed: No access token!")
            return
        
        print(f"🔌 Starting WebSocket with token: {self.kite.access_token[:10]}...", flush=True)
        
        self.ticker = KiteTicker(api_key=api_key, access_token=self.kite.access_token)
        
        tokens = [config["instrument_token"] for config in SECURITIES.values()]
        token_to_symbol = {config["instrument_token"]: symbol for symbol, config in SECURITIES.items()}
        
        def on_connect(ws, response):
            msg = f"WebSocket connected: {list(SECURITIES.keys())}"
            self._log(msg)
            print(f"✅ {msg}", flush=True)
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
            msg = f"WebSocket closed: {code} - {reason}"
            self._log(msg)
            print(f"⚠️ {msg}", flush=True)
        
        def on_error(ws, code, reason):
            msg = f"WebSocket error: {code} - {reason}"
            self._log(msg)
            print(f"❌ {msg}", flush=True)
        
        self.ticker.on_connect = on_connect
        self.ticker.on_ticks = on_ticks
        self.ticker.on_close = on_close
        self.ticker.on_error = on_error
        
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
        """Stop the bot - does NOT close positions (hold till expiry)."""
        self.is_running = False
        self.stop_event.set()
        
        # Log open positions but DON'T close them (hold till expiry)
        for symbol, trader in self.traders.items():
            if trader.position:
                self._log(f"[{symbol}] Open position held: {trader.position.option_type} {trader.position.strike}")
                print(f"📌 [{symbol}] Holding position: {trader.position.option_type} {trader.position.strike}", flush=True)


def main():
    bot = SupertrendBot()
    bot.run()


if __name__ == "__main__":
    main()
