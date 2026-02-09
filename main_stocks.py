#!/usr/bin/env python3
"""
Triple-Confirmation Trading Bot for F&O STOCKS
Uses MACD + SuperTrend + VWAP + PCR for entry confirmation.

Stocks: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK

Usage:
    python3 main_stocks.py
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

# Import Angel One PCR tracker
try:
    from angel_one import AngelOnePCR, get_pcr_tracker
    ANGEL_ONE_AVAILABLE = True
except ImportError:
    ANGEL_ONE_AVAILABLE = False
    AngelOnePCR = None
    get_pcr_tracker = None


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

CORPUS = 100000
TIMEFRAME_MINUTES = 15

# Indicator Parameters
ATR_PERIOD = 20
ATR_MULTIPLIER = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk Management
MAX_CONCURRENT_POSITIONS = 3
PER_STOCK_CAPITAL_LIMIT = 25000
DAILY_LOSS_LIMIT = 15000

# 5 F&O Stocks Configuration
STOCKS = {
    "RELIANCE": {
        "name": "Reliance Industries",
        "lot_size": 250,
        "strike_gap": 20,
        "instrument_token": None,  # Will be fetched dynamically
    },
    "TCS": {
        "name": "Tata Consultancy",
        "lot_size": 175,
        "strike_gap": 50,
        "instrument_token": None,
    },
    "INFY": {
        "name": "Infosys",
        "lot_size": 400,
        "strike_gap": 20,
        "instrument_token": None,
    },
    "HDFCBANK": {
        "name": "HDFC Bank",
        "lot_size": 550,
        "strike_gap": 25,
        "instrument_token": None,
    },
    "ICICIBANK": {
        "name": "ICICI Bank",
        "lot_size": 700,
        "strike_gap": 12.5,
        "instrument_token": None,
    },
}


def now_ist():
    if IST:
        return datetime.now(IST)
    return datetime.now()


# ============================================================
# INDICATORS (Same as main.py)
# ============================================================
def calculate_supertrend(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    df['hl2'] = (df['high'] + df['low']) / 2
    df['basic_up'] = df['hl2'] - (multiplier * df['atr'])
    df['basic_dn'] = df['hl2'] + (multiplier * df['atr'])
    df['up'] = df['basic_up']
    df['dn'] = df['basic_dn']
    df['trend'] = 1
    df['supertrend'] = 0.0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > df['up'].iloc[i-1]:
            df.loc[df.index[i], 'up'] = max(df['basic_up'].iloc[i], df['up'].iloc[i-1])
        else:
            df.loc[df.index[i], 'up'] = df['basic_up'].iloc[i]
        
        if df['close'].iloc[i-1] < df['dn'].iloc[i-1]:
            df.loc[df.index[i], 'dn'] = min(df['basic_dn'].iloc[i], df['dn'].iloc[i-1])
        else:
            df.loc[df.index[i], 'dn'] = df['basic_dn'].iloc[i]
        
        prev_trend = df['trend'].iloc[i-1]
        if prev_trend == -1 and df['close'].iloc[i] > df['dn'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = 1
        elif prev_trend == 1 and df['close'].iloc[i] < df['up'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = prev_trend
        
        if df['trend'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['up'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['dn'].iloc[i]
    
    df['prev_trend'] = df['trend'].shift(1)
    df['signal'] = None
    df.loc[(df['trend'] == 1) & (df['prev_trend'] == -1), 'signal'] = 'BUY'
    df.loc[(df['trend'] == -1) & (df['prev_trend'] == 1), 'signal'] = 'SELL'
    
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
    df['macd_prev_line'] = df['macd_line'].shift(1)
    df['macd_prev_signal'] = df['macd_signal'].shift(1)
    df['macd_bullish_cross'] = (df['macd_line'] > df['macd_signal']) & (df['macd_prev_line'] <= df['macd_prev_signal'])
    df['macd_bearish_cross'] = (df['macd_line'] < df['macd_signal']) & (df['macd_prev_line'] >= df['macd_prev_signal'])
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_real'] = False
        return df
    
    # Reset VWAP daily
    df['date'] = df['timestamp'].dt.date
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['tp'] * df['volume']
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    df['vwap_real'] = True
    return df


# ============================================================
# POSITION & METRICS TRACKING
# ============================================================
class Position:
    def __init__(self, symbol: str, option_type: str, strike: float,
                 entry_price: float, sl: float, quantity: int, 
                 entry_time: datetime, spot_at_entry: float):
        self.symbol = symbol
        self.option_type = option_type
        self.strike = strike
        self.entry_price = entry_price
        self.initial_sl = sl
        self.sl = sl
        self.quantity = quantity
        self.entry_time = entry_time
        self.spot_at_entry = spot_at_entry
        self.peak_price = entry_price
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
    
    def close(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_time = now_ist()
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.quantity


class MetricsTracker:
    """Track P&L, drawdown, and other metrics."""
    
    def __init__(self):
        self.trades: List[Position] = []
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.winning_streak_equity = 0.0
        self.max_drawdown_from_winners = 0.0  # Drawdown for profitable trades
    
    def add_trade(self, position: Position):
        self.trades.append(position)
        self.daily_pnl += position.pnl
        
        # Track peak equity
        if self.daily_pnl > self.peak_equity:
            self.peak_equity = self.daily_pnl
        
        # Calculate drawdown
        current_dd = self.peak_equity - self.daily_pnl
        self.max_drawdown = max(self.max_drawdown, current_dd)
        
        # Track drawdown from profitable trades
        if position.pnl > 0:
            self.winning_streak_equity += position.pnl
        else:
            dd_from_winners = self.winning_streak_equity + position.pnl
            if dd_from_winners < 0:
                self.max_drawdown_from_winners = max(
                    self.max_drawdown_from_winners, 
                    abs(dd_from_winners)
                )
            self.winning_streak_equity = max(0, self.winning_streak_equity + position.pnl)
    
    def get_stats(self) -> Dict:
        if not self.trades:
            return {}
        
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        
        return {
            'total_trades': len(self.trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'max_dd_from_winners': self.max_drawdown_from_winners,
            'avg_win': sum(t.pnl for t in winners) / len(winners) if winners else 0,
            'avg_loss': sum(t.pnl for t in losers) / len(losers) if losers else 0,
        }


# ============================================================
# PCR TRACKER (Simplified - uses stock PCR if available)
# ============================================================
class StockPCRTracker:
    """Simple PCR tracker for stocks (uses default if not available)."""
    
    def __init__(self):
        self.pcr_values = {}
    
    def get_pcr(self, symbol: str) -> float:
        """Get PCR for stock. Returns 1.0 (neutral) if not available."""
        return self.pcr_values.get(symbol, 1.0)
    
    def update_pcr(self, symbol: str, pcr: float):
        self.pcr_values[symbol] = pcr


# ============================================================
# STOCK TRADER
# ============================================================
class StockTrader:
    """
    Quad-Confirmation Trading for one stock.
    
    Entry Logic (with MACD lookback):
    - MACD crossover triggers a "pending signal" that stays valid for 2 candles
    - If SuperTrend confirms within those 2 candles → ENTRY
    - VWAP and PCR must still be valid at the time of entry
    """
    
    # MACD signal lookback window (in candles)
    MACD_LOOKBACK_CANDLES = 2
    
    def __init__(self, symbol: str, config: Dict, logger, telegram=None, pcr_tracker=None, metrics=None):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.telegram = telegram
        self.pcr_tracker = pcr_tracker
        self.metrics = metrics
        
        self.candles: List[Dict] = []
        self.current_candle: Optional[Dict] = None
        self.last_candle_time: Optional[datetime] = None
        
        # Indicators
        self.current_trend = 0
        self.prev_trend = 0  # Track previous trend for SuperTrend flip detection
        self.supertrend_value = 0
        self.current_atr = 0
        self.macd_line = 0
        self.macd_signal = 0
        self.macd_bullish = False
        self.macd_bearish = False
        self.vwap = 0
        self.vwap_is_real = False
        
        # MACD Signal Lookback Tracking
        # When MACD crosses, we remember it for N candles
        self.pending_macd_bullish = 0  # Countdown of candles remaining
        self.pending_macd_bearish = 0  # Countdown of candles remaining
        
        self.position: Optional[Position] = None
        self.trades: List[Position] = []
        self.signals: List[Dict] = []
        
        self.tick_count = 0
        self.candle_count = 0
        self.lock = Lock()

    
    def process_tick(self, ltp: float, tick_time: datetime, volume: int = 0):
        with self.lock:
            self.tick_count += 1
            
            if self.tick_count % 100 == 0:
                print(f"[TICK] [{self.symbol}] #{self.tick_count} LTP: {ltp:.2f}", flush=True)
            
            candle_minute = (tick_time.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
            candle_ts = tick_time.replace(minute=candle_minute, second=0, microsecond=0)
            
            if self.current_candle is None or candle_ts != self.last_candle_time:
                if self.current_candle:
                    self.candles.append(self.current_candle)
                    self.candle_count += 1
                    if self.candle_count >= ATR_PERIOD + 5:
                        self._process_candle(self.current_candle)
                
                self.current_candle = {
                    "timestamp": candle_ts,
                    "open": ltp,
                    "high": ltp,
                    "low": ltp,
                    "close": ltp,
                    "volume": volume
                }
                self.last_candle_time = candle_ts
            else:
                self.current_candle["high"] = max(self.current_candle["high"], ltp)
                self.current_candle["low"] = min(self.current_candle["low"], ltp)
                self.current_candle["close"] = ltp
                self.current_candle["volume"] += volume
            
            if self.position:
                self._check_exit(ltp)
    
    def _process_candle(self, candle: Dict):
        if len(self.candles) < ATR_PERIOD + 5:
            return
        
        df = pd.DataFrame(self.candles[-50:])
        df = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
        df = calculate_macd(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df = calculate_vwap(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        prev_trend = self.current_trend
        self.current_trend = int(latest['trend'])
        self.supertrend_value = latest['supertrend']
        self.current_atr = latest['atr']
        self.macd_line = latest['macd_line']
        self.macd_signal_val = latest['macd_signal']
        self.macd_bullish = bool(latest['macd_bullish_cross'])
        self.macd_bearish = bool(latest['macd_bearish_cross'])
        self.vwap = latest['vwap']
        self.vwap_is_real = latest.get('vwap_real', False)
        
        trend_str = "🟢 BULL" if self.current_trend == 1 else "🔴 BEAR"
        macd_str = "↑CROSS" if self.macd_bullish else ("↓CROSS" if self.macd_bearish else "—")
        vwap_tag = "(REAL)" if self.vwap_is_real else "(EST)"
        
        self.logger(f"[{self.symbol}] {candle['timestamp'].strftime('%H:%M')} | C:{candle['close']:.0f} | ST:{self.supertrend_value:.0f} | MACD:{macd_str} | VWAP{vwap_tag} | {trend_str}")
        
        # Get PCR - Check if real PCR is available
        pcr = None  # None means not available
        pcr_available = False
        if self.pcr_tracker:
            try:
                pcr_value = self.pcr_tracker.get_pcr(self.symbol)
                # If PCR is exactly 1.0, it's likely the default (not real data)
                if pcr_value != 1.0:
                    pcr = pcr_value
                    pcr_available = True
            except Exception as e:
                print(f"   ⚠️ [{self.symbol}] PCR fetch error: {e}", flush=True)
                pcr = None
        
        
        # === EXIT LOGIC ===
        if self.position:
            exit_reason = None
            
            if self.position.option_type == "CE" and self.macd_bearish:
                exit_reason = "MACD_REVERSAL"
            elif self.position.option_type == "PE" and self.macd_bullish:
                exit_reason = "MACD_REVERSAL"
            
            if self.position.option_type == "CE" and self.current_trend == -1 and prev_trend == 1:
                exit_reason = "SUPERTREND_REVERSAL"
            elif self.position.option_type == "PE" and self.current_trend == 1 and prev_trend == -1:
                exit_reason = "SUPERTREND_REVERSAL"
            
            if exit_reason:
                self._close_position(candle["close"], exit_reason)
        
        # === ENTRY LOGIC (with MACD Lookback) ===
        # MACD crossover triggers a "pending signal" that stays valid for N candles
        # SuperTrend confirmation can come on crossover candle OR next N candles
        if self.position is None:
            
            # === Step 1: Track new MACD crossovers ===
            if self.macd_bullish:
                self.pending_macd_bullish = self.MACD_LOOKBACK_CANDLES + 1  # +1 for this candle
                self.pending_macd_bearish = 0  # Cancel opposite signal
                print(f"   ⚡ [{self.symbol}] MACD Bullish Cross detected (valid for {self.MACD_LOOKBACK_CANDLES} more candles)", flush=True)
            
            if self.macd_bearish:
                self.pending_macd_bearish = self.MACD_LOOKBACK_CANDLES + 1
                self.pending_macd_bullish = 0  # Cancel opposite signal
                print(f"   ⚡ [{self.symbol}] MACD Bearish Cross detected (valid for {self.MACD_LOOKBACK_CANDLES} more candles)", flush=True)
            
            # === Step 2: Check for SuperTrend confirmation ===
            # SuperTrend flip: trend changed from previous candle
            st_bullish_flip = (self.current_trend == 1 and prev_trend == -1)
            st_bearish_flip = (self.current_trend == -1 and prev_trend == 1)
            
            # OR: SuperTrend already in correct trend
            st_bullish = (self.current_trend == 1)
            st_bearish = (self.current_trend == -1)
            
            buy_confirmed = False
            sell_confirmed = False
            
            # BUY: Pending MACD bullish + ST bullish + Below VWAP + (PCR < 1.0 if available)
            if self.pending_macd_bullish > 0 and st_bullish:
                if candle['close'] < self.vwap:
                    # PCR check: Skip if not available (use Triple-Confirmation), require <1.0 if available
                    if pcr is None:
                        # PCR not available - use Triple-Confirmation
                        buy_confirmed = True
                        confirmation_type = "FLIP" if st_bullish_flip else "ALIGN"
                        print(f"\n✅ [{self.symbol}] TRIPLE-CONFIRMATION BUY! (PCR unavailable)", flush=True)
                        print(f"   MACD: ↑ (pending) | ST: Bullish ({confirmation_type}) | VWAP: Below | PCR: N/A", flush=True)
                    elif pcr < 1.0:
                        # PCR available and bullish - Quad-Confirmation
                        buy_confirmed = True
                        confirmation_type = "FLIP" if st_bullish_flip else "ALIGN"
                        print(f"\n✅ [{self.symbol}] QUAD-CONFIRMATION BUY!", flush=True)
                        print(f"   MACD: ↑ (pending) | ST: Bullish ({confirmation_type}) | VWAP: Below | PCR: {pcr:.2f}", flush=True)
                    else:
                        print(f"   ⏳ [{self.symbol}] BUY blocked: PCR {pcr:.2f} >= 1.0 (bearish)", flush=True)
                else:
                    print(f"   ⏳ [{self.symbol}] BUY blocked: Price above VWAP", flush=True)
            
            # SELL: Pending MACD bearish + ST bearish + Above VWAP + (PCR > 1.0 if available)
            if self.pending_macd_bearish > 0 and st_bearish:
                if candle['close'] > self.vwap:
                    # PCR check: Skip if not available, require >1.0 if available
                    if pcr is None:
                        # PCR not available - use Triple-Confirmation
                        sell_confirmed = True
                        confirmation_type = "FLIP" if st_bearish_flip else "ALIGN"
                        print(f"\n✅ [{self.symbol}] TRIPLE-CONFIRMATION SELL! (PCR unavailable)", flush=True)
                        print(f"   MACD: ↓ (pending) | ST: Bearish ({confirmation_type}) | VWAP: Above | PCR: N/A", flush=True)
                    elif pcr > 1.0:
                        # PCR available and bearish - Quad-Confirmation
                        sell_confirmed = True
                        confirmation_type = "FLIP" if st_bearish_flip else "ALIGN"
                        print(f"\n✅ [{self.symbol}] QUAD-CONFIRMATION SELL!", flush=True)
                        print(f"   MACD: ↓ (pending) | ST: Bearish ({confirmation_type}) | VWAP: Above | PCR: {pcr:.2f}", flush=True)
                    else:
                        print(f"   ⏳ [{self.symbol}] SELL blocked: PCR {pcr:.2f} <= 1.0 (bullish)", flush=True)
                else:
                    print(f"   ⏳ [{self.symbol}] SELL blocked: Price below VWAP", flush=True)
            
            # === Step 3: Enter position if confirmed ===
            if buy_confirmed:
                self._enter_position("BUY", candle, prev)
                self.pending_macd_bullish = 0  # Reset after entry
            elif sell_confirmed:
                self._enter_position("SELL", candle, prev)
                self.pending_macd_bearish = 0  # Reset after entry
            
            # === Step 4: Decrement pending counters ===
            if self.pending_macd_bullish > 0:
                self.pending_macd_bullish -= 1
                if self.pending_macd_bullish == 0:
                    print(f"   ⌛ [{self.symbol}] MACD Bullish signal expired (no ST confirmation)", flush=True)
            
            if self.pending_macd_bearish > 0:
                self.pending_macd_bearish -= 1
                if self.pending_macd_bearish == 0:
                    print(f"   ⌛ [{self.symbol}] MACD Bearish signal expired (no ST confirmation)", flush=True)
    
    def _enter_position(self, signal: str, candle: Dict, prev_candle: Dict = None):
        spot = candle["close"]
        option_type = "CE" if signal == "BUY" else "PE"
        
        strike_gap = self.config["strike_gap"]
        strike = round(spot / strike_gap) * strike_gap
        
        # Estimate premium
        itm = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
        entry_price = itm + spot * 0.004 + 15
        
        # Dynamic SL from previous candle
        if prev_candle:
            sl_spot = prev_candle["low"] if signal == "BUY" else prev_candle["high"]
            delta = 0.5 if option_type == "CE" else -0.5
            sl = max(0.05, entry_price + ((sl_spot - spot) * delta))
        else:
            sl = entry_price * 0.85
        
        self.position = Position(
            symbol=self.symbol,
            option_type=option_type,
            strike=strike,
            entry_price=entry_price,
            sl=sl,
            quantity=self.config["lot_size"],
            entry_time=now_ist(),
            spot_at_entry=spot
        )
        
        emoji = "🟢" if signal == "BUY" else "🔴"
        print(f"\n{'='*50}", flush=True)
        print(f"{emoji} [{self.symbol}] ENTRY - {option_type} {strike}", flush=True)
        print(f"   Entry: ₹{entry_price:.2f} | SL: ₹{sl:.2f}", flush=True)
        print(f"   Qty: {self.config['lot_size']}", flush=True)
        print(f"{'='*50}\n", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_entry(
                self.symbol, option_type, strike, entry_price, 0, sl,
                self.config["lot_size"], signal
            )
    
    def _close_position(self, current_spot: float, reason: str):
        if not self.position:
            return
        
        move = current_spot - self.position.spot_at_entry
        delta = 0.5 if self.position.option_type == "CE" else -0.5
        exit_price = max(0.05, self.position.entry_price + (move * delta))
        
        self.position.close(exit_price, reason)
        self.trades.append(self.position)
        
        if self.metrics:
            self.metrics.add_trade(self.position)
        
        emoji = "✅" if self.position.pnl > 0 else "🛑"
        print(f"\n{emoji} [{self.symbol}] CLOSED - {reason}", flush=True)
        print(f"   Entry: ₹{self.position.entry_price:.2f} → Exit: ₹{exit_price:.2f}", flush=True)
        print(f"   P&L: ₹{self.position.pnl:,.2f}", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_exit(
                self.symbol, self.position.option_type, self.position.strike,
                self.position.entry_price, exit_price, self.position.pnl, reason
            )
        
        self.position = None
    
    def _check_exit(self, current_ltp: float):
        if not self.position:
            return
        
        move = current_ltp - self.position.spot_at_entry
        delta = 0.5 if self.position.option_type == "CE" else -0.5
        estimated_price = self.position.entry_price + (move * delta)
        
        if estimated_price <= self.position.sl:
            self._close_position(current_ltp, "SL_HIT")


# ============================================================
# MAIN BOT
# ============================================================
class StockOptionsBot:
    def __init__(self):
        self.is_running = False
        self.stop_event = Event()
        
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        
        self.telegram = TelegramNotifier() if TELEGRAM_AVAILABLE else None
        
        # Initialize PCR tracker - try Angel One, fallback to dummy
        self.pcr_tracker = None
        self.pcr_available = False
        if ANGEL_ONE_AVAILABLE:
            try:
                self.pcr_tracker = get_pcr_tracker(self._log)
                print("📊 Angel One PCR tracker initialized", flush=True)
            except Exception as e:
                print(f"⚠️ Angel One PCR init failed: {e}", flush=True)
                self.pcr_tracker = StockPCRTracker()
        else:
            print("⚠️ Angel One not available, using fallback PCR", flush=True)
            self.pcr_tracker = StockPCRTracker()
        
        self.metrics = MetricsTracker()
        
        self.traders: Dict[str, StockTrader] = {}
        self.token_to_symbol: Dict[int, str] = {}
        
        self.log_file = LOG_DIR / f"stocks_{now_ist().strftime('%Y%m%d')}.log"
    
    def _log(self, message: str):
        timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {message}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    
    def _load_credentials(self):
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
            print("🔐 KITE LOGIN")
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
            
            self._log(f"✅ Logged in: {data.get('user_name', 'N/A')}")
            return True
            
        except Exception as e:
            self._log(f"❌ Auth failed: {e}")
            return False
    
    def fetch_stock_tokens(self):
        """Fetch instrument tokens for all stocks."""
        print("\n📋 Fetching stock tokens...")
        
        try:
            instruments = self.kite.instruments("NSE")
            
            for symbol in STOCKS.keys():
                for inst in instruments:
                    if inst['tradingsymbol'] == symbol:
                        STOCKS[symbol]['instrument_token'] = inst['instrument_token']
                        self.token_to_symbol[inst['instrument_token']] = symbol
                        print(f"   ✓ {symbol}: {inst['instrument_token']}")
                        break
            
            # Initialize traders
            for symbol, config in STOCKS.items():
                if config['instrument_token']:
                    self.traders[symbol] = StockTrader(
                        symbol, config, self._log, self.telegram, 
                        self.pcr_tracker, self.metrics
                    )
            
            print(f"   ✅ {len(self.traders)} stocks ready")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def fetch_historical(self):
        """Fetch historical data for all stocks."""
        if not self.kite:
            return
        
        for symbol, trader in self.traders.items():
            try:
                config = STOCKS[symbol]
                to_date = now_ist()
                from_date = to_date - timedelta(days=5)
                
                print(f"📊 Fetching {symbol}...", flush=True)
                
                data = self.kite.historical_data(
                    instrument_token=config["instrument_token"],
                    from_date=from_date,
                    to_date=to_date,
                    interval="15minute"
                )
                
                for candle in data[-50:]:
                    trader.candles.append({
                        "timestamp": candle["date"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle.get("volume", 0)
                    })
                
                print(f"   ✓ {symbol}: {len(trader.candles)} candles", flush=True)
                
            except Exception as e:
                print(f"   ✗ {symbol}: {e}", flush=True)
    
    def start_live_feed(self):
        """Start WebSocket feed for all stocks."""
        api_key, _ = self._load_credentials()
        self.ticker = KiteTicker(api_key, self.kite.access_token)
        
        tokens = list(self.token_to_symbol.keys())
        
        def on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            self._log(f"WebSocket connected. Subscribed: {list(self.token_to_symbol.values())}")
        
        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick.get("instrument_token")
                ltp = tick.get("last_price")
                volume = tick.get("volume_traded", 0)
                tick_time = now_ist()
                
                if token in self.token_to_symbol and ltp:
                    symbol = self.token_to_symbol[token]
                    if symbol in self.traders:
                        self.traders[symbol].process_tick(ltp, tick_time, volume)
        
        def on_close(ws, code, reason):
            self._log(f"WebSocket closed: {code} - {reason}")
        
        def on_error(ws, code, reason):
            self._log(f"WebSocket error: {code} - {reason}")
        
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
        
        print(f"\n{'='*60}")
        print(f"📊 DAILY REPORT - {today}")
        print(f"{'='*60}")
        
        stats = self.metrics.get_stats()
        
        for symbol, trader in self.traders.items():
            pnl = sum(t.pnl for t in trader.trades)
            wins = len([t for t in trader.trades if t.pnl > 0])
            losses = len([t for t in trader.trades if t.pnl <= 0])
            print(f"{symbol}: {len(trader.trades)} trades | W:{wins} L:{losses} | P&L: ₹{pnl:,.2f}")
        
        print(f"\n📈 COMBINED STATS:")
        print(f"   Total Trades: {stats.get('total_trades', 0)}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"   Total P&L: ₹{stats.get('total_pnl', 0):,.2f}")
        print(f"   Max Drawdown: ₹{stats.get('max_drawdown', 0):,.2f}")
        print(f"   Max DD from Winners: ₹{stats.get('max_dd_from_winners', 0):,.2f}")
        print(f"{'='*60}\n")
        
        # Save report
        report = {
            "date": today,
            "stats": stats,
            "stocks": {s: {"trades": len(t.trades), "pnl": sum(p.pnl for p in t.trades)} 
                       for s, t in self.traders.items()}
        }
        
        report_path = LOG_DIR / f"stocks_report_{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    
    def run(self):
        self.is_running = True
        
        def signal_handler(sig, frame):
            print("\n⏹️ Stopping...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\n" + "=" * 60)
        print("🚀 STOCK OPTIONS TRADING BOT")
        print("=" * 60)
        print(f"Stocks: {', '.join(STOCKS.keys())}")
        print(f"Strategy: MACD + SuperTrend + VWAP + PCR")
        print(f"Timeframe: {TIMEFRAME_MINUTES}min")
        print(f"Max Positions: {MAX_CONCURRENT_POSITIONS}")
        print("=" * 60 + "\n")
        
        if self.kite is None:
            if not self.authenticate():
                return
        
        if not self.fetch_stock_tokens():
            return
        
        if not self.is_market_open():
            self.wait_for_market()
        
        if not self.is_running:
            return
        
        self._log("Starting stock trading session...")
        
        if self.telegram:
            self.telegram.notify_bot_start(list(STOCKS.keys()))
        
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


def main():
    bot = StockOptionsBot()
    bot.run()


if __name__ == "__main__":
    main()
