#!/usr/bin/env python3
"""
Production Autonomous Stock Options Trading Bot (Quad-Confirmation + Alligator Golden Zone)
Stocks: RELIANCE, ICICIBANK, SBIN, AXISBANK, LT

Upgrades:
1. Native Kite API Real-Time PCR calculation (Queries Open Interest directly from Zerodha Kite Connect)
2. Real ATM Option Contract Resolution & Live Market Quotes (No synthetic delta/premium formulas)
3. Dynamic Lot Size & Instrument Token auto-loading from Kite API on startup
4. Strictly configured for ₹1,00,000 (₹1 Lakh) capital
5. Full Diagnostic Telemetry: Real-time Near-Miss alerts (Score ≥ 1.5), 12:00 PM Mid-Day Heartbeat, and EOD Filter Matrix
"""

import os
import sys
import time
import signal
import json
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List
from threading import Event, Lock
import pandas as pd
import numpy as np

# Add directory to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

try:
    from telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramNotifier = None

from auto_login import KiteAutoLogin, load_credentials
from execution_engine import ExecutionEngine

# ============================================================
# CONFIGURATION & CAPITAL ALLOCATION (₹1.0 LAKH)
# ============================================================
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TOTAL_CAPITAL = 100000.0           # ₹1,00,000 Total Capital for Stock Bot
MAX_CONCURRENT_POSITIONS = 3       # Max 3 open positions simultaneously
PER_STOCK_CAPITAL_LIMIT = 25000.0  # Max ₹25K allocation per trade
DAILY_LOSS_LIMIT = 10000.0         # Circuit breaker: stop trading if down ₹10K in a day

PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() == "true"

# Indicator Parameters
ATR_PERIOD = 20
ATR_MULTIPLIER = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Default fallback specs (Overwritten dynamically from Kite API on startup)
STOCKS = {
    "RELIANCE": {"name": "Reliance Industries", "timeframe": 15, "macd_lookback": 3, "lot_size": 500, "strike_gap": 20.0, "token": None},
    "ICICIBANK": {"name": "ICICI Bank", "timeframe": 30, "macd_lookback": 3, "lot_size": 700, "strike_gap": 20.0, "token": None},
    "SBIN": {"name": "State Bank of India", "timeframe": 30, "macd_lookback": 3, "lot_size": 750, "strike_gap": 20.0, "token": None},
    "AXISBANK": {"name": "Axis Bank", "timeframe": 30, "macd_lookback": 5, "lot_size": 625, "strike_gap": 40.0, "token": None},
    "LT": {"name": "Larsen & Toubro", "timeframe": 15, "macd_lookback": 5, "lot_size": 175, "strike_gap": 100.0, "token": None},
}


def now_ist():
    return datetime.now(IST) if IST else datetime.now()


# ============================================================
# NATIVE KITE STOCK PCR TRACKER
# ============================================================
class NativeKitePCRTracker:
    """
    Calculates authentic Put-Call Ratio (PCR) for individual stocks
    using real Open Interest (OI) fetched directly from Kite Connect API.
    """
    def __init__(self, kite=None, logger=None):
        self.kite = kite
        self.logger = logger or print
        self.pcr_cache: Dict[str, float] = {}
        self.last_updated: Optional[datetime] = None

    def update_stock_pcr(self, symbol: str, nfo_df: pd.DataFrame) -> float:
        """Fetch live OI for all strikes of the stock and compute Put OI / Call OI."""
        if self.kite is None or nfo_df is None or nfo_df.empty:
            return 1.0
        try:
            today = date.today()
            stock_opts = nfo_df[(nfo_df['name'] == symbol) & (nfo_df['expiry'] >= today)]
            if stock_opts.empty:
                return 1.0
                
            near_expiry = stock_opts['expiry'].min()
            near_opts = stock_opts[stock_opts['expiry'] == near_expiry]
            
            # Query quotes for instruments (batch of up to 50 instruments)
            symbols = [f"NFO:{ts}" for ts in near_opts['tradingsymbol'].tolist()[:40]]
            quotes = self.kite.quote(symbols)
            
            call_oi = 0
            put_oi = 0
            for ts, q in quotes.items():
                oi = q.get('oi', 0)
                if ts.endswith('CE'): call_oi += oi
                elif ts.endswith('PE'): put_oi += oi
                
            pcr = (put_oi / call_oi) if call_oi > 0 else 1.0
            self.pcr_cache[symbol] = round(pcr, 2)
            return self.pcr_cache[symbol]
        except Exception as e:
            self.logger(f"⚠️ PCR calculation warning for {symbol}: {e}")
            return self.pcr_cache.get(symbol, 1.0)

    def get_pcr(self, symbol: str) -> float:
        return self.pcr_cache.get(symbol, 1.0)


# ============================================================
# POSITION MODEL
# ============================================================
class Position:
    def __init__(self, symbol: str, option_type: str, strike: float,
                 tradingsymbol: str, option_token: int, entry_price: float,
                 sl: float, quantity: int, entry_time: datetime, spot_at_entry: float):
        self.symbol = symbol
        self.option_type = option_type          # "CE" or "PE"
        self.strike = strike
        self.tradingsymbol = tradingsymbol      # e.g., "RELIANCE26SEP1360CE"
        self.option_token = option_token
        self.entry_price = entry_price          # Real Option LTP
        self.sl = sl                            # Real Option SL
        self.quantity = quantity
        self.entry_time = entry_time
        self.spot_at_entry = spot_at_entry
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.net_pnl = 0.0

    def close(self, exit_price: float, reason: str):
        self.exit_price = exit_price
        self.exit_time = now_ist()
        self.exit_reason = reason
        pts = self.exit_price - self.entry_price
        self.pnl = pts * self.quantity
        
        # Real exchange brokerage & taxes for Stock Options
        tot_val = (self.entry_price + exit_price) * self.quantity
        stt = (exit_price * self.quantity) * 0.0010
        brokerage = 40.0
        exchange = tot_val * 0.000505
        stamp = (self.entry_price * self.quantity) * 0.00003
        sebi = tot_val * 0.000001
        gst = (brokerage + exchange + sebi) * 0.18
        tax = brokerage + stt + exchange + stamp + sebi + gst
        self.net_pnl = self.pnl - tax


# ============================================================
# PERSISTENT CAPITAL & RISK TRACKER
# ============================================================
class CapitalTracker:
    """
    Manages session trading capital, passive income counter (overall P&L),
    and enforces dynamic concurrent position guardrails.
    """
    def __init__(self, base_capital: float = TOTAL_CAPITAL,
                 per_slot_capital: float = PER_STOCK_CAPITAL_LIMIT,
                 state_file: Path = BASE_DIR / "capital_state.json",
                 logger=print):
        self.base_capital = base_capital
        self.per_slot_capital = per_slot_capital
        self.state_file = state_file
        self.logger = logger
        self.lock = Lock()  # Thread safety for concurrent trader access
        
        self.session_capital = self.base_capital
        self.overall_pnl = 0.0
        self.daily_realized_pnl = 0.0  # Track intraday P&L for daily loss limit
        self.load_state()

    def load_state(self):
        # 1. Local state file persistence (authoritative for active container)
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.session_capital = float(data.get("session_capital", self.base_capital))
                self.overall_pnl = float(data.get("overall_pnl", 0.0))
                os.environ["SESSION_CAPITAL"] = str(self.session_capital)
                os.environ["OVERALL_PNL"] = str(self.overall_pnl)
                self.logger(f"💼 Capital state loaded: Session Capital=₹{self.session_capital:,.2f}, Overall P&L=₹{self.overall_pnl:,.2f}")
                return
            except Exception as e:
                self.logger(f"⚠️ Error reading capital_state.json: {e}. Checking ENV fallback...")

        # 2. Environment variable fallback (used on fresh container deploy where local disk was wiped)
        env_session = os.environ.get("SESSION_CAPITAL")
        env_overall = os.environ.get("OVERALL_PNL")
        if env_session is not None or env_overall is not None:
            if env_session:
                try: self.session_capital = float(env_session)
                except: pass
            if env_overall:
                try: self.overall_pnl = float(env_overall)
                except: pass
            self.logger(f"💼 Capital initialized from ENV: Session Capital=₹{self.session_capital:,.2f}, Overall P&L=₹{self.overall_pnl:,.2f}")
            self.save_state()
            return
            
        self.session_capital = self.base_capital
        self.overall_pnl = 0.0
        self.save_state()

    def save_state(self):
        # Keep in-process os.environ in sync so re-instantiations within same container stay aligned
        os.environ["SESSION_CAPITAL"] = str(self.session_capital)
        os.environ["OVERALL_PNL"] = str(self.overall_pnl)

        # 1. Save to local file (fast, for same-container restarts)
        try:
            data = {
                "base_capital": self.base_capital,
                "session_capital": self.session_capital,
                "overall_pnl": self.overall_pnl,
                "last_updated": now_ist().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self.logger(f"⚠️ Error saving capital_state.json: {e}")

        # 2. Push to Render env vars (survives redeployments / new containers)
        self._push_to_render_env()

    def _push_to_render_env(self):
        """Persist SESSION_CAPITAL and OVERALL_PNL as Render env vars so they
        survive container teardowns and new deployments.
        Uses GET-merge-PUT pattern to preserve all existing env vars."""
        import math
        api_key = os.environ.get("RENDER_API_KEY")
        service_id = os.environ.get("RENDER_SERVICE_ID")
        if not api_key or not service_id:
            return  # Not on Render or keys not configured — silently skip
        
        # Guard against persisting NaN or Inf
        if not math.isfinite(self.session_capital) or not math.isfinite(self.overall_pnl):
            self.logger("⚠️ Refusing to persist non-finite capital values to Render")
            return
        
        try:
            import urllib.request
            base_url = f"https://api.render.com/v1/services/{service_id}/env-vars"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            # 1. GET all existing env vars
            get_req = urllib.request.Request(base_url, method="GET", headers=headers)
            with urllib.request.urlopen(get_req, timeout=10) as resp:
                existing = json.loads(resp.read().decode())
            
            # 2. Build env var dict from existing, update our keys
            env_dict = {}
            for item in existing:
                env_dict[item["envVar"]["key"]] = item["envVar"]["value"]
            
            env_dict["SESSION_CAPITAL"] = str(self.session_capital)
            env_dict["OVERALL_PNL"] = str(self.overall_pnl)
            
            # 3. PUT full list back (preserves KITE_*, TELEGRAM_*, etc.)
            payload = json.dumps([{"key": k, "value": v} for k, v in env_dict.items()]).encode()
            put_req = urllib.request.Request(base_url, data=payload, method="PUT", headers=headers)
            urllib.request.urlopen(put_req, timeout=10)
            
            self.logger(f"💾 Capital persisted to Render — Session: ₹{self.session_capital:,.2f} | P&L: ₹{self.overall_pnl:+,.2f}")
        except Exception as e:
            self.logger(f"⚠️ Could not push capital to Render env vars: {e}")

    def get_max_concurrent_positions(self) -> int:
        if self.session_capital < self.per_slot_capital:
            return 0
        return min(MAX_CONCURRENT_POSITIONS, int(self.session_capital // self.per_slot_capital))

    def can_open_position(self, current_open_count: int) -> tuple:
        with self.lock:
            if self.session_capital <= 0:
                return False, "Session capital is zero or depleted. Trading halted."
            # Daily loss limit circuit breaker
            if self.daily_realized_pnl <= -DAILY_LOSS_LIMIT:
                return False, f"Daily loss limit hit: ₹{self.daily_realized_pnl:,.2f} exceeds -₹{DAILY_LOSS_LIMIT:,.2f}. Trading halted for today."
            if self.session_capital < self.per_slot_capital:
                return False, f"Session capital (₹{self.session_capital:,.2f}) is below minimum required per-slot capital (₹{self.per_slot_capital:,.2f})."
            max_allowed = self.get_max_concurrent_positions()
            if current_open_count >= max_allowed:
                return False, f"Maximum concurrent positions reached ({current_open_count}/{max_allowed}) for capital ₹{self.session_capital:,.2f}."
            return True, "OK"
    
    def record_trade_pnl(self, pnl: float):
        """Record a closed trade's P&L for daily loss limit tracking."""
        with self.lock:
            self.daily_realized_pnl += pnl

    def end_day(self, day_net_pnl: float) -> dict:
        """
        Processes End-of-Day P&L with two-tier principal recovery:
        1. Principal is calculated first:
           new_capital = session_capital + day_net_pnl
        2. If new_capital >= base_capital:
           - Principal is fully intact/restored to base_capital (₹1.0L).
           - Excess above base is profit reaped: profit_reaped = new_capital - base_capital.
           - Tomorrow starts at base_capital.
        3. If new_capital < base_capital:
           - Principal is still in deficit (loss carried forward or partial recovery).
           - No profit is reaped (profit_reaped = 0.0).
           - Tomorrow starts at new_capital (recovering counter).
        4. Overall P&L accumulates continuously:
           overall_pnl += day_net_pnl
        """
        capital_used = self.session_capital
        self.overall_pnl += day_net_pnl
        new_capital = self.session_capital + day_net_pnl
        
        if new_capital >= self.base_capital:
            next_day_capital = self.base_capital
            profit_reaped = new_capital - self.base_capital
            is_base_restored = True
        else:
            next_day_capital = max(0.0, new_capital)
            profit_reaped = 0.0
            is_base_restored = False
            
        capital_remaining = next_day_capital
        deficit = max(0.0, self.base_capital - next_day_capital)
        
        summary = {
            "capital_used": capital_used,
            "day_pnl": day_net_pnl,
            "is_profit": is_base_restored and profit_reaped > 0,
            "is_base_restored": is_base_restored,
            "profit_reaped": profit_reaped,
            "deficit": deficit,
            "capital_remaining": capital_remaining,
            "next_day_capital": next_day_capital,
            "overall_pnl": self.overall_pnl,
            "base_capital": self.base_capital
        }
        
        self.session_capital = next_day_capital
        self.save_state()
        return summary


# ============================================================
# STOCK TRADER ENGINE
# ============================================================
class StockTrader:
    def __init__(self, symbol: str, config: Dict, logger, kite=None,
                 nfo_df=None, telegram=None, pcr_tracker=None,
                 capital_tracker=None, bot_controller=None, execution_engine=None):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.kite = kite
        self.nfo_df = nfo_df
        self.telegram = telegram
        self.pcr_tracker = pcr_tracker
        self.capital_tracker = capital_tracker
        self.bot_controller = bot_controller
        self.execution_engine = execution_engine
        
        self.timeframe_minutes = config.get("timeframe", 15)
        self.macd_lookback = config.get("macd_lookback", 3)
        self.lot_size = config.get("lot_size", 500)
        self.strike_gap = config.get("strike_gap", 20.0)
        
        self.candles: List[Dict] = []
        self.current_candle: Optional[Dict] = None
        self.last_candle_time: Optional[datetime] = None
        
        self.current_trend = 0
        self.supertrend_value = 0.0
        self.vwap = 0.0
        self.pending_macd_bullish = 0
        self.pending_macd_bearish = 0
        
        # SuperTrend recursive state (proper trailing bands)
        self.prev_final_ub = None
        self.prev_final_lb = None
        self.prev_st_trend = 0  # 1=up, -1=down, 0=uninitialized
        self.prev_close_for_st = None
        
        # VWAP incremental volume tracking
        self.prev_cumulative_volume = 0
        
        # Alligator Golden Zone (75M)
        self.gz_bull_top = None
        self.gz_bull_bot = None
        self.gz_bear_top = None
        self.gz_bear_bot = None
        
        self.position: Optional[Position] = None
        self.trades: List[Position] = []
        
        # Diagnostics
        self.tick_count = 0
        self.candle_count = 0
        self.ltp = 0.0
        self.peak_score = 0.0
        self.peak_score_breakdown = []
        self.peak_score_direction = None
        self.last_near_miss_notified = None
        self.primary_block_reason = "No confirmed MACD crossover formed"
        self.lock = Lock()
        self.last_option_ltp = 0.0  # Updated from WebSocket ticks for SL monitoring

    def process_tick(self, ltp: float, tick_time: datetime, volume: int = 0):
        with self.lock:
            self.tick_count += 1
            self.ltp = ltp
            
            # Compute incremental volume from Kite's cumulative volume_traded
            incremental_vol = max(0, volume - self.prev_cumulative_volume) if volume > 0 else 0
            self.prev_cumulative_volume = volume if volume > 0 else self.prev_cumulative_volume
            
            # Session-anchored candle alignment (09:15 open anchor, handles 15m & 30m)
            total_mins = tick_time.hour * 60 + tick_time.minute
            market_open_mins = 9 * 60 + 15
            if total_mins >= market_open_mins:
                elapsed = total_mins - market_open_mins
                bucket_start_mins = market_open_mins + (elapsed // self.timeframe_minutes) * self.timeframe_minutes
                c_hour = bucket_start_mins // 60
                c_min = bucket_start_mins % 60
                candle_ts = tick_time.replace(hour=c_hour, minute=c_min, second=0, microsecond=0)
            else:
                candle_min = (tick_time.minute // self.timeframe_minutes) * self.timeframe_minutes
                candle_ts = tick_time.replace(minute=candle_min, second=0, microsecond=0)
            
            if self.current_candle is None or candle_ts != self.last_candle_time:
                if self.current_candle:
                    self.candles.append(self.current_candle)
                    self.candle_count += 1
                    # Truncate candle list to prevent memory growth
                    if len(self.candles) > 120:
                        self.candles = self.candles[-100:]
                    if len(self.candles) >= ATR_PERIOD + 5:
                        self._process_candle(self.current_candle)
                
                self.current_candle = {
                    "timestamp": candle_ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": incremental_vol
                }
                self.last_candle_time = candle_ts
            else:
                self.current_candle["high"] = max(self.current_candle["high"], ltp)
                self.current_candle["low"] = min(self.current_candle["low"], ltp)
                self.current_candle["close"] = ltp
                self.current_candle["volume"] += incremental_vol
                
            # Real-time tick SL check
            if self.position:
                self._check_exit(ltp)

    def _process_candle(self, candle: Dict):
        c_time = candle["timestamp"].time()
        c_close = candle["close"]
        c_ts = candle["timestamp"]
        
        # 1. Calculate Technical Indicators on Closed Candle
        df = pd.DataFrame(self.candles[-60:])
        
        # SuperTrend (20, 2) — Proper Recursive Trailing Bands
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1]
        
        hl2_val = (candle['high'] + candle['low']) / 2
        basic_ub = hl2_val + (ATR_MULTIPLIER * atr)
        basic_lb = hl2_val - (ATR_MULTIPLIER * atr)
        
        # Recursive trailing: bands only tighten, never widen against the trend
        prev_close = self.prev_close_for_st
        if self.prev_final_ub is not None and prev_close is not None:
            final_ub = min(basic_ub, self.prev_final_ub) if prev_close <= self.prev_final_ub else basic_ub
            final_lb = max(basic_lb, self.prev_final_lb) if prev_close >= self.prev_final_lb else basic_lb
        else:
            final_ub = basic_ub
            final_lb = basic_lb
        
        # Determine trend: only flip when price crosses through the trailing band
        if self.prev_st_trend == 1:  # Was uptrend (using lower band)
            if c_close < final_lb:
                st_trend = -1  # Flip to downtrend
            else:
                st_trend = 1   # Stay uptrend
        elif self.prev_st_trend == -1:  # Was downtrend (using upper band)
            if c_close > final_ub:
                st_trend = 1   # Flip to uptrend
            else:
                st_trend = -1  # Stay downtrend
        else:
            # First candle — initialize based on price vs bands
            st_trend = 1 if c_close > final_ub else -1
        
        self.supertrend_value = final_lb if st_trend == 1 else final_ub
        st_bullish = (st_trend == 1)
        st_bearish = (st_trend == -1)
        
        # Save state for next candle's recursion
        self.prev_final_ub = final_ub
        self.prev_final_lb = final_lb
        self.prev_close_for_st = c_close
        
        prev_trend = self.current_trend
        self.current_trend = st_trend
        st_bull_flip = (prev_trend != 1 and self.current_trend == 1)
        st_bear_flip = (prev_trend != -1 and self.current_trend == -1)
        
        # MACD (12, 26, 9)
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        
        macd_bull_cross = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
        macd_bear_cross = (macd_line.iloc[-1] < signal_line.iloc[-1]) and (macd_line.iloc[-2] >= signal_line.iloc[-2])
        
        # Clear opposite pending on new cross (1D fix)
        if macd_bull_cross:
            self.pending_macd_bullish = self.macd_lookback
            self.pending_macd_bearish = 0
        if macd_bear_cross:
            self.pending_macd_bearish = self.macd_lookback
            self.pending_macd_bullish = 0
        
        # VWAP — Daily reset: only use today's candles
        today_date = c_ts.date() if hasattr(c_ts, 'date') else c_ts
        today_candles = [c for c in self.candles[-60:] if hasattr(c['timestamp'], 'date') and c['timestamp'].date() == today_date]
        if today_candles:
            df_today = pd.DataFrame(today_candles)
            df_today['vol_p'] = df_today['close'] * df_today['volume']
            total_vol = df_today['volume'].sum()
            self.vwap = (df_today['vol_p'].sum() / total_vol) if total_vol > 0 else c_close
        else:
            self.vwap = c_close
        
        # PCR from live Kite Tracker
        pcr = self.pcr_tracker.get_pcr(self.symbol) if self.pcr_tracker else 1.0
        
        # Golden Zone Checks
        in_bull_gz = (self.gz_bull_bot <= c_close <= self.gz_bull_top) if (self.gz_bull_top and self.gz_bull_bot) else False
        in_bear_gz = (self.gz_bear_bot <= c_close <= self.gz_bear_top) if (self.gz_bear_top and self.gz_bear_bot) else False
        
        # 2. Intraday Auto Square-Off at 15:15 IST
        if c_time >= datetime.strptime("15:15", "%H:%M").time():
            if self.position:
                self._close_position("EOD_SQUAREOFF")
            return
            
        # 3. Position Exit on Indicator Reversal
        if self.position:
            if self.position.option_type == "CE" and (st_bear_flip or macd_bear_cross):
                self._close_position("INDICATOR_REVERSAL_BEAR")
            elif self.position.option_type == "PE" and (st_bull_flip or macd_bull_cross):
                self._close_position("INDICATOR_REVERSAL_BULL")
            return

        # 4. Entry Evaluation (09:30 to 14:30)
        if self.position is None and datetime.strptime("09:30", "%H:%M").time() <= c_time <= datetime.strptime("14:30", "%H:%M").time():
            # --- BUY SCORING ---
            if self.pending_macd_bullish > 0:
                buy_score = 1.0  # MACD pending
                buy_bd = ["MACD:+1.0"]
                if st_bullish:
                    buy_score += 1.5 if st_bull_flip else 1.0
                    buy_bd.append("ST:+1.5(FLIP)" if st_bull_flip else "ST:+1.0(ALIGN)")
                else: buy_bd.append("ST:0")
                
                if c_close > self.vwap:
                    buy_score += 0.5
                    buy_bd.append("VWAP:+0.5")
                else: buy_bd.append("VWAP:0")
                
                if pcr < 1.0:
                    buy_score += 0.5
                    buy_bd.append(f"PCR:+0.5({pcr:.2f})")
                else: buy_bd.append(f"PCR:0({pcr:.2f})")
                
                if in_bull_gz:
                    buy_score += 1.0
                    buy_bd.append("GZ_BOOST:+1.0")
                if in_bear_gz and not in_bull_gz:
                    buy_score = 0.0
                    buy_bd.append("BLOCKED_BEAR_GZ")
                    
                if buy_score > self.peak_score:
                    self.peak_score = buy_score
                    self.peak_score_breakdown = list(buy_bd)
                    self.peak_score_direction = "BUY"
                    
                if buy_score >= 2.0:
                    self._enter_position("BUY", c_close, in_bull_gz)
                    self.pending_macd_bullish = 0
                    self.primary_block_reason = "Order Executed"
                else:
                    if not st_bullish: self.primary_block_reason = "SuperTrend is Bearish"
                    elif c_close <= self.vwap: self.primary_block_reason = "Price below session VWAP"
                    else: self.primary_block_reason = f"Score {buy_score:.1f} < 2.0 threshold"
                    
                    if buy_score >= 1.5 and self.telegram:
                        if self.last_near_miss_notified is None or (c_ts - self.last_near_miss_notified).total_seconds() >= 1800:
                            self.telegram.notify_near_miss(self.symbol, "BUY (CE)", buy_score, buy_bd, c_close)
                            self.last_near_miss_notified = c_ts

            # --- SELL SCORING ---
            elif self.pending_macd_bearish > 0:
                sell_score = 1.0  # MACD pending
                sell_bd = ["MACD:+1.0"]
                if st_bearish:
                    sell_score += 1.5 if st_bear_flip else 1.0
                    sell_bd.append("ST:+1.5(FLIP)" if st_bear_flip else "ST:+1.0(ALIGN)")
                else: sell_bd.append("ST:0")
                
                if c_close < self.vwap:
                    sell_score += 0.5
                    sell_bd.append("VWAP:+0.5")
                else: sell_bd.append("VWAP:0")
                
                if pcr > 1.0:
                    sell_score += 0.5
                    sell_bd.append(f"PCR:+0.5({pcr:.2f})")
                else: sell_bd.append(f"PCR:0({pcr:.2f})")
                
                if in_bear_gz:
                    sell_score += 1.0
                    sell_bd.append("GZ_BOOST:+1.0")
                if in_bull_gz and not in_bear_gz:
                    sell_score = 0.0
                    sell_bd.append("BLOCKED_BULL_GZ")
                    
                if sell_score > self.peak_score:
                    self.peak_score = sell_score
                    self.peak_score_breakdown = list(sell_bd)
                    self.peak_score_direction = "SELL"
                    
                if sell_score >= 2.0:
                    self._enter_position("SELL", c_close, in_bear_gz)
                    self.pending_macd_bearish = 0
                    self.primary_block_reason = "Order Executed"
                else:
                    if not st_bearish: self.primary_block_reason = "SuperTrend is Bullish"
                    elif c_close >= self.vwap: self.primary_block_reason = "Price above session VWAP"
                    else: self.primary_block_reason = f"Score {sell_score:.1f} < 2.0 threshold"
                    
                    if sell_score >= 1.5 and self.telegram:
                        if self.last_near_miss_notified is None or (c_ts - self.last_near_miss_notified).total_seconds() >= 1800:
                            self.telegram.notify_near_miss(self.symbol, "SELL (PE)", sell_score, sell_bd, c_close)
                            self.last_near_miss_notified = c_ts

        # Decrement pending lookbacks
        if self.pending_macd_bullish > 0: self.pending_macd_bullish -= 1
        if self.pending_macd_bearish > 0: self.pending_macd_bearish -= 1

    def _get_live_atm_contract(self, spot: float, opt_type: str) -> Optional[Dict]:
        """Fetch the exact live ATM option contract from Kite API."""
        if self.nfo_df is None or self.nfo_df.empty: return None
        try:
            atm_strike = round(spot / self.strike_gap) * self.strike_gap
            today = date.today()
            opts = self.nfo_df[(self.nfo_df['name'] == self.symbol) &
                               (self.nfo_df['strike'] == atm_strike) &
                               (self.nfo_df['instrument_type'] == opt_type) &
                               (self.nfo_df['expiry'] >= today)].sort_values('expiry')
            if opts.empty: return None
            contract = opts.iloc[0]
            tsym = contract['tradingsymbol']
            token = int(contract['instrument_token'])
            lot = int(contract['lot_size'])
            
            # Fetch live market quote from Kite API
            quote = self.kite.ltp([f"NFO:{tsym}"])
            live_price = quote.get(f"NFO:{tsym}", {}).get("last_price", 0.0)
            if live_price <= 0:
                self.logger(f"⚠️ [{self.symbol}] LTP unavailable for {tsym} — aborting entry (refusing to fabricate price)")
                return None
            
            return {
                "tradingsymbol": tsym, "token": token, "lot_size": lot, "strike": atm_strike, "live_ltp": live_price
            }
        except Exception as e:
            self.logger(f"⚠️ Live option resolution error for {self.symbol}: {e}")
            return None

    def _enter_position(self, signal_type: str, spot: float, in_gz: bool):
        # Capital Risk Guardrail Check
        if self.capital_tracker and self.bot_controller:
            active_count = sum(1 for t in self.bot_controller.traders.values() if t.position is not None)
            can_open, reason = self.capital_tracker.can_open_position(active_count)
            if not can_open:
                self.logger(f"⚠️ [{self.symbol}] Entry blocked: {reason}")
                if self.telegram and (self.capital_tracker.session_capital < self.capital_tracker.per_slot_capital or self.capital_tracker.session_capital <= 0):
                    self.telegram.notify_capital_alert(
                        self.symbol,
                        self.capital_tracker.session_capital,
                        self.capital_tracker.per_slot_capital,
                        self.capital_tracker.overall_pnl,
                        reason
                    )
                return

        opt_type = "CE" if signal_type == "BUY" else "PE"
        contract = self._get_live_atm_contract(spot, opt_type)
        if not contract:
            self.logger(f"❌ Failed to resolve live ATM option for {self.symbol}")
            return
            
        entry_price = contract["live_ltp"]
        sl = max(0.50, entry_price * 0.75)  # 25% option stop-loss
        
        lot_mult = 2 if in_gz else 1
        qty = contract["lot_size"] * lot_mult
        
        # Trade cost validation — qty × premium must not exceed session capital
        if self.capital_tracker:
            trade_cost = qty * entry_price
            if trade_cost > self.capital_tracker.session_capital:
                # Try reducing to 1 lot if GZ doubled
                if lot_mult == 2:
                    qty = contract["lot_size"]
                    trade_cost = qty * entry_price
                    lot_mult = 1
                    self.logger(f"⚠️ [{self.symbol}] GZ 2-lot (₹{qty * entry_price * 2:,.0f}) exceeds capital — reduced to 1 lot")
                if trade_cost > self.capital_tracker.session_capital:
                    self.logger(f"⚠️ [{self.symbol}] Trade cost ₹{trade_cost:,.0f} exceeds session capital ₹{self.capital_tracker.session_capital:,.0f} — entry blocked")
                    return
        
        # Execute via ExecutionEngine (paper or live)
        if self.execution_engine:
            result = self.execution_engine.place_entry_order(
                contract["tradingsymbol"], qty, entry_price, "BUY"
            )
            if result["status"] != "COMPLETE":
                self.logger(f"❌ [{self.symbol}] Entry order failed: {result.get('error', result['status'])}")
                return
            entry_price = result["fill_price"]  # Use actual fill price
            qty = result.get("fill_qty", qty)
        
        self.position = Position(
            symbol=self.symbol,
            option_type=opt_type,
            strike=contract["strike"],
            tradingsymbol=contract["tradingsymbol"],
            option_token=contract["token"],
            entry_price=entry_price,
            sl=sl,
            quantity=qty,
            entry_time=now_ist(),
            spot_at_entry=spot
        )
        
        # Subscribe option token to WebSocket for real-time SL monitoring
        self.last_option_ltp = entry_price
        if self.bot_controller and self.bot_controller.ticker:
            try:
                self.bot_controller.ticker.subscribe([contract["token"]])
                self.bot_controller.ticker.set_mode(self.bot_controller.ticker.MODE_FULL, [contract["token"]])
                self.bot_controller.token_to_symbol[contract["token"]] = f"OPT_{self.symbol}"
                self.logger(f"📡 Subscribed option token {contract['token']} ({contract['tradingsymbol']}) to WebSocket")
            except Exception as e:
                self.logger(f"⚠️ Failed to subscribe option token: {e}")
        
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        gz_tag = "2 LOTS (Golden Zone Confluence)" if in_gz and lot_mult == 2 else "1 LOT (Standard)"
        trade_amount = qty * entry_price
        print(f"\n{'='*60}", flush=True)
        print(f"{emoji} [{self.symbol} ENTRY] {contract['tradingsymbol']} | {gz_tag}", flush=True)
        print(f"   Entry LTP: ₹{entry_price:.2f} | Stop-Loss: ₹{sl:.2f} | Quantity: {qty} | Amount Required: ₹{trade_amount:,.2f}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_entry(
                self.symbol, opt_type, contract["strike"], entry_price, 0, sl, qty, f"{signal_type} ({gz_tag})",
                amount_required=trade_amount
            )

    def _check_exit(self, current_spot: float):
        """Tick SL check using WebSocket-fed option LTP (no blocking REST calls)."""
        if not self.position: return
        # Use last option LTP from WebSocket feed (updated in on_ticks)
        current_opt_ltp = getattr(self, 'last_option_ltp', 0.0)
        if current_opt_ltp > 0 and current_opt_ltp <= self.position.sl:
            self.logger(f"🔔 [{self.symbol}] SL triggered: Option LTP ₹{current_opt_ltp:.2f} ≤ SL ₹{self.position.sl:.2f}")
            self._close_position("SL_HIT", current_opt_ltp)

    def _close_position(self, reason: str, exit_price: float = None):
        if not self.position: return
        
        # Get exit price with retry + WebSocket fallback
        if exit_price is None or exit_price <= 0:
            # Try REST with retries
            for attempt in range(3):
                try:
                    quote = self.kite.ltp([f"NFO:{self.position.tradingsymbol}"])
                    fetched = quote.get(f"NFO:{self.position.tradingsymbol}", {}).get("last_price", 0.0)
                    if fetched > 0:
                        exit_price = fetched
                        break
                except Exception as e:
                    self.logger(f"⚠️ Exit quote retry {attempt+1}/3 failed: {e}")
                    if attempt < 2:
                        time.sleep(0.5)
            
            # WebSocket LTP fallback
            if exit_price is None or exit_price <= 0:
                ws_ltp = getattr(self, 'last_option_ltp', 0.0)
                if ws_ltp > 0:
                    exit_price = ws_ltp
                    self.logger(f"⚠️ [{self.symbol}] Using WebSocket LTP ₹{ws_ltp:.2f} for exit (REST unavailable)")
                else:
                    # Absolute last resort — log warning, use entry price
                    exit_price = self.position.entry_price
                    self.logger(f"⚠️ [{self.symbol}] EXIT PRICE UNAVAILABLE — using entry price ₹{exit_price:.2f} (P&L may be inaccurate)")
        
        # Execute exit via ExecutionEngine
        if self.execution_engine:
            result = self.execution_engine.place_exit_order(
                self.position.tradingsymbol, self.position.quantity, "SELL", exit_price
            )
            if result["status"] == "COMPLETE" and result["fill_price"] > 0:
                exit_price = result["fill_price"]
        
        self.position.close(exit_price, reason)
        self.trades.append(self.position)
        
        # Record to capital tracker for daily loss limit
        if self.capital_tracker:
            self.capital_tracker.record_trade_pnl(self.position.net_pnl)
        
        emoji = "✅" if self.position.net_pnl > 0 else "🛑"
        print(f"\n{emoji} [{self.symbol} EXIT] {self.position.tradingsymbol} - {reason}", flush=True)
        print(f"   Entry: ₹{self.position.entry_price:.2f} → Exit: ₹{exit_price:.2f} | Net P&L: ₹{self.position.net_pnl:+,.2f}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_exit(
                self.symbol, self.position.option_type, self.position.strike,
                self.position.entry_price, exit_price, self.position.net_pnl, reason
            )
        
        # Unsubscribe option token from WebSocket
        if self.bot_controller and self.bot_controller.ticker and self.position:
            try:
                opt_token = self.position.option_token
                self.bot_controller.ticker.unsubscribe([opt_token])
                self.bot_controller.token_to_symbol.pop(opt_token, None)
            except Exception:
                pass
        
        self.position = None
        self.last_option_ltp = 0.0

    def get_diagnostics(self) -> Dict:
        st_state = "BULLISH" if (self.supertrend_value > 0 and self.ltp > self.supertrend_value) else ("BEARISH" if (self.supertrend_value > 0 and self.ltp < self.supertrend_value) else "NEUTRAL")
        return {
            "symbol": self.symbol,
            "ticks": self.tick_count,
            "candles": self.candle_count,
            "ltp": self.ltp,
            "trend": st_state,
            "lot_size": self.lot_size,
            "pcr": self.pcr_tracker.get_pcr(self.symbol) if self.pcr_tracker else 1.0,
            "peak_score": round(self.peak_score, 1),
            "peak_direction": self.peak_score_direction,
            "peak_breakdown": self.peak_score_breakdown,
            "block_reason": self.primary_block_reason,
            "active_position": True if self.position else False
        }


# ============================================================
# MASTER STOCK BOT CONTROLLER
# ============================================================
class StockOptionsBot:
    def __init__(self):
        self.is_running = False
        self.stop_event = Event()
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        self.telegram = TelegramNotifier() if TELEGRAM_AVAILABLE else None
        self.pcr_tracker: Optional[NativeKitePCRTracker] = None
        self.nfo_df: Optional[pd.DataFrame] = None
        self.traders: Dict[str, StockTrader] = {}
        self.token_to_symbol: Dict[int, str] = {}
        self.log_file = LOG_DIR / f"stocks_{now_ist().strftime('%Y%m%d')}.log"
        self.capital_tracker = CapitalTracker(base_capital=TOTAL_CAPITAL, per_slot_capital=PER_STOCK_CAPITAL_LIMIT, logger=self._log)
        self.execution_engine = ExecutionEngine(paper_trading=PAPER_TRADING, logger=self._log)
        self._needs_restart = False  # Flag for safe WebSocket restart (avoids deadlock)

    def _log(self, message: str):
        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {message}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def authenticate(self) -> bool:
        if not KITE_AVAILABLE: return False
        creds = load_credentials()
        auto_login = KiteAutoLogin(
            api_key=creds["api_key"], api_secret=creds["api_secret"],
            user_id=creds["user_id"], password=creds["password"],
            totp_secret=creds["totp_secret"], headless=True
        )
        saved = auto_login.get_saved_token()
        if saved:
            self.kite = KiteConnect(api_key=creds["api_key"])
            self.kite.set_access_token(saved)
            try:
                prof = self.kite.profile()
                self._log(f"✅ Reusing valid access token. Logged in as: {prof.get('user_name')}")
                self.execution_engine.kite = self.kite
                return True
            except Exception:
                pass
        token = auto_login.login()
        if token:
            self.kite = auto_login.kite
            self.execution_engine.kite = self.kite
            self._log("✅ Fresh auto-login successful!")
            return True
        return False

    def load_market_metadata(self):
        """Fetch live NSE spot tokens and NFO lot sizes directly from Kite API."""
        if not self.kite: return
        self._log("📊 Downloading live Kite market metadata (NSE & NFO)...")
        
        # 1. Download NFO instruments
        nfo_list = self.kite.instruments('NFO')
        self.nfo_df = pd.DataFrame(nfo_list)
        
        # 2. Download NSE Spot tokens
        nse_list = self.kite.instruments('NSE')
        df_nse = pd.DataFrame(nse_list)
        
        self.pcr_tracker = NativeKitePCRTracker(self.kite, self._log)
        
        for sym, cfg in STOCKS.items():
            # Get Spot token
            spot_match = df_nse[df_nse['tradingsymbol'] == sym]
            if not spot_match.empty:
                cfg["token"] = int(spot_match.iloc[0]['instrument_token'])
                self.token_to_symbol[cfg["token"]] = sym
                
            # Get real lot size & strike gap from NFO
            sym_futs = self.nfo_df[(self.nfo_df['name'] == sym) & (self.nfo_df['instrument_type'] == 'FUT')]
            if not sym_futs.empty:
                cfg["lot_size"] = int(sym_futs.iloc[0]['lot_size'])
                
            sym_opts = self.nfo_df[(self.nfo_df['name'] == sym) & (self.nfo_df['instrument_type'] == 'CE')]
            strikes = sorted(sym_opts['strike'].unique())
            if len(strikes) > 1:
                cfg["strike_gap"] = float(strikes[1] - strikes[0])
                
            self.traders[sym] = StockTrader(sym, cfg, self._log, self.kite, self.nfo_df, self.telegram, self.pcr_tracker, self.capital_tracker, self, self.execution_engine)
            self._log(f"   ✓ {sym:<10} | Spot Token: {cfg['token']} | Lot Size: {cfg['lot_size']} | Strike Step: {cfg['strike_gap']}")

    def fetch_historical_and_gz(self):
        """Fetch historical candles and calculate the 75-Minute Alligator Golden Zone.

        Kite Connect does not natively support 75min intervals.
        We synthesize 75-minute candles from 15-minute data by grouping every 5 consecutive
        candles — producing OHLCV values mathematically identical to native 75min candles.
        (375 min trading session ÷ 75 = 5 clean candles/day — the backtested rhythm.)

        Kite supported intervals: minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day
        """
        if not self.kite: return
        self._log("📊 Fetching historical candles & calculating 75M Alligator Golden Zones...")
        to_d = now_ist()
        from_d = to_d - timedelta(days=15)

        for sym, trader in self.traders.items():
            token = STOCKS[sym]["token"]
            interval = f"{trader.timeframe_minutes}minute"

            # 1. Fetch intraday candles for indicator warm-up
            try:
                data = self.kite.historical_data(token, from_date=from_d, to_date=to_d, interval=interval)
                trader.candles = []  # Clear previous candles to prevent duplicates on retry
                for c in data[-50:]:
                    trader.candles.append({
                        "timestamp": c["date"], "open": c["open"], "high": c["high"],
                        "low": c["low"], "close": c["close"], "volume": c.get("volume", 0)
                    })
            except Exception as e:
                self._log(f"⚠️ Historical candle fetch failed for {sym}: {e}")
            
            time.sleep(0.5)  # Rate limit: Kite allows 3 req/sec for historical

            # 2. Synthesize 75-minute candles from 15-minute data
            # Group every 5 consecutive 15-minute candles → 1 synthetic 75-minute candle
            # Session-anchored: only group within same trading day (09:15–15:30)
            try:
                d15 = self.kite.historical_data(token, from_date=from_d, to_date=to_d, interval="15minute")
                df15 = pd.DataFrame(d15)

                if not df15.empty and len(df15) >= 5:
                    # Filter to market hours only (09:15–15:30) to avoid overnight contamination
                    df15['date_col'] = df15['date'].apply(lambda x: x.date() if hasattr(x, 'date') else x)
                    df15['time_col'] = df15['date'].apply(lambda x: x.time() if hasattr(x, 'time') else None)
                    market_open = datetime.strptime("09:15", "%H:%M").time()
                    market_close = datetime.strptime("15:30", "%H:%M").time()
                    df15 = df15[(df15['time_col'] >= market_open) & (df15['time_col'] < market_close)]
                    
                    # Group by trading day, then within each day group consecutive 5 candles
                    groups = []
                    for day_date, day_df in df15.groupby('date_col'):
                        day_df = day_df.reset_index(drop=True)
                        n_complete = (len(day_df) // 5) * 5
                        for i in range(0, n_complete, 5):
                            block = day_df.iloc[i:i+5]
                            if len(block) == 5:
                                groups.append({
                                    "open":  block['open'].iloc[0],
                                    "high":  block['high'].max(),
                                    "low":   block['low'].min(),
                                    "close": block['close'].iloc[-1],
                                })

                    df75 = pd.DataFrame(groups)

                    if not df75.empty:
                        # Use last 20 synthetic candles (20 × 75min = ~4 trading days)
                        lookback = df75.iloc[-20:]
                        p_high = lookback['high'].max()
                        p_low  = lookback['low'].min()
                        rng    = p_high - p_low

                        trader.gz_bull_bot = p_low  + (0.50 * rng)
                        trader.gz_bull_top = p_low  + (0.65 * rng)
                        trader.gz_bear_top = p_high - (0.50 * rng)
                        trader.gz_bear_bot = p_high - (0.65 * rng)
                        self._log(
                            f"   {sym} 75M GZ: Bull [{trader.gz_bull_bot:.1f}–{trader.gz_bull_top:.1f}]"
                            f" | Bear [{trader.gz_bear_bot:.1f}–{trader.gz_bear_top:.1f}]"
                            f"  (built from {len(df75)} session-anchored 75M candles)"
                        )
            except Exception as e:
                self._log(f"⚠️ 75M GZ skipped for {sym}: {e} (GZ boost disabled today)")
            
            time.sleep(0.5)  # Rate limit between stocks

    def start_live_feed(self):
        creds = load_credentials()
        self.ticker = KiteTicker(creds["api_key"], self.kite.access_token)
        tokens = list(self.token_to_symbol.keys())
        self._last_tick_time = None  # Set only when first real tick arrives
        self._feed_start_time = now_ist()  # Set immediately on feed launch so watchdog tracks connection time
        self._log(f"📡 Initializing WebSocket feed for {len(tokens)} tokens: {tokens}")

        def on_connect(ws, resp):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            self._feed_start_time = now_ist()  # Refresh on successful connection
            self._log(f"✅ WebSocket connected — subscribed to {len(tokens)} stock tokens.")
            
        def on_ticks(ws, ticks):
            self._last_tick_time = now_ist()
            for t in ticks:
                tok = t.get("instrument_token")
                if tok in self.token_to_symbol:
                    sym_key = self.token_to_symbol[tok]
                    ltp = t.get("last_price")
                    vol = t.get("volume_traded", 0)
                    
                    if sym_key.startswith("OPT_"):
                        # Option tick — update last_option_ltp for SL monitoring
                        real_sym = sym_key[4:]  # Remove "OPT_" prefix
                        if real_sym in self.traders and ltp:
                            self.traders[real_sym].last_option_ltp = ltp
                    elif ltp and sym_key in self.traders:
                        # Spot tick — process for candle building
                        self.traders[sym_key].process_tick(ltp, now_ist(), vol)

        def on_error(ws, code, reason):
            self._log(f"⚠️ WebSocket error [{code}]: {reason} — will attempt reconnect.")

        def on_close(ws, code, reason):
            self._log(f"⚠️ WebSocket closed [{code}]: {reason}.")
            # KiteTicker with reconnect=True will auto-retry; log it so we know
            if self.is_running and self.is_market_open():
                self._log("🔄 Market is open — waiting for KiteTicker auto-reconnect...")

        def on_reconnect(ws, attempt):
            self._log(f"🔄 WebSocket reconnecting... attempt #{attempt}")

        def on_noreconnect(ws):
            self._log("❌ WebSocket exhausted all reconnect attempts — setting restart flag.")
            # Don't restart from WS callback thread (deadlock risk) — let app.py watchdog handle it
            self._needs_restart = True

        self.ticker.on_connect = on_connect
        self.ticker.on_ticks = on_ticks
        self.ticker.on_error = on_error
        self.ticker.on_close = on_close
        self.ticker.on_reconnect = on_reconnect
        self.ticker.on_noreconnect = on_noreconnect
        self.ticker.connect(threaded=True)

    def _restart_ticker(self):
        """Hard-restart the KiteTicker. MUST be called from main thread, not WS callback."""
        self._needs_restart = False
        try:
            if self.ticker:
                self.ticker.close()
        except Exception:
            pass
        time.sleep(5)
        self.start_live_feed()
        self._log("✅ WebSocket ticker restarted successfully.")

    def is_market_open(self) -> bool:
        now = now_ist()
        if now.weekday() >= 5: return False
        return now.replace(hour=9, minute=15, second=0) <= now <= now.replace(hour=15, minute=30, second=0)

    def generate_report(self):
        today = now_ist().strftime("%Y-%m-%d")
        # Ensure any remaining open positions are force-closed so their P&L enters EOD accounting
        for sym, tr in self.traders.items():
            if tr.position is not None:
                self._log(f"⚠️ [{sym}] Force-closing open position before generating EOD report")
                try:
                    tr._close_position("EOD_REPORT_CLOSE")
                except Exception as e:
                    self._log(f"❌ Failed to close {sym} before EOD report: {e}")

        all_trades = [t for tr in self.traders.values() for t in tr.trades]
        tot_pnl = sum(t.net_pnl for t in all_trades)
        wins = [t for t in all_trades if t.net_pnl > 0]
        
        sec_data = {}
        for sym, tr in self.traders.items():
            sym_pnl = sum(t.net_pnl for t in tr.trades)
            sym_wins = len([t for t in tr.trades if t.net_pnl > 0])
            sec_data[sym] = {"trades": len(tr.trades), "pnl": sym_pnl, "wins": sym_wins, "losses": len(tr.trades) - sym_wins}
            
        diagnostics = {s: t.get_diagnostics() for s, t in self.traders.items()}
        
        # End of day capital processing
        cap_summary = self.capital_tracker.end_day(tot_pnl)
        
        if self.telegram:
            self.telegram.notify_daily_summary(today, sec_data, tot_pnl, diagnostics=diagnostics, capital_summary=cap_summary)
            
        report = {
            "date": today,
            "total_capital": TOTAL_CAPITAL,
            "capital_summary": cap_summary,
            "total_trades": len(all_trades),
            "net_pnl": tot_pnl,
            "diagnostics": diagnostics
        }
        with open(LOG_DIR / f"stocks_report_{today}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    def stop(self):
        self.is_running = False
        self.stop_event.set()
        if self.ticker:
            try:
                self.ticker.close()
            except Exception:
                pass


if __name__ == "__main__":
    from app import run_trading_bot
    run_trading_bot()
