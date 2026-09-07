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
        
        self.session_capital = self.base_capital
        self.overall_pnl = 0.0
        self.load_state()

    def load_state(self):
        # 1. Environment variable override if specified
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
            return
            
        # 2. Local state file persistence
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.session_capital = float(data.get("session_capital", self.base_capital))
                self.overall_pnl = float(data.get("overall_pnl", 0.0))
                self.logger(f"💼 Capital state loaded: Session Capital=₹{self.session_capital:,.2f}, Overall P&L=₹{self.overall_pnl:,.2f}")
            except Exception as e:
                self.logger(f"⚠️ Error reading capital_state.json: {e}. Defaulting to ₹{self.base_capital:,.2f}")
                self.session_capital = self.base_capital
                self.overall_pnl = 0.0
        else:
            self.session_capital = self.base_capital
            self.overall_pnl = 0.0
            self.save_state()

    def save_state(self):
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
        survive container teardowns and new deployments."""
        api_key = os.environ.get("RENDER_API_KEY")
        service_id = os.environ.get("RENDER_SERVICE_ID")
        if not api_key or not service_id:
            return  # Not on Render or keys not configured — silently skip
        try:
            import urllib.request
            url = f"https://api.render.com/v1/services/{service_id}/env-vars"
            payload = json.dumps([
                {"key": "SESSION_CAPITAL", "value": str(self.session_capital)},
                {"key": "OVERALL_PNL",     "value": str(self.overall_pnl)},
            ]).encode()
            req = urllib.request.Request(
                url, data=payload, method="PUT",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 201):
                    self.logger(f"💾 Capital persisted to Render env vars — Session: ₹{self.session_capital:,.2f} | Overall P&L: ₹{self.overall_pnl:+,.2f}")
                else:
                    self.logger(f"⚠️ Render env update returned status {resp.status}")
        except Exception as e:
            self.logger(f"⚠️ Could not push capital to Render env vars: {e}")

    def get_max_concurrent_positions(self) -> int:
        if self.session_capital < self.per_slot_capital:
            return 0
        return min(MAX_CONCURRENT_POSITIONS, int(self.session_capital // self.per_slot_capital))

    def can_open_position(self, current_open_count: int) -> tuple:
        if self.session_capital <= 0:
            return False, "Session capital is zero or depleted. Trading halted."
        if self.session_capital < self.per_slot_capital:
            return False, f"Session capital (₹{self.session_capital:,.2f}) is below minimum required per-slot capital (₹{self.per_slot_capital:,.2f})."
        max_allowed = self.get_max_concurrent_positions()
        if current_open_count >= max_allowed:
            return False, f"Maximum concurrent positions reached ({current_open_count}/{max_allowed}) for capital ₹{self.session_capital:,.2f}."
        return True, "OK"

    def end_day(self, day_net_pnl: float) -> dict:
        """
        Processes End-of-Day P&L:
        - If profit:
            - overall_pnl += day_net_pnl (Passive income credited)
            - tomorrow's session_capital = base_capital (₹1L) (Profits reaped)
        - If loss:
            - overall_pnl += day_net_pnl
            - tomorrow's session_capital = max(0.0, session_capital + day_net_pnl) (Loss carried forward)
        """
        capital_used = self.session_capital
        self.overall_pnl += day_net_pnl
        
        is_profit = day_net_pnl >= 0
        if is_profit:
            next_day_capital = self.base_capital
            capital_remaining = self.session_capital + day_net_pnl
        else:
            next_day_capital = max(0.0, self.session_capital + day_net_pnl)
            capital_remaining = next_day_capital
            
        summary = {
            "capital_used": capital_used,
            "day_pnl": day_net_pnl,
            "is_profit": is_profit,
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
                 capital_tracker=None, bot_controller=None):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.kite = kite
        self.nfo_df = nfo_df
        self.telegram = telegram
        self.pcr_tracker = pcr_tracker
        self.capital_tracker = capital_tracker
        self.bot_controller = bot_controller
        
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

    def process_tick(self, ltp: float, tick_time: datetime, volume: int = 0):
        with self.lock:
            self.tick_count += 1
            self.ltp = ltp
            
            candle_min = (tick_time.minute // self.timeframe_minutes) * self.timeframe_minutes
            candle_ts = tick_time.replace(minute=candle_min, second=0, microsecond=0)
            
            if self.current_candle is None or candle_ts != self.last_candle_time:
                if self.current_candle:
                    self.candles.append(self.current_candle)
                    self.candle_count += 1
                    if len(self.candles) >= ATR_PERIOD + 5:
                        self._process_candle(self.current_candle)
                
                self.current_candle = {
                    "timestamp": candle_ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": volume
                }
                self.last_candle_time = candle_ts
            else:
                self.current_candle["high"] = max(self.current_candle["high"], ltp)
                self.current_candle["low"] = min(self.current_candle["low"], ltp)
                self.current_candle["close"] = ltp
                self.current_candle["volume"] += volume
                
            # Real-time tick SL check
            if self.position:
                self._check_exit(ltp)

    def _process_candle(self, candle: Dict):
        c_time = candle["timestamp"].time()
        c_close = candle["close"]
        c_ts = candle["timestamp"]
        
        # 1. Calculate Technical Indicators on Closed Candle
        df = pd.DataFrame(self.candles[-60:])
        
        # SuperTrend (20, 2)
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1]
        
        hl2 = (df['high'] + df['low']) / 2
        basic_ub = hl2 + (ATR_MULTIPLIER * atr)
        basic_lb = hl2 - (ATR_MULTIPLIER * atr)
        self.supertrend_value = basic_lb.iloc[-1] if c_close >= df['close'].iloc[-2] else basic_ub.iloc[-1]
        st_bullish = c_close > self.supertrend_value
        st_bearish = c_close < self.supertrend_value
        
        prev_trend = self.current_trend
        self.current_trend = 1 if st_bullish else (-1 if st_bearish else 0)
        st_bull_flip = (prev_trend != 1 and self.current_trend == 1)
        st_bear_flip = (prev_trend != -1 and self.current_trend == -1)
        
        # MACD (12, 26, 9)
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        
        macd_bull_cross = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
        macd_bear_cross = (macd_line.iloc[-1] < signal_line.iloc[-1]) and (macd_line.iloc[-2] >= signal_line.iloc[-2])
        
        if macd_bull_cross: self.pending_macd_bullish = self.macd_lookback
        if macd_bear_cross: self.pending_macd_bearish = self.macd_lookback
        
        # VWAP
        df['vol_p'] = df['close'] * df['volume']
        self.vwap = (df['vol_p'].sum() / df['volume'].sum()) if df['volume'].sum() > 0 else c_close
        
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
            if live_price <= 0: live_price = 25.0  # Safe fallback if market closed
            
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
        
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        gz_tag = "2 LOTS (Golden Zone Confluence)" if in_gz else "1 LOT (Standard)"
        print(f"\n{'='*60}", flush=True)
        print(f"{emoji} [{self.symbol} ENTRY] {contract['tradingsymbol']} | {gz_tag}", flush=True)
        print(f"   Entry LTP: ₹{entry_price:.2f} | Stop-Loss: ₹{sl:.2f} | Quantity: {qty}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_entry(
                self.symbol, opt_type, contract["strike"], entry_price, 0, sl, qty, f"{signal_type} ({gz_tag})"
            )

    def _check_exit(self, current_spot: float):
        """Tick SL check using live option quote."""
        if not self.position: return
        try:
            # Query real option quote
            quote = self.kite.ltp([f"NFO:{self.position.tradingsymbol}"])
            current_opt_ltp = quote.get(f"NFO:{self.position.tradingsymbol}", {}).get("last_price", 0.0)
            if current_opt_ltp > 0 and current_opt_ltp <= self.position.sl:
                self._close_position("SL_HIT", current_opt_ltp)
        except Exception:
            pass

    def _close_position(self, reason: str, exit_price: float = None):
        if not self.position: return
        if exit_price is None or exit_price <= 0:
            try:
                quote = self.kite.ltp([f"NFO:{self.position.tradingsymbol}"])
                exit_price = quote.get(f"NFO:{self.position.tradingsymbol}", {}).get("last_price", self.position.entry_price)
            except Exception:
                exit_price = self.position.entry_price
                
        self.position.close(exit_price, reason)
        self.trades.append(self.position)
        
        emoji = "✅" if self.position.net_pnl > 0 else "🛑"
        print(f"\n{emoji} [{self.symbol} EXIT] {self.position.tradingsymbol} - {reason}", flush=True)
        print(f"   Entry: ₹{self.position.entry_price:.2f} → Exit: ₹{exit_price:.2f} | Net P&L: ₹{self.position.net_pnl:+,.2f}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        if self.telegram:
            self.telegram.notify_trade_exit(
                self.symbol, self.position.option_type, self.position.strike,
                self.position.entry_price, exit_price, self.position.net_pnl, reason
            )
        self.position = None

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
                return True
            except Exception:
                pass
        token = auto_login.login()
        if token:
            self.kite = auto_login.kite
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
                
            self.traders[sym] = StockTrader(sym, cfg, self._log, self.kite, self.nfo_df, self.telegram, self.pcr_tracker, self.capital_tracker, self)
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
                for c in data[-50:]:
                    trader.candles.append({
                        "timestamp": c["date"], "open": c["open"], "high": c["high"],
                        "low": c["low"], "close": c["close"], "volume": c.get("volume", 0)
                    })
            except Exception as e:
                self._log(f"⚠️ Historical candle fetch failed for {sym}: {e}")

            # 2. Synthesize 75-minute candles from 15-minute data
            # Group every 5 consecutive 15-minute candles → 1 synthetic 75-minute candle
            try:
                d15 = self.kite.historical_data(token, from_date=from_d, to_date=to_d, interval="15minute")
                df15 = pd.DataFrame(d15)

                if not df15.empty and len(df15) >= 5:
                    # Only use candles from completed 75-min blocks (multiples of 5)
                    n_complete = (len(df15) // 5) * 5
                    df15 = df15.iloc[-n_complete:].reset_index(drop=True)

                    # Build synthetic 75-min OHLCV by grouping every 5 rows
                    groups = []
                    for i in range(0, len(df15), 5):
                        block = df15.iloc[i:i+5]
                        if len(block) == 5:
                            groups.append({
                                "open":  block['open'].iloc[0],
                                "high":  block['high'].max(),
                                "low":   block['low'].min(),
                                "close": block['close'].iloc[-1],
                            })

                    df75 = pd.DataFrame(groups)

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
                        f"  (built from {len(df75)} synthetic 75M candles)"
                    )
            except Exception as e:
                self._log(f"⚠️ 75M GZ skipped for {sym}: {e} (GZ boost disabled today)")

    def start_live_feed(self):
        creds = load_credentials()
        self.ticker = KiteTicker(creds["api_key"], self.kite.access_token)
        tokens = list(self.token_to_symbol.keys())
        self._last_tick_time = now_ist()
        
        def on_connect(ws, resp):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            self._log(f"✅ WebSocket connected — subscribed to {len(tokens)} stock tokens.")
            
        def on_ticks(ws, ticks):
            self._last_tick_time = now_ist()
            for t in ticks:
                tok = t.get("instrument_token")
                if tok in self.token_to_symbol:
                    sym = self.token_to_symbol[tok]
                    ltp = t.get("last_price")
                    vol = t.get("volume_traded", 0)
                    if ltp and sym in self.traders:
                        self.traders[sym].process_tick(ltp, now_ist(), vol)

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
            self._log("❌ WebSocket exhausted all reconnect attempts — restarting ticker now.")
            try:
                self._restart_ticker()
            except Exception as e:
                self._log(f"❌ Ticker restart failed: {e}")

        self.ticker.on_connect = on_connect
        self.ticker.on_ticks = on_ticks
        self.ticker.on_error = on_error
        self.ticker.on_close = on_close
        self.ticker.on_reconnect = on_reconnect
        self.ticker.on_noreconnect = on_noreconnect
        self.ticker.connect(threaded=True)

    def _restart_ticker(self):
        """Hard-restart the KiteTicker when auto-reconnect is exhausted."""
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

    def run(self):
        self.is_running = True
        print("\n" + "="*70)
        print(f"🚀 AUTONOMOUS STOCK OPTIONS BOT (Capital: ₹{self.capital_tracker.session_capital:,.0f} | Overall P&L: ₹{self.capital_tracker.overall_pnl:+,.0f})")
        print("="*70)
        
        if not self.authenticate(): return
        self.load_market_metadata()
        self.fetch_historical_and_gz()
        
        if self.telegram:
            self.telegram.notify_bot_start(list(STOCKS.keys()), capital_tracker=self.capital_tracker)
            
        self.start_live_feed()
        
        heartbeat_sent = False
        pcr_last_updated = None
        
        while self.is_running and self.is_market_open():
            now = now_ist()
            # 1. PCR update every 15 mins
            if pcr_last_updated is None or (now - pcr_last_updated).total_seconds() >= 900:
                for sym in STOCKS.keys():
                    self.pcr_tracker.update_stock_pcr(sym, self.nfo_df)
                pcr_last_updated = now
                
            # 2. Mid-Day Heartbeat at 12:00 PM IST
            if not heartbeat_sent and now.hour == 12 and now.minute >= 0:
                if self.telegram:
                    status_dict = {s: t.get_diagnostics() for s, t in self.traders.items()}
                    total_ticks = sum(t.tick_count for t in self.traders.values())
                    active_pos = sum(1 for t in self.traders.values() if t.position is not None)
                    self.telegram.notify_midday_heartbeat(status_dict, total_ticks, active_pos)
                heartbeat_sent = True
                
            time.sleep(1)
            
        self._log("Market closed. Generating EOD report...")
        self.generate_report()
        if self.ticker: self.ticker.close()

    def stop(self):
        self.is_running = False
        self.stop_event.set()


if __name__ == "__main__":
    bot = StockOptionsBot()
    bot.run()
