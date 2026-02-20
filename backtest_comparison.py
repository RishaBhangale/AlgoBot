#!/usr/bin/env python3
"""
Backtest Comparison: Old Strategy vs New Strategy

Compares:
  OLD: AND-gate Quad-Confirmation (MACD ∧ ST ∧ VWAP_below ∧ PCR)
  NEW: Scoring System (MACD:1 + ST:1/1.5 + VWAP_above:0.5 + PCR:0.5 >= 2.0)

Data: Real historical data from Kite Connect API.
      Uses ONLY real data — if not enough, reduces timeframe.
      No synthetic or generated data.

Usage:
    python3 backtest_comparison.py

Environment Variables Required:
    KITE_API_KEY, KITE_API_SECRET (+ valid access token from auto_login)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================
TIMEFRAME_MINUTES = 15
ATR_PERIOD = 20
ATR_MULTIPLIER = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

STOCKS = {
    "RELIANCE": {"lot_size": 250, "strike_gap": 20},
    "HDFCBANK": {"lot_size": 550, "strike_gap": 25},
    "ICICIBANK":{"lot_size": 700, "strike_gap": 12.5},
}


# ============================================================
# INDICATORS (same as main_stocks.py)
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
    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_prev_line'] = df['macd_line'].shift(1)
    df['macd_prev_signal'] = df['macd_signal'].shift(1)
    df['macd_bullish'] = (df['macd_line'] > df['macd_signal']) & (df['macd_prev_line'] <= df['macd_prev_signal'])
    df['macd_bearish'] = (df['macd_line'] < df['macd_signal']) & (df['macd_prev_line'] >= df['macd_prev_signal'])
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        return df
    
    df['date'] = df['timestamp'].dt.date
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['tp'] * df['volume']
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    return df


# ============================================================
# BACKTEST TRADE
# ============================================================
class BacktestTrade:
    def __init__(self, symbol, direction, entry_price, entry_time, spot, strike_gap):
        self.symbol = symbol
        self.direction = direction  # "BUY" or "SELL"
        self.option_type = "CE" if direction == "BUY" else "PE"
        self.strike = round(spot / strike_gap) * strike_gap
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.spot_entry = spot
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.peak_price = entry_price
        
        # Estimate option premium
        itm = max(0, spot - self.strike) if self.option_type == "CE" else max(0, self.strike - spot)
        self.premium = itm + spot * 0.004 + 15
        
        # SL at 15% of premium
        self.sl = self.premium * 0.85
    
    def close(self, exit_price, exit_time, reason):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) if self.direction == "BUY" else (self.entry_price - exit_price)


# ============================================================
# STRATEGY RUNNER
# ============================================================
class StrategyRunner:
    """Runs a strategy on historical data and collects trades."""
    
    def __init__(self, name: str, macd_lookback: int, use_scoring: bool, 
                 vwap_flipped: bool, entry_threshold: float = 2.0):
        self.name = name
        self.macd_lookback = macd_lookback
        self.use_scoring = use_scoring
        self.vwap_flipped = vwap_flipped
        self.entry_threshold = entry_threshold
        self.trades: List[BacktestTrade] = []
        self.blocked_signals: List[Dict] = []
    
    def run(self, df: pd.DataFrame, symbol: str, config: Dict) -> List[BacktestTrade]:
        """Run strategy on prepared DataFrame."""
        trades = []
        position = None
        pending_macd_bullish = 0
        pending_macd_bearish = 0
        
        for i in range(ATR_PERIOD + 5, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            close = row['close']
            vwap = row['vwap']
            trend = row['trend']
            prev_trend = row['prev_trend']
            macd_bullish = row['macd_bullish']
            macd_bearish = row['macd_bearish']
            
            # Exit logic: SuperTrend reversal or end of day
            if position:
                # ST reversal exit
                if position.direction == "BUY" and trend == -1 and prev_trend == 1:
                    position.close(close, row['timestamp'], "ST_REVERSAL")
                    trades.append(position)
                    position = None
                elif position.direction == "SELL" and trend == 1 and prev_trend == -1:
                    position.close(close, row['timestamp'], "ST_REVERSAL")
                    trades.append(position)
                    position = None
                
                # EOD exit (3:15 PM)
                if position and hasattr(row['timestamp'], 'hour'):
                    ts = row['timestamp']
                    if ts.hour >= 15 and ts.minute >= 15:
                        position.close(close, ts, "EOD")
                        trades.append(position)
                        position = None
                
                continue  # Skip entry logic if in position
            
            # MACD crossover tracking
            if macd_bullish:
                pending_macd_bullish = self.macd_lookback + 1
                pending_macd_bearish = 0
            if macd_bearish:
                pending_macd_bearish = self.macd_lookback + 1
                pending_macd_bullish = 0
            
            st_bullish = (trend == 1)
            st_bearish = (trend == -1)
            st_bullish_flip = (trend == 1 and prev_trend == -1)
            st_bearish_flip = (trend == -1 and prev_trend == 1)
            
            entered = False
            
            if self.use_scoring:
                # === NEW: Scoring System ===
                
                # BUY scoring
                if pending_macd_bullish > 0 and not entered:
                    score = 1.0  # MACD
                    reasons = {"MACD": True, "ST": False, "ST_FLIP": False, "VWAP": False}
                    if st_bullish:
                        score += 1.5 if st_bullish_flip else 1.0
                        reasons["ST"] = True
                        reasons["ST_FLIP"] = st_bullish_flip
                    if close > vwap:
                        score += 0.5
                        reasons["VWAP"] = True
                    
                    if score >= self.entry_threshold:
                        position = BacktestTrade(symbol, "BUY", close, row['timestamp'], close, config['strike_gap'])
                        pending_macd_bullish = 0
                        entered = True
                    elif score >= 1.0:
                        blocked_reason = "ST_MISSING" if not reasons["ST"] else "VWAP_MISSING"
                        self._track_blocked("BUY", close, score, row['timestamp'], df, i, symbol, reasons, blocked_reason)
                
                # SELL scoring
                if pending_macd_bearish > 0 and not entered:
                    score = 1.0  # MACD
                    reasons = {"MACD": True, "ST": False, "ST_FLIP": False, "VWAP": False}
                    if st_bearish:
                        score += 1.5 if st_bearish_flip else 1.0
                        reasons["ST"] = True
                        reasons["ST_FLIP"] = st_bearish_flip
                    if close < vwap:
                        score += 0.5
                        reasons["VWAP"] = True
                    
                    if score >= self.entry_threshold:
                        position = BacktestTrade(symbol, "SELL", close, row['timestamp'], close, config['strike_gap'])
                        pending_macd_bearish = 0
                        entered = True
                    elif score >= 1.0:
                        blocked_reason = "ST_MISSING" if not reasons["ST"] else "VWAP_MISSING"
                        self._track_blocked("SELL", close, score, row['timestamp'], df, i, symbol, reasons, blocked_reason)
            else:
                # === OLD: AND-gate Quad-Confirmation ===
                if pending_macd_bullish > 0 and st_bullish:
                    if close < vwap:
                        position = BacktestTrade(symbol, "BUY", close, row['timestamp'], close, config['strike_gap'])
                        pending_macd_bullish = 0
                        entered = True
                    else:
                        self._track_blocked("BUY", close, 0, row['timestamp'], df, i, symbol, 
                                           {"MACD": True, "ST": True, "VWAP": False}, "VWAP_WRONG_SIDE")
                
                if pending_macd_bearish > 0 and st_bearish and not entered:
                    if close > vwap:
                        position = BacktestTrade(symbol, "SELL", close, row['timestamp'], close, config['strike_gap'])
                        pending_macd_bearish = 0
                        entered = True
                    else:
                        self._track_blocked("SELL", close, 0, row['timestamp'], df, i, symbol,
                                           {"MACD": True, "ST": True, "VWAP": False}, "VWAP_WRONG_SIDE")
            
            # Decrement counters
            if pending_macd_bullish > 0:
                pending_macd_bullish -= 1
            if pending_macd_bearish > 0:
                pending_macd_bearish -= 1
        
        # Close any remaining position
        if position:
            position.close(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "BACKTEST_END")
            trades.append(position)
        
        self.trades.extend(trades)
        return trades
    
    def _track_blocked(self, direction: str, price: float, score: float, 
                       time, df: pd.DataFrame, idx: int, symbol: str = "",
                       reasons: Dict = None, blocked_reason: str = ""):
        """Track MFE and simulate what-if for blocked signals."""
        # MFE over next 5 candles
        mfe = 0.0
        mae = 0.0  # Max Adverse Excursion (worst drawdown)
        for j in range(1, min(6, len(df) - idx)):
            future_close = df.iloc[idx + j]['close']
            if direction == "BUY":
                move = future_close - price
            else:
                move = price - future_close
            mfe = max(mfe, move)
            mae = min(mae, move)
        
        # Simulate what-if: take this trade and exit at next ST reversal or EOD
        whatif_pnl = 0.0
        whatif_exit_reason = "NO_EXIT"
        for j in range(1, min(50, len(df) - idx)):  # Look up to 50 candles ahead
            future_row = df.iloc[idx + j]
            future_trend = future_row['trend']
            future_prev = future_row['prev_trend']
            
            # ST reversal exit
            if direction == "BUY" and future_trend == -1 and future_prev == 1:
                whatif_pnl = future_row['close'] - price
                whatif_exit_reason = "ST_REVERSAL"
                break
            elif direction == "SELL" and future_trend == 1 and future_prev == -1:
                whatif_pnl = price - future_row['close']
                whatif_exit_reason = "ST_REVERSAL"
                break
            
            # EOD exit
            ts = future_row['timestamp']
            if hasattr(ts, 'hour') and ts.hour >= 15 and ts.minute >= 15:
                if direction == "BUY":
                    whatif_pnl = future_row['close'] - price
                else:
                    whatif_pnl = price - future_row['close']
                whatif_exit_reason = "EOD"
                break
        
        self.blocked_signals.append({
            "symbol": symbol,
            "direction": direction,
            "price": round(price, 2),
            "score": score,
            "time": str(time),
            "mfe_5_candles": round(mfe, 2),
            "mae_5_candles": round(mae, 2),
            "whatif_pnl": round(whatif_pnl, 2),
            "whatif_exit": whatif_exit_reason,
            "blocked_reason": blocked_reason,
            "reasons": reasons or {},
        })
    
    def get_stats(self) -> Dict:
        if not self.trades:
            return {
                "total_trades": 0, "winners": 0, "losers": 0,
                "win_rate": 0, "total_pnl": 0, "avg_pnl": 0,
                "profit_factor": 0, "max_drawdown": 0,
                "avg_winner": 0, "avg_loser": 0,
                "blocked_count": len(self.blocked_signals),
                "avg_blocked_mfe": 0
            }
        
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        
        # Max drawdown
        equity = 0
        peak = 0
        max_dd = 0
        for t in self.trades:
            equity += t.pnl
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        
        avg_blocked_mfe = 0
        if self.blocked_signals:
            avg_blocked_mfe = np.mean([b['mfe_5_candles'] for b in self.blocked_signals])
        
        return {
            "total_trades": len(self.trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(self.trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(self.trades), 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "max_drawdown": round(max_dd, 2),
            "avg_winner": round(gross_profit / len(winners), 2) if winners else 0,
            "avg_loser": round(-gross_loss / len(losers), 2) if losers else 0,
            "blocked_count": len(self.blocked_signals),
            "avg_blocked_mfe": round(avg_blocked_mfe, 2)
        }


# ============================================================
# DATA LOADING
# ============================================================
def load_data_from_cache(symbol: str) -> Optional[pd.DataFrame]:
    """Load data from cached CSV if available."""
    cache_path = LOG_DIR / f"backtest_data_{symbol}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  📁 {symbol}: {len(df)} candles (cached)")
        return df
    return None


def save_data_to_cache(df: pd.DataFrame, symbol: str):
    """Cache data for offline re-runs."""
    cache_path = LOG_DIR / f"backtest_data_{symbol}.csv"
    df.to_csv(cache_path, index=False)


def load_all_stocks_from_kite(symbols: List[str], days: int = 15) -> Dict[str, pd.DataFrame]:
    """
    Authenticate ONCE then fetch all stocks in a single session.
    Returns dict of symbol -> DataFrame.
    """
    # Load .env file for local runs
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    
    try:
        from kiteconnect import KiteConnect
        from auto_login import KiteAutoLogin, load_credentials
    except ImportError:
        print("  ❌ kiteconnect or auto_login not available")
        return {}
    
    try:
        creds = load_credentials()
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
            print("  ❌ Login failed")
            return {}
        
        kite = auto_login.kite
        print("  ✅ Authenticated successfully")
        
        # Get all instrument tokens at once
        instruments = kite.instruments("NSE")
        token_map = {}
        for inst in instruments:
            if inst['tradingsymbol'] in symbols:
                token_map[inst['tradingsymbol']] = inst['instrument_token']
        
        print(f"  📋 Found tokens for: {list(token_map.keys())}")
        
        # Fetch data for each stock
        results = {}
        to_date = datetime.now(IST) if IST else datetime.now()
        from_date = to_date - timedelta(days=days)
        
        for symbol in symbols:
            if symbol not in token_map:
                print(f"  ⚠️ {symbol}: No instrument token found")
                continue
            
            try:
                import time as _time
                _time.sleep(0.5)  # Rate limit: don't hammer Kite API
                
                data = kite.historical_data(
                    instrument_token=token_map[symbol],
                    from_date=from_date,
                    to_date=to_date,
                    interval="15minute"
                )
                
                if data:
                    df = pd.DataFrame(data)
                    df.rename(columns={"date": "timestamp"}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    results[symbol] = df
                    save_data_to_cache(df, symbol)
                    print(f"  ✅ {symbol}: {len(df)} candles ({days} days)")
                else:
                    print(f"  ⚠️ {symbol}: No data returned")
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Auth failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================
# MAIN BACKTEST
# ============================================================
def run_backtest():
    print("\n" + "=" * 70)
    print("📊 BACKTEST COMPARISON: Old Strategy vs New Strategy")
    print("=" * 70)
    print(f"\nOLD: AND-gate (MACD ∧ ST ∧ VWAP_below), lookback=2")
    print(f"NEW: Scoring (MACD:1 + ST:1/1.5 + VWAP_above:0.5 + PCR:0.5 ≥ 2.0), lookback=3")
    print(f"Timeframe: {TIMEFRAME_MINUTES}min | ATR: {ATR_PERIOD}/{ATR_MULTIPLIER}")
    
    # Initialize strategies
    old_strategy = StrategyRunner(
        name="OLD (AND-gate)",
        macd_lookback=2,
        use_scoring=False,
        vwap_flipped=False
    )
    
    new_strategy = StrategyRunner(
        name="NEW (Scoring)",
        macd_lookback=3,
        use_scoring=True,
        vwap_flipped=True,
        entry_threshold=2.0
    )
    
    # Load data: try cache first, then Kite (single login)
    print(f"\n📥 Loading real market data...")
    
    all_data = {}
    symbols_needed = []
    days = 60  # Kite max for intraday is 60 days
    
    for symbol in STOCKS:
        df = load_data_from_cache(symbol)
        if df is not None and len(df) >= ATR_PERIOD + 10:
            all_data[symbol] = df
        else:
            symbols_needed.append(symbol)
    
    # Fetch missing stocks from Kite (single authentication)
    if symbols_needed:
        print(f"\n🔐 Fetching {len(symbols_needed)} stocks from Kite (single login)...")
        kite_data = load_all_stocks_from_kite(symbols_needed, days=days)
        for symbol, df in kite_data.items():
            if len(df) >= ATR_PERIOD + 10:
                all_data[symbol] = df
            else:
                print(f"  ⚠️ {symbol}: Only {len(df)} candles, need {ATR_PERIOD + 10}+")
    
    if not all_data:
        print("\n❌ No data available. Ensure Kite credentials are in .env")
        print("   and run during market hours to cache data.")
        sys.exit(1)
    
    # Run both strategies on each stock
    print(f"\n🔄 Running backtest on {len(all_data)} stocks...")
    
    per_stock_results = {}
    for symbol, raw_df in all_data.items():
        # Prepare indicators
        df = calculate_supertrend(raw_df, ATR_PERIOD, ATR_MULTIPLIER)
        df = calculate_macd(df)
        df = calculate_vwap(df)
        
        print(f"\n  📊 {symbol}: {len(df)} candles")
        
        old_trades = old_strategy.run(df, symbol, STOCKS[symbol])
        new_trades = new_strategy.run(df, symbol, STOCKS[symbol])
        
        per_stock_results[symbol] = {
            "candles": len(df),
            "old_trades": len(old_trades),
            "new_trades": len(new_trades),
            "old_pnl": round(sum(t.pnl for t in old_trades), 2),
            "new_pnl": round(sum(t.pnl for t in new_trades), 2),
        }
        
        print(f"     OLD: {len(old_trades)} trades, PnL: ₹{per_stock_results[symbol]['old_pnl']:,.2f}")
        print(f"     NEW: {len(new_trades)} trades, PnL: ₹{per_stock_results[symbol]['new_pnl']:,.2f}")
    
    # Final comparison
    old_stats = old_strategy.get_stats()
    new_stats = new_strategy.get_stats()
    
    print(f"\n{'=' * 70}")
    print(f"📈 COMPARISON RESULTS")
    print(f"{'=' * 70}")
    
    metrics = [
        ("Total Trades", "total_trades", ""),
        ("Winners", "winners", ""),
        ("Losers", "losers", ""),
        ("Win Rate", "win_rate", "%"),
        ("Total P&L", "total_pnl", " ₹"),
        ("Avg P&L/Trade", "avg_pnl", " ₹"),
        ("Profit Factor", "profit_factor", ""),
        ("Max Drawdown", "max_drawdown", " ₹"),
        ("Avg Winner", "avg_winner", " ₹"),
        ("Avg Loser", "avg_loser", " ₹"),
        ("Blocked Signals", "blocked_count", ""),
        ("Avg Blocked MFE", "avg_blocked_mfe", " ₹"),
    ]
    
    print(f"\n{'Metric':<22} {'OLD (AND-gate)':<18} {'NEW (Scoring)':<18} {'Change':<15}")
    print(f"{'-'*22} {'-'*18} {'-'*18} {'-'*15}")
    
    for label, key, unit in metrics:
        old_val = old_stats.get(key, 0)
        new_val = new_stats.get(key, 0)
        
        if isinstance(old_val, float):
            old_str = f"{old_val:,.2f}{unit}"
            new_str = f"{new_val:,.2f}{unit}"
        else:
            old_str = f"{old_val}{unit}"
            new_str = f"{new_val}{unit}"
        
        # Calculate change
        if old_val != 0:
            change_pct = ((new_val - old_val) / abs(old_val)) * 100
            change_str = f"{change_pct:+.1f}%"
        elif new_val != 0:
            change_str = "NEW"
        else:
            change_str = "—"
        
        # Highlight improvements
        emoji = ""
        if key in ("total_trades", "win_rate", "total_pnl", "profit_factor", "avg_pnl") and new_val > old_val:
            emoji = " ✅"
        elif key in ("max_drawdown", "avg_loser") and abs(new_val) < abs(old_val):
            emoji = " ✅"
        elif key == "avg_blocked_mfe" and new_val > 0:
            emoji = " ⚠️"
        
        print(f"{label:<22} {old_str:<18} {new_str:<18} {change_str}{emoji}")
    
    # Per-stock breakdown
    print(f"\n{'=' * 70}")
    print(f"📊 PER-STOCK BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"\n{'Stock':<12} {'Old Trades':<12} {'New Trades':<12} {'Old PnL':<14} {'New PnL':<14}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
    
    for symbol, data in per_stock_results.items():
        print(f"{symbol:<12} {data['old_trades']:<12} {data['new_trades']:<12} "
              f"₹{data['old_pnl']:<13,.2f} ₹{data['new_pnl']:<13,.2f}")
    
    # ================================================================
    # DEEP BLOCKED SIGNAL ANALYSIS (NEW strategy)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"🔍 DEEP BLOCKED SIGNAL ANALYSIS (NEW Strategy)")
    print(f"{'=' * 70}")
    
    blocked = new_strategy.blocked_signals
    if blocked:
        # --- 1. WHY were signals blocked? ---
        print(f"\n  📋 BLOCK REASONS:")
        reason_counts = {}
        for b in blocked:
            r = b.get('blocked_reason', 'UNKNOWN')
            reason_counts[r] = reason_counts.get(r, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            pct = count / len(blocked) * 100
            print(f"    {reason}: {count} ({pct:.0f}%)")
        
        # --- 2. WHAT-IF: What if we took all blocked signals? ---
        print(f"\n  💰 WHAT-IF ANALYSIS (if all {len(blocked)} blocked signals were taken):")
        whatif_pnls = [b['whatif_pnl'] for b in blocked]
        whatif_winners = [b for b in blocked if b['whatif_pnl'] > 0]
        whatif_losers = [b for b in blocked if b['whatif_pnl'] <= 0]
        whatif_total = sum(whatif_pnls)
        
        print(f"    Total what-if P&L: ₹{whatif_total:.2f} (per share)")
        print(f"    Winners: {len(whatif_winners)} | Losers: {len(whatif_losers)}")
        if blocked:
            print(f"    Win rate: {len(whatif_winners)/len(blocked)*100:.0f}%")
        if whatif_winners:
            print(f"    Avg winner: ₹{np.mean([b['whatif_pnl'] for b in whatif_winners]):.2f}")
        if whatif_losers:
            print(f"    Avg loser: ₹{np.mean([b['whatif_pnl'] for b in whatif_losers]):.2f}")
        
        # --- 3. Breakdown by block reason ---
        print(f"\n  📊 WHAT-IF BY BLOCK REASON:")
        print(f"    {'Reason':<16} {'Count':<8} {'WinRate':<10} {'Avg P&L':<12} {'Total P&L':<12} {'Verdict'}")
        print(f"    {'-'*16} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        
        for reason in reason_counts:
            reason_signals = [b for b in blocked if b.get('blocked_reason') == reason]
            reason_pnls = [b['whatif_pnl'] for b in reason_signals]
            reason_winners = [p for p in reason_pnls if p > 0]
            reason_wr = len(reason_winners) / len(reason_signals) * 100 if reason_signals else 0
            reason_avg = np.mean(reason_pnls) if reason_pnls else 0
            reason_total = sum(reason_pnls)
            
            verdict = "🟢 LOWER threshold" if reason_wr > 55 and reason_avg > 0 else "🔴 KEEP blocking"
            print(f"    {reason:<16} {len(reason_signals):<8} {reason_wr:<10.0f}% ₹{reason_avg:<11.2f} ₹{reason_total:<11.2f} {verdict}")
        
        # --- 4. Per-stock blocked analysis ---
        print(f"\n  📈 BLOCKED PER STOCK:")
        print(f"    {'Stock':<12} {'Blocked':<10} {'WhatIf P&L':<14} {'Avg MFE':<12}")
        print(f"    {'-'*12} {'-'*10} {'-'*14} {'-'*12}")
        
        stock_blocked = {}
        for b in blocked:
            sym = b.get('symbol', '?')
            if sym not in stock_blocked:
                stock_blocked[sym] = []
            stock_blocked[sym].append(b)
        
        for sym, sigs in sorted(stock_blocked.items()):
            total_whatif = sum(b['whatif_pnl'] for b in sigs)
            avg_mfe = np.mean([b['mfe_5_candles'] for b in sigs])
            lot_size = STOCKS.get(sym, {}).get('lot_size', 1)
            print(f"    {sym:<12} {len(sigs):<10} ₹{total_whatif:<13.2f} ₹{avg_mfe:<11.2f}")
        
        # --- 5. Top missed opportunities ---
        print(f"\n  🎯 TOP 10 MISSED OPPORTUNITIES (highest what-if P&L):")
        sorted_blocked = sorted(blocked, key=lambda b: b['whatif_pnl'], reverse=True)
        for i, b in enumerate(sorted_blocked[:10]):
            print(f"    {i+1}. [{b.get('symbol','?')}] {b['direction']} @ ₹{b['price']:.0f} "
                  f"| Score: {b['score']:.1f} | What-if: ₹{b['whatif_pnl']:+.2f} "
                  f"| Exit: {b.get('whatif_exit','?')} | Reason: {b.get('blocked_reason','?')} "
                  f"| {b['time'][:16]}")
        
        # --- 6. Optimization recommendations ---
        print(f"\n{'=' * 70}")
        print(f"💡 OPTIMIZATION RECOMMENDATIONS")
        print(f"{'=' * 70}")
        
        st_missing = [b for b in blocked if b.get('blocked_reason') == 'ST_MISSING']
        vwap_missing = [b for b in blocked if b.get('blocked_reason') == 'VWAP_MISSING']
        
        if st_missing:
            st_whatif = sum(b['whatif_pnl'] for b in st_missing)
            st_wr = len([b for b in st_missing if b['whatif_pnl'] > 0]) / len(st_missing) * 100
            print(f"\n  1️⃣  ST_MISSING: {len(st_missing)} signals blocked because SuperTrend wasn't aligned")
            print(f"     What-if win rate: {st_wr:.0f}% | What-if total P&L: ₹{st_whatif:.2f}")
            if st_wr < 45:
                print(f"     → ✅ CORRECT to block. These have low win rate, ST confirmation adds value.")
            else:
                print(f"     → ⚠️ Consider lowering threshold to 1.5 for MACD+VWAP signals")
        
        if vwap_missing:
            vwap_whatif = sum(b['whatif_pnl'] for b in vwap_missing)
            vwap_wr = len([b for b in vwap_missing if b['whatif_pnl'] > 0]) / len(vwap_missing) * 100
            print(f"\n  2️⃣  VWAP_MISSING: {len(vwap_missing)} signals blocked because VWAP wasn't aligned")
            print(f"     These have MACD+ST confirmed (score 2.0) but VWAP on wrong side")
            print(f"     What-if win rate: {vwap_wr:.0f}% | What-if total P&L: ₹{vwap_whatif:.2f}")
            if vwap_wr > 50 and vwap_whatif > 0:
                print(f"     → ⚠️ VWAP may be hurting! These signals would have been profitable.")
                print(f"     → Consider: MACD+ST alone (score 2.0) is enough to enter.")
            else:
                print(f"     → ✅ VWAP filter is correctly blocking losing trades.")
    else:
        print(f"\n  No blocked signals to analyze.")
    
    # ================================================================
    # OLD strategy blocked analysis (brief)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"📐 OLD STRATEGY BLOCKED SIGNALS (brief)")
    print(f"{'=' * 70}")
    old_blocked = old_strategy.blocked_signals
    if old_blocked:
        old_whatif_total = sum(b['whatif_pnl'] for b in old_blocked)
        old_whatif_winners = len([b for b in old_blocked if b['whatif_pnl'] > 0])
        print(f"  Blocked: {len(old_blocked)} | What-if winners: {old_whatif_winners} ({old_whatif_winners/len(old_blocked)*100:.0f}%)")
        print(f"  What-if total P&L: ₹{old_whatif_total:.2f} (per share)")
        print(f"  → OLD strategy was blocking {old_whatif_winners} profitable trades")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_days": days,
        "stocks_tested": list(all_data.keys()),
        "old_strategy": old_stats,
        "new_strategy": new_stats,
        "per_stock": per_stock_results,
        "blocked_analysis": {
            "new_blocked": len(blocked),
            "reason_breakdown": reason_counts if blocked else {},
            "whatif_total_pnl": round(sum(b['whatif_pnl'] for b in blocked), 2) if blocked else 0,
            "whatif_winners": len([b for b in blocked if b['whatif_pnl'] > 0]) if blocked else 0,
        },
        "blocked_signals_detail": [{
            "symbol": b.get('symbol'), "direction": b['direction'], 
            "score": b['score'], "blocked_reason": b.get('blocked_reason'),
            "whatif_pnl": b['whatif_pnl'], "mfe": b['mfe_5_candles'],
            "time": b['time']
        } for b in blocked] if blocked else []
    }
    
    results_path = LOG_DIR / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"{'=' * 70}")
    
    # Final verdict
    print(f"\n🏆 VERDICT:")
    if new_stats['total_trades'] > old_stats['total_trades']:
        print(f"  ✅ NEW generates {new_stats['total_trades'] - old_stats['total_trades']} more trades")
    if new_stats['win_rate'] > old_stats['win_rate']:
        print(f"  ✅ NEW has higher win rate ({new_stats['win_rate']}% vs {old_stats['win_rate']}%)")
    if new_stats['profit_factor'] > old_stats['profit_factor']:
        print(f"  ✅ NEW has better profit factor ({new_stats['profit_factor']} vs {old_stats['profit_factor']})")
    
    # Lot-size-adjusted P&L
    print(f"\n  💰 LOT-SIZE-ADJUSTED P&L:")
    for symbol, data in per_stock_results.items():
        lot = STOCKS[symbol]['lot_size']
        print(f"    {symbol}: ₹{data['new_pnl']:.2f}/share × {lot} lots = ₹{data['new_pnl'] * lot:,.0f}")
    total_adj = sum(data['new_pnl'] * STOCKS[sym]['lot_size'] for sym, data in per_stock_results.items())
    print(f"    TOTAL: ₹{total_adj:,.0f}")
    print()


if __name__ == "__main__":
    run_backtest()
