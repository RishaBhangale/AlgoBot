#!/usr/bin/env python3
"""
Multi-Config Backtest: Find optimal lookback + timeframe + stock combination

Tests:
  - Lookback: 3 vs 5 candles
  - Timeframe: 15min vs 30min (Kite doesn't support 25min)
  - Stocks: 3 current (RELIANCE, HDFCBANK, ICICIBANK) + 2 candidates (SBIN, TATAMOTORS)

Usage: python3 backtest_multiconfig.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Load .env
env_path = BASE_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ============================================================
# ALL STOCKS TO TEST
# ============================================================
ALL_STOCKS = {
    # Fixed 3
    "RELIANCE":  {"lot_size": 250, "strike_gap": 20,   "group": "FIXED"},
    "ICICIBANK": {"lot_size": 700, "strike_gap": 12.5, "group": "FIXED"},
    "SBIN":      {"lot_size": 750, "strike_gap": 5,    "group": "FIXED"},
    # Candidates — testing all TATAMOTORS variants + 3 others
    "TATAMOTORS":{"lot_size": 575, "strike_gap": 10,  "group": "CANDIDATE"},
    "TATAMOTOR": {"lot_size": 575, "strike_gap": 10,  "group": "CANDIDATE"},
    "TATAMTRDVR":{"lot_size": 575, "strike_gap": 10,  "group": "CANDIDATE"},
    "BHARTIARTL":{"lot_size": 1851,"strike_gap": 20,  "group": "CANDIDATE"},
    "AXISBANK":  {"lot_size": 625, "strike_gap": 25,  "group": "CANDIDATE"},
    "LT":        {"lot_size": 150, "strike_gap": 25,  "group": "CANDIDATE"},
}

# Configs to test
CONFIGS = [
    {"name": "15min_LB3", "interval": "15minute", "lookback": 3, "label": "15min / 3 candles (current)"},
    {"name": "15min_LB5", "interval": "15minute", "lookback": 5, "label": "15min / 5 candles"},
    {"name": "30min_LB3", "interval": "30minute", "lookback": 3, "label": "30min / 3 candles"},
    {"name": "30min_LB5", "interval": "30minute", "lookback": 5, "label": "30min / 5 candles"},
]

ATR_PERIOD = 20
ATR_MULTIPLIER = 2.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL_P = 12, 26, 9
DAYS = 60


# ============================================================
# INDICATORS
# ============================================================
def calculate_supertrend(df, period=20, multiplier=2.0):
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    df['hl2'] = (df['high'] + df['low']) / 2
    df['basic_up'] = df['hl2'] - (multiplier * df['atr'])
    df['basic_dn'] = df['hl2'] + (multiplier * df['atr'])
    df['up'] = df['basic_up']; df['dn'] = df['basic_dn']
    df['trend'] = 1; df['supertrend'] = 0.0
    for i in range(1, len(df)):
        df.loc[df.index[i], 'up'] = max(df['basic_up'].iloc[i], df['up'].iloc[i-1]) if df['close'].iloc[i-1] > df['up'].iloc[i-1] else df['basic_up'].iloc[i]
        df.loc[df.index[i], 'dn'] = min(df['basic_dn'].iloc[i], df['dn'].iloc[i-1]) if df['close'].iloc[i-1] < df['dn'].iloc[i-1] else df['basic_dn'].iloc[i]
        pt = df['trend'].iloc[i-1]
        if pt == -1 and df['close'].iloc[i] > df['dn'].iloc[i-1]: df.loc[df.index[i], 'trend'] = 1
        elif pt == 1 and df['close'].iloc[i] < df['up'].iloc[i-1]: df.loc[df.index[i], 'trend'] = -1
        else: df.loc[df.index[i], 'trend'] = pt
        df.loc[df.index[i], 'supertrend'] = df['up'].iloc[i] if df['trend'].iloc[i] == 1 else df['dn'].iloc[i]
    df['prev_trend'] = df['trend'].shift(1)
    return df

def calculate_macd(df):
    df = df.copy()
    ef = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    es = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_line'] = ef - es
    df['macd_signal'] = df['macd_line'].ewm(span=MACD_SIGNAL_P, adjust=False).mean()
    df['macd_bullish'] = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_bearish'] = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    return df

def calculate_vwap(df):
    df = df.copy()
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        return df
    df['date'] = df['timestamp'].dt.date
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / df['volume'].groupby(df['date']).cumsum()
    return df


# ============================================================
# STRATEGY RUNNER
# ============================================================
def run_strategy(df, symbol, config, lookback, entry_threshold=2.0):
    """Run scoring strategy, return (trades_list, blocked_list)."""
    trades = []
    blocked = []
    position = None
    pending_bull = 0
    pending_bear = 0
    
    for i in range(ATR_PERIOD + 5, len(df)):
        row = df.iloc[i]
        close = row['close']
        vwap = row['vwap']
        trend = row['trend']
        prev_trend = row['prev_trend']
        
        # Exit
        if position:
            if position['dir'] == "BUY" and trend == -1 and prev_trend == 1:
                pnl = close - position['entry']
                trades.append({**position, "exit": close, "pnl": pnl, "reason": "ST_REV"})
                position = None
            elif position['dir'] == "SELL" and trend == 1 and prev_trend == -1:
                pnl = position['entry'] - close
                trades.append({**position, "exit": close, "pnl": pnl, "reason": "ST_REV"})
                position = None
            if position:
                ts = row['timestamp']
                if hasattr(ts, 'hour') and ts.hour >= 15 and ts.minute >= 15:
                    pnl = (close - position['entry']) if position['dir'] == "BUY" else (position['entry'] - close)
                    trades.append({**position, "exit": close, "pnl": pnl, "reason": "EOD"})
                    position = None
            continue
        
        # MACD tracking
        if row['macd_bullish']: pending_bull = lookback + 1; pending_bear = 0
        if row['macd_bearish']: pending_bear = lookback + 1; pending_bull = 0
        
        st_bull = trend == 1; st_bear = trend == -1
        st_bull_flip = trend == 1 and prev_trend == -1
        st_bear_flip = trend == -1 and prev_trend == 1
        entered = False
        
        # BUY
        if pending_bull > 0 and not entered:
            score = 1.0
            if st_bull: score += 1.5 if st_bull_flip else 1.0
            if close > vwap: score += 0.5
            if score >= entry_threshold:
                position = {"dir": "BUY", "entry": close, "time": str(row['timestamp']), "sym": symbol}
                pending_bull = 0; entered = True
            elif score >= 1.0:
                # Track what-if
                wif = _whatif(df, i, "BUY", close)
                blocked.append({"sym": symbol, "dir": "BUY", "score": score, "whatif": wif})
        
        # SELL
        if pending_bear > 0 and not entered:
            score = 1.0
            if st_bear: score += 1.5 if st_bear_flip else 1.0
            if close < vwap: score += 0.5
            if score >= entry_threshold:
                position = {"dir": "SELL", "entry": close, "time": str(row['timestamp']), "sym": symbol}
                pending_bear = 0; entered = True
            elif score >= 1.0:
                wif = _whatif(df, i, "SELL", close)
                blocked.append({"sym": symbol, "dir": "SELL", "score": score, "whatif": wif})
        
        if pending_bull > 0: pending_bull -= 1
        if pending_bear > 0: pending_bear -= 1
    
    if position:
        close = df.iloc[-1]['close']
        pnl = (close - position['entry']) if position['dir'] == "BUY" else (position['entry'] - close)
        trades.append({**position, "exit": close, "pnl": pnl, "reason": "END"})
    
    return trades, blocked


def _whatif(df, idx, direction, price):
    """Simulate what-if P&L using ST reversal exit."""
    for j in range(1, min(50, len(df) - idx)):
        r = df.iloc[idx + j]
        t, pt = r['trend'], r['prev_trend']
        if direction == "BUY" and t == -1 and pt == 1: return round(r['close'] - price, 2)
        if direction == "SELL" and t == 1 and pt == -1: return round(price - r['close'], 2)
        ts = r['timestamp']
        if hasattr(ts, 'hour') and ts.hour >= 15 and ts.minute >= 15:
            return round((r['close'] - price) if direction == "BUY" else (price - r['close']), 2)
    return 0.0


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(symbols, interval, days=60):
    """Load from cache or Kite. Returns dict of symbol->DataFrame."""
    cache_suffix = interval.replace("minute", "m")
    results = {}
    need_fetch = []
    
    for sym in symbols:
        cache = LOG_DIR / f"mc_{sym}_{cache_suffix}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            results[sym] = df
            print(f"  📁 {sym} ({interval}): {len(df)} candles (cached)")
        else:
            need_fetch.append(sym)
    
    if need_fetch:
        print(f"\n  🔐 Fetching {len(need_fetch)} stocks ({interval}) from Kite...")
        try:
            from auto_login import KiteAutoLogin, load_credentials
            creds = load_credentials()
            auto = KiteAutoLogin(api_key=creds["api_key"], api_secret=creds["api_secret"],
                                 user_id=creds["user_id"], password=creds["password"],
                                 totp_secret=creds["totp_secret"], headless=True)
            token = auto.login()
            if not token:
                print("  ❌ Login failed"); return results
            
            kite = auto.kite
            instruments = kite.instruments("NSE")
            token_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments if i['tradingsymbol'] in need_fetch}
            
            to_date = datetime.now(IST) if IST else datetime.now()
            from_date = to_date - timedelta(days=days)
            
            import time as _time
            for sym in need_fetch:
                if sym not in token_map:
                    print(f"  ⚠️ {sym}: no token"); continue
                _time.sleep(0.5)
                try:
                    data = kite.historical_data(instrument_token=token_map[sym],
                                                from_date=from_date, to_date=to_date, interval=interval)
                    if data:
                        df = pd.DataFrame(data)
                        df.rename(columns={"date": "timestamp"}, inplace=True)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        results[sym] = df
                        cache = LOG_DIR / f"mc_{sym}_{cache_suffix}.csv"
                        df.to_csv(cache, index=False)
                        print(f"  ✅ {sym} ({interval}): {len(df)} candles")
                    else:
                        print(f"  ⚠️ {sym}: no data")
                except Exception as e:
                    print(f"  ❌ {sym}: {e}")
        except Exception as e:
            print(f"  ❌ Kite error: {e}")
            import traceback; traceback.print_exc()
    
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("🔬 MULTI-CONFIG BACKTEST: Lookback × Timeframe × Stocks")
    print("=" * 80)
    
    symbols = list(ALL_STOCKS.keys())
    all_results = []
    
    for cfg in CONFIGS:
        interval = cfg["interval"]
        lookback = cfg["lookback"]
        config_name = cfg["name"]
        
        print(f"\n{'─' * 80}")
        print(f"📊 CONFIG: {cfg['label']}")
        print(f"{'─' * 80}")
        
        # Load data for this interval
        data = load_all_data(symbols, interval, DAYS)
        
        for sym in symbols:
            if sym not in data or len(data[sym]) < ATR_PERIOD + 10:
                print(f"  ⚠️ {sym}: insufficient data, skipping")
                continue
            
            df = calculate_supertrend(data[sym], ATR_PERIOD, ATR_MULTIPLIER)
            df = calculate_macd(df)
            df = calculate_vwap(df)
            
            trades, blocked = run_strategy(df, sym, ALL_STOCKS[sym], lookback)
            
            lot = ALL_STOCKS[sym]['lot_size']
            total_pnl = sum(t['pnl'] for t in trades)
            winners = [t for t in trades if t['pnl'] > 0]
            losers = [t for t in trades if t['pnl'] <= 0]
            wr = len(winners) / len(trades) * 100 if trades else 0
            gp = sum(t['pnl'] for t in winners) if winners else 0
            gl = abs(sum(t['pnl'] for t in losers)) if losers else 1
            pf = round(gp / gl, 2) if gl > 0 else 0
            
            all_results.append({
                "config": config_name,
                "config_label": cfg['label'],
                "symbol": sym,
                "group": ALL_STOCKS[sym]['group'],
                "lot_size": lot,
                "trades": len(trades),
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": round(wr, 1),
                "pnl_per_share": round(total_pnl, 2),
                "pnl_lot_adjusted": round(total_pnl * lot, 0),
                "profit_factor": pf,
                "avg_winner": round(gp / len(winners), 2) if winners else 0,
                "avg_loser": round(-gl / len(losers), 2) if losers else 0,
                "blocked": len(blocked),
                "blocked_whatif_pnl": round(sum(b['whatif'] for b in blocked), 2) if blocked else 0,
            })
    
    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"📈 RESULTS: ALL CONFIGS × ALL STOCKS")
    print(f"{'=' * 80}")
    
    print(f"\n{'Config':<18} {'Stock':<12} {'Group':<10} {'Trades':<8} {'WR%':<7} {'PF':<6} {'P&L/sh':<10} {'Lot-Adj P&L':<14}")
    print(f"{'-'*18} {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*14}")
    
    for r in all_results:
        emoji = "✅" if r['pnl_lot_adjusted'] > 0 else "❌"
        print(f"{r['config']:<18} {r['symbol']:<12} {r['group']:<10} {r['trades']:<8} "
              f"{r['win_rate']:<7} {r['profit_factor']:<6} ₹{r['pnl_per_share']:<9} "
              f"₹{r['pnl_lot_adjusted']:<13,.0f} {emoji}")
    
    # ================================================================
    # AGGREGATE BY CONFIG
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"📊 AGGREGATE BY CONFIG (all stocks combined)")
    print(f"{'=' * 80}")
    
    config_agg = {}
    for r in all_results:
        c = r['config_label']
        if c not in config_agg:
            config_agg[c] = {"trades": 0, "winners": 0, "losers": 0, "total_lot_pnl": 0, "gp": 0, "gl": 0}
        config_agg[c]["trades"] += r['trades']
        config_agg[c]["winners"] += r['winners']
        config_agg[c]["losers"] += r['losers']
        config_agg[c]["total_lot_pnl"] += r['pnl_lot_adjusted']
        config_agg[c]["gp"] += r['avg_winner'] * r['winners'] if r['winners'] else 0
        config_agg[c]["gl"] += abs(r['avg_loser']) * r['losers'] if r['losers'] else 0
    
    print(f"\n{'Config':<32} {'Trades':<8} {'WR%':<8} {'PF':<7} {'Total Lot-Adj P&L':<18}")
    print(f"{'-'*32} {'-'*8} {'-'*8} {'-'*7} {'-'*18}")
    
    best_config = None
    best_pnl = -999999
    for label, agg in config_agg.items():
        wr = agg['winners'] / agg['trades'] * 100 if agg['trades'] else 0
        pf = round(agg['gp'] / agg['gl'], 2) if agg['gl'] > 0 else 0
        emoji = "🏆" if agg['total_lot_pnl'] == max(a['total_lot_pnl'] for a in config_agg.values()) else "  "
        print(f"{label:<32} {agg['trades']:<8} {wr:<8.1f} {pf:<7} ₹{agg['total_lot_pnl']:<17,.0f} {emoji}")
        if agg['total_lot_pnl'] > best_pnl:
            best_pnl = agg['total_lot_pnl']
            best_config = label
    
    # ================================================================
    # CANDIDATE STOCKS ANALYSIS
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"🆕 CANDIDATE STOCK ANALYSIS (pick best 2 to add)")
    print(f"{'=' * 80}")
    
    candidate_syms = [s for s, c in ALL_STOCKS.items() if c['group'] == 'CANDIDATE']
    candidate_rankings = []
    
    for sym in candidate_syms:
        sym_results = [r for r in all_results if r['symbol'] == sym]
        if not sym_results:
            print(f"\n  {sym}: No data available on NSE")
            continue
        
        print(f"\n  📊 {sym} (lot size: {ALL_STOCKS[sym]['lot_size']}):")
        best = max(sym_results, key=lambda r: r['pnl_lot_adjusted'])
        worst = min(sym_results, key=lambda r: r['pnl_lot_adjusted'])
        
        for r in sym_results:
            marker = " 🏆" if r == best else " ❌" if r == worst else ""
            print(f"    {r['config']:<18} {r['trades']} trades | WR: {r['win_rate']}% | "
                  f"PF: {r['profit_factor']} | Lot P&L: ₹{r['pnl_lot_adjusted']:,.0f}{marker}")
        
        profitable_configs = [r for r in sym_results if r['pnl_lot_adjusted'] > 0 and r['profit_factor'] >= 1.0]
        best_pnl_val = best['pnl_lot_adjusted']
        avg_pnl = np.mean([r['pnl_lot_adjusted'] for r in sym_results])
        
        if len(profitable_configs) >= 3:
            verdict = f"✅ STRONG ADD"
        elif len(profitable_configs) >= 2:
            verdict = f"⚠️ CONDITIONAL ADD"
        else:
            verdict = f"❌ SKIP"
        print(f"    → {verdict} | Profitable in {len(profitable_configs)}/4 configs | Best: ₹{best_pnl_val:,.0f} | Avg: ₹{avg_pnl:,.0f}")
        
        candidate_rankings.append({"sym": sym, "profitable_configs": len(profitable_configs),
                                    "best_pnl": best_pnl_val, "avg_pnl": avg_pnl, "verdict": verdict})
    
    # Rank candidates
    if candidate_rankings:
        candidate_rankings.sort(key=lambda x: (-x['profitable_configs'], -x['avg_pnl']))
        print(f"\n  🏅 CANDIDATE RANKING (best to worst):")
        for i, c in enumerate(candidate_rankings):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "  "
            print(f"    {emoji} {i+1}. {c['sym']:<14} {c['verdict']:<20} Avg P&L: ₹{c['avg_pnl']:>10,.0f} | Best: ₹{c['best_pnl']:>10,.0f}")
    
    # ================================================================
    # CURRENT STOCKS CHECK
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"📋 FIXED STOCKS: Performance across configs")
    print(f"{'=' * 80}")
    
    fixed_syms = [s for s, c in ALL_STOCKS.items() if c['group'] == 'FIXED']
    for sym in fixed_syms:
        sym_results = [r for r in all_results if r['symbol'] == sym]
        if not sym_results:
            continue
        best = max(sym_results, key=lambda r: r['pnl_lot_adjusted'])
        print(f"\n  {sym}: Best config = {best['config']} (₹{best['pnl_lot_adjusted']:,.0f})")
        for r in sym_results:
            marker = " 🏆" if r == best else ""
            print(f"    {r['config']:<18} {r['trades']} trades | WR: {r['win_rate']}% | "
                  f"PF: {r['profit_factor']} | ₹{r['pnl_lot_adjusted']:,.0f}{marker}")
    
    # ================================================================
    # FINAL RECOMMENDATION
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"🏆 RECOMMENDATION")
    print(f"{'=' * 80}")
    print(f"\n  Best overall config: {best_config}")
    print(f"  Total lot-adjusted P&L: ₹{best_pnl:,.0f}")
    
    # Save
    save_path = LOG_DIR / f"multiconfig_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(save_path, "w") as f:
        json.dump({"results": all_results, "best_config": best_config, 
                   "best_pnl": best_pnl, "timestamp": datetime.now().isoformat()}, f, indent=2, default=str)
    print(f"  Results saved: {save_path}\n")


if __name__ == "__main__":
    main()
