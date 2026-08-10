#!/usr/bin/env python3
"""
Long-Period Backtest: 90/120/270/360 days with per-stock optimal config.
Each stock uses its own timeframe + MACD lookback.
No concurrent position limit — each stock tested independently.

Capital calculation: option premium × lot_size per trade.
"""
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import pytz; IST = pytz.timezone("Asia/Kolkata")
except: IST = None

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"; LOG_DIR.mkdir(exist_ok=True)

# Load .env
env_path = BASE_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ============================================================
# STOCK CONFIGS (each with optimal timeframe + lookback)
# ============================================================
STOCKS = {
    "SBIN":      {"lot_size": 750, "strike_gap": 5,    "interval": "30minute", "timeframe": 30, "lookback": 3},
    "AXISBANK":  {"lot_size": 625, "strike_gap": 25,   "interval": "30minute", "timeframe": 30, "lookback": 5},
    "RELIANCE":  {"lot_size": 250, "strike_gap": 20,   "interval": "15minute", "timeframe": 15, "lookback": 3},
    "ICICIBANK": {"lot_size": 700, "strike_gap": 12.5, "interval": "30minute", "timeframe": 30, "lookback": 3},
    "LT":        {"lot_size": 150, "strike_gap": 25,   "interval": "15minute", "timeframe": 15, "lookback": 5},
}

PERIODS = [90, 120, 270, 360]
ATR_PERIOD = 20; ATR_MULTIPLIER = 2.0
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# ============================================================
# INDICATORS
# ============================================================
def calc_supertrend(df, period=20, mult=2.0):
    df = df.copy()
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    hl2 = (df['high']+df['low'])/2
    df['basic_up'] = hl2 - mult*df['atr']; df['basic_dn'] = hl2 + mult*df['atr']
    df['up'] = df['basic_up']; df['dn'] = df['basic_dn']; df['trend'] = 1
    for i in range(1, len(df)):
        df.loc[df.index[i],'up'] = max(df['basic_up'].iloc[i], df['up'].iloc[i-1]) if df['close'].iloc[i-1] > df['up'].iloc[i-1] else df['basic_up'].iloc[i]
        df.loc[df.index[i],'dn'] = min(df['basic_dn'].iloc[i], df['dn'].iloc[i-1]) if df['close'].iloc[i-1] < df['dn'].iloc[i-1] else df['basic_dn'].iloc[i]
        pt = df['trend'].iloc[i-1]
        if pt==-1 and df['close'].iloc[i]>df['dn'].iloc[i-1]: df.loc[df.index[i],'trend']=1
        elif pt==1 and df['close'].iloc[i]<df['up'].iloc[i-1]: df.loc[df.index[i],'trend']=-1
        else: df.loc[df.index[i],'trend']=pt
    df['prev_trend'] = df['trend'].shift(1)
    return df

def calc_macd(df):
    df = df.copy()
    ef = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    es = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_line'] = ef-es; df['macd_signal'] = df['macd_line'].ewm(span=MACD_SIG, adjust=False).mean()
    df['macd_bullish'] = (df['macd_line']>df['macd_signal']) & (df['macd_line'].shift(1)<=df['macd_signal'].shift(1))
    df['macd_bearish'] = (df['macd_line']<df['macd_signal']) & (df['macd_line'].shift(1)>=df['macd_signal'].shift(1))
    return df

def calc_vwap(df):
    df = df.copy()
    if 'volume' not in df.columns or df['volume'].sum()==0:
        df['vwap'] = (df['high']+df['low']+df['close'])/3; return df
    df['date'] = df['timestamp'].dt.date
    tp = (df['high']+df['low']+df['close'])/3
    df['vwap'] = (tp*df['volume']).groupby(df['date']).cumsum() / df['volume'].groupby(df['date']).cumsum()
    return df

# ============================================================
# STRATEGY
# ============================================================
def run_strategy(df, sym, cfg):
    trades, lookback = [], cfg['lookback']
    pos = None; pb, ps = 0, 0
    lot = cfg['lot_size']; sg = cfg['strike_gap']
    
    for i in range(ATR_PERIOD+5, len(df)):
        r = df.iloc[i]; c = r['close']; v = r['vwap']; t = r['trend']; pt = r['prev_trend']
        # Exit
        if pos:
            if pos['dir']=="BUY" and t==-1 and pt==1:
                pos['exit']=c; pos['pnl']=c-pos['entry']; pos['reason']="ST_REV"; trades.append(pos); pos=None
            elif pos['dir']=="SELL" and t==1 and pt==-1:
                pos['exit']=c; pos['pnl']=pos['entry']-c; pos['reason']="ST_REV"; trades.append(pos); pos=None
            if pos:
                ts=r['timestamp']
                if hasattr(ts,'hour') and ts.hour>=15 and ts.minute>=15:
                    pos['exit']=c
                    pos['pnl']=(c-pos['entry']) if pos['dir']=="BUY" else (pos['entry']-c)
                    pos['reason']="EOD"; trades.append(pos); pos=None
            continue
        # MACD
        if r['macd_bullish']: pb=lookback+1; ps=0
        if r['macd_bearish']: ps=lookback+1; pb=0
        stb=t==1; sts=t==-1; stbf=t==1 and pt==-1; stsf=t==-1 and pt==1
        entered=False
        # BUY
        if pb>0 and not entered:
            sc=1.0
            if stb: sc+=1.5 if stbf else 1.0
            if c>v: sc+=0.5
            if sc>=2.0:
                # Estimate option premium for capital calc
                strike = round(c/sg)*sg
                itm = max(0, c-strike)
                premium = itm + c*0.004 + 15  # rough ATM premium
                capital = premium * lot
                pos={"dir":"BUY","entry":c,"time":str(r['timestamp']),"sym":sym,
                     "strike":strike,"premium":round(premium,2),"capital":round(capital,0)}
                pb=0; entered=True
        if ps>0 and not entered:
            sc=1.0
            if sts: sc+=1.5 if stsf else 1.0
            if c<v: sc+=0.5
            if sc>=2.0:
                strike = round(c/sg)*sg
                itm = max(0, strike-c)
                premium = itm + c*0.004 + 15
                capital = premium * lot
                pos={"dir":"SELL","entry":c,"time":str(r['timestamp']),"sym":sym,
                     "strike":strike,"premium":round(premium,2),"capital":round(capital,0)}
                ps=0; entered=True
        if pb>0: pb-=1
        if ps>0: ps-=1
    
    if pos:
        c=df.iloc[-1]['close']
        pos['exit']=c; pos['pnl']=(c-pos['entry']) if pos['dir']=="BUY" else (pos['entry']-c)
        pos['reason']="END"; trades.append(pos)
    return trades

# ============================================================
# DATA LOADING (chunked for 15min > 60 days)
# ============================================================
def fetch_data(symbols_by_interval, days):
    """Fetch data, chunking in 60-day windows for 15min."""
    results = {}
    
    try:
        from auto_login import KiteAutoLogin, load_credentials
        creds = load_credentials()
        auto = KiteAutoLogin(api_key=creds["api_key"], api_secret=creds["api_secret"],
                             user_id=creds["user_id"], password=creds["password"],
                             totp_secret=creds["totp_secret"], headless=True)
        token = auto.login()
        if not token: print("  ❌ Login failed"); return {}
        kite = auto.kite; print("  ✅ Authenticated")
        
        instruments = kite.instruments("NSE")
        token_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
        
        import time as _t
        to_date = datetime.now(IST) if IST else datetime.now()
        
        for interval, syms in symbols_by_interval.items():
            # Max days per chunk: 60 for <=15min, 400 for 30min+
            chunk_days = 60 if "15" in interval or "10" in interval or "5" in interval or interval=="minute" else 400
            
            for sym in syms:
                if sym not in token_map:
                    print(f"  ⚠️ {sym}: no token"); continue
                
                cache = LOG_DIR / f"lp_{sym}_{interval}_{days}d.csv"
                if cache.exists():
                    df = pd.read_csv(cache); df['timestamp'] = pd.to_datetime(df['timestamp'])
                    print(f"  📁 {sym} ({interval}, {days}d): {len(df)} candles (cached)")
                    results[sym] = df; continue
                
                all_data = []
                remaining = days
                end = to_date
                
                while remaining > 0:
                    chunk = min(remaining, chunk_days)
                    start = end - timedelta(days=chunk)
                    _t.sleep(0.4)
                    try:
                        data = kite.historical_data(instrument_token=token_map[sym],
                                                    from_date=start, to_date=end, interval=interval)
                        if data: all_data = data + all_data
                        print(f"  📥 {sym}: {len(data)} candles ({start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')})")
                    except Exception as e:
                        print(f"  ⚠️ {sym} chunk error: {e}")
                        break
                    end = start
                    remaining -= chunk
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    df.rename(columns={"date":"timestamp"}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                    df.to_csv(cache, index=False)
                    results[sym] = df
                    print(f"  ✅ {sym}: {len(df)} total candles ({days}d)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback; traceback.print_exc()
    
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*85)
    print("📊 LONG-PERIOD BACKTEST: 90 / 120 / 270 / 360 days")
    print("="*85)
    print("\n⚠️  No 3-trade concurrent limit — each stock is tested independently.")
    print("    This shows theoretical max performance per stock.\n")
    
    # Group symbols by interval for efficient fetching
    interval_groups = {}
    for sym, cfg in STOCKS.items():
        iv = cfg['interval']
        if iv not in interval_groups: interval_groups[iv] = []
        interval_groups[iv].append(sym)
    
    all_results = []
    
    for days in PERIODS:
        print(f"\n{'━'*85}")
        print(f"📅 PERIOD: {days} DAYS")
        print(f"{'━'*85}")
        
        data = fetch_data(interval_groups, days)
        
        for sym, cfg in STOCKS.items():
            if sym not in data or len(data[sym]) < ATR_PERIOD+10:
                print(f"  ⚠️ {sym}: insufficient data for {days}d")
                all_results.append({"days": days, "sym": sym, "trades": 0, "status": "NO_DATA"})
                continue
            
            df = calc_supertrend(data[sym], ATR_PERIOD, ATR_MULTIPLIER)
            df = calc_macd(df); df = calc_vwap(df)
            
            trades = run_strategy(df, sym, cfg)
            lot = cfg['lot_size']
            
            if not trades:
                all_results.append({"days": days, "sym": sym, "trades": 0, "status": "NO_TRADES"})
                continue
            
            pnls = [t['pnl'] for t in trades]
            winners = [t for t in trades if t['pnl']>0]
            losers = [t for t in trades if t['pnl']<=0]
            gp = sum(t['pnl'] for t in winners) if winners else 0
            gl = abs(sum(t['pnl'] for t in losers)) if losers else 1
            capitals = [t.get('capital',0) for t in trades]
            
            # Equity curve for max drawdown
            eq = 0; peak = 0; mdd = 0
            for p in pnls:
                eq += p * lot; peak = max(peak, eq); mdd = max(mdd, peak - eq)
            
            r = {
                "days": days, "sym": sym, "status": "OK",
                "config": f"{cfg['interval'].replace('minute','m')}/LB{cfg['lookback']}",
                "lot_size": lot,
                "candles": len(df),
                "trades": len(trades),
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": round(len(winners)/len(trades)*100, 1),
                "pnl_per_share": round(sum(pnls), 2),
                "pnl_lot": round(sum(pnls)*lot, 0),
                "profit_factor": round(gp/gl, 2) if gl>0 else 0,
                "avg_winner": round(gp/len(winners), 2) if winners else 0,
                "avg_loser": round(-gl/len(losers), 2) if losers else 0,
                "max_drawdown_lot": round(mdd, 0),
                "avg_capital_per_trade": round(np.mean(capitals), 0) if capitals else 0,
                "max_capital_per_trade": round(max(capitals), 0) if capitals else 0,
                "trades_per_day": round(len(trades) / max(1, days * 0.7), 2),  # ~70% are trading days
            }
            all_results.append(r)
    
    # ================================================================
    # PRINT RESULTS
    # ================================================================
    for days in PERIODS:
        period_results = [r for r in all_results if r['days']==days and r.get('status')=='OK']
        if not period_results: continue
        
        print(f"\n{'='*85}")
        print(f"📊 {days}-DAY RESULTS")
        print(f"{'='*85}")
        
        print(f"\n{'Stock':<12} {'Config':<10} {'Trades':<8} {'W/L':<8} {'WR%':<6} {'PF':<6} "
              f"{'P&L/sh':<10} {'Lot P&L':<12} {'MaxDD':<10} {'Avg Cap':<10}")
        print(f"{'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6} "
              f"{'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        
        total_lot_pnl = 0
        for r in period_results:
            e = "✅" if r['pnl_lot']>0 else "❌"
            print(f"{r['sym']:<12} {r['config']:<10} {r['trades']:<8} "
                  f"{r['winners']}/{r['losers']:<5} {r['win_rate']:<6} {r['profit_factor']:<6} "
                  f"₹{r['pnl_per_share']:<9} ₹{r['pnl_lot']:<11,.0f} ₹{r['max_drawdown_lot']:<9,.0f} "
                  f"₹{r['avg_capital_per_trade']:<9,.0f} {e}")
            total_lot_pnl += r['pnl_lot']
        
        print(f"\n  💰 TOTAL LOT-ADJUSTED P&L ({days}d): ₹{total_lot_pnl:,.0f}")
        print(f"  📊 Monthly avg: ₹{total_lot_pnl / (days/30):,.0f}/month")
        
        # Capital summary
        avg_caps = [r['avg_capital_per_trade'] for r in period_results if r['avg_capital_per_trade']>0]
        if avg_caps:
            print(f"  💵 Avg capital per trade: ₹{np.mean(avg_caps):,.0f}")
            print(f"  💵 Max capital single trade: ₹{max(r['max_capital_per_trade'] for r in period_results):,.0f}")
            print(f"  💵 Capital for 3 concurrent (max): ₹{3 * max(r['max_capital_per_trade'] for r in period_results):,.0f}")
    
    # ================================================================
    # CROSS-PERIOD COMPARISON
    # ================================================================
    print(f"\n{'='*85}")
    print(f"📈 CROSS-PERIOD COMPARISON (Monthly P&L)")
    print(f"{'='*85}")
    
    print(f"\n{'Stock':<12}", end="")
    for d in PERIODS: print(f" {d}d/month{'':>4}", end="")
    print()
    print(f"{'-'*12}", end="")
    for _ in PERIODS: print(f" {'-'*14}", end="")
    print()
    
    for sym in STOCKS:
        print(f"{sym:<12}", end="")
        for d in PERIODS:
            r = next((x for x in all_results if x['days']==d and x['sym']==sym and x.get('status')=='OK'), None)
            if r:
                monthly = r['pnl_lot'] / (d/30)
                e = "✅" if monthly>0 else "❌"
                print(f" ₹{monthly:>9,.0f} {e} ", end="")
            else:
                print(f" {'N/A':>12}  ", end="")
        print()
    
    # Totals
    print(f"{'TOTAL':<12}", end="")
    for d in PERIODS:
        total = sum(r['pnl_lot'] for r in all_results if r['days']==d and r.get('status')=='OK')
        monthly = total / (d/30)
        print(f" ₹{monthly:>9,.0f} 📊 ", end="")
    print()
    
    # Save
    save_path = LOG_DIR / f"longperiod_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Saved: {save_path}\n")


if __name__ == "__main__":
    main()
