#!/usr/bin/env python3
"""
Supertrend Strategy Backtest - 3 Minute Timeframe
- ATR Period: 10, Multiplier: 3.0
- Timeframe: 3 minutes
- Target: +20%, SL: -10%

This is a separate analysis file - does not affect cloud deployment.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Strategy Parameters
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
TARGET_PCT = 20
SL_PCT = 10
TIMEFRAME_MINUTES = 3  # Changed to 3 minutes
CAPITAL = 100000

SECURITIES = {
    "NIFTY": {
        "instrument_token": 256265,
        "lot_size": 50,
        "strike_interval": 50,
    },
    "BANKNIFTY": {
        "instrument_token": 260105,
        "lot_size": 25,
        "strike_interval": 100,
    }
}

BASE_DIR = Path(__file__).parent


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['trend'] = 1
    df['supertrend'] = df['lower_band']
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = 1
        elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = -1
        else:
            df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
        
        if df['trend'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
    
    df['prev_trend'] = df['trend'].shift(1)
    df['signal'] = None
    df.loc[(df['trend'] == 1) & (df['prev_trend'] == -1), 'signal'] = 'BUY'
    df.loc[(df['trend'] == -1) & (df['prev_trend'] == 1), 'signal'] = 'SELL'
    
    return df


def estimate_option_premium(spot: float, strike: float, option_type: str) -> float:
    itm = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
    time_value = spot * 0.003 + 20
    return itm + time_value


def backtest_security(df: pd.DataFrame, symbol: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Backtesting {symbol} (3-min timeframe)")
    print(f"{'='*60}")
    print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    df = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
    
    trades = []
    position = None
    initial_position_taken = False
    
    for i in range(ATR_PERIOD + 2, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        spot = row['close']
        timestamp = df.index[i]
        current_trend = row['trend']
        prev_trend = prev_row['trend']
        
        if position:
            move = spot - position['spot_at_entry']
            delta = 0.5 if position['type'] == 'CE' else -0.5
            current_opt_price = position['entry_price'] + (move * delta)
            exit_reason = None
            
            if current_opt_price >= position['target']:
                exit_reason = 'TARGET'
                exit_price = position['target']
            elif current_opt_price <= position['sl']:
                exit_reason = 'SL'
                exit_price = position['sl']
            elif current_trend != prev_trend:
                exit_reason = 'SIGNAL_CHANGE'
                exit_price = max(0.05, current_opt_price)
            
            if exit_reason:
                pnl = (exit_price - position['entry_price']) * config['lot_size']
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'type': position['type'],
                    'strike': position['strike'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                })
                position = None
        
        if position is None:
            enter = False
            if not initial_position_taken and current_trend != 0:
                enter = True
                initial_position_taken = True
            elif current_trend != prev_trend and prev_trend != 0:
                enter = True
            
            if enter:
                option_type = 'CE' if current_trend == 1 else 'PE'
                strike = round(spot / config['strike_interval']) * config['strike_interval']
                entry_price = estimate_option_premium(spot, strike, option_type)
                target = entry_price * (1 + TARGET_PCT / 100)
                sl = entry_price * (1 - SL_PCT / 100)
                
                position = {
                    'entry_time': timestamp,
                    'type': option_type,
                    'strike': strike,
                    'entry_price': entry_price,
                    'target': target,
                    'sl': sl,
                    'spot_at_entry': spot
                }
    
    if position:
        spot = df.iloc[-1]['close']
        move = spot - position['spot_at_entry']
        delta = 0.5 if position['type'] == 'CE' else -0.5
        exit_price = max(0.05, position['entry_price'] + (move * delta))
        pnl = (exit_price - position['entry_price']) * config['lot_size']
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'type': position['type'],
            'exit_reason': 'END_OF_DATA',
            'pnl': pnl,
        })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        winners = len(trades_df[trades_df['pnl'] > 0])
        losers = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winners / len(trades_df) * 100
        target_hits = len(trades_df[trades_df['exit_reason'] == 'TARGET'])
        sl_hits = len(trades_df[trades_df['exit_reason'] == 'SL'])
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losers > 0 else 0
        
        stats = {
            'symbol': symbol,
            'timeframe': '3min',
            'total_trades': len(trades_df),
            'winners': winners,
            'losers': losers,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'target_hits': target_hits,
            'sl_hits': sl_hits,
            'return_pct': round((total_pnl / CAPITAL) * 100, 2)
        }
    else:
        stats = {'symbol': symbol, 'total_trades': 0, 'total_pnl': 0}
    
    print(f"\n📊 {symbol} Results (3-min):")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats.get('win_rate', 0)}%")
    print(f"   Winners: {stats.get('winners', 0)} | Losers: {stats.get('losers', 0)}")
    print(f"   Target Hits: {stats.get('target_hits', 0)} | SL Hits: {stats.get('sl_hits', 0)}")
    print(f"   Avg Win: ₹{stats.get('avg_win', 0):,.2f} | Avg Loss: ₹{stats.get('avg_loss', 0):,.2f}")
    print(f"   Total P&L: ₹{stats['total_pnl']:,.2f}")
    print(f"   Return: {stats.get('return_pct', 0)}% on ₹{CAPITAL:,}")
    
    return {'trades': trades, 'stats': stats}


def load_sample_data(symbol: str) -> pd.DataFrame:
    """Generate 3-minute sample data."""
    print(f"Generating 3-year sample data for {symbol} (3-min candles)...")
    
    np.random.seed(42 if symbol == "NIFTY" else 123)
    base_price = 18000 if symbol == "NIFTY" else 40000
    
    trading_days = 252 * 3
    candles_per_day = 125  # 6.25 hours * 20 three-min candles per hour
    
    all_timestamps = []
    current_date = datetime(2023, 1, 2, 9, 15)
    
    for day in range(trading_days):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        for candle in range(candles_per_day):
            timestamp = current_date.replace(hour=9, minute=15) + timedelta(minutes=candle * 3)
            if timestamp.hour < 16:
                all_timestamps.append(timestamp)
        
        current_date += timedelta(days=1)
    
    prices = [base_price]
    trend = 1
    trend_duration = 0
    trend_length = np.random.randint(80, 300)
    
    for i in range(1, len(all_timestamps)):
        trend_duration += 1
        if trend_duration > trend_length:
            trend *= -1
            trend_duration = 0
            trend_length = np.random.randint(80, 300)
        
        hour = all_timestamps[i].hour
        volatility = 0.0015 if hour in [9, 15] else 0.0006
        drift = trend * 0.00008
        change = np.random.normal(drift, volatility)
        
        if np.random.random() < 0.003:
            change = np.random.choice([-1, 1]) * np.random.uniform(0.004, 0.012)
        
        prices.append(prices[-1] * (1 + change))
    
    df_data = []
    for i, (timestamp, close) in enumerate(zip(all_timestamps, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.0008)))
        low = close * (1 - abs(np.random.normal(0, 0.0008)))
        open_price = prices[i-1] if i > 0 else close
        
        df_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close
        })
    
    df = pd.DataFrame(df_data)
    df = df.set_index('timestamp')
    print(f"   Generated {len(df)} candles")
    
    return df


def main():
    print("\n" + "="*60)
    print("🎯 SUPERTREND BACKTEST - 3 MINUTE TIMEFRAME")
    print("="*60)
    print(f"Strategy: Supertrend (ATR:{ATR_PERIOD}, Mult:{ATR_MULTIPLIER})")
    print(f"Timeframe: {TIMEFRAME_MINUTES} minutes")
    print(f"Target: +{TARGET_PCT}% | SL: -{SL_PCT}%")
    print(f"Capital: ₹{CAPITAL:,}")
    print("="*60)
    
    results = {}
    
    for symbol, config in SECURITIES.items():
        df = load_sample_data(symbol)
        if len(df) > 0:
            result = backtest_security(df, symbol, config)
            results[symbol] = result
    
    print("\n" + "="*60)
    print("📈 OVERALL SUMMARY (3-MIN TIMEFRAME)")
    print("="*60)
    
    total_pnl = sum(r['stats']['total_pnl'] for r in results.values())
    total_trades = sum(r['stats']['total_trades'] for r in results.values())
    total_wins = sum(r['stats'].get('winners', 0) for r in results.values())
    
    print(f"Total Trades: {total_trades}")
    print(f"Total Winners: {total_wins}")
    print(f"Overall Win Rate: {total_wins/total_trades*100:.2f}%" if total_trades > 0 else "N/A")
    print(f"Total P&L: ₹{total_pnl:,.2f}")
    print(f"Return on ₹{CAPITAL:,}: {total_pnl/CAPITAL*100:.2f}%")
    print("="*60)
    
    # Generate chart
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Supertrend Strategy - 3 Minute Timeframe (3 Years)', fontsize=16, fontweight='bold')
        
        for idx, (symbol, config) in enumerate(SECURITIES.items()):
            if symbol not in results:
                continue
            
            df = load_sample_data(symbol)
            trades = results[symbol]['trades']
            
            strategy_equity = [CAPITAL]
            equity_dates = [df.index[0]]
            running_pnl = 0
            
            for trade in trades:
                running_pnl += trade['pnl']
                strategy_equity.append(CAPITAL + running_pnl)
                equity_dates.append(pd.to_datetime(trade['exit_time']))
            
            initial_price = df.iloc[0]['close']
            daily_df = df.resample('D').last().dropna()
            daily_buy_hold = (daily_df['close'] / initial_price) * CAPITAL
            
            ax1 = axes[idx, 0]
            ax1.plot(daily_df.index, daily_buy_hold.values, label=f'{symbol} Buy & Hold', color='gray', alpha=0.7)
            ax1.step(equity_dates, strategy_equity, label='Supertrend (3-min)', 
                    color='green' if strategy_equity[-1] > CAPITAL else 'red', linewidth=2, where='post')
            ax1.axhline(y=CAPITAL, color='black', linestyle='--', alpha=0.3)
            ax1.set_title(f'{symbol} - Capital Growth (3-min)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[idx, 1]
            if trades:
                pnls = [t['pnl'] for t in trades]
                colors = ['green' if p > 0 else 'red' for p in pnls]
                ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                ax2.axhline(y=0, color='black', linewidth=1)
                ax2.set_title(f'{symbol} - Trade P&L (3-min)', fontsize=12)
                stats = results[symbol]['stats']
                ax2.text(0.02, 0.98, f"Trades: {stats['total_trades']}\nWin Rate: {stats['win_rate']}%\nP&L: ₹{stats['total_pnl']:,.0f}",
                        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_file = BASE_DIR / f"backtest_3min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart saved to: {chart_file}")
        plt.show()
        
    except ImportError:
        print("\n⚠️ matplotlib not installed")
    
    return results


if __name__ == "__main__":
    main()
