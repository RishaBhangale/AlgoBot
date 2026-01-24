#!/usr/bin/env python3
"""
Supertrend Strategy Backtest
- ATR Period: 10, Multiplier: 3.0
- Timeframe: 5 minutes
- Target: +20%, SL: -10%
- Entry: On current trend (immediate)
- Exit: Target/SL/Signal change
- Hold: Till exit (no daily square-off)

Usage:
    python3 backtest.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Try to import kiteconnect for real data
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

# Strategy Parameters
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
TARGET_PCT = 20
SL_PCT = 10
TIMEFRAME = "5minute"
CAPITAL = 100000  # 1 Lakh

# Securities
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
    """Calculate Supertrend indicator."""
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
    
    # Basic bands
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    
    # Supertrend
    df['trend'] = 1  # 1 = bullish, -1 = bearish
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
    
    # Signal on trend change
    df['prev_trend'] = df['trend'].shift(1)
    df['signal'] = None
    df.loc[(df['trend'] == 1) & (df['prev_trend'] == -1), 'signal'] = 'BUY'
    df.loc[(df['trend'] == -1) & (df['prev_trend'] == 1), 'signal'] = 'SELL'
    
    return df


def estimate_option_premium(spot: float, strike: float, option_type: str) -> float:
    """Estimate option premium (simplified)."""
    itm = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
    time_value = spot * 0.003 + 20
    return itm + time_value


def backtest_security(df: pd.DataFrame, symbol: str, config: dict) -> dict:
    """
    Backtest Supertrend strategy on a single security.
    
    Returns dict with trades, stats, and summary.
    """
    print(f"\n{'='*60}")
    print(f"Backtesting {symbol}")
    print(f"{'='*60}")
    print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Calculate Supertrend
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
        
        # Check for exit on existing position
        if position:
            # Estimate current option price
            move = spot - position['spot_at_entry']
            delta = 0.5 if position['type'] == 'CE' else -0.5
            current_opt_price = position['entry_price'] + (move * delta)
            
            exit_reason = None
            
            # Check target
            if current_opt_price >= position['target']:
                exit_reason = 'TARGET'
                exit_price = position['target']
            # Check SL
            elif current_opt_price <= position['sl']:
                exit_reason = 'SL'
                exit_price = position['sl']
            # Check signal change
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
                    'target': position['target'],
                    'sl': position['sl'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'lot_size': config['lot_size']
                })
                position = None
        
        # Entry logic
        if position is None:
            enter = False
            
            # Initial position on first trend reading
            if not initial_position_taken and current_trend != 0:
                enter = True
                initial_position_taken = True
            # Signal change entry
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
    
    # Close any remaining position at last price
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
            'strike': position['strike'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'target': position['target'],
            'sl': position['sl'],
            'exit_reason': 'END_OF_DATA',
            'pnl': pnl,
            'lot_size': config['lot_size']
        })
    
    # Calculate stats
    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        winners = len(trades_df[trades_df['pnl'] > 0])
        losers = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winners / len(trades_df) * 100
        
        target_hits = len(trades_df[trades_df['exit_reason'] == 'TARGET'])
        sl_hits = len(trades_df[trades_df['exit_reason'] == 'SL'])
        signal_changes = len(trades_df[trades_df['exit_reason'] == 'SIGNAL_CHANGE'])
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losers > 0 else 0
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        stats = {
            'symbol': symbol,
            'total_trades': len(trades_df),
            'winners': winners,
            'losers': losers,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'target_hits': target_hits,
            'sl_hits': sl_hits,
            'signal_changes': signal_changes,
            'return_pct': round((total_pnl / CAPITAL) * 100, 2)
        }
    else:
        stats = {
            'symbol': symbol,
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'return_pct': 0
        }
    
    # Print summary
    print(f"\n📊 {symbol} Results:")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']}%")
    print(f"   Winners: {stats['winners']} | Losers: {stats['losers']}")
    print(f"   Target Hits: {stats.get('target_hits', 0)} | SL Hits: {stats.get('sl_hits', 0)} | Signal Changes: {stats.get('signal_changes', 0)}")
    print(f"   Avg Win: ₹{stats.get('avg_win', 0):,.2f} | Avg Loss: ₹{stats.get('avg_loss', 0):,.2f}")
    print(f"   Max Win: ₹{stats.get('max_win', 0):,.2f} | Max Loss: ₹{stats.get('max_loss', 0):,.2f}")
    print(f"   Total P&L: ₹{stats['total_pnl']:,.2f}")
    print(f"   Return: {stats['return_pct']}% on ₹{CAPITAL:,} capital")
    
    return {'trades': trades, 'stats': stats}


def fetch_historical_data(kite, symbol: str, config: dict, years: int = 3) -> pd.DataFrame:
    """Fetch historical data from Kite Connect."""
    all_data = []
    to_date = datetime.now()
    
    # Kite allows max 60 days of 5-min data per request
    days_per_request = 60
    total_days = years * 365
    
    print(f"Fetching {years} years of data for {symbol}...")
    
    while total_days > 0:
        from_date = to_date - timedelta(days=min(days_per_request, total_days))
        
        try:
            data = kite.historical_data(
                instrument_token=config['instrument_token'],
                from_date=from_date,
                to_date=to_date,
                interval=TIMEFRAME
            )
            all_data.extend(data)
            print(f"   Fetched {len(data)} candles ({from_date.date()} to {to_date.date()})")
        except Exception as e:
            print(f"   Error: {e}")
        
        to_date = from_date
        total_days -= days_per_request
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df


def load_sample_data(symbol: str) -> pd.DataFrame:
    """Generate realistic sample data with trends and volatility."""
    print(f"Generating 3-year sample data for {symbol}...")
    
    np.random.seed(42 if symbol == "NIFTY" else 123)
    
    base_price = 18000 if symbol == "NIFTY" else 40000
    
    # Generate 3 years of data (252 trading days x 3 years x ~75 candles per day)
    trading_days = 252 * 3
    candles_per_day = 75  # 9:15 to 3:30 = 6.25 hours * 12 five-min candles
    total_candles = trading_days * candles_per_day
    
    # Create trading timestamps
    all_timestamps = []
    current_date = datetime(2023, 1, 2, 9, 15)
    
    for day in range(trading_days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # Generate candles for this day
        for candle in range(candles_per_day):
            timestamp = current_date.replace(hour=9, minute=15) + timedelta(minutes=candle * 5)
            if timestamp.hour < 16:  # Before 4 PM
                all_timestamps.append(timestamp)
        
        current_date += timedelta(days=1)
    
    # Generate price with realistic volatility and trends
    prices = [base_price]
    trend = 1  # 1 = up, -1 = down
    trend_duration = 0
    trend_length = np.random.randint(50, 200)  # Trend lasts 50-200 candles
    
    for i in range(1, len(all_timestamps)):
        # Check if we should change trend
        trend_duration += 1
        if trend_duration > trend_length:
            trend *= -1  # Reverse trend
            trend_duration = 0
            trend_length = np.random.randint(50, 200)
        
        # Daily volatility (higher during market hours)
        hour = all_timestamps[i].hour
        if hour == 9 or hour == 15:  # Opening and closing hours more volatile
            volatility = 0.002
        else:
            volatility = 0.0008
        
        # Price movement with trend bias
        drift = trend * 0.0001  # Small trend bias
        change = np.random.normal(drift, volatility)
        
        # Occasional larger moves (news events)
        if np.random.random() < 0.005:  # 0.5% chance of big move
            change = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.015)
        
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLC data
    df_data = []
    for i, (timestamp, close) in enumerate(zip(all_timestamps, prices)):
        intraday_volatility = 0.001
        high = close * (1 + abs(np.random.normal(0, intraday_volatility)))
        low = close * (1 - abs(np.random.normal(0, intraday_volatility)))
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
    
    print(f"   Generated {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    return df


def main():
    print("\n" + "="*60)
    print("🎯 SUPERTREND BACKTEST")
    print("="*60)
    print(f"Strategy: Supertrend (ATR:{ATR_PERIOD}, Mult:{ATR_MULTIPLIER})")
    print(f"Timeframe: 5 minutes")
    print(f"Target: +{TARGET_PCT}% | SL: -{SL_PCT}%")
    print(f"Capital: ₹{CAPITAL:,}")
    print("="*60)
    
    results = {}
    
    # Try to use Kite for real data
    kite = None
    if KITE_AVAILABLE:
        try:
            api_file = BASE_DIR / "api_key.txt"
            token_file = BASE_DIR / "access_token.json"
            
            if api_file.exists() and token_file.exists():
                lines = api_file.read_text().strip().split("\n")
                api_key = lines[0].strip()
                
                with open(token_file) as f:
                    token_data = json.load(f)
                
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(token_data['access_token'])
                print("\n✅ Using real Kite Connect data")
        except Exception as e:
            print(f"\n⚠️ Could not connect to Kite: {e}")
            kite = None
    
    for symbol, config in SECURITIES.items():
        if kite:
            df = fetch_historical_data(kite, symbol, config, years=3)
        else:
            df = load_sample_data(symbol)
        
        if len(df) > 0:
            result = backtest_security(df, symbol, config)
            results[symbol] = result
    
    # Overall summary
    print("\n" + "="*60)
    print("📈 OVERALL SUMMARY")
    print("="*60)
    
    total_pnl = sum(r['stats']['total_pnl'] for r in results.values())
    total_trades = sum(r['stats']['total_trades'] for r in results.values())
    total_wins = sum(r['stats']['winners'] for r in results.values())
    
    print(f"Total Trades: {total_trades}")
    print(f"Total Winners: {total_wins}")
    print(f"Overall Win Rate: {total_wins/total_trades*100:.2f}%" if total_trades > 0 else "N/A")
    print(f"Total P&L: ₹{total_pnl:,.2f}")
    print(f"Return on ₹{CAPITAL:,}: {total_pnl/CAPITAL*100:.2f}%")
    print("="*60)
    
    # Generate comparison chart
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Supertrend Strategy vs Buy & Hold (3 Years)', fontsize=16, fontweight='bold')
        
        for idx, (symbol, config) in enumerate(SECURITIES.items()):
            if symbol not in results:
                continue
            
            # Get the data
            if kite:
                df = fetch_historical_data(kite, symbol, config, years=3)
            else:
                df = load_sample_data(symbol)
            
            if len(df) == 0:
                continue
            
            trades = results[symbol]['trades']
            
            # Calculate equity curves
            # Strategy equity curve
            strategy_equity = [CAPITAL]
            equity_dates = [df.index[0]]
            running_pnl = 0
            
            for trade in trades:
                running_pnl += trade['pnl']
                strategy_equity.append(CAPITAL + running_pnl)
                equity_dates.append(pd.to_datetime(trade['exit_time']))
            
            # Buy & Hold equity curve (simplified - just track index value)
            initial_price = df.iloc[0]['close']
            buy_hold_equity = (df['close'] / initial_price) * CAPITAL
            
            # Daily resampling for smoother chart
            daily_df = df.resample('D').last().dropna()
            daily_buy_hold = (daily_df['close'] / initial_price) * CAPITAL
            
            # Plot 1: Equity comparison
            ax1 = axes[idx, 0]
            ax1.plot(daily_df.index, daily_buy_hold.values, label=f'{symbol} Buy & Hold', 
                    color='gray', alpha=0.7, linewidth=1)
            ax1.step(equity_dates, strategy_equity, label='Supertrend Strategy', 
                    color='green' if strategy_equity[-1] > CAPITAL else 'red', linewidth=2, where='post')
            ax1.axhline(y=CAPITAL, color='black', linestyle='--', alpha=0.3, label='Initial Capital')
            ax1.set_title(f'{symbol} - Capital Growth', fontsize=12)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value (₹)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Final values
            strategy_final = strategy_equity[-1]
            buyhold_final = daily_buy_hold.iloc[-1] if len(daily_buy_hold) > 0 else CAPITAL
            
            ax1.annotate(f'Strategy: ₹{strategy_final:,.0f}\n({(strategy_final/CAPITAL-1)*100:.1f}%)', 
                        xy=(equity_dates[-1], strategy_final), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if strategy_final > CAPITAL else 'lightcoral'))
            ax1.annotate(f'B&H: ₹{buyhold_final:,.0f}\n({(buyhold_final/CAPITAL-1)*100:.1f}%)', 
                        xy=(daily_df.index[-1], buyhold_final), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgray'))
            
            # Plot 2: Trade distribution
            ax2 = axes[idx, 1]
            if trades:
                pnls = [t['pnl'] for t in trades]
                colors = ['green' if p > 0 else 'red' for p in pnls]
                ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                ax2.axhline(y=0, color='black', linewidth=1)
                ax2.set_title(f'{symbol} - Trade P&L Distribution', fontsize=12)
                ax2.set_xlabel('Trade #')
                ax2.set_ylabel('P&L (₹)')
                ax2.grid(True, alpha=0.3)
                
                # Add stats
                stats = results[symbol]['stats']
                stats_text = f"Trades: {stats['total_trades']}\nWin Rate: {stats['win_rate']}%\nTotal P&L: ₹{stats['total_pnl']:,.0f}"
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save chart
        chart_file = BASE_DIR / f"backtest_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart saved to: {chart_file}")
        
        plt.show()
        
    except ImportError:
        print("\n⚠️ matplotlib not installed. Run: pip install matplotlib")
        print("   Chart generation skipped.")
    
    # Save results
    report_file = BASE_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        # Convert trades to serializable format
        for sym in results:
            for trade in results[sym]['trades']:
                trade['entry_time'] = str(trade['entry_time'])
                trade['exit_time'] = str(trade['exit_time'])
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()
