# 🚀 Quad-Confirmation + Alligator Golden Zone Stock Trading Engine

Automated F&O Stock Options trading bot for **RELIANCE, ICICIBANK, SBIN, AXISBANK, LT** utilizing the **Quad-Confirmation + Alligator Golden Zone Overlay** strategy with headless 2FA auto-login, dynamic conviction position sizing, and real-time Telegram alerting.

---

## 🎯 Core Architecture & Strategy

| Component | Specification | Description |
| :--- | :--- | :--- |
| **Securities** | RELIANCE, ICICIBANK, SBIN, AXISBANK, LT | Top 5 liquid institutional momentum F&O stocks |
| **Execution Timeframes** | 15m (RELIANCE, LT), 30m (ICICIBANK, SBIN, AXISBANK) | Individual timeframes optimized for max Sharpe ratio |
| **Macro Regime Filter** | 75-minute Alligator Swing Pivots | Dynamically tracks $P_{\text{low}}, P_{\text{high}}$ to define Fibonacci Golden Zone ($50\% - 65\%$) |
| **Intraday Anchors** | 15-Minute Opening Range (09:15–09:30) & Rolling VWAP | Anchors intraday institutional equilibrium |
| **Entry Engine** | Quad-Confirmation Scoring + Golden Zone Boost | MACD (+1.0) + SuperTrend (+1.0/+1.5) + VWAP (+0.5) + PCR (+0.5) + **GZ Confluence (+1.0)** |
| **Conviction Sizing** | **2 Lots in Golden Zone**, **1 Lot Standard** | Amplifies sizing on high-probability institutional reload zones |
| **Risk & SL** | Dynamic Structural Stop-Loss | Placed at previous swing pivot / candle extreme (delta-adjusted) |
| **Auto Square-Off** | 15:15 IST (EOD Square-off) | Eliminates overnight gap risk |

---

## 📈 Scoring & Conviction System

| Condition | BUY Score | SELL Score | Notes |
| :--- | :---: | :---: | :--- |
| **MACD Pending Cross** | +1.0 | +1.0 | Lookback of 3 to 5 candles |
| **SuperTrend Aligned** | +1.0 | +1.0 | SuperTrend (20, 2) |
| **SuperTrend Flip Bonus** | +0.5 | +0.5 | Fresh structural trend transition |
| **VWAP Confirmation** | +0.5 | +0.5 | Price above/below cumulative session VWAP |
| **PCR Alignment** | +0.5 | +0.5 | PCR < 1.0 (Bullish) or > 1.0 (Bearish) |
| **Golden Zone Confluence** | **+1.0** | **+1.0** | Price / ORB inside $50\% - 65\%$ Fibonacci retracement |
| **Counter-Trend Protection** | **BLOCKED** | **BLOCKED** | Suppresses CE buys inside Bear GZ (and PE buys inside Bull GZ) |
| **Entry Threshold** | **≥ 2.0** | **≥ 2.0** | Max score: 4.5 |

### 💎 Position Sizing Rules
* **Inside Golden Zone**: Triggers **2 LOTS** (Amplified payoff on mean-reverting institutional rallies).
* **Outside Golden Zone**: Triggers **1 LOT** (Standard trend-following breakout).

---

## 🛠️ Automated Daily Lifecycle (`run_bot.py`)

No manual terminal interaction or token pasting required. The system runs autonomously:

```
 08:50 AM IST ───► Headless Auto-Login (Generates fresh Kite token via HTTP + TOTP)
 09:14 AM IST ───► Pre-Market Setup (Subscribes to live WebSocket tick feed & fetches 75M GZ levels)
 09:15 AM IST ───► Market Opens (Establishes 15M ORB, processes 15m/30m candles in real-time)
 Intraday     ───► Live Order Execution + Instant Telegram Notifications (Entry, SL, Exit, P&L)
 03:15 PM IST ───► Intraday Auto Square-Off (Closes any lingering open positions)
 03:30 PM IST ───► Market Closes (Computes metrics, writes logs/stocks_report_YYYY-MM-DD.json, sends EOD Telegram summary)
 03:31 PM IST ───► Standby Mode (Sleeps until 08:50 AM next trading day)
```

---

## 🚀 How to Run

### 1. Requirements & Dependencies
Ensure Python 3.10+ is available:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials (`.env`)
Create or edit `.env` in the project root:
```ini
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_USER_ID=your_zerodha_user_id
KITE_PASSWORD=your_zerodha_password
KITE_TOTP_SECRET=your_totp_secret_key

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 3. Start the Autonomous Daily Bot
```bash
python3 run_bot.py
```
To run permanently in the background (surviving terminal closure / disconnect):
```bash
nohup python3 run_bot.py > bot_output.log 2>&1 &
```

---

## 📱 Telegram Alerts

The bot delivers rich HTML notifications directly to your phone:
* **Bot Start**: Alerts symbols loaded and system initialization.
* **Trade Entry**: Security, Option Type (CE/PE), Strike, Entry Price, Dynamic SL, and Lot Size (`1 LOT` or `2 LOTS (Golden Zone Boost)`).
* **Trade Exit**: Exit price, points captured, exit reason (`MACD_REVERSAL`, `SUPERTREND_REVERSAL`, `SL_HIT`, `EOD_SQUAREOFF`), and realized P&L.
* **EOD Daily Summary**: Breakdown of trades, win rate, and total daily net P&L.

---

## 📊 Visualizing on TradingView

The exact indicator script matching the Python bot logic is included in:
* [`tradingview_stock_hybrid.pine`](file:///Users/rishabhbhangale/Desktop/Trading/supertrend-bot/tradingview_stock_hybrid.pine)

Open TradingView $\to$ Pine Editor $\to$ Paste script $\to$ "Add to Chart". It visually highlights the 15M ORB, Alligator Golden Zone boxes, and displays exact Buy CE / Buy PE label tags with 1x/2x lot sizes.

---

## 📁 Clean Directory Structure

```
supertrend-bot/
├── auto_login.py                 # Fast headless HTTP + TOTP Zerodha auto-login
├── main_stocks.py                # Core Quad-Confirmation + Alligator Golden Zone Bot
├── run_bot.py                    # Autonomous daily scheduler & lifecycle manager
├── telegram_notifier.py          # Real-time Telegram alerting engine
├── tradingview_stock_hybrid.pine # TradingView Pine Script v5 indicator visualizer
├── angel_one.py                  # PCR data module
├── requirements.txt              # Production Python dependencies
├── .env                          # Local credentials (gitignored)
└── logs/                         # Automated daily reports (stocks_report_YYYY-MM-DD.json)
```
