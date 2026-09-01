# Stock Options Momentum Trading Engine

Autonomous algorithmic trading system for NSE F&O Stock Options (**RELIANCE, ICICIBANK, SBIN, AXISBANK, LT**) utilizing a **Multi-Confirmation Scoring Framework + Alligator Fibonacci Golden Zone Overlay**. Built for 24/7 cloud execution on **Render** (monitored via **UptimeRobot**), sub-second headless Zerodha Kite Connect 2FA auto-login, real-time Open Interest Put-Call Ratio (PCR) tracking, live exchange ATM option quotes, and automated Telegram alerting.

---

## 🎯 System Architecture & Strategy

| Component | Specification | Description |
| :--- | :--- | :--- |
| **Securities** | RELIANCE, ICICIBANK, SBIN, AXISBANK, LT | Top 5 liquid institutional momentum F&O stocks |
| **Execution Timeframes** | 15m (RELIANCE, LT), 30m (ICICIBANK, SBIN, AXISBANK) | Per-stock timeframes optimized for trend stability |
| **Macro Regime Filter** | 75-minute Alligator Swing Pivots | Tracks $P_{\text{low}}, P_{\text{high}}$ to define Fibonacci Golden Zone ($50\% - 65\%$) |
| **Intraday Anchors** | 15-Minute Opening Range (09:15–09:30) & Rolling VWAP | Anchors intraday institutional equilibrium |
| **Entry Scoring Engine** | Weighted Multi-Confirmation Scoring | MACD (+1.0) + SuperTrend (+1.0/+1.5) + VWAP (+0.5) + Native PCR (+0.5) + **Golden Zone Boost (+1.0)** |
| **Native Stock PCR** | Live Kite API Open Interest | Calculates $\text{PCR} = \frac{\sum \text{Put OI}}{\sum \text{Call OI}}$ across active stock option strikes |
| **Real Option Execution** | Live ATM Market Quotes | Resolves exact nearest ATM contract from Kite API; exits on real option LTP (no synthetic formulas) |
| **Conviction Sizing** | **2 Lots in Golden Zone**, **1 Lot Standard** | Amplifies sizing on high-probability institutional reload zones |
| **Dynamic Risk & SL** | 25% Option Stop-Loss / Structure SL | Real-time tick monitoring cuts losses instantly |
| **Intraday Square-Off** | 15:15 IST Auto Square-Off | Eliminates overnight gap and black-swan risk |

---

## 💰 Capital & Risk Management

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **Total Account Capital** | **₹1,00,000 (₹1.0 Lakh)** | Total allocated trading capital |
| **Max Concurrent Positions** | **3 stocks** | Prevents over-concentration across correlated names |
| **Per-Stock Capital Limit** | **₹25,000** | Maximum premium allocated per trade |
| **Daily Loss Limit** | **₹10,000** | Circuit breaker: Halts trading if daily loss hits 10% |

---

## 📈 Entry Scoring & Sizing Framework

| Signal Condition | BUY Score | SELL Score | Notes |
| :--- | :---: | :---: | :--- |
| **MACD Pending Cross** | +1.0 | +1.0 | Valid for 3 to 5 candle lookback |
| **SuperTrend Aligned** | +1.0 | +1.0 | SuperTrend (20, 2) |
| **SuperTrend Flip Bonus** | +0.5 | +0.5 | Fresh structural trend transition |
| **VWAP Confirmation** | +0.5 | +0.5 | Price above/below cumulative session VWAP |
| **Native PCR Alignment** | +0.5 | +0.5 | Live Stock PCR < 1.0 (Bullish) or > 1.0 (Bearish) |
| **Golden Zone Confluence** | **+1.0** | **+1.0** | Price / ORB inside $50\% - 65\%$ Fibonacci retracement |
| **Counter-Trend Protection** | **BLOCKED** | **BLOCKED** | Suppresses CE buys inside Bear GZ (and PE buys inside Bull GZ) |
| **Entry Threshold** | **≥ 2.0** | **≥ 2.0** | Maximum score: 4.5 |

### 💎 Position Sizing Rules
* **Inside Golden Zone**: Triggers **2 LOTS** (High-conviction reload setup).
* **Outside Golden Zone**: Triggers **1 LOT** (Standard breakout trade).

---

## 🛠️ Autonomous Daily Lifecycle

Runs autonomously without requiring manual intervention:

```
 08:50 AM IST ───► Headless Auto-Login (Refreshes Zerodha Kite token in <1 sec via HTTP + 2FA TOTP)
 09:14 AM IST ───► Pre-Market Setup (Downloads live NFO lot sizes, NSE spot tokens, & 75M GZ levels)
 09:15 AM IST ───► Market Opens (Subscribes to live WebSocket tick feed, builds 15m/30m candles)
 09:30–15:15  ───► Live Execution (Updates PCR every 15m, evaluates entries on candle close, manages real-time SL)
 12:00 PM IST ───► Mid-Day Heartbeat (Sends live container health, tick counts, and trend status to Telegram)
 03:15 PM IST ───► Auto Square-Off (Closes open intraday positions)
 03:30 PM IST ───► EOD Reporting (Sends P&L & Filter Diagnostic Matrix to Telegram, writes JSON logs)
 03:31 PM IST ───► Standby Mode (Sleeps until 08:50 AM next trading morning)
```

---

## 🚀 Deployment Guide

### Option 1: Cloud Deployment (Render + UptimeRobot)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "feat(stock-bot): update production configuration"
   git push origin main
   ```
2. **Deploy on Render**:
   * Create a **New Web Service** $\to$ Connect repository $\to$ Runtime: **Docker**.
   * Add Environment Variables:
     * `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID`, `KITE_PASSWORD`, `KITE_TOTP_SECRET`
     * `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
     * `PAPER_TRADING` = `true` (or `false` for live execution)
3. **Keep Awake via UptimeRobot**:
   * Create a free HTTP monitor at [UptimeRobot.com](https://uptimerobot.com) pointing to `https://<your-render-app>.onrender.com/ping` at 5-minute intervals.

---

### Option 2: Local Mac / VPS Runner

Run locally in the background (surviving terminal closure / sleep):
```bash
caffeinate -dis nohup python3 run_bot.py > bot_output.log 2>&1 &
```

* **Monitor live output**:
  ```bash
  tail -f bot_output.log
  ```
* **Stop bot**:
  ```bash
  pkill -f run_bot.py
  ```

---

## 📱 Telegram Notifications

* **Bot Initialization**: Confirms Kite login, symbols loaded, and active capital limits.
* **Trade Entry**: Security, Option Contract (e.g. `RELIANCE26SEP1360CE`), Entry LTP, Stop-Loss, and Quantity.
* **Near-Miss Watch**: Instant low-priority alert when a stock reaches Score $\ge 1.5$ showing exact component breakdown.
* **Mid-Day Heartbeat (12:00 PM)**: Confirms container health, total ticks processed, and live stock trends.
* **Trade Exit**: Realized exit LTP, points captured, exit reason, and net realized P&L.
* **EOD Diagnostic Summary**: Complete P&L breakdown plus the **Daily Filter Matrix** showing candle counts, peak scores, and block reasons for every stock.

---

## 📁 Repository Structure

```
supertrend-bot/
├── app.py                         # FastAPI web server + keepalive pinger for Render
├── main_stocks.py                 # Core Stock Options Execution Engine & Native PCR Tracker
├── run_bot.py                     # Standalone runner for local/VPS execution
├── auto_login.py                  # Sub-second headless HTTP + 2FA TOTP Zerodha auto-login
├── telegram_notifier.py           # Real-time Telegram alerting engine
├── tradingview_stock_hybrid.pine  # TradingView Pine Script v5 Visualizer
├── Dockerfile                     # Docker container configuration
├── render.yaml                    # Render cloud infrastructure blueprint
├── requirements.txt               # Production Python dependencies
├── .env.example                   # Environment variable template
└── logs/                          # Daily JSON execution logs
```
