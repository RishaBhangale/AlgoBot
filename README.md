# Scoring-Based Trading Bot

Automated paper trading bot for **NIFTY/BANKNIFTY** and **5 F&O Stocks** using a **Scoring-Based** entry system with Telegram notifications.

## 🎯 Strategy

| Parameter | Value |
|-----------|-------|
| **Indicators** | SuperTrend (20, 2) + MACD (12, 26, 9) + VWAP + PCR |
| **Timeframe** | 15 minutes |
| **MACD Lookback** | 3 candles (waits for SuperTrend confirmation) |
| **Entry** | Score ≥ 2.0 (MACD:1 + ST:1/1.5 + VWAP:0.5 + PCR:0.5) |
| **Exit** | MACD reversal OR SuperTrend reversal |
| **SL** | Previous candle low/high (dynamic) |
| **Target** | None - hold till reversal |

## 📊 Trading Modes

### Mode 1: Index Options (`main.py`)
| Index | Lot Size | Strike Interval |
|-------|----------|-----------------|
| NIFTY | 50 | 50 |
| BANKNIFTY | 25 | 100 |

### Mode 2: Stock Options (`main_stocks.py`)
| Stock | Lot Size | Strike Gap |
|-------|----------|------------|
| RELIANCE | 250 | ₹20 |
| TCS | 175 | ₹50 |
| INFY | 400 | ₹20 |
| HDFCBANK | 550 | ₹25 |
| ICICIBANK | 700 | ₹12.5 |

## 🔄 Entry Logic (Scoring System)

Each indicator contributes a weighted score:

| Indicator | BUY Score | SELL Score |
|-----------|-----------|------------|
| MACD pending | +1.0 | +1.0 |
| SuperTrend aligned | +1.0 | +1.0 |
| SuperTrend FLIP bonus | +0.5 | +0.5 |
| VWAP (above for BUY, below for SELL) | +0.5 | +0.5 |
| PCR (< 1.0 for BUY, > 1.0 for SELL) | +0.5 | +0.5 |
| **Threshold** | **≥ 2.0** | **≥ 2.0** |

```
Candle 1: ⚡ MACD Bullish Cross (valid for 3 more candles)
Candle 2: ST still bearish... score 1.0 (MACD only)
Candle 3: ✅ ST flips bullish + VWAP above → Score 3.0 → BUY!
```

## 🚪 Exit Conditions

| Trigger | Action |
|---------|--------|
| MACD reversal | Exit immediately |
| SuperTrend reversal | Exit immediately |
| SL hit (prev candle) | Exit on tick |

## 📁 Files

```
supertrend-bot/
├── app.py                  # FastAPI server + daily loop
├── main.py                 # Index options bot (NIFTY/BANKNIFTY)
├── main_stocks.py          # Stock options bot (5 F&O stocks)
├── backtest_comparison.py  # Old vs New strategy comparison
├── angel_one.py            # PCR via Angel One SmartAPI
├── auto_login.py           # Selenium auto-login (Kite)
├── telegram_notifier.py    # Telegram notifications
├── Dockerfile              # Docker config with Chromium
├── render.yaml             # Render deployment config
├── requirements.txt        # Python dependencies
└── .env.example            # Environment template
```

## 🏃 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/RishaBhangale/AlgoBot.git
cd AlgoBot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run Index Bot
python main.py

# OR Run Stock Bot
python main_stocks.py
```

### Run with Docker

```bash
docker build -t AlgoBot .
docker run -p 10000:10000 --env-file .env AlgoBot
```

## ☁️ Render Deployment

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Quad-Confirmation with MACD lookback"
git push
```

### Step 2: Add Environment Variables

In Render Dashboard → Environment:

**Zerodha:**
```
KITE_API_KEY
KITE_API_SECRET
KITE_USER_ID
KITE_PASSWORD
KITE_TOTP_SECRET
```

**Telegram:**
```
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

**Angel One (PCR):**
```
ANGEL_API_KEY
ANGEL_SECRET_KEY
ANGEL_CLIENT_ID
ANGEL_MPIN
ANGEL_TOTP_SECRET
```

### Step 3: Choose Bot Mode

Edit `Dockerfile` CMD line:

```dockerfile
# For Index Options (NIFTY/BANKNIFTY):
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

# For Stock Options (5 stocks):
# Modify app.py to import main_stocks instead of main
```

## 🌐 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /status` | Bot status JSON |
| `GET /logs` | Recent bot logs |

## 📈 Indicators

| Indicator | Purpose |
|-----------|---------|
| **SuperTrend (20,2)** | Trend direction + dynamic SL levels |
| **MACD (12,26,9)** | Entry timing via crossovers |
| **VWAP** | Fair value filter (real for stocks, estimated for indices) |
| **PCR** | Market sentiment from Angel One OI data |

## 📅 Daily Schedule

| Time | Action |
|------|--------|
| 8:30 AM | Bot wakes up |
| 8:45 AM | Selenium auto-login (fresh Kite token) |
| 9:15 AM | Fetch data, calculate indicators |
| 9:15 - 3:30 PM | Monitor for signals |
| 3:30 PM | Generate daily report |

## 📱 Telegram Alerts

```
✅ [RELIANCE] BUY SIGNAL! Score: 3.0/3.5
   MACD:+1.0 | ST:+1.5(FLIP) | VWAP:+0.5(above) | PCR:N/A

🔔 [RELIANCE] EXIT TRIGGERED: ST_REVERSAL
   Entry: ₹245.00 → Exit: ₹278.50
   P&L: ₹1,675
```

## 🛡️ Risk Management

| Parameter | Value |
|-----------|-------|
| Max Concurrent Positions | 3 (stocks mode) |
| Per-Stock Capital | ₹25,000 |
| Daily Loss Limit | ₹15,000 |
| Position Sizing | 1 lot per signal |

---

## Support

For issues, check the logs at `/logs` endpoint or Telegram alerts.
