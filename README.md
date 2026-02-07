# Quad-Confirmation Trading Bot

Automated paper trading bot for **NIFTY/BANKNIFTY** and **5 F&O Stocks** using the **Quad-Confirmation** strategy with Telegram notifications.

## 🎯 Strategy

| Parameter | Value |
|-----------|-------|
| **Indicators** | SuperTrend (20, 2) + MACD (12, 26, 9) + VWAP + PCR |
| **Timeframe** | 15 minutes |
| **MACD Lookback** | 2 candles (waits for SuperTrend confirmation) |
| **Entry** | All 4 indicators must align |
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

## 🔄 Entry Logic (with MACD Lookback)

```
Candle 1: ⚡ MACD Bullish Cross (valid for 2 more candles)
Candle 2: ST still bearish... waiting
Candle 3: ✅ ST flips bullish + VWAP + PCR → QUAD-CONFIRMATION BUY!
```

### BUY (Long CE)
- ✓ MACD crosses UP (or pending from last 2 candles)
- ✓ SuperTrend = Bullish (FLIP or ALIGN)
- ✓ Price < VWAP
- ✓ PCR < 1.0

### SELL (Long PE)
- ✓ MACD crosses DOWN (or pending from last 2 candles)
- ✓ SuperTrend = Bearish (FLIP or ALIGN)
- ✓ Price > VWAP
- ✓ PCR > 1.0

## 🚪 Exit Conditions

| Trigger | Action |
|---------|--------|
| MACD reversal | Exit immediately |
| SuperTrend reversal | Exit immediately |
| SL hit (prev candle) | Exit on tick |

## 📁 Files

```
supertrend-bot/
├── app.py               # FastAPI server + daily loop
├── main.py              # Index options bot (NIFTY/BANKNIFTY)
├── main_stocks.py       # Stock options bot (5 F&O stocks)
├── angel_one.py         # PCR via Angel One SmartAPI
├── auto_login.py        # Selenium auto-login (Kite)
├── telegram_notifier.py # Telegram notifications
├── Dockerfile           # Docker config with Chromium
├── render.yaml          # Render deployment config
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template
```

## 🏃 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/supertrend-bot.git
cd supertrend-bot

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
docker build -t supertrend-bot .
docker run -p 10000:10000 --env-file .env supertrend-bot
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
✅ [NIFTY] QUAD-CONFIRMATION BUY!
   MACD: ↑ (pending) | ST: Bullish (FLIP) | VWAP: Below | PCR: 0.85

🔔 [NIFTY] EXIT TRIGGERED: MACD_REVERSAL
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
