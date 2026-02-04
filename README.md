# Triple-Confirmation Trading Bot

Fully automated paper trading bot for NIFTY and BANKNIFTY using **4-indicator Triple-Confirmation** strategy with Telegram notifications.

## Strategy

| Parameter | Value |
|-----------|-------|
| **Indicators** | SuperTrend (20, 2) + MACD (12, 26, 9) + VWAP + PCR |
| **Timeframe** | 15 minutes |
| **Entry** | All 4 indicators must align |
| **Exit** | MACD reversal OR SuperTrend reversal |
| **SL** | Previous candle low/high (dynamic) |
| **Target** | None - hold till reversal |
| **Capital** | ₹1,00,000 (paper trading) |

## Entry Conditions

### BUY (Long CE)
- ✓ MACD crosses UP (bullish crossover)
- ✓ SuperTrend = Bullish (trend = 1)
- ✓ Price < VWAP
- ✓ PCR < 1.0 (bullish sentiment)

### SELL (Long PE)
- ✓ MACD crosses DOWN (bearish crossover)
- ✓ SuperTrend = Bearish (trend = -1)
- ✓ Price > VWAP
- ✓ PCR > 1.0 (bearish sentiment)

## Exit Conditions

| Trigger | Action |
|---------|--------|
| MACD reversal crossover | Exit immediately |
| SuperTrend reversal | Exit immediately |
| SL hit (prev candle low/high) | Exit on tick |

## Securities

| Index | Lot Size | Strike Interval |
|-------|----------|-----------------|
| NIFTY | 50 | 50 |
| BANKNIFTY | 25 | 100 |

## Daily Schedule

| Time | Action |
|------|--------|
| 8:30 AM | Bot wakes up |
| 8:45 AM | Selenium auto-login (fresh Kite token) |
| 9:15 AM | Fetch data, calculate indicators, wait for entry |
| 9:15 - 3:30 PM | Monitor for signals, exits, SL |
| 3:30 PM | Generate daily report |
| Next day | Repeat with fresh login |

## Tech Stack

- **Trading**: Zerodha Kite Connect
- **PCR Data**: Angel One SmartAPI
- **Framework**: FastAPI + Uvicorn
- **Automation**: Selenium + Chromium (headless)
- **Containerization**: Docker
- **Hosting**: Render (free tier)
- **Notifications**: Telegram Bot

## Files

```
supertrend-bot/
├── app.py               # FastAPI server + daily loop
├── main.py              # Trading bot core + Triple-Confirmation
├── angel_one.py         # PCR via Angel One SmartAPI
├── auto_login.py        # Selenium auto-login (Kite)
├── telegram_notifier.py # Telegram notifications
├── Dockerfile           # Docker config with Chromium
├── render.yaml          # Render deployment config
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template
```

---

## Local Setup

### Step 1: Install Dependencies

```bash
cd supertrend-bot
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Step 3: Run Locally

```bash
python app.py
```

---

## Render Deployment

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Triple-Confirmation strategy"
git push
```

### Step 2: Add Environment Variables

In Render Dashboard → Environment, add:

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

### Step 3: Deploy

Render auto-deploys on push. Monitor logs for status.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /status` | Bot status JSON |
| `GET /logs` | Recent bot logs |

---

## Indicators Explained

### SuperTrend (20, 2)
- Trend-following indicator using ATR
- Bullish when close > SuperTrend line
- Bearish when close < SuperTrend line

### MACD (12, 26, 9)
- Momentum indicator
- Bullish crossover: MACD line crosses above Signal line
- Bearish crossover: MACD line crosses below Signal line

### VWAP
- Volume Weighted Average Price
- BUY when price below VWAP (undervalued)
- SELL when price above VWAP (overvalued)

### PCR (Put-Call Ratio)
- Market sentiment indicator from Angel One OI data
- PCR < 1.0 = Bullish (more calls than puts)
- PCR > 1.0 = Bearish (more puts than calls)

---

## Support

For issues or questions, check the logs at `/logs` endpoint.
