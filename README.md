# Supertrend Trading Bot

Fully automated options trading bot for NIFTY and BANKNIFTY with Telegram notifications.

## Strategy

| Parameter | Value |
|-----------|-------|
| **Indicator** | Supertrend (ATR Period: 10, Multiplier: 3.0) |
| **Timeframe** | 5 minutes |
| **Target** | +20% profit |
| **Stop Loss** | -10% loss |
| **Entry** | Immediate on current trend |
| **Exit** | Target hit, SL hit, or Signal change |
| **Hold** | Till expiry (not daily) |
| **Capital** | ₹1,00,000 (simulated) |

## Trading Rules

1. **Bot starts** → Checks current Supertrend trend
2. **BULLISH trend** → Buys 1 lot CE (ATM strike)
3. **BEARISH trend** → Buys 1 lot PE (ATM strike)
4. **Monitors** every tick for Target (20%) / SL (10%)
5. **Signal changes** → Closes old position, opens new one
6. **Market closes** → Position held till next day/expiry

## Securities

| Index | Lot Size | Strike Interval |
|-------|----------|-----------------|
| NIFTY | 50 | 50 |
| BANKNIFTY | 25 | 100 |

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Automation**: Selenium + Chromium (headless)
- **Containerization**: Docker
- **Hosting**: Render (free tier)

## Architecture

```
┌─────────────────────────────────────────┐
│       Docker Container (Render)         │
│  ┌─────────────────────────────────┐    │
│  │  FastAPI Server (uvicorn)       │    │
│  │  - /       → Health check       │◄───┼─── UptimeRobot (every 5 min)
│  │  - /ping   → Ping               │    │
│  │  - /status → Bot status JSON    │    │
│  │  - /logs   → Recent bot logs    │    │
│  └─────────────┬───────────────────┘    │
│                │                        │
│  ┌─────────────▼───────────────────┐    │
│  │  Background Thread              │    │
│  │  - 8:45 AM: Selenium login      │    │
│  │  - 9:15 AM: Start trading       │───►├─── Telegram Alerts
│  │  - 3:30 PM: Generate report     │    │
│  └─────────────────────────────────┘    │
│                                         │
│  System: Chromium + ChromeDriver        │
└─────────────────────────────────────────┘
```

## Files

```
supertrend-bot/
├── app.py               # FastAPI server (entry point)
├── main.py              # Trading bot core
├── auto_login.py        # Selenium auto-login
├── telegram_notifier.py # Telegram notifications
├── Dockerfile           # Docker config with Chromium
├── render.yaml          # Render deployment config
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .gitignore           # Git ignore rules
└── logs/                # Daily logs & reports
```

---

## Local Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_USER_ID=AB1234
KITE_PASSWORD=your_password
KITE_TOTP_SECRET=your_totp_secret

TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=your_chat_id
```

### Step 3: Run Locally

```bash
python3 app.py
# or
uvicorn app:app --host 0.0.0.0 --port 5000
```

---

## Cloud Deployment (Render)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Deploy trading bot"
git push origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Render auto-detects `Dockerfile`
5. Add **Environment Variables**:

| Variable | Description |
|----------|-------------|
| `KITE_API_KEY` | Kite Connect API key |
| `KITE_API_SECRET` | Kite Connect API secret |
| `KITE_USER_ID` | Zerodha user ID |
| `KITE_PASSWORD` | Zerodha password |
| `KITE_TOTP_SECRET` | TOTP secret for 2FA |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |

6. Click **Create Web Service**

### Step 3: Setup UptimeRobot

Render free tier sleeps after 15 mins inactivity.

1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Add New Monitor:
   - Type: HTTP(s)
   - URL: `https://your-app.onrender.com/ping`
   - Interval: 5 minutes

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Health check status |
| `/ping` | Simple ping for UptimeRobot |
| `/status` | Detailed bot status (JSON) |
| `/logs` | Recent 50 log entries |
| `/logs/all` | All log entries |
| `/docs` | FastAPI Swagger docs |

---

## Telegram Notifications

| Event | Message |
|-------|---------|
| Bot Start | Securities, strategy details |
| Trade Entry | CE/PE, Strike, Entry, Target, SL, Qty |
| Trade Exit | Exit price, P&L, Reason (TARGET/SL/SIGNAL_CHANGE) |
| Daily Summary | All trades, wins, losses, total P&L |

---

## Daily Schedule

| Time | Action |
|------|--------|
| 8:45 AM | Selenium auto-login |
| 9:15 AM | Fetch data, enter position |
| 9:15 - 3:30 PM | Monitor for Target/SL/Signals |
| 3:30 PM | Generate daily report |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 0 candles loaded | Check Kite subscription has Historical API |
| WebSocket 403 | Check Kite subscription has Streaming API |
| Bot stuck waiting | Trigger Manual Deploy on Render |
| No Telegram | Verify bot token and chat ID |
| Login fails | Check credentials, reset TOTP if needed |

---

## Kite Connect Requirements

Your Kite Connect subscription must include:
- ✅ Historical Data API
- ✅ WebSocket Streaming API

Check at [developers.kite.trade](https://developers.kite.trade/)

---

## ⚠️ Disclaimer

- Automating Zerodha login may violate their Terms of Service
- This bot uses paper/simulated capital (₹1 lakh)
- Use at your own risk
- Monitor trades via Telegram
