# Supertrend Trading Bot

Fully automated options trading bot for NIFTY and BANKNIFTY with Telegram notifications.

## Features

- **Supertrend Indicator**: ATR Period 10, Multiplier 3.0
- **Timeframe**: 5 minutes
- **Auto-Login**: Selenium-based daily authentication
- **Telegram Notifications**: Real-time trade alerts & daily summaries
- **NIFTY**: Target +20%, SL -5%
- **BANKNIFTY**: Target +10%, SL -5%

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
│  │  - /docs   → API documentation  │    │
│  └─────────────┬───────────────────┘    │
│                │                         │
│  ┌─────────────▼───────────────────┐    │
│  │  Background Thread              │    │
│  │  - SupertrendBot                │───►├─── Telegram Alerts
│  │  - Selenium Auto-Login          │    │
│  │  - Kite Connect API             │    │
│  └─────────────────────────────────┘    │
│                                          │
│  System: Chromium + ChromeDriver         │
└─────────────────────────────────────────┘
```

## Files

```
supertrend-bot/
├── app.py              # FastAPI server (entry point)
├── main.py             # Trading bot core
├── auto_login.py       # Selenium auto-login
├── telegram_notifier.py # Telegram notifications
├── Dockerfile          # Docker config with Chromium
├── render.yaml         # Render deployment config
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── logs/               # Daily logs
```

---

## Local Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Credentials

Create `.env` from template:
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
# With uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 5000

# Or using Python
python3 app.py
```

Visit `http://localhost:5000/docs` for FastAPI auto-generated documentation.

---

## Cloud Deployment (Render - Free Tier)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Supertrend bot"
git remote add origin https://github.com/yourusername/supertrend-bot.git
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com) and sign up
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Render will auto-detect `Dockerfile`
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

### Step 3: Setup UptimeRobot (Prevent Sleep)

Render free tier sleeps after 15 minutes of inactivity.

1. Go to [uptimerobot.com](https://uptimerobot.com) (free)
2. Sign up and click **Add New Monitor**
3. Settings:
   - **Monitor Type**: HTTP(s)
   - **URL**: `https://your-app.onrender.com/ping`
   - **Monitoring Interval**: 5 minutes
4. Click **Create Monitor**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check ("Bot is running!") |
| `/ping` | GET | Simple ping for UptimeRobot |
| `/status` | GET | Detailed bot status (JSON) |
| `/docs` | GET | FastAPI Swagger documentation |
| `/redoc` | GET | FastAPI ReDoc documentation |

---

## Telegram Notifications

| Event | Message |
|-------|---------|
| Bot Start | Securities, strategy |
| Trade Entry | Option type, strike, entry, target, SL |
| Trade Exit | Exit price, P&L, reason |
| Daily Summary | All trades, total P&L |

---

## Docker

Build and run locally:
```bash
docker build -t supertrend-bot .
docker run -p 10000:10000 --env-file .env supertrend-bot
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App sleeping | Verify UptimeRobot is pinging `/ping` |
| No Telegram | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` |
| Login fails | Verify Kite credentials in env vars |
| Chrome error | Ensure Dockerfile installs chromium |

---

## ⚠️ Disclaimer

- Automating Zerodha login may violate their Terms of Service
- This bot uses paper/simulated capital
- Use at your own risk
