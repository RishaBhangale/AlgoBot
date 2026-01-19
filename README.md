# Supertrend Trading Bot

Fully automated options trading bot for NIFTY and BANKNIFTY with Telegram notifications.

## Features

- **Supertrend Indicator**: ATR Period 10, Multiplier 3.0
- **Timeframe**: 5 minutes
- **Auto-Login**: Selenium-based daily authentication
- **Telegram Notifications**: Real-time trade alerts & daily summaries
- **NIFTY**: Target +20%, SL -5%
- **BANKNIFTY**: Target +10%, SL -5%

## Architecture

```
┌─────────────────────────────────────────┐
│           Render (Free Tier)            │
│  ┌─────────────────────────────────┐    │
│  │  Flask Web Server (gunicorn)    │    │
│  │  - Health check endpoint        │◄───┼─── UptimeRobot (every 5 min)
│  │  - /status endpoint             │    │
│  └─────────────┬───────────────────┘    │
│                │                         │
│  ┌─────────────▼───────────────────┐    │
│  │  Background Thread              │    │
│  │  - SupertrendBot                │───►├─── Telegram Alerts
│  │  - Kite Connect                 │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Files

```
supertrend-bot/
├── app.py              # Flask wrapper (entry point for Render)
├── main.py             # Trading bot core
├── auto_login.py       # Selenium auto-login
├── telegram_notifier.py # Telegram notifications
├── render.yaml         # Render config
├── requirements.txt    # Python dependencies
├── .env                # Credentials (not in git)
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
KITE_USER_ID=AB1234
KITE_PASSWORD=your_password
KITE_TOTP_SECRET=your_totp_secret

TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=your_chat_id
```

### Step 3: Create api_key.txt

```
your_kite_api_key
your_kite_api_secret
```

### Step 4: Test Locally

```bash
# Test Telegram
python3 telegram_notifier.py

# Run bot locally
python3 app.py
```

---

## Cloud Deployment (Render - Free Tier)

### Step 1: Push to GitHub

```bash
cd supertrend-bot
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
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Add **Environment Variables**:

| Key | Value |
|-----|-------|
| `KITE_USER_ID` | Your Zerodha ID |
| `KITE_PASSWORD` | Your password |
| `KITE_TOTP_SECRET` | Your TOTP secret |
| `TELEGRAM_BOT_TOKEN` | Your bot token |
| `TELEGRAM_CHAT_ID` | Your chat ID |

6. Click **Create Web Service**

### Step 3: Setup UptimeRobot (Prevent Sleep)

Render free tier sleeps after 15 minutes of inactivity. UptimeRobot pings your app to keep it awake.

1. Go to [uptimerobot.com](https://uptimerobot.com) (free)
2. Sign up and click **Add New Monitor**
3. Settings:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: Supertrend Bot
   - **URL**: `https://your-app.onrender.com/ping`
   - **Monitoring Interval**: 5 minutes
4. Click **Create Monitor**

### Step 4: Verify

- Visit `https://your-app.onrender.com/` → Should show "Bot is running!"
- Visit `https://your-app.onrender.com/status` → Shows bot details
- Check Telegram for notifications

---

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/` | Health check (returns "Bot is running!") |
| `/ping` | Simple ping for UptimeRobot |
| `/status` | Detailed bot status JSON |

---

## Telegram Notifications

| Event | Message |
|-------|---------|
| Bot Start | Securities, strategy |
| Trade Entry | Option type, strike, entry, target, SL |
| Trade Exit | Exit price, P&L, reason |
| Daily Summary | All trades, total P&L |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App sleeping | Check UptimeRobot is configured |
| No Telegram messages | Run `python3 telegram_notifier.py` locally |
| Login fails | Verify .env credentials |
| Check logs | Render Dashboard → Logs |

---

## ⚠️ Disclaimer

- Automating Zerodha login may violate their Terms of Service
- This bot uses paper/simulated capital
- Use at your own risk
