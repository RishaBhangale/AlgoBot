# Supertrend Trading Bot

Fully automated options trading bot for NIFTY and BANKNIFTY with Telegram notifications.

## Features

- **Supertrend Indicator**: ATR Period 10, Multiplier 3.0
- **Timeframe**: 5 minutes
- **Auto-Login**: Selenium-based daily authentication
- **Telegram Notifications**: Real-time trade alerts & daily summaries
- **NIFTY**: Target +20%, SL -5%
- **BANKNIFTY**: Target +10%, SL -5%

## Files

```
supertrend-bot/
├── main.py             # Trading bot core
├── auto_login.py       # Selenium auto-login
├── run_bot.py          # Fully automated runner
├── telegram_notifier.py # Telegram notifications
├── api_key.txt         # Kite API credentials
├── .env                # Login & Telegram credentials
├── requirements.txt    # Python dependencies
└── logs/               # Daily logs & reports
```

---

## Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Credentials

**API Key** (`api_key.txt`):
```
your_kite_api_key
your_kite_api_secret
```

**Login & Telegram** (`.env`):
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

### Step 3: Setup Telegram Bot

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`, follow prompts
3. Copy the **bot token** to `.env`
4. Open Telegram, search for **@userinfobot**
5. It will show your **chat ID** - copy to `.env`

### Step 4: Test

```bash
# Test Telegram
python3 telegram_notifier.py

# Test auto-login
python3 auto_login.py

# Run bot
python3 run_bot.py
```

---

## Telegram Notifications

You'll receive:

| Event | Notification |
|-------|--------------|
| Bot Start | Securities, strategy details |
| Trade Entry | Option type, strike, entry, target, SL |
| Trade Exit | Exit price, P&L, reason (TARGET/SL/REVERSAL) |
| Daily Summary | Trades, winners, losers, total P&L |

---

## Cloud Deployment (Render - Free)

Render offers 750 free hours/month - enough for market hours (6+ hrs/day).

### Step 1: Create render.yaml

Create a `render.yaml` file in your project:

```yaml
services:
  - type: worker
    name: supertrend-bot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python run_bot.py
    envVars:
      - key: KITE_USER_ID
        sync: false
      - key: KITE_PASSWORD
        sync: false
      - key: KITE_TOTP_SECRET
        sync: false
      - key: TELEGRAM_BOT_TOKEN
        sync: false
      - key: TELEGRAM_CHAT_ID
        sync: false
```

### Step 2: Push to GitHub

```bash
cd supertrend-bot
git init
git add .
git commit -m "Supertrend bot"
git remote add origin https://github.com/yourusername/supertrend-bot.git
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** → **Background Worker**
3. Connect your GitHub repository
4. Select **Python** environment
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `python run_bot.py`

### Step 4: Add Environment Variables

In Render dashboard → Your service → **Environment**:

| Key | Value |
|-----|-------|
| `KITE_USER_ID` | Your Zerodha ID |
| `KITE_PASSWORD` | Your password |
| `KITE_TOTP_SECRET` | Your TOTP secret |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |

### Step 5: Deploy

Click **Manual Deploy** → **Deploy latest commit**

### Step 6: Monitor

- View logs in Render dashboard
- Get Telegram notifications on your phone

---

## Alternative: AWS Free Tier

AWS offers 750 hours/month for 12 months.

### Step 1: Create EC2 Instance

1. Go to [AWS Console](https://console.aws.amazon.com)
2. Launch EC2 → Amazon Linux 2 → t2.micro (free tier)
3. Download key pair (.pem file)

### Step 2: Connect

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ec2-user@your-public-ip
```

### Step 3: Setup

```bash
# Install dependencies
sudo yum install -y python3 python3-pip google-chrome-stable

# Upload bot files
scp -i your-key.pem -r supertrend-bot/ ec2-user@your-ip:~/

# Install packages
cd supertrend-bot
pip3 install -r requirements.txt
```

### Step 4: Run with Screen

```bash
screen -S trading
python3 run_bot.py

# Detach: Ctrl+A, D
# Reattach: screen -r trading
```

### Step 5: Auto-start (systemd)

```bash
sudo nano /etc/systemd/system/supertrend.service
```

```ini
[Unit]
Description=Supertrend Bot
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/supertrend-bot
ExecStart=/usr/bin/python3 run_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable supertrend
sudo systemctl start supertrend
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Telegram not working | Run `python3 telegram_notifier.py` to test |
| Auto-login fails | Check .env credentials, reset TOTP if needed |
| Chrome not found | Install: `apt install google-chrome-stable` |
| Token expired | Bot auto-refreshes daily at 9 AM |

---

## ⚠️ Disclaimer

- Automating Zerodha login may violate their Terms of Service
- This bot uses paper/simulated capital
- Use at your own risk
- Always monitor your trades via Telegram
