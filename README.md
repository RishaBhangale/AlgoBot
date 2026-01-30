# Supertrend Trading Bot

Fully automated paper trading bot for NIFTY and BANKNIFTY with Telegram notifications and **hybrid trailing stop-loss**.

## Strategy

| Parameter | Value |
|-----------|-------|
| **Indicator** | Supertrend (ATR Period: 10, Multiplier: 3.0) |
| **Timeframe** | 5 minutes |
| **Target** | +20% profit |
| **Initial SL** | -10% loss |
| **Trailing SL** | Hybrid (Supertrend + ATR based) |
| **Entry** | Immediate on current trend |
| **Exit** | Target hit, Trailing SL hit, or Signal change |
| **Hold** | Till expiry (not daily) |
| **Capital** | ₹1,00,000 (paper trading) |

## Trading Rules

1. **Bot starts** → Fetches 5 days historical data
2. **Calculates Supertrend** → Determines current trend
3. **BULLISH trend** → Buys 1 lot CE (ATM strike)
4. **BEARISH trend** → Buys 1 lot PE (ATM strike)
5. **Every candle** → Updates trailing SL (never goes backward)
6. **Every tick** → Checks for Target (20%) / Trailing SL hit
7. **Signal changes** → Closes position, opens new one

## Hybrid Trailing Stop-Loss

The bot uses a hybrid trailing SL that combines:

| Method | Formula |
|--------|---------|
| **Supertrend-based** | SL at Supertrend value ± buffer |
| **ATR-based** | SL at Peak - (2 × ATR) |
| **Final SL** | `MAX(Supertrend SL, ATR SL)` - Uses tighter/more protective |

### Example Flow
```
Entry CE @ ₹100, Spot: 23,000
Initial SL: ₹90 (-10%)

Candle 1: Spot rises to 23,100
  → Trailing SL moves to ₹100 (breakeven!)

Candle 2: Spot rises to 23,200  
  → Trailing SL moves to ₹105

Candle 3: Spot dips to 23,100
  → Exit at ₹105 (locked in +5% profit)
```

## Securities

| Index | Lot Size | Strike Interval |
|-------|----------|-----------------|
| NIFTY | 50 | 50 |
| BANKNIFTY | 25 | 100 |

## Daily Schedule

| Time | Action |
|------|--------|
| 8:30 AM | Bot wakes up |
| 8:45 AM | Selenium auto-login (fresh token) |
| 9:15 AM | Fetch data, calculate Supertrend, enter position |
| 9:15 - 3:30 PM | Monitor for Target/SL/Signals, trail SL |
| 3:30 PM | Generate daily report |
| Next day | Repeat with fresh login |

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Automation**: Selenium + Chromium (headless)
- **Containerization**: Docker
- **Hosting**: Render (free tier)
- **Notifications**: Telegram Bot

## Files

```
supertrend-bot/
├── app.py               # FastAPI server + daily loop
├── main.py              # Trading bot core + trailing SL
├── auto_login.py        # Selenium auto-login
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
```

---

## Cloud Deployment (Render)

1. Push to GitHub
2. Create Web Service on [render.com](https://render.com)
3. Add environment variables
4. Setup UptimeRobot for keep-alive pings

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Health check with status and trading day |
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
| **Trailing SL Update** | Old SL → New SL, Peak price |
| Trade Exit | Exit price, P&L, Reason |
| Daily Summary | All trades, wins, losses, total P&L |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 0 candles loaded | Historical data API issue - wait or check subscription |
| Extra signals vs TradingView | Ensure same ATR period (10) and multiplier (3.0) |
| WebSocket 403 | Kite streaming subscription required |
| Bot not trading | Check `/logs` endpoint for errors |
| No Telegram | Verify bot token and chat ID |

---

## Kite Connect Requirements

Your Kite Connect subscription must include:
- ✅ Historical Data API
- ✅ WebSocket Streaming API

---

## ⚠️ Disclaimer

- This bot uses paper/simulated capital (₹1 lakh)
- No real orders are placed
- Automating Zerodha login may violate their TOS
- Use at your own risk
