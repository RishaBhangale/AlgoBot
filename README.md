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
| MACD reversal | Exit immediately |
| SuperTrend reversal | Exit immediately |
| SL hit | Exit on tick |

## Securities

| Index | Lot Size | Strike Interval |
|-------|----------|-----------------|
| NIFTY | 50 | 50 |
| BANKNIFTY | 25 | 100 |

## Environment Variables

### Zerodha (Kite Connect)
```
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_secret
KITE_USER_ID=your_user_id
KITE_PASSWORD=your_password
KITE_TOTP_SECRET=your_totp_secret
```

### Telegram
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Angel One (for PCR)
```
ANGEL_API_KEY=your_angel_api_key
ANGEL_SECRET_KEY=your_angel_secret_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_MPIN=your_4_digit_mpin
ANGEL_TOTP_SECRET=your_totp_secret
```

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
└── requirements.txt     # Python dependencies
```

## Deploy to Render

1. Push to GitHub:
```bash
git add .
git commit -m "Triple-Confirmation strategy"
git push
```

2. Add environment variables in Render Dashboard

3. Deploy - bot auto-starts on push

## Tech Stack

- **Trading**: Zerodha Kite Connect
- **PCR Data**: Angel One SmartAPI
- **Framework**: FastAPI + Uvicorn
- **Hosting**: Render (free tier)
- **Notifications**: Telegram Bot
