#!/usr/bin/env python3
"""
Telegram Bot Integration for Trading Notifications
Sends real-time trade alerts, capital risk telemetry, and daily summaries.
"""
import os
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

BASE_DIR = Path(__file__).parent


def now_ist():
    return datetime.now(IST) if IST else datetime.now()


class TelegramNotifier:
    """
    Sends trading notifications to Telegram.
    
    Features:
    - Real-time trade alerts (entry/exit)
    - Position status updates
    - Daily P&L and Capital Risk summary
    """
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or self._load_from_env("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or self._load_from_env("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            print("⚠️ Telegram notifications disabled (missing credentials)")
    
    def _load_from_env(self, key: str) -> Optional[str]:
        """Load value from environment or .env file."""
        value = os.environ.get(key)
        if value:
            return value
        
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip().strip('"').strip("'")
        return None
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def notify_bot_start(self, securities: List[str], capital_tracker=None,
                         atr_period: int = 20, atr_mult: float = 2.0, timeframe: int = 15):
        """Notify that bot has started with bold capital & risk metrics."""
        cap_section = ""
        if capital_tracker:
            cap_today = capital_tracker.session_capital
            overall_pnl = capital_tracker.overall_pnl
            max_pos = capital_tracker.get_max_concurrent_positions()
            overall_sign = "+" if overall_pnl >= 0 else ""
            cap_section = (
                f"<b>Capital Today:</b> ₹{cap_today:,.2f}\n"
                f"<b>Overall P&L:</b> ₹{overall_sign}{overall_pnl:,.2f}\n"
                f"<b>Max Concurrent Positions:</b> {max_pos} (₹{capital_tracker.per_slot_capital:,.0f}/slot)\n\n"
            )

        message = (
            f"🟢 <b>BOT STARTED — {now_ist().strftime('%Y-%m-%d %H:%M')} IST</b>\n\n"
            f"{cap_section}"
            f"<b>Securities:</b> {', '.join(securities)}\n"
            f"<b>SuperTrend:</b> ATR {atr_period} × {atr_mult}\n"
            f"<b>Entry Threshold:</b> Score ≥ 2.0\n\n"
            f"<i>Scanning 15m / 30m candle closes...</i>"
        )
        self.send_message(message)
    
    def notify_trade_entry(self, security: str, option_type: str, strike: float,
                           entry_price: float, target: float, sl: float,
                           quantity: int, signal: str, amount_required: Optional[float] = None):
        """Notify new trade entry."""
        is_call = option_type.upper() in ("CE", "CALL") or "BUY" in signal.upper() or "LONG" in signal.upper()
        emoji = "🟢" if is_call else "🔴"
        direction = "BULLISH" if is_call else "BEARISH"
        sl_pct = ((1 - sl / entry_price) * 100) if entry_price > 0 else 0
        if amount_required is None and entry_price > 0 and quantity > 0:
            amount_required = quantity * entry_price

        amt_line = f"<b>Amount Required:</b> ₹{amount_required:,.2f}\n" if amount_required else ""

        message = (
            f"{emoji} <b>NEW TRADE — {security}</b>\n\n"
            f"<b>Signal:</b> {direction}  |  <b>Type:</b> {option_type} {int(strike)}\n\n"
            f"<b>Entry:</b> ₹{entry_price:.2f}\n"
            f"<b>SL:</b>    ₹{sl:.2f}  (–{sl_pct:.1f}%)\n"
            f"<b>Qty:</b>   {quantity}\n"
            f"{amt_line}\n"
            f"<b>Time:</b>  {now_ist().strftime('%H:%M')} IST"
        )
        self.send_message(message)
    
    def notify_trade_exit(self, security: str, option_type: str, strike: float,
                          entry_price: float, exit_price: float, pnl: float,
                          reason: str):
        """Notify trade exit."""
        emoji = "✅" if pnl > 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""

        reason_map = {
            "SL_HIT": "Stop-Loss Hit",
            "INDICATOR_REVERSAL_BEAR": "Indicator Reversal (Bearish)",
            "INDICATOR_REVERSAL_BULL": "Indicator Reversal (Bullish)",
            "EOD_SQUAREOFF": "End-of-Day Square-Off",
        }
        reason_text = reason_map.get(reason, reason)

        message = (
            f"{emoji} <b>TRADE CLOSED — {security}</b>\n\n"
            f"<b>Contract:</b> {option_type} {int(strike)}  |  <b>Reason:</b> {reason_text}\n\n"
            f"<b>Entry:</b> ₹{entry_price:.2f}  →  <b>Exit:</b> ₹{exit_price:.2f}\n"
            f"<b>Net P&L:</b> ₹{pnl_sign}{pnl:,.2f}\n\n"
            f"<b>Time:</b> {now_ist().strftime('%H:%M')} IST"
        )
        self.send_message(message)
    
    def notify_daily_summary(self, date: str, securities_data: Dict, total_pnl: float,
                             diagnostics: Optional[Dict] = None, capital_summary: Optional[Dict] = None):
        """Send daily trading summary with bold capital risk tracking."""
        pnl_sign = "+" if total_pnl >= 0 else ""
        
        summary_lines = []
        total_trades = 0
        total_wins = 0
        total_losses = 0
        
        for symbol, data in securities_data.items():
            trades = data.get("trades", 0)
            pnl = data.get("pnl", 0)
            wins = data.get("wins", 0)
            losses = data.get("losses", 0)
            total_trades += trades
            total_wins += wins
            total_losses += losses
            sym_pnl_sign = "+" if pnl >= 0 else ""
            summary_lines.append(f"  <b>{symbol}:</b> {trades} trades | W:{wins} L:{losses} | ₹{sym_pnl_sign}{pnl:,.2f}")
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        cap_section = ""
        if capital_summary:
            cap_used = capital_summary.get("capital_used", 100000.0)
            is_base_restored = capital_summary.get("is_base_restored", False)
            profit_reaped = capital_summary.get("profit_reaped", 0.0)
            deficit = capital_summary.get("deficit", 0.0)
            day_pnl = capital_summary.get("day_pnl", 0.0)
            next_day_cap = capital_summary.get("next_day_capital", 100000.0)
            overall = capital_summary.get("overall_pnl", 0.0)
            overall_sign = "+" if overall >= 0 else ""
            
            if is_base_restored:
                if profit_reaped > 0:
                    cap_details = (
                        f"<b>Capital Used Today:</b> ₹{cap_used:,.2f}\n"
                        f"<b>Profit Credited:</b> ₹+{profit_reaped:,.2f} (Added to passive income)\n"
                        f"<b>Tomorrow Starts:</b> ₹{next_day_cap:,.2f} (Reset to base — profits reaped)\n"
                        f"<b>Overall P&L:</b> ₹{overall_sign}{overall:,.2f} (since inception)"
                    )
                else:
                    cap_details = (
                        f"<b>Capital Used Today:</b> ₹{cap_used:,.2f}\n"
                        f"<b>Principal Restored:</b> ₹{next_day_cap:,.2f} (Base capital intact)\n"
                        f"<b>Tomorrow Starts:</b> ₹{next_day_cap:,.2f}\n"
                        f"<b>Overall P&L:</b> ₹{overall_sign}{overall:,.2f} (since inception)"
                    )
            else:
                status_note = f"(Recovering — ₹{deficit:,.2f} below base)" if day_pnl >= 0 else "(Loss carried forward)"
                cap_details = (
                    f"<b>Capital Used Today:</b> ₹{cap_used:,.2f}\n"
                    f"<b>Capital Remaining:</b> ₹{next_day_cap:,.2f}\n"
                    f"<b>Tomorrow Starts:</b> ₹{next_day_cap:,.2f} {status_note}\n"
                    f"<b>Overall P&L:</b> ₹{overall_sign}{overall:,.2f} (since inception)"
                )
            cap_section = f"\n━━━━━━━━━━━━━━━━━━━━━━\n{cap_details}\n━━━━━━━━━━━━━━━━━━━━━━\n"

        diag_section = ""
        if diagnostics:
            diag_lines = []
            for sym, d in diagnostics.items():
                candles = d.get("candles", 0)
                ticks = d.get("ticks", 0)
                peak = d.get("peak_score", 0.0)
                reason = d.get("block_reason", "Criteria not met")
                diag_lines.append(f"  <b>{sym}:</b> {candles} candles ({ticks:,} ticks) | Peak: {peak:.1f}/2.0\n    <i>{reason}</i>")
            verdict = "<b>System Verdict:</b> No trades — strict filters held." if total_trades == 0 else f"<b>System Verdict:</b> {total_trades} trade(s) executed."
            diag_section = f"\n<b>Filter Diagnostics:</b>\n" + "\n".join(diag_lines) + f"\n\n{verdict}\n"
        
        message = (
            f"📊 <b>EOD SUMMARY — {date}</b>\n\n"
            + "\n".join(summary_lines) +
            f"\n\n<b>Total Trades:</b> {total_trades}  |  <b>Winners:</b> {total_wins}  <b>Losers:</b> {total_losses}  |  <b>Win Rate:</b> {win_rate:.0f}%\n"
            f"<b>P&L Today:</b> ₹{pnl_sign}{total_pnl:,.2f}\n"
            f"{cap_section}"
            f"{diag_section}\n"
            f"<i>Session closed at {now_ist().strftime('%H:%M')} IST</i>"
        )
        self.send_message(message)
    
    def notify_capital_alert(self, symbol: str, current_capital: float, min_required: float, overall_pnl: float, reason: str):
        """Notify when capital drops below operational threshold or hits 0."""
        overall_sign = "+" if overall_pnl >= 0 else ""
        message = (
            f"⚠️ <b>CAPITAL RISK ALERT</b>\n\n"
            f"<b>Status:</b> Trading Halted\n"
            f"<b>Current Capital:</b> ₹{current_capital:,.2f}\n"
            f"<b>Min Required Slot:</b> ₹{min_required:,.2f}\n"
            f"<b>Overall P&L:</b> ₹{overall_sign}{overall_pnl:,.2f}\n\n"
            f"<b>Reason:</b> {reason}\n\n"
            f"<i>{now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST</i>"
        )
        self.send_message(message)
    
    def notify_midday_heartbeat(self, status_dict: Dict, total_ticks: int, active_positions: int):
        """Send mid-day heartbeat ping at 12:00 PM IST."""
        lines = []
        for sym, d in status_dict.items():
            trend = d.get("trend", "NEUTRAL")
            dot = "🟢" if trend == "BULLISH" else ("🔴" if trend == "BEARISH" else "⚪")
            lines.append(f"  <b>{sym}:</b> {dot} {trend} | LTP ₹{d.get('ltp', 0):.2f} | Peak {d.get('peak_score', 0.0):.1f}/2.0")

        message = (
            f"💓 <b>MID-DAY CHECK — 12:00 PM IST</b>\n\n"
            f"<b>Ticks Processed:</b> {total_ticks:,}\n"
            f"<b>Open Positions:</b> {active_positions}\n\n"
            + "\n".join(lines) +
            f"\n\n<i>Bot healthy — scanning 15m/30m candle closes.</i>"
        )
        self.send_message(message)
    
    def notify_near_miss(self, symbol: str, direction: str, score: float, breakdown: List[str], ltp: float):
        """Send near-miss alert when score >= 1.5."""
        message = (
            f"🟡 <b>NEAR-MISS — {symbol}</b>\n\n"
            f"<b>Direction:</b> {direction}  |  <b>Score:</b> {score:.1f}/2.0\n"
            f"<b>LTP:</b> ₹{ltp:.2f}  |  <b>Time:</b> {now_ist().strftime('%H:%M')} IST\n\n"
            f"<code>{' | '.join(breakdown)}</code>\n\n"
            f"<i>0.5 pts from trigger — waiting for confirmation.</i>"
        )
        self.send_message(message)
    
    def notify_error(self, error: str):
        """Notify about an error."""
        message = f"⚠️ <b>BOT ERROR</b>\n\n{error}\n\n<i>{now_ist().strftime('%H:%M:%S')} IST</i>"
        self.send_message(message)
    
    def notify_market_waiting(self, hours: int, mins: int):
        """Notify that bot is waiting for market."""
        message = (
            f"⏳ <b>WAITING FOR MARKET</b>\n\n"
            f"<b>Market opens in:</b> {hours}h {mins}m\n"
            f"Will auto-login and start trading.\n\n"
            f"<i>{now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST</i>"
        )
        self.send_message(message)


def test_telegram():
    """Test Telegram connection."""
    print("\n" + "=" * 50)
    print("🔔 TELEGRAM NOTIFICATION TEST")
    print("=" * 50)
    
    notifier = TelegramNotifier()
    
    if not notifier.enabled:
        print("❌ Telegram not configured")
        return
    
    print("Sending test message...")
    success = notifier.send_message("🧪 <b>Test message from Supertrend Bot!</b>\n\nIf you see this, Telegram is configured correctly.")
    
    if success:
        print("✅ Test message sent!")
    else:
        print("❌ Failed to send message")


if __name__ == "__main__":
    test_telegram()
