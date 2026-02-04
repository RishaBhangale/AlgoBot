#!/usr/bin/env python3
"""
Angel One SmartAPI Integration for PCR (Put-Call Ratio) calculation.
Uses WebSocket to get real-time Open Interest data.
"""
import os
import time
import threading
from datetime import datetime
from typing import Optional, Dict

try:
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False
    SmartConnect = None
    SmartWebSocketV2 = None

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False


class AngelOnePCR:
    """
    Calculates Put-Call Ratio (PCR) using Angel One SmartAPI.
    PCR = Total Put OI / Total Call OI
    
    PCR > 1.0 = Bearish sentiment
    PCR < 1.0 = Bullish sentiment
    """
    
    def __init__(self, logger=None):
        self.logger = logger or print
        self.smart_api: Optional[SmartConnect] = None
        self.websocket: Optional[SmartWebSocketV2] = None
        self.auth_token: str = ""
        self.feed_token: str = ""
        self.refresh_token: str = ""
        
        # OI data storage
        self.nifty_call_oi: int = 0
        self.nifty_put_oi: int = 0
        self.banknifty_call_oi: int = 0
        self.banknifty_put_oi: int = 0
        
        # PCR values
        self.nifty_pcr: float = 1.0
        self.banknifty_pcr: float = 1.0
        
        self.connected: bool = False
        self.last_update: Optional[datetime] = None
        
        # Credentials (MPIN-based login)
        self.api_key = os.environ.get("ANGEL_API_KEY", "")
        self.secret_key = os.environ.get("ANGEL_SECRET_KEY", "")
        self.client_id = os.environ.get("ANGEL_CLIENT_ID", "")
        self.mpin = os.environ.get("ANGEL_MPIN", "")
        self.totp_secret = os.environ.get("ANGEL_TOTP_SECRET", "")
    
    def login(self) -> bool:
        """Login to Angel One SmartAPI using MPIN + TOTP."""
        if not SMARTAPI_AVAILABLE:
            self.logger("⚠️ SmartAPI not installed. Run: pip install smartapi-python")
            return False
        
        if not PYOTP_AVAILABLE:
            self.logger("⚠️ pyotp not installed. Run: pip install pyotp")
            return False
        
        if not self.api_key or not self.client_id or not self.mpin:
            self.logger("⚠️ Angel One credentials not configured")
            return False
        
        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret).now() if self.totp_secret else ""
            
            # Login with MPIN + TOTP
            data = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.mpin,
                totp=totp
            )
            
            if data.get('status'):
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                self.connected = True
                self.logger("✅ Angel One login successful")
                return True
            else:
                self.logger(f"❌ Angel One login failed: {data.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger(f"❌ Angel One login error: {str(e)}")
            return False
    
    def get_nifty_options_tokens(self) -> list:
        """Get instrument tokens for NIFTY option chain."""
        # These are sample tokens - in production, fetch from scrip master
        # For now, return empty to indicate manual setup needed
        return []
    
    def start_oi_stream(self):
        """Start WebSocket stream for OI data."""
        if not self.connected:
            self.logger("⚠️ Not connected to Angel One")
            return
        
        try:
            # WebSocket for streaming OI data
            self.websocket = SmartWebSocketV2(
                self.auth_token,
                self.api_key,
                self.client_id,
                self.feed_token
            )
            
            def on_data(wsapp, message):
                self._process_oi_data(message)
            
            def on_open(wsapp):
                self.logger("📊 Angel One WebSocket connected")
                # Subscribe to OI data (implementation depends on scrip master)
            
            def on_error(wsapp, error):
                self.logger(f"⚠️ WebSocket error: {error}")
            
            def on_close(wsapp):
                self.logger("WebSocket closed")
            
            self.websocket.on_data = on_data
            self.websocket.on_open = on_open
            self.websocket.on_error = on_error
            self.websocket.on_close = on_close
            
            # Start in background thread
            thread = threading.Thread(target=self.websocket.connect, daemon=True)
            thread.start()
            
        except Exception as e:
            self.logger(f"❌ WebSocket setup error: {str(e)}")
    
    def _process_oi_data(self, data: Dict):
        """Process incoming OI data and update PCR."""
        try:
            # Extract OI from message (format depends on Angel One API)
            # This is a placeholder - actual implementation depends on message format
            if 'oi' in data:
                # Update OI values based on symbol
                pass
            
            self._calculate_pcr()
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger(f"⚠️ OI processing error: {str(e)}")
    
    def _calculate_pcr(self):
        """Calculate PCR from OI data."""
        if self.nifty_call_oi > 0:
            self.nifty_pcr = self.nifty_put_oi / self.nifty_call_oi
        
        if self.banknifty_call_oi > 0:
            self.banknifty_pcr = self.banknifty_put_oi / self.banknifty_call_oi
    
    def get_pcr(self, symbol: str = "NIFTY") -> float:
        """
        Get current PCR for a symbol.
        
        Returns:
            float: PCR value (>1 = bearish, <1 = bullish)
        """
        if symbol == "NIFTY":
            return self.nifty_pcr
        elif symbol == "BANKNIFTY":
            return self.banknifty_pcr
        return 1.0  # Neutral default
    
    def is_bullish(self, symbol: str = "NIFTY") -> bool:
        """Check if PCR indicates bullish sentiment (PCR < 1.0)."""
        return self.get_pcr(symbol) < 1.0
    
    def is_bearish(self, symbol: str = "NIFTY") -> bool:
        """Check if PCR indicates bearish sentiment (PCR > 1.0)."""
        return self.get_pcr(symbol) > 1.0
    
    def validate_signal(self, symbol: str, signal: str) -> bool:
        """
        Check if a trading signal aligns with PCR sentiment.
        
        Args:
            symbol: "NIFTY" or "BANKNIFTY"
            signal: "BUY" or "SELL"
        
        Returns:
            bool: True if signal aligns with PCR, False otherwise
        """
        pcr = self.get_pcr(symbol)
        
        if signal == "BUY":
            # BUY signal valid if PCR < 1.0 (bullish sentiment)
            return pcr < 1.0
        elif signal == "SELL":
            # SELL signal valid if PCR > 1.0 (bearish sentiment)
            return pcr > 1.0
        
        return True  # Unknown signal, allow by default
    
    def get_status(self) -> Dict:
        """Get current PCR status for all symbols."""
        return {
            "connected": self.connected,
            "last_update": str(self.last_update) if self.last_update else None,
            "nifty": {
                "pcr": round(self.nifty_pcr, 2),
                "sentiment": "BEARISH" if self.nifty_pcr > 1.0 else "BULLISH",
                "call_oi": self.nifty_call_oi,
                "put_oi": self.nifty_put_oi,
            },
            "banknifty": {
                "pcr": round(self.banknifty_pcr, 2),
                "sentiment": "BEARISH" if self.banknifty_pcr > 1.0 else "BULLISH",
                "call_oi": self.banknifty_call_oi,
                "put_oi": self.banknifty_put_oi,
            }
        }
    
    def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            try:
                self.websocket.close_connection()
            except:
                pass
        self.connected = False


# Singleton instance
_pcr_instance: Optional[AngelOnePCR] = None

def get_pcr_tracker(logger=None) -> AngelOnePCR:
    """Get or create PCR tracker singleton."""
    global _pcr_instance
    if _pcr_instance is None:
        _pcr_instance = AngelOnePCR(logger)
    return _pcr_instance
