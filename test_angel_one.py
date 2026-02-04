#!/usr/bin/env python3
"""
Test script to verify Angel One SmartAPI connection.
Run this locally to confirm your credentials work.
"""
import os

# Set credentials (for testing only - don't commit these!)
os.environ["ANGEL_API_KEY"] = "FZ3b7eCt"
os.environ["ANGEL_SECRET_KEY"] = "5752ce60-7115-451b-9e8f-8c03a7bbb47f"
os.environ["ANGEL_CLIENT_ID"] = "AABM714130"
os.environ["ANGEL_PASSWORD"] = "Ri$habhB@0210"
os.environ["ANGEL_TOTP_SECRET"] = "HWFHLSCHCOGKKQVN3QJ36FDVUA"

print("=" * 50)
print("Angel One SmartAPI Connection Test")
print("=" * 50)

# Check if smartapi is installed
try:
    from SmartApi import SmartConnect
    print("✅ SmartApi package installed")
except ImportError:
    print("❌ SmartApi not installed. Run: pip install smartapi-python")
    exit(1)

# Check pyotp for TOTP
try:
    import pyotp
    print("✅ pyotp package installed")
except ImportError:
    print("❌ pyotp not installed. Run: pip install pyotp")
    exit(1)

# Test connection
try:
    api_key = os.environ.get("ANGEL_API_KEY")
    client_id = os.environ.get("ANGEL_CLIENT_ID")
    password = os.environ.get("ANGEL_PASSWORD")
    totp_secret = os.environ.get("ANGEL_TOTP_SECRET")
    
    print(f"\nConnecting as: {client_id}")
    print("API Key:", api_key[:4] + "***")
    
    # Generate TOTP
    totp = pyotp.TOTP(totp_secret).now()
    print(f"TOTP Generated: {totp}")
    
    smart_api = SmartConnect(api_key=api_key)
    
    # Login WITH TOTP
    data = smart_api.generateSession(
        clientCode=client_id,
        password=password,
        totp=totp
    )
    
    if data.get('status'):
        print("\n✅ LOGIN SUCCESSFUL!")
        print(f"   JWT Token: {data['data']['jwtToken'][:30]}...")
        print(f"   Refresh Token: {data['data']['refreshToken'][:30]}...")
        
        # Get profile
        profile = smart_api.getProfile(data['data']['refreshToken'])
        if profile.get('status'):
            print(f"\n📊 Profile:")
            print(f"   Name: {profile['data'].get('name', 'N/A')}")
            print(f"   Email: {profile['data'].get('email', 'N/A')}")
        
        # Get feed token for WebSocket
        feed_token = smart_api.getfeedToken()
        print(f"\n📡 Feed Token: {feed_token[:20]}..." if feed_token else "❌ No feed token")
        
        print("\n" + "=" * 50)
        print("✅ Angel One connection is WORKING!")
        print("   You can now use PCR data in your bot.")
        print("=" * 50)
        
    else:
        print(f"\n❌ LOGIN FAILED: {data.get('message', 'Unknown error')}")
        print("\nPossible issues:")
        print("1. Wrong password")
        print("2. 2FA/TOTP is enabled (need to add TOTP code)")
        print("3. Account not activated")
        print("4. Wrong client ID")
        
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nCheck:")
    print("1. Internet connection")
    print("2. Credentials are correct")
    print("3. Angel One account is active")
