#!/usr/bin/env python3
"""
Automated runner for integrated futures trading system
Runs with pre-configured settings without user input
"""

import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the system class
from integrated_futures_trading_system import IntegratedFuturesTradingSystem, get_top_100_futures_pairs

def auto_run():
    """Run the system with pre-configured settings"""
    
    print("🚀 AUTO-RUN: INTEGRATED FUTURES TRADING SYSTEM")
    print("="*60)
    print("📱 Telegram Chat ID: 1166227057")
    print("🔧 Mode: LIVE TRADING (REAL MONEY)")
    print("⚠️  WARNING: HIGH RISK - REAL MONEY WILL BE USED")
    print("⏱️ Duration: 5 minutes")
    print("📊 Pairs: Top 20 (for faster testing)")
    print("🔔 Alerts: Enabled")
    print("="*60)
    
    # Pre-configured settings
    TELEGRAM_BOT_TOKEN = "8201084480:AAEsc-cLl8KIelwV7PffT4cGoclZquRpFak"
    BINANCE_API_KEY = "QVa0FqEvAtf1s5vtQE0grudNUkg3sl0IIPcdFx99cW50Cb80gPwHexW9VPGk7h0y"
    BINANCE_SECRET_KEY = "9A8hpWaTRvnaEApeCCwl7in0FvTBPIdFqXO4zidYugJgXXA9FO6TWMU3kn4JKgb0"
    TELEGRAM_CHAT_ID = "1166227057"
    
    # Get top 20 pairs for faster testing
    print("🔍 Fetching top 20 futures pairs...")
    all_pairs = get_top_100_futures_pairs()
    symbols = all_pairs[:20]  # Use top 20 for faster testing
    
    print(f"📊 Using top 20 pairs: {', '.join(symbols)}")
    
    # Create system with pre-configured settings
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_SECRET_KEY,
        enable_file_output=True,
        enable_live_trading=True,   # LIVE TRADING
        paper_trading=False,        # No paper trading
        max_position_size=0.05,     # 5% (reduced for live trading)
        risk_per_trade=0.01,        # 1% (reduced for live trading)
        enable_backtesting=False,
        enable_alerts=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        max_concurrent_trades=3,    # Reduced for live trading
        max_portfolio_risk=0.05     # 5% (reduced for live trading)
    )
    
    # Test Telegram connection
    print("🔔 Testing Telegram connection...")
    if system.test_telegram_connection():
        print("✅ Telegram connection successful!")
    else:
        print("❌ Telegram connection failed, but continuing...")
    
    # Final confirmation for live trading
    print("\n⚠️  FINAL WARNING: LIVE TRADING MODE")
    print("="*60)
    print("💰 REAL MONEY WILL BE USED")
    print("📊 Max Position Size: 5% of account")
    print("⚠️  Risk Per Trade: 1% of account")
    print("📈 Max Concurrent Trades: 3")
    print("⚠️  Max Portfolio Risk: 5%")
    print("="*60)
    
    confirm = input("Type 'LIVE' to confirm live trading: ").strip()
    if confirm != "LIVE":
        print("❌ Live trading cancelled. Exiting...")
        return
    
    print("🚀 Starting LIVE TRADING for 5 minutes...")
    try:
        system.start_monitoring(duration_minutes=5, update_interval=30)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        system.stop()
        print("✅ System stopped successfully!")

if __name__ == "__main__":
    auto_run() 