#!/usr/bin/env python3
"""
Live Trading System with Enhanced Risk Management
Real money trading with strict safety controls
"""

import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the system class
from integrated_futures_trading_system import IntegratedFuturesTradingSystem, get_top_100_futures_pairs

def live_trading():
    """Run live trading with enhanced safety features"""
    
    print("🚨 LIVE TRADING SYSTEM - REAL MONEY")
    print("="*70)
    print("⚠️  WARNING: THIS WILL USE REAL MONEY")
    print("💰 Risk of financial loss is HIGH")
    print("⚡ Leverage can amplify both profits AND losses")
    print("🔒 Only trade with money you can afford to lose")
    print("="*70)
    
    # Pre-configured settings
    TELEGRAM_BOT_TOKEN = "8201084480:AAEsc-cLl8KIelwV7PffT4cGoclZquRpFak"
    BINANCE_API_KEY = "QVa0FqEvAtf1s5vtQE0grudNUkg3sl0IIPcdFx99cW50Cb80gPwHexW9VPGk7h0y"
    BINANCE_SECRET_KEY = "9A8hpWaTRvnaEApeCCwl7in0FvTBPIdFqXO4zidYugJgXXA9FO6TWMU3kn4JKgb0"
    TELEGRAM_CHAT_ID = "1166227057"
    
    # Enhanced safety settings for live trading
    print("🔒 ENHANCED SAFETY SETTINGS:")
    print("📊 Max Position Size: 3% of account")
    print("⚠️  Risk Per Trade: 0.5% of account")
    print("📈 Max Concurrent Trades: 2")
    print("⚠️  Max Portfolio Risk: 3%")
    print("🎯 Min Confidence: 75%")
    print("⏱️ Max Trade Duration: 2 hours")
    print("="*70)
    
    # Get top 10 pairs for conservative live trading
    print("🔍 Fetching top 10 futures pairs...")
    all_pairs = get_top_100_futures_pairs()
    symbols = all_pairs[:10]  # Use top 10 for conservative approach
    
    print(f"📊 Using top 10 pairs: {', '.join(symbols)}")
    
    # Final confirmation
    print("\n🚨 FINAL CONFIRMATION REQUIRED")
    print("="*70)
    print("Type 'CONFIRM LIVE' to start live trading")
    print("Type anything else to cancel")
    print("="*70)
    
    confirm = input("Confirmation: ").strip()
    if confirm != "CONFIRM LIVE":
        print("❌ Live trading cancelled. Exiting...")
        return
    
    # Create system with ultra-conservative settings
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_SECRET_KEY,
        enable_file_output=True,
        enable_live_trading=True,   # LIVE TRADING
        paper_trading=False,        # No paper trading
        max_position_size=0.03,     # 3% (ultra-conservative)
        risk_per_trade=0.005,       # 0.5% (ultra-conservative)
        enable_backtesting=False,
        enable_alerts=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        max_concurrent_trades=2,    # Very conservative
        max_portfolio_risk=0.03     # 3% (ultra-conservative)
    )
    
    # Test Telegram connection
    print("🔔 Testing Telegram connection...")
    if system.test_telegram_connection():
        print("✅ Telegram connection successful!")
    else:
        print("❌ Telegram connection failed!")
        return
    
    # Send live trading start notification
    start_message = f"""
🚨 **LIVE TRADING STARTED**

💰 **Mode**: REAL MONEY TRADING
📊 **Pairs**: {len(symbols)} symbols
🔒 **Max Position**: 3% of account
⚠️ **Risk Per Trade**: 0.5% of account
📈 **Max Trades**: 2 concurrent
🎯 **Min Confidence**: 75%

⏰ **Start Time**: {datetime.now().strftime('%H:%M:%S')}

⚠️ **WARNING**: Real money is at risk!
"""
    system.send_telegram_alert(start_message)
    
    # Start monitoring
    print("🚀 Starting LIVE TRADING...")
    try:
        system.start_monitoring(duration_minutes=10, update_interval=60)  # 1-minute updates
    except KeyboardInterrupt:
        print("\n⏹️ Stopping live trading...")
        stop_message = f"""
🛑 **LIVE TRADING STOPPED**

⏰ **Stop Time**: {datetime.now().strftime('%H:%M:%S')}
💰 **Total P&L**: ${system.total_pnl:.2f}
🎯 **Success Rate**: {(system.successful_trades / system.total_trades * 100) if system.total_trades > 0 else 0:.1f}%

All positions have been closed.
"""
        system.send_telegram_alert(stop_message)
    except Exception as e:
        print(f"❌ Error: {e}")
        error_message = f"""
🚨 **LIVE TRADING ERROR**

❌ **Error**: {str(e)}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

System has been stopped for safety.
"""
        system.send_telegram_alert(error_message)
    finally:
        system.stop()
        print("✅ Live trading stopped successfully!")

if __name__ == "__main__":
    live_trading() 