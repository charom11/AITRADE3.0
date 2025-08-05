#!/usr/bin/env python3
"""
Trading Mode Switcher
Easy way to switch between paper trading and live trading modes
"""

import os
import sys

def switch_to_paper_trading():
    """Switch to paper trading mode"""
    try:
        with open('trading_config.py', 'r') as f:
            content = f.read()
        
        # Replace the configuration
        content = content.replace('ENABLE_LIVE_TRADING = True', 'ENABLE_LIVE_TRADING = False')
        content = content.replace('PAPER_TRADING = False', 'PAPER_TRADING = True')
        
        with open('trading_config.py', 'w') as f:
            f.write(content)
        
        print("🟡 SWITCHED TO PAPER TRADING MODE")
        print("   - No real trades will be executed")
        print("   - All trades are simulations")
        print("   - Safe for testing and development")
        
    except Exception as e:
        print(f"❌ Error switching to paper trading: {e}")

def switch_to_live_trading():
    """Switch to live trading mode"""
    try:
        with open('trading_config.py', 'r') as f:
            content = f.read()
        
        # Replace the configuration
        content = content.replace('ENABLE_LIVE_TRADING = False', 'ENABLE_LIVE_TRADING = True')
        content = content.replace('PAPER_TRADING = True', 'PAPER_TRADING = False')
        
        with open('trading_config.py', 'w') as f:
            f.write(content)
        
        print("🔴 SWITCHED TO LIVE TRADING MODE")
        print("   - Real trades will be executed on Binance")
        print("   - Real money will be used")
        print("   - ⚠️  WARNING: Use at your own risk!")
        
    except Exception as e:
        print(f"❌ Error switching to live trading: {e}")

def show_current_mode():
    """Show current trading mode"""
    try:
        from trading_config import get_trading_mode, ENABLE_LIVE_TRADING, PAPER_TRADING
        
        mode = get_trading_mode()
        print(f"📊 CURRENT TRADING MODE: {mode}")
        print(f"   Live trading enabled: {ENABLE_LIVE_TRADING}")
        print(f"   Paper trading enabled: {PAPER_TRADING}")
        
        if mode == "LIVE_TRADING":
            print("   🔴 REAL TRADES WILL BE EXECUTED ON BINANCE")
        elif mode == "PAPER_TRADING":
            print("   🟡 SIMULATION MODE - NO REAL TRADES")
        else:
            print("   ⚠️  MIXED MODE - CHECK CONFIGURATION")
            
    except Exception as e:
        print(f"❌ Error reading current mode: {e}")

def main():
    """Main function"""
    print("🎯 TRADING MODE SWITCHER")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python switch_mode.py paper    - Switch to paper trading")
        print("  python switch_mode.py live     - Switch to live trading")
        print("  python switch_mode.py status   - Show current mode")
        print()
        show_current_mode()
        return
    
    command = sys.argv[1].lower()
    
    if command == "paper":
        switch_to_paper_trading()
    elif command == "live":
        switch_to_live_trading()
    elif command == "status":
        show_current_mode()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: paper, live, or status")

if __name__ == "__main__":
    main() 