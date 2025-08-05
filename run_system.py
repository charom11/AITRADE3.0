#!/usr/bin/env python3
"""
Quick runner for integrated futures trading system with pre-configured settings
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function
from integrated_futures_trading_system import main

if __name__ == "__main__":
    print("🚀 Quick Start - Integrated Futures Trading System")
    print("="*60)
    print("📱 Telegram Chat ID: 1166227057")
    print("🔧 Mode: Paper Trading")
    print("⏱️ Duration: 5 minutes")
    print("📊 Pairs: Top 100")
    print("="*60)
    
    # Run the main function
    main() 