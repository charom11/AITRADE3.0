#!/usr/bin/env python3
"""
Trading Configuration
Easy way to switch between paper trading and live trading modes
"""

# =============================================================================
# TRADING MODE CONFIGURATION
# =============================================================================

# Set to True for live trading on Binance, False for paper trading (simulation)
ENABLE_LIVE_TRADING = True
PAPER_TRADING = False

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================

# Maximum position size as percentage of portfolio (0.05 = 5%)
MAX_POSITION_SIZE = 0.05

# Risk per trade as percentage of portfolio (0.01 = 1%)
RISK_PER_TRADE = 0.01

# Maximum concurrent trades
MAX_CONCURRENT_TRADES = 3

# Maximum portfolio risk (0.05 = 5%)
MAX_PORTFOLIO_RISK = 0.05

# =============================================================================
# TRADING PAIRS CONFIGURATION
# =============================================================================

# Number of top symbols to trade (by volume)
TOP_SYMBOLS_COUNT = 20

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Binance API credentials
BINANCE_API_KEY = "QVa0FqEvAtf1s5vtQE0grudNUkg3sl0IIPcdFx99cW50Cb80gPwHexW9VPGk7h0y"
BINANCE_SECRET_KEY = "9A8hpWaTRvnaEApeCCwl7in0FvTBPIdFqXO4zidYugJgXXA9FO6TWMU3kn4JKgb0"

# Telegram configuration
TELEGRAM_BOT_TOKEN = "8201084480:AAEsc-cLl8KIelwV7PffT4cGoclZquRpFak"
TELEGRAM_CHAT_ID = "1166227057"

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

# Enable file output for logging
ENABLE_FILE_OUTPUT = True

# Enable backtesting
ENABLE_BACKTESTING = False

# Enable alerts
ENABLE_ALERTS = True

# =============================================================================
# QUICK MODE SWITCHES
# =============================================================================

def enable_paper_trading():
    """Enable paper trading mode (simulation)"""
    global ENABLE_LIVE_TRADING, PAPER_TRADING
    ENABLE_LIVE_TRADING = True
    PAPER_TRADING = False
    print("🟡 PAPER TRADING MODE ENABLED - No real trades will be executed")

def enable_live_trading():
    """Enable live trading mode (real trades on Binance)"""
    global ENABLE_LIVE_TRADING, PAPER_TRADING
    ENABLE_LIVE_TRADING = True
    PAPER_TRADING = False
    print("🔴 LIVE TRADING MODE ENABLED - Real trades will be executed on Binance")

def get_trading_mode():
    """Get current trading mode"""
    if ENABLE_LIVE_TRADING and not PAPER_TRADING:
        return "LIVE_TRADING"
    elif not ENABLE_LIVE_TRADING and PAPER_TRADING:
        return "PAPER_TRADING"
    else:
        return "MIXED_MODE"

if __name__ == "__main__":
    print(f"Current trading mode: {get_trading_mode()}")
    print(f"Live trading enabled: {ENABLE_LIVE_TRADING}")
    print(f"Paper trading enabled: {PAPER_TRADING}") 