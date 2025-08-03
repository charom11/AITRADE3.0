"""
Configuration file for the Mini Hedge Fund
Defines trading parameters, risk management, and strategy settings
"""

import os
from datetime import datetime, timedelta

# Trading Configuration
TRADING_CONFIG = {
    'initial_capital': 100000,  # Starting capital in USD
    'max_position_size': 0.1,   # Maximum 10% of portfolio per position
    'max_drawdown': 0.15,       # Maximum 15% drawdown before stopping
    'risk_per_trade': 0.02,     # 2% risk per trade
    'commission': 0.001,        # 0.1% commission per trade
    'slippage': 0.0005,         # 0.05% slippage
}

# Asset Universe - Multiple markets for diversification
ASSETS = {
    'equities': {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'IWM': 'Russell 2000 ETF',
        'EFA': 'International Developed Markets ETF',
        'EEM': 'Emerging Markets ETF'
    },
    'commodities': {
        'GLD': 'Gold ETF',
        'SLV': 'Silver ETF',
        'USO': 'Oil ETF',
        'UNG': 'Natural Gas ETF'
    },
    'bonds': {
        'TLT': '20+ Year Treasury Bond ETF',
        'IEF': '7-10 Year Treasury Bond ETF',
        'LQD': 'Investment Grade Corporate Bond ETF'
    },
    'currencies': {
        'UUP': 'US Dollar Bull ETF',
        'FXE': 'Euro ETF',
        'FXY': 'Japanese Yen ETF'
    }
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'momentum': {
        'short_window': 20,      # Short-term moving average
        'long_window': 200,      # Long-term moving average
        'rsi_period': 14,        # RSI period
        'rsi_overbought': 70,    # RSI overbought threshold
        'rsi_oversold': 30,      # RSI oversold threshold
    },
    'mean_reversion': {
        'bb_period': 20,         # Bollinger Bands period
        'bb_std': 2,             # Bollinger Bands standard deviation
        'atr_period': 14,        # Average True Range period
        'rsi_overbought': 70,    # RSI overbought threshold
        'rsi_oversold': 30,      # RSI oversold threshold
    },
    'pairs_trading': {
        'lookback_period': 252,  # 1 year for correlation calculation
        'correlation_threshold': 0.7,  # Minimum correlation for pairs
        'z_score_threshold': 2.0,      # Z-score threshold for entry
    },
    'divergence': {
        'rsi_period': 14,        # RSI calculation period
        'macd_fast': 12,         # MACD fast EMA period
        'macd_slow': 26,         # MACD slow EMA period
        'macd_signal': 9,        # MACD signal line period
        'min_candles': 50,       # Minimum candles to analyze
        'swing_threshold': 0.02, # Minimum swing size to consider (2%)
        'confirm_with_support_resistance': True,  # Confirm with S/R levels
        'confirm_with_candlestick': True,        # Confirm with candlestick patterns
        'risk_per_trade': 0.025, # 2.5% risk per trade (higher for divergence)
    }
}

# Risk Management
RISK_CONFIG = {
    'max_correlation': 0.8,      # Maximum correlation between positions
    'max_sector_exposure': 0.3,  # Maximum 30% exposure to any sector
    'stop_loss': 0.05,           # 5% stop loss
    'take_profit': 0.15,         # 15% take profit
    'trailing_stop': 0.03,       # 3% trailing stop
}

# Performance Tracking
PERFORMANCE_CONFIG = {
    'benchmark': 'SPY',          # Benchmark for comparison
    'risk_free_rate': 0.02,      # 2% risk-free rate
    'rebalance_frequency': 'weekly',  # Portfolio rebalancing frequency
    'performance_window': 252,   # 1 year rolling window
}

# Data Configuration
DATA_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'data_source': 'yfinance',   # Data source
    'update_frequency': 'daily', # Data update frequency
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'trading_log.txt',
    'performance_log': 'performance_log.csv',
}

# Environment Variables
ENV_VARS = {
    'API_KEY': os.getenv('API_KEY', ''),
    'SECRET_KEY': os.getenv('SECRET_KEY', ''),
    'PAPER_TRADING': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
} 