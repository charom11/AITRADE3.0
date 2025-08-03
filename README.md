# 🚀 Mini Hedge Fund - Algorithmic Trading System

> **⚠️ IMPORTANT: VIEW-ONLY REPOSITORY** ⚠️
> 
> **This repository is for educational and viewing purposes ONLY.**
> 
> - ❌ **DO NOT use this code for live trading**
> - ❌ **DO NOT use this code with real money**
> - ❌ **DO NOT use this code without proper testing**
> - ✅ **This is for learning and educational purposes only**

A comprehensive algorithmic trading system that implements the concepts from the "How to create your own mini hedge fund" thread. This system demonstrates how to build a professional-grade trading platform using Python.

## 📋 Table of Contents

- [Repository Status](#repository-status)
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Trading Strategies](#trading-strategies)
- [Risk Management](#risk-management)
- [Performance Analysis](#performance-analysis)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

## 🔒 Repository Status

### Current Status: **VIEW-ONLY**
- **Purpose**: Educational demonstration and code review
- **Live Trading**: Disabled and not recommended
- **Updates**: Code may be updated for educational purposes only
- **Security**: All sensitive data and API keys are excluded via `.gitignore`

### What This Repository Contains:
- ✅ Educational trading algorithms
- ✅ Backtesting frameworks
- ✅ Risk management examples
- ✅ Performance analysis tools
- ✅ Documentation and guides

### What This Repository Does NOT Contain:
- ❌ API keys or credentials
- ❌ Live trading configurations
- ❌ Real trading data
- ❌ Production-ready code for live trading

## 🎯 Overview

This mini hedge fund system implements the core concepts from the algorithmic trading thread:

1. **Multiple Asset Classes**: Diversification across equities, commodities, bonds, and currencies
2. **Multiple Strategies**: Momentum, mean reversion, and pairs trading
3. **Risk Management**: Position sizing, stop losses, drawdown limits
4. **Performance Tracking**: Comprehensive metrics and reporting
5. **Professional Tools**: Free Python quant stack (pandas, numpy, yfinance, etc.)

## ✨ Key Features

### 🎯 Multi-Strategy Approach
- **Momentum Strategy**: Based on moving averages and RSI
- **Mean Reversion Strategy**: Using Bollinger Bands and ATR
- **Pairs Trading Strategy**: Statistical arbitrage on correlated assets

### 📊 Risk Management
- Position sizing based on signal strength
- Stop losses and take profit levels
- Maximum drawdown protection
- Correlation and sector exposure limits
- Trailing stops for profit protection

### 📈 Performance Analysis
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown analysis
- Win rate and profit factor
- Rolling performance metrics
- Benchmark comparison (vs S&P 500)

### 🔄 Live Trading Capabilities
- Real-time data fetching
- Automated signal generation
- Position management
- Performance monitoring

## 🏗️ System Architecture

```
Mini Hedge Fund
├── Data Manager (data_manager.py)
│   ├── Historical data fetching
│   ├── Technical indicators calculation
│   └── Data caching and storage
├── Strategy Manager (strategies.py)
│   ├── Momentum Strategy
│   ├── Mean Reversion Strategy
│   └── Pairs Trading Strategy
├── Portfolio Manager (portfolio_manager.py)
│   ├── Position management
│   ├── Risk management
│   └── Performance tracking
├── Performance Analyzer (performance_analyzer.py)
│   ├── Metrics calculation
│   ├── Report generation
│   └── Visualization
└── Main Controller (main.py)
    ├── System orchestration
    ├── Backtesting engine
    └── Live trading interface
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd AiTrading

# Install required packages
pip install -r requirements.txt
```

### Required Packages
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data
- `matplotlib` & `seaborn` - Visualization
- `scikit-learn` - Machine learning (for future enhancements)
- `ta` - Technical analysis indicators
- `plotly` & `dash` - Interactive dashboards
- `backtrader` - Backtesting framework
- `vectorbt` - Vectorized backtesting
- `empyrical` - Financial risk and performance metrics

## 🚀 Quick Start

### 1. Basic Backtest

```python
from main import MiniHedgeFund

# Initialize the hedge fund
fund = MiniHedgeFund(initial_capital=100000)

# Run backtest
results = fund.run_backtest(
    start_date='2022-01-01',
    end_date='2023-12-31'
)

# Generate reports
fund.generate_reports(results)
```

### 2. Live Trading Simulation

```python
# Run live trading (paper trading)
fund.run_live_trading()
```

### 3. Command Line Execution

```bash
python main.py
```

## 📊 Trading Strategies

### 1. Momentum Strategy
**Concept**: Follow the trend when price is above long-term moving average

**Signals**:
- **Long**: Price > 200-day SMA AND RSI < 70 AND 20-day SMA > 50-day SMA
- **Short**: Price < 200-day SMA AND RSI > 30 AND 20-day SMA < 50-day SMA

**Risk Management**:
- Maximum 10% position size
- 2% risk per trade
- Stop loss: 5%
- Take profit: 15%

### 2. Mean Reversion Strategy
**Concept**: Trade against extreme moves using Bollinger Bands

**Signals**:
- **Long**: Price near lower Bollinger Band AND low volatility AND RSI < 40
- **Short**: Price near upper Bollinger Band AND high volatility AND RSI > 60

**Risk Management**:
- Maximum 5% position size (tighter due to higher risk)
- 1% risk per trade
- Stop loss: 5%
- Take profit: 15%

### 3. Pairs Trading Strategy
**Concept**: Trade the spread between highly correlated assets

**Signals**:
- Find pairs with correlation > 0.7
- **Long Pair 1, Short Pair 2**: When spread Z-score < -2
- **Short Pair 1, Long Pair 2**: When spread Z-score > 2

**Risk Management**:
- Maximum 8% position size
- 1.5% risk per trade
- Equal position sizes for both legs

## 🛡️ Risk Management

### Position Sizing
- Base position size: 2% of portfolio per trade
- Adjusted by signal strength (0-100%)
- Maximum position size: 10% of portfolio
- Sector exposure limit: 30% per sector

### Stop Losses
- Fixed stop loss: 5% from entry
- Trailing stop: 3% from peak
- Take profit: 15% from entry

### Portfolio Limits
- Maximum drawdown: 15%
- Maximum correlation between positions: 0.8
- Maximum number of positions: 10

### Asset Allocation
```
Equities: 40% (SPY, QQQ, IWM, EFA, EEM)
Commodities: 25% (GLD, SLV, USO, UNG)
Bonds: 20% (TLT, IEF, LQD)
Currencies: 15% (UUP, FXE, FXY)
```

## 📈 Performance Analysis

### Key Metrics
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Yearly return rate
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return vs maximum drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Benchmark Comparison
- Compare against S&P 500 (SPY)
- Calculate alpha and beta
- Information ratio
- Tracking error

## ⚙️ Configuration

### Trading Parameters (`config.py`)

```python
TRADING_CONFIG = {
    'initial_capital': 100000,  # Starting capital
    'max_position_size': 0.1,   # 10% max position
    'max_drawdown': 0.15,       # 15% max drawdown
    'risk_per_trade': 0.02,     # 2% risk per trade
    'commission': 0.001,        # 0.1% commission
    'slippage': 0.0005,         # 0.05% slippage
}
```

### Asset Universe

```python
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
    # ... more assets
}
```

## 💡 Usage Examples

### Example 1: Custom Strategy

```python
from strategies import BaseStrategy

class CustomStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Your custom logic here
        signals = data.copy()
        signals['signal'] = 0
        
        # Example: Buy when price crosses above 50-day SMA
        signals.loc[data['close'] > data['sma_50'], 'signal'] = 1
        
        return signals

# Add to strategy manager
fund.strategy_manager.add_strategy(CustomStrategy('Custom', {}))
```

### Example 2: Custom Risk Management

```python
# Modify position sizing
def custom_position_size(signal, price, strategy, symbol):
    # Your custom position sizing logic
    base_size = portfolio_value * 0.01  # 1% base size
    
    if signal > 0.5:  # Strong signal
        return base_size * 2
    elif signal > 0.2:  # Medium signal
        return base_size
    else:
        return 0
```

### Example 3: Performance Monitoring

```python
# Get real-time performance
summary = fund.portfolio_manager.get_portfolio_summary()
print(f"Total Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")

# Get current positions
positions = fund.portfolio_manager.get_position_summary()
print(positions)
```

## 📁 File Structure

```
AiTrading/
├── main.py                 # Main execution file
├── config.py              # Configuration settings
├── data_manager.py        # Data fetching and processing
├── strategies.py          # Trading strategies
├── portfolio_manager.py   # Portfolio and risk management
├── performance_analyzer.py # Performance analysis
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data_cache/           # Cached market data
├── reports/              # Generated reports
└── trading_log.txt       # Trading logs
```

## 🔧 Advanced Features

### 1. Machine Learning Integration
```python
# Add ML-based signal generation
from sklearn.ensemble import RandomForestClassifier

class MLStrategy(BaseStrategy):
    def __init__(self):
        self.model = RandomForestClassifier()
        # Train model on historical data
```

### 2. Real-time Data Feeds
```python
# Integrate with real-time data providers
import ccxt

class RealTimeDataManager:
    def __init__(self):
        self.exchange = ccxt.binance()
        # Set up real-time data feeds
```

### 3. Web Dashboard
```python
# Create interactive dashboard
import dash
from dash import dcc, html

app = dash.Dash(__name__)
# Add dashboard components
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**⚠️ CRITICAL WARNING: VIEW-ONLY EDUCATIONAL REPOSITORY ⚠️**

### 🚫 PROHIBITED USES
- **DO NOT use this code for live trading**
- **DO NOT use this code with real money**
- **DO NOT use this code for investment decisions**
- **DO NOT use this code without proper testing**
- **DO NOT rely on this code for financial advice**

### ✅ INTENDED USES
- **Educational learning and study**
- **Code review and analysis**
- **Understanding trading algorithms**
- **Learning system architecture**
- **Academic research purposes**

### ⚖️ Legal Disclaimers
- **This software is for educational purposes ONLY**
- **Past performance does not guarantee future results**
- **No financial advice is provided**
- **No investment recommendations are made**
- **Use at your own risk and responsibility**
- **Consult with qualified financial advisors before any investment decisions**
- **Be aware of tax implications and regulatory requirements**

### 🔒 Security Notice
- **All sensitive data is excluded via .gitignore**
- **No API keys or credentials are stored in this repository**
- **No real trading data is included**
- **This is a demonstration system only**

## 📚 Learning Resources

### Books
- "Quantitative Trading" by Ernie Chan
- "Algorithmic Trading" by Ernie Chan
- "Building Algorithmic Trading Systems" by Kevin Davey

### Online Courses
- Coursera: Financial Engineering and Risk Management
- edX: Quantitative Finance
- Udemy: Algorithmic Trading with Python

### Communities
- QuantConnect Forum
- Reddit r/algotrading
- Stack Overflow (quantitative-finance tag)

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the code comments

---

**Happy Trading! 🚀📈**

*Remember: The goal is not just to maximize returns, but to achieve consistent, risk-adjusted performance over time.* 
