# Real-Time Live Trading System

A comprehensive cryptocurrency trading system that integrates multiple technical analysis modules for robust signal generation and automated trading.

## 🚀 Features

### Core Components
- **Backtesting System**: Historical strategy testing with comprehensive performance metrics
- **Support/Resistance Detector**: Real-time identification of key price levels
- **Fibonacci Detector**: Dynamic retracement and extension level calculation
- **Divergence Detector**: RSI/MACD divergence analysis for reversal signals
- **Multi-Strategy Manager**: Momentum, Mean Reversion, Pairs Trading, and Divergence strategies
- **Real-Time Data Streaming**: Live OHLCV data from Binance
- **Signal Validation**: Multi-condition confirmation for high-confidence trades
- **Performance Tracking**: Comprehensive metrics and visualization
- **Alert System**: Telegram notifications for trading signals

### Signal Generation Logic
The system generates trading signals only when **2 or more conditions** are met:

1. **Price at Support/Resistance**: Within 2% of identified levels
2. **Fibonacci Level Interaction**: Price at key retracement levels (23.6%, 38.2%, 50%, 61.8%)
3. **Divergence Confirmation**: RSI or MACD divergence with price
4. **Strategy Signals**: Momentum, mean reversion, or other strategy confirmations

## 📁 Project Structure

```
AITRADE/
├── real_live_trading_system.py      # Main trading system
├── backtest_system.py               # Historical backtesting
├── live_support_resistance_detector.py  # Support/resistance detection
├── live_fibonacci_detector.py       # Fibonacci level analysis
├── divergence_detector.py           # Divergence analysis
├── strategies.py                    # Trading strategies
├── config.py                        # Configuration settings
├── test_real_time_system.py         # System testing
├── DATA/                            # Historical data storage
├── models/                          # ML models
└── requirements.txt                 # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Binance API credentials (optional for live trading)
- Telegram Bot Token (optional for alerts)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AITRADE
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables** (create `.env` file)
   ```env
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

## 🚀 Usage

### 1. Testing the System
Run the comprehensive test suite to verify all components:

```bash
python test_real_time_system.py
```

This will test:
- Backtesting system functionality
- Support/resistance detection
- Fibonacci level calculation
- Divergence analysis
- Strategy signal generation
- Integrated system performance

### 2. Running Backtests
Test strategies on historical data:

```bash
python backtest_system.py
```

**Configuration Options:**
- Initial capital: $100,000 (default)
- Commission: 0.1% (default)
- Date range: 2024-01-01 to 2024-12-31 (default)
- Trading pairs: BTC/USDT, ETH/USDT, BNB/USDT, etc.

### 3. Live Trading System
Start the real-time trading system:

```bash
python real_live_trading_system.py
```

**Features:**
- Real-time data streaming from Binance
- Multi-condition signal validation
- Optional live trading execution
- Performance tracking and alerts
- SQLite database for signal storage

## 📊 System Architecture

### Data Flow
```
Live Market Data → Technical Analysis → Signal Generation → Validation → Execution
     ↓                    ↓                    ↓              ↓           ↓
  OHLCV Data    Support/Resistance    Multi-Condition   Backtesting   Trade Execution
                 Fibonacci Levels     Signal Logic      Validation    (Optional)
                 Divergence Analysis  Strategy Signals  Confidence    Performance Tracking
```

### Signal Validation Process
1. **Data Collection**: Fetch live OHLCV data from Binance
2. **Technical Analysis**: Apply all detectors and strategies
3. **Condition Checking**: Validate proximity to support/resistance, Fibonacci levels
4. **Signal Generation**: Combine multiple conditions for high-confidence signals
5. **Backtesting**: Optional historical validation
6. **Execution**: Execute trades if live trading is enabled

## 🔧 Configuration

### Trading Parameters (`config.py`)
```python
TRADING_CONFIG = {
    'max_position_size': 0.1,      # 10% of capital per trade
    'risk_per_trade': 0.02,        # 2% risk per trade
    'stop_loss': 0.02,             # 2% stop loss
    'take_profit': 0.04,           # 4% take profit
    'commission': 0.001            # 0.1% commission
}
```

### Strategy Parameters
```python
STRATEGY_CONFIG = {
    'momentum': {
        'sma_short': 20,
        'sma_long': 50,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    },
    'mean_reversion': {
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14
    },
    'divergence': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'min_candles': 20,
        'swing_threshold': 0.02
    }
}
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Backtesting Results
- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Yearly return rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win/Loss**: Average profit and loss per trade

### Real-Time Metrics
- **Total Signals**: Number of signals generated
- **Executed Trades**: Number of trades executed
- **Current P&L**: Real-time profit/loss
- **Signal Confidence**: Average signal confidence level

## 🔔 Alert System

### Telegram Notifications
The system can send real-time alerts via Telegram:

```
🚨 TRADING SIGNAL ALERT 🚨

Symbol: BTC/USDT
Signal: BUY
Price: $45,234.56
Strength: 0.85
Confidence: 0.92
Time: 2024-01-15 14:30:25

Conditions Met:
• Near Support $44,800.00
• At 61.8% Fib Level
• Bullish RSI Divergence
• Momentum Signal

Support/Resistance:
• Support Zones: 3
• Resistance Zones: 2

Fibonacci Levels:
• Active Levels: 5

Divergence:
• Signals: 1
```

## 📊 Data Storage

### SQLite Database
The system uses SQLite for persistent storage:

- **trading_signals**: All generated signals with metadata
- **market_conditions**: Historical market analysis data

### File Outputs
- **Backtest Results**: JSON files with detailed performance data
- **Charts**: PNG files with equity curves, drawdowns, and P&L distributions
- **Logs**: Detailed system logs for debugging

## 🧪 Testing Results

Recent test results show:
- ✅ Backtest System: PASSED
- ✅ Support/Resistance Detector: PASSED  
- ✅ Fibonacci Detector: PASSED
- ✅ Divergence Detector: PASSED
- ✅ Strategy Manager: PASSED
- ⚠️ Integrated System: Minor issues (83.3% pass rate)

## 🚨 Risk Management

### Built-in Protections
1. **Position Sizing**: Maximum 10% of capital per trade
2. **Stop Loss**: Automatic 2% stop loss on all positions
3. **Take Profit**: 4% take profit targets
4. **Multi-Condition Validation**: Requires 2+ conditions for signals
5. **Confidence Thresholds**: Minimum 70% confidence for execution
6. **Paper Trading Mode**: Test without real money

### Risk Warnings
- **Cryptocurrency trading is highly volatile**
- **Past performance does not guarantee future results**
- **Always test strategies thoroughly before live trading**
- **Never risk more than you can afford to lose**

## 🔧 Customization

### Adding New Strategies
1. Create strategy class in `strategies.py`
2. Implement `generate_signals()` method
3. Add to `StrategyManager` in main system
4. Update configuration parameters

### Adding New Detectors
1. Create detector class
2. Implement analysis methods
3. Integrate with main trading system
4. Add to signal validation logic

### Modifying Signal Logic
Edit the `validate_signal_conditions()` method in `real_live_trading_system.py` to:
- Change condition weights
- Add new validation rules
- Modify confidence calculations
- Adjust signal thresholds

## 📚 API Reference

### Main Classes

#### `RealLiveTradingSystem`
Main trading system class with methods:
- `start()`: Start real-time trading
- `stop()`: Stop trading system
- `analyze_market_conditions()`: Analyze current market state
- `generate_trading_signal()`: Generate trading signals
- `execute_trade()`: Execute trades

#### `BacktestSystem`
Historical testing system with methods:
- `run_backtest()`: Run historical backtest
- `calculate_performance_metrics()`: Calculate performance stats
- `generate_charts()`: Create visualization charts

#### `LiveSupportResistanceDetector`
Support/resistance detection with methods:
- `identify_zones()`: Find support/resistance zones
- `check_zone_breaks()`: Monitor zone breakouts

#### `LiveFibonacciDetector`
Fibonacci analysis with methods:
- `update_fibonacci_levels()`: Calculate Fibonacci levels
- `check_price_alerts()`: Monitor level interactions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🆘 Support

For issues and questions:
1. Check the test results first
2. Review the logs in `real_live_trading_system.log`
3. Verify your API credentials and configuration
4. Test with paper trading before live execution

## 🔮 Future Enhancements

- [ ] Machine learning signal enhancement
- [ ] Portfolio optimization algorithms
- [ ] Advanced risk management features
- [ ] Web-based dashboard
- [ ] Multi-exchange support
- [ ] Options and futures trading
- [ ] Social sentiment analysis
- [ ] News impact analysis

---

**Disclaimer**: This system is for educational and research purposes. Cryptocurrency trading involves substantial risk. Always test thoroughly and never invest more than you can afford to lose. 