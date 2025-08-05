# Integrated Futures Trading System

A comprehensive futures trading system that integrates Binance Futures signals with support/resistance detection, divergence analysis, and strategy management, featuring intelligent file prevention to avoid duplicates.

## 🚀 Features

### Core Integration
- **Binance Futures API Integration**: Real-time futures data from Binance
- **Support/Resistance Detection**: Live detection of key price levels
- **Divergence Analysis**: Class A bullish/bearish divergence detection
- **Strategy Management**: Multiple trading strategies (Momentum, Mean Reversion, Divergence, Support/Resistance)
- **Signal Combination**: Intelligent combination of multiple analysis methods

### File Management & Prevention
- **Duplicate Prevention**: Prevents creation of identical files
- **Content Hash Verification**: MD5 hash checking for content integrity
- **Safe File Operations**: Automatic directory creation and error handling
- **File Registry**: Tracks all created files with metadata
- **Cleanup Utilities**: Automatic cleanup of old files

### Risk Management
- **Confidence Scoring**: 0-1 scale confidence for each signal
- **Risk Assessment**: Risk score calculation based on signal strength
- **Position Sizing**: Dynamic position size calculation
- **Leverage Suggestions**: Appropriate leverage recommendations
- **Stop Loss/Take Profit**: Automatic calculation of risk levels

## 📁 File Structure

```
AITRADE/
├── integrated_futures_trading_system.py  # Main integrated system
├── file_manager.py                       # File management utility
├── test_integrated_system.py             # Test suite
├── INTEGRATED_SYSTEM_README.md           # This file
├── binance_futures_signals.py            # Binance futures API client
├── live_support_resistance_detector.py   # Support/resistance detection
├── divergence_detector.py                # Divergence analysis
├── strategies.py                         # Trading strategies
└── integrated_signals/                   # Output directory (created automatically)
    ├── .file_registry.json              # File tracking registry
    └── integrated_signal_*.json         # Generated signals
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install pandas numpy requests ccxt matplotlib
```

### Optional: Binance API Setup
1. Create a Binance account
2. Generate API keys with futures trading permissions
3. Set appropriate leverage limits
4. **Important**: Only use API keys with read permissions for safety

## 🚀 Quick Start

### Basic Usage
```python
from integrated_futures_trading_system import IntegratedFuturesTradingSystem

# Create system
system = IntegratedFuturesTradingSystem(
    symbols=['BTCUSDT', 'ETHUSDT'],
    enable_file_output=True
)

# Start monitoring
system.start_monitoring(duration_minutes=10)
```

### Advanced Usage
```python
# With API credentials
system = IntegratedFuturesTradingSystem(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    api_key='your_api_key',
    api_secret='your_api_secret',
    enable_file_output=True,
    output_directory='custom_signals'
)

# Custom monitoring
system.start_monitoring(
    duration_minutes=30,
    update_interval=60  # Check every minute
)
```

## 📊 System Components

### 1. IntegratedFuturesTradingSystem
Main system class that orchestrates all components:

- **Data Fetching**: Real-time futures data from Binance
- **Signal Generation**: Multiple analysis methods
- **Signal Combination**: Intelligent signal fusion
- **File Management**: Automatic file handling with prevention
- **Risk Management**: Comprehensive risk assessment

### 2. FileManager
Intelligent file management with prevention features:

```python
from file_manager import FileManager

fm = FileManager("output_directory")

# Safe save operations
fm.safe_save_json(data, "signal.json")  # Prevents duplicates
fm.safe_save_csv(df, "data.csv")        # CSV with prevention
fm.safe_save_text(content, "log.txt")   # Text with prevention

# File tracking
files = fm.list_created_files()
fm.print_summary()
fm.cleanup_old_files(max_age_hours=24)
```

### 3. Signal Components

#### Futures Signals
- RSI, MACD, Moving Averages
- Funding rate analysis
- Open interest tracking
- Volume analysis

#### Support/Resistance Detection
- Swing high/low detection
- Multiple touch confirmations
- Volume spike analysis
- Zone break detection

#### Divergence Analysis
- Class A bullish/bearish divergence
- RSI and MACD divergence
- Signal confirmation with patterns
- Strength calculation

#### Strategy Management
- Momentum Strategy
- Mean Reversion Strategy
- Divergence Strategy
- Support/Resistance Strategy

## 📈 Signal Output Format

Each generated signal includes:

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-01T12:00:00",
  "signal_type": "LONG",
  "confidence": 0.85,
  "price": 45000.0,
  "stop_loss": 42750.0,
  "take_profit": 47250.0,
  "risk_score": 0.15,
  "position_size": 0.085,
  "leverage_suggestion": 2.0,
  "funding_rate": 0.0001,
  "open_interest": 1000000,
  "market_regime": "bullish_funding",
  "futures_signal": {...},
  "support_resistance_signal": {...},
  "divergence_signal": {...},
  "strategy_signals": {...}
}
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_integrated_system.py
```

The test suite demonstrates:
- File manager functionality
- Duplicate prevention
- Integrated system initialization
- File tracking and cleanup

## ⚠️ Risk Warnings

### Futures Trading Risks
- **High Leverage Risk**: Leverage can amplify both profits AND losses
- **Liquidation Risk**: Positions can be liquidated if price moves against you
- **Funding Rate Risk**: Funding rates can impact profitability
- **Market Volatility**: Crypto markets are highly volatile

### System Limitations
- **API Rate Limits**: Respect Binance API rate limits
- **Data Quality**: Ensure reliable internet connection
- **Strategy Risk**: No strategy guarantees profits
- **Educational Purpose**: This system is for educational purposes only

## 🔧 Configuration

### System Configuration
```python
# Default configuration
DEFAULT_CONFIG = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    'timeframe': '5m',
    'update_interval': 30,  # seconds
    'min_confidence': 0.3,
    'max_position_size': 0.1,  # 10% of portfolio
    'enable_file_output': True
}
```

### Strategy Configuration
```python
# Momentum Strategy
momentum_config = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'atr_period': 14,
    'position_size_pct': 0.1
}

# Mean Reversion Strategy
mean_reversion_config = {
    'bb_period': 20,
    'bb_std': 2,
    'rsi_period': 14,
    'position_size_pct': 0.08
}
```

## 📊 Performance Monitoring

### File Management Metrics
- Total files created
- Storage usage
- Duplicate prevention rate
- Cleanup statistics

### Signal Quality Metrics
- Signal generation rate
- Confidence distribution
- Risk score analysis
- Strategy performance

## 🛡️ Security Features

### API Security
- Optional API credentials
- Read-only permissions recommended
- Rate limiting compliance
- Error handling

### File Security
- Content hash verification
- Safe file operations
- Directory traversal protection
- Error logging

## 🔄 Maintenance

### Regular Tasks
1. **Cleanup Old Files**: Remove files older than 24 hours
2. **Monitor Storage**: Check disk usage
3. **Update Strategies**: Review and adjust strategy parameters
4. **API Key Rotation**: Regularly rotate API keys

### Troubleshooting
1. **API Errors**: Check internet connection and API limits
2. **File Errors**: Verify directory permissions
3. **Memory Issues**: Reduce symbol count or update interval
4. **Performance Issues**: Optimize strategy parameters

## 📚 API Reference

### IntegratedFuturesTradingSystem

#### Methods
- `__init__(symbols, api_key, api_secret, enable_file_output, output_directory)`
- `start_monitoring(duration_minutes, update_interval)`
- `stop()`
- `_process_symbol(symbol)`
- `_save_signal_to_file(signal)`

#### Properties
- `symbols`: List of trading symbols
- `current_signals`: Current signals for each symbol
- `signal_history`: Historical signals
- `file_manager`: File manager instance

### FileManager

#### Methods
- `safe_save_json(data, filepath, check_existing, content_hash)`
- `safe_save_csv(data, filepath, check_existing)`
- `safe_save_text(content, filepath, check_existing)`
- `list_created_files(pattern)`
- `cleanup_old_files(max_age_hours, file_pattern)`
- `print_summary()`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is for educational purposes only. Use at your own risk.

## ⚡ Quick Commands

```bash
# Run the integrated system
python integrated_futures_trading_system.py

# Run tests
python test_integrated_system.py

# Check file manager
python file_manager.py

# Monitor specific symbols
python -c "
from integrated_futures_trading_system import IntegratedFuturesTradingSystem
system = IntegratedFuturesTradingSystem(['BTCUSDT'])
system.start_monitoring(5)
"
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test suite
3. Examine the logs in `integrated_futures_system.log`
4. Verify file permissions and disk space

---

**⚠️ DISCLAIMER**: This system is for educational purposes only. Futures trading involves substantial risk of loss. Only trade with money you can afford to lose. Past performance does not guarantee future results. 