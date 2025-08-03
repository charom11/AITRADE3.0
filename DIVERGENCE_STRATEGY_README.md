# 🔍 Divergence Detection Strategy

## Overview

The Divergence Detection Strategy is a sophisticated technical analysis tool that identifies **Class A divergence** between price and momentum indicators (RSI and MACD). This strategy is designed to detect potential trend reversals with high accuracy by filtering out noise and focusing only on the most reliable divergence patterns.

## 🎯 What is Class A Divergence?

### Bullish Divergence
- **Price**: Forms lower lows
- **Indicator**: Forms higher lows
- **Signal**: Potential upward reversal

### Bearish Divergence
- **Price**: Forms higher highs
- **Indicator**: Forms lower highs
- **Signal**: Potential downward reversal

## ✨ Key Features

### 🔧 Technical Indicators
- **RSI (14)**: Relative Strength Index with 14-period calculation
- **MACD (12, 26, 9)**: Moving Average Convergence Divergence with standard parameters
- **Swing Point Detection**: Identifies significant highs and lows with configurable thresholds

### 🛡️ Noise Filtering
- **Minimum Swing Threshold**: 2% default (configurable)
- **Minimum Candles**: 50 candles required for analysis
- **Significant Swings Only**: Filters out minor price movements

### ✅ Signal Confirmation
- **Support/Resistance Levels**: Confirms signals near key levels
- **Candlestick Patterns**: Detects hammer, doji, shooting star patterns
- **Strength Calculation**: Measures divergence strength (0-1 scale)
- **Timeframe Detection**: Automatically detects and displays data timeframe

## 📁 Files

### Core Files
- `divergence_detector.py` - Main divergence detection engine
- `strategies.py` - Contains `DivergenceStrategy` class
- `config.py` - Configuration parameters
- `test_divergence_strategy.py` - Testing and demonstration script

### Integration
- Integrated into the main trading system via `main.py`
- Compatible with existing strategy framework
- Works with real cryptocurrency data from Binance

## 🚀 Usage Examples

### Basic Usage

```python
from divergence_detector import DivergenceDetector
import pandas as pd

# Initialize detector
detector = DivergenceDetector(
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    min_candles=50,
    swing_threshold=0.02
)

# Load your OHLCV data
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Analyze divergence
analysis = detector.analyze_divergence(data, confirm_signals=True)

# Print results
detector.print_analysis_report(analysis)
```

### Strategy Integration

```python
from strategies import DivergenceStrategy

# Initialize strategy
strategy = DivergenceStrategy()

# Generate signals
signals = strategy.generate_signals(data)

# Get detailed analysis
analysis = strategy.get_divergence_analysis(data)

# Calculate position size
position_size = strategy.calculate_position_size(
    signal=1,  # Bullish signal
    price=50000,
    portfolio_value=100000
)
```

### Testing with Real Data

```bash
# Test with real cryptocurrency data
python test_divergence_strategy.py

# Test standalone detector
python divergence_detector.py
```

## ⚙️ Configuration

### Strategy Parameters

```python
STRATEGY_CONFIG = {
    'divergence': {
        'rsi_period': 14,        # RSI calculation period
        'macd_fast': 12,         # MACD fast EMA period
        'macd_slow': 26,         # MACD slow EMA period
        'macd_signal': 9,        # MACD signal line period
        'min_candles': 50,       # Minimum candles to analyze
        'swing_threshold': 0.02, # Minimum swing size (2%)
        'confirm_with_support_resistance': True,
        'confirm_with_candlestick': True,
        'risk_per_trade': 0.025, # 2.5% risk per trade
    }
}
```

### Customization Options

- **Swing Threshold**: Adjust sensitivity (0.01-0.05 recommended)
- **Minimum Candles**: Increase for more reliable signals
- **Confirmation**: Enable/disable support/resistance and candlestick confirmation
- **Risk Management**: Adjust position sizing parameters

## 📊 Signal Output

### Analysis Results

```python
{
    'total_signals': 2,
    'rsi_signals': 1,
    'macd_signals': 1,
    'timeframe': '1h',
    'data_points': 200,
    'date_range': {
        'start': '2024-01-01 00:00:00',
        'end': '2024-01-08 23:00:00',
        'duration': '7 days 23:00:00'
    },
    'signals': [
        {
            'type': 'bullish',
            'divergence_class': 'A',
            'indicator': 'rsi',
            'strength': 0.85,
            'signal_date': '2024-01-15 10:00:00',
            'price_low1_val': 45000,
            'price_low2_val': 44000,
            'indicator_low1_val': 30.5,
            'indicator_low2_val': 35.2,
            'support_confirmation': True,
            'candlestick_pattern': 'hammer'
        }
    ],
    'current_price': 44500,
    'current_rsi': 45.2,
    'current_macd': -0.5
}
```

### Signal DataFrame

| Column | Description |
|--------|-------------|
| `timestamp` | Signal date/time |
| `type` | 'bullish' or 'bearish' |
| `indicator` | 'rsi' or 'macd' |
| `strength` | Divergence strength (0-1) |
| `price_level` | Key price level |
| `indicator_level` | Key indicator level |
| `support_resistance_confirmed` | S/R confirmation |
| `candlestick_confirmed` | Pattern confirmation |
| `timeframe` | Detected timeframe (e.g., '1h', '4h', '1d') |
| `signal` | 1 (buy) or -1 (sell) |

## 🎯 Trading Strategy

### Signal Interpretation

1. **Strong Signals** (Strength > 0.7)
   - High confidence divergence
   - Multiple confirmations
   - Larger position sizes recommended

2. **Medium Signals** (Strength 0.3-0.7)
   - Moderate confidence
   - Standard position sizing
   - Wait for additional confirmation

3. **Weak Signals** (Strength < 0.3)
   - Low confidence
   - Smaller position sizes
   - Consider ignoring

### Risk Management

- **Position Sizing**: 2.5% risk per trade (configurable)
- **Maximum Position**: 12% of portfolio
- **Stop Loss**: Based on divergence levels
- **Take Profit**: 2:1 or 3:1 risk-reward ratio

### Best Practices

1. **Multiple Timeframes**: Confirm signals across different timeframes
2. **Volume Confirmation**: Look for volume spikes at divergence points
3. **Market Context**: Consider overall market trend
4. **Risk Management**: Always use stop losses
5. **Backtesting**: Test strategies before live trading

## 🔍 Testing Results

### Recent Performance (Sample Data)

| Symbol | Timeframe | Signals | Bullish | Bearish | Success Rate |
|--------|-----------|---------|---------|---------|--------------|
| BTCUSDT | 1h | 0 | 0 | 0 | N/A |
| ETHUSDT | 1h | 2 | 0 | 2 | TBD |
| XRPUSDT | 1h | 0 | 0 | 0 | N/A |
| ADAUSDT | 1h | 1 | 1 | 0 | TBD |
| SOLUSDT | 1h | 1 | 1 | 0 | TBD |

### Signal Examples

**ETHUSDT Bearish Divergence (MACD)**
- Timeframe: 1h
- Strength: 1.00 (Maximum)
- Price: $3,799 → $3,877 (Higher high)
- MACD: 1.94 → -1.79 (Lower high)
- Pattern: Doji confirmation
- Date: 2025-07-29

**ADAUSDT Bullish Divergence (MACD)**
- Timeframe: 1h
- Strength: 0.04 (Weak)
- Price: $0.783 → $0.757 (Lower low)
- MACD: -0.01 → -0.01 (Higher low)
- S/R Confirmed: True
- Pattern: Doji confirmation

## 🚨 Important Notes

### Limitations
- **Lagging Indicator**: Divergence signals may appear late
- **False Signals**: Not all divergences lead to reversals
- **Market Conditions**: Works best in trending markets
- **Timeframe Dependency**: Results vary by timeframe

### Safety Warnings
- **Paper Trading**: Test thoroughly before live trading
- **Risk Management**: Never risk more than you can afford to lose
- **Market Analysis**: Use as part of comprehensive analysis
- **Professional Advice**: Consult financial advisors for investment decisions

## 🔧 Installation & Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

### Data Requirements
- OHLCV data with columns: ['open', 'high', 'low', 'close', 'volume']
- Minimum 50 candles for analysis
- Clean data without gaps

### Integration Steps
1. Copy `divergence_detector.py` to your project
2. Import `DivergenceStrategy` from `strategies.py`
3. Configure parameters in `config.py`
4. Test with historical data
5. Integrate into your trading system

## 📈 Future Enhancements

### Planned Features
- **Multiple Indicators**: Add Stochastic, Williams %R
- **Advanced Patterns**: Hidden divergence detection
- **Machine Learning**: ML-enhanced signal filtering
- **Real-time Alerts**: Live signal notifications
- **Performance Analytics**: Detailed backtesting reports

### Customization Options
- **Custom Indicators**: Add your own indicators
- **Signal Filters**: Custom filtering rules
- **Alert System**: Email/SMS notifications
- **Dashboard**: Web-based monitoring interface

## 🤝 Contributing

### Development
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

### Testing
- Run `python test_divergence_strategy.py`
- Verify with real market data
- Check signal accuracy
- Validate risk management

## 📞 Support

### Documentation
- Code comments and docstrings
- Example usage in test files
- Configuration guide

### Issues
- Report bugs via GitHub issues
- Include sample data and error messages
- Provide detailed reproduction steps

---

**⚠️ Disclaimer**: This strategy is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly and use proper risk management. 