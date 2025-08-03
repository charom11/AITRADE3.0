# Live Fibonacci Retracement and Extension Detector

Real-time Python trading tool that detects and visualizes Fibonacci levels from live OHLCV market data.

## 🚀 Features

### 🔁 Live Market Data
- **Multi-Exchange**: Binance, Bybit, and other CCXT exchanges
- **Real-time Streaming**: Continuous OHLCV data with configurable timeframes
- **Rolling Window**: 100-200 recent candles for analysis
- **Auto Updates**: Configurable intervals (1m, 5m, 15m, 1h, 4h, 1d)

### 📐 Fibonacci Logic
- **Auto Detection**: Identifies swing high and swing low
- **Retracement Levels**: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- **Extension Levels**: 127.2%, 161.8%, 200%
- **Price Zones**: ±0.3% buffer for noise handling
- **Swing Strength**: Calculated from price and volume

### 📊 Visualization
- **Real-time Charts**: Candlestick charts with Fibonacci overlays
- **Swing Markers**: Visual indicators for highs and lows
- **Zone Shading**: Color-coded Fibonacci zones
- **Export**: High-resolution PNG charts

### 🚨 Alerts & Output
- **Price Alerts**: When price approaches/touches levels
- **Telegram Integration**: Real-time notifications
- **Console Logging**: Detailed logging
- **CSV Export**: Fibonacci levels export
- **Status Reports**: Comprehensive status display

## 🚀 Quick Start

### Basic Usage
```python
from live_fibonacci_detector import LiveFibonacciDetector

# Create detector
detector = LiveFibonacciDetector(
    exchange_name='binance',
    symbol='BTC/USDT',
    timeframe='5m'
)

# Start monitoring
detector.start_monitoring()

# Print status
detector.print_status()

# Create chart
detector.create_chart('fibonacci_chart.png')
```

### Interactive Setup
```bash
python live_fibonacci_detector.py
```

### Testing
```bash
python test_fibonacci_detector.py
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exchange_name` | 'binance' | Exchange name |
| `symbol` | 'BTC/USDT' | Trading pair |
| `timeframe` | '5m' | Data timeframe |
| `lookback_periods` | 200 | Candles to maintain |
| `buffer_percentage` | 0.003 | Price buffer (±0.3%) |
| `min_swing_strength` | 0.6 | Minimum swing strength |

## 📊 Fibonacci Levels

### Retracement Levels
- **23.6%**: Shallow retracement
- **38.2%**: Common level
- **50.0%**: Psychological level
- **61.8%**: Golden ratio
- **78.6%**: Deep retracement

### Extension Levels
- **127.2%**: First extension
- **161.8%**: Golden ratio extension
- **200%**: Double extension

## 🔧 Core Components

### LiveFibonacciDetector Class
```python
# Key Methods
fetch_live_data() -> pd.DataFrame
detect_swing_points(data) -> Tuple[List[SwingPoint], List[SwingPoint]]
calculate_fibonacci_levels(swing_high, swing_low) -> List[FibonacciLevel]
start_monitoring(update_interval: int = 60)
check_price_alerts() -> List[Dict]
print_status()
create_chart(save_path: str = None)
export_levels_to_csv(filename: str = None)
```

### SwingPoint Dataclass
```python
@dataclass
class SwingPoint:
    point_type: str      # 'high' or 'low'
    price: float         # Price level
    timestamp: datetime  # Time of swing
    volume: float        # Volume at swing
    strength: float      # Calculated strength (0-1)
```

### FibonacciLevel Dataclass
```python
@dataclass
class FibonacciLevel:
    level_type: str              # 'retracement' or 'extension'
    percentage: float            # Fibonacci percentage
    price: float                 # Calculated price level
    zone_high: float            # Upper zone boundary
    zone_low: float             # Lower zone boundary
    strength: float             # Level strength (0-1)
    touches: int                # Number of price touches
    last_touch: Optional[datetime] = None
```

## 📈 Usage Examples

### Example 1: Basic Monitoring
```python
detector = LiveFibonacciDetector(symbol='BTC/USDT', timeframe='5m')
detector.start_monitoring(update_interval=60)
time.sleep(3600)  # Monitor for 1 hour
detector.stop_monitoring()
```

### Example 2: Historical Analysis
```python
detector = LiveFibonacciDetector(symbol='BTC/USDT')
detector.data = historical_df
detector.current_price = historical_df['close'].iloc[-1]

swing_highs, swing_lows = detector.detect_swing_points(historical_df)
if swing_highs and swing_lows:
    latest_high = max(swing_highs, key=lambda x: x.timestamp)
    latest_low = max(swing_lows, key=lambda x: x.timestamp)
    levels = detector.calculate_fibonacci_levels(latest_high, latest_low)
    
    for level in levels:
        print(f"{level.level_type.title()} {level.percentage}%: ${level.price:.4f}")
```

### Example 3: Telegram Alerts
```python
detector = LiveFibonacciDetector(
    symbol='ETH/USDT',
    timeframe='15m',
    telegram_token='YOUR_BOT_TOKEN',
    telegram_chat_id='YOUR_CHAT_ID'
)
detector.start_monitoring()
```

## 🧪 Testing

### Test Options
1. **Historical Data**: Test with real market data
2. **Sample Data**: Test with synthetic data
3. **Swing Detection**: Validate swing point algorithm
4. **Fibonacci Math**: Verify level calculations
5. **Alert System**: Test price alerts
6. **Chart Creation**: Validate chart generation
7. **Live Simulation**: Demonstrate real-time functionality

### Running Tests
```bash
python test_fibonacci_detector.py
# Select option 1-9 for specific tests
```

## 🔗 Integration with AITRADE

### Adding to Main System
```python
# In main.py
from live_fibonacci_detector import LiveFibonacciDetector

self.fibonacci_detector = LiveFibonacciDetector(
    symbol='BTC/USDT',
    timeframe='5m'
)
self.fibonacci_detector.start_monitoring()
```

### Using in Strategies
```python
# Check if price is at Fibonacci level
fibonacci_levels = self.fibonacci_detector.fibonacci_levels
current_price = self.get_current_price()

for level in fibonacci_levels:
    if (current_price >= level.zone_low and 
        current_price <= level.zone_high):
        # Price at Fibonacci level - use in strategy
        pass
```

## 📊 Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Swing Point Accuracy | >85% | Correct swing identification |
| Fibonacci Precision | >95% | Accurate level calculations |
| Alert Responsiveness | <5s | Time from touch to alert |
| False Positive Rate | <10% | Incorrect alerts |

## 🚨 Important Notes

### Rate Limits
- **Binance**: 1200 requests/minute
- **Bybit**: 100 requests/second
- Respect exchange rate limits

### Data Requirements
- **Minimum**: 50 candles for analysis
- **Timeframe**: Match detector setting
- **Quality**: Handle missing data appropriately

### Risk Management
- **Not Financial Advice**: Analysis tool only
- **Backtesting**: Test before live trading
- **Risk Limits**: Set appropriate position sizes

## 🔮 Future Enhancements

### Planned Features
- **WebSocket Support**: Real-time streaming
- **Multiple Timeframes**: Simultaneous analysis
- **Advanced Patterns**: Elliott Wave, harmonics
- **Machine Learning**: ML-based detection
- **Web Interface**: Browser visualization
- **Mobile App**: iOS/Android notifications
- **Backtesting Engine**: Historical analysis
- **Strategy Integration**: Direct trading integration

## 📞 Support

### Troubleshooting
1. **Connection Errors**: Check exchange connectivity
2. **No Swing Points**: Lower strength threshold
3. **Chart Issues**: Verify matplotlib installation

### Getting Help
- **Documentation**: Check this README
- **Testing**: Run test suite
- **Logs**: Check `fibonacci_detector.log`

## 📄 License

Part of AITRADE system - same licensing terms apply.

---

**Disclaimer**: Educational/analysis tool only. Not financial advice. Always do your own research. 