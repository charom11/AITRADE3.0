# 🛡️ Live Support and Resistance Detector

A real-time support and resistance level detection system that analyzes live market data to identify key price zones with multiple touch confirmations, volume analysis, and real-time alerts.

## ✨ Key Features

### 🔍 Real-Time Detection
- **Live Data Streaming**: Connects to Binance API for real-time OHLCV data
- **Multiple Timeframes**: Supports 1m, 5m, 15m, 1h, 4h, 1d timeframes
- **Rolling Window**: Maintains 100-200 candle history for analysis
- **Continuous Updates**: Automatically updates zones as new data arrives

### 🎯 Zone Identification
- **Swing Point Detection**: Identifies local highs and lows with configurable sensitivity
- **Multiple Touch Confirmation**: Requires minimum 2 touches for zone validation
- **Volume Spike Analysis**: Confirms zones with above-average volume
- **Price Range Zones**: Creates zones with ±0.3% buffer instead of exact lines
- **Strength Calculation**: Measures zone strength based on touch count and volume

### 🚨 Real-Time Alerts
- **Zone Break Detection**: Alerts when price breaks support/resistance
- **Zone Approach Alerts**: Warns when price approaches zones (within 1%)
- **Telegram Integration**: Sends formatted alerts to Telegram
- **Console Notifications**: Real-time status updates in terminal

### 📊 Visualization & Export
- **Live Charts**: Optional matplotlib-based real-time charts
- **Zone Visualization**: Color-coded support/resistance levels
- **JSON Export**: Export zones with metadata to files
- **Status Reports**: Detailed console status updates

## 📁 Files

| File | Description |
|------|-------------|
| `live_support_resistance_detector.py` | Main detector class with all functionality |
| `test_support_resistance_detector.py` | Test suite for historical data analysis |
| `SUPPORT_RESISTANCE_README.md` | This documentation file |

## 🚀 Quick Start

### 1. Basic Usage

```python
from live_support_resistance_detector import LiveSupportResistanceDetector

# Create detector
detector = LiveSupportResistanceDetector(
    symbol='BTC/USDT',
    timeframe='5m',
    enable_alerts=True
)

# Start live detection
detector.start()
```

### 2. Interactive Setup

```bash
python live_support_resistance_detector.py
```

Follow the prompts to configure:
- Trading pair symbol
- Timeframe (1m, 5m, 15m)
- Chart visualization
- Alert settings
- Telegram integration

### 3. Testing with Historical Data

```bash
python test_support_resistance_detector.py
```

Choose from:
- Single symbol test
- Multiple symbols comparison
- Parameter sensitivity analysis
- Live simulation demo

## ⚙️ Configuration

### Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | 'BTC/USDT' | Trading pair symbol |
| `exchange` | 'binance' | Exchange name |
| `timeframe` | '5m' | Candle timeframe |
| `window_size` | 200 | Number of candles to analyze |
| `min_touches` | 2 | Minimum touches for zone confirmation |
| `zone_buffer` | 0.003 | Price buffer around zone (±0.3%) |
| `volume_threshold` | 1.5 | Volume spike multiplier |
| `swing_sensitivity` | 0.02 | Minimum swing size (2%) |
| `enable_charts` | False | Enable real-time charts |
| `enable_alerts` | True | Enable price alerts |

### Alert Settings

```python
# Telegram setup
detector = LiveSupportResistanceDetector(
    telegram_bot_token='YOUR_BOT_TOKEN',
    telegram_chat_id='YOUR_CHAT_ID'
)
```

## 📊 Zone Analysis

### SupportResistanceZone Object

```python
@dataclass
class SupportResistanceZone:
    level: float                    # Zone price level
    zone_type: str                  # 'support' or 'resistance'
    strength: float                 # 0-1 scale based on touches
    touches: int                    # Number of touches
    first_touch: datetime           # First touch timestamp
    last_touch: datetime            # Last touch timestamp
    volume_confirmed: bool          # Volume spike confirmation
    price_range: Tuple[float, float] # Zone range with buffer
    is_active: bool = True          # Zone active status
    break_count: int = 0            # Number of breaks
    last_break: Optional[datetime]  # Last break timestamp
```

### Zone Strength Calculation

- **Base Strength**: `min(1.0, touches / 5.0)`
- **Volume Bonus**: Additional confirmation from volume spikes
- **Touch Count**: More touches = stronger zone
- **Recency**: Recent touches weighted higher

## 🔍 Detection Algorithm

### 1. Swing Point Detection

```python
def detect_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """
    Identifies swing highs and lows using 5-point pattern:
    - Swing High: H[i] > H[i-1], H[i-2] AND H[i] > H[i+1], H[i+2]
    - Swing Low: L[i] < L[i-1], L[i-2] AND L[i] < L[i+1], L[i+2]
    - Minimum swing size: 2% of price
    """
```

### 2. Zone Formation

```python
def identify_zones(self, data: pd.DataFrame) -> Tuple[List[SupportResistanceZone], List[SupportResistanceZone]]:
    """
    Process:
    1. Find swing points
    2. Group nearby levels (within buffer)
    3. Count touches for each level
    4. Filter by minimum touches
    5. Calculate strength and metadata
    """
```

### 3. Volume Confirmation

```python
def check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
    """
    Checks if swing point has volume spike:
    - Compare swing volume to 11-period average
    - Volume > (average * threshold) = confirmed
    """
```

## 🚨 Alert System

### Alert Types

1. **Zone Break Alerts**
   - Triggered when price breaks support/resistance
   - Includes zone strength and break details

2. **Zone Approach Alerts**
   - Triggered when price approaches zone (within 1%)
   - Shows distance and zone information

### Telegram Alert Format

```
🚨 Zone Break Alert 🚨

Symbol: BTC/USDT
Type: Support Break
Price: $45,234.56
Zone Level: $45,000.00
Strength: 0.85
Touches: 4
Time: 2024-01-15 14:30:25
```

## 📈 Chart Visualization

### Features

- **Real-time Price Line**: Current price tracking
- **Support Zones**: Green dashed lines (volume confirmed = darker)
- **Resistance Zones**: Red dashed lines (volume confirmed = darker)
- **Zone Strength**: Opacity indicates strength
- **Legend**: Shows all zones with strength values

### Enable Charts

```python
detector = LiveSupportResistanceDetector(
    enable_charts=True,
    # ... other parameters
)
```

## 📊 Testing & Validation

### Historical Data Testing

```python
# Test with historical data
support_zones, resistance_zones = test_detector_with_historical_data('BTCUSDT', 200)

# Results include:
# - Zone count and strength
# - Touch analysis
# - Volume confirmation
# - Break/approach detection
```

### Parameter Sensitivity

Test different settings:
- **More Touches**: Higher confirmation threshold
- **Wider Buffer**: Larger zone ranges
- **More Sensitive**: Lower swing detection threshold
- **Less Sensitive**: Higher swing detection threshold

### Multiple Symbol Analysis

Compare detection across:
- BTCUSDT, ETHUSDT, XRPUSDT
- ADAUSDT, SOLUSDT
- Different market conditions

## 🔧 Integration with AITRADE

### Strategy Integration

```python
# Add to strategies.py
class SupportResistanceStrategy(BaseStrategy):
    def __init__(self):
        self.detector = LiveSupportResistanceDetector(
            symbol=self.symbol,
            timeframe='5m'
        )
    
    def generate_signals(self, data):
        # Use detector zones for signal generation
        support_zones, resistance_zones = self.detector.identify_zones(data)
        # Generate buy/sell signals based on zone interactions
```

### Portfolio Management

```python
# Add zone-based position sizing
def calculate_position_size(self, signal, price, portfolio_value):
    # Consider zone strength for position sizing
    # Stronger zones = larger positions
```

## 📋 Usage Examples

### Example 1: Basic Live Detection

```python
from live_support_resistance_detector import LiveSupportResistanceDetector

# Create detector
detector = LiveSupportResistanceDetector(
    symbol='ETH/USDT',
    timeframe='15m',
    min_touches=3,
    enable_alerts=True
)

# Start detection
detector.start()
```

### Example 2: Custom Configuration

```python
detector = LiveSupportResistanceDetector(
    symbol='BTC/USDT',
    timeframe='1h',
    window_size=300,
    min_touches=2,
    zone_buffer=0.005,  # 0.5% buffer
    volume_threshold=2.0,  # 2x volume spike
    swing_sensitivity=0.03,  # 3% minimum swing
    enable_charts=True,
    enable_alerts=True,
    telegram_bot_token='YOUR_TOKEN',
    telegram_chat_id='YOUR_CHAT_ID'
)
```

### Example 3: Export Zones

```python
# Export zones to JSON
detector.export_zones('btc_zones_20240115.json')

# Export format:
{
  "symbol": "BTC/USDT",
  "timeframe": "5m",
  "export_time": "2024-01-15T14:30:25",
  "support_zones": [...],
  "resistance_zones": [...]
}
```

## 🎯 Performance Metrics

### Detection Accuracy

| Metric | Description |
|--------|-------------|
| **Zone Strength** | 0-1 scale based on touches and volume |
| **Touch Count** | Number of times price touched zone |
| **Volume Confirmation** | Percentage of zones with volume spikes |
| **Break Rate** | Frequency of zone breaks |

### Sample Results (BTCUSDT 5m)

| Symbol | Support Zones | Resistance Zones | Avg Strength | Volume Confirmed |
|--------|---------------|------------------|--------------|------------------|
| BTCUSDT | 3 | 2 | 0.72 | 80% |
| ETHUSDT | 2 | 3 | 0.68 | 75% |
| XRPUSDT | 1 | 2 | 0.55 | 60% |

## ⚠️ Important Notes

### Risk Management

- **Not Financial Advice**: This tool is for analysis only
- **Backtesting**: Always test with historical data first
- **Parameter Tuning**: Adjust sensitivity based on market conditions
- **Multiple Timeframes**: Consider higher timeframe context

### Technical Considerations

- **API Limits**: Respect exchange rate limits
- **Network Issues**: Handle connection failures gracefully
- **Memory Usage**: Large window sizes increase memory usage
- **CPU Usage**: Real-time charts can be CPU intensive

### Best Practices

1. **Start with Testing**: Use `test_support_resistance_detector.py` first
2. **Parameter Tuning**: Test different settings on historical data
3. **Multiple Symbols**: Don't rely on single symbol analysis
4. **Timeframe Selection**: Choose appropriate timeframe for your strategy
5. **Alert Management**: Configure alerts to avoid notification spam

## 🔄 Updates & Maintenance

### Regular Maintenance

- **Data Quality**: Monitor for data gaps or anomalies
- **Zone Updates**: Review and clean old/inactive zones
- **Performance**: Monitor CPU and memory usage
- **Alerts**: Test alert delivery regularly

### Future Enhancements

- **Machine Learning**: ML-based zone strength prediction
- **Multi-Exchange**: Support for additional exchanges
- **Advanced Patterns**: Candlestick pattern integration
- **Risk Metrics**: Zone-based risk calculations
- **Web Interface**: Browser-based visualization

## 📞 Support

For issues or questions:
1. Check the test suite first
2. Review parameter settings
3. Verify data availability
4. Check network connectivity
5. Review log files for errors

---

**Disclaimer**: This tool is for educational and analysis purposes only. Always conduct your own research and consider professional financial advice before making trading decisions. 