"""
Test Suite for Live Fibonacci Detector
Comprehensive testing of Fibonacci retracement and extension detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from live_fibonacci_detector import LiveFibonacciDetector, FibonacciLevel, SwingPoint

def load_historical_data(symbol: str, data_folder: str = 'DATA') -> pd.DataFrame:
    """Load historical data from CSV file"""
    try:
        file_path = os.path.join(data_folder, f"{symbol.replace('/', '_')}_binance_historical_data.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ Data file not found: {file_path}")
            print("💡 Run binance_data_extractor.py first to download data")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df.columns = [col.lower() for col in df.columns]
        
        print(f"✅ Loaded {len(df)} data points for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

def create_sample_data(periods: int = 200, trend: str = 'uptrend') -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate base prices
    if trend == 'uptrend':
        base_prices = np.linspace(100, 150, periods) + np.random.normal(0, 2, periods)
    elif trend == 'downtrend':
        base_prices = np.linspace(150, 100, periods) + np.random.normal(0, 2, periods)
    else:  # sideways
        base_prices = 125 + np.random.normal(0, 5, periods)
    
    # Create OHLCV data
    data = []
    for i, base_price in enumerate(base_prices):
        # Add some volatility
        high = base_price + abs(np.random.normal(0, 1))
        low = base_price - abs(np.random.normal(0, 1))
        open_price = base_price + np.random.normal(0, 0.5)
        close_price = base_price + np.random.normal(0, 0.5)
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    # Create DataFrame with timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq='5min')
    df = pd.DataFrame(data, index=timestamps)
    
    return df

def test_fibonacci_detector_with_historical_data():
    """Test Fibonacci detector with historical data"""
    print("\n🧪 TESTING WITH HISTORICAL DATA")
    print("="*50)
    
    # Load historical data
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 30)
        
        df = load_historical_data(symbol)
        if df.empty:
            continue
        
        # Use last 200 data points
        recent_data = df.tail(200)
        
        # Create detector
        detector = LiveFibonacciDetector(
            symbol=symbol,
            timeframe='5m',
            lookback_periods=200
        )
        
        # Set data manually
        detector.data = recent_data
        detector.current_price = recent_data['close'].iloc[-1]
        
        # Test swing point detection
        swing_highs, swing_lows = detector.detect_swing_points(recent_data)
        print(f"🎯 Swing Highs: {len(swing_highs)}")
        print(f"🎯 Swing Lows: {len(swing_lows)}")
        
        if swing_highs and swing_lows:
            # Test Fibonacci level calculation
            latest_high = max(swing_highs, key=lambda x: x.timestamp)
            latest_low = max(swing_lows, key=lambda x: x.timestamp)
            
            print(f"📈 Latest Swing High: ${latest_high.price:.4f}")
            print(f"📉 Latest Swing Low: ${latest_low.price:.4f}")
            
            # Calculate Fibonacci levels
            levels = detector.calculate_fibonacci_levels(latest_high, latest_low)
            print(f"📐 Fibonacci Levels: {len(levels)}")
            
            # Display levels
            for level in levels:
                print(f"  {level.level_type.title()} {level.percentage}%: ${level.price:.4f}")
        
        print()

def test_fibonacci_detector_with_sample_data():
    """Test Fibonacci detector with synthetic data"""
    print("\n🧪 TESTING WITH SAMPLE DATA")
    print("="*50)
    
    # Test different trends
    trends = ['uptrend', 'downtrend', 'sideways']
    
    for trend in trends:
        print(f"\n📈 Testing {trend.upper()} pattern")
        print("-" * 30)
        
        # Create sample data
        df = create_sample_data(200, trend)
        
        # Create detector
        detector = LiveFibonacciDetector(
            symbol='TEST/USDT',
            timeframe='5m',
            lookback_periods=200
        )
        
        # Set data
        detector.data = df
        detector.current_price = df['close'].iloc[-1]
        
        # Test swing point detection
        swing_highs, swing_lows = detector.detect_swing_points(df)
        print(f"🎯 Swing Highs: {len(swing_highs)}")
        print(f"🎯 Swing Lows: {len(swing_lows)}")
        
        if swing_highs and swing_lows:
            # Get strongest swing points
            strongest_high = max(swing_highs, key=lambda x: x.strength)
            strongest_low = max(swing_lows, key=lambda x: x.strength)
            
            print(f"💪 Strongest Swing High: ${strongest_high.price:.4f} (Strength: {strongest_high.strength:.2f})")
            print(f"💪 Strongest Swing Low: ${strongest_low.price:.4f} (Strength: {strongest_low.strength:.2f})")
            
            # Calculate Fibonacci levels
            levels = detector.calculate_fibonacci_levels(strongest_high, strongest_low)
            
            print(f"📐 Fibonacci Levels: {len(levels)}")
            
            # Group by type
            retracements = [l for l in levels if l.level_type == 'retracement']
            extensions = [l for l in levels if l.level_type == 'extension']
            
            print(f"  📉 Retracements: {len(retracements)}")
            print(f"  📈 Extensions: {len(extensions)}")
            
            # Show key levels
            for level in levels[:5]:  # Show first 5 levels
                print(f"    {level.level_type.title()} {level.percentage}%: ${level.price:.4f}")
        
        print()

def test_swing_point_detection():
    """Test swing point detection algorithm"""
    print("\n🎯 TESTING SWING POINT DETECTION")
    print("="*50)
    
    # Create data with known swing points
    np.random.seed(42)
    
    # Create a pattern with clear swing highs and lows
    periods = 100
    prices = []
    
    for i in range(periods):
        if i < 20:
            # Uptrend
            price = 100 + i * 0.5 + np.random.normal(0, 0.5)
        elif i < 40:
            # Downtrend
            price = 110 - (i - 20) * 0.3 + np.random.normal(0, 0.5)
        elif i < 60:
            # Uptrend
            price = 104 + (i - 40) * 0.4 + np.random.normal(0, 0.5)
        elif i < 80:
            # Downtrend
            price = 112 - (i - 60) * 0.2 + np.random.normal(0, 0.5)
        else:
            # Final uptrend
            price = 108 + (i - 80) * 0.3 + np.random.normal(0, 0.5)
        
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price + abs(np.random.normal(0, 0.3))
        low = price - abs(np.random.normal(0, 0.3))
        open_price = price + np.random.normal(0, 0.2)
        close_price = price + np.random.normal(0, 0.2)
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq='5min')
    df = pd.DataFrame(data, index=timestamps)
    
    # Create detector
    detector = LiveFibonacciDetector(
        symbol='TEST/USDT',
        timeframe='5m',
        min_swing_strength=0.3  # Lower threshold for testing
    )
    
    # Test swing point detection
    swing_highs, swing_lows = detector.detect_swing_points(df)
    
    print(f"🎯 Detected Swing Points:")
    print(f"  📈 Swing Highs: {len(swing_highs)}")
    print(f"  📉 Swing Lows: {len(swing_lows)}")
    
    # Display swing points
    print(f"\n📈 Swing Highs:")
    for i, swing in enumerate(swing_highs[:5]):  # Show first 5
        print(f"  {i+1}. ${swing.price:.4f} (Strength: {swing.strength:.2f}, Time: {swing.timestamp})")
    
    print(f"\n📉 Swing Lows:")
    for i, swing in enumerate(swing_lows[:5]):  # Show first 5
        print(f"  {i+1}. ${swing.price:.4f} (Strength: {swing.strength:.2f}, Time: {swing.timestamp})")
    
    # Test Fibonacci calculation
    if swing_highs and swing_lows:
        latest_high = max(swing_highs, key=lambda x: x.timestamp)
        latest_low = max(swing_lows, key=lambda x: x.timestamp)
        
        print(f"\n📐 Fibonacci Levels from latest swings:")
        print(f"  High: ${latest_high.price:.4f}")
        print(f"  Low: ${latest_low.price:.4f}")
        
        levels = detector.calculate_fibonacci_levels(latest_high, latest_low)
        
        for level in levels:
            print(f"  {level.level_type.title()} {level.percentage}%: ${level.price:.4f}")

def test_fibonacci_calculations():
    """Test Fibonacci level calculations"""
    print("\n📐 TESTING FIBONACCI CALCULATIONS")
    print("="*50)
    
    # Create test swing points
    swing_high = SwingPoint(
        point_type='high',
        price=100.0,
        timestamp=datetime.now(),
        volume=5000,
        strength=0.8
    )
    
    swing_low = SwingPoint(
        point_type='low',
        price=80.0,
        timestamp=datetime.now(),
        volume=5000,
        strength=0.8
    )
    
    # Create detector
    detector = LiveFibonacciDetector(
        symbol='TEST/USDT',
        timeframe='5m'
    )
    
    # Calculate levels
    levels = detector.calculate_fibonacci_levels(swing_high, swing_low)
    
    print(f"📊 Test Case: High=${swing_high.price}, Low=${swing_low.price}")
    print(f"📏 Price Range: ${swing_high.price - swing_low.price:.2f}")
    print()
    
    print("📉 Retracement Levels:")
    retracements = [l for l in levels if l.level_type == 'retracement']
    for level in retracements:
        print(f"  {level.percentage}%: ${level.price:.4f} (Zone: ${level.zone_low:.4f}-${level.zone_high:.4f})")
    
    print("\n📈 Extension Levels:")
    extensions = [l for l in levels if l.level_type == 'extension']
    for level in extensions:
        print(f"  {level.percentage}%: ${level.price:.4f} (Zone: ${level.zone_low:.4f}-${level.zone_high:.4f})")
    
    # Test with different price ranges
    print(f"\n🧮 Testing different price ranges:")
    test_cases = [
        (100, 90),   # Small range
        (100, 50),   # Large range
        (1000, 800), # High prices
        (1, 0.8),    # Low prices
    ]
    
    for high, low in test_cases:
        test_high = SwingPoint('high', high, datetime.now(), 5000, 0.8)
        test_low = SwingPoint('low', low, datetime.now(), 5000, 0.8)
        
        test_levels = detector.calculate_fibonacci_levels(test_high, test_low)
        
        print(f"\n  High=${high}, Low=${low}:")
        print(f"    Range: ${high - low:.2f}")
        print(f"    Levels: {len(test_levels)}")
        
        # Show 61.8% retracement
        for level in test_levels:
            if level.percentage == 61.8 and level.level_type == 'retracement':
                print(f"    61.8% Retracement: ${level.price:.4f}")
                break

def test_alert_system():
    """Test price alert system"""
    print("\n🚨 TESTING ALERT SYSTEM")
    print("="*50)
    
    # Create detector
    detector = LiveFibonacciDetector(
        symbol='TEST/USDT',
        timeframe='5m'
    )
    
    # Create test levels
    test_levels = [
        FibonacciLevel('retracement', 23.6, 95.0, 95.285, 94.715, 0.5, 0),
        FibonacciLevel('retracement', 38.2, 92.0, 92.276, 91.724, 0.5, 0),
        FibonacciLevel('retracement', 61.8, 87.0, 87.261, 86.739, 0.5, 0),
        FibonacciLevel('extension', 127.2, 125.0, 125.375, 124.625, 0.5, 0),
    ]
    
    detector.fibonacci_levels = test_levels
    
    # Test different price scenarios
    test_prices = [95.0, 92.0, 87.0, 125.0, 100.0, 80.0]
    
    print("💰 Testing price alerts:")
    for price in test_prices:
        detector.current_price = price
        alerts = detector.check_price_alerts()
        
        if alerts:
            print(f"  Price ${price:.2f}: {len(alerts)} alerts")
            for alert in alerts:
                print(f"    {alert['type']} {alert['level'].level_type} {alert['level'].percentage}%")
        else:
            print(f"  Price ${price:.2f}: No alerts")

def test_chart_creation():
    """Test chart creation functionality"""
    print("\n📊 TESTING CHART CREATION")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(100, 'uptrend')
    
    # Create detector
    detector = LiveFibonacciDetector(
        symbol='TEST/USDT',
        timeframe='5m'
    )
    
    # Set data
    detector.data = df
    detector.current_price = df['close'].iloc[-1]
    
    # Detect swing points and calculate levels
    swing_highs, swing_lows = detector.detect_swing_points(df)
    detector.swing_highs = swing_highs
    detector.swing_lows = swing_lows
    
    if swing_highs and swing_lows:
        latest_high = max(swing_highs, key=lambda x: x.timestamp)
        latest_low = max(swing_lows, key=lambda x: x.timestamp)
        
        levels = detector.calculate_fibonacci_levels(latest_high, latest_low)
        detector.fibonacci_levels = levels
        
        # Create chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_fibonacci_chart_{timestamp}.png"
        
        try:
            detector.create_chart(filename)
            print(f"✅ Chart created: {filename}")
            
            # Check if file exists
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"📁 File size: {file_size} bytes")
            else:
                print("❌ Chart file not found")
                
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    else:
        print("❌ No swing points detected for charting")

def demo_live_simulation():
    """Demonstrate live simulation"""
    print("\n🔄 LIVE SIMULATION DEMO")
    print("="*50)
    
    # Create detector
    detector = LiveFibonacciDetector(
        symbol='DEMO/USDT',
        timeframe='5m',
        lookback_periods=50
    )
    
    # Create initial data
    df = create_sample_data(50, 'uptrend')
    detector.data = df
    detector.current_price = df['close'].iloc[-1]
    
    print("🚀 Starting live simulation...")
    print("📊 Initial data loaded")
    
    # Simulate live updates
    for i in range(5):
        print(f"\n⏰ Update {i+1}/5")
        print("-" * 20)
        
        # Add new data point
        last_price = detector.current_price
        new_price = last_price + np.random.normal(0, 0.5)
        
        new_data = {
            'open': last_price,
            'high': max(last_price, new_price),
            'low': min(last_price, new_price),
            'close': new_price,
            'volume': np.random.randint(1000, 10000)
        }
        
        new_timestamp = detector.data.index[-1] + timedelta(minutes=5)
        new_row = pd.DataFrame([new_data], index=[new_timestamp])
        
        detector.data = pd.concat([detector.data, new_row])
        detector.data = detector.data.tail(50)  # Keep only last 50 points
        detector.current_price = new_price
        
        # Update Fibonacci levels
        detector.update_fibonacci_levels(detector.data)
        
        # Print status
        print(f"💰 Current Price: ${detector.current_price:.4f}")
        print(f"📈 Swing Highs: {len(detector.swing_highs)}")
        print(f"📉 Swing Lows: {len(detector.swing_lows)}")
        print(f"📐 Fibonacci Levels: {len(detector.fibonacci_levels)}")
        
        # Check alerts
        alerts = detector.check_price_alerts()
        if alerts:
            print(f"🚨 Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  {alert['type']} {alert['level'].percentage}% level")
        
        time.sleep(1)  # Simulate time delay
    
    print("\n✅ Live simulation completed")

def main():
    """Main test function"""
    print("🧪 FIBONACCI DETECTOR TEST SUITE")
    print("="*60)
    print("Comprehensive testing of Fibonacci retracement and extension detection")
    print("="*60)
    
    while True:
        print("\n📋 TEST OPTIONS:")
        print("1. Test with historical data")
        print("2. Test with sample data")
        print("3. Test swing point detection")
        print("4. Test Fibonacci calculations")
        print("5. Test alert system")
        print("6. Test chart creation")
        print("7. Live simulation demo")
        print("8. Run all tests")
        print("9. Exit")
        
        try:
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                test_fibonacci_detector_with_historical_data()
                
            elif choice == '2':
                test_fibonacci_detector_with_sample_data()
                
            elif choice == '3':
                test_swing_point_detection()
                
            elif choice == '4':
                test_fibonacci_calculations()
                
            elif choice == '5':
                test_alert_system()
                
            elif choice == '6':
                test_chart_creation()
                
            elif choice == '7':
                demo_live_simulation()
                
            elif choice == '8':
                print("\n🔄 Running all tests...")
                test_fibonacci_detector_with_historical_data()
                test_fibonacci_detector_with_sample_data()
                test_swing_point_detection()
                test_fibonacci_calculations()
                test_alert_system()
                test_chart_creation()
                demo_live_simulation()
                print("\n✅ All tests completed!")
                
            elif choice == '9':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid option")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 