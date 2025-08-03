"""
Timeframe Detection Demo
Demonstrates the automatic timeframe detection feature of the divergence detector
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from divergence_detector import DivergenceDetector

def create_sample_data(timeframe: str, periods: int = 100):
    """
    Create sample data with specified timeframe
    
    Args:
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '4h', '1d')
        periods: Number of periods to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Convert timeframe to timedelta
    timeframe_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1)
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    delta = timeframe_map[timeframe]
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + i * delta for i in range(periods)]
    
    # Generate sample price data
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(periods):
        if i < 30:
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 60:
            price = base_price + 15 - (i - 30) * 0.3 + np.random.normal(0, 1)
        else:
            price = base_price + 6 + (i - 60) * 0.2 + np.random.normal(0, 1)
        prices.append(price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in range(periods)]
    }, index=dates)
    
    return data

def demo_timeframe_detection():
    """Demonstrate timeframe detection with different timeframes"""
    print("🔍 TIMEFRAME DETECTION DEMO")
    print("=" * 60)
    
    # Test different timeframes
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    detector = DivergenceDetector()
    
    for tf in timeframes:
        print(f"\n📊 Testing {tf} Timeframe")
        print("-" * 40)
        
        # Create sample data
        data = create_sample_data(tf, 100)
        
        # Analyze divergence
        analysis = detector.analyze_divergence(data, confirm_signals=False)
        
        if 'error' not in analysis:
            print(f"⏰ Detected Timeframe: {analysis['timeframe']}")
            print(f"📊 Data Points: {analysis['data_points']}")
            print(f"📅 Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            print(f"⏱️ Duration: {analysis['date_range']['duration']}")
            print(f"📈 Signals: {analysis['total_signals']}")
            print(f"💰 Current Price: ${analysis['current_price']:.4f}")
        else:
            print(f"❌ Error: {analysis['error']}")
    
    print("\n" + "=" * 60)
    print("✅ Timeframe detection demo completed!")

def demo_real_data_timeframes():
    """Demonstrate timeframe detection with real data resampling"""
    print("\n📈 REAL DATA TIMEFRAME DEMO")
    print("=" * 60)
    
    try:
        # Load real data (if available)
        import os
        if os.path.exists('DATA/BTCUSDT_binance_historical_data.csv'):
            # Load BTC data
            df = pd.read_csv('DATA/BTCUSDT_binance_historical_data.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df.columns = [col.lower() for col in df.columns]
            
            print("✅ Loaded real BTCUSDT data")
            print(f"📊 Original data points: {len(df)}")
            print(f"📅 Original date range: {df.index[0]} to {df.index[-1]}")
            
            detector = DivergenceDetector()
            
            # Test different resampled timeframes
            resample_configs = {
                '1h': '1H',
                '4h': '4H', 
                '1d': '1D'
            }
            
            for name, freq in resample_configs.items():
                print(f"\n📊 Testing {name} Resampled Data")
                print("-" * 40)
                
                # Resample data
                resampled = df.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Use last 100 points
                recent_data = resampled.tail(100)
                
                # Analyze
                analysis = detector.analyze_divergence(recent_data, confirm_signals=False)
                
                if 'error' not in analysis:
                    print(f"⏰ Detected Timeframe: {analysis['timeframe']}")
                    print(f"📊 Data Points: {analysis['data_points']}")
                    print(f"📅 Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
                    print(f"⏱️ Duration: {analysis['date_range']['duration']}")
                    print(f"📈 Signals: {analysis['total_signals']}")
                else:
                    print(f"❌ Error: {analysis['error']}")
        else:
            print("❌ Real data not found. Run binance_data_extractor.py first.")
            
    except Exception as e:
        print(f"❌ Error loading real data: {e}")

def main():
    """Main demonstration function"""
    print("🚀 DIVERGENCE DETECTOR - TIMEFRAME DETECTION")
    print("=" * 60)
    print("This demo shows how the divergence detector automatically")
    print("detects the timeframe of your data and includes it in analysis.")
    print("=" * 60)
    
    # Run demos
    demo_timeframe_detection()
    demo_real_data_timeframes()
    
    print("\n🎯 KEY FEATURES:")
    print("• Automatic timeframe detection from data intervals")
    print("• Support for seconds, minutes, hours, days, weeks, months")
    print("• Timeframe information included in all analysis results")
    print("• Works with both synthetic and real market data")
    print("• Helps users understand the context of their signals")

if __name__ == "__main__":
    main() 