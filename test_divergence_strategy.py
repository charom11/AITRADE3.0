"""
Test Divergence Strategy
Demonstrates the divergence detection strategy with real cryptocurrency data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from divergence_detector import DivergenceDetector
from strategies import DivergenceStrategy
from data_manager import DataManager

def load_crypto_data(symbol: str, data_folder: str = 'DATA') -> pd.DataFrame:
    """
    Load cryptocurrency data from CSV file
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        data_folder: Folder containing CSV files
        
    Returns:
        DataFrame with OHLCV data
    """
    file_path = os.path.join(data_folder, f'{symbol}_binance_historical_data.csv')
    
    if not os.path.exists(file_path):
        print(f"❌ Data file not found: {file_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert Date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Rename columns to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Missing required columns in {file_path}")
            print(f"   Available columns: {list(df.columns)}")
            return None
        
        # Sort by timestamp
        df = df.sort_index()
        
        print(f"✅ Loaded {len(df)} records for {symbol}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data for {symbol}: {e}")
        return None

def test_divergence_detector():
    """Test the divergence detector with real data"""
    print("🔍 TESTING DIVERGENCE DETECTOR")
    print("=" * 60)
    
    # Test with multiple cryptocurrencies
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 40)
        
        # Load data
        data = load_crypto_data(symbol)
        if data is None:
            continue
        
        # Use last 200 candles for analysis
        recent_data = data.tail(200)
        
        # Initialize divergence detector
        detector = DivergenceDetector(
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            min_candles=50,
            swing_threshold=0.02
        )
        
        # Analyze divergence
        analysis = detector.analyze_divergence(recent_data, confirm_signals=True)
        
        # Print results
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
        else:
            print(f"⏰ Timeframe: {analysis.get('timeframe', 'Unknown')}")
            print(f"📊 Data Points: {analysis.get('data_points', 'Unknown')}")
            print(f"📈 Total Signals: {analysis['total_signals']}")
            print(f"📊 RSI Signals: {analysis['rsi_signals']}")
            print(f"📉 MACD Signals: {analysis['macd_signals']}")
            print(f"💰 Current Price: ${analysis['current_price']:.4f}")
            print(f"📊 Current RSI: {analysis['current_rsi']:.2f}")
            print(f"📈 Current MACD: {analysis['current_macd']:.6f}")
            
            if analysis['signals']:
                print("\n🎯 DETECTED SIGNALS:")
                for i, signal in enumerate(analysis['signals'][:3], 1):  # Show top 3
                    print(f"  {i}. {signal['type'].upper()} Divergence ({signal['indicator'].upper()})")
                    print(f"     Strength: {signal['strength']:.2f}")
                    print(f"     Date: {signal['signal_date']}")
                    
                    if signal['type'] == 'bullish':
                        print(f"     Price: ${signal['price_low1_val']:.4f} → ${signal['price_low2_val']:.4f}")
                        print(f"     {signal['indicator'].upper()}: {signal['indicator_low1_val']:.2f} → {signal['indicator_low2_val']:.2f}")
                    else:
                        print(f"     Price: ${signal['price_high1_val']:.4f} → ${signal['price_high2_val']:.4f}")
                        print(f"     {signal['indicator'].upper()}: {signal['indicator_high1_val']:.2f} → {signal['indicator_high2_val']:.2f}")
                    
                    if 'support_confirmation' in signal:
                        print(f"     S/R Confirmed: {signal.get('support_confirmation', signal.get('resistance_confirmation', False))}")
                    
                    if 'pattern_confirmation' in signal:
                        print(f"     Pattern: {signal.get('candlestick_pattern', 'None')}")
                    
                    print()
            else:
                print("❌ No Class A divergence signals detected")

def test_divergence_strategy():
    """Test the divergence strategy integration"""
    print("\n🚀 TESTING DIVERGENCE STRATEGY INTEGRATION")
    print("=" * 60)
    
    # Test with BTCUSDT
    symbol = 'BTCUSDT'
    print(f"\n📊 Testing {symbol} Strategy")
    print("-" * 40)
    
    # Load data
    data = load_crypto_data(symbol)
    if data is None:
        return
    
    # Use last 200 candles for analysis
    recent_data = data.tail(200)
    
    # Initialize divergence strategy
    strategy = DivergenceStrategy()
    
    # Generate signals
    signals = strategy.generate_signals(recent_data)
    
    # Get detailed analysis
    analysis = strategy.get_divergence_analysis(recent_data)
    
    # Print results
    print(f"📈 Total Signals Generated: {len(signals[signals['signal'] != 0])}")
    print(f"📊 Current Signal: {signals['signal'].iloc[-1]}")
    
    if signals['signal'].iloc[-1] != 0:
        print(f"🎯 Signal Type: {signals['divergence_type'].iloc[-1]}")
        print(f"📊 Indicator: {signals['divergence_indicator'].iloc[-1]}")
        print(f"💪 Strength: {signals['divergence_strength'].iloc[-1]:.2f}")
        print(f"⏰ Timeframe: {signals['timeframe'].iloc[-1]}")
        print(f"🛡️ S/R Confirmed: {signals['support_resistance_confirmed'].iloc[-1]}")
        print(f"🕯️ Pattern Confirmed: {signals['candlestick_confirmed'].iloc[-1]}")
    
    # Calculate position size
    current_price = recent_data['close'].iloc[-1]
    portfolio_value = 100000  # $100k portfolio
    
    if signals['signal'].iloc[-1] != 0:
        position_size = strategy.calculate_position_size(
            signals['signal'].iloc[-1], 
            current_price, 
            portfolio_value
        )
        print(f"💰 Position Size: ${abs(position_size):.2f}")
        print(f"📊 Position %: {abs(position_size) / portfolio_value * 100:.2f}%")

def test_multiple_timeframes():
    """Test divergence detection across multiple timeframes"""
    print("\n⏰ TESTING MULTIPLE TIMEFRAMES")
    print("=" * 60)
    
    symbol = 'BTCUSDT'
    data = load_crypto_data(symbol)
    if data is None:
        return
    
    # Test different timeframes
    timeframes = {
        '1H': 24 * 7,      # 1 week of hourly data
        '4H': 24 * 7,      # 1 week of 4-hour data
        '1D': 30,          # 30 days of daily data
    }
    
    detector = DivergenceDetector()
    
    for timeframe, periods in timeframes.items():
        print(f"\n📊 {timeframe} Timeframe ({periods} periods)")
        print("-" * 40)
        
        # Resample data to different timeframe
        if timeframe == '1H':
            resampled_data = data.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        elif timeframe == '4H':
            resampled_data = data.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:  # 1D
            resampled_data = data.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Use recent data
        recent_data = resampled_data.tail(periods)
        
        # Analyze divergence
        analysis = detector.analyze_divergence(recent_data, confirm_signals=True)
        
        if 'error' not in analysis:
            print(f"⏰ Detected Timeframe: {analysis.get('timeframe', 'Unknown')}")
            print(f"📈 Signals: {analysis['total_signals']}")
            print(f"💰 Current Price: ${analysis['current_price']:.4f}")
            print(f"📊 Current RSI: {analysis['current_rsi']:.2f}")
            
            if analysis['signals']:
                latest_signal = analysis['signals'][0]
                print(f"🎯 Latest: {latest_signal['type'].upper()} ({latest_signal['indicator'].upper()})")
                print(f"💪 Strength: {latest_signal['strength']:.2f}")
            else:
                print("❌ No signals")

def main():
    """Main test function"""
    print("🚀 DIVERGENCE STRATEGY TESTING")
    print("=" * 60)
    print("Testing Class A divergence detection with real cryptocurrency data")
    print("=" * 60)
    
    # Check if DATA folder exists
    if not os.path.exists('DATA'):
        print("❌ DATA folder not found!")
        print("Please run binance_data_extractor.py first to download data")
        return
    
    # List available data files
    data_files = [f for f in os.listdir('DATA') if f.endswith('_binance_historical_data.csv')]
    if not data_files:
        print("❌ No data files found in DATA folder!")
        print("Please run binance_data_extractor.py first to download data")
        return
    
    print(f"✅ Found {len(data_files)} data files")
    
    # Run tests
    try:
        test_divergence_detector()
        test_divergence_strategy()
        test_multiple_timeframes()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 