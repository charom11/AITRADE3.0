"""
Test Support and Resistance Detector
Demonstrates the detector functionality using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from live_support_resistance_detector import LiveSupportResistanceDetector, SupportResistanceZone

def load_historical_data(symbol: str, data_folder: str = 'DATA') -> pd.DataFrame:
    """
    Load historical data from CSV file
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        data_folder: Folder containing CSV files
        
    Returns:
        DataFrame with OHLCV data
    """
    file_path = os.path.join(data_folder, f"{symbol}_binance_historical_data.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = [col.lower() for col in df.columns]
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def simulate_live_data(df: pd.DataFrame, chunk_size: int = 50) -> pd.DataFrame:
    """
    Simulate live data by taking chunks of historical data
    
    Args:
        df: Historical DataFrame
        chunk_size: Number of candles per chunk
        
    Returns:
        DataFrame chunk for simulation
    """
    # Take the most recent chunk_size candles
    return df.tail(chunk_size)

def test_detector_with_historical_data(symbol: str = 'BTCUSDT', chunk_size: int = 200):
    """
    Test the detector with historical data
    
    Args:
        symbol: Trading pair symbol
        chunk_size: Number of candles to analyze
    """
    print(f"🔍 TESTING SUPPORT/RESISTANCE DETECTOR")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Chunk Size: {chunk_size} candles")
    print(f"{'='*60}")
    
    try:
        # Load historical data
        print(f"📊 Loading historical data for {symbol}...")
        df = load_historical_data(symbol)
        print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Simulate live data
        live_data = simulate_live_data(df, chunk_size)
        print(f"📈 Using {len(live_data)} recent candles for analysis")
        
        # Create detector instance
        detector = LiveSupportResistanceDetector(
            symbol=symbol.replace('USDT', '/USDT'),
            timeframe='5m',
            window_size=chunk_size,
            min_touches=2,
            zone_buffer=0.003,
            volume_threshold=1.5,
            swing_sensitivity=0.02,
            enable_charts=False,
            enable_alerts=False
        )
        
        # Analyze the data
        print(f"\n🔍 Analyzing support and resistance zones...")
        support_zones, resistance_zones = detector.identify_zones(live_data)
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Current Price: ${live_data['close'].iloc[-1]:.4f}")
        print(f"Analysis Period: {live_data.index[0]} to {live_data.index[-1]}")
        print(f"Data Points: {len(live_data)}")
        
        print(f"\n🛡️ SUPPORT ZONES ({len(support_zones)}):")
        if support_zones:
            for i, zone in enumerate(support_zones, 1):
                volume_icon = "📈" if zone.volume_confirmed else "📊"
                print(f"  {i}. ${zone.level:.4f} | S:{zone.strength:.2f} | T:{zone.touches} | {volume_icon}")
                print(f"     Range: ${zone.price_range[0]:.4f} - ${zone.price_range[1]:.4f}")
                print(f"     First Touch: {zone.first_touch}")
                print(f"     Last Touch: {zone.last_touch}")
        else:
            print("  No support zones detected")
        
        print(f"\n🚀 RESISTANCE ZONES ({len(resistance_zones)}):")
        if resistance_zones:
            for i, zone in enumerate(resistance_zones, 1):
                volume_icon = "📈" if zone.volume_confirmed else "📊"
                print(f"  {i}. ${zone.level:.4f} | S:{zone.strength:.2f} | T:{zone.touches} | {volume_icon}")
                print(f"     Range: ${zone.price_range[0]:.4f} - ${zone.price_range[1]:.4f}")
                print(f"     First Touch: {zone.first_touch}")
                print(f"     Last Touch: {zone.last_touch}")
        else:
            print("  No resistance zones detected")
        
        # Test zone break detection
        current_price = live_data['close'].iloc[-1]
        breaks = detector.check_zone_breaks(current_price)
        approaches = detector.check_zone_approaches(current_price)
        
        print(f"\n🚨 ZONE INTERACTIONS")
        print(f"{'='*60}")
        
        if breaks:
            print(f"🔴 Zone Breaks ({len(breaks)}):")
            for break_alert in breaks:
                zone = break_alert['zone']
                print(f"  {break_alert['type']}: ${zone.level:.4f} (S:{zone.strength:.2f})")
        else:
            print("✅ No zone breaks detected")
        
        if approaches:
            print(f"⚠️ Zone Approaches ({len(approaches)}):")
            for approach_alert in approaches:
                zone = approach_alert['zone']
                distance_pct = approach_alert['distance'] * 100
                print(f"  {approach_alert['type']}: ${zone.level:.4f} ({distance_pct:.2f}% away)")
        else:
            print("✅ No zone approaches detected")
        
        print(f"{'='*60}")
        print("✅ Analysis completed successfully!")
        
        return support_zones, resistance_zones
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return [], []

def test_multiple_symbols():
    """Test detector with multiple symbols"""
    print(f"\n🔍 TESTING MULTIPLE SYMBOLS")
    print(f"{'='*60}")
    
    # Available symbols from DATA folder
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT']
    
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            support_zones, resistance_zones = test_detector_with_historical_data(symbol, 150)
            
            results[symbol] = {
                'support_count': len(support_zones),
                'resistance_count': len(resistance_zones),
                'total_zones': len(support_zones) + len(resistance_zones),
                'avg_support_strength': np.mean([z.strength for z in support_zones]) if support_zones else 0,
                'avg_resistance_strength': np.mean([z.strength for z in resistance_zones]) if resistance_zones else 0
            }
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    # Summary
    print(f"\n📊 SUMMARY RESULTS")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'Support':<8} {'Resistance':<10} {'Total':<6} {'Avg S':<6} {'Avg R':<6}")
    print(f"{'-'*60}")
    
    for symbol, result in results.items():
        if 'error' not in result:
            print(f"{symbol:<10} {result['support_count']:<8} {result['resistance_count']:<10} "
                  f"{result['total_zones']:<6} {result['avg_support_strength']:<6.2f} "
                  f"{result['avg_resistance_strength']:<6.2f}")
        else:
            print(f"{symbol:<10} {'ERROR':<8} {'ERROR':<10} {'ERROR':<6} {'ERROR':<6} {'ERROR':<6}")

def test_parameter_sensitivity(symbol: str = 'BTCUSDT'):
    """Test different parameter settings"""
    print(f"\n🔍 PARAMETER SENSITIVITY TEST")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Load data
        df = load_historical_data(symbol)
        live_data = simulate_live_data(df, 200)
        
        # Test different parameters
        parameters = [
            {'min_touches': 2, 'zone_buffer': 0.003, 'swing_sensitivity': 0.02, 'name': 'Default'},
            {'min_touches': 3, 'zone_buffer': 0.003, 'swing_sensitivity': 0.02, 'name': 'More Touches'},
            {'min_touches': 2, 'zone_buffer': 0.005, 'swing_sensitivity': 0.02, 'name': 'Wider Buffer'},
            {'min_touches': 2, 'zone_buffer': 0.003, 'swing_sensitivity': 0.01, 'name': 'More Sensitive'},
            {'min_touches': 2, 'zone_buffer': 0.003, 'swing_sensitivity': 0.05, 'name': 'Less Sensitive'}
        ]
        
        results = []
        
        for params in parameters:
            print(f"\n🔧 Testing {params['name']} parameters...")
            
            detector = LiveSupportResistanceDetector(
                symbol=symbol.replace('USDT', '/USDT'),
                timeframe='5m',
                window_size=200,
                min_touches=params['min_touches'],
                zone_buffer=params['zone_buffer'],
                swing_sensitivity=params['swing_sensitivity'],
                enable_charts=False,
                enable_alerts=False
            )
            
            support_zones, resistance_zones = detector.identify_zones(live_data)
            
            result = {
                'name': params['name'],
                'support_count': len(support_zones),
                'resistance_count': len(resistance_zones),
                'total_zones': len(support_zones) + len(resistance_zones),
                'avg_strength': np.mean([z.strength for z in support_zones + resistance_zones]) if (support_zones or resistance_zones) else 0
            }
            results.append(result)
            
            print(f"  Support: {result['support_count']}, Resistance: {result['resistance_count']}, "
                  f"Total: {result['total_zones']}, Avg Strength: {result['avg_strength']:.2f}")
        
        # Summary table
        print(f"\n📊 PARAMETER COMPARISON")
        print(f"{'='*60}")
        print(f"{'Parameters':<15} {'Support':<8} {'Resistance':<10} {'Total':<6} {'Avg Strength':<12}")
        print(f"{'-'*60}")
        
        for result in results:
            print(f"{result['name']:<15} {result['support_count']:<8} {result['resistance_count']:<10} "
                  f"{result['total_zones']:<6} {result['avg_strength']:<12.2f}")
        
    except Exception as e:
        print(f"❌ Error during parameter test: {e}")

def demo_live_simulation(symbol: str = 'BTCUSDT', duration_minutes: int = 10):
    """Simulate live detection with historical data"""
    print(f"\n🚀 LIVE SIMULATION DEMO")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"{'='*60}")
    
    try:
        # Load data
        df = load_historical_data(symbol)
        
        # Create detector
        detector = LiveSupportResistanceDetector(
            symbol=symbol.replace('USDT', '/USDT'),
            timeframe='5m',
            window_size=200,
            enable_charts=False,
            enable_alerts=True  # Enable alerts for demo
        )
        
        # Simulate live updates
        start_idx = len(df) - 200
        end_idx = len(df)
        step_size = 10  # Update every 10 candles
        
        print(f"📊 Starting live simulation...")
        print(f"Press Ctrl+C to stop early")
        
        for i in range(start_idx, end_idx, step_size):
            # Get data chunk
            chunk_data = df.iloc[i:i+200]
            
            # Update detector data
            detector.candles.clear()
            detector.candles.extend(chunk_data.to_dict('records'))
            
            # Identify zones
            support_zones, resistance_zones = detector.identify_zones(chunk_data)
            detector.support_zones = support_zones
            detector.resistance_zones = resistance_zones
            
            # Update current price
            current_price = chunk_data['close'].iloc[-1]
            detector.current_price = current_price
            detector.last_update = datetime.now()
            
            # Check interactions
            breaks = detector.check_zone_breaks(current_price)
            approaches = detector.check_zone_approaches(current_price)
            
            # Print status
            print(f"\n⏰ {chunk_data.index[-1]} | Price: ${current_price:.4f}")
            print(f"   Support Zones: {len(support_zones)}, Resistance Zones: {len(resistance_zones)}")
            
            if breaks:
                print(f"   🚨 BREAKS: {len(breaks)}")
            if approaches:
                print(f"   ⚠️ APPROACHES: {len(approaches)}")
            
            # Simulate time passing
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")

def main():
    """Main test function"""
    print("🧪 SUPPORT/RESISTANCE DETECTOR TEST SUITE")
    print("="*60)
    print("This script tests the detector using historical data")
    print("="*60)
    
    while True:
        print(f"\n📋 TEST OPTIONS:")
        print(f"1. Test single symbol (BTCUSDT)")
        print(f"2. Test multiple symbols")
        print(f"3. Parameter sensitivity test")
        print(f"4. Live simulation demo")
        print(f"5. Exit")
        
        choice = input(f"\nSelect option (1-5): ").strip()
        
        if choice == '1':
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            test_detector_with_historical_data(symbol)
            
        elif choice == '2':
            test_multiple_symbols()
            
        elif choice == '3':
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            test_parameter_sensitivity(symbol)
            
        elif choice == '4':
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            duration = input("Enter duration in minutes (default: 10): ").strip()
            duration = int(duration) if duration.isdigit() else 10
            demo_live_simulation(symbol, duration)
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-5.")

if __name__ == "__main__":
    main() 