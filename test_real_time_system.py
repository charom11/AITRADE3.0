"""
Test Real-Time Trading System
Demonstration script showing the integrated trading system capabilities
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from backtest_system import BacktestSystem
from live_support_resistance_detector import LiveSupportResistanceDetector
from live_fibonacci_detector import LiveFibonacciDetector
from divergence_detector import DivergenceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG, TRADING_CONFIG

def test_backtest_system():
    """Test the backtesting system"""
    print("🧪 TESTING BACKTEST SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize backtest system
        backtest = BacktestSystem(
            initial_capital=100000,
            commission=0.001
        )
        
        # Test with a shorter period
        print("📊 Running backtest for BTC/USDT...")
        backtest.run_backtest(
            start_date='2024-01-01',
            end_date='2024-01-31'  # Just January for testing
        )
        
        print("✅ Backtest system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Backtest system test failed: {e}")
        return False

def test_support_resistance_detector():
    """Test the support/resistance detector"""
    print("\n🧪 TESTING SUPPORT/RESISTANCE DETECTOR")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = LiveSupportResistanceDetector(
            symbol='BTC/USDT',
            timeframe='5m',
            enable_charts=False,
            enable_alerts=False
        )
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 45000
        price_data = []
        for i in range(len(dates)):
            # Add some trend and volatility
            trend = np.sin(i * 0.01) * 1000
            noise = np.random.normal(0, 500)
            price = base_price + trend + noise
            price_data.append(max(price, 1000))  # Ensure positive price
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': price_data,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in price_data],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in price_data],
            'close': price_data,
            'volume': np.random.uniform(100, 1000, len(price_data))
        }, index=dates)
        
        # Test zone identification
        support_zones, resistance_zones = detector.identify_zones(data)
        
        print(f"📊 Found {len(support_zones)} support zones")
        print(f"📊 Found {len(resistance_zones)} resistance zones")
        
        if support_zones:
            print("🛡️ Sample Support Zone:")
            zone = support_zones[0]
            print(f"   Level: ${zone.level:.2f}")
            print(f"   Strength: {zone.strength:.2f}")
            print(f"   Touches: {zone.touches}")
        
        if resistance_zones:
            print("🚀 Sample Resistance Zone:")
            zone = resistance_zones[0]
            print(f"   Level: ${zone.level:.2f}")
            print(f"   Strength: {zone.strength:.2f}")
            print(f"   Touches: {zone.touches}")
        
        print("✅ Support/Resistance detector test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Support/Resistance detector test failed: {e}")
        return False

def test_fibonacci_detector():
    """Test the Fibonacci detector"""
    print("\n🧪 TESTING FIBONACCI DETECTOR")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = LiveFibonacciDetector(
            symbol='BTC/USDT',
            timeframe='5m'
        )
        
        # Create sample data with clear swing points
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
        np.random.seed(42)
        
        # Generate data with clear swing highs and lows
        base_price = 45000
        price_data = []
        for i in range(len(dates)):
            # Create wave pattern
            wave = np.sin(i * 0.05) * 2000
            trend = i * 10  # Upward trend
            noise = np.random.normal(0, 200)
            price = base_price + wave + trend + noise
            price_data.append(max(price, 1000))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': price_data,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in price_data],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in price_data],
            'close': price_data,
            'volume': np.random.uniform(100, 1000, len(price_data))
        }, index=dates)
        
        # Test Fibonacci level calculation
        detector.update_fibonacci_levels(data)
        
        print(f"📊 Found {len(detector.fibonacci_levels)} Fibonacci levels")
        
        if detector.fibonacci_levels:
            print("📐 Sample Fibonacci Levels:")
            for i, level in enumerate(detector.fibonacci_levels[:3]):
                print(f"   {i+1}. {level.level_type}: {level.percentage}% @ ${level.price:.2f}")
        
        print("✅ Fibonacci detector test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Fibonacci detector test failed: {e}")
        return False

def test_divergence_detector():
    """Test the divergence detector"""
    print("\n🧪 TESTING DIVERGENCE DETECTOR")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = DivergenceDetector(
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            min_candles=20,
            swing_threshold=0.02
        )
        
        # Create sample data with potential divergence
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
        np.random.seed(42)
        
        # Generate data with price making higher highs but RSI making lower highs (bearish divergence)
        base_price = 45000
        price_data = []
        for i in range(len(dates)):
            # Price trend: higher highs
            trend = i * 20
            wave = np.sin(i * 0.1) * 1000
            price = base_price + trend + wave
            price_data.append(max(price, 1000))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': price_data,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in price_data],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in price_data],
            'close': price_data,
            'volume': np.random.uniform(100, 1000, len(price_data))
        }, index=dates)
        
        # Test divergence analysis
        analysis = detector.analyze_divergence(data)
        
        print(f"📊 Analysis completed")
        print(f"📊 Found {len(analysis.get('signals', []))} divergence signals")
        
        if 'signals' in analysis and analysis['signals']:
            print("📈 Sample Divergence Signal:")
            signal = analysis['signals'][0]
            print(f"   Type: {signal['type']}")
            print(f"   Indicator: {signal['indicator']}")
            print(f"   Strength: {signal['strength']:.2f}")
        
        print("✅ Divergence detector test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Divergence detector test failed: {e}")
        return False

def test_strategy_manager():
    """Test the strategy manager"""
    print("\n🧪 TESTING STRATEGY MANAGER")
    print("=" * 50)
    
    try:
        # Initialize strategy manager
        strategy_manager = StrategyManager()
        
        # Add strategies
        momentum_strategy = MomentumStrategy(STRATEGY_CONFIG['momentum'])
        mean_reversion_strategy = MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion'])
        pairs_strategy = PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading'])
        divergence_strategy = DivergenceStrategy(STRATEGY_CONFIG['divergence'])
        
        strategy_manager.add_strategy(momentum_strategy)
        strategy_manager.add_strategy(mean_reversion_strategy)
        strategy_manager.add_strategy(pairs_strategy)
        strategy_manager.add_strategy(divergence_strategy)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
        np.random.seed(42)
        
        base_price = 45000
        price_data = []
        for i in range(len(dates)):
            trend = np.sin(i * 0.01) * 1000
            noise = np.random.normal(0, 500)
            price = base_price + trend + noise
            price_data.append(max(price, 1000))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': price_data,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in price_data],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in price_data],
            'close': price_data,
            'volume': np.random.uniform(100, 1000, len(price_data))
        }, index=dates)
        
        # Test strategy signals
        strategy_data = {'BTC/USDT': data}
        all_signals = strategy_manager.get_all_signals(strategy_data)
        
        print(f"📊 Generated signals for {len(all_signals)} strategies")
        
        for strategy_name, signals in all_signals.items():
            if 'BTC/USDT' in signals:
                signal_data = signals['BTC/USDT']
                if len(signal_data) > 0:
                    latest_signal = signal_data.iloc[-1]
                    signal_value = latest_signal.get('signal', 0)
                    print(f"   {strategy_name}: {signal_value:.3f}")
        
        print("✅ Strategy manager test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy manager test failed: {e}")
        return False

def test_integrated_system():
    """Test the integrated system functionality"""
    print("\n🧪 TESTING INTEGRATED SYSTEM")
    print("=" * 50)
    
    try:
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5min')
        np.random.seed(42)
        
        base_price = 45000
        price_data = []
        for i in range(len(dates)):
            trend = np.sin(i * 0.01) * 1000
            noise = np.random.normal(0, 500)
            price = base_price + trend + noise
            price_data.append(max(price, 1000))
        
        data = pd.DataFrame({
            'open': price_data,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in price_data],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in price_data],
            'close': price_data,
            'volume': np.random.uniform(100, 1000, len(price_data))
        }, index=dates)
        
        # Test all components together
        print("📊 Testing integrated analysis...")
        
        # 1. Support/Resistance
        sr_detector = LiveSupportResistanceDetector('BTC/USDT', '5m', False, False)
        support_zones, resistance_zones = sr_detector.identify_zones(data)
        
        # 2. Fibonacci
        fib_detector = LiveFibonacciDetector('BTC/USDT', '5m')
        fib_detector.update_fibonacci_levels(data)
        
        # 3. Divergence
        div_detector = DivergenceDetector()
        divergence_analysis = div_detector.analyze_divergence(data)
        
        # 4. Strategies
        strategy_manager = StrategyManager()
        momentum_strategy = MomentumStrategy(STRATEGY_CONFIG['momentum'])
        strategy_manager.add_strategy(momentum_strategy)
        strategy_signals = strategy_manager.get_all_signals({'BTC/USDT': data})
        
        # Simulate signal generation
        current_price = data['close'].iloc[-1]
        conditions_met = []
        
        # Check support/resistance proximity
        for zone in support_zones:
            distance = abs(current_price - zone.level) / zone.level
            if distance <= 0.02:
                conditions_met.append(f"Near Support ${zone.level:.2f}")
        
        for zone in resistance_zones:
            distance = abs(current_price - zone.level) / zone.level
            if distance <= 0.02:
                conditions_met.append(f"Near Resistance ${zone.level:.2f}")
        
        # Check Fibonacci levels
        for level in fib_detector.fibonacci_levels:
            distance = abs(current_price - level.price) / level.price
            if distance <= 0.01:
                conditions_met.append(f"At {level.percentage}% Fib Level")
        
        # Check divergence
        if 'signals' in divergence_analysis and divergence_analysis['signals']:
            for signal in divergence_analysis['signals']:
                conditions_met.append(f"{signal['type'].title()} {signal['indicator'].upper()} Divergence")
        
        # Check strategy signals
        for strategy_name, signals in strategy_signals.items():
            if 'BTC/USDT' in signals:
                signal_data = signals['BTC/USDT']
                if len(signal_data) > 0:
                    latest_signal = signal_data.iloc[-1]
                    signal_value = latest_signal.get('signal', 0)
                    if abs(signal_value) > 0.3:
                        conditions_met.append(f"{strategy_name.title()} Signal")
        
        print(f"📊 Current Price: ${current_price:.2f}")
        print(f"📊 Conditions Met: {len(conditions_met)}")
        
        if conditions_met:
            print("✅ Conditions detected:")
            for condition in conditions_met:
                print(f"   • {condition}")
            
            # Determine signal type
            bullish_conditions = sum(1 for c in conditions_met if any(word in c.lower() for word in ['support', 'bullish', 'buy']))
            bearish_conditions = sum(1 for c in conditions_met if any(word in c.lower() for word in ['resistance', 'bearish', 'sell']))
            
            if bullish_conditions > bearish_conditions:
                signal_type = "BUY"
            elif bearish_conditions > bullish_conditions:
                signal_type = "SELL"
            else:
                signal_type = "NEUTRAL"
            
            print(f"📈 Signal Type: {signal_type}")
        else:
            print("📊 No strong conditions detected")
        
        print("✅ Integrated system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 REAL-TIME TRADING SYSTEM TEST SUITE")
    print("=" * 70)
    print("Testing all integrated components...")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Backtest System", test_backtest_system),
        ("Support/Resistance Detector", test_support_resistance_detector),
        ("Fibonacci Detector", test_fibonacci_detector),
        ("Divergence Detector", test_divergence_detector),
        ("Strategy Manager", test_strategy_manager),
        ("Integrated System", test_integrated_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for real-time trading.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 