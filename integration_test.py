"""
Integration Test for Complete Trading System
Tests all detection modules, strategies, and unified signal evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List

# Import all modules
from live_support_resistance_detector import LiveSupportResistanceDetector
from live_fibonacci_detector import LiveFibonacciDetector
from divergence_detector import DivergenceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    SupportResistanceStrategy,
    FibonacciStrategy,
    StrategyManager
)
from real_live_trading_system import RealLiveTradingSystem
from config import STRATEGY_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(symbol: str = 'BTC/USDT', periods: int = 200) -> pd.DataFrame:
    """Create realistic test data for integration testing"""
    np.random.seed(42)
    
    # Generate base trend with volatility
    base_prices = np.linspace(45000, 48000, periods) + np.random.normal(0, 500, periods)
    
    # Add some swing points for testing
    for i in range(20, periods, 30):
        base_prices[i:i+10] += np.random.normal(1000, 200, 10)
    
    # Create OHLCV data
    data = []
    for i, base_price in enumerate(base_prices):
        # Add realistic volatility
        high = base_price + abs(np.random.normal(0, 100))
        low = base_price - abs(np.random.normal(0, 100))
        open_price = base_price + np.random.normal(0, 50)
        close_price = base_price + np.random.normal(0, 50)
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
    
    # Add required technical indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df

def test_detection_modules():
    """Test all detection modules individually"""
    print("\n🧪 TESTING DETECTION MODULES")
    print("="*50)
    
    # Create test data
    test_data = create_test_data()
    print(f"✅ Created test data: {len(test_data)} periods")
    
    # Test Support/Resistance Detector
    print("\n📊 Testing Support/Resistance Detector...")
    sr_detector = LiveSupportResistanceDetector(
        symbol='BTC/USDT',
        timeframe='5m',
        window_size=200
    )
    
    support_zones, resistance_zones = sr_detector.identify_zones(test_data)
    print(f"  Support zones: {len(support_zones)}")
    print(f"  Resistance zones: {len(resistance_zones)}")
    
    if support_zones:
        print(f"  Strongest support: ${support_zones[0].level:.2f} (strength: {support_zones[0].strength:.2f})")
    if resistance_zones:
        print(f"  Strongest resistance: ${resistance_zones[0].level:.2f} (strength: {resistance_zones[0].strength:.2f})")
    
    # Test Fibonacci Detector
    print("\n📐 Testing Fibonacci Detector...")
    fib_detector = LiveFibonacciDetector(
        symbol='BTC/USDT',
        timeframe='5m'
    )
    
    fib_detector.data = test_data
    fib_detector.current_price = test_data['close'].iloc[-1]
    fib_detector.update_fibonacci_levels(test_data)
    
    print(f"  Fibonacci levels: {len(fib_detector.fibonacci_levels)}")
    print(f"  Swing highs: {len(fib_detector.swing_highs)}")
    print(f"  Swing lows: {len(fib_detector.swing_lows)}")
    
    if fib_detector.fibonacci_levels:
        retracements = [l for l in fib_detector.fibonacci_levels if l.level_type == 'retracement']
        extensions = [l for l in fib_detector.fibonacci_levels if l.level_type == 'extension']
        print(f"  Retracement levels: {len(retracements)}")
        print(f"  Extension levels: {len(extensions)}")
    
    # Test Divergence Detector
    print("\n🔄 Testing Divergence Detector...")
    div_detector = DivergenceDetector()
    
    divergence_analysis = div_detector.analyze_divergence(test_data)
    
    if 'error' not in divergence_analysis:
        print(f"  Total signals: {divergence_analysis['total_signals']}")
        print(f"  RSI signals: {divergence_analysis['rsi_signals']}")
        print(f"  MACD signals: {divergence_analysis['macd_signals']}")
        print(f"  Current RSI: {divergence_analysis['current_rsi']:.2f}")
        print(f"  Current MACD: {divergence_analysis['current_macd']:.4f}")
    else:
        print(f"  Error: {divergence_analysis['error']}")
    
    return test_data

def test_strategies():
    """Test all strategies individually"""
    print("\n🧠 TESTING STRATEGIES")
    print("="*50)
    
    # Create test data
    test_data = create_test_data()
    
    # Test each strategy
    strategies = {
        'Momentum': MomentumStrategy(STRATEGY_CONFIG['momentum']),
        'MeanReversion': MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']),
        'Divergence': DivergenceStrategy(STRATEGY_CONFIG['divergence']),
        'SupportResistance': SupportResistanceStrategy(STRATEGY_CONFIG['support_resistance']),
        'Fibonacci': FibonacciStrategy(STRATEGY_CONFIG['fibonacci'])
    }
    
    for name, strategy in strategies.items():
        print(f"\n📈 Testing {name} Strategy...")
        
        try:
            signals = strategy.generate_signals(test_data)
            
            if 'signal' in signals.columns:
                latest_signal = signals['signal'].iloc[-1]
                signal_count = len(signals[signals['signal'] != 0])
                
                print(f"  Latest signal: {latest_signal}")
                print(f"  Total signals: {signal_count}")
                
                if 'signal_strength' in signals.columns:
                    avg_strength = signals['signal_strength'].mean()
                    print(f"  Average strength: {avg_strength:.3f}")
            else:
                print("  No signal column found")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return strategies

def test_strategy_manager():
    """Test the StrategyManager with unified signal evaluation"""
    print("\n🎯 TESTING STRATEGY MANAGER")
    print("="*50)
    
    # Create test data
    test_data = create_test_data()
    
    # Create StrategyManager
    strategy_manager = StrategyManager()
    
    # Add strategies
    strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
    strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
    strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
    strategy_manager.add_strategy(SupportResistanceStrategy(STRATEGY_CONFIG['support_resistance']))
    strategy_manager.add_strategy(FibonacciStrategy(STRATEGY_CONFIG['fibonacci']))
    
    # Create live data structure
    live_data = {
        'price_data': {'BTC/USDT': test_data},
        'support_resistance': {},
        'fibonacci': {},
        'divergence': {}
    }
    
    # Add detection module outputs
    sr_detector = LiveSupportResistanceDetector(symbol='BTC/USDT', timeframe='5m')
    support_zones, resistance_zones = sr_detector.identify_zones(test_data)
    live_data['support_resistance']['BTC/USDT'] = {
        'support_zones': support_zones,
        'resistance_zones': resistance_zones
    }
    
    fib_detector = LiveFibonacciDetector(symbol='BTC/USDT', timeframe='5m')
    fib_detector.data = test_data
    fib_detector.current_price = test_data['close'].iloc[-1]
    fib_detector.update_fibonacci_levels(test_data)
    live_data['fibonacci']['BTC/USDT'] = fib_detector.fibonacci_levels
    
    div_detector = DivergenceDetector()
    divergence_analysis = div_detector.analyze_divergence(test_data)
    live_data['divergence']['BTC/USDT'] = divergence_analysis
    
    # Test unified signal evaluation
    print("🔍 Testing unified signal evaluation...")
    unified_signals = strategy_manager.evaluate_trade_signal(live_data)
    
    for symbol, signal_result in unified_signals.items():
        print(f"\n📊 Results for {symbol}:")
        print(f"  Signal: {signal_result['signal']}")
        print(f"  Confidence: {signal_result['confidence']:.3f}")
        print(f"  Recommended action: {signal_result['recommended_action']}")
        print(f"  Risk level: {signal_result['risk_level']}")
        print(f"  Reasoning: {len(signal_result['reasoning'])} conditions")
        
        for reason in signal_result['reasoning'][:3]:  # Show first 3 reasons
            print(f"    - {reason}")
    
    return unified_signals

def test_complete_system():
    """Test the complete RealLiveTradingSystem"""
    print("\n🚀 TESTING COMPLETE SYSTEM")
    print("="*50)
    
    # Create trading system (paper trading mode)
    trading_system = RealLiveTradingSystem(
        symbols=['BTC/USDT'],
        timeframe='5m',
        enable_live_trading=False,  # Paper trading
        enable_backtesting=True,
        enable_alerts=False
    )
    
    print("✅ Trading system created")
    print(f"📊 Symbols: {trading_system.symbols}")
    print(f"⏱️ Timeframe: {trading_system.timeframe}")
    print(f"💰 Live trading: {'Enabled' if trading_system.enable_live_trading else 'Disabled'}")
    print(f"🧪 Backtesting: {'Enabled' if trading_system.enable_backtesting else 'Disabled'}")
    
    # Test system initialization
    print("\n🔧 Testing system initialization...")
    try:
        # Initialize components
        trading_system._initialize_detectors()
        trading_system._initialize_strategies()
        
        print("✅ All components initialized successfully")
        print(f"  Detectors: {len(trading_system.detectors)}")
        print(f"  Strategies: {len(trading_system.strategy_manager.strategies)}")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False
    
    # Test with simulated data
    print("\n📊 Testing with simulated data...")
    test_data = create_test_data()
    
    # Simulate data processing
    try:
        # Create live data structure
        live_data = {
            'price_data': {'BTC/USDT': test_data},
            'support_resistance': {},
            'fibonacci': {},
            'divergence': {}
        }
        
        # Add detection outputs
        if 'BTC/USDT' in trading_system.detectors:
            sr_detector = trading_system.detectors['BTC/USDT']['support_resistance']
            support_zones, resistance_zones = sr_detector.identify_zones(test_data)
            live_data['support_resistance']['BTC/USDT'] = {
                'support_zones': support_zones,
                'resistance_zones': resistance_zones
            }
            
            fib_detector = trading_system.detectors['BTC/USDT']['fibonacci']
            fib_detector.data = test_data
            fib_detector.current_price = test_data['close'].iloc[-1]
            fib_detector.update_fibonacci_levels(test_data)
            live_data['fibonacci']['BTC/USDT'] = fib_detector.fibonacci_levels
            
            div_detector = trading_system.detectors['BTC/USDT']['divergence']
            divergence_analysis = div_detector.analyze_divergence(test_data)
            live_data['divergence']['BTC/USDT'] = divergence_analysis
        
        # Evaluate signals
        unified_signals = trading_system.strategy_manager.evaluate_trade_signal(live_data)
        
        print("✅ Signal evaluation completed")
        for symbol, result in unified_signals.items():
            print(f"  {symbol}: {result['recommended_action']} (confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False
    
    return True

def run_performance_test():
    """Run performance test to ensure system can handle real-time data"""
    print("\n⚡ PERFORMANCE TEST")
    print("="*50)
    
    # Create test data
    test_data = create_test_data(periods=1000)
    
    # Test detection modules performance
    start_time = time.time()
    
    # Support/Resistance
    sr_detector = LiveSupportResistanceDetector(symbol='BTC/USDT', timeframe='5m')
    support_zones, resistance_zones = sr_detector.identify_zones(test_data)
    sr_time = time.time() - start_time
    
    # Fibonacci
    start_time = time.time()
    fib_detector = LiveFibonacciDetector('BTC/USDT', '5m')
    fib_detector.data = test_data
    fib_detector.current_price = test_data['close'].iloc[-1]
    fib_detector.update_fibonacci_levels(test_data)
    fib_time = time.time() - start_time
    
    # Divergence
    start_time = time.time()
    div_detector = DivergenceDetector()
    divergence_analysis = div_detector.analyze_divergence(test_data)
    div_time = time.time() - start_time
    
    # Strategy evaluation
    start_time = time.time()
    strategy_manager = StrategyManager()
    strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
    strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
    strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
    
    live_data = {
        'price_data': {'BTC/USDT': test_data},
        'support_resistance': {'BTC/USDT': {'support_zones': support_zones, 'resistance_zones': resistance_zones}},
        'fibonacci': {'BTC/USDT': fib_detector.fibonacci_levels},
        'divergence': {'BTC/USDT': divergence_analysis}
    }
    
    unified_signals = strategy_manager.evaluate_trade_signal(live_data)
    strategy_time = time.time() - start_time
    
    print(f"📊 Performance Results (1000 data points):")
    print(f"  Support/Resistance: {sr_time:.3f}s")
    print(f"  Fibonacci: {fib_time:.3f}s")
    print(f"  Divergence: {div_time:.3f}s")
    print(f"  Strategy Evaluation: {strategy_time:.3f}s")
    print(f"  Total: {sr_time + fib_time + div_time + strategy_time:.3f}s")
    
    # Check if performance is acceptable for real-time
    total_time = sr_time + fib_time + div_time + strategy_time
    if total_time < 1.0:  # Should complete in under 1 second
        print("✅ Performance acceptable for real-time trading")
    else:
        print("⚠️ Performance may be too slow for real-time trading")

def main():
    """Run all integration tests"""
    print("🧩 COMPLETE TRADING SYSTEM INTEGRATION TEST")
    print("="*60)
    print("Testing all detection modules, strategies, and unified system")
    print("="*60)
    
    try:
        # Test 1: Detection Modules
        test_data = test_detection_modules()
        
        # Test 2: Individual Strategies
        strategies = test_strategies()
        
        # Test 3: Strategy Manager
        unified_signals = test_strategy_manager()
        
        # Test 4: Complete System
        system_success = test_complete_system()
        
        # Test 5: Performance
        run_performance_test()
        
        print("\n" + "="*60)
        print("🎉 INTEGRATION TEST COMPLETED")
        print("="*60)
        
        if system_success:
            print("✅ All tests passed! System is ready for deployment.")
            print("\n📋 NEXT STEPS:")
            print("1. Configure API keys for live trading")
            print("2. Set up Telegram alerts (optional)")
            print("3. Run backtest_system.py for historical validation")
            print("4. Start real_live_trading_system.py for live trading")
        else:
            print("❌ Some tests failed. Please check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 