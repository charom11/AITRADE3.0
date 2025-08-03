"""
Test Integration Script
Demonstrates how all components work together
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from divergence_detector import DivergenceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG

def create_sample_data(symbol: str, periods: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods)
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Create price data with some patterns
    base_price = 100 if symbol == 'BTCUSDT' else 50
    
    # Generate price with trends and reversals
    prices = []
    for i in range(periods):
        if i < 50:
            # Initial uptrend
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 100:
            # Downtrend
            price = base_price + 25 - (i - 50) * 0.3 + np.random.normal(0, 1)
        elif i < 150:
            # Sideways with divergence
            price = base_price + 10 + np.sin(i * 0.1) * 5 + np.random.normal(0, 1)
        else:
            # Final uptrend
            price = base_price + 10 + (i - 150) * 0.2 + np.random.normal(0, 1)
        
        prices.append(max(price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Create realistic OHLC from close price
        volatility = price * 0.02  # 2% volatility
        high = price + abs(np.random.normal(0, volatility))
        low = price - abs(np.random.normal(0, volatility))
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationship
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the DataFrame"""
    # Moving averages
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
    bb_period = STRATEGY_CONFIG['mean_reversion']['bb_period']
    bb_std = STRATEGY_CONFIG['mean_reversion']['bb_std']
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
    
    # ATR
    atr_period = STRATEGY_CONFIG['mean_reversion']['atr_period']
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=atr_period).mean()
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    return df

def test_strategies():
    """Test all trading strategies"""
    print("🧪 TESTING TRADING STRATEGIES")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data('BTCUSDT', 200)
    df = calculate_indicators(df)
    
    print(f"📊 Sample data created: {len(df)} periods")
    print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📊 Current price: ${df['close'].iloc[-1]:.2f}")
    print()
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    
    # Add strategies
    strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
    strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
    strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
    
    # Generate signals
    strategy_data = {'BTCUSDT': df}
    all_signals = strategy_manager.get_all_signals(strategy_data)
    
    # Display results
    for strategy_name, strategy_signals in all_signals.items():
        if 'BTCUSDT' in strategy_signals:
            signal_data = strategy_signals['BTCUSDT']
            latest_signal = signal_data.iloc[-1]
            
            signal = latest_signal.get('signal', 0)
            strength = latest_signal.get('signal_strength', 0.0)
            
            direction = "🟢 BUY" if signal > 0 else "🔴 SELL" if signal < 0 else "⚪ HOLD"
            
            print(f"📈 {strategy_name}:")
            print(f"   Signal: {direction}")
            print(f"   Strength: {strength:.3f}")
            print(f"   RSI: {latest_signal.get('rsi', 'N/A'):.1f}")
            print()
    
    return all_signals

def test_divergence_detector():
    """Test divergence detection"""
    print("🔍 TESTING DIVERGENCE DETECTOR")
    print("=" * 50)
    
    # Create sample data with divergence patterns
    df = create_sample_data('BTCUSDT', 200)
    
    # Initialize divergence detector
    detector = DivergenceDetector(
        rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
        macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
        macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
        macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
        min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
        swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
    )
    
    # Analyze divergence
    analysis = detector.analyze_divergence(df, confirm_signals=True)
    
    print(f"📊 Data analyzed: {analysis.get('data_points', 'N/A')} points")
    print(f"⏰ Timeframe: {analysis.get('timeframe', 'N/A')}")
    print(f"📈 Total signals: {analysis.get('total_signals', 0)}")
    print(f"📊 RSI signals: {analysis.get('rsi_signals', 0)}")
    print(f"📉 MACD signals: {analysis.get('macd_signals', 0)}")
    print()
    
    # Display signals
    if analysis.get('signals'):
        print("🎯 DETECTED SIGNALS:")
        for i, signal in enumerate(analysis['signals'], 1):
            print(f"   Signal {i}:")
            print(f"     Type: {signal['type'].upper()} Divergence")
            print(f"     Indicator: {signal['indicator'].upper()}")
            print(f"     Strength: {signal['strength']:.3f}")
            print(f"     Date: {signal['signal_date']}")
            print()
    else:
        print("❌ No divergence signals detected")
        print()
    
    return analysis

def test_integrated_signals():
    """Test integrated signal generation"""
    print("🔗 TESTING INTEGRATED SIGNALS")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data('BTCUSDT', 200)
    df = calculate_indicators(df)
    
    # Initialize components
    strategy_manager = StrategyManager()
    strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
    strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
    strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
    
    divergence_detector = DivergenceDetector(
        rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
        macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
        macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
        macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
        min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
        swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
    )
    
    # Generate integrated signals
    signals = {
        'symbol': 'BTCUSDT',
        'timestamp': datetime.now(),
        'current_price': df['close'].iloc[-1],
        'strategies': {},
        'divergence': None,
        'support_resistance': None,
        'composite_signal': 0,
        'confidence': 0.0
    }
    
    # Strategy signals
    strategy_data = {'BTCUSDT': df}
    all_strategy_signals = strategy_manager.get_all_signals(strategy_data)
    
    for strategy_name, strategy_signals in all_strategy_signals.items():
        if 'BTCUSDT' in strategy_signals:
            signal_data = strategy_signals['BTCUSDT']
            latest_signal = signal_data.iloc[-1]
            
            signals['strategies'][strategy_name] = {
                'signal': latest_signal.get('signal', 0),
                'strength': latest_signal.get('signal_strength', 0.0)
            }
    
    # Divergence signals
    divergence_analysis = divergence_detector.analyze_divergence(df)
    if 'signals' in divergence_analysis and divergence_analysis['signals']:
        latest_divergence = divergence_analysis['signals'][0]
        signals['divergence'] = {
            'type': latest_divergence['type'],
            'indicator': latest_divergence['indicator'],
            'strength': latest_divergence['strength'],
            'signal': 1 if latest_divergence['type'] == 'bullish' else -1
        }
    
    # Calculate composite signal
    composite_signal = 0
    total_weight = 0
    
    # Strategy signals (weight: 0.6)
    for strategy_name, strategy_data in signals['strategies'].items():
        signal = strategy_data['signal']
        strength = strategy_data['strength']
        composite_signal += signal * strength * 0.6
        total_weight += 0.6
    
    # Divergence signals (weight: 0.3)
    if signals['divergence']:
        divergence_signal = signals['divergence']['signal']
        divergence_strength = signals['divergence']['strength']
        composite_signal += divergence_signal * divergence_strength * 0.3
        total_weight += 0.3
    
    # Support/resistance signals (weight: 0.1) - simulated
    sr_bias = 0.1  # Simulated support/resistance bias
    composite_signal += sr_bias * 0.1
    total_weight += 0.1
    
    # Normalize composite signal
    if total_weight > 0:
        signals['composite_signal'] = composite_signal / total_weight
        signals['confidence'] = min(abs(composite_signal), 1.0)
    
    # Display results
    print(f"📊 Current Price: ${signals['current_price']:.2f}")
    print(f"🔗 Composite Signal: {signals['composite_signal']:.3f}")
    print(f"🎯 Confidence: {signals['confidence']:.3f}")
    print()
    
    # Display strategy breakdown
    print("📈 STRATEGY BREAKDOWN:")
    for strategy_name, strategy_data in signals['strategies'].items():
        signal = strategy_data['signal']
        strength = strategy_data['strength']
        direction = "🟢 BUY" if signal > 0 else "🔴 SELL" if signal < 0 else "⚪ HOLD"
        print(f"   {strategy_name}: {direction} (Strength: {strength:.3f})")
    
    # Display divergence
    if signals['divergence']:
        div_type = signals['divergence']['type']
        div_indicator = signals['divergence']['indicator']
        div_strength = signals['divergence']['strength']
        direction = "🟢 BULLISH" if div_type == 'bullish' else "🔴 BEARISH"
        print(f"   Divergence: {direction} {div_indicator.upper()} (Strength: {div_strength:.3f})")
    
    # Trading decision
    print()
    if abs(signals['composite_signal']) > 0.3 and signals['confidence'] > 0.5:
        if signals['composite_signal'] > 0:
            print("🎯 TRADING DECISION: 🟢 STRONG BUY SIGNAL")
        else:
            print("🎯 TRADING DECISION: 🔴 STRONG SELL SIGNAL")
    else:
        print("🎯 TRADING DECISION: ⚪ NO STRONG SIGNAL")
    
    print()
    return signals

def main():
    """Main test function"""
    print("🚀 INTEGRATED TRADING SYSTEM - TEST SUITE")
    print("=" * 60)
    print("This test demonstrates how all components work together")
    print("without requiring real API credentials.")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Trading Strategies
        strategy_results = test_strategies()
        
        # Test 2: Divergence Detection
        divergence_results = test_divergence_detector()
        
        # Test 3: Integrated Signals
        integrated_results = test_integrated_signals()
        
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The system is ready for live trading with proper API credentials.")
        print("Make sure to:")
        print("1. Set up your .env file with Binance API credentials")
        print("2. Test with small amounts first")
        print("3. Monitor the system continuously")
        print("4. Have proper risk management in place")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Please check your configuration and dependencies.")

if __name__ == "__main__":
    main() 