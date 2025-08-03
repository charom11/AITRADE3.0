"""
Integrated Trading System
Connects all strategies and detectors for comprehensive trading
"""

import os
import sys
import time
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import warnings
import logging
warnings.filterwarnings('ignore')

# Import our modules
from divergence_detector import DivergenceDetector
from live_support_resistance_detector import LiveSupportResistanceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG, TRADING_CONFIG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    """Integrated trading system with all strategies and detectors"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            print("❌ API credentials not found!")
            return
        
        # Initialize exchange
        self.exchange = self._setup_exchange()
        
        # Trading parameters
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.leverage = 100
        self.auto_trading = True
        
        # Initialize strategies and detectors
        self._initialize_components()
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0
        }
        
        print("🚀 INTEGRATED TRADING SYSTEM INITIALIZED")
        print("=" * 60)
        print(f"📈 Strategies: {list(self.strategy_manager.strategies.keys())}")
        print(f"🔍 Detectors: Divergence, Support/Resistance")
        print("=" * 60)
    
    def _setup_exchange(self):
        """Setup Binance exchange connection"""
        try:
            exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': False,
                'options': {'defaultType': 'future'}
            })
            
            exchange.load_markets()
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            print(f"✅ Connected to Binance Live")
            print(f"💰 USDT Balance: ${usdt_balance:.4f}")
            
            return exchange
            
        except Exception as e:
            print(f"❌ Error connecting to Binance: {e}")
            return None
    
    def _initialize_components(self):
        """Initialize all strategies and detectors"""
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        
        # Add strategies
        self.strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
        self.strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
        self.strategy_manager.add_strategy(PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading']))
        self.strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
        
        # Initialize divergence detector
        self.divergence_detector = DivergenceDetector(
            rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
            macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
            macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
            macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
            min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
            swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
        )
        
        # Initialize support/resistance detectors
        self.sr_detectors = {}
        for symbol in self.trading_pairs:
            detector = LiveSupportResistanceDetector(
                symbol=symbol,
                timeframe='15m',
                enable_charts=False,
                enable_alerts=False
            )
            self.sr_detectors[symbol] = detector
        
        logger.info("All components initialized successfully")
    
    def fetch_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process market data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=200)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        bb_period = STRATEGY_CONFIG['mean_reversion']['bb_period']
        bb_std = STRATEGY_CONFIG['mean_reversion']['bb_std']
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        
        # ATR
        atr_period = STRATEGY_CONFIG['mean_reversion']['atr_period']
        df['atr'] = self._calculate_atr(df, atr_period)
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def generate_integrated_signals(self, symbol: str) -> Dict:
        """Generate integrated trading signals using all components"""
        try:
            # Fetch market data
            df = self.fetch_market_data(symbol)
            if df is None or len(df) < 200:
                return None
            
            signals = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': df['close'].iloc[-1],
                'strategies': {},
                'divergence': None,
                'support_resistance': None,
                'composite_signal': 0,
                'confidence': 0.0
            }
            
            # Generate strategy signals
            strategy_data = {symbol: df}
            all_strategy_signals = self.strategy_manager.get_all_signals(strategy_data)
            
            for strategy_name, strategy_signals in all_strategy_signals.items():
                if symbol in strategy_signals:
                    signal_data = strategy_signals[symbol]
                    latest_signal = signal_data.iloc[-1]
                    
                    signals['strategies'][strategy_name] = {
                        'signal': latest_signal.get('signal', 0),
                        'strength': latest_signal.get('signal_strength', 0.0)
                    }
            
            # Generate divergence signals
            divergence_analysis = self.divergence_detector.analyze_divergence(df)
            if 'signals' in divergence_analysis and divergence_analysis['signals']:
                latest_divergence = divergence_analysis['signals'][0]
                signals['divergence'] = {
                    'type': latest_divergence['type'],
                    'indicator': latest_divergence['indicator'],
                    'strength': latest_divergence['strength'],
                    'signal': 1 if latest_divergence['type'] == 'bullish' else -1
                }
            
            # Get support/resistance levels
            if symbol in self.sr_detectors:
                detector = self.sr_detectors[symbol]
                current_price = df['close'].iloc[-1]
                
                support_levels = [zone.level for zone in detector.support_zones if zone.is_active]
                resistance_levels = [zone.level for zone in detector.resistance_zones if zone.is_active]
                
                if support_levels and resistance_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=None)
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                    
                    signals['support_resistance'] = {
                        'nearest_support': nearest_support,
                        'nearest_resistance': nearest_resistance,
                        'support_distance': (current_price - nearest_support) / current_price if nearest_support else None,
                        'resistance_distance': (nearest_resistance - current_price) / current_price if nearest_resistance else None
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
            
            # Support/resistance signals (weight: 0.1)
            if signals['support_resistance']:
                sr_data = signals['support_resistance']
                if sr_data['support_distance'] and sr_data['support_distance'] < 0.02:
                    composite_signal += 0.5 * 0.1  # Bullish bias near support
                elif sr_data['resistance_distance'] and sr_data['resistance_distance'] < 0.02:
                    composite_signal += -0.5 * 0.1  # Bearish bias near resistance
                total_weight += 0.1
            
            # Normalize composite signal
            if total_weight > 0:
                signals['composite_signal'] = composite_signal / total_weight
                signals['confidence'] = min(abs(composite_signal), 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return None
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            print(f"\n📊 EXECUTING INTEGRATED TRADING CYCLE - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 60)
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f"💰 USDT Balance: ${usdt_balance:.4f}")
            
            # Process each trading pair
            for symbol in self.trading_pairs:
                try:
                    # Generate integrated signals
                    signals = self.generate_integrated_signals(symbol)
                    if not signals:
                        continue
                    
                    current_price = signals['current_price']
                    composite_signal = signals['composite_signal']
                    confidence = signals['confidence']
                    
                    print(f"\n📈 {symbol}: ${current_price:.4f}")
                    print(f"   Composite Signal: {composite_signal:.3f} (Confidence: {confidence:.2f})")
                    
                    # Display strategy signals
                    for strategy_name, strategy_data in signals['strategies'].items():
                        signal = strategy_data['signal']
                        strength = strategy_data['strength']
                        if signal != 0:
                            direction = "🟢 BUY" if signal > 0 else "🔴 SELL"
                            print(f"   {strategy_name}: {direction} (Strength: {strength:.2f})")
                    
                    # Display divergence signal
                    if signals['divergence']:
                        div_type = signals['divergence']['type']
                        div_indicator = signals['divergence']['indicator']
                        div_strength = signals['divergence']['strength']
                        direction = "🟢 BULLISH" if div_type == 'bullish' else "🔴 BEARISH"
                        print(f"   Divergence: {direction} {div_indicator.upper()} (Strength: {div_strength:.2f})")
                    
                    # Display support/resistance info
                    if signals['support_resistance']:
                        sr_data = signals['support_resistance']
                        if sr_data['nearest_support']:
                            print(f"   Support: ${sr_data['nearest_support']:.4f} ({sr_data['support_distance']*100:.1f}% away)")
                        if sr_data['nearest_resistance']:
                            print(f"   Resistance: ${sr_data['nearest_resistance']:.4f} ({sr_data['resistance_distance']*100:.1f}% away)")
                    
                    # Execute trades based on composite signal
                    if abs(composite_signal) > 0.3 and confidence > 0.5:
                        if composite_signal > 0:
                            print(f"   🟢 STRONG BUY SIGNAL")
                        else:
                            print(f"   🔴 STRONG SELL SIGNAL")
                    else:
                        print(f"   ⚪ No strong signal")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Display performance summary
            self._display_performance_summary()
            
        except Exception as e:
            logger.error(f"Error executing trading cycle: {e}")
    
    def _display_performance_summary(self):
        """Display performance summary"""
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Winning Trades: {self.performance_metrics['winning_trades']}")
        
        if self.performance_metrics['total_trades'] > 0:
            win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            print(f"Win Rate: {win_rate*100:.1f}%")
        
        print(f"Total P&L: ${self.performance_metrics['total_pnl']:.4f}")
    
    def run_trading_session(self, duration_minutes: int = 60, update_interval: int = 30):
        """Run integrated trading session"""
        print(f"\n🚀 STARTING INTEGRATED TRADING SESSION")
        print("=" * 60)
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🔄 Update Interval: {update_interval} seconds")
        print(f"📈 Strategies: {list(self.strategy_manager.strategies.keys())}")
        print(f"🔍 Detectors: Divergence, Support/Resistance")
        print(f"🛑 Press Ctrl+C to stop early")
        print("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            iteration = 0
            while datetime.now() < end_time:
                iteration += 1
                current_time = datetime.now()
                
                print(f"\n--- Trading Cycle {iteration} ({current_time.strftime('%H:%M:%S')}) ---")
                
                # Execute trading cycle
                self.execute_trading_cycle()
                
                # Check if we should stop
                if current_time >= end_time:
                    print(f"\n⏰ Trading session completed!")
                    break
                
                # Wait for next update
                remaining_time = (end_time - current_time).total_seconds()
                if remaining_time > update_interval:
                    print(f"\n⏳ Waiting {update_interval} seconds...")
                    time.sleep(update_interval)
                else:
                    print(f"\n⏳ Waiting {remaining_time:.0f} seconds...")
                    time.sleep(remaining_time)
        
        except KeyboardInterrupt:
            print(f"\n🛑 Trading stopped by user!")
        
        except Exception as e:
            print(f"\n❌ Error in trading session: {e}")
        
        finally:
            # Final performance summary
            print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
            print("=" * 60)
            self._display_performance_summary()
            print(f"✅ Integrated trading session completed!")

def main():
    """Main function to run integrated trading"""
    print("🚀 INTEGRATED TRADING SYSTEM")
    print("=" * 60)
    
    # Initialize trading system
    trading_system = IntegratedTradingSystem()
    
    if not trading_system.exchange:
        print("❌ Failed to initialize trading system")
        return
    
    # Confirm real trading
    print("\n⚠️  WARNING: This will use REAL MONEY!")
    print("💰 Your current balance will be used for trading")
    print("⚡ Trading with 100x leverage")
    print("📈 Using integrated strategies and detectors")
    
    confirm = input("\nAre you sure you want to proceed with REAL MONEY trading? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Trading cancelled by user")
        return
    
    # Get trading parameters
    try:
        duration = int(input("Enter trading duration in minutes (default 60): ") or "60")
        interval = int(input("Enter update interval in seconds (default 30): ") or "30")
    except ValueError:
        duration = 60
        interval = 30
    
    # Start trading session
    trading_system.run_trading_session(duration_minutes=duration, update_interval=interval)

if __name__ == "__main__":
    main() 