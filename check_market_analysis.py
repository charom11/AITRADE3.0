#!/usr/bin/env python3
"""
Market Analysis Checker
Check current market data and analysis results from the trading system
"""

import pandas as pd
import numpy as np
import ccxt
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

# Import our custom modules
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    SupportResistanceStrategy,
    FibonacciStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG
from live_support_resistance_detector import LiveSupportResistanceDetector
from live_fibonacci_detector import LiveFibonacciDetector
from divergence_detector import DivergenceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketAnalysisChecker:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': None,
            'secret': None,
            'sandbox': False,
            'enableRateLimit': True
        })
        
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'USDT/USDT', 'BNB/USDT', 'SOL/USDT', 'USDC/USDT',
            'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'HYPE/USDT', 'XLM/USDT', 'SUI/USDT', 'LINK/USDT',
            'HBAR/USDT', 'AVAX/USDT', 'BCH/USDT', 'SHIB/USDT', 'LTC/USDT', 'TON/USDT', 'LEO/USDT',
            'DOT/USDT', 'UNI/USDT', 'USDe/USDT', 'XMR/USDT', 'PEPE/USDT', 'BGB/USDT', 'DAI/USDT',
            'AAVE/USDT', 'TAO/USDT', 'CRO/USDT', 'NEAR/USDT', 'ETC/USDT', 'APT/USDT', 'ONDO/USDT',
            'PI/USDT', 'ICP/USDT', 'OKB/USDT', 'BONK/USDT', 'KAS/USDT', 'MNT/USDT', 'POL/USDT'
        ]
        self.timeframe = '5m'
        
        # Initialize strategies
        self.strategy_manager = StrategyManager()
        self.strategy_manager.add_strategy(MomentumStrategy(STRATEGY_CONFIG['momentum']))
        self.strategy_manager.add_strategy(MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']))
        self.strategy_manager.add_strategy(PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading']))
        self.strategy_manager.add_strategy(DivergenceStrategy(STRATEGY_CONFIG['divergence']))
        self.strategy_manager.add_strategy(SupportResistanceStrategy(STRATEGY_CONFIG['support_resistance']))
        self.strategy_manager.add_strategy(FibonacciStrategy(STRATEGY_CONFIG['fibonacci']))
        
        # Initialize detectors
        self.detectors = {}
        for symbol in self.symbols:
            self.detectors[symbol] = {
                'support_resistance': LiveSupportResistanceDetector(symbol, 'binance', self.timeframe),
                'fibonacci': LiveFibonacciDetector('binance', symbol, self.timeframe),
                'divergence': DivergenceDetector()
            }
    
    def fetch_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent market data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a single symbol with all strategies and detectors"""
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING {symbol}")
        print(f"{'='*60}")
        
        # Fetch data
        data = self.fetch_market_data(symbol)
        if data.empty:
            return {}
        
        current_price = data['close'].iloc[-1]
        
        # Calculate basic indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['rsi'] = self._calculate_rsi(data['close'], 14)
        data['atr'] = self._calculate_atr(data, 14)
        
        # Get latest values
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        atr = data['atr'].iloc[-1]
        
        print(f"💰 Current Price: ${current_price:.4f}")
        print(f"📈 24h Change: {((current_price / data['close'].iloc[-288] - 1) * 100):.2f}%" if len(data) > 288 else "N/A")
        print(f"📊 Volume: {data['volume'].iloc[-1]:,.0f}")
        print(f"📊 SMA 20: ${sma_20:.4f} | SMA 50: ${sma_50:.4f}")
        print(f"📊 RSI: {rsi:.2f} | ATR: ${atr:.4f}")
        
        # Market trend analysis
        if current_price > sma_20 > sma_50:
            trend = "🟢 BULLISH"
        elif current_price < sma_20 < sma_50:
            trend = "🔴 BEARISH"
        else:
            trend = "🟡 SIDEWAYS"
        print(f"📈 Trend: {trend}")
        
        # RSI analysis
        if rsi > 70:
            rsi_status = "🔴 OVERBOUGHT"
        elif rsi < 30:
            rsi_status = "🟢 OVERSOLD"
        else:
            rsi_status = "🟡 NEUTRAL"
        print(f"📊 RSI Status: {rsi_status}")
        
        # Get strategy signals
        strategy_signals = {}
        for strategy_name, strategy in self.strategy_manager.strategies.items():
            try:
                if isinstance(strategy, PairsTradingStrategy):
                    # Pairs trading needs multiple symbols
                    continue
                else:
                    signals = strategy.generate_signals(data)
                    if not signals.empty:
                        latest_signal = signals.iloc[-1]
                        if latest_signal['signal'] != 0:
                            strategy_signals[strategy_name] = {
                                'signal': latest_signal['signal'],
                                'strength': latest_signal.get('signal_strength', 0),
                                'confidence': abs(latest_signal.get('signal_strength', 0))
                            }
            except Exception as e:
                logger.error(f"Error in {strategy_name}: {e}")
        
        # Get detector analysis
        detector_analysis = {}
        
        # Support/Resistance
        try:
            sr_detector = self.detectors[symbol]['support_resistance']
            sr_detector.data = data
            # Use identify_zones method instead of get_current_zones
            support_zones, resistance_zones = sr_detector.identify_zones(data)
            if support_zones or resistance_zones:
                detector_analysis['support_resistance'] = {
                    'support_zones': support_zones,
                    'resistance_zones': resistance_zones
                }
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
        
        # Fibonacci
        try:
            fib_detector = self.detectors[symbol]['fibonacci']
            fib_detector.data = data
            fib_detector.current_price = current_price
            fib_detector.update_fibonacci_levels(data)
            if fib_detector.fibonacci_levels:
                detector_analysis['fibonacci'] = [
                    {
                        'level_type': level.level_type,
                        'percentage': level.percentage,
                        'price': level.price,
                        'strength': level.strength
                    } for level in fib_detector.fibonacci_levels
                ]
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
        
        # Divergence
        try:
            div_detector = self.detectors[symbol]['divergence']
            div_analysis = div_detector.analyze_divergence(data)
            if div_analysis and div_analysis.get('type'):
                detector_analysis['divergence'] = div_analysis
        except Exception as e:
            logger.error(f"Error in divergence analysis: {e}")
        
        # Print results
        print(f"\n🎯 STRATEGY SIGNALS:")
        if strategy_signals:
            for strategy, info in strategy_signals.items():
                signal_type = "BUY" if info['signal'] == 1 else "SELL"
                print(f"  • {strategy}: {signal_type} (Strength: {info['strength']:.3f}, Confidence: {info['confidence']:.3f})")
        else:
            print("  • No signals generated")
        
        print(f"\n🔍 DETECTOR ANALYSIS:")
        
        if 'support_resistance' in detector_analysis:
            zones = detector_analysis['support_resistance']
            print(f"  📍 Support/Resistance Zones:")
            if zones.get('support_zones'):
                for zone in zones['support_zones'][:3]:  # Show top 3
                    print(f"    • Support: ${zone.price_range[0]:.4f} - ${zone.price_range[1]:.4f} (Strength: {zone.strength:.3f})")
            if zones.get('resistance_zones'):
                for zone in zones['resistance_zones'][:3]:  # Show top 3
                    print(f"    • Resistance: ${zone.price_range[0]:.4f} - ${zone.price_range[1]:.4f} (Strength: {zone.strength:.3f})")
        
        if 'fibonacci' in detector_analysis:
            print(f"  📐 Fibonacci Levels:")
            for level in detector_analysis['fibonacci'][:5]:  # Show top 5
                print(f"    • {level['level_type'].title()} {level['percentage']}%: ${level['price']:.4f} (Strength: {level['strength']:.3f})")
        
        if 'divergence' in detector_analysis:
            div = detector_analysis['divergence']
            print(f"  📊 Divergence: {div['type'].upper()} (Strength: {div.get('strength', 0):.3f})")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'strategy_signals': strategy_signals,
            'detector_analysis': detector_analysis
        }
    
    def run_analysis(self):
        """Run analysis for all symbols"""
        print(f"🚀 MARKET ANALYSIS CHECKER")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Symbols: {len(self.symbols)}")
        print(f"⏱️ Timeframe: {self.timeframe}")
        
        results = {}
        
        for symbol in self.symbols:
            try:
                result = self.analyze_symbol(symbol)
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📋 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        total_signals = 0
        buy_signals = 0
        sell_signals = 0
        
        for symbol, result in results.items():
            signals = result['strategy_signals']
            if signals:
                total_signals += len(signals)
                for strategy, info in signals.items():
                    if info['signal'] == 1:
                        buy_signals += 1
                    elif info['signal'] == -1:
                        sell_signals += 1
        
        print(f"📊 Total Signals Generated: {total_signals}")
        print(f"🟢 Buy Signals: {buy_signals}")
        print(f"🔴 Sell Signals: {sell_signals}")
        
        # Save results
        with open('market_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: market_analysis_results.json")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr

if __name__ == "__main__":
    checker = MarketAnalysisChecker()
    checker.run_analysis() 