#!/usr/bin/env python3
"""
Strategy Optimizer
Analyze and optimize trading strategies based on signal quality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from check_market_analysis import MarketAnalysisChecker
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    DivergenceStrategy,
    SupportResistanceStrategy,
    FibonacciStrategy
)
from config import STRATEGY_CONFIG

class StrategyOptimizer:
    def __init__(self):
        self.checker = MarketAnalysisChecker()
        self.signal_history = []
        self.optimization_results = {}
        
    def analyze_signal_quality(self, symbol: str, timeframe: str = '1h', days: int = 7):
        """Analyze signal quality for a specific symbol"""
        print(f"\n🔍 ANALYZING SIGNAL QUALITY: {symbol}")
        print("=" * 60)
        
        # Get historical data
        data = self.checker.fetch_market_data(symbol, limit=1000)
        if data.empty:
            print("❌ No data available")
            return {}
        
        # Initialize strategies
        strategies = {
            'Momentum': MomentumStrategy(STRATEGY_CONFIG['momentum']),
            'MeanReversion': MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion']),
            'Divergence': DivergenceStrategy(STRATEGY_CONFIG['divergence']),
            'SupportResistance': SupportResistanceStrategy(STRATEGY_CONFIG['support_resistance']),
            'Fibonacci': FibonacciStrategy(STRATEGY_CONFIG['fibonacci'])
        }
        
        results = {}
        
        for strategy_name, strategy in strategies.items():
            print(f"\n📊 Testing {strategy_name}...")
            
            try:
                # Generate signals
                signals = strategy.generate_signals(data)
                
                if not signals.empty:
                    # Analyze signal quality
                    quality_metrics = self._calculate_signal_quality(signals, data)
                    results[strategy_name] = quality_metrics
                    
                    print(f"  ✅ Signals Generated: {len(signals[signals['signal'] != 0])}")
                    print(f"  📈 Signal Quality Score: {quality_metrics['quality_score']:.3f}")
                    print(f"  🎯 Win Rate: {quality_metrics['win_rate']:.2f}%")
                    print(f"  📊 Average Strength: {quality_metrics['avg_strength']:.3f}")
                else:
                    print(f"  ⚠️ No signals generated")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return results
    
    def _calculate_signal_quality(self, signals: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """Calculate signal quality metrics"""
        # Filter non-zero signals
        active_signals = signals[signals['signal'] != 0].copy()
        
        if active_signals.empty:
            return {
                'quality_score': 0.0,
                'win_rate': 0.0,
                'avg_strength': 0.0,
                'signal_count': 0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate forward returns
        active_signals['future_return'] = 0.0
        active_signals['signal_outcome'] = 0  # 1 for win, -1 for loss
        
        for idx, signal in active_signals.iterrows():
            if idx + 5 < len(data):  # Look 5 periods ahead
                future_price = data['close'].iloc[idx + 5]
                current_price = data['close'].iloc[idx]
                
                if signal['signal'] == 1:  # BUY signal
                    return_pct = (future_price - current_price) / current_price
                    active_signals.loc[idx, 'future_return'] = return_pct
                    active_signals.loc[idx, 'signal_outcome'] = 1 if return_pct > 0 else -1
                elif signal['signal'] == -1:  # SELL signal
                    return_pct = (current_price - future_price) / current_price
                    active_signals.loc[idx, 'future_return'] = return_pct
                    active_signals.loc[idx, 'signal_outcome'] = 1 if return_pct > 0 else -1
        
        # Calculate metrics
        win_rate = (active_signals['signal_outcome'] == 1).mean() * 100
        avg_strength = active_signals.get('signal_strength', pd.Series([0.5] * len(active_signals))).mean()
        avg_return = active_signals['future_return'].mean()
        
        # Quality score (0-1)
        quality_score = (
            (win_rate / 100) * 0.4 +  # Win rate weight
            (avg_strength) * 0.3 +     # Signal strength weight
            (min(avg_return * 100, 1)) * 0.3  # Return weight
        )
        
        return {
            'quality_score': quality_score,
            'win_rate': win_rate,
            'avg_strength': avg_strength,
            'signal_count': len(active_signals),
            'avg_return': avg_return,
            'profit_factor': self._calculate_profit_factor(active_signals),
            'max_drawdown': self._calculate_max_drawdown(active_signals)
        }
    
    def _calculate_profit_factor(self, signals: pd.DataFrame) -> float:
        """Calculate profit factor"""
        if signals.empty:
            return 0.0
        
        gains = signals[signals['future_return'] > 0]['future_return'].sum()
        losses = abs(signals[signals['future_return'] < 0]['future_return'].sum())
        
        return gains / losses if losses > 0 else gains
    
    def _calculate_max_drawdown(self, signals: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if signals.empty:
            return 0.0
        
        cumulative_returns = signals['future_return'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(drawdown.min())
    
    def optimize_strategy_parameters(self, symbol: str, strategy_name: str):
        """Optimize strategy parameters using grid search"""
        print(f"\n⚙️ OPTIMIZING {strategy_name} FOR {symbol}")
        print("=" * 60)
        
        # Get current config
        current_config = STRATEGY_CONFIG.get(strategy_name.lower().replace('strategy', ''), {})
        
        # Define parameter ranges for optimization
        param_ranges = {
            'Momentum': {
                'rsi_period': [10, 14, 20],
                'rsi_overbought': [65, 70, 75],
                'rsi_oversold': [25, 30, 35],
                'volume_multiplier': [1.0, 1.5, 2.0]
            },
            'MeanReversion': {
                'bb_std': [1.5, 2.0, 2.5],
                'rsi_period': [10, 14, 20],
                'min_bounce_pct': [0.01, 0.02, 0.03]
            },
            'Divergence': {
                'rsi_period': [10, 14, 20],
                'min_divergence_strength': [0.5, 0.7, 0.9],
                'volume_confirmation': [True, False]
            },
            'SupportResistance': {
                'min_touches': [2, 3, 4],
                'zone_buffer': [0.005, 0.01, 0.015],
                'bounce_threshold': [0.01, 0.02, 0.03]
            },
            'Fibonacci': {
                'swing_window': [10, 20, 30],
                'min_swing_strength': [0.5, 0.7, 0.9],
                'tolerance': [0.01, 0.02, 0.03]
            }
        }
        
        ranges = param_ranges.get(strategy_name, {})
        if not ranges:
            print(f"❌ No optimization ranges defined for {strategy_name}")
            return current_config
        
        # Get test data
        data = self.checker.fetch_market_data(symbol, limit=500)
        if data.empty:
            print("❌ No data available for optimization")
            return current_config
        
        best_config = current_config.copy()
        best_score = 0.0
        
        # Grid search
        total_combinations = np.prod([len(v) for v in ranges.values()])
        print(f"🔍 Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(self._generate_param_combinations(ranges)):
            try:
                # Create test config
                test_config = current_config.copy()
                test_config.update(params)
                
                # Test strategy with new config
                strategy_class = self._get_strategy_class(strategy_name)
                if strategy_class:
                    strategy = strategy_class(test_config)
                    signals = strategy.generate_signals(data)
                    
                    if not signals.empty:
                        quality = self._calculate_signal_quality(signals, data)
                        score = quality['quality_score']
                        
                        if score > best_score:
                            best_score = score
                            best_config = test_config.copy()
                            
                        if (i + 1) % 10 == 0:
                            print(f"  Progress: {i+1}/{total_combinations} - Best Score: {best_score:.3f}")
                            
            except Exception as e:
                continue
        
        print(f"\n✅ Optimization Complete!")
        print(f"📈 Best Quality Score: {best_score:.3f}")
        print(f"⚙️ Optimized Parameters:")
        for param, value in best_config.items():
            if param in ranges:
                print(f"  • {param}: {value}")
        
        return best_config
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _get_strategy_class(self, strategy_name: str):
        """Get strategy class by name"""
        strategy_map = {
            'Momentum': MomentumStrategy,
            'MeanReversion': MeanReversionStrategy,
            'Divergence': DivergenceStrategy,
            'SupportResistance': SupportResistanceStrategy,
            'Fibonacci': FibonacciStrategy
        }
        return strategy_map.get(strategy_name)
    
    def generate_optimization_report(self, symbols: List[str] = None):
        """Generate comprehensive optimization report"""
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        
        print(f"\n📊 GENERATING OPTIMIZATION REPORT")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': symbols,
            'strategy_performance': {},
            'recommendations': []
        }
        
        # Analyze each symbol
        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}...")
            symbol_results = self.analyze_signal_quality(symbol)
            report['strategy_performance'][symbol] = symbol_results
        
        # Generate recommendations
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        # Save report
        with open('strategy_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Report saved: strategy_optimization_report.json")
        
        # Print summary
        self._print_optimization_summary(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate strategy recommendations"""
        recommendations = []
        
        # Analyze overall performance
        all_scores = []
        for symbol, strategies in report['strategy_performance'].items():
            for strategy, metrics in strategies.items():
                all_scores.append((strategy, metrics['quality_score']))
        
        if all_scores:
            # Find best performing strategies
            all_scores.sort(key=lambda x: x[1], reverse=True)
            best_strategies = [s[0] for s in all_scores[:3]]
            
            recommendations.append(f"Top performing strategies: {', '.join(best_strategies)}")
            
            # Find strategies needing optimization
            poor_strategies = [s[0] for s in all_scores if s[1] < 0.3]
            if poor_strategies:
                recommendations.append(f"Strategies needing optimization: {', '.join(poor_strategies)}")
        
        return recommendations
    
    def _print_optimization_summary(self, report: Dict):
        """Print optimization summary"""
        print(f"\n📋 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        for symbol, strategies in report['strategy_performance'].items():
            print(f"\n📊 {symbol}:")
            for strategy, metrics in strategies.items():
                print(f"  • {strategy}: Score {metrics['quality_score']:.3f}, Win Rate {metrics['win_rate']:.1f}%")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")

if __name__ == "__main__":
    optimizer = StrategyOptimizer()
    
    # Generate optimization report
    optimizer.generate_optimization_report()
    
    # Optimize specific strategies
    print("\n" + "="*60)
    optimizer.optimize_strategy_parameters('BTC/USDT', 'Momentum')
    optimizer.optimize_strategy_parameters('ETH/USDT', 'Divergence') 