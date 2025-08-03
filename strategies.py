"""
Trading Strategies for Mini Hedge Fund
Implements momentum, mean reversion, pairs trading, and divergence strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from abc import ABC, abstractmethod

from config import STRATEGY_CONFIG

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.positions = {}
        self.signals = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for the strategy"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """Calculate position size based on signal and risk management"""
        pass
    
    def get_strategy_summary(self) -> Dict:
        """Get strategy performance summary"""
        return {
            'name': self.name,
            'positions': len(self.positions),
            'total_signals': len(self.signals)
        }

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy based on moving averages and RSI
    - Long when price > long-term MA and RSI not overbought
    - Short when price < long-term MA and RSI not oversold
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = STRATEGY_CONFIG['momentum']
        super().__init__('Momentum', config)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals"""
        signals = data.copy()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Long signals: price above long MA and RSI not overbought
        long_condition = (
            (signals['close'] > signals['sma_200']) & 
            (signals['rsi'] < self.config['rsi_overbought']) &
            (signals['sma_20'] > signals['sma_50'])  # Short-term momentum
        )
        signals.loc[long_condition, 'signal'] = 1
        
        # Short signals: price below long MA and RSI not oversold
        short_condition = (
            (signals['close'] < signals['sma_200']) & 
            (signals['rsi'] > self.config['rsi_oversold']) &
            (signals['sma_20'] < signals['sma_50'])  # Short-term momentum
        )
        signals.loc[short_condition, 'signal'] = -1
        
        # Add signal strength
        signals['signal_strength'] = self._calculate_signal_strength(signals)
        
        return signals
    
    def _calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on multiple factors"""
        strength = pd.Series(0.0, index=data.index)
        
        # Price momentum
        price_momentum = (data['close'] - data['sma_200']) / data['sma_200']
        
        # RSI position
        rsi_position = (data['rsi'] - 50) / 50  # Normalize RSI to [-1, 1]
        
        # Volume confirmation
        volume_confirmation = (data['volume'] - data['volume_sma']) / data['volume_sma']
        volume_confirmation = np.clip(volume_confirmation, -1, 1)
        
        # Combine factors
        strength = (price_momentum * 0.4 + rsi_position * 0.3 + volume_confirmation * 0.3)
        
        return np.clip(strength, -1, 1)
    
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """Calculate position size for momentum strategy"""
        if signal == 0:
            return 0
        
        # Base position size
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.02)  # 2% risk
        
        # Adjust for signal strength
        position_size = base_size * abs(signal)
        
        # Apply maximum position size limit
        max_position = portfolio_value * 0.1  # 10% max position
        position_size = min(position_size, max_position)
        
        return position_size if signal > 0 else -position_size

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy based on Bollinger Bands and ATR
    - Long when price touches lower band and RSI oversold
    - Short when price touches upper band and RSI overbought
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = STRATEGY_CONFIG['mean_reversion']
        super().__init__('MeanReversion', config)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        signals = data.copy()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Calculate Bollinger Bands position
        bb_position = (signals['close'] - signals['bb_lower']) / (signals['bb_upper'] - signals['bb_lower'])
        
        # Calculate volatility z-score
        volatility_zscore = (signals['atr'] - signals['atr'].rolling(window=20).mean()) / signals['atr'].rolling(window=20).std()
        
        # Long signals: price near lower band, RSI oversold, low volatility
        long_condition = (
            (bb_position < 0.2) &  # Price near lower band
            (signals['rsi'] < self.config['rsi_oversold']) &  # RSI oversold
            (volatility_zscore < 1.0)  # Low volatility
        )
        signals.loc[long_condition, 'signal'] = 1
        
        # Short signals: price near upper band, RSI overbought, high volatility
        short_condition = (
            (bb_position > 0.8) &  # Price near upper band
            (signals['rsi'] > self.config['rsi_overbought']) &  # RSI overbought
            (volatility_zscore > 1.0)  # High volatility
        )
        signals.loc[short_condition, 'signal'] = -1
        
        # Add signal strength
        signals['signal_strength'] = self._calculate_signal_strength(signals, bb_position, volatility_zscore)
        
        return signals
    
    def _calculate_signal_strength(self, data: pd.DataFrame, bb_position: pd.Series, 
                                 volatility_zscore: pd.Series) -> pd.Series:
        """Calculate signal strength for mean reversion"""
        strength = pd.Series(0.0, index=data.index)
        
        # Bollinger Band position strength
        bb_strength = 1 - bb_position  # Stronger signal when closer to bands
        
        # RSI extremity strength
        rsi_strength = np.abs(data['rsi'] - 50) / 50  # How far from neutral
        
        # Volatility strength
        vol_strength = np.clip(np.abs(volatility_zscore), 0, 2) / 2
        
        # Combine factors
        strength = (bb_strength * 0.4 + rsi_strength * 0.4 + vol_strength * 0.2)
        
        return np.clip(strength, 0, 1)
    
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """Calculate position size for mean reversion strategy"""
        if signal == 0:
            return 0
        
        # Smaller position sizes for mean reversion (higher risk)
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.015)  # 1.5% risk
        
        # Adjust for signal strength
        position_size = base_size * abs(signal)
        
        # Apply maximum position size limit
        max_position = portfolio_value * 0.08  # 8% max position
        position_size = min(position_size, max_position)
        
        return position_size if signal > 0 else -position_size

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy based on statistical arbitrage
    - Find highly correlated pairs
    - Trade when spread deviates from mean
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = STRATEGY_CONFIG['pairs_trading']
        super().__init__('PairsTrading', config)
        self.pairs = []
        self.spreads = {}
        
    def find_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find highly correlated pairs for trading"""
        symbols = list(data.keys())
        pairs = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Align data
                df1, df2 = data[symbol1], data[symbol2]
                common_dates = df1.index.intersection(df2.index)
                
                if len(common_dates) < self.config['lookback_period']:
                    continue
                
                # Calculate correlation
                returns1 = df1.loc[common_dates, 'close'].pct_change().dropna()
                returns2 = df2.loc[common_dates, 'close'].pct_change().dropna()
                
                if len(returns1) < 50:  # Need sufficient data
                    continue
                
                correlation = returns1.corr(returns2)
                
                if abs(correlation) > self.config['correlation_threshold']:
                    pairs.append((symbol1, symbol2))
        
        return pairs
    
    def calculate_spread(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.Series:
        """Calculate spread between two assets"""
        # Use log prices for better stationarity
        log_price1 = np.log(data1['close'])
        log_price2 = np.log(data2['close'])
        spread = log_price1 - log_price2
        return spread
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate pairs trading signals"""
        signals = {}
        
        # Find trading pairs
        self.pairs = self.find_pairs(data)
        
        for symbol1, symbol2 in self.pairs:
            if symbol1 in data and symbol2 in data:
                # Align data
                df1, df2 = data[symbol1], data[symbol2]
                common_dates = df1.index.intersection(df2.index)
                
                if len(common_dates) < self.config['lookback_period']:
                    continue
                
                df1_aligned = df1.loc[common_dates]
                df2_aligned = df2.loc[common_dates]
                
                # Calculate spread
                spread = self.calculate_spread(df1_aligned, df2_aligned)
                self.spreads[f"{symbol1}_{symbol2}"] = spread
                
                # Calculate spread statistics
                spread_mean = spread.rolling(window=self.config['lookback_period']).mean()
                spread_std = spread.rolling(window=self.config['lookback_period']).std()
                z_score = (spread - spread_mean) / spread_std
                
                # Generate signals
                signal_df = df1_aligned.copy()
                signal_df['signal'] = 0
                signal_df['z_score'] = z_score
                signal_df['spread'] = spread
                
                # Long signal1, short signal2 when spread is low
                long_condition = z_score < -self.config['z_score_threshold']
                signal_df.loc[long_condition, 'signal'] = 1
                
                # Short signal1, long signal2 when spread is high
                short_condition = z_score > self.config['z_score_threshold']
                signal_df.loc[short_condition, 'signal'] = -1
                
                signals[f"{symbol1}_{symbol2}"] = signal_df
        
        return signals
    
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """Calculate position size for pairs trading"""
        if signal == 0:
            return 0
        
        # Equal position sizes for both legs of the pair
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.015)  # 1.5% risk
        
        # Adjust for signal strength
        position_size = base_size * abs(signal)
        
        # Apply maximum position size limit
        max_position = portfolio_value * 0.08  # 8% max position
        position_size = min(position_size, max_position)
        
        return position_size if signal > 0 else -position_size

class DivergenceStrategy(BaseStrategy):
    """
    Divergence Strategy based on Class A divergence detection
    - Bullish divergence: Price lower lows, indicator higher lows
    - Bearish divergence: Price higher highs, indicator lower highs
    - Uses RSI and MACD indicators
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'min_candles': 50,
                'swing_threshold': 0.02,
                'confirm_with_support_resistance': True,
                'confirm_with_candlestick': True
            }
        super().__init__('Divergence', config)
        
        # Import divergence detector
        from divergence_detector import DivergenceDetector
        self.detector = DivergenceDetector(
            rsi_period=config['rsi_period'],
            macd_fast=config['macd_fast'],
            macd_slow=config['macd_slow'],
            macd_signal=config['macd_signal'],
            min_candles=config['min_candles'],
            swing_threshold=config['swing_threshold']
        )
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate divergence signals"""
        signals = data.copy()
        
        # Initialize signal columns
        signals['signal'] = 0
        signals['divergence_type'] = ''
        signals['divergence_indicator'] = ''
        signals['divergence_strength'] = 0.0
        signals['support_resistance_confirmed'] = False
        signals['candlestick_confirmed'] = False
        signals['timeframe'] = ''
        
        # Analyze divergence
        analysis = self.detector.analyze_divergence(
            data, 
            confirm_signals=self.config['confirm_with_support_resistance']
        )
        
        if 'error' in analysis:
            self.logger.warning(f"Divergence analysis error: {analysis['error']}")
            return signals
        
        # Apply signals to the most recent data point
        if analysis['signals']:
            latest_signal = analysis['signals'][0]  # Get strongest signal
            
            # Find the index of the signal date
            signal_date = latest_signal['signal_date']
            if signal_date in signals.index:
                idx = signals.index.get_loc(signal_date)
                
                # Set signal
                if latest_signal['type'] == 'bullish':
                    signals.iloc[idx, signals.columns.get_loc('signal')] = 1
                else:
                    signals.iloc[idx, signals.columns.get_loc('signal')] = -1
                
                # Set additional information
                signals.iloc[idx, signals.columns.get_loc('divergence_type')] = latest_signal['type']
                signals.iloc[idx, signals.columns.get_loc('divergence_indicator')] = latest_signal['indicator']
                signals.iloc[idx, signals.columns.get_loc('divergence_strength')] = latest_signal['strength']
                
                if 'support_confirmation' in latest_signal:
                    signals.iloc[idx, signals.columns.get_loc('support_resistance_confirmed')] = latest_signal.get('support_confirmation', latest_signal.get('resistance_confirmation', False))
                
                if 'pattern_confirmation' in latest_signal:
                    signals.iloc[idx, signals.columns.get_loc('candlestick_confirmed')] = latest_signal.get('pattern_confirmation', False)
                
                # Set timeframe information
                if 'timeframe' in analysis:
                    signals.iloc[idx, signals.columns.get_loc('timeframe')] = analysis['timeframe']
        
        return signals
    
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """Calculate position size for divergence strategy"""
        if signal == 0:
            return 0
        
        # Divergence signals are high-confidence, so use larger position sizes
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.025)  # 2.5% risk
        
        # Adjust for signal strength (divergence strength)
        position_size = base_size * abs(signal)
        
        # Apply maximum position size limit
        max_position = portfolio_value * 0.12  # 12% max position for divergence
        position_size = min(position_size, max_position)
        
        return position_size if signal > 0 else -position_size
    
    def get_divergence_analysis(self, data: pd.DataFrame) -> Dict:
        """Get detailed divergence analysis"""
        return self.detector.analyze_divergence(
            data, 
            confirm_signals=self.config['confirm_with_support_resistance']
        )

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.logger = logging.getLogger(__name__)
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Added strategy: {strategy.name}")
    
    def get_all_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get signals from all strategies"""
        all_signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                if isinstance(strategy, PairsTradingStrategy):
                    signals = strategy.generate_signals(data)
                else:
                    signals = {}
                    for symbol, df in data.items():
                        signals[symbol] = strategy.generate_signals(df)
                
                all_signals[name] = signals
                self.logger.info(f"Generated signals for strategy: {name}")
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {name}: {str(e)}")
                continue
        
        return all_signals
    
    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary of all strategies"""
        summaries = []
        for name, strategy in self.strategies.items():
            summary = strategy.get_strategy_summary()
            summaries.append(summary)
        
        return pd.DataFrame(summaries) 