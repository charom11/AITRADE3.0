"""
Trading Strategies for Mini Hedge Fund
Implements momentum, mean reversion, pairs trading, divergence, support/resistance, and fibonacci strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from config import STRATEGY_CONFIG

# Import detection modules
try:
    from live_support_resistance_detector import LiveSupportResistanceDetector, SupportResistanceZone
    from live_fibonacci_detector import LiveFibonacciDetector, FibonacciLevel, SwingPoint
    from divergence_detector import DivergenceDetector
except ImportError as e:
    logging.warning(f"Some detection modules not available: {e}")
    LiveSupportResistanceDetector = None
    LiveFibonacciDetector = None
    DivergenceDetector = None
from live_support_resistance_detector import LiveSupportResistanceDetector
from live_fibonacci_detector import LiveFibonacciDetector

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
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
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
        """Generate enhanced momentum signals with multiple confirmations"""
        signals = data.copy()
        
        # Calculate required indicators if not present
        if 'sma_20' not in signals.columns:
            signals['sma_20'] = signals['close'].rolling(window=self.config['sma_short']).mean()
        if 'sma_50' not in signals.columns:
            signals['sma_50'] = signals['close'].rolling(window=self.config['sma_medium']).mean()
        if 'sma_200' not in signals.columns:
            signals['sma_200'] = signals['close'].rolling(window=self.config['sma_long']).mean()
        if 'rsi' not in signals.columns:
            signals['rsi'] = self._calculate_rsi(signals['close'], self.config['rsi_period'])
        if 'volume_sma' not in signals.columns:
            signals['volume_sma'] = signals['volume'].rolling(window=20).mean()
        if 'macd' not in signals.columns:
            signals['macd'], signals['macd_signal'] = self._calculate_macd(signals['close'])
        if 'atr' not in signals.columns:
            signals['atr'] = self._calculate_atr(signals, self.config['atr_period'])
        
        # Initialize signal column
        signals['signal'] = 0
        signals['signal_strength'] = 0.0
        
        # Calculate trend strength
        signals['trend_strength'] = self._calculate_trend_strength(signals)
        
        # Enhanced long signals with multiple confirmations
        long_condition = (
            (signals['close'] > signals['sma_200']) &  # Primary trend
            (signals['sma_20'] > signals['sma_50']) &  # Short-term momentum
            (signals['rsi'] < self.config['rsi_overbought']) &  # Not overbought
            (signals['volume'] > signals['volume_sma'] * self.config['volume_multiplier']) &  # Volume confirmation
            (signals['macd'] > signals['macd_signal']) &  # MACD confirmation
            (signals['trend_strength'] > self.config['min_trend_strength']) &  # Trend strength
            (signals['atr'] > signals['atr'].rolling(20).mean())  # Volatility filter
        )
        
        # Enhanced short signals with multiple confirmations
        short_condition = (
            (signals['close'] < signals['sma_200']) &  # Primary trend
            (signals['sma_20'] < signals['sma_50']) &  # Short-term momentum
            (signals['rsi'] > self.config['rsi_oversold']) &  # Not oversold
            (signals['volume'] > signals['volume_sma'] * self.config['volume_multiplier']) &  # Volume confirmation
            (signals['macd'] < signals['macd_signal']) &  # MACD confirmation
            (signals['trend_strength'] > self.config['min_trend_strength']) &  # Trend strength
            (signals['atr'] > signals['atr'].rolling(20).mean())  # Volatility filter
        )
        
        # Apply signals
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        # Calculate signal strength for long signals
        long_strength = (
            (signals['close'] / signals['sma_200'] - 1) * 2 +  # Price vs MA
            (signals['sma_20'] / signals['sma_50'] - 1) * 2 +  # Momentum
            (70 - signals['rsi']) / 70 +  # RSI distance from overbought
            (signals['volume'] / signals['volume_sma'] - 1) +  # Volume strength
            (signals['macd'] - signals['macd_signal']) / signals['close'] * 100 +  # MACD strength
            signals['trend_strength']  # Trend strength
        ) / 6
        
        # Calculate signal strength for short signals
        short_strength = (
            (signals['sma_200'] / signals['close'] - 1) * 2 +  # Price vs MA
            (signals['sma_50'] / signals['sma_20'] - 1) * 2 +  # Momentum
            (signals['rsi'] - 30) / 70 +  # RSI distance from oversold
            (signals['volume'] / signals['volume_sma'] - 1) +  # Volume strength
            (signals['macd_signal'] - signals['macd']) / signals['close'] * 100 +  # MACD strength
            signals['trend_strength']  # Trend strength
        ) / 6
        
        signals.loc[long_condition, 'signal_strength'] = long_strength[long_condition]
        signals.loc[short_condition, 'signal_strength'] = short_strength[short_condition]
        
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
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
        """Calculate position size using Kelly Criterion and dynamic risk management"""
        if signal == 0:
            return 0
        
        # Base position size from config
        base_position = self.config['max_position']
        
        # Kelly Criterion calculation
        if hasattr(self, 'win_rate') and hasattr(self, 'avg_win') and hasattr(self, 'avg_loss'):
            # Use historical performance for Kelly
            if self.avg_loss != 0:
                kelly_fraction = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) / self.avg_win
                kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
            else:
                kelly_fraction = 0.1  # Default 10%
        else:
            # Default Kelly fraction based on signal strength
            kelly_fraction = 0.1 + (signal_strength * 0.15)  # 10-25% range
        
        # Dynamic position sizing based on volatility
        if 'atr' in self.last_data.columns if hasattr(self, 'last_data') else False:
            current_atr = self.last_data['atr'].iloc[-1]
            avg_atr = self.last_data['atr'].rolling(20).mean().iloc[-1]
            volatility_factor = avg_atr / current_atr if current_atr > 0 else 1.0
            volatility_factor = np.clip(volatility_factor, 0.5, 2.0)
        else:
            volatility_factor = 1.0
        
        # Market condition adjustment
        market_condition_factor = self._calculate_market_condition_factor()
        
        # Final position size calculation
        position_size = (
            base_position * 
            kelly_fraction * 
            volatility_factor * 
            market_condition_factor * 
            abs(signal_strength)
        )
        
        # Risk management limits
        max_risk_position = portfolio_value * self.config.get('risk_per_trade', 0.01) / price
        position_size = min(position_size * portfolio_value, max_risk_position)
        
        return position_size if signal > 0 else -position_size
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=self.config['macd_fast']).mean()
        ema_slow = prices.ewm(span=self.config['macd_slow']).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.config['macd_signal']).mean()
        return macd, macd_signal
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength based on price action and moving averages"""
        # Price momentum
        price_momentum = data['close'].pct_change(periods=20)
        
        # MA alignment
        ma_alignment = (
            (data['sma_20'] > data['sma_50']).astype(int) +
            (data['sma_50'] > data['sma_200']).astype(int)
        ) / 2
        
        # Price vs MA distance
        price_vs_ma = np.abs(data['close'] / data['sma_200'] - 1)
        
        # Combine factors
        trend_strength = (
            price_momentum.rolling(10).mean() * 0.4 +
            ma_alignment * 0.4 +
            (1 - price_vs_ma) * 0.2
        )
        
        return trend_strength.clip(0, 1)
    
    def _calculate_market_condition_factor(self) -> float:
        """Calculate market condition factor for position sizing"""
        if not hasattr(self, 'last_data') or self.last_data is None:
            return 1.0
        
        try:
            # Trend strength
            if 'sma_200' in self.last_data.columns:
                trend_strength = (self.last_data['close'].iloc[-1] / self.last_data['sma_200'].iloc[-1] - 1)
                trend_factor = 1.0 + np.clip(trend_strength, -0.2, 0.2)
            else:
                trend_factor = 1.0
            
            # Volatility condition
            if 'atr' in self.last_data.columns:
                current_vol = self.last_data['atr'].iloc[-1]
                avg_vol = self.last_data['atr'].rolling(20).mean().iloc[-1]
                vol_factor = avg_vol / current_vol if current_vol > 0 else 1.0
                vol_factor = np.clip(vol_factor, 0.7, 1.3)
            else:
                vol_factor = 1.0
            
            # Volume condition
            if 'volume' in self.last_data.columns and 'volume_sma' in self.last_data.columns:
                volume_ratio = self.last_data['volume'].iloc[-1] / self.last_data['volume_sma'].iloc[-1]
                volume_factor = np.clip(volume_ratio, 0.8, 1.2)
            else:
                volume_factor = 1.0
            
            return trend_factor * vol_factor * volume_factor
            
        except Exception:
            return 1.0

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
        """Generate enhanced mean reversion signals with multiple confirmations"""
        signals = data.copy()
        
        # Calculate required indicators if not present
        if 'bb_upper' not in signals.columns or 'bb_lower' not in signals.columns:
            bb_upper, bb_lower = self._calculate_bollinger_bands(signals['close'], self.config['bb_period'], self.config['bb_std'])
            signals['bb_upper'] = bb_upper
            signals['bb_lower'] = bb_lower
        if 'rsi' not in signals.columns:
            signals['rsi'] = self._calculate_rsi(signals['close'], self.config['rsi_period'])
        if 'atr' not in signals.columns:
            signals['atr'] = self._calculate_atr(signals, self.config['atr_period'])
        if 'volume_sma' not in signals.columns:
            signals['volume_sma'] = signals['volume'].rolling(window=20).mean()
        
        # Initialize signal column
        signals['signal'] = 0
        signals['signal_strength'] = 0.0
        
        # Calculate Bollinger Bands position and deviation
        bb_position = (signals['close'] - signals['bb_lower']) / (signals['bb_upper'] - signals['bb_lower'])
        bb_deviation = np.abs(signals['close'] - signals['close'].rolling(self.config['bb_period']).mean()) / signals['close'].rolling(self.config['bb_period']).std()
        
        # Calculate volatility z-score
        volatility_zscore = (signals['atr'] - signals['atr'].rolling(window=20).mean()) / signals['atr'].rolling(window=20).std()
        
        # Enhanced long signals (oversold bounce)
        long_condition = (
            (signals['close'] <= signals['bb_lower']) &  # Price at or below lower band
            (signals['rsi'] < self.config['rsi_oversold']) &  # RSI oversold
            (bb_deviation > self.config['max_deviation']) &  # Significant deviation
            (volatility_zscore > 1.0) &  # High volatility
            (signals['volume'] > signals['volume_sma'] * 1.2) &  # Volume confirmation
            (signals['close'].pct_change() > self.config['min_bounce_pct'])  # Initial bounce
        )
        
        # Enhanced short signals (overbought reversal)
        short_condition = (
            (signals['close'] >= signals['bb_upper']) &  # Price at or above upper band
            (signals['rsi'] > self.config['rsi_overbought']) &  # RSI overbought
            (bb_deviation > self.config['max_deviation']) &  # Significant deviation
            (volatility_zscore > 1.0) &  # High volatility
            (signals['volume'] > signals['volume_sma'] * 1.2) &  # Volume confirmation
            (signals['close'].pct_change() < -self.config['min_bounce_pct'])  # Initial reversal
        )
        
        # Apply signals
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        # Calculate signal strength for long signals
        long_strength = (
            (signals['bb_lower'] / signals['close'] - 1) * 3 +  # Distance from lower band
            (30 - signals['rsi']) / 30 +  # RSI oversold strength
            bb_deviation / 3 +  # Deviation strength
            volatility_zscore / 2 +  # Volatility strength
            (signals['volume'] / signals['volume_sma'] - 1)  # Volume strength
        ) / 5
        
        # Calculate signal strength for short signals
        short_strength = (
            (signals['close'] / signals['bb_upper'] - 1) * 3 +  # Distance from upper band
            (signals['rsi'] - 70) / 30 +  # RSI overbought strength
            bb_deviation / 3 +  # Deviation strength
            volatility_zscore / 2 +  # Volatility strength
            (signals['volume'] / signals['volume_sma'] - 1)  # Volume strength
        ) / 5
        
        signals.loc[long_condition, 'signal_strength'] = long_strength[long_condition]
        signals.loc[short_condition, 'signal_strength'] = short_strength[short_condition]
        
        return signals
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
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
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
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
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
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
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
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


class SupportResistanceStrategy(BaseStrategy):
    """
    Support and Resistance Strategy based on key price levels
    - Long when price bounces off support with volume confirmation
    - Short when price rejects from resistance with volume confirmation
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                'min_touches': 2,
                'zone_buffer': 0.003,  # 0.3%
                'volume_threshold': 1.5,
                'swing_sensitivity': 0.02,  # 2%
                'risk_per_trade': 0.02,
                'enable_charts': False,
                'enable_alerts': True
            }
        super().__init__('SupportResistance', config)
        
        # Initialize detector if available
        if LiveSupportResistanceDetector:
            self.detector = LiveSupportResistanceDetector(
                symbol='BTC/USDT',
                exchange='binance',
                timeframe='5m',
                min_touches=config['min_touches'],
                zone_buffer=config['zone_buffer'],
                volume_threshold=config['volume_threshold'],
                swing_sensitivity=config['swing_sensitivity'],
                enable_charts=config['enable_charts'],
                enable_alerts=config['enable_alerts']
            )
        else:
            self.detector = None
            self.logger.warning("LiveSupportResistanceDetector not available")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['signal'] = 0
        signals['support_level'] = 0.0
        signals['resistance_level'] = 0.0
        signals['zone_strength'] = 0.0
        signals['volume_confirmed'] = False
        
        if self.detector is None:
            return signals
            
        try:
            # Set data in detector
            self.detector.data = data
            self.detector.current_price = data['close'].iloc[-1]
            
            # Identify zones
            support_zones, resistance_zones = self.detector.identify_zones(data)
            
            current_price = data['close'].iloc[-1]
            
            # Check for support bounce
            for zone in support_zones:
                if (current_price >= zone.price_range[0] and 
                    current_price <= zone.price_range[1] and 
                    zone.strength >= 0.6):
                    
                    # Check for bounce pattern (price was lower, now higher)
                    recent_low = data['low'].tail(5).min()
                    if current_price > recent_low * 1.005:  # 0.5% bounce
                        signals.iloc[-1, signals.columns.get_loc('signal')] = 1
                        signals.iloc[-1, signals.columns.get_loc('support_level')] = zone.level
                        signals.iloc[-1, signals.columns.get_loc('zone_strength')] = zone.strength
                        signals.iloc[-1, signals.columns.get_loc('volume_confirmed')] = zone.volume_confirmed
                        break
            
            # Check for resistance rejection
            for zone in resistance_zones:
                if (current_price >= zone.price_range[0] and 
                    current_price <= zone.price_range[1] and 
                    zone.strength >= 0.6):
                    
                    # Check for rejection pattern (price was higher, now lower)
                    recent_high = data['high'].tail(5).max()
                    if current_price < recent_high * 0.995:  # 0.5% rejection
                        signals.iloc[-1, signals.columns.get_loc('signal')] = -1
                        signals.iloc[-1, signals.columns.get_loc('resistance_level')] = zone.level
                        signals.iloc[-1, signals.columns.get_loc('zone_strength')] = zone.strength
                        signals.iloc[-1, signals.columns.get_loc('volume_confirmed')] = zone.volume_confirmed
                        break
                        
        except Exception as e:
            self.logger.error(f"Error in support/resistance signal generation: {e}")
            
        return signals

    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
        if signal == 0:
            return 0
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.02)
        position_size = base_size * abs(signal)
        max_position = portfolio_value * 0.10
        position_size = min(position_size, max_position)
        return position_size if signal > 0 else -position_size

    def get_current_zones(self) -> Dict:
        """Get current support and resistance zones"""
        if self.detector is None:
            return {'support_zones': [], 'resistance_zones': []}
        
        return {
            'support_zones': self.detector.support_zones,
            'resistance_zones': self.detector.resistance_zones
        }


class FibonacciStrategy(BaseStrategy):
    """
    Fibonacci Retracement and Extension Strategy
    - Long when price bounces off Fibonacci support levels
    - Short when price rejects from Fibonacci resistance levels
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                'buffer_percentage': 0.003,  # 0.3%
                'min_swing_strength': 0.6,
                'risk_per_trade': 0.02,
                'enable_charts': False,
                'enable_alerts': True
            }
        super().__init__('Fibonacci', config)
        
        # Initialize detector if available
        if LiveFibonacciDetector:
            self.detector = LiveFibonacciDetector(
                buffer_percentage=config['buffer_percentage'],
                min_swing_strength=config['min_swing_strength']
            )
        else:
            self.detector = None
            self.logger.warning("LiveFibonacciDetector not available")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['signal'] = 0
        signals['fibonacci_level'] = 0.0
        signals['level_type'] = ''
        signals['level_percentage'] = 0.0
        signals['level_strength'] = 0.0
        
        if self.detector is None:
            return signals
            
        try:
            # Set data in detector
            self.detector.data = data
            self.detector.current_price = data['close'].iloc[-1]
            
            # Update Fibonacci levels
            self.detector.update_fibonacci_levels(data)
            
            current_price = data['close'].iloc[-1]
            
            # Check for Fibonacci level interactions
            for level in self.detector.fibonacci_levels:
                if (current_price >= level.zone_low and 
                    current_price <= level.zone_high and 
                    level.strength >= 0.5):
                    
                    # Determine signal based on level type and price action
                    if level.level_type == 'retracement':
                        # Retracement levels act as support/resistance
                        if level.percentage <= 61.8:  # Strong support levels
                            # Check for bounce from support
                            recent_low = data['low'].tail(3).min()
                            if current_price > recent_low * 1.002:  # Small bounce
                                signals.iloc[-1, signals.columns.get_loc('signal')] = 1
                                signals.iloc[-1, signals.columns.get_loc('fibonacci_level')] = level.price
                                signals.iloc[-1, signals.columns.get_loc('level_type')] = level.level_type
                                signals.iloc[-1, signals.columns.get_loc('level_percentage')] = level.percentage
                                signals.iloc[-1, signals.columns.get_loc('level_strength')] = level.strength
                                break
                        else:  # 78.6% - potential reversal
                            # Check for rejection from deep retracement
                            recent_high = data['high'].tail(3).max()
                            if current_price < recent_high * 0.998:  # Small rejection
                                signals.iloc[-1, signals.columns.get_loc('signal')] = -1
                                signals.iloc[-1, signals.columns.get_loc('fibonacci_level')] = level.price
                                signals.iloc[-1, signals.columns.get_loc('level_type')] = level.level_type
                                signals.iloc[-1, signals.columns.get_loc('level_percentage')] = level.percentage
                                signals.iloc[-1, signals.columns.get_loc('level_strength')] = level.strength
                                break
                    
                    elif level.level_type == 'extension':
                        # Extension levels act as targets
                        if level.percentage >= 161.8:  # Strong extension
                            # Check for rejection from extension
                            recent_high = data['high'].tail(3).max()
                            if current_price < recent_high * 0.998:  # Small rejection
                                signals.iloc[-1, signals.columns.get_loc('signal')] = -1
                                signals.iloc[-1, signals.columns.get_loc('fibonacci_level')] = level.price
                                signals.iloc[-1, signals.columns.get_loc('level_type')] = level.level_type
                                signals.iloc[-1, signals.columns.get_loc('level_percentage')] = level.percentage
                                signals.iloc[-1, signals.columns.get_loc('level_strength')] = level.strength
                                break
                        
        except Exception as e:
            self.logger.error(f"Error in Fibonacci signal generation: {e}")
            
        return signals

    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float, signal_strength: float = 1.0) -> float:
        if signal == 0:
            return 0
        base_size = portfolio_value * self.config.get('risk_per_trade', 0.02)
        position_size = base_size * abs(signal)
        max_position = portfolio_value * 0.10
        position_size = min(position_size, max_position)
        return position_size if signal > 0 else -position_size

    def get_fibonacci_levels(self) -> List[FibonacciLevel]:
        """Get current Fibonacci levels"""
        if self.detector is None:
            return []
        return self.detector.fibonacci_levels


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
    
    def evaluate_trade_signal(self, live_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Unified signal evaluation combining all detection modules
        
        Args:
            live_data: Dictionary with keys:
                - 'price_data': OHLCV data for each symbol
                - 'support_resistance': Support/resistance zones
                - 'fibonacci': Fibonacci levels
                - 'divergence': Divergence analysis
                
        Returns:
            Dictionary with unified trading signals
        """
        unified_signals = {}
        
        try:
            # Get signals from all strategies
            all_strategy_signals = self.get_all_signals(live_data.get('price_data', {}))
            
            # Process each symbol
            for symbol in live_data.get('price_data', {}).keys():
                symbol_signals = {}
                
                # Collect signals from each strategy
                for strategy_name, strategy_signals in all_strategy_signals.items():
                    if symbol in strategy_signals:
                        symbol_signals[strategy_name] = strategy_signals[symbol]
                
                # Evaluate combined signal
                combined_signal = self._combine_strategy_signals(symbol_signals)
                unified_signals[symbol] = combined_signal
                
        except Exception as e:
            self.logger.error(f"Error in unified signal evaluation: {e}")
            
        return unified_signals
    
    def _combine_strategy_signals(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Enhanced signal combination with weighted voting and confidence thresholds"""
        if not strategy_signals:
            return pd.DataFrame()
        
        # Get the first strategy's data as base
        base_strategy = list(strategy_signals.keys())[0]
        combined_signals = strategy_signals[base_strategy].copy()
        
        # Initialize combined signal columns
        combined_signals['combined_signal'] = 0
        combined_signals['signal_confidence'] = 0.0
        combined_signals['signal_count'] = 0
        combined_signals['bullish_votes'] = 0
        combined_signals['bearish_votes'] = 0
        
        # Strategy weights based on historical performance (can be dynamic)
        strategy_weights = {
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'pairs_trading': 0.15,
            'divergence': 0.20,
            'support_resistance': 0.10,
            'fibonacci': 0.10
        }
        
        # Market regime detection
        market_regime = self._detect_market_regime(combined_signals)
        
        # Adjust weights based on market regime
        if market_regime == 'trending':
            strategy_weights['momentum'] *= 1.5
            strategy_weights['divergence'] *= 1.2
            strategy_weights['mean_reversion'] *= 0.7
        elif market_regime == 'ranging':
            strategy_weights['mean_reversion'] *= 1.5
            strategy_weights['support_resistance'] *= 1.3
            strategy_weights['fibonacci'] *= 1.2
            strategy_weights['momentum'] *= 0.7
        
        # Normalize weights
        total_weight = sum(strategy_weights.values())
        strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
        
        # Combine signals with weighted voting
        for strategy_name, signals_df in strategy_signals.items():
            if strategy_name not in strategy_weights:
                continue
                
            weight = strategy_weights[strategy_name]
            
            # Count votes
            bullish_mask = signals_df['signal'] > 0
            bearish_mask = signals_df['signal'] < 0
            
            combined_signals.loc[bullish_mask, 'bullish_votes'] += weight
            combined_signals.loc[bearish_mask, 'bearish_votes'] += weight
            
            # Add weighted signal strength
            if 'signal_strength' in signals_df.columns:
                combined_signals.loc[bullish_mask, 'signal_confidence'] += (
                    signals_df.loc[bullish_mask, 'signal_strength'] * weight
                )
                combined_signals.loc[bearish_mask, 'signal_confidence'] += (
                    -signals_df.loc[bearish_mask, 'signal_strength'] * weight
                )
        
        # Generate final combined signal
        min_vote_threshold = 0.3  # At least 30% of strategies must agree
        min_confidence_threshold = 0.4  # Minimum confidence level
        
        # Long signal conditions
        long_condition = (
            (combined_signals['bullish_votes'] > min_vote_threshold) &
            (combined_signals['bullish_votes'] > combined_signals['bearish_votes'] * 1.5) &
            (combined_signals['signal_confidence'] > min_confidence_threshold)
        )
        
        # Short signal conditions
        short_condition = (
            (combined_signals['bearish_votes'] > min_vote_threshold) &
            (combined_signals['bearish_votes'] > combined_signals['bullish_votes'] * 1.5) &
            (combined_signals['signal_confidence'] < -min_confidence_threshold)
        )
        
        # Apply combined signals
        combined_signals.loc[long_condition, 'combined_signal'] = 1
        combined_signals.loc[short_condition, 'combined_signal'] = -1
        
        # Calculate final confidence
        combined_signals['signal_confidence'] = np.clip(
            combined_signals['signal_confidence'], -1, 1
        )
        
        # Count total signals
        combined_signals['signal_count'] = (
            (combined_signals['bullish_votes'] > 0).astype(int) +
            (combined_signals['bearish_votes'] > 0).astype(int)
        )
        
        return combined_signals
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            # Calculate trend strength
            if 'sma_200' in data.columns and 'sma_20' in data.columns:
                trend_strength = np.abs(
                    (data['sma_20'].iloc[-1] / data['sma_200'].iloc[-1] - 1)
                )
            else:
                trend_strength = 0
            
            # Calculate volatility
            if 'atr' in data.columns:
                current_vol = data['atr'].iloc[-1]
                avg_vol = data['atr'].rolling(20).mean().iloc[-1]
                volatility_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            else:
                volatility_ratio = 1.0
            
            # Determine regime
            if trend_strength > 0.05 and volatility_ratio < 1.5:
                return 'trending'
            elif volatility_ratio > 1.8:
                return 'volatile'
            else:
                return 'ranging'
                
        except Exception:
            return 'ranging'  # Default to ranging
    
    def _get_signal_confidence(self, signals: pd.DataFrame, strategy_name: str) -> float:
        """Calculate confidence level for a strategy signal"""
        if len(signals) == 0:
            return 0.0
            
        # Base confidence from signal strength
        if 'signal_strength' in signals.columns:
            confidence = abs(signals['signal_strength'].iloc[-1])
        else:
            confidence = 0.5  # Default confidence
            
        # Strategy-specific confidence adjustments
        if strategy_name == 'Divergence':
            if 'divergence_strength' in signals.columns:
                confidence = max(confidence, signals['divergence_strength'].iloc[-1])
                
        elif strategy_name == 'SupportResistance':
            if 'zone_strength' in signals.columns:
                confidence = max(confidence, signals['zone_strength'].iloc[-1])
                
        elif strategy_name == 'Fibonacci':
            if 'level_strength' in signals.columns:
                confidence = max(confidence, signals['level_strength'].iloc[-1])
        
        return min(1.0, confidence)
    
    def _check_detection_confirmations(self, live_data: Dict, symbol: str) -> Dict:
        """Check for confirmations from detection modules"""
        confirmations = {
            'support_resistance': False,
            'fibonacci_support': False,
            'fibonacci_resistance': False,
            'divergence_bullish': False,
            'divergence_bearish': False
        }
        
        try:
            # Check support/resistance confirmations
            if 'support_resistance' in live_data:
                sr_data = live_data['support_resistance']
                if symbol in sr_data:
                    zones = sr_data[symbol]
                    current_price = live_data['price_data'][symbol]['close'].iloc[-1]
                    
                    # Check if price is at support
                    for zone in zones.get('support_zones', []):
                        if (current_price >= zone.price_range[0] and 
                            current_price <= zone.price_range[1]):
                            confirmations['support_resistance'] = True
                            break
                    
                    # Check if price is at resistance
                    for zone in zones.get('resistance_zones', []):
                        if (current_price >= zone.price_range[0] and 
                            current_price <= zone.price_range[1]):
                            confirmations['resistance_rejection'] = True
                            break
            
            # Check Fibonacci confirmations
            if 'fibonacci' in live_data:
                fib_data = live_data['fibonacci']
                if symbol in fib_data:
                    levels = fib_data[symbol]
                    current_price = live_data['price_data'][symbol]['close'].iloc[-1]
                    
                    for level in levels:
                        if (current_price >= level.zone_low and 
                            current_price <= level.zone_high):
                            if level.level_type == 'retracement' and level.percentage <= 61.8:
                                confirmations['fibonacci_support'] = True
                            elif level.level_type == 'retracement' and level.percentage >= 78.6:
                                confirmations['fibonacci_resistance'] = True
                            break
            
            # Check divergence confirmations
            if 'divergence' in live_data:
                div_data = live_data['divergence']
                if symbol in div_data:
                    divergence_info = div_data[symbol]
                    if divergence_info.get('type') == 'bullish':
                        confirmations['divergence_bullish'] = True
                    elif divergence_info.get('type') == 'bearish':
                        confirmations['divergence_bearish'] = True
                        
        except Exception as e:
            self.logger.error(f"Error checking detection confirmations: {e}")
            
        return confirmations
    
    def _enhance_signals_with_detections(self, combined_result: Dict, detection_confirmations: Dict) -> Dict:
        """
        Enhance signals based on detection module confirmations
        
        Args:
            combined_result: Combined signal result
            detection_confirmations: Confirmations from detection modules
            
        Returns:
            Enhanced signal result
        """
        enhanced_result = combined_result.copy()
        
        # Boost confidence for detection confirmations
        detection_bonus = 0.0
        
        if detection_confirmations.get('support_resistance', False):
            detection_bonus += 0.1
            enhanced_result['reasoning'].append("Support/Resistance zone confirmation")
            
        if detection_confirmations.get('fibonacci_support', False):
            detection_bonus += 0.1
            enhanced_result['reasoning'].append("Fibonacci support level confirmation")
            
        if detection_confirmations.get('fibonacci_resistance', False):
            detection_bonus += 0.1
            enhanced_result['reasoning'].append("Fibonacci resistance level confirmation")
            
        if detection_confirmations.get('divergence_bullish', False):
            detection_bonus += 0.15
            enhanced_result['reasoning'].append("Bullish divergence confirmation")
            
        if detection_confirmations.get('divergence_bearish', False):
            detection_bonus += 0.15
            enhanced_result['reasoning'].append("Bearish divergence confirmation")
        
        # Apply detection bonus to confidence
        enhanced_result['confidence'] = min(1.0, enhanced_result['confidence'] + detection_bonus)
        
        # Determine risk level based on confidence and confirmations
        if enhanced_result['confidence'] >= 0.8:
            enhanced_result['risk_level'] = 'low'
        elif enhanced_result['confidence'] >= 0.6:
            enhanced_result['risk_level'] = 'medium'
        else:
            enhanced_result['risk_level'] = 'high'
        
        # Determine recommended action
        if enhanced_result['signal'] == 1 and enhanced_result['confidence'] >= 0.6:
            enhanced_result['recommended_action'] = 'buy'
        elif enhanced_result['signal'] == -1 and enhanced_result['confidence'] >= 0.6:
            enhanced_result['recommended_action'] = 'sell'
        else:
            enhanced_result['recommended_action'] = 'hold'
        
        return enhanced_result 