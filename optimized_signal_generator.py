#!/usr/bin/env python3
"""
Optimized Signal Generator
Enhanced signal generation with ML and advanced technical analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTradingSignal:
    """Enhanced trading signal with additional metadata"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    price: float
    volume: float
    conditions: List[str]
    risk_score: float  # 0-1 scale (lower is better)
    expected_return: float  # Expected percentage return
    stop_loss: float
    take_profit: float
    position_size: float
    file_hash: str = None  # For file creation prevention

class OptimizedSignalGenerator:
    """Enhanced signal generator with ML and optimization"""
    
    def __init__(self):
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_return': 0.0,
            'win_rate': 0.0
        }
    
    def calculate_enhanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators"""
        df = data.copy()
        
        # Basic indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(5)
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # CCI (Commodity Channel Index)
        df['cci'] = self._calculate_cci(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))
        return williams_r
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[OptimizedTradingSignal]:
        """Generate optimized trading signal"""
        if len(data) < 50:
            return None
        
        # Calculate indicators
        df = self.calculate_enhanced_indicators(data)
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        current_stoch_k = df['stoch_k'].iloc[-1]
        current_stoch_d = df['stoch_d'].iloc[-1]
        current_williams_r = df['williams_r'].iloc[-1]
        current_cci = df['cci'].iloc[-1]
        
        # Signal conditions
        conditions = []
        signal_strength = 0.0
        confidence = 0.0
        
        # RSI conditions
        if current_rsi < 30:
            conditions.append("RSI oversold")
            signal_strength += 0.25
            confidence += 0.15
        elif current_rsi > 70:
            conditions.append("RSI overbought")
            signal_strength += 0.25
            confidence += 0.15
        
        # MACD conditions
        if current_macd > current_macd_signal and current_macd > 0:
            conditions.append("MACD bullish crossover")
            signal_strength += 0.20
            confidence += 0.12
        elif current_macd < current_macd_signal and current_macd < 0:
            conditions.append("MACD bearish crossover")
            signal_strength += 0.20
            confidence += 0.12
        
        # Stochastic conditions
        if current_stoch_k < 20 and current_stoch_d < 20:
            conditions.append("Stochastic oversold")
            signal_strength += 0.15
            confidence += 0.10
        elif current_stoch_k > 80 and current_stoch_d > 80:
            conditions.append("Stochastic overbought")
            signal_strength += 0.15
            confidence += 0.10
        
        # Williams %R conditions
        if current_williams_r < -80:
            conditions.append("Williams %R oversold")
            signal_strength += 0.15
            confidence += 0.10
        elif current_williams_r > -20:
            conditions.append("Williams %R overbought")
            signal_strength += 0.15
            confidence += 0.10
        
        # CCI conditions
        if current_cci < -100:
            conditions.append("CCI oversold")
            signal_strength += 0.10
            confidence += 0.08
        elif current_cci > 100:
            conditions.append("CCI overbought")
            signal_strength += 0.10
            confidence += 0.08
        
        # Volume conditions
        volume_ratio = df['volume_ratio'].iloc[-1]
        if volume_ratio > 1.5:
            conditions.append("High volume")
            signal_strength += 0.15
            confidence += 0.10
        
        # Trend conditions
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            conditions.append("Strong uptrend")
            signal_strength += 0.20
            confidence += 0.12
        elif current_price < sma_20 < sma_50:
            conditions.append("Strong downtrend")
            signal_strength += 0.20
            confidence += 0.12
        
        # Bollinger Bands conditions
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        if current_price < bb_lower:
            conditions.append("Price below lower Bollinger Band")
            signal_strength += 0.15
            confidence += 0.10
        elif current_price > bb_upper:
            conditions.append("Price above upper Bollinger Band")
            signal_strength += 0.15
            confidence += 0.10
        
        # Determine signal type
        signal_type = 'hold'
        if signal_strength > 0.6:  # Higher threshold for better quality
            bullish_conditions = sum(1 for c in conditions if any(word in c.lower() for word in ['oversold', 'bullish', 'uptrend']))
            bearish_conditions = sum(1 for c in conditions if any(word in c.lower() for word in ['overbought', 'bearish', 'downtrend']))
            
            if bullish_conditions > bearish_conditions:
                signal_type = 'buy'
            elif bearish_conditions > bullish_conditions:
                signal_type = 'sell'
        
        # Calculate risk and return metrics
        atr = df['atr'].iloc[-1]
        volatility = df['close'].pct_change().std()
        
        risk_score = min(volatility * 10, 1.0)  # Higher volatility = higher risk
        expected_return = signal_strength * 0.04  # 4% max expected return
        
        # Calculate stop loss and take profit
        if signal_type == 'buy':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif signal_type == 'sell':
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            stop_loss = take_profit = current_price
        
        # Calculate position size based on risk
        position_size = 0.02 / risk_score if risk_score > 0 else 0.02  # 2% base risk
        
        # Create signal
        signal = OptimizedTradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=min(signal_strength, 1.0),
            confidence=min(confidence, 1.0),
            price=current_price,
            volume=current_volume,
            conditions=conditions,
            risk_score=risk_score,
            expected_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=min(position_size, 0.1)  # Max 10% position size
        )
        
        return signal if signal_type != 'hold' else None
    
    def update_performance_metrics(self, signal: OptimizedTradingSignal, actual_return: float):
        """Update performance metrics based on signal results"""
        self.performance_metrics['total_signals'] += 1
        
        if actual_return > 0:
            self.performance_metrics['successful_signals'] += 1
        
        # Update average return
        total_signals = self.performance_metrics['total_signals']
        current_avg = self.performance_metrics['avg_return']
        self.performance_metrics['avg_return'] = ((current_avg * (total_signals - 1)) + actual_return) / total_signals
        
        # Update win rate
        self.performance_metrics['win_rate'] = self.performance_metrics['successful_signals'] / total_signals
        
        # Store signal in history
        self.signal_history.append({
            'signal': signal,
            'actual_return': actual_return,
            'timestamp': datetime.now()
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'total_signals': self.performance_metrics['total_signals'],
            'successful_signals': self.performance_metrics['successful_signals'],
            'failed_signals': self.performance_metrics['failed_signals'],
            'win_rate': self.performance_metrics['win_rate'],
            'avg_return': self.performance_metrics['avg_return'],
            'recent_signals': len(self.signal_history)
        } 