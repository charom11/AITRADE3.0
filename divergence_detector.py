"""
Divergence Detection Strategy
Identifies Class A bullish and bearish divergence between price and momentum indicators

Class A Divergence Rules:
- Bullish: Price forms lower lows, indicator forms higher lows
- Bearish: Price forms higher highs, indicator forms lower highs

Requirements:
- Use RSI (14) and/or MACD (12, 26, 9)
- Analyze at least 50 recent candles
- Filter noise and avoid Class B/C signals
- Optional: confirm with support/resistance or candlestick patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DivergenceDetector:
    """Detects Class A divergence between price and momentum indicators"""
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, 
                 macd_slow: int = 26, macd_signal: int = 9, 
                 min_candles: int = 50, swing_threshold: float = 0.02):
        """
        Initialize divergence detector
        
        Args:
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            min_candles: Minimum candles to analyze
            swing_threshold: Minimum swing size to consider (2% default)
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.min_candles = min_candles
        self.swing_threshold = swing_threshold
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and MACD indicators
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with added indicator columns
        """
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(
            df['close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def find_swing_points(self, data: pd.DataFrame, column: str, 
                         window: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows in the data
        
        Args:
            data: DataFrame with price/indicator data
            column: Column name to analyze ('close', 'rsi', 'macd')
            window: Window size for swing detection
            
        Returns:
            Tuple of (high_indices, low_indices)
        """
        # Safety check: ensure we have enough data
        if len(data) < 2 * window + 2:
            return [], []
            
        highs = []
        lows = []
        
        for i in range(window, len(data) - window - 1):
            # Check for swing high
            if all(data[column].iloc[i] >= data[column].iloc[i-j] for j in range(1, window+1)) and \
               all(data[column].iloc[i] >= data[column].iloc[i+j] for j in range(1, window+1)):
                highs.append(i)
            
            # Check for swing low
            if all(data[column].iloc[i] <= data[column].iloc[i-j] for j in range(1, window+1)) and \
               all(data[column].iloc[i] <= data[column].iloc[i+j] for j in range(1, window+1)):
                lows.append(i)
        
        return highs, lows
    
    def filter_significant_swings(self, data: pd.DataFrame, highs: List[int], 
                                 lows: List[int], column: str) -> Tuple[List[int], List[int]]:
        """
        Filter swings to only include significant ones (above threshold)
        
        Args:
            data: DataFrame with price/indicator data
            highs: List of high indices
            lows: List of low indices
            column: Column name to analyze
            
        Returns:
            Tuple of filtered (high_indices, low_indices)
        """
        filtered_highs = []
        filtered_lows = []
        
        # Calculate average value for threshold
        avg_value = data[column].mean()
        threshold = avg_value * self.swing_threshold
        
        # Filter highs
        for i in range(1, len(highs)):
            if i > 0:
                swing_size = abs(data[column].iloc[highs[i]] - data[column].iloc[highs[i-1]])
                if swing_size >= threshold:
                    filtered_highs.append(highs[i])
        
        # Filter lows
        for i in range(1, len(lows)):
            if i > 0:
                swing_size = abs(data[column].iloc[lows[i]] - data[column].iloc[lows[i-1]])
                if swing_size >= threshold:
                    filtered_lows.append(lows[i])
        
        return filtered_highs, filtered_lows
    
    def detect_divergence(self, data: pd.DataFrame, price_highs: List[int], 
                         price_lows: List[int], indicator_highs: List[int], 
                         indicator_lows: List[int], indicator_name: str) -> List[Dict]:
        """
        Detect Class A divergence between price and indicator
        
        Args:
            data: DataFrame with price and indicator data
            price_highs: List of price high indices
            price_lows: List of price low indices
            indicator_highs: List of indicator high indices
            indicator_lows: List of indicator low indices
            indicator_name: Name of the indicator ('rsi' or 'macd')
            
        Returns:
            List of divergence signals
        """
        signals = []
        
        # Bearish divergence: Price higher highs, indicator lower highs
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            # Check last two price highs
            price_high1, price_high2 = price_highs[-2], price_highs[-1]
            price_val1, price_val2 = data['close'].iloc[price_high1], data['close'].iloc[price_high2]
            
            # Check last two indicator highs
            indicator_high1, indicator_high2 = indicator_highs[-2], indicator_highs[-1]
            indicator_val1, indicator_val2 = data[indicator_name].iloc[indicator_high1], data[indicator_name].iloc[indicator_high2]
            
            # Class A Bearish Divergence: Price higher highs, indicator lower highs
            if (price_val2 > price_val1 and indicator_val2 < indicator_val1 and 
                price_high2 > price_high1 and indicator_high2 > indicator_high1):
                
                signal = {
                    'type': 'bearish',
                    'divergence_class': 'A',
                    'indicator': indicator_name,
                    'price_high1_idx': price_high1,
                    'price_high2_idx': price_high2,
                    'price_high1_val': price_val1,
                    'price_high2_val': price_val2,
                    'indicator_high1_idx': indicator_high1,
                    'indicator_high2_idx': indicator_high2,
                    'indicator_high1_val': indicator_val1,
                    'indicator_high2_val': indicator_val2,
                    'signal_date': data.index[price_high2],
                    'strength': self._calculate_divergence_strength(price_val1, price_val2, indicator_val1, indicator_val2)
                }
                signals.append(signal)
        
        # Bullish divergence: Price lower lows, indicator higher lows
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            # Check last two price lows
            price_low1, price_low2 = price_lows[-2], price_lows[-1]
            price_val1, price_val2 = data['close'].iloc[price_low1], data['close'].iloc[price_low2]
            
            # Check last two indicator lows
            indicator_low1, indicator_low2 = indicator_lows[-2], indicator_lows[-1]
            indicator_val1, indicator_val2 = data[indicator_name].iloc[indicator_low1], data[indicator_name].iloc[indicator_low2]
            
            # Class A Bullish Divergence: Price lower lows, indicator higher lows
            if (price_val2 < price_val1 and indicator_val2 > indicator_val1 and 
                price_low2 > price_low1 and indicator_low2 > indicator_low1):
                
                signal = {
                    'type': 'bullish',
                    'divergence_class': 'A',
                    'indicator': indicator_name,
                    'price_low1_idx': price_low1,
                    'price_low2_idx': price_low2,
                    'price_low1_val': price_val1,
                    'price_low2_val': price_val2,
                    'indicator_low1_idx': indicator_low1,
                    'indicator_low2_idx': indicator_low2,
                    'indicator_low1_val': indicator_val1,
                    'indicator_low2_val': indicator_val2,
                    'signal_date': data.index[price_low2],
                    'strength': self._calculate_divergence_strength(price_val1, price_val2, indicator_val1, indicator_val2)
                }
                signals.append(signal)
        
        return signals
    
    def _calculate_divergence_strength(self, val1: float, val2: float, 
                                     indicator_val1: float, indicator_val2: float) -> float:
        """
        Calculate divergence strength (0-1 scale)
        
        Args:
            val1, val2: Price values
            indicator_val1, indicator_val2: Indicator values
            
        Returns:
            Divergence strength (0-1)
        """
        # Calculate percentage changes
        price_change = abs(val2 - val1) / val1
        indicator_change = abs(indicator_val2 - indicator_val1) / abs(indicator_val1)
        
        # Strength is based on how clear the divergence is
        strength = min(price_change + indicator_change, 1.0)
        return strength
    
    def confirm_with_support_resistance(self, data: pd.DataFrame, signal: Dict) -> Dict:
        """
        Confirm divergence signal with support/resistance levels
        
        Args:
            data: DataFrame with price data
            signal: Divergence signal dictionary
            
        Returns:
            Signal with confirmation data
        """
        signal = signal.copy()
        
        if signal['type'] == 'bullish':
            # Check if price is near support level
            current_price = data['close'].iloc[-1]
            support_level = signal['price_low2_val']
            support_distance = abs(current_price - support_level) / support_level
            
            signal['support_confirmation'] = support_distance < 0.05  # Within 5%
            signal['support_level'] = support_level
            signal['support_distance'] = support_distance
            
        elif signal['type'] == 'bearish':
            # Check if price is near resistance level
            current_price = data['close'].iloc[-1]
            resistance_level = signal['price_high2_val']
            resistance_distance = abs(current_price - resistance_level) / resistance_level
            
            signal['resistance_confirmation'] = resistance_distance < 0.05  # Within 5%
            signal['resistance_level'] = resistance_level
            signal['resistance_distance'] = resistance_distance
        
        return signal
    
    def confirm_with_candlestick_patterns(self, data: pd.DataFrame, signal: Dict) -> Dict:
        """
        Confirm divergence signal with candlestick patterns
        
        Args:
            data: DataFrame with price data
            signal: Divergence signal dictionary
            
        Returns:
            Signal with candlestick confirmation
        """
        signal = signal.copy()
        
        # Get recent candlestick data
        recent_data = data.tail(5)
        
        # Check for bullish patterns (hammer, doji, engulfing)
        if signal['type'] == 'bullish':
            pattern = self._detect_bullish_patterns(recent_data)
            signal['candlestick_pattern'] = pattern
            signal['pattern_confirmation'] = pattern is not None
        
        # Check for bearish patterns (shooting star, doji, engulfing)
        elif signal['type'] == 'bearish':
            pattern = self._detect_bearish_patterns(recent_data)
            signal['candlestick_pattern'] = pattern
            signal['pattern_confirmation'] = pattern is not None
        
        return signal
    
    def _detect_bullish_patterns(self, data: pd.DataFrame) -> Optional[str]:
        """Detect bullish candlestick patterns"""
        if len(data) < 2:
            return None
        
        # Hammer pattern
        for i in range(len(data)):
            candle = data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            
            if (lower_shadow > 2 * body_size and upper_shadow < body_size):
                return 'hammer'
        
        # Doji pattern
        for i in range(len(data)):
            candle = data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if body_size < 0.1 * total_range:
                return 'doji'
        
        return None
    
    def _detect_bearish_patterns(self, data: pd.DataFrame) -> Optional[str]:
        """Detect bearish candlestick patterns"""
        if len(data) < 2:
            return None
        
        # Shooting star pattern
        for i in range(len(data)):
            candle = data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            
            if (upper_shadow > 2 * body_size and lower_shadow < body_size):
                return 'shooting_star'
        
        # Doji pattern
        for i in range(len(data)):
            candle = data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if body_size < 0.1 * total_range:
                return 'doji'
        
        return None
    
    def analyze_divergence(self, data: pd.DataFrame, confirm_signals: bool = True) -> Dict:
        """
        Complete divergence analysis
        
        Args:
            data: OHLCV DataFrame
            confirm_signals: Whether to confirm signals with additional analysis
            
        Returns:
            Dictionary with divergence analysis results
        """
        if len(data) < self.min_candles:
            return {
                'error': f'Insufficient data. Need at least {self.min_candles} candles, got {len(data)}'
            }
        
        # Detect timeframe from data
        timeframe = self._detect_timeframe(data)
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Find swing points for price and indicators
        price_highs, price_lows = self.find_swing_points(df, 'close')
        rsi_highs, rsi_lows = self.find_swing_points(df, 'rsi')
        macd_highs, macd_lows = self.find_swing_points(df, 'macd')
        
        # Filter significant swings
        price_highs, price_lows = self.filter_significant_swings(df, price_highs, price_lows, 'close')
        rsi_highs, rsi_lows = self.filter_significant_swings(df, rsi_highs, rsi_lows, 'rsi')
        macd_highs, macd_lows = self.filter_significant_swings(df, macd_highs, macd_lows, 'macd')
        
        # Detect divergence
        rsi_signals = self.detect_divergence(df, price_highs, price_lows, rsi_highs, rsi_lows, 'rsi')
        macd_signals = self.detect_divergence(df, price_highs, price_lows, macd_highs, macd_lows, 'macd')
        
        all_signals = rsi_signals + macd_signals
        
        # Confirm signals if requested
        if confirm_signals and all_signals:
            for i, signal in enumerate(all_signals):
                signal = self.confirm_with_support_resistance(df, signal)
                signal = self.confirm_with_candlestick_patterns(df, signal)
                all_signals[i] = signal
        
        # Sort signals by strength
        all_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'total_signals': len(all_signals),
            'rsi_signals': len(rsi_signals),
            'macd_signals': len(macd_signals),
            'signals': all_signals,
            'current_price': df['close'].iloc[-1],
            'current_rsi': df['rsi'].iloc[-1],
            'current_macd': df['macd'].iloc[-1],
            'timeframe': timeframe,
            'data_points': len(data),
            'date_range': {
                'start': data.index[0],
                'end': data.index[-1],
                'duration': data.index[-1] - data.index[0]
            },
            'analysis_date': datetime.now()
        }
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """
        Detect the timeframe of the data based on time intervals
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            String indicating the detected timeframe
        """
        if len(data) < 2:
            return 'Unknown'
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            return 'Unknown'
        
        # Calculate time differences between consecutive rows
        time_diffs = data.index.to_series().diff().dropna()
        
        if len(time_diffs) == 0:
            return 'Unknown'
        
        # Get the most common time difference
        most_common_diff = time_diffs.mode().iloc[0]
        
        # Check if it's a timedelta object
        if not hasattr(most_common_diff, 'total_seconds'):
            return 'Unknown'
        
        # Convert to total seconds
        total_seconds = most_common_diff.total_seconds()
        
        # Determine timeframe
        if total_seconds < 60:
            return f'{int(total_seconds)}s'
        elif total_seconds < 3600:
            minutes = int(total_seconds / 60)
            return f'{minutes}m'
        elif total_seconds < 86400:
            hours = int(total_seconds / 3600)
            return f'{hours}h'
        elif total_seconds < 604800:
            days = int(total_seconds / 86400)
            return f'{days}d'
        elif total_seconds < 2592000:
            weeks = int(total_seconds / 604800)
            return f'{weeks}w'
        else:
            months = int(total_seconds / 2592000)
            return f'{months}M'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals DataFrame
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with trading signals
        """
        analysis = self.analyze_divergence(data)
        
        if 'error' in analysis:
            return pd.DataFrame()
        
        signals = []
        for signal in analysis['signals']:
            signal_row = {
                'timestamp': signal['signal_date'],
                'type': signal['type'],
                'indicator': signal['indicator'],
                'strength': signal['strength'],
                'price_level': signal.get('price_high2_val', signal.get('price_low2_val')),
                'indicator_level': signal.get('indicator_high2_val', signal.get('indicator_low2_val')),
                'support_resistance_confirmed': signal.get('support_confirmation', signal.get('resistance_confirmation', False)),
                'candlestick_confirmed': signal.get('pattern_confirmation', False),
                'signal': 1 if signal['type'] == 'bullish' else -1
            }
            signals.append(signal_row)
        
        return pd.DataFrame(signals)
    
    def print_analysis_report(self, analysis: Dict):
        """Print a formatted analysis report"""
        print("🔍 DIVERGENCE ANALYSIS REPORT")
        print("=" * 60)
        print(f"⏰ Timeframe: {analysis.get('timeframe', 'Unknown')}")
        print(f"📊 Data Points: {analysis.get('data_points', 'Unknown')}")
        if 'date_range' in analysis:
            print(f"📅 Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            print(f"⏱️ Duration: {analysis['date_range']['duration']}")
        print(f"📊 Total Signals: {analysis['total_signals']}")
        print(f"📈 RSI Signals: {analysis['rsi_signals']}")
        print(f"📉 MACD Signals: {analysis['macd_signals']}")
        print(f"💰 Current Price: ${analysis['current_price']:.4f}")
        print(f"📊 Current RSI: {analysis['current_rsi']:.2f}")
        print(f"📈 Current MACD: {analysis['current_macd']:.6f}")
        print()
        
        if analysis['signals']:
            print("🎯 DETECTED SIGNALS:")
            print("-" * 60)
            
            for i, signal in enumerate(analysis['signals'], 1):
                print(f"Signal {i}:")
                print(f"  Type: {signal['type'].upper()} Divergence")
                print(f"  Indicator: {signal['indicator'].upper()}")
                print(f"  Strength: {signal['strength']:.2f}")
                print(f"  Date: {signal['signal_date']}")
                
                if signal['type'] == 'bullish':
                    print(f"  Price: ${signal['price_low1_val']:.4f} → ${signal['price_low2_val']:.4f}")
                    print(f"  {signal['indicator'].upper()}: {signal['indicator_low1_val']:.2f} → {signal['indicator_low2_val']:.2f}")
                else:
                    print(f"  Price: ${signal['price_high1_val']:.4f} → ${signal['price_high2_val']:.4f}")
                    print(f"  {signal['indicator'].upper()}: {signal['indicator_high1_val']:.2f} → {signal['indicator_high2_val']:.2f}")
                
                if 'support_confirmation' in signal:
                    print(f"  Support/Resistance Confirmed: {signal.get('support_confirmation', signal.get('resistance_confirmation', False))}")
                
                if 'pattern_confirmation' in signal:
                    print(f"  Candlestick Pattern: {signal.get('candlestick_pattern', 'None')}")
                
                print()
        else:
            print("❌ No Class A divergence signals detected")
        
        print("=" * 60)


def main():
    """Example usage of the divergence detector"""
    # Example data (you would load your actual OHLCV data here)
    print("🚀 DIVERGENCE DETECTOR - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create sample data for demonstration
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Create sample price data with some divergence patterns
    base_price = 100
    prices = []
    for i in range(100):
        if i < 30:
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 60:
            price = base_price + 15 - (i - 30) * 0.3 + np.random.normal(0, 1)
        else:
            price = base_price + 6 + (i - 60) * 0.2 + np.random.normal(0, 1)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in range(100)]
    }, index=dates)
    
    # Initialize divergence detector
    detector = DivergenceDetector(
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        min_candles=50,
        swing_threshold=0.02
    )
    
    # Analyze divergence
    analysis = detector.analyze_divergence(sample_data, confirm_signals=True)
    
    # Print report
    detector.print_analysis_report(analysis)
    
    # Generate signals DataFrame
    signals_df = detector.generate_signals(sample_data)
    if not signals_df.empty:
        print("📋 SIGNALS DATAFRAME:")
        print(signals_df.to_string(index=False))
    else:
        print("📋 No signals generated")


if __name__ == "__main__":
    main() 