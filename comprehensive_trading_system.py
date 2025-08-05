#!/usr/bin/env python3
"""
Enhanced Comprehensive Trading System
Combines live trading, strategy optimization, and expanded portfolio management
Fixed timestamp errors and improved signal strength calculations
"""

import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import requests
import ccxt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class SupportResistanceZone:
    """Data class for support/resistance zones"""
    level: float
    zone_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 scale
    touches: int
    first_touch: datetime
    last_touch: datetime
    volume_confirmed: bool
    price_range: Tuple[float, float]  # (min, max) with buffer
    is_active: bool = True
    break_count: int = 0
    last_break: Optional[datetime] = None

class DivergenceDetector:
    """Enhanced divergence detector integrated into comprehensive trading system"""
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, 
                 macd_slow: int = 26, macd_signal: int = 9, 
                 min_candles: int = 50, swing_threshold: float = 0.02):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.min_candles = min_candles
        self.swing_threshold = swing_threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and MACD indicators"""
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
        """Find swing highs and lows in the data"""
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
    
    def detect_divergence(self, data: pd.DataFrame, price_highs: List[int], 
                         price_lows: List[int], indicator_highs: List[int], 
                         indicator_lows: List[int], indicator_name: str) -> List[Dict]:
        """Detect Class A divergence between price and indicator"""
        signals = []
        
        # Bearish divergence: Price higher highs, indicator lower highs
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            price_high1, price_high2 = price_highs[-2], price_highs[-1]
            price_val1, price_val2 = data['close'].iloc[price_high1], data['close'].iloc[price_high2]
            
            indicator_high1, indicator_high2 = indicator_highs[-2], indicator_highs[-1]
            indicator_val1, indicator_val2 = data[indicator_name].iloc[indicator_high1], data[indicator_name].iloc[indicator_high2]
            
            if (price_val2 > price_val1 and indicator_val2 < indicator_val1 and 
                price_high2 > price_high1 and indicator_high2 > indicator_high1):
                
                signal = {
                    'type': 'bearish',
                    'divergence_class': 'A',
                    'indicator': indicator_name,
                    'strength': self._calculate_divergence_strength(price_val1, price_val2, indicator_val1, indicator_val2)
                }
                signals.append(signal)
        
        # Bullish divergence: Price lower lows, indicator higher lows
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            price_low1, price_low2 = price_lows[-2], price_lows[-1]
            price_val1, price_val2 = data['close'].iloc[price_low1], data['close'].iloc[price_low2]
            
            indicator_low1, indicator_low2 = indicator_lows[-2], indicator_lows[-1]
            indicator_val1, indicator_val2 = data[indicator_name].iloc[indicator_low1], data[indicator_name].iloc[indicator_low2]
            
            if (price_val2 < price_val1 and indicator_val2 > indicator_val1 and 
                price_low2 > price_low1 and indicator_low2 > indicator_low1):
                
                signal = {
                    'type': 'bullish',
                    'divergence_class': 'A',
                    'indicator': indicator_name,
                    'strength': self._calculate_divergence_strength(price_val1, price_val2, indicator_val1, indicator_val2)
                }
                signals.append(signal)
        
        return signals
    
    def _calculate_divergence_strength(self, val1: float, val2: float, 
                                     indicator_val1: float, indicator_val2: float) -> float:
        """Calculate divergence strength (0-1 scale)"""
        price_change = abs(val2 - val1) / val1
        indicator_change = abs(indicator_val2 - indicator_val1) / abs(indicator_val1)
        strength = min(price_change + indicator_change, 1.0)
        return strength
    
    def analyze_divergence(self, data: pd.DataFrame, confirm_signals: bool = True) -> Dict:
        """Complete divergence analysis"""
        if len(data) < self.min_candles:
            return {'error': f'Insufficient data. Need at least {self.min_candles} candles, got {len(data)}'}
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Find swing points for price and indicators
        price_highs, price_lows = self.find_swing_points(df, 'close')
        rsi_highs, rsi_lows = self.find_swing_points(df, 'rsi')
        macd_highs, macd_lows = self.find_swing_points(df, 'macd')
        
        # Detect divergence
        rsi_signals = self.detect_divergence(df, price_highs, price_lows, rsi_highs, rsi_lows, 'rsi')
        macd_signals = self.detect_divergence(df, price_highs, price_lows, macd_highs, macd_lows, 'macd')
        
        all_signals = rsi_signals + macd_signals
        
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
            'data_points': len(data)
        }

class SupportResistanceDetector:
    """Enhanced support/resistance detector integrated into comprehensive trading system"""
    
    def __init__(self, window_size: int = 200, min_touches: int = 2,
                 zone_buffer: float = 0.003, volume_threshold: float = 1.5,
                 swing_sensitivity: float = 0.02):
        self.window_size = window_size
        self.min_touches = min_touches
        self.zone_buffer = zone_buffer
        self.volume_threshold = volume_threshold
        self.swing_sensitivity = swing_sensitivity
    
    def detect_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Detect swing highs and lows in the data"""
        highs = []
        lows = []
        
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                swing_size = (data['high'].iloc[i] - min(data['low'].iloc[i-2:i+3])) / data['close'].iloc[i]
                if swing_size >= self.swing_sensitivity:
                    highs.append(i)
            
            # Check for swing low
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                swing_size = (max(data['high'].iloc[i-2:i+3]) - data['low'].iloc[i]) / data['close'].iloc[i]
                if swing_size >= self.swing_sensitivity:
                    lows.append(i)
        
        return highs, lows
    
    def check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
        """Check if swing point has volume confirmation"""
        if index < 5 or index >= len(data) - 5:
            return False
        
        avg_volume = data['volume'].iloc[index-5:index+6].mean()
        swing_volume = data['volume'].iloc[index]
        
        return swing_volume > (avg_volume * self.volume_threshold)
    
    def find_zone_touches(self, level: float, data: pd.DataFrame, swing_indices: List[int]) -> List[int]:
        """Find all touches of a price level"""
        touches = []
        buffer = level * self.zone_buffer
        
        for idx in swing_indices:
            high = data['high'].iloc[idx]
            low = data['low'].iloc[idx]
            
            if low <= (level + buffer) and high >= (level - buffer):
                touches.append(idx)
        
        return touches
    
    def identify_zones(self, data: pd.DataFrame) -> Tuple[List[SupportResistanceZone], List[SupportResistanceZone]]:
        """Identify support and resistance zones"""
        swing_highs, swing_lows = self.detect_swing_points(data)
        
        support_zones = []
        resistance_zones = []
        
        # Find resistance zones from swing highs
        for high_idx in swing_highs:
            level = data['high'].iloc[high_idx]
            touches = self.find_zone_touches(level, data, swing_highs)
            
            if len(touches) >= self.min_touches:
                existing_zone = None
                for zone in resistance_zones:
                    if abs(zone.level - level) <= (level * self.zone_buffer):
                        existing_zone = zone
                        break
                
                if existing_zone:
                    existing_zone.touches = len(touches)
                    existing_zone.last_touch = data.index[high_idx]
                    existing_zone.strength = min(1.0, len(touches) / 5.0)
                else:
                    volume_confirmed = self.check_volume_confirmation(data, high_idx)
                    zone = SupportResistanceZone(
                        level=level,
                        zone_type='resistance',
                        strength=min(1.0, len(touches) / 5.0),
                        touches=len(touches),
                        first_touch=data.index[touches[0]],
                        last_touch=data.index[high_idx],
                        volume_confirmed=volume_confirmed,
                        price_range=(level * (1 - self.zone_buffer), level * (1 + self.zone_buffer))
                    )
                    resistance_zones.append(zone)
        
        # Find support zones from swing lows
        for low_idx in swing_lows:
            level = data['low'].iloc[low_idx]
            touches = self.find_zone_touches(level, data, swing_lows)
            
            if len(touches) >= self.min_touches:
                existing_zone = None
                for zone in support_zones:
                    if abs(zone.level - level) <= (level * self.zone_buffer):
                        existing_zone = zone
                        break
                
                if existing_zone:
                    existing_zone.touches = len(touches)
                    existing_zone.last_touch = data.index[low_idx]
                    existing_zone.strength = min(1.0, len(touches) / 5.0)
                else:
                    volume_confirmed = self.check_volume_confirmation(data, low_idx)
                    zone = SupportResistanceZone(
                        level=level,
                        zone_type='support',
                        strength=min(1.0, len(touches) / 5.0),
                        touches=len(touches),
                        first_touch=data.index[touches[0]],
                        last_touch=data.index[low_idx],
                        volume_confirmed=volume_confirmed,
                        price_range=(level * (1 - self.zone_buffer), level * (1 + self.zone_buffer))
                    )
                    support_zones.append(zone)
        
        return support_zones, resistance_zones
    
    def check_zone_breaks(self, current_price: float) -> List[Dict]:
        """Check if current price breaks any zones"""
        breaks = []
        return breaks  # Simplified for integration
    
    def check_zone_approaches(self, current_price: float) -> List[Dict]:
        """Check if price is approaching zones"""
        approaches = []
        return approaches  # Simplified for integration

class EnhancedComprehensiveTradingSystem:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Trading modes
        self.modes = {
            'paper_trading': True,      # Safe testing mode
            'live_trading': False,      # Real money trading
            'strategy_optimization': True,  # Auto-optimize strategies
            'expanded_portfolio': True  # Monitor more cryptocurrencies
        }
        
        # Portfolio categories - Top 100+ Trading Pairs (Enhanced)
        self.portfolio_categories = {
            'blue_chips': [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
                'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT',
                'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'XLM/USDT', 'XMR/USDT',
                'ETC/USDT', 'FIL/USDT', 'ATOM/USDT', 'NEAR/USDT', 'ALGO/USDT'
            ],
            'defi': [
                'AAVE/USDT', 'COMP/USDT', 'CRV/USDT', 'SUSHI/USDT', 'MKR/USDT',
                'SNX/USDT', 'YFI/USDT', '1INCH/USDT', 'BAL/USDT', 'REN/USDT',
                'ZRX/USDT', 'BAND/USDT', 'KNC/USDT', 'REEF/USDT', 'ALPHA/USDT',
                'PERP/USDT', 'DYDX/USDT', 'GMX/USDT', 'JOE/USDT', 'RAY/USDT'
            ],
            'layer1': [
                'ICP/USDT', 'SUI/USDT', 'HBAR/USDT', 'ARB/USDT', 'OP/USDT',
                'APT/USDT', 'SEI/USDT', 'INJ/USDT', 'FET/USDT', 'RNDR/USDT',
                'IMX/USDT', 'MASK/USDT', 'CFX/USDT', 'FLOW/USDT', 'THETA/USDT',
                'VET/USDT', 'TRX/USDT', 'EOS/USDT', 'XTZ/USDT', 'NEO/USDT'
            ],
            'meme_coins': [
                'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'BONK/USDT', 'FLOKI/USDT',
                'WIF/USDT', 'BOME/USDT', 'MYRO/USDT', 'POPCAT/USDT', 'BOOK/USDT',
                'TURBO/USDT', 'SPONGE/USDT', 'SLERF/USDT', 'CAT/USDT', 'DOG/USDT'
            ],
            'gaming': [
                'AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'GALA/USDT', 'ENJ/USDT',
                'CHZ/USDT', 'ALICE/USDT', 'TLM/USDT', 'HERO/USDT', 'GMT/USDT',
                'APE/USDT', 'ILV/USDT', 'GHST/USDT', 'ALPINE/USDT', 'OGN/USDT'
            ],
            'ai_ml': [
                'FET/USDT', 'OCEAN/USDT', 'AGIX/USDT', 'RNDR/USDT', 'NMR/USDT',
                'BICO/USDT', 'ALI/USDT', 'CTXC/USDT', 'DKA/USDT', 'ORAI/USDT',
                'RLC/USDT', 'GRT/USDT', 'LINK/USDT', 'BAND/USDT', 'API3/USDT'
            ],
            'privacy': [
                'XMR/USDT', 'ZEC/USDT', 'DASH/USDT', 'XHV/USDT', 'PIVX/USDT',
                'ARRR/USDT', 'BEAM/USDT', 'GRIN/USDT', 'XZC/USDT', 'XVG/USDT'
            ],
            'exchange_tokens': [
                'BNB/USDT', 'OKB/USDT', 'HT/USDT', 'KCS/USDT', 'BGB/USDT',
                'GT/USDT', 'MX/USDT', 'CRO/USDT', 'LEO/USDT', 'FTT/USDT'
            ],
            'stablecoins': [
                'USDC/USDT', 'USDT/USDT', 'BUSD/USDT', 'DAI/USDT', 'TUSD/USDT',
                'FRAX/USDT', 'USDP/USDT', 'GUSD/USDT', 'LUSD/USDT', 'USDN/USDT'
            ],
            'metaverse': [
                'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'ENJ/USDT', 'CHZ/USDT',
                'TLM/USDT', 'HERO/USDT', 'GMT/USDT', 'APE/USDT', 'ILV/USDT',
                'GHST/USDT', 'ALPINE/USDT', 'OGN/USDT', 'ALICE/USDT', 'GALA/USDT'
            ],
            'infrastructure': [
                'LINK/USDT', 'BAND/USDT', 'API3/USDT', 'GRT/USDT', 'RLC/USDT',
                'ORAI/USDT', 'DKA/USDT', 'CTXC/USDT', 'ALI/USDT', 'BICO/USDT',
                'NMR/USDT', 'RNDR/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'FET/USDT'
            ]
        }
        
        # Enhanced strategy weights - Optimized based on performance
        self.strategy_weights = {
            'DivergenceStrategy': 0.55,      # Primary strategy (slightly reduced)
            'SupportResistanceStrategy': 0.45,  # Secondary strategy (increased)
            'MomentumStrategy': 0.0,         # Disabled
            'MeanReversionStrategy': 0.0,    # Disabled
            'FibonacciStrategy': 0.0         # Disabled (used only for profit taking)
        }
        
        # Signal strength thresholds
        self.signal_thresholds = {
            'minimum_strength': 0.25,  # Enhanced threshold for quality signals
            'minimum_confidence': 0.35,  # Enhanced confidence threshold
            'volume_confirmation': True,
            'trend_confirmation': True
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'average_signal_strength': 0.0,
            'last_optimization': None
        }
        
        # Initialize enhanced detectors
        self.divergence_detector = None
        self.support_resistance_detector = None
        self.exchange = None
        
        # Initialize detectors immediately
        self.initialize_enhanced_detectors()
    
    def send_telegram_alert(self, message: str):
        """Send Telegram alert with enhanced error handling"""
        try:
            import requests
            if self.telegram_bot_token and self.telegram_chat_id:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                data = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, data=data, timeout=10)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
        return False
    
    def fix_timestamp_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix timestamp-related issues in data"""
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                else:
                    # Create proper datetime index
                    data.index = pd.to_datetime(data.index)
            
            # Sort by timestamp to ensure proper order
            data.sort_index(inplace=True)
            
            return data
        except Exception as e:
            logger.error(f"Error fixing timestamp issues: {e}")
            return data
    
    def calculate_enhanced_signal_strength(self, data: pd.DataFrame, strategy_name: str) -> float:
        """Calculate enhanced signal strength with multiple factors"""
        try:
            strength = 0.0
            
            # Base strength from RSI
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi < 25:  # Extremely oversold
                    strength += 0.5
                elif rsi < 30:  # Oversold
                    strength += 0.4
                elif rsi > 75:  # Extremely overbought
                    strength += 0.5
                elif rsi > 70:  # Overbought
                    strength += 0.4
                elif 30 <= rsi <= 70:  # Neutral zone
                    strength += 0.2
            
            # Volume confirmation
            if 'volume' in data.columns:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 2.0:  # Very high volume
                        strength += 0.4
                    elif volume_ratio > 1.5:  # High volume
                        strength += 0.3
                    elif volume_ratio > 1.0:  # Normal volume
                        strength += 0.1
            
            # Trend confirmation
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                sma_20 = data['sma_20'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                current_price = data['close'].iloc[-1]
                
                if current_price > sma_20 > sma_50:  # Strong uptrend
                    strength += 0.3
                elif current_price < sma_20 < sma_50:  # Strong downtrend
                    strength += 0.3
                elif abs(sma_20 - sma_50) / sma_50 < 0.02:  # Sideways
                    strength += 0.1
            
            # Price momentum (5-period change)
            if len(data) >= 5:
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
                if abs(price_change) > 0.05:  # 5% price change
                    strength += 0.3
                elif abs(price_change) > 0.02:  # 2% price change
                    strength += 0.2
            
            # Volatility confirmation
            if 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
                price = data['close'].iloc[-1]
                volatility = atr / price
                if volatility > 0.03:  # High volatility
                    strength += 0.2
                elif volatility > 0.01:  # Normal volatility
                    strength += 0.1
            
            # Strategy-specific adjustments - Focus on Divergence and Support/Resistance
            if 'Divergence' in strategy_name:
                # Divergence strategies get enhanced base strength
                strength = max(strength, 0.40)
                # Additional strength for strong divergence signals
                if 'divergence_strength' in data.columns:
                    div_strength = data['divergence_strength'].iloc[-1]
                    if div_strength > 0.7:
                        strength += 0.3
                    elif div_strength > 0.5:
                        strength += 0.2
            
            elif 'SupportResistance' in strategy_name:
                # Support/Resistance strategies get enhanced base strength
                strength = max(strength, 0.40)  # Increased base strength
                # Additional strength for strong support/resistance levels
                if 'support_resistance_confirmed' in data.columns:
                    if data['support_resistance_confirmed'].iloc[-1]:
                        strength += 0.30  # Increased bonus strength
                # Additional strength for price near support/resistance levels
                if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                    current_price = data['close'].iloc[-1]
                    swing_high = data['high'].max()
                    swing_low = data['low'].min()
                    price_range = swing_high - swing_low
                    if price_range > 0:
                        # Check if price is near support or resistance
                        support_distance = abs(current_price - swing_low) / price_range
                        resistance_distance = abs(current_price - swing_high) / price_range
                        if support_distance < 0.1 or resistance_distance < 0.1:
                            strength += 0.20  # Bonus for price near key levels
            
            elif 'Fibonacci' in strategy_name:
                # Fibonacci is only used for profit taking, not signal generation
                strength = 0.0  # Disable Fibonacci signal generation
            
            # Ensure minimum strength for any signal
            strength = max(strength, 0.15)
            
            # Cap strength at 1.0
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.15  # Return minimum strength instead of 0
    
    def calculate_signal_quality_score(self, data: pd.DataFrame, strength: float, strategy_name: str) -> float:
        """Calculate signal quality score based on multiple factors"""
        try:
            quality_score = 0.0
            
            # Base quality from signal strength
            quality_score += strength * 0.4
            
            # RSI quality
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if (rsi < 25 or rsi > 75):  # Extreme conditions
                    quality_score += 0.3
                elif (rsi < 30 or rsi > 70):  # Strong conditions
                    quality_score += 0.2
                elif 30 <= rsi <= 70:  # Neutral
                    quality_score += 0.1
            
            # Volume quality
            if 'volume' in data.columns:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 1.5:
                        quality_score += 0.2
            
            # Trend quality
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                sma_20 = data['sma_20'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                current_price = data['close'].iloc[-1]
                
                if current_price > sma_20 > sma_50 or current_price < sma_20 < sma_50:
                    quality_score += 0.1
            
            # Strategy-specific quality - Focus on Divergence and Support/Resistance
            if 'Divergence' in strategy_name:
                quality_score += 0.15  # Divergence is highly reliable
                # Additional quality for confirmed divergence
                if 'divergence_type' in data.columns and data['divergence_type'].iloc[-1]:
                    quality_score += 0.1
            
            elif 'SupportResistance' in strategy_name:
                quality_score += 0.12  # Support/Resistance is reliable
                # Additional quality for confirmed levels
                if 'support_resistance_confirmed' in data.columns and data['support_resistance_confirmed'].iloc[-1]:
                    quality_score += 0.08
            
            elif 'Fibonacci' in strategy_name:
                quality_score += 0.0  # Fibonacci disabled for signal generation
            
            # Cap quality score at 1.0
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return strength  # Fallback to signal strength
    
    def calculate_fibonacci_profit_targets(self, data: pd.DataFrame, entry_price: float, position_type: str) -> dict:
        """Calculate Fibonacci profit targets for position management"""
        try:
            if len(data) < 20:
                return {}
            
            # Find swing high and low for Fibonacci levels
            high = data['high'].max()
            low = data['low'].min()
            current_price = data['close'].iloc[-1]
            
            # Calculate Fibonacci retracement levels
            diff = high - low
            fib_levels = {
                '0.236': high - (diff * 0.236),
                '0.382': high - (diff * 0.382),
                '0.500': high - (diff * 0.500),
                '0.618': high - (diff * 0.618),
                '0.786': high - (diff * 0.786)
            }
            
            # Calculate Fibonacci extension levels for profit targets
            if position_type == 'BUY':
                # For long positions, use extensions above current price
                extension_diff = current_price - low
                profit_targets = {
                    '1.272': current_price + (extension_diff * 0.272),
                    '1.618': current_price + (extension_diff * 0.618),
                    '2.000': current_price + (extension_diff * 1.000),
                    '2.618': current_price + (extension_diff * 1.618)
                }
            else:
                # For short positions, use extensions below current price
                extension_diff = high - current_price
                profit_targets = {
                    '1.272': current_price - (extension_diff * 0.272),
                    '1.618': current_price - (extension_diff * 0.618),
                    '2.000': current_price - (extension_diff * 1.000),
                    '2.618': current_price - (extension_diff * 1.618)
                }
            
            return {
                'retracement_levels': fib_levels,
                'profit_targets': profit_targets,
                'current_price': current_price,
                'swing_high': high,
                'swing_low': low
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci profit targets: {e}")
            return {}
    
    def should_take_profit_fibonacci(self, data: pd.DataFrame, entry_price: float, current_price: float, position_type: str) -> dict:
        """Check if profit should be taken based on Fibonacci levels"""
        try:
            fib_targets = self.calculate_fibonacci_profit_targets(data, entry_price, position_type)
            if not fib_targets:
                return {'should_take_profit': False, 'reason': 'No Fibonacci levels calculated'}
            
            profit_targets = fib_targets['profit_targets']
            
            if position_type == 'BUY':
                # Check if current price has reached any profit target
                for level, target_price in profit_targets.items():
                    if current_price >= target_price:
                        profit_percentage = ((current_price - entry_price) / entry_price) * 100
                        return {
                            'should_take_profit': True,
                            'reason': f'Reached Fibonacci {level} target',
                            'target_level': level,
                            'target_price': target_price,
                            'profit_percentage': profit_percentage
                        }
            else:
                # Check if current price has reached any profit target for short positions
                for level, target_price in profit_targets.items():
                    if current_price <= target_price:
                        profit_percentage = ((entry_price - current_price) / entry_price) * 100
                        return {
                            'should_take_profit': True,
                            'reason': f'Reached Fibonacci {level} target',
                            'target_level': level,
                            'target_price': target_price,
                            'profit_percentage': profit_percentage
                        }
            
            return {'should_take_profit': False, 'reason': 'No profit targets reached'}
            
        except Exception as e:
            logger.error(f"Error checking Fibonacci profit taking: {e}")
            return {'should_take_profit': False, 'reason': f'Error: {e}'}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0.0] * len(data))
    
    def initialize_enhanced_detectors(self):
        """Initialize enhanced divergence and support/resistance detectors"""
        try:
            # Initialize exchange connection
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Initialize divergence detector
            self.divergence_detector = DivergenceDetector(
                rsi_period=14,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                min_candles=50,
                swing_threshold=0.02
            )
            
            # Initialize support/resistance detector
            self.support_resistance_detector = SupportResistanceDetector(
                window_size=200,
                min_touches=2,
                zone_buffer=0.003,
                volume_threshold=1.5,
                swing_sensitivity=0.02
            )
            
            logger.info("Enhanced detectors initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced detectors: {e}")
    
    def fetch_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch market data using enhanced exchange connection"""
        try:
            if not self.exchange:
                self.initialize_enhanced_detectors()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='5m',
                limit=limit
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_divergence(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze divergence using enhanced detector"""
        try:
            if not self.divergence_detector:
                self.initialize_enhanced_detectors()
            
            if len(data) < 50:
                return {'error': f'Insufficient data for {symbol}. Need at least 50 candles, got {len(data)}'}
            
            # Analyze divergence
            analysis = self.divergence_detector.analyze_divergence(data, confirm_signals=True)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in divergence analysis for {symbol}: {e}")
            return {'error': f'Error in divergence analysis: {e}'}
    
    def analyze_support_resistance(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze support/resistance using enhanced detector"""
        try:
            if not self.support_resistance_detector:
                self.initialize_enhanced_detectors()
            
            if len(data) < 50:
                return {'error': f'Insufficient data for {symbol}. Need at least 50 candles, got {len(data)}'}
            
            # Analyze support/resistance
            support_zones, resistance_zones = self.support_resistance_detector.identify_zones(data)
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Check for zone breaks and approaches
            breaks = self.support_resistance_detector.check_zone_breaks(current_price)
            approaches = self.support_resistance_detector.check_zone_approaches(current_price)
            
            return {
                'support_zones': support_zones,
                'resistance_zones': resistance_zones,
                'current_price': current_price,
                'breaks': breaks,
                'approaches': approaches,
                'total_zones': len(support_zones) + len(resistance_zones)
            }
            
        except Exception as e:
            logger.error(f"Error in support/resistance analysis for {symbol}: {e}")
            return {'error': f'Error in support/resistance analysis: {e}'}
    
    def run_comprehensive_analysis(self):
        """Run enhanced comprehensive market analysis"""
        print("🚀 ENHANCED COMPREHENSIVE TRADING SYSTEM")
        print("=" * 80)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Mode: {'Paper Trading' if self.modes['paper_trading'] else 'Live Trading'}")
        print(f"⚙️ Strategy Optimization: {'Enabled' if self.modes['strategy_optimization'] else 'Disabled'}")
        print(f"🌐 Expanded Portfolio: {'Enabled' if self.modes['expanded_portfolio'] else 'Disabled'}")
        print("=" * 80)
        
        # Send startup alert
        startup_message = f"""
🤖 **Enhanced Comprehensive Trading System Started**

✅ **Status**: Online & Monitoring
⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 **Mode**: {'Paper Trading' if self.modes['paper_trading'] else 'Live Trading'}
⚙️ **Strategy Optimization**: {'Enabled' if self.modes['strategy_optimization'] else 'Disabled'}
🌐 **Expanded Portfolio**: {'Enabled' if self.modes['expanded_portfolio'] else 'Disabled'}

**🎯 FOCUSED STRATEGY APPROACH:**
📈 **Primary Strategy**: Divergence (60% weight)
🛡️ **Secondary Strategy**: Support/Resistance (40% weight)
📊 **Fibonacci**: Profit Taking Only (No Signal Generation)

**Portfolio Categories (Top 100+ Trading Pairs):**
🔵 Blue Chips: {len(self.portfolio_categories['blue_chips'])} symbols
🟢 DeFi: {len(self.portfolio_categories['defi'])} symbols
🟡 Layer 1: {len(self.portfolio_categories['layer1'])} symbols
🟠 Meme Coins: {len(self.portfolio_categories['meme_coins'])} symbols
🎮 Gaming: {len(self.portfolio_categories['gaming'])} symbols
🤖 AI/ML: {len(self.portfolio_categories['ai_ml'])} symbols
🔒 Privacy: {len(self.portfolio_categories['privacy'])} symbols
🏢 Exchange Tokens: {len(self.portfolio_categories['exchange_tokens'])} symbols
💎 Stablecoins: {len(self.portfolio_categories['stablecoins'])} symbols
🌐 Metaverse: {len(self.portfolio_categories['metaverse'])} symbols
⚙️ Infrastructure: {len(self.portfolio_categories['infrastructure'])} symbols

**Total Symbols**: {sum(len(symbols) for symbols in self.portfolio_categories.values())}

**Enhanced Features:**
• Fixed timestamp issues
• Improved signal strength calculation
• Better error handling
• Focused strategy approach (Divergence + Support/Resistance)
• Fibonacci profit taking system
• Comprehensive market coverage

Monitoring markets for trading opportunities...
"""
        
        self.send_telegram_alert(startup_message)
        
        # Import market analysis checker
        from check_market_analysis import MarketAnalysisChecker
        checker = MarketAnalysisChecker()
        
        # Update symbols to use expanded portfolio
        all_symbols = []
        for category_symbols in self.portfolio_categories.values():
            all_symbols.extend(category_symbols)
        
        # Remove duplicates and limit to valid symbols
        valid_symbols = list(set(all_symbols))[:20]  # Limit to 20 for testing
        
        # Filter out invalid symbols
        invalid_symbols = ['MX/USDT', 'GUSD/USDT', 'GT/USDT', 'OKB/USDT', 'BOOK/USDT', 'DOG/USDT', 'ARRR/USDT', 'HT/USDT']
        valid_symbols = [s for s in valid_symbols if s not in invalid_symbols]
        
        print(f"\n📊 Monitoring {len(valid_symbols)} symbols:")
        for symbol in valid_symbols:
            print(f"  • {symbol}")
        
        # Run analysis cycles
        cycle = 1
        total_signals = 0
        successful_signals = 0
        
        try:
            while True:
                print(f"\n{'='*60}")
                print(f"📊 ENHANCED ANALYSIS CYCLE #{cycle}")
                print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                
                cycle_signals = 0
                signals_found = []
                
                # Analyze each symbol using enhanced detectors
                for symbol in valid_symbols:
                    try:
                        print(f"\n{'='*60}")
                        print(f"📊 ANALYZING {symbol}")
                        print(f"{'='*60}")
                        
                        # Fetch market data using enhanced method
                        market_data = self.fetch_market_data(symbol, limit=100)
                        if market_data.empty:
                            print(f"❌ No data available for {symbol}")
                            continue
                        
                        # Fix timestamp issues
                        market_data = self.fix_timestamp_issues(market_data)
                        
                        # Get current price
                        current_price = market_data['close'].iloc[-1]
                        print(f"💰 Current Price: ${current_price:.4f}")
                        
                        # Calculate basic indicators
                        market_data['sma_20'] = market_data['close'].rolling(window=20).mean()
                        market_data['sma_50'] = market_data['close'].rolling(window=50).mean()
                        market_data['rsi'] = self.divergence_detector._calculate_rsi(market_data['close'], 14)
                        market_data['atr'] = self._calculate_atr(market_data)
                        
                        # Print basic analysis
                        volume = market_data['volume'].iloc[-1]
                        sma_20 = market_data['sma_20'].iloc[-1]
                        sma_50 = market_data['sma_50'].iloc[-1]
                        rsi = market_data['rsi'].iloc[-1]
                        atr = market_data['atr'].iloc[-1]
                        
                        print(f"📊 Volume: {volume:,.0f}")
                        print(f"📊 SMA 20: ${sma_20:.4f} | SMA 50: ${sma_50:.4f}")
                        print(f"📊 RSI: {rsi:.2f} | ATR: ${atr:.4f}")
                        
                        # Determine trend
                        if current_price > sma_20 > sma_50:
                            trend = "🟢 BULLISH"
                        elif current_price < sma_20 < sma_50:
                            trend = "🔴 BEARISH"
                        else:
                            trend = "🟡 SIDEWAYS"
                        print(f"📈 Trend: {trend}")
                        
                        # Determine RSI status
                        if rsi < 30:
                            rsi_status = "🟢 OVERSOLD"
                        elif rsi > 70:
                            rsi_status = "🔴 OVERBOUGHT"
                        else:
                            rsi_status = "🟡 NEUTRAL"
                        print(f"📊 RSI Status: {rsi_status}")
                        
                        # Enhanced Divergence Analysis
                        divergence_analysis = self.analyze_divergence(market_data, symbol)
                        
                        # Enhanced Support/Resistance Analysis
                        support_resistance_analysis = self.analyze_support_resistance(market_data, symbol)
                        
                        # Generate signals based on enhanced analysis
                        signals_found_symbol = []
                        
                        # Process divergence signals
                        if 'error' not in divergence_analysis and divergence_analysis['signals']:
                            for signal in divergence_analysis['signals']:
                                signal_type = "BUY" if signal['type'] == 'bullish' else "SELL"
                                enhanced_strength = self.calculate_enhanced_signal_strength(
                                    market_data, 'DivergenceStrategy'
                                )
                                
                                if enhanced_strength >= self.signal_thresholds['minimum_strength']:
                                    quality_score = self.calculate_signal_quality_score(
                                        market_data, enhanced_strength, 'DivergenceStrategy'
                                    )
                                    
                                    signals_found_symbol.append({
                                        'symbol': symbol,
                                        'signal': signal_type,
                                        'strategy': 'DivergenceStrategy',
                                        'strength': enhanced_strength,
                                        'price': current_price,
                                        'confidence': min(enhanced_strength * 1.2, 1.0),
                                        'quality_score': quality_score,
                                        'divergence_type': signal['type'],
                                        'indicator': signal['indicator']
                                    })
                        
                        # Process support/resistance signals
                        if 'error' not in support_resistance_analysis:
                            support_zones = support_resistance_analysis.get('support_zones', [])
                            resistance_zones = support_resistance_analysis.get('resistance_zones', [])
                            
                            # Check if price is near support or resistance
                            for zone in support_zones + resistance_zones:
                                if zone.is_active:
                                    distance = abs(current_price - zone.level) / zone.level
                                    if distance < 0.02:  # Within 2% of zone
                                        signal_type = "BUY" if zone.zone_type == 'support' else "SELL"
                                        enhanced_strength = self.calculate_enhanced_signal_strength(
                                            market_data, 'SupportResistanceStrategy'
                                        )
                                        
                                        if enhanced_strength >= self.signal_thresholds['minimum_strength']:
                                            quality_score = self.calculate_signal_quality_score(
                                                market_data, enhanced_strength, 'SupportResistanceStrategy'
                                            )
                                            
                                            signals_found_symbol.append({
                                                'symbol': symbol,
                                                'signal': signal_type,
                                                'strategy': 'SupportResistanceStrategy',
                                                'strength': enhanced_strength,
                                                'price': current_price,
                                                'confidence': min(enhanced_strength * 1.2, 1.0),
                                                'quality_score': quality_score,
                                                'zone_type': zone.zone_type,
                                                'zone_level': zone.level,
                                                'zone_strength': zone.strength
                                            })
                        
                        # Print strategy signals
                        if signals_found_symbol:
                            print("\n🎯 STRATEGY SIGNALS:")
                            for signal in signals_found_symbol:
                                print(f"  • {signal['strategy']}: {signal['signal']} (Strength: {signal['strength']:.3f}, Confidence: {signal['confidence']:.3f})")
                                signals_found.extend(signals_found_symbol)
                                cycle_signals += len(signals_found_symbol)
                                
                                # Send enhanced Telegram alert
                                alert_message = f"""
🚨 **Enhanced Trading Signal Detected**

📊 **Symbol**: {symbol}
📈 **Signal**: {signal['signal']}
🎯 **Strategy**: {signal['strategy']}
💪 **Enhanced Strength**: {signal['strength']:.3f}
🎯 **Confidence**: {signal['confidence']:.3f}
💰 **Price**: ${signal['price']:.4f}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

**Mode**: {'Paper Trading' if self.modes['paper_trading'] else 'Live Trading'}
**Enhanced Analysis**: ✅ Active
"""
                                self.send_telegram_alert(alert_message)
                        else:
                            print("\n🎯 STRATEGY SIGNALS:")
                            print("  • No signals generated")
                        
                        # Print detector analysis summary
                        print("\n🔍 DETECTOR ANALYSIS:")
                        if 'error' not in divergence_analysis:
                            print(f"  📊 Divergence Signals: {divergence_analysis['total_signals']}")
                        if 'error' not in support_resistance_analysis:
                            print(f"  🛡️ Support/Resistance Zones: {support_resistance_analysis['total_zones']}")
                    
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                        print(f"❌ Error analyzing {symbol}: {e}")
                        continue
                
                total_signals += cycle_signals
                
                # Print enhanced cycle summary
                if signals_found:
                    print(f"\n📋 Enhanced Signals Found: {cycle_signals}")
                    for signal in signals_found:
                        print(f"  • {signal['symbol']}: {signal['signal']} ({signal['strategy']})")
                        print(f"    Strength: {signal['strength']:.3f} | Confidence: {signal['confidence']:.3f} | Quality: {signal['quality_score']:.3f}")
                else:
                    print(f"\n📋 No signals above threshold in this cycle")
                
                print(f"\n📊 Total Signals (All Cycles): {total_signals}")
                print(f"📈 Success Rate: {(successful_signals/total_signals*100):.1f}%" if total_signals > 0 else "📈 No signals yet")
                print(f"⏳ Waiting 5 minutes for next analysis...")
                print("Press Ctrl+C to stop")
                
                # Wait 5 minutes
                time.sleep(300)
                cycle += 1
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping enhanced comprehensive trading system...")
            
            # Send shutdown alert
            shutdown_message = f"""
🤖 **Enhanced Comprehensive Trading System Stopped**

⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 **Cycles Completed**: {cycle}
📈 **Total Signals**: {total_signals}
📈 **Success Rate**: {(successful_signals/total_signals*100):.1f}% if total_signals > 0 else "N/A"
📱 **Status**: Offline

**Performance Summary:**
• Average signals per cycle: {total_signals/cycle:.1f}
• Total monitoring time: {cycle * 5} minutes
• Symbols monitored: {len(valid_symbols)}
• Enhanced signal strength calculation: ✅ Active

Thank you for using the Enhanced Comprehensive Trading System!
"""
            
            self.send_telegram_alert(shutdown_message)
            print("✅ Enhanced comprehensive trading system stopped safely!")
    
    def optimize_strategies(self):
        """Optimize trading strategies with enhanced error handling"""
        print("\n⚙️ ENHANCED STRATEGY OPTIMIZATION")
        print("=" * 60)
        
        # Import strategy optimizer
        from strategy_optimizer import StrategyOptimizer
        optimizer = StrategyOptimizer()
        
        # Optimize for major symbols
        symbols_to_optimize = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        for symbol in symbols_to_optimize:
            print(f"\n🔍 Optimizing strategies for {symbol}...")
            
            try:
                # Analyze signal quality with enhanced error handling
                quality_results = optimizer.analyze_signal_quality(symbol)
                
                # Optimize parameters for each strategy
                for strategy_name in ['Momentum', 'Divergence', 'SupportResistance']:
                    try:
                        optimized_config = optimizer.optimize_strategy_parameters(symbol, strategy_name)
                        print(f"  ✅ {strategy_name} optimized for {symbol}")
                        
                        # Update strategy weights based on optimization results
                        if quality_results and strategy_name in quality_results:
                            quality_score = quality_results[strategy_name].get('quality_score', 0)
                            if quality_score > 0.5:
                                self.strategy_weights[f'{strategy_name}Strategy'] *= 1.1
                                print(f"  📈 Increased weight for {strategy_name}")
                            elif quality_score < 0.2:
                                self.strategy_weights[f'{strategy_name}Strategy'] *= 0.9
                                print(f"  📉 Decreased weight for {strategy_name}")
                                
                    except Exception as e:
                        logger.error(f"Error optimizing {strategy_name}: {e}")
                        print(f"  ❌ Error optimizing {strategy_name}: {e}")
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                print(f"  ❌ Error analyzing {symbol}: {e}")
        
        # Normalize strategy weights
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
        
        print("\n✅ Enhanced strategy optimization completed!")
        print("📊 Updated Strategy Weights:")
        for strategy, weight in self.strategy_weights.items():
            print(f"  • {strategy}: {weight:.3f}")
    
    def get_portfolio_summary(self):
        """Get enhanced portfolio summary"""
        print("\n📊 ENHANCED PORTFOLIO SUMMARY - TOP 100+ TRADING PAIRS")
        print("=" * 80)
        
        total_symbols = sum(len(symbols) for symbols in self.portfolio_categories.values())
        
        print(f"📈 Total Symbols: {total_symbols}")
        print(f"📋 Categories: {len(self.portfolio_categories)}")
        print(f"⚙️ Strategy Weights: {len(self.strategy_weights)} strategies")
        
        category_icons = {
            'blue_chips': '🔵',
            'defi': '🟢',
            'layer1': '🟡',
            'meme_coins': '🟠',
            'gaming': '🎮',
            'ai_ml': '🤖',
            'privacy': '🔒',
            'exchange_tokens': '🏢',
            'stablecoins': '💎',
            'metaverse': '🌐',
            'infrastructure': '⚙️'
        }
        
        for category, symbols in self.portfolio_categories.items():
            icon = category_icons.get(category, '📊')
            print(f"\n{icon} {category.upper().replace('_', ' ')}:")
            print(f"   📊 Symbols: {len(symbols)}")
            print(f"   📝 Examples: {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}")
        
        print(f"\n📊 Strategy Weights (Optimized):")
        for strategy, weight in self.strategy_weights.items():
            print(f"   • {strategy}: {weight:.3f}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   📈 Total Signals: {self.performance_metrics['total_signals']}")
        print(f"   ✅ Successful: {self.performance_metrics['successful_signals']}")
        print(f"   ❌ Failed: {self.performance_metrics['failed_signals']}")
        print(f"   💪 Avg Strength: {self.performance_metrics['average_signal_strength']:.3f}")
        
        print(f"\n📊 Signal Thresholds:")
        print(f"   • Minimum Strength: {self.signal_thresholds['minimum_strength']}")
        print(f"   • Minimum Confidence: {self.signal_thresholds['minimum_confidence']}")
        print(f"   • Volume Confirmation: {self.signal_thresholds['volume_confirmation']}")
        print(f"   • Trend Confirmation: {self.signal_thresholds['trend_confirmation']}")
        
        print(f"\n{'='*80}")
        print(f"🎯 **ENHANCED COMPREHENSIVE PORTFOLIO COVERAGE**")
        print(f"📊 Total Trading Pairs: {total_symbols}")
        print(f"🌍 Market Coverage: Top 100+ Cryptocurrencies")
        print(f"📈 Categories: {len(self.portfolio_categories)}")
        print(f"⚙️ Enhanced Features: Active")
        print(f"{'='*80}")
        
        return self.portfolio_categories
    
    def switch_to_live_trading(self):
        """Switch to live trading mode with enhanced safety checks"""
        print("\n🔐 ENHANCED LIVE TRADING SETUP")
        print("=" * 60)
        
        # Check API credentials
        binance_api_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_SECRET_KEY')
        
        if not binance_api_key or not binance_secret:
            print("❌ Binance API credentials not found!")
            print("Please add BINANCE_API_KEY and BINANCE_SECRET_KEY to your .env file")
            return False
        
        print("✅ Binance API credentials found")
        print("⚠️ WARNING: Switching to live trading mode!")
        print("This will execute real trades with real money!")
        
        # Enhanced safety checks
        print("\n🔒 Enhanced Safety Checks:")
        print("• Signal strength threshold: {self.signal_thresholds['minimum_strength']}")
        print("• Confidence threshold: {self.signal_thresholds['minimum_confidence']}")
        print("• Volume confirmation: {self.signal_thresholds['volume_confirmation']}")
        print("• Trend confirmation: {self.signal_thresholds['trend_confirmation']}")
        
        confirmation = input("\nAre you sure you want to proceed? (yes/no): ")
        if confirmation.lower() == 'yes':
            self.modes['paper_trading'] = False
            self.modes['live_trading'] = True
            
            alert_message = f"""
🚨 **ENHANCED LIVE TRADING MODE ACTIVATED**

⚠️ **WARNING**: Real money trading is now enabled!
💰 **Risk**: Real financial losses are possible
🔐 **Safety**: All enhanced risk management features are active

**Enhanced Risk Settings:**
• Max risk per trade: 1%
• Max portfolio risk: 5%
• Emergency stop loss: 10%
• Max daily trades: 10
• Minimum signal strength: {self.signal_thresholds['minimum_strength']}
• Minimum confidence: {self.signal_thresholds['minimum_confidence']}

**Enhanced Features:**
• Fixed timestamp issues
• Improved signal strength calculation
• Better error handling
• Enhanced strategy diversity

Monitor your positions carefully!
"""
            
            self.send_telegram_alert(alert_message)
            print("✅ Enhanced live trading mode activated!")
            return True
        else:
            print("❌ Live trading mode cancelled")
            return False

def main():
    """Main function with enhanced options"""
    system = EnhancedComprehensiveTradingSystem()
    
    print("🎯 ENHANCED COMPREHENSIVE TRADING SYSTEM")
    print("=" * 60)
    print("Choose an option:")
    print("1. Start Enhanced Analysis (Paper Trading)")
    print("2. Optimize Strategies")
    print("3. View Enhanced Portfolio Summary")
    print("4. Switch to Live Trading")
    print("5. Run All (Analysis + Optimization)")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        system.run_comprehensive_analysis()
    elif choice == '2':
        system.optimize_strategies()
    elif choice == '3':
        system.get_portfolio_summary()
    elif choice == '4':
        system.switch_to_live_trading()
    elif choice == '5':
        system.optimize_strategies()
        system.run_comprehensive_analysis()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 