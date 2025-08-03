"""
Enhanced Real Live Trading System
Integrated trading system with multiple strategies and detectors
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
import threading
from collections import deque
warnings.filterwarnings('ignore')

# Import our custom modules
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedLiveTradingSystem:
    """Enhanced Real Live Trading System with multiple strategies and detectors"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize enhanced live trading system
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            print("❌ API credentials not found!")
            print("Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in .env file")
            return
        
        # Initialize exchange
        self.exchange = self._setup_exchange()
        
        # Trading parameters
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
            'DOGEUSDT', 'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
            'AVAXUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'BCHUSDT'
        ]
        self.leverage = 100
        self.max_position_size = TRADING_CONFIG['max_position_size']
        self.stop_loss_pct = TRADING_CONFIG.get('stop_loss', 0.02)
        self.take_profit_pct = TRADING_CONFIG.get('take_profit', 0.04)
        self.risk_per_trade = TRADING_CONFIG['risk_per_trade']
        
        # Strategy and detector initialization
        self._initialize_strategies()
        self._initialize_detectors()
        
        # Data storage
        self.market_data = {}
        self.signal_history = []
        self.trade_history = []
        self.current_positions = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Trading state
        self.auto_trading = True
        self.enable_tp_sl = True
        self.running = False
        
        print("🚀 ENHANCED REAL LIVE TRADING SYSTEM INITIALIZED")
        print("=" * 70)
        print(f"💰 Trading with REAL MONEY")
        print(f"📊 Trading Pairs: {len(self.trading_pairs)} pairs")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"💰 Max Position Size: {self.max_position_size*100}%")
        print(f"🛡️ Stop Loss: {self.stop_loss_pct*100}%")
        print(f"🎯 Take Profit: {self.take_profit_pct*100}%")
        print(f"🤖 Auto-Trading: {'ENABLED' if self.auto_trading else 'DISABLED'}")
        print(f"📈 Strategies: {list(self.strategy_manager.strategies.keys())}")
        print(f"🔍 Detectors: Divergence, Support/Resistance")
        print("=" * 70)
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        self.strategy_manager = StrategyManager()
        
        # Add momentum strategy
        momentum_strategy = MomentumStrategy(STRATEGY_CONFIG['momentum'])
        self.strategy_manager.add_strategy(momentum_strategy)
        
        # Add mean reversion strategy
        mean_reversion_strategy = MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion'])
        self.strategy_manager.add_strategy(mean_reversion_strategy)
        
        # Add pairs trading strategy
        pairs_strategy = PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading'])
        self.strategy_manager.add_strategy(pairs_strategy)
        
        # Add divergence strategy
        divergence_strategy = DivergenceStrategy(STRATEGY_CONFIG['divergence'])
        self.strategy_manager.add_strategy(divergence_strategy)
        
        logger.info("All strategies initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize technical detectors"""
        # Initialize divergence detector
        self.divergence_detector = DivergenceDetector(
            rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
            macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
            macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
            macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
            min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
            swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
        )
        
        # Initialize support/resistance detector for main pairs
        self.support_resistance_detectors = {}
        for symbol in self.trading_pairs[:5]:  # Use top 5 pairs for S/R detection
            detector = LiveSupportResistanceDetector(
                symbol=symbol,
                timeframe='15m',
                enable_charts=False,
                enable_alerts=False
            )
            self.support_resistance_detectors[symbol] = detector
        
        logger.info("All detectors initialized successfully")
    
    def _setup_exchange(self):
        """Setup Binance exchange connection"""
        try:
            exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': False,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            
            # Load markets
            print("📡 Loading markets...")
            exchange.load_markets()
            
            # Test connection
            print("📡 Testing connection...")
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            print(f"✅ Connected to Binance Live")
            print(f"💰 USDT Balance: ${usdt_balance:.4f}")
            
            return exchange
            
        except Exception as e:
            print(f"❌ Error connecting to Binance: {e}")
            return None
    
    def fetch_market_data(self, symbol: str, timeframe: str = '15m', limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch and process market data for a symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the DataFrame"""
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
    
    def generate_comprehensive_signals(self, symbol: str) -> Dict:
        """Generate comprehensive trading signals using all strategies and detectors"""
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
                        'strength': latest_signal.get('signal_strength', 0.0),
                        'confidence': abs(latest_signal.get('signal', 0)) * 0.8
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
            if symbol in self.support_resistance_detectors:
                detector = self.support_resistance_detectors[symbol]
                current_price = df['close'].iloc[-1]
                
                # Get nearest support and resistance
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
            strategy_weight = 0.6
            for strategy_name, strategy_data in signals['strategies'].items():
                signal = strategy_data['signal']
                strength = strategy_data['strength']
                composite_signal += signal * strength * strategy_weight
                total_weight += strategy_weight
            
            # Divergence signals (weight: 0.3)
            if signals['divergence']:
                divergence_weight = 0.3
                divergence_signal = signals['divergence']['signal']
                divergence_strength = signals['divergence']['strength']
                composite_signal += divergence_signal * divergence_strength * divergence_weight
                total_weight += divergence_weight
            
            # Support/resistance signals (weight: 0.1)
            if signals['support_resistance']:
                sr_weight = 0.1
                support_distance = signals['support_resistance']['support_distance']
                resistance_distance = signals['support_resistance']['resistance_distance']
                
                if support_distance and support_distance < 0.02:  # Within 2% of support
                    composite_signal += 0.5 * sr_weight  # Bullish bias
                elif resistance_distance and resistance_distance < 0.02:  # Within 2% of resistance
                    composite_signal += -0.5 * sr_weight  # Bearish bias
                
                total_weight += sr_weight
            
            # Normalize composite signal
            if total_weight > 0:
                signals['composite_signal'] = composite_signal / total_weight
                signals['confidence'] = min(abs(composite_signal), 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, signal: float, price: float, confidence: float) -> float:
        """Calculate position size based on signal strength and confidence"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance <= 0:
                return 0
            
            # Base position size
            base_size = usdt_balance * self.risk_per_trade
            
            # Adjust for signal strength and confidence
            position_value = base_size * abs(signal) * confidence
            
            # Apply maximum position size limit
            max_position_value = usdt_balance * self.max_position_size
            position_value = min(position_value, max_position_value)
            
            # Calculate quantity
            quantity = position_value / price
            
            # Apply leverage
            leveraged_quantity = quantity * self.leverage
            
            return leveraged_quantity if signal > 0 else -leveraged_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def place_trade(self, symbol: str, side: str, quantity: float, price: float = None):
        """Place a trade order"""
        try:
            print(f"📊 Placing {side.upper()} order for {abs(quantity):.4f} {symbol} @ ${price or 'MARKET'}")
            
            order_params = {
                'symbol': symbol,
                'type': 'market' if price is None else 'limit',
                'side': side,
                'amount': abs(quantity)
            }
            
            if price:
                order_params['price'] = price
            
            # Set leverage
            try:
                self.exchange.set_leverage(self.leverage, symbol)
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")
            
            # Place the order
            order = self.exchange.create_order(**order_params)
            
            print(f"✅ Order placed successfully!")
            print(f"   Order ID: {order['id']}")
            print(f"   Status: {order['status']}")
            print(f"   Filled: {order.get('filled', 0)}")
            
            # Record the trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': abs(quantity),
                'price': order.get('price', price),
                'order_id': order['id'],
                'status': order['status']
            }
            
            self.trade_history.append(trade_record)
            self.performance_metrics['total_trades'] += 1
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None
    
    def execute_trading_strategy(self):
        """Execute the comprehensive trading strategy"""
        try:
            print(f"\n📊 EXECUTING ENHANCED TRADING STRATEGY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)
            
            # Get account info
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f"💰 USDT Balance: ${usdt_balance:.4f}")
            
            # Process each trading pair
            for symbol in self.trading_pairs:
                try:
                    # Generate comprehensive signals
                    signals = self.generate_comprehensive_signals(symbol)
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
                    if abs(composite_signal) > 0.3 and confidence > 0.5:  # Strong signal threshold
                        if composite_signal > 0:  # Buy signal
                            print(f"   🟢 EXECUTING BUY SIGNAL")
                            
                            if self.auto_trading:
                                quantity = self.calculate_position_size(composite_signal, current_price, confidence)
                                if quantity > 0:
                                    order = self.place_trade(symbol, 'buy', quantity)
                                    if order:
                                        # Place TP/SL orders
                                        self._place_tp_sl_orders(symbol, 'buy', current_price, abs(quantity))
                        else:  # Sell signal
                            print(f"   🔴 EXECUTING SELL SIGNAL")
                            
                            if self.auto_trading:
                                # Check for existing long position to close
                                positions = self.exchange.fetch_positions([symbol])
                                long_position = None
                                for position in positions:
                                    if position.get('size', 0) > 0 and position.get('side') == 'long':
                                        long_position = position
                                        break
                                
                                if long_position:
                                    # Close long position
                                    order = self.place_trade(symbol, 'sell', long_position['size'])
                                    if order:
                                        self.performance_metrics['winning_trades'] += 1
                                else:
                                    # Open short position
                                    quantity = self.calculate_position_size(abs(composite_signal), current_price, confidence)
                                    if quantity > 0:
                                        order = self.place_trade(symbol, 'sell', quantity)
                                        if order:
                                            # Place TP/SL orders
                                            self._place_tp_sl_orders(symbol, 'sell', current_price, quantity)
                    else:
                        print(f"   ⚪ No strong signal (Threshold: 0.3, Confidence: 0.5)")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Check and execute TP/SL orders
            self._check_tp_sl_orders()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Display performance summary
            self._display_performance_summary()
            
        except Exception as e:
            logger.error(f"Error executing trading strategy: {e}")
    
    def _place_tp_sl_orders(self, symbol: str, side: str, entry_price: float, quantity: float):
        """Place Take Profit and Stop Loss orders"""
        try:
            if not self.enable_tp_sl:
                return None, None
            
            # Calculate TP and SL prices
            if side == 'buy':  # Long position
                tp_price = entry_price * (1 + self.take_profit_pct)
                sl_price = entry_price * (1 - self.stop_loss_pct)
                tp_side = 'sell'
                sl_side = 'sell'
            else:  # Short position
                tp_price = entry_price * (1 - self.take_profit_pct)
                sl_price = entry_price * (1 + self.stop_loss_pct)
                tp_side = 'buy'
                sl_side = 'buy'
            
            # Place Take Profit order
            tp_order = None
            try:
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=tp_side,
                    amount=quantity,
                    price=tp_price,
                    params={'reduceOnly': True}
                )
                print(f"   🎯 TP Order placed: {tp_side.upper()} {quantity} {symbol} @ ${tp_price:.4f}")
            except Exception as e:
                logger.warning(f"Could not place TP order: {e}")
            
            # Place Stop Loss order
            sl_order = None
            try:
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=sl_side,
                    amount=quantity,
                    price=sl_price,
                    params={'stopPrice': sl_price, 'reduceOnly': True}
                )
                print(f"   🛡️ SL Order placed: {sl_side.upper()} {quantity} {symbol} @ ${sl_price:.4f}")
            except Exception as e:
                logger.warning(f"Could not place SL order: {e}")
            
            return tp_order, sl_order
            
        except Exception as e:
            logger.error(f"Error placing TP/SL orders: {e}")
            return None, None
    
    def _check_tp_sl_orders(self):
        """Check and execute Take Profit and Stop Loss orders"""
        try:
            positions = self.exchange.fetch_positions()
            
            for position in positions:
                if position.get('size', 0) > 0:
                    symbol = position['symbol']
                    side = position['side']
                    size = position['size']
                    entry_price = position.get('entryPrice', 0)
                    current_price = position.get('markPrice', 0)
                    
                    if entry_price == 0 or current_price == 0:
                        continue
                    
                    # Calculate P&L percentage
                    if side == 'long':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Check for TP/SL conditions
                    if pnl_pct >= self.take_profit_pct:
                        # Take Profit hit
                        close_side = 'sell' if side == 'long' else 'buy'
                        order = self.place_trade(symbol, close_side, size)
                        if order:
                            print(f"🎯 TP EXECUTED: {symbol} {side.upper()} position closed at {current_price:.4f}")
                            self.performance_metrics['winning_trades'] += 1
                    
                    elif pnl_pct <= -self.stop_loss_pct:
                        # Stop Loss hit
                        close_side = 'sell' if side == 'long' else 'buy'
                        order = self.place_trade(symbol, close_side, size)
                        if order:
                            print(f"🛡️ SL EXECUTED: {symbol} {side.upper()} position closed at {current_price:.4f}")
                    
                    # Display position status
                    if abs(pnl_pct) > 0.01:  # Show positions with >1% P&L
                        status = "🟢" if pnl_pct > 0 else "🔴"
                        print(f"   {status} {symbol} {side.upper()}: {pnl_pct*100:+.2f}% (Entry: ${entry_price:.4f}, Current: ${current_price:.4f})")
            
        except Exception as e:
            logger.error(f"Error checking TP/SL: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate total P&L
            positions = self.exchange.fetch_positions()
            total_pnl = 0
            
            for position in positions:
                if position.get('size', 0) > 0:
                    pnl = position.get('unrealizedPnl', 0)
                    total_pnl += pnl
            
            self.performance_metrics['total_pnl'] = total_pnl
            
            # Calculate win rate
            if self.performance_metrics['total_trades'] > 0:
                win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
                self.performance_metrics['win_rate'] = win_rate
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _display_performance_summary(self):
        """Display trading performance summary"""
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Winning Trades: {self.performance_metrics['winning_trades']}")
        
        if self.performance_metrics['total_trades'] > 0:
            win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            print(f"Win Rate: {win_rate*100:.1f}%")
        
        print(f"Total P&L: ${self.performance_metrics['total_pnl']:.4f}")
        
        # Get current account value
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f"Current Balance: ${usdt_balance:.4f}")
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
    
    def run_live_trading(self, duration_minutes: int = 60, update_interval: int = 30):
        """Run enhanced live trading for specified duration"""
        print(f"\n🚀 STARTING ENHANCED REAL LIVE TRADING")
        print("=" * 70)
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🔄 Update Interval: {update_interval} seconds")
        print(f"📈 Strategies: {list(self.strategy_manager.strategies.keys())}")
        print(f"🔍 Detectors: Divergence, Support/Resistance")
        print(f"🛑 Press Ctrl+C to stop early")
        print("=" * 70)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        self.running = True
        
        try:
            iteration = 0
            while datetime.now() < end_time and self.running:
                iteration += 1
                current_time = datetime.now()
                
                print(f"\n--- Enhanced Trading Session {iteration} ({current_time.strftime('%H:%M:%S')}) ---")
                
                # Execute trading strategy
                self.execute_trading_strategy()
                
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
            print(f"\n❌ Error in live trading: {e}")
        
        finally:
            self.running = False
            
            # Final performance summary
            print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
            print("=" * 70)
            self._display_performance_summary()
            
            # Save trade history
            if self.trade_history:
                filename = f"enhanced_trading_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump([{
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'quantity': trade['quantity'],
                        'price': trade['price'],
                        'order_id': trade['order_id'],
                        'status': trade['status']
                    } for trade in self.trade_history], f, indent=2)
                
                print(f"💾 Trade history saved to: {filename}")
            
            print(f"✅ Enhanced real live trading completed!")

def main():
    """Main function to run enhanced live trading"""
    print("🚀 ENHANCED REAL LIVE TRADING SYSTEM")
    print("=" * 70)
    
    # Initialize trading system
    trading_system = EnhancedLiveTradingSystem()
    
    if not trading_system.exchange:
        print("❌ Failed to initialize trading system")
        return
    
    # Confirm real trading
    print("\n⚠️  WARNING: This will use REAL MONEY!")
    print("💰 Your current balance will be used for trading")
    print("⚡ Trading with 100x leverage")
    print("📈 Using multiple strategies and detectors")
    
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
    
    # Start live trading
    trading_system.run_live_trading(duration_minutes=duration, update_interval=interval)

if __name__ == "__main__":
    main() 