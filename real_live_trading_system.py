"""
Real-Time Live Trading System
Integrated system combining multiple technical analysis modules for robust signal generation

Features:
- Live OHLCV data streaming from Binance
- Real-time support/resistance detection
- Dynamic Fibonacci retracement levels
- Divergence analysis (RSI/MACD)
- Multi-condition signal validation
- Optional backtesting before live trading
- Real-time alerts and notifications
- Performance tracking and visualization
"""

import os
import sys
import time
import json
import threading
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import requests
import sqlite3
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our custom modules
from backtest_system import BacktestSystem
from live_support_resistance_detector import LiveSupportResistanceDetector
from live_fibonacci_detector import LiveFibonacciDetector
from divergence_detector import DivergenceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    SupportResistanceStrategy,
    FibonacciStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG, TRADING_CONFIG

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_live_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy' or 'sell'
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    price: float
    conditions: List[str]  # List of met conditions
    support_resistance: Optional[Dict] = None
    fibonacci_level: Optional[Dict] = None
    divergence: Optional[Dict] = None
    strategy_signals: Optional[Dict] = None
    backtest_result: Optional[Dict] = None

@dataclass
class MarketCondition:
    """Data class for market conditions"""
    timestamp: datetime
    symbol: str
    price: float
    support_zones: List[Dict]
    resistance_zones: List[Dict]
    fibonacci_levels: List[Dict]
    divergence_signals: List[Dict]
    strategy_signals: Dict
    volume: float
    volatility: float

class RealLiveTradingSystem:
    """
    Comprehensive real-time trading system with integrated technical analysis
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None,
                 symbols: List[str] = None,
                 timeframe: str = '5m',
                 enable_live_trading: bool = False,
                 enable_backtesting: bool = True,
                 enable_alerts: bool = True,
                 telegram_bot_token: str = None,
                 telegram_chat_id: str = None,
                 database_path: str = 'trading_signals.db',
                 auto_analysis_interval: int = 300,  # 5 minutes
                 auto_export_interval: int = 3600,   # 1 hour
                 max_concurrent_trades: int = 5,
                 risk_per_trade: float = 0.02,       # 2% per trade
                 max_portfolio_risk: float = 0.10):  # 10% total portfolio
        """
        Initialize the real-time trading system
        """
        # API credentials
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY')
        
        # Trading configuration
        self.symbols = symbols or [
            'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'USDC/USDT',
            'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'XLM/USDT', 'SUI/USDT', 'LINK/USDT',
            'HBAR/USDT', 'AVAX/USDT', 'BCH/USDT', 'SHIB/USDT', 'LTC/USDT', 'TON/USDT',
            'DOT/USDT', 'UNI/USDT', 'XMR/USDT', 'PEPE/USDT', 'DAI/USDT', 'AAVE/USDT',
            'NEAR/USDT', 'ETC/USDT', 'APT/USDT', 'ONDO/USDT', 'ICP/USDT', 'BONK/USDT',
            'KAS/USDT', 'POL/USDT'
        ]
        self.timeframe = timeframe
        self.enable_live_trading = enable_live_trading
        self.enable_backtesting = enable_backtesting
        self.enable_alerts = enable_alerts
        
        # Automation settings
        self.auto_analysis_interval = auto_analysis_interval
        self.auto_export_interval = auto_export_interval
        self.max_concurrent_trades = max_concurrent_trades
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
        # Telegram configuration
        self.telegram_bot_token = telegram_bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        # Database setup
        self.database_path = database_path
        self._setup_database()
        
        # Exchange connection
        self.exchange = self._setup_exchange()
        
        # Data storage
        self.market_data = {}
        self.signal_history = []
        self.condition_history = []
        self.current_positions = {}
        self.analysis_history = []
        self.last_analysis_time = None
        self.last_export_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'executed_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Initialize backtest system
        if self.enable_backtesting:
            self.backtest_system = BacktestSystem(
                initial_capital=100000,
                commission=0.001
            )
        
        # Threading and async
        self.running = False
        self.data_thread = None
        self.analysis_thread = None
        self.export_thread = None
        
        # Signal validation settings
        self.min_conditions = 2  # Minimum conditions required for signal
        self.min_signal_strength = 0.6  # Minimum signal strength
        self.min_confidence = 0.7  # Minimum confidence level
        
        print("🚀 REAL-TIME LIVE TRADING SYSTEM INITIALIZED")
        print("=" * 70)
        print(f"📊 Symbols: {len(self.symbols)} pairs")
        print(f"⏱️ Timeframe: {self.timeframe}")
        print(f"💰 Live Trading: {'ENABLED' if self.enable_live_trading else 'DISABLED'}")
        print(f"📈 Backtesting: {'ENABLED' if self.enable_backtesting else 'DISABLED'}")
        print(f"🔔 Alerts: {'ENABLED' if self.enable_alerts else 'DISABLED'}")
        print(f"📱 Telegram: {'ENABLED' if self.telegram_bot_token else 'DISABLED'}")
        print(f"🤖 Auto Analysis: {'ENABLED' if self.auto_analysis_interval > 0 else 'DISABLED'}")
        print(f"📤 Auto Export: {'ENABLED' if self.auto_export_interval > 0 else 'DISABLED'}")
        print(f"⚙️ Risk Management: {self.risk_per_trade:.1%} per trade, {self.max_portfolio_risk:.1%} max portfolio")
        print("=" * 70)
    
    def _setup_database(self):
        """Setup SQLite database for signal storage"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    conditions TEXT NOT NULL,
                    support_resistance TEXT,
                    fibonacci_level TEXT,
                    divergence TEXT,
                    strategy_signals TEXT,
                    backtest_result TEXT,
                    executed BOOLEAN DEFAULT FALSE,
                    pnl REAL DEFAULT 0.0
                )
            ''')
            
            # Create market conditions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    support_zones TEXT,
                    resistance_zones TEXT,
                    fibonacci_levels TEXT,
                    divergence_signals TEXT,
                    strategy_signals TEXT,
                    volume REAL NOT NULL,
                    volatility REAL NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database setup complete: {self.database_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def _setup_exchange(self):
        """Setup exchange connection"""
        try:
            exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            
            # Test connection
            exchange.load_markets()
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            print(f"✅ Connected to Binance")
            print(f"💰 USDT Balance: ${usdt_balance:.4f}")
            
            return exchange
            
        except Exception as e:
            print(f"❌ Error connecting to exchange: {e}")
            return None
    
    def _initialize_detectors(self):
        """Initialize all technical detectors"""
        self.detectors = {}
        
        for symbol in self.symbols:
            symbol_detectors = {
                'support_resistance': LiveSupportResistanceDetector(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    enable_charts=False,
                    enable_alerts=False
                ),
                'fibonacci': LiveFibonacciDetector(
                    symbol=symbol,
                    timeframe=self.timeframe
                ),
                'divergence': DivergenceDetector(
                    rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
                    macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
                    macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
                    macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
                    min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
                    swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
                )
            }
            self.detectors[symbol] = symbol_detectors
        
        logger.info(f"Initialized detectors for {len(self.symbols)} symbols")
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        self.strategy_manager = StrategyManager()
        
        # Add strategies
        momentum_strategy = MomentumStrategy(STRATEGY_CONFIG['momentum'])
        mean_reversion_strategy = MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion'])
        pairs_strategy = PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading'])
        divergence_strategy = DivergenceStrategy(STRATEGY_CONFIG['divergence'])
        support_resistance_strategy = SupportResistanceStrategy(STRATEGY_CONFIG['support_resistance'])
        fibonacci_strategy = FibonacciStrategy(STRATEGY_CONFIG['fibonacci'])
        
        self.strategy_manager.add_strategy(momentum_strategy)
        self.strategy_manager.add_strategy(mean_reversion_strategy)
        self.strategy_manager.add_strategy(pairs_strategy)
        self.strategy_manager.add_strategy(divergence_strategy)
        self.strategy_manager.add_strategy(support_resistance_strategy)
        self.strategy_manager.add_strategy(fibonacci_strategy)
        
        logger.info("All strategies initialized successfully")
    
    def fetch_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch live OHLCV data for a symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=200
            )
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_market_conditions(self, symbol: str, data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions using all detectors"""
        try:
            current_price = data['close'].iloc[-1]
            volume = data['volume'].iloc[-1]
            volatility = data['close'].pct_change().std()
            
            # Get support/resistance zones
            support_zones = []
            resistance_zones = []
            if symbol in self.detectors:
                sr_detector = self.detectors[symbol]['support_resistance']
                support_zones, resistance_zones = sr_detector.identify_zones(data)
                
                # Convert to dict format
                support_zones = [
                    {
                        'level': zone.level,
                        'strength': zone.strength,
                        'touches': zone.touches,
                        'volume_confirmed': zone.volume_confirmed
                    }
                    for zone in support_zones
                ]
                
                resistance_zones = [
                    {
                        'level': zone.level,
                        'strength': zone.strength,
                        'touches': zone.touches,
                        'volume_confirmed': zone.volume_confirmed
                    }
                    for zone in resistance_zones
                ]
            
            # Get Fibonacci levels
            fibonacci_levels = []
            if symbol in self.detectors:
                fib_detector = self.detectors[symbol]['fibonacci']
                fib_detector.update_fibonacci_levels(data)
                fibonacci_levels = [
                    {
                        'level_type': level.level_type,
                        'percentage': level.percentage,
                        'price': level.price,
                        'strength': level.strength,
                        'touches': level.touches
                    }
                    for level in fib_detector.fibonacci_levels
                ]
            
            # Get divergence signals
            divergence_signals = []
            if symbol in self.detectors:
                div_detector = self.detectors[symbol]['divergence']
                divergence_analysis = div_detector.analyze_divergence(data)
                if 'signals' in divergence_analysis:
                    divergence_signals = divergence_analysis['signals']
            
            # Get strategy signals
            strategy_data = {symbol: data}
            strategy_signals = self.strategy_manager.get_all_signals(strategy_data)
            
            # Create market condition
            condition = MarketCondition(
                timestamp=datetime.now(),
                symbol=symbol,
                price=current_price,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                fibonacci_levels=fibonacci_levels,
                divergence_signals=divergence_signals,
                strategy_signals=strategy_signals,
                volume=volume,
                volatility=volatility
            )
            
            return condition
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions for {symbol}: {e}")
            return None
            
    def validate_signal_conditions(self, condition: MarketCondition) -> Tuple[bool, List[str], float, float]:
        """
        Validate if market conditions meet signal criteria
        
        Returns:
            Tuple of (is_valid, conditions_met, signal_strength, confidence)
        """
        conditions_met = []
        signal_strength = 0.0
        confidence = 0.0
        
        current_price = condition.price
        
        # Check support/resistance proximity
        for zone in condition.support_zones:
            distance = abs(current_price - zone['level']) / zone['level']
            if distance <= 0.02:  # Within 2% of support
                conditions_met.append(f"Near Support ${zone['level']:.4f}")
                signal_strength += zone['strength'] * 0.3
                confidence += 0.2
        
        for zone in condition.resistance_zones:
            distance = abs(current_price - zone['level']) / zone['level']
            if distance <= 0.02:  # Within 2% of resistance
                conditions_met.append(f"Near Resistance ${zone['level']:.4f}")
                signal_strength += zone['strength'] * 0.3
                confidence += 0.2
        
        # Check Fibonacci level proximity
        for level in condition.fibonacci_levels:
            distance = abs(current_price - level['price']) / level['price']
            if distance <= 0.01:  # Within 1% of Fibonacci level
                conditions_met.append(f"At {level['percentage']}% Fib Level")
                signal_strength += level['strength'] * 0.25
                confidence += 0.15
        
        # Check divergence signals
        for signal in condition.divergence_signals:
            conditions_met.append(f"{signal['type'].title()} {signal['indicator'].upper()} Divergence")
            signal_strength += signal['strength'] * 0.4
            confidence += 0.3
        
        # Check strategy signals
        for strategy_name, strategy_data in condition.strategy_signals.items():
            if symbol in strategy_data:
                signal_data = strategy_data[symbol]
                if len(signal_data) > 0:
                    latest_signal = signal_data.iloc[-1]
                    signal_value = latest_signal.get('signal', 0)
                    if abs(signal_value) > 0.3:
                        conditions_met.append(f"{strategy_name.title()} Signal")
                        signal_strength += abs(signal_value) * 0.2
                        confidence += 0.15
        
        # Normalize values
        signal_strength = min(signal_strength, 1.0)
        confidence = min(confidence, 1.0)
        
        # Check minimum conditions
        is_valid = (len(conditions_met) >= self.min_conditions and 
                   signal_strength >= self.min_signal_strength and 
                   confidence >= self.min_confidence)
        
        return is_valid, conditions_met, signal_strength, confidence
    
    def generate_trading_signal(self, condition: MarketCondition) -> Optional[TradingSignal]:
        """Generate trading signal based on market conditions"""
        try:
            is_valid, conditions_met, signal_strength, confidence = self.validate_signal_conditions(condition)
            
            if not is_valid:
                return None
            
            # Determine signal type based on conditions
            signal_type = 'buy'
            bullish_conditions = 0
            bearish_conditions = 0
            
            for condition_str in conditions_met:
                if any(word in condition_str.lower() for word in ['support', 'bullish', 'buy']):
                    bullish_conditions += 1
                elif any(word in condition_str.lower() for word in ['resistance', 'bearish', 'sell']):
                    bearish_conditions += 1
            
            if bearish_conditions > bullish_conditions:
                signal_type = 'sell'
            
            # Create trading signal
            signal = TradingSignal(
                timestamp=condition.timestamp,
                symbol=condition.symbol,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=confidence,
                price=condition.price,
                conditions=conditions_met,
                support_resistance={
                    'support_zones': condition.support_zones,
                    'resistance_zones': condition.resistance_zones
                },
                fibonacci_level=condition.fibonacci_levels,
                divergence=condition.divergence_signals,
                strategy_signals=condition.strategy_signals
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def backtest_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """Backtest the signal using historical data"""
        if not self.enable_backtesting:
            return None
        
        try:
            # Load historical data for backtesting
            symbol = signal.symbol
            end_date = signal.timestamp.strftime('%Y-%m-%d')
            start_date = (signal.timestamp - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Initialize backtest system
            backtest_system = BacktestSystem(
                initial_capital=10000,  # Small capital for signal testing
                commission=0.001
            )
            
            # Load historical data
            if backtest_system.load_historical_data(start_date=start_date, end_date=end_date):
                # Run focused backtest for this signal
                backtest_result = self._run_signal_backtest(backtest_system, signal)
                return backtest_result
            else:
                logger.warning(f"Could not load historical data for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error backtesting signal: {e}")
            return None
    
    def _run_signal_backtest(self, backtest_system: BacktestSystem, signal: TradingSignal) -> Dict:
        """Run focused backtest for a specific signal"""
        try:
            # Get historical data for the signal period
            symbol = signal.symbol
            if symbol not in backtest_system.historical_data:
                return {'error': 'No historical data available'}
            
            data = backtest_system.historical_data[symbol]
            
            # Find signal date in historical data
            signal_date = signal.timestamp
            if signal_date not in data.index:
                # Find closest date
                closest_date = data.index[data.index.get_indexer([signal_date], method='nearest')[0]]
                signal_date = closest_date
            
            # Get data up to signal date
            historical_data = data[data.index <= signal_date]
            
            if len(historical_data) < 50:
                return {'error': 'Insufficient historical data'}
            
            # Generate signals for the historical period
            signals = backtest_system.generate_backtest_signals(symbol, signal_date)
            
            if not signals:
                return {'error': 'Could not generate signals'}
            
            # Calculate performance metrics
            performance_metrics = self._calculate_signal_performance(
                historical_data, signals, signal
            )
            
            return {
                'signal_type': signal.signal_type,
                'entry_price': signal.price,
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'backtest_period': f"{historical_data.index[0].strftime('%Y-%m-%d')} to {signal_date.strftime('%Y-%m-%d')}",
                'historical_performance': performance_metrics,
                'data_points': len(historical_data),
                'signal_conditions': signal.conditions
            }
            
        except Exception as e:
            logger.error(f"Error in signal backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_signal_performance(self, data: pd.DataFrame, signals: Dict, original_signal: TradingSignal) -> Dict:
        """Calculate performance metrics for signal validation"""
        try:
            # Calculate basic metrics
            returns = data['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate signal accuracy (if we have multiple signals)
            signal_accuracy = 0.0
            if 'composite_signal' in signals:
                # Simple accuracy calculation based on signal direction vs price movement
                signal_direction = np.sign(signals['composite_signal'])
                price_movement = np.sign(data['close'].diff().shift(-1))  # Next period's movement
                
                # Calculate accuracy for non-zero signals
                valid_signals = (signal_direction != 0) & (price_movement != 0)
                if valid_signals.sum() > 0:
                    accuracy = (signal_direction[valid_signals] == price_movement[valid_signals]).mean()
                    signal_accuracy = accuracy
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': (data['close'].iloc[-1] / data['close'].iloc[0]) - 1,
                'signal_accuracy': signal_accuracy,
                'price_range': {
                    'min': data['close'].min(),
                    'max': data['close'].max(),
                    'current': data['close'].iloc[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal performance: {e}")
            return {'error': str(e)}
    
    def execute_trade(self, signal: TradingSignal):
        """Execute trade based on signal"""
        if not self.enable_live_trading:
            logger.info(f"Live trading disabled - would execute {signal.signal_type.upper()} for {signal.symbol}")
            return
        
        try:
            # Calculate position size
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance <= 0:
                logger.warning("Insufficient balance for trading")
                return
            
            # Simple position sizing (1% of balance)
            position_value = usdt_balance * 0.01 * signal.strength
            quantity = position_value / signal.price
            
            # Place order
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type='market',
                side=signal.signal_type,
                amount=quantity
            )
            
            logger.info(f"Trade executed: {signal.signal_type.upper()} {quantity:.4f} {signal.symbol} @ ${signal.price:.4f}")
            
            # Update performance metrics
            self.performance_metrics['executed_trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def save_signal_to_database(self, signal: TradingSignal):
        """Save signal to SQLite database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_signals (
                    timestamp, symbol, signal_type, strength, confidence, price,
                    conditions, support_resistance, fibonacci_level, divergence,
                    strategy_signals, backtest_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.strength,
                signal.confidence,
                signal.price,
                json.dumps(signal.conditions),
                json.dumps(signal.support_resistance) if signal.support_resistance else None,
                json.dumps(signal.fibonacci_level) if signal.fibonacci_level else None,
                json.dumps(signal.divergence) if signal.divergence else None,
                json.dumps(signal.strategy_signals) if signal.strategy_signals else None,
                json.dumps(signal.backtest_result) if signal.backtest_result else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")
    
    def save_market_condition_to_database(self, condition: MarketCondition):
        """Save market condition to SQLite database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_conditions (
                    timestamp, symbol, price, support_zones, resistance_zones,
                    fibonacci_levels, divergence_signals, strategy_signals,
                    volume, volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                condition.timestamp.isoformat(),
                condition.symbol,
                condition.price,
                json.dumps(condition.support_zones),
                json.dumps(condition.resistance_zones),
                json.dumps(condition.fibonacci_levels),
                json.dumps(condition.divergence_signals),
                json.dumps(condition.strategy_signals),
                condition.volume,
                condition.volatility
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving market condition to database: {e}")
    
    def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def send_signal_alert(self, signal: TradingSignal):
        """Send signal alert"""
        if not self.enable_alerts:
            return
        
        timestamp = signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"""
🚨 <b>TRADING SIGNAL ALERT</b> 🚨

Symbol: {signal.symbol}
Signal: {signal.signal_type.upper()}
Price: ${signal.price:.4f}
Strength: {signal.strength:.2f}
Confidence: {signal.confidence:.2f}
Time: {timestamp}

<b>Conditions Met:</b>
{chr(10).join(f"• {condition}" for condition in signal.conditions)}

<b>Support/Resistance:</b>
• Support Zones: {len(signal.support_resistance['support_zones']) if signal.support_resistance else 0}
• Resistance Zones: {len(signal.support_resistance['resistance_zones']) if signal.support_resistance else 0}

<b>Fibonacci Levels:</b>
• Active Levels: {len(signal.fibonacci_level) if signal.fibonacci_level else 0}

<b>Divergence:</b>
• Signals: {len(signal.divergence) if signal.divergence else 0}
        """
        
        # Send to console
        print(f"\n{message}")
        
        # Send to Telegram
        self.send_telegram_alert(message)
    
    def automated_analysis_loop(self):
        """Automated market analysis loop"""
        while self.running:
            try:
                # Run comprehensive market analysis
                analysis_result = self.run_automated_market_analysis()
                
                if analysis_result:
                    # Print analysis summary
                    self.print_analysis_summary(analysis_result)
                
                # Sleep for analysis interval
                time.sleep(self.auto_analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in automated analysis loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def automated_export_loop(self):
        """Automated export loop"""
        while self.running:
            try:
                # Export results periodically
                self.export_results()
                
                # Sleep for export interval
                time.sleep(self.auto_export_interval)
                
            except Exception as e:
                logger.error(f"Error in automated export loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def data_processing_loop(self):
        """Main data processing loop with unified signal evaluation"""
        while self.running:
            try:
                # Collect live data for all symbols
                live_data = {
                    'price_data': {},
                    'support_resistance': {},
                    'fibonacci': {},
                    'divergence': {}
                }
                
                for symbol in self.symbols:
                    # Fetch live data
                    data = self.fetch_live_data(symbol)
                    if data is None:
                        continue
                    
                    live_data['price_data'][symbol] = data
                    
                    # Get detection module outputs
                    if symbol in self.detectors:
                        # Support/Resistance zones
                        sr_detector = self.detectors[symbol]['support_resistance']
                        support_zones, resistance_zones = sr_detector.identify_zones(data)
                        live_data['support_resistance'][symbol] = {
                            'support_zones': support_zones,
                            'resistance_zones': resistance_zones
                        }
                        
                        # Fibonacci levels
                        fib_detector = self.detectors[symbol]['fibonacci']
                        fib_detector.update_fibonacci_levels(data)
                        live_data['fibonacci'][symbol] = fib_detector.fibonacci_levels
                        
                        # Divergence analysis
                        div_detector = self.detectors[symbol]['divergence']
                        divergence_analysis = div_detector.analyze_divergence(data)
                        live_data['divergence'][symbol] = divergence_analysis
                
                # Evaluate unified signals using StrategyManager
                if live_data['price_data']:
                    unified_signals = self.strategy_manager.evaluate_trade_signal(live_data)
                    
                    # Process signals
                    for symbol, signal_result in unified_signals.items():
                        if signal_result['recommended_action'] != 'hold':
                            # Create trading signal
                            signal = TradingSignal(
                                timestamp=signal_result['timestamp'],
                    symbol=symbol,
                                signal_type=signal_result['recommended_action'],
                                strength=abs(signal_result['signal']),
                                confidence=signal_result['confidence'],
                                price=signal_result['current_price'],
                                conditions=signal_result['reasoning'],
                                support_resistance=live_data['support_resistance'].get(symbol),
                                fibonacci_level=live_data['fibonacci'].get(symbol),
                                divergence=live_data['divergence'].get(symbol),
                                strategy_signals=signal_result['strategy_contributions']
                            )
                            
                            # Backtest signal if enabled
                            if self.enable_backtesting:
                                backtest_result = self.backtest_signal(signal)
                                signal.backtest_result = backtest_result
                            
                            # Store signal
                            self.signal_history.append(signal)
                            self.save_signal_to_database(signal)
                            
                            # Send alerts
                            self.send_signal_alert(signal)
                            
                            # Execute trade if confidence is high enough
                            if signal.confidence >= 0.7:
                                self.execute_trade(signal)
                            
                            # Update performance metrics
                            self.performance_metrics['total_signals'] += 1
                
                # Sleep based on timeframe
                if self.timeframe == '1m':
                    time.sleep(30)
                elif self.timeframe == '5m':
                    time.sleep(60)
                else:
                    time.sleep(120)
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                time.sleep(10)
    
    def print_status(self):
        """Print system status"""
        print(f"\n{'='*70}")
        print(f"📊 REAL-TIME TRADING SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 Running: {'Yes' if self.running else 'No'}")
        print(f"📈 Symbols: {len(self.symbols)}")
        print(f"⏱️ Timeframe: {self.timeframe}")
        
        print(f"\n🤖 AUTOMATION STATUS:")
        print(f"• Analysis Interval: {self.auto_analysis_interval}s")
        print(f"• Export Interval: {self.auto_export_interval}s")
        print(f"• Max Concurrent Trades: {self.max_concurrent_trades}")
        print(f"• Risk Per Trade: {self.risk_per_trade:.1%}")
        print(f"• Max Portfolio Risk: {self.max_portfolio_risk:.1%}")
        print(f"• Last Analysis: {self.last_analysis_time.strftime('%H:%M:%S') if self.last_analysis_time else 'Never'}")
        print(f"• Last Export: {self.last_export_time.strftime('%H:%M:%S') if self.last_export_time else 'Never'}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"• Total Signals: {self.performance_metrics['total_signals']}")
        print(f"• Executed Trades: {self.performance_metrics['executed_trades']}")
        print(f"• Winning Trades: {self.performance_metrics['winning_trades']}")
        print(f"• Total P&L: ${self.performance_metrics['total_pnl']:.2f}")
        print(f"• Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
        
        print(f"\n📋 RECENT SIGNALS:")
        for signal in self.signal_history[-5:]:
            print(f"• {signal.timestamp.strftime('%H:%M:%S')} | {signal.symbol} | {signal.signal_type.upper()} | ${signal.price:.4f} | S:{signal.strength:.2f}")
        
        print(f"{'='*70}")
    
    def print_analysis_summary(self, analysis_result: Dict):
        """Print automated analysis summary"""
        print(f"\n{'='*70}")
        print(f"🤖 AUTOMATED MARKET ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"⏰ Time: {analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Symbols Analyzed: {analysis_result['symbols_analyzed']}")
        print(f"🎯 Total Signals: {analysis_result['total_signals']}")
        print(f"🟢 Buy Signals: {analysis_result['buy_signals']}")
        print(f"🔴 Sell Signals: {analysis_result['sell_signals']}")
        
        if analysis_result['total_signals'] > 0:
            print(f"\n📈 SIGNAL BREAKDOWN:")
            for symbol, data in analysis_result['market_conditions'].items():
                if data['signal_generated']:
                    signal_icon = "🟢" if data['signal_type'] == 'buy' else "🔴"
                    print(f"  {signal_icon} {symbol}: {data['signal_type'].upper()} | "
                          f"Strength: {data['signal_strength']:.3f} | "
                          f"Confidence: {data['signal_confidence']:.3f}")
        
        print(f"\n📊 MARKET CONDITIONS:")
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for symbol, data in analysis_result['market_conditions'].items():
            if data['signal_type'] == 'buy':
                bullish_count += 1
            elif data['signal_type'] == 'sell':
                bearish_count += 1
            else:
                neutral_count += 1
        
        print(f"  🟢 Bullish: {bullish_count} symbols")
        print(f"  🔴 Bearish: {bearish_count} symbols")
        print(f"  🟡 Neutral: {neutral_count} symbols")
        
        print(f"{'='*70}")
    
    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return
        
        self.running = True
        
        # Start data processing thread
        self.data_thread = threading.Thread(target=self.data_processing_loop, daemon=True)
        self.data_thread.start()
        
        # Start automated analysis thread
        self.analysis_thread = threading.Thread(target=self.automated_analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        # Start automated export thread
        self.export_thread = threading.Thread(target=self.automated_export_loop, daemon=True)
        self.export_thread.start()
        
        logger.info("Real-time trading system started with automated analysis")
        
        # Main loop for status updates
        try:
            while self.running:
                time.sleep(30)  # Print status every 30 seconds
                self.print_status()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        logger.info("Trading system stopped")
        
        # Export final results
        self.export_results()
    
    def run_automated_market_analysis(self):
        """Run comprehensive market analysis for all symbols"""
        try:
            logger.info("Starting automated market analysis...")
            
            analysis_results = {}
            total_signals = 0
            buy_signals = 0
            sell_signals = 0
            
            for symbol in self.symbols:
                try:
                    # Fetch current data
                    data = self.fetch_live_data(symbol)
                    if data is None or data.empty:
                        continue
                    
                    # Analyze market conditions
                    condition = self.analyze_market_conditions(symbol, data)
                    if condition is None:
                        continue
                    
                    # Generate trading signal
                    signal = self.generate_trading_signal(condition)
                    
                    # Store analysis results
                    analysis_results[symbol] = {
                        'current_price': condition.price,
                        'volume': condition.volume,
                        'volatility': condition.volatility,
                        'support_zones': len(condition.support_zones),
                        'resistance_zones': len(condition.resistance_zones),
                        'fibonacci_levels': len(condition.fibonacci_levels),
                        'divergence_signals': len(condition.divergence_signals),
                        'signal_generated': signal is not None,
                        'signal_type': signal.signal_type if signal else None,
                        'signal_strength': signal.strength if signal else 0,
                        'signal_confidence': signal.confidence if signal else 0,
                        'conditions_met': signal.conditions if signal else []
                    }
                    
                    # Count signals
                    if signal:
                        total_signals += 1
                        if signal.signal_type == 'buy':
                            buy_signals += 1
                        elif signal.signal_type == 'sell':
                            sell_signals += 1
                        
                        # Store signal
                        self.signal_history.append(signal)
                        self.save_signal_to_database(signal)
                        
                        # Send alerts for high-confidence signals
                        if signal.confidence >= 0.7:
                            self.send_signal_alert(signal)
                        
                        # Execute trade if live trading is enabled
                        if self.enable_live_trading and signal.confidence >= 0.8:
                            self.execute_trade(signal)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Store analysis results
            analysis_summary = {
                'timestamp': datetime.now(),
                'symbols_analyzed': len(analysis_results),
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'market_conditions': analysis_results
            }
            
            self.analysis_history.append(analysis_summary)
            self.last_analysis_time = datetime.now()
            
            # Log analysis summary
            logger.info(f"Market analysis completed: {total_signals} signals generated ({buy_signals} buy, {sell_signals} sell)")
            
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error in automated market analysis: {e}")
            return None
    
    def export_results(self, filename: str = None):
        """Export trading results"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trading_results_{timestamp}.json'
        
        results = {
            'system_info': {
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'start_time': self.signal_history[0].timestamp.isoformat() if self.signal_history else None,
                'end_time': self.signal_history[-1].timestamp.isoformat() if self.signal_history else None,
                'enable_live_trading': self.enable_live_trading,
                'enable_backtesting': self.enable_backtesting
            },
            'performance_metrics': self.performance_metrics,
            'signals': [
                {
                    'timestamp': signal.timestamp.isoformat(),
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'conditions': signal.conditions
                }
                for signal in self.signal_history
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

def main():
    """Main function to run the trading system"""
    print("🚀 REAL-TIME LIVE TRADING SYSTEM")
    print("=" * 70)
    
    # Configuration
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'USDC/USDT',
        'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'XLM/USDT', 'SUI/USDT', 'LINK/USDT',
        'HBAR/USDT', 'AVAX/USDT', 'BCH/USDT', 'SHIB/USDT', 'LTC/USDT', 'TON/USDT',
        'DOT/USDT', 'UNI/USDT', 'XMR/USDT', 'PEPE/USDT', 'DAI/USDT', 'AAVE/USDT',
        'NEAR/USDT', 'ETC/USDT', 'APT/USDT', 'ONDO/USDT', 'ICP/USDT', 'BONK/USDT',
        'KAS/USDT', 'POL/USDT'
    ]
    timeframe = '5m'
    enable_live_trading = False  # Set to True for live trading
    enable_backtesting = True
    enable_alerts = True
    
    # Automation settings
    auto_analysis_interval = 300  # 5 minutes
    auto_export_interval = 3600   # 1 hour
    max_concurrent_trades = 5
    risk_per_trade = 0.02         # 2% per trade
    max_portfolio_risk = 0.10     # 10% total portfolio
    
    # Telegram setup (optional)
    telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Create trading system
    trading_system = RealLiveTradingSystem(
        symbols=symbols,
        timeframe=timeframe,
        enable_live_trading=enable_live_trading,
        enable_backtesting=enable_backtesting,
        enable_alerts=enable_alerts,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        auto_analysis_interval=auto_analysis_interval,
        auto_export_interval=auto_export_interval,
        max_concurrent_trades=max_concurrent_trades,
        risk_per_trade=risk_per_trade,
        max_portfolio_risk=max_portfolio_risk
    )
    
    try:
        # Start the system
        trading_system.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping trading system...")
        trading_system.stop()
        
        # Export results
        export = input("\nExport results? (y/n): ").strip().lower() == 'y'
        if export:
            trading_system.export_results()

if __name__ == "__main__":
    main() 
