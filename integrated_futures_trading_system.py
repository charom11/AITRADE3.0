#!/usr/bin/env python3
"""
Integrated Futures Trading System
Combines Binance Futures signals with support/resistance detection, divergence analysis, and strategy management
"""

import os
import time
import json
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import warnings
import ccxt
import hmac
import hashlib
import requests
from urllib.parse import urlencode
import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

# Import existing components
from binance_futures_signals import BinanceFuturesAPIClient, FuturesMarketData, FuturesSignalGenerator
from live_support_resistance_detector import LiveSupportResistanceDetector
from divergence_detector import DivergenceDetector
from strategies import (
    StrategyManager, MomentumStrategy, MeanReversionStrategy, 
    DivergenceStrategy, SupportResistanceStrategy, PairsTradingStrategy
)
from file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_futures_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedSignal:
    """Integrated trading signal combining multiple analysis methods"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0-1 scale
    price: float
    stop_loss: float
    take_profit: float
    
    # Component signals
    futures_signal: Optional[Dict] = None
    support_resistance_signal: Optional[Dict] = None
    divergence_signal: Optional[Dict] = None
    strategy_signals: Optional[Dict] = None
    
    # Risk metrics
    risk_score: float = 0.0
    position_size: float = 0.0
    leverage_suggestion: float = 1.0
    
    # Market context
    funding_rate: float = 0.0
    open_interest: float = 0.0
    volume_ratio: float = 1.0
    market_regime: str = "unknown"
    
    # Trading execution
    executed: bool = False
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    trade_status: str = "PENDING"  # PENDING, EXECUTED, FAILED, CANCELLED

@dataclass
class TradePosition:
    """Trade position data"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    order_id: str
    timestamp: datetime
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    strategy: str = ""
    confidence: float = 0.0
    risk_score: float = 0.0

@dataclass
class MarketCondition:
    """Market condition data"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    volatility: float
    trend: str  # 'bullish', 'bearish', 'sideways'
    rsi: float
    macd: float
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    divergence_signals: List[Dict] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    average_trade_duration: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

class IntegratedFuturesTradingSystem:
    """
    Integrated futures trading system combining multiple analysis methods with live trading execution
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 api_key: str = None,
                 api_secret: str = None,
                 enable_file_output: bool = True,
                 output_directory: str = "integrated_signals",
                 enable_live_trading: bool = False,
                 paper_trading: bool = True,
                 max_position_size: float = 0.1,
                 risk_per_trade: float = 0.02,
                 enable_backtesting: bool = True,
                 enable_alerts: bool = True,
                 telegram_bot_token: str = None,
                 telegram_chat_id: str = None,
                 database_path: str = 'trading_signals.db',
                 auto_analysis_interval: int = 300,
                 auto_export_interval: int = 3600,
                 max_concurrent_trades: int = 5,
                 max_portfolio_risk: float = 0.10):
        """
        Initialize the integrated system
        
        Args:
            symbols: List of trading symbols
            api_key: Binance API key
            api_secret: Binance API secret
            enable_file_output: Whether to save signals to files
            output_directory: Directory for output files
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.api_key = api_key
        self.api_secret = api_secret
        self.enable_file_output = enable_file_output
        self.output_directory = output_directory
        self.enable_live_trading = enable_live_trading
        self.paper_trading = paper_trading
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.enable_backtesting = enable_backtesting
        self.enable_alerts = enable_alerts
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.database_path = database_path
        self.auto_analysis_interval = auto_analysis_interval
        self.auto_export_interval = auto_export_interval
        self.max_concurrent_trades = max_concurrent_trades
        self.max_portfolio_risk = max_portfolio_risk
        
        # Trading state
        self.open_positions = {}
        self.trade_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.performance_metrics = PerformanceMetrics()
        self.market_conditions = {}
        
        # Database
        self._setup_database()
        
        # Initialize file manager
        self.file_manager = FileManager(output_directory)
        
        # Create output directory if it doesn't exist
        if self.enable_file_output:
            self.file_manager.safe_create_directory(output_directory)
        
        # Initialize components
        self.futures_client = BinanceFuturesAPIClient(api_key, api_secret)
        self.futures_signal_generator = FuturesSignalGenerator()
        
        # Initialize CCXT exchange for live trading
        if self.enable_live_trading and self.api_key and self.api_secret:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            logger.info("Live trading enabled with CCXT")
        else:
            self.exchange = None
            logger.info("Paper trading mode enabled")
        
        # Initialize detectors
        self.support_resistance_detectors = {}
        self.divergence_detectors = {}
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        self._initialize_strategies()
        
        # Data storage
        self.historical_data = {symbol: [] for symbol in symbols}
        self.current_signals = {symbol: None for symbol in symbols}
        self.signal_history = {symbol: [] for symbol in symbols}
        
        # System state
        self.running = False
        self.last_update = None
        
        logger.info(f"Initialized integrated futures trading system for {len(symbols)} symbols")
        logger.info(f"Trading mode: {'Live Trading' if self.enable_live_trading else 'Paper Trading'}")
        logger.info(f"Backtesting: {'Enabled' if self.enable_backtesting else 'Disabled'}")
        logger.info(f"Alerts: {'Enabled' if self.enable_alerts else 'Disabled'}")
    
    def _setup_database(self):
        """Setup SQLite database for storing trading data"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    strategy TEXT,
                    executed BOOLEAN,
                    order_id TEXT,
                    execution_price REAL,
                    execution_time TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    volatility REAL,
                    trend TEXT,
                    rsi REAL,
                    macd REAL,
                    support_levels TEXT,
                    resistance_levels TEXT,
    
                    divergence_signals TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    strategy TEXT,
                    duration REAL,
                    order_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        try:
            if self.telegram_bot_token and self.telegram_chat_id:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                data = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Telegram alert sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
        return False
    
    def test_telegram_connection(self):
        """Test Telegram bot connection"""
        try:
            if self.telegram_bot_token and self.telegram_chat_id:
                test_message = f"""
🚨 **AITRADE Bot Test**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: Connection Test
💪 **Enhanced Analysis**: ✅ Active

✅ **Bot connection successful!**
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}
🔧 **Status**: Ready for enhanced trading signals

You will now receive enhanced trading alerts and notifications.
"""
                success = self.send_telegram_alert(test_message)
                if success:
                    print("✅ Telegram connection test successful!")
                    return True
                else:
                    print("❌ Telegram connection test failed!")
                    return False
            else:
                print("⚠️ Telegram credentials not configured")
                return False
        except Exception as e:
            print(f"❌ Telegram test error: {e}")
            return False
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        # Add strategies to manager
        momentum_config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sma_short': 20,
            'sma_medium': 50,
            'sma_long': 200,
            'atr_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_multiplier': 1.5,
            'min_trend_strength': 0.3,
            'position_size_pct': 0.1
        }
        
        mean_reversion_config = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'atr_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'max_deviation': 2.0,
            'min_bounce_pct': 0.005,
            'position_size_pct': 0.08
        }
        
        divergence_config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'min_candles': 50,
            'swing_threshold': 0.02,
            'confirm_with_support_resistance': True,
            'position_size_pct': 0.12
        }
        
        support_resistance_config = {
            'window_size': 200,
            'min_touches': 2,
            'zone_buffer': 0.003,
            'swing_sensitivity': 0.02,
            'volume_threshold': 1.5,
            'enable_charts': False,
            'enable_alerts': False,
            'confirm_with_candlestick': True,
            'position_size_pct': 0.15
        }
        
        self.strategy_manager.add_strategy(MomentumStrategy(momentum_config))
        self.strategy_manager.add_strategy(MeanReversionStrategy(mean_reversion_config))
        self.strategy_manager.add_strategy(DivergenceStrategy(divergence_config))
        self.strategy_manager.add_strategy(SupportResistanceStrategy(support_resistance_config))
        
        # Add additional strategies
        pairs_config = {
            'lookback_period': 252,
            'correlation_threshold': 0.7,
            'z_score_threshold': 2.0,
            'position_size_pct': 0.08
        }
        
        self.strategy_manager.add_strategy(PairsTradingStrategy(pairs_config))
        
        logger.info("Initialized trading strategies")
    
    def _get_or_create_detector(self, symbol: str, detector_type: str):
        """Get existing detector or create new one if it doesn't exist"""
        if detector_type == "support_resistance":
            if symbol not in self.support_resistance_detectors:
                self.support_resistance_detectors[symbol] = LiveSupportResistanceDetector(
                    symbol=symbol.replace('USDT', '/USDT'),
                    timeframe='5m',
                    enable_charts=False,
                    enable_alerts=False
                )
            return self.support_resistance_detectors[symbol]
        
        elif detector_type == "divergence":
            if symbol not in self.divergence_detectors:
                self.divergence_detectors[symbol] = DivergenceDetector(
                    rsi_period=14,
                    macd_fast=12,
                    macd_slow=26,
                    min_candles=50
                )
            return self.divergence_detectors[symbol]
        
        return None
    
    def _fetch_futures_data(self, symbol: str) -> Optional[FuturesMarketData]:
        """Fetch current futures data for a symbol"""
        try:
            # Get klines data
            klines = self.futures_client.get_klines(symbol, interval="5m", limit=1)
            if not klines:
                return None
            
            # Get funding rate
            try:
                funding_data = self.futures_client.get_funding_rate(symbol)
                funding_rate = float(funding_data[0]['fundingRate']) if funding_data else 0.0
            except:
                funding_rate = 0.0
            
            # Get open interest
            try:
                oi_data = self.futures_client.get_open_interest(symbol)
                open_interest = float(oi_data['openInterest']) if oi_data else 0.0
            except:
                open_interest = 0.0
            
            # Create market data object
            market_data = FuturesMarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(klines[0][0]/1000),
                open=float(klines[0][1]),
                high=float(klines[0][2]),
                low=float(klines[0][3]),
                close=float(klines[0][4]),
                volume=float(klines[0][5]),
                funding_rate=funding_rate,
                open_interest=open_interest,
                source="binance_futures_api"
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching futures data for {symbol}: {e}")
            return None
    
    def _fetch_historical_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical data for analysis"""
        try:
            klines = self.futures_client.get_klines(symbol, interval="5m", limit=limit)
            if not klines:
                return None
            
            # Convert to DataFrame
            data = []
            for kline in klines:
                data.append({
                    'timestamp': datetime.fromtimestamp(kline[0]/1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _generate_futures_signal(self, symbol: str, market_data: FuturesMarketData) -> Optional[Dict]:
        """Generate futures signal using existing generator"""
        try:
            if len(self.historical_data[symbol]) >= 30:
                signal = self.futures_signal_generator.generate_futures_signal(
                    market_data, self.historical_data[symbol]
                )
                return signal
        except Exception as e:
            logger.error(f"Error generating futures signal for {symbol}: {e}")
        return None
    
    def _generate_support_resistance_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Generate support/resistance signal"""
        try:
            detector = self._get_or_create_detector(symbol, "support_resistance")
            
            # Identify zones
            support_zones, resistance_zones = detector.identify_zones(data)
            
            current_price = data['close'].iloc[-1]
            
            # Check for zone breaks and approaches
            breaks = detector.check_zone_breaks(current_price)
            approaches = detector.check_zone_approaches(current_price)
            
            if breaks or approaches:
                return {
                    'type': 'support_resistance',
                    'breaks': breaks,
                    'approaches': approaches,
                    'support_zones': support_zones,
                    'resistance_zones': resistance_zones,
                    'current_price': current_price
                }
                
        except Exception as e:
            logger.error(f"Error generating support/resistance signal for {symbol}: {e}")
        return None
    
    def _generate_divergence_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Generate divergence signal"""
        try:
            detector = self._get_or_create_detector(symbol, "divergence")
            
            # Analyze divergence
            analysis = detector.analyze_divergence(data, confirm_signals=True)
            
            if analysis.get('total_signals', 0) > 0:
                return {
                    'type': 'divergence',
                    'analysis': analysis,
                    'signals': analysis.get('signals', [])
                }
                
        except Exception as e:
            logger.error(f"Error generating divergence signal for {symbol}: {e}")
        return None
    
    def _generate_strategy_signals(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Generate strategy signals"""
        try:
            # Prepare data for strategy manager
            strategy_data = {symbol: data}
            
            # Get all strategy signals
            all_signals = self.strategy_manager.get_all_signals(strategy_data)
            
            if all_signals and symbol in all_signals:
                return {
                    'type': 'strategy',
                    'signals': all_signals[symbol],
                    'summary': self.strategy_manager.get_strategy_summary()
                }
                
        except Exception as e:
            logger.error(f"Error generating strategy signals for {symbol}: {e}")
        return None
    
    def analyze_market_conditions(self, symbol: str, data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            current_price = data['close'].iloc[-1]
            volume = data['volume'].iloc[-1]
            
            # Calculate volatility
            returns = data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate RSI
            rsi = self._calculate_rsi(data['close'], 14).iloc[-1]
            
            # Calculate MACD
            macd = self._calculate_macd(data['close'])[0].iloc[-1]
            
            # Determine trend
            sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
            
            if current_price > sma_20 > sma_50:
                trend = "bullish"
            elif current_price < sma_20 < sma_50:
                trend = "bearish"
            else:
                trend = "sideways"
            
            # Get support/resistance levels
            support_levels = []
            resistance_levels = []
            try:
                detector = self._get_or_create_detector(symbol, "support_resistance")
                support_zones, resistance_zones = detector.identify_zones(data)
                support_levels = [zone.level for zone in support_zones[:3]]
                resistance_levels = [zone.level for zone in resistance_zones[:3]]
            except:
                pass
            

            
            # Get divergence signals
            divergence_signals = []
            try:
                div_detector = self._get_or_create_detector(symbol, "divergence")
                div_analysis = div_detector.analyze_divergence(data)
                if div_analysis.get('signals'):
                    divergence_signals = div_analysis['signals'][:3]
            except:
                pass
            
            return MarketCondition(
                timestamp=datetime.now(),
                symbol=symbol,
                price=current_price,
                volume=volume,
                volatility=volatility,
                trend=trend,
                rsi=rsi,
                macd=macd,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                divergence_signals=divergence_signals
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions for {symbol}: {e}")
            return None
    
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
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _combine_signals(self, symbol: str, market_data: FuturesMarketData, 
                        futures_signal: Optional[Dict],
                        support_resistance_signal: Optional[Dict],
                        divergence_signal: Optional[Dict],
                        strategy_signals: Optional[Dict]) -> Optional[IntegratedSignal]:
        """Combine all signals into an integrated signal"""
        
        # Calculate base confidence from futures signal
        base_confidence = 0.0
        signal_type = "NEUTRAL"
        stop_loss = take_profit = market_data.close
        
        if futures_signal:
            base_confidence = futures_signal.get('confidence', 0.0)
            signal_type = futures_signal.get('signal_type', 'NEUTRAL')
            stop_loss = futures_signal.get('stop_loss', market_data.close)
            take_profit = futures_signal.get('take_profit', market_data.close)
        
        # Adjust confidence based on confirmations
        confirmation_bonus = 0.0
        
        # Support/resistance confirmation
        if support_resistance_signal:
            if support_resistance_signal.get('breaks'):
                confirmation_bonus += 0.1
            if support_resistance_signal.get('approaches'):
                confirmation_bonus += 0.05
        
        # Divergence confirmation
        if divergence_signal:
            signals = divergence_signal.get('signals', [])
            if signals:
                # Check if divergence aligns with signal direction
                for signal in signals:
                    if signal['type'] == 'bullish' and signal_type == 'LONG':
                        confirmation_bonus += 0.15
                    elif signal['type'] == 'bearish' and signal_type == 'SHORT':
                        confirmation_bonus += 0.15
        
        # Strategy confirmation
        if strategy_signals:
            strategy_summary = strategy_signals.get('summary', pd.DataFrame())
            if not strategy_summary.empty:
                # Check if strategies agree with signal direction
                bullish_strategies = len(strategy_summary[strategy_summary['signal'] > 0])
                bearish_strategies = len(strategy_summary[strategy_summary['signal'] < 0])
                
                if signal_type == 'LONG' and bullish_strategies > bearish_strategies:
                    confirmation_bonus += 0.1
                elif signal_type == 'SHORT' and bearish_strategies > bullish_strategies:
                    confirmation_bonus += 0.1
        
        # Calculate final confidence
        final_confidence = min(1.0, base_confidence + confirmation_bonus)
        
        # Calculate risk score
        risk_score = 1.0 - final_confidence
        
        # Calculate position size (simplified)
        position_size = final_confidence * 0.1  # 10% max position
        
        # Suggest leverage based on confidence
        if final_confidence > 0.8:
            leverage = 3.0
        elif final_confidence > 0.6:
            leverage = 2.0
        else:
            leverage = 1.0
        
        # Determine market regime
        market_regime = "unknown"
        if market_data.funding_rate > 0.01:
            market_regime = "bullish_funding"
        elif market_data.funding_rate < -0.01:
            market_regime = "bearish_funding"
        else:
            market_regime = "neutral"
        
        return IntegratedSignal(
            symbol=symbol,
            timestamp=market_data.timestamp,
            signal_type=signal_type,
            confidence=final_confidence,
            price=market_data.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            futures_signal=futures_signal,
            support_resistance_signal=support_resistance_signal,
            divergence_signal=divergence_signal,
            strategy_signals=strategy_signals,
            risk_score=risk_score,
            position_size=position_size,
            leverage_suggestion=leverage,
            funding_rate=market_data.funding_rate,
            open_interest=market_data.open_interest,
            volume_ratio=1.0,  # Would need to calculate from historical data
            market_regime=market_regime
        )
    
    def _save_signal_to_file(self, signal: IntegratedSignal):
        """Save signal to file if enabled and file doesn't exist"""
        if not self.enable_file_output:
            return
        
        timestamp = signal.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_signal_{signal.symbol}_{timestamp}.json"
        
        # Convert signal to JSON-serializable format
        signal_data = {
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'signal_type': signal.signal_type,
            'confidence': signal.confidence,
            'price': signal.price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'risk_score': signal.risk_score,
            'position_size': signal.position_size,
            'leverage_suggestion': signal.leverage_suggestion,
            'funding_rate': signal.funding_rate,
            'open_interest': signal.open_interest,
            'volume_ratio': signal.volume_ratio,
            'market_regime': signal.market_regime,
            'futures_signal': signal.futures_signal,
            'support_resistance_signal': signal.support_resistance_signal,
            'divergence_signal': signal.divergence_signal,
            'strategy_signals': signal.strategy_signals
        }
        
        # Use file manager to save with existence check
        success = self.file_manager.safe_save_json(signal_data, filename)
        if success:
            logger.info(f"Signal saved/verified: {filename}")
        else:
            logger.warning(f"Failed to save signal: {filename}")
    
    def _process_symbol(self, symbol: str):
        """Process a single symbol and generate integrated signal"""
        try:
            # Fetch current data
            market_data = self._fetch_futures_data(symbol)
            if not market_data:
                return
            
            # Update historical data
            self.historical_data[symbol].append(market_data)
            if len(self.historical_data[symbol]) > 100:
                self.historical_data[symbol] = self.historical_data[symbol][-100:]
            
            # Fetch historical data for analysis
            historical_df = self._fetch_historical_data(symbol, limit=100)
            if historical_df is None:
                return
            
            # Generate component signals
            futures_signal = self._generate_futures_signal(symbol, market_data)
            support_resistance_signal = self._generate_support_resistance_signal(symbol, historical_df)
            divergence_signal = self._generate_divergence_signal(symbol, historical_df)
            strategy_signals = self._generate_strategy_signals(symbol, historical_df)
            
            # Combine signals
            integrated_signal = self._combine_signals(
                symbol, market_data, futures_signal, 
                support_resistance_signal, divergence_signal, strategy_signals
            )
            
            if integrated_signal and integrated_signal.confidence > 0.3:  # Only show significant signals
                self.current_signals[symbol] = integrated_signal
                self.signal_history[symbol].append(integrated_signal)
                
                # Analyze market conditions
                market_condition = self.analyze_market_conditions(symbol, historical_df)
                if market_condition:
                    self.market_conditions[symbol] = market_condition
                
                # Enhanced trade execution with validation - Lowered threshold for more trades
                if integrated_signal.confidence > 0.4 and integrated_signal.signal_type in ['LONG', 'SHORT']:
                    # Check if we already have an open position for this symbol
                    if symbol not in self.open_positions or self.open_positions[symbol].status != "OPEN":
                        # Validate signal with market conditions
                        if self._validate_signal_with_market_conditions(integrated_signal, market_condition):
                            # Execute trade automatically
                            trade_executed = self.execute_trade(integrated_signal)
                            if trade_executed:
                                logger.info(f"🎯 TRADE EXECUTED: {symbol} {integrated_signal.signal_type}")
                                
                                # Send trade execution alert
                                if self.enable_alerts:
                                    # Get strategy summary
                                    strategy_summary = integrated_signal.strategy_signals.get('summary', {})
                                    best_strategy = strategy_summary.get('best_strategy', 'Multiple') if strategy_summary else 'Multiple'
                                    
                                    # Calculate potential profit/loss
                                    if integrated_signal.signal_type == 'LONG':
                                        potential_profit = ((integrated_signal.take_profit - integrated_signal.price) / integrated_signal.price) * 100
                                        potential_loss = ((integrated_signal.price - integrated_signal.stop_loss) / integrated_signal.price) * 100
                                    else:
                                        potential_profit = ((integrated_signal.price - integrated_signal.take_profit) / integrated_signal.price) * 100
                                        potential_loss = ((integrated_signal.stop_loss - integrated_signal.price) / integrated_signal.price) * 100
                                    
                                    alert_message = f"""
🚨 **TRADE EXECUTED - LIVE TRADING**

📊 **Symbol**: {symbol}
📈 **Signal**: {integrated_signal.signal_type}
🎯 **Strategy**: {best_strategy}
💪 **Confidence**: {integrated_signal.confidence:.3f}
💰 **Entry Price**: ${integrated_signal.price:.4f}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

🎯 **Risk Management:**
🛑 **Stop Loss**: ${integrated_signal.stop_loss:.4f}
🎯 **Take Profit**: ${integrated_signal.take_profit:.4f}
⚡ **Leverage**: {integrated_signal.leverage_suggestion}x

📊 **Market Conditions:**
📈 **Trend**: {market_condition.trend if market_condition else 'Unknown'}
📊 **RSI**: {market_condition.rsi:.1f if market_condition else 'N/A'}
💸 **Funding Rate**: {integrated_signal.funding_rate*100:.4f}%

✅ **Status**: Trade executed on Binance
🔧 **Mode**: Automated Live Trading
"""
                                    self.send_telegram_alert(alert_message)
                            else:
                                logger.warning(f"❌ TRADE FAILED: {symbol} {integrated_signal.signal_type}")
                        else:
                            logger.info(f"⚠️ SIGNAL REJECTED: Market conditions validation failed for {symbol}")
                    else:
                        logger.info(f"⏸️ SKIPPING TRADE: Already have open position for {symbol}")
                
                # Save to database
                self._save_signal_to_database(integrated_signal)
                
                # Send signal alert for high-confidence signals (even if not executed)
                if integrated_signal.confidence > 0.3 and self.enable_alerts:
                    # Get strategy summary
                    strategy_summary = integrated_signal.strategy_signals.get('summary', {})
                    best_strategy = strategy_summary.get('best_strategy', 'Multiple') if strategy_summary else 'Multiple'
                    
                    signal_alert = f"""
📊 **TRADING SIGNAL DETECTED**

📊 **Symbol**: {symbol}
📈 **Signal**: {integrated_signal.signal_type}
🎯 **Strategy**: {best_strategy}
💪 **Confidence**: {integrated_signal.confidence:.3f}
💰 **Price**: ${integrated_signal.price:.4f}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

📊 **Market Conditions:**
📈 **Trend**: {market_condition.trend if market_condition else 'Unknown'}
📊 **RSI**: {market_condition.rsi:.1f if market_condition else 'N/A'}
💸 **Funding Rate**: {integrated_signal.funding_rate*100:.4f}%

🔧 **Status**: Monitoring (Confidence: {integrated_signal.confidence:.1%})
"""
                    self.send_telegram_alert(signal_alert)
                
                # Save to file
                self._save_signal_to_file(integrated_signal)
                
                # Display signal (only in console, not Telegram)
                self._display_signal(integrated_signal)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _display_signal(self, signal: IntegratedSignal):
        """Display integrated signal"""
        print(f"\n🚀 INTEGRATED FUTURES SIGNAL - {signal.timestamp.strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📊 Symbol: {signal.symbol}")
        print(f"🎯 Signal Type: {signal.signal_type}")
        print(f"💰 Price: ${signal.price:,.2f}")
        print(f"📈 Confidence: {signal.confidence:.2f}")
        print(f"⚠️  Risk Score: {signal.risk_score:.2f}")
        print(f"🛑 Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"🎯 Take Profit: ${signal.take_profit:,.2f}")
        print(f"📊 Position Size: {signal.position_size:.1%}")
        print(f"⚡ Suggested Leverage: {signal.leverage_suggestion}x")
        print(f"💸 Funding Rate: {signal.funding_rate*100:.4f}%")
        print(f"📊 Open Interest: {signal.open_interest:,.0f}")
        print(f"🌍 Market Regime: {signal.market_regime}")
        
        # Component confirmations
        confirmations = []
        if signal.futures_signal:
            confirmations.append("✅ Futures Analysis")
        if signal.support_resistance_signal:
            confirmations.append("✅ Support/Resistance")
        if signal.divergence_signal:
            confirmations.append("✅ Divergence")
        if signal.strategy_signals:
            confirmations.append("✅ Strategy Signals")
        
        print(f"🔍 Confirmations: {', '.join(confirmations)}")
        
        # Trading execution status
        if signal.executed:
            print(f"🚀 TRADE EXECUTED: {signal.trade_status}")
            print(f"📋 Order ID: {signal.order_id}")
            print(f"💰 Execution Price: ${signal.execution_price:.4f}")
            print(f"⏰ Execution Time: {signal.execution_time.strftime('%H:%M:%S')}")
        elif signal.confidence > 0.6:
            print(f"🎯 TRADE READY: High confidence signal ready for execution")
        else:
            print(f"📊 SIGNAL ONLY: Confidence below execution threshold")
        
        print(f"{'='*80}")
    
    def start_monitoring(self, duration_minutes: int = 10, update_interval: int = 30):
        """Start monitoring symbols and generating integrated signals"""
        print("🚀 STARTING INTEGRATED FUTURES TRADING SYSTEM")
        print("="*80)
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🔄 Update Interval: {update_interval} seconds")
        print(f"💾 File Output: {'Enabled' if self.enable_file_output else 'Disabled'}")
        print(f"🔧 Trading Mode: {'LIVE TRADING' if self.enable_live_trading else 'PAPER TRADING'}")
        print(f"💰 Max Position Size: {self.max_position_size*100:.1f}%")
        print(f"⚠️ Risk Per Trade: {self.risk_per_trade*100:.1f}%")
        print("="*80)
        
        # Send startup notification to Telegram
        if self.enable_alerts and self.telegram_bot_token and self.telegram_chat_id:
            startup_message = f"""
🚨 **AITRADE AUTOMATED TRADING STARTED**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: Automated Trading Active
💪 **Mode**: Background Strategy Execution

📊 **Monitoring**: {len(self.symbols)} symbols
⏱️ **Duration**: {duration_minutes} minutes
🔄 **Update Interval**: {update_interval} seconds
🔧 **Trading Mode**: {'AUTOMATED LIVE TRADING' if self.enable_live_trading else 'PAPER TRADING'}
💰 **Max Position Size**: {self.max_position_size*100:.1f}%
⚠️ **Risk Per Trade**: {self.risk_per_trade*100:.1f}%

📈 **Top Symbols**: {', '.join(self.symbols[:5])}
⏰ **Start Time**: {datetime.now().strftime('%H:%M:%S')}

✅ **Automated Features:**
• Background strategy analysis
• Automatic trade execution
• Real-time Telegram alerts
• Risk management enforcement

You will receive signal alerts and trade execution notifications.
"""
            self.send_telegram_alert(startup_message)
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        signal_count = 0
        
        try:
            while datetime.now() < end_time and self.running:
                current_time = datetime.now()
                
                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Update open positions
                self.update_positions()
                
                # Update last update time
                self.last_update = current_time
                
                # Count total signals
                total_signals = sum(len(history) for history in self.signal_history.values())
                if total_signals > signal_count:
                    signal_count = total_signals
                
                # Display current positions
                self._display_positions()
                
                # Send periodic status update to Telegram (every 10 minutes)
                if self.enable_alerts and self.telegram_bot_token and self.telegram_chat_id:
                    elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed_minutes > 0 and elapsed_minutes % 10 < update_interval / 60:  # Every ~10 minutes
                        status_message = f"""
🚨 **AITRADE AUTOMATED STATUS**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: Automated Trading Active
💪 **Mode**: Background Execution

⏱️ **Elapsed Time**: {elapsed_minutes:.1f} minutes
📈 **Signals Generated**: {signal_count}
💰 **Total P&L**: ${self.total_pnl:.2f}
🎯 **Success Rate**: {(self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.1f}%
📊 **Open Positions**: {len([p for p in self.open_positions.values() if p.status == "OPEN"])}
🔧 **Trades Executed**: {self.total_trades}

⏰ **Last Update**: {datetime.now().strftime('%H:%M:%S')}
"""
                        self.send_telegram_alert(status_message)
                
                # Sleep until next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping integrated futures trading system...")
        
        finally:
            self.running = False
            
            # Send shutdown notification to Telegram
            if self.enable_alerts and self.telegram_bot_token and self.telegram_chat_id:
                shutdown_message = f"""
🚨 **AITRADE AUTOMATED SESSION ENDED**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: Automated Trading Complete
💪 **Mode**: Background Execution Finished

📊 **Session Summary:**
⏱️ **Duration**: {duration_minutes} minutes
📈 **Total Signals**: {signal_count}
🔄 **Average Signals/Min**: {signal_count/duration_minutes:.1f if duration_minutes > 0 else 0.0}
💰 **Total P&L**: ${self.total_pnl:.2f}
🎯 **Success Rate**: {(self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.1f}%
🔧 **Trades Executed**: {self.total_trades}
📊 **Open Positions**: {len([p for p in self.open_positions.values() if p.status == "OPEN"])}

⏰ **End Time**: {datetime.now().strftime('%H:%M:%S')}

✅ **Automated Features Used:**
• Background strategy analysis
• Automatic trade execution
• Real-time risk management
• Telegram notifications

Thank you for using AITRADE Automated Trading!
"""
                self.send_telegram_alert(shutdown_message)
            
            self._print_summary(signal_count, duration_minutes)
    
    def _print_summary(self, signal_count: int, duration_minutes: int):
        """Print system summary"""
        print(f"\n📈 INTEGRATED SYSTEM SUMMARY")
        print(f"{'='*50}")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"📊 Total Signals Generated: {signal_count}")
        print(f"🔄 Average Signals/Minute: {signal_count/duration_minutes:.1f}")
        print(f"📁 Output Directory: {self.output_directory}")
        
        # Use file manager to get accurate file count
        created_files = self.file_manager.list_created_files()
        print(f"💾 Files Created: {len(created_files)}")
        
        # Print file manager summary
        self.file_manager.print_summary()
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        # Get trading summary
        trading_summary = self.get_trading_summary()
        
        print(f"\n📊 TRADING PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"🎯 Total Trades: {trading_summary['total_trades']}")
        print(f"✅ Successful Trades: {trading_summary['successful_trades']}")
        print(f"📈 Success Rate: {trading_summary['success_rate']:.1f}%")
        print(f"💰 Total P&L: ${trading_summary['total_pnl']:.2f}")
        print(f"📊 Open Positions: {trading_summary['open_positions']}")
        print(f"🔧 Trading Mode: {trading_summary['trading_mode']}")
        
        # Enhanced performance metrics
        if self.performance_metrics.total_trades > 0:
            print(f"\n📈 ENHANCED PERFORMANCE METRICS")
            print(f"{'='*50}")
            print(f"🎯 Win Rate: {self.performance_metrics.win_rate:.1%}")
            print(f"📊 Profit Factor: {self.performance_metrics.profit_factor:.2f}")
            print(f"💰 Largest Win: ${self.performance_metrics.largest_win:.2f}")
            print(f"💸 Largest Loss: ${self.performance_metrics.largest_loss:.2f}")
            print(f"📊 Average Win: ${self.performance_metrics.largest_win:.2f}")
            print(f"📉 Average Loss: ${self.performance_metrics.largest_loss:.2f}")
        
        print(f"\n📊 Symbols Monitored: {', '.join(self.symbols)}")
        print(f"⚠️  REMEMBER: Futures trading involves high risk due to leverage!")
        print(f"{'='*50}")
    
    def calculate_position_size(self, signal: IntegratedSignal, account_balance: float = 10000.0) -> float:
        """Calculate position size based on risk management"""
        try:
            # Calculate position size based on risk per trade
            risk_amount = account_balance * self.risk_per_trade
            price = signal.price
            
            # Calculate stop loss distance
            if signal.signal_type == 'LONG':
                stop_distance = price - signal.stop_loss
            else:  # SHORT
                stop_distance = signal.stop_loss - price
            
            if stop_distance <= 0:
                stop_distance = price * 0.02  # Default 2% stop loss
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Apply max position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def execute_trade(self, signal: IntegratedSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.enable_live_trading or not self.exchange:
                # Paper trading mode
                return self._execute_paper_trade(signal)
            else:
                # Live trading mode
                return self._execute_live_trade(signal)
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _execute_paper_trade(self, signal: IntegratedSignal) -> bool:
        """Execute paper trade (simulation)"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {signal.symbol}")
                return False
            
            # Create trade position
            position = TradePosition(
                symbol=signal.symbol,
                side=signal.signal_type,
                entry_price=signal.price,
                quantity=position_size,
                leverage=signal.leverage_suggestion,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_id=f"PAPER_{signal.symbol}_{int(time.time())}",
                timestamp=signal.timestamp
            )
            
            # Store position
            self.open_positions[signal.symbol] = position
            self.trade_history.append(position)
            self.total_trades += 1
            
            # Update signal
            signal.executed = True
            signal.order_id = position.order_id
            signal.execution_price = signal.price
            signal.execution_time = signal.timestamp
            signal.trade_status = "EXECUTED"
            
            logger.info(f"📊 PAPER TRADE EXECUTED: {signal.symbol} {signal.signal_type} at ${signal.price:.4f}")
            logger.info(f"   Quantity: {position_size:.4f} | Leverage: {signal.leverage_suggestion}x")
            logger.info(f"   Stop Loss: ${signal.stop_loss:.4f} | Take Profit: ${signal.take_profit:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return False
    
    def _execute_live_trade(self, signal: IntegratedSignal) -> bool:
        """Execute live trade on Binance Futures"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {signal.symbol}")
                return False
            
            # Set leverage
            try:
                self.exchange.set_leverage(signal.leverage_suggestion, signal.symbol)
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")
            
            # Prepare order parameters
            side = 'buy' if signal.signal_type == 'LONG' else 'sell'
            
            order_params = {
                'symbol': signal.symbol,
                'type': 'market',
                'side': side,
                'amount': position_size,
                'params': {
                    'timeInForce': 'IOC'  # Immediate or Cancel
                }
            }
            
            # Execute order
            order = self.exchange.create_order(**order_params)
            
            if order and order.get('status') in ['closed', 'filled']:
                # Create trade position
                position = TradePosition(
                    symbol=signal.symbol,
                    side=signal.signal_type,
                    entry_price=float(order.get('price', signal.price)),
                    quantity=position_size,
                    leverage=signal.leverage_suggestion,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    order_id=order.get('id', f"LIVE_{signal.symbol}_{int(time.time())}"),
                    timestamp=signal.timestamp
                )
                
                # Store position
                self.open_positions[signal.symbol] = position
                self.trade_history.append(position)
                self.total_trades += 1
                
                # Update signal
                signal.executed = True
                signal.order_id = position.order_id
                signal.execution_price = position.entry_price
                signal.execution_time = signal.timestamp
                signal.trade_status = "EXECUTED"
                
                logger.info(f"🚀 LIVE TRADE EXECUTED: {signal.symbol} {signal.signal_type} at ${position.entry_price:.4f}")
                logger.info(f"   Quantity: {position_size:.4f} | Leverage: {signal.leverage_suggestion}x")
                logger.info(f"   Stop Loss: ${signal.stop_loss:.4f} | Take Profit: ${signal.take_profit:.4f}")
                logger.info(f"   Order ID: {position.order_id}")
                
                return True
            else:
                logger.error(f"Order failed for {signal.symbol}: {order}")
                signal.trade_status = "FAILED"
                return False
                
        except Exception as e:
            logger.error(f"Error executing live trade: {e}")
            signal.trade_status = "FAILED"
            return False
    
    def update_positions(self):
        """Update open positions with current P&L"""
        try:
            for symbol, position in self.open_positions.items():
                if position.status != "OPEN":
                    continue
                
                # Get current price
                current_data = self._fetch_futures_data(symbol)
                if not current_data:
                    continue
                
                current_price = current_data.close
                
                # Calculate P&L
                if position.side == 'LONG':
                    pnl = (current_price - position.entry_price) * position.quantity
                    pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
                else:  # SHORT
                    pnl = (position.entry_price - current_price) * position.quantity
                    pnl_percentage = ((position.entry_price - current_price) / position.entry_price) * 100
                
                position.pnl = pnl
                position.pnl_percentage = pnl_percentage
                
                # Check stop loss and take profit
                if self._should_close_position(position, current_price):
                    self._close_position(position, current_price)
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _should_close_position(self, position: TradePosition, current_price: float) -> bool:
        """Check if position should be closed based on stop loss or take profit"""
        if position.side == 'LONG':
            # Check stop loss
            if current_price <= position.stop_loss:
                return True
            # Check take profit
            if current_price >= position.take_profit:
                return True
        else:  # SHORT
            # Check stop loss
            if current_price >= position.stop_loss:
                return True
            # Check take profit
            if current_price <= position.take_profit:
                return True
        
        return False
    
    def _close_position(self, position: TradePosition, current_price: float):
        """Close position"""
        try:
            if self.enable_live_trading and self.exchange:
                # Live trading - close position
                side = 'sell' if position.side == 'LONG' else 'buy'
                
                order_params = {
                    'symbol': position.symbol,
                    'type': 'market',
                    'side': side,
                    'amount': position.quantity,
                    'params': {
                        'timeInForce': 'IOC'
                    }
                }
                
                order = self.exchange.create_order(**order_params)
                
                if order and order.get('status') in ['closed', 'filled']:
                    position.status = "CLOSED"
                    self.successful_trades += 1
                    self.total_pnl += position.pnl
                    
                    logger.info(f"🔒 POSITION CLOSED: {position.symbol} at ${current_price:.4f}")
                    logger.info(f"   P&L: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
                    
                    # Send position closure alert
                    if self.enable_alerts:
                        closure_alert = f"""
🔒 **POSITION CLOSED - AUTOMATED**

📊 **Symbol**: {position.symbol}
📈 **Side**: {position.side}
💰 **Exit Price**: ${current_price:.4f}
💰 **Entry Price**: ${position.entry_price:.4f}
📊 **P&L**: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)
⚡ **Leverage**: {position.leverage}x
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

✅ **Status**: Position closed automatically
🔧 **Mode**: Automated Live Trading
"""
                        self.send_telegram_alert(closure_alert)
                    
            else:
                # Paper trading - simulate close
                position.status = "CLOSED"
                self.successful_trades += 1
                self.total_pnl += position.pnl
                
                logger.info(f"🔒 PAPER POSITION CLOSED: {position.symbol} at ${current_price:.4f}")
                logger.info(f"   P&L: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_trading_summary(self) -> Dict:
        """Get trading performance summary"""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'open_positions': len([p for p in self.open_positions.values() if p.status == "OPEN"]),
            'trading_mode': 'Live Trading' if self.enable_live_trading else 'Paper Trading'
        }
    
    def _validate_signal_with_market_conditions(self, signal: IntegratedSignal, market_condition: MarketCondition) -> bool:
        """Validate signal against market conditions"""
        if not market_condition:
            return True  # Allow if no market condition data
        
        # Check trend alignment
        if signal.signal_type == 'LONG' and market_condition.trend == 'bearish':
            if signal.confidence < 0.8:  # Require higher confidence for counter-trend
                return False
        
        if signal.signal_type == 'SHORT' and market_condition.trend == 'bullish':
            if signal.confidence < 0.8:  # Require higher confidence for counter-trend
                return False
        
        # Check RSI extremes
        if market_condition.rsi > 80 and signal.signal_type == 'LONG':
            return False  # Don't buy when extremely overbought
        
        if market_condition.rsi < 20 and signal.signal_type == 'SHORT':
            return False  # Don't sell when extremely oversold
        
        # Check volatility
        if market_condition.volatility > 0.5:  # Very high volatility
            if signal.confidence < 0.7:  # Require higher confidence
                return False
        
        return True
    
    def _save_signal_to_database(self, signal: IntegratedSignal):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_signals 
                (timestamp, symbol, signal_type, confidence, price, stop_loss, take_profit, 
                 strategy, executed, order_id, execution_price, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.price,
                signal.stop_loss,
                signal.take_profit,
                str(signal.strategy_signals),
                signal.executed,
                signal.order_id,
                signal.execution_price,
                signal.execution_time.isoformat() if signal.execution_time else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")
    
    def _save_market_condition_to_database(self, condition: MarketCondition):
        """Save market condition to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_conditions 
                (timestamp, symbol, price, volume, volatility, trend, rsi, macd,
                 support_levels, resistance_levels, divergence_signals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                condition.timestamp.isoformat(),
                condition.symbol,
                condition.price,
                condition.volume,
                condition.volatility,
                condition.trend,
                condition.rsi,
                condition.macd,
                json.dumps(condition.support_levels),
                json.dumps(condition.resistance_levels),
                json.dumps(condition.divergence_signals)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving market condition to database: {e}")
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return
        
        # Basic metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trade_history)
        winning_pnl = [t.pnl for t in self.trade_history if t.pnl > 0]
        losing_pnl = [t.pnl for t in self.trade_history if t.pnl < 0]
        
        avg_win = np.mean(winning_pnl) if winning_pnl else 0
        avg_loss = np.mean(losing_pnl) if losing_pnl else 0
        largest_win = max(winning_pnl) if winning_pnl else 0
        largest_loss = min(losing_pnl) if losing_pnl else 0
        
        # Profit factor
        total_wins = sum(winning_pnl) if winning_pnl else 0
        total_losses = abs(sum(losing_pnl)) if losing_pnl else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Update performance metrics
        self.performance_metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss
        )
    
    def _display_positions(self):
        """Display current open positions"""
        open_positions = [p for p in self.open_positions.values() if p.status == "OPEN"]
        
        if open_positions:
            print(f"\n📊 OPEN POSITIONS ({len(open_positions)}):")
            print(f"{'='*60}")
            for position in open_positions:
                pnl_color = "🟢" if position.pnl >= 0 else "🔴"
                print(f"{pnl_color} {position.symbol} {position.side}")
                print(f"   Entry: ${position.entry_price:.4f} | Qty: {position.quantity:.4f}")
                print(f"   P&L: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
                print(f"   Stop: ${position.stop_loss:.4f} | Target: ${position.take_profit:.4f}")
                print(f"   Leverage: {position.leverage}x | Order: {position.order_id}")
                print(f"   {'='*40}")
        else:
            print(f"\n📊 No open positions")
    
    def stop(self):
        """Stop the system"""
        self.running = False
        logger.info("Integrated futures trading system stopped")

def get_top_100_futures_pairs():
    """Get top 100 non-stablecoin futures pairs from Binance"""
    import requests
    import pandas as pd
    
    print("🔍 Fetching top 100 non-stablecoin futures pairs from Binance...")
    
    try:
        # Get 24hr ticker statistics for all futures pairs
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter out stablecoins and keep only USDT pairs (most liquid)
        filtered_pairs = []
        for pair in df['symbol']:
            # Keep pairs that end with USDT but are not stablecoin pairs
            if pair.endswith('USDT') and not pair.startswith('USDT'):
                # Exclude stablecoin pairs
                base_asset = pair.replace('USDT', '')
                stablecoins = ['USDT', 'BUSD', 'USDC', 'TUSD', 'DAI', 'FRAX', 'USDP', 'USDD']
                if base_asset not in stablecoins:
                    filtered_pairs.append(pair)
        
        # Filter DataFrame to only include non-stablecoin pairs
        df_filtered = df[df['symbol'].isin(filtered_pairs)].copy()
        
        # Convert volume to numeric
        df_filtered['volume'] = pd.to_numeric(df_filtered['volume'])
        df_filtered['quoteVolume'] = pd.to_numeric(df_filtered['quoteVolume'])
        
        # Sort by 24hr volume (descending)
        df_sorted = df_filtered.sort_values('quoteVolume', ascending=False)
        
        # Get top 100
        top_100 = df_sorted.head(100)
        
        print(f"✅ Found {len(top_100)} top non-stablecoin futures pairs")
        print(f"📊 Total volume: ${top_100['quoteVolume'].sum():,.0f}")
        
        # Display top 20 for reference
        print("\n📈 TOP 20 BY VOLUME:")
        print("=" * 80)
        for i, (_, row) in enumerate(top_100.head(20).iterrows(), 1):
            volume_usd = float(row['quoteVolume'])
            price_change = float(row['priceChangePercent'])
            print(f"{i:2d}. {row['symbol']:<12} | Vol: ${volume_usd:>12,.0f} | Change: {price_change:>6.2f}%")
        
        # Get just the symbols
        symbols = top_100['symbol'].tolist()
        
        print(f"\n📋 Total pairs: {len(symbols)}")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching futures pairs: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Fallback to default

def main():
    """Main function"""
    print("🎯 INTEGRATED FUTURES TRADING SYSTEM - TOP 100 PAIRS")
    print("="*80)
    print("⚠️  WARNING: FUTURES TRADING IS HIGH RISK!")
    print("   - Leverage can amplify both profits AND losses")
    print("   - Risk of liquidation if price moves against you")
    print("   - Only trade with money you can afford to lose")
    print("   - This is for educational purposes only!")
    print("="*80)
    
    # Pre-configured credentials
    TELEGRAM_BOT_TOKEN = "8201084480:AAEsc-cLl8KIelwV7PffT4cGoclZquRpFak"
    BINANCE_API_KEY = "QVa0FqEvAtf1s5vtQE0grudNUkg3sl0IIPcdFx99cW50Cb80gPwHexW9VPGk7h0y"
    BINANCE_SECRET_KEY = "9A8hpWaTRvnaEApeCCwl7in0FvTBPIdFqXO4zidYugJgXXA9FO6TWMU3kn4JKgb0"
    DEFAULT_CHAT_ID = "1166227057"  # Pre-configured Chat ID
    
    print("🔑 Pre-configured credentials loaded:")
    print(f"   📱 Telegram Bot: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"   📊 Binance API: {BINANCE_API_KEY[:20]}...")
    print(f"   📱 Default Chat ID: {DEFAULT_CHAT_ID}")
    print("   💡 To get your Telegram Chat ID, message @userinfobot on Telegram")
    
    # Get top 100 futures pairs
    print("\n🚀 Loading top 100 non-stablecoin futures pairs...")
    symbols = get_top_100_futures_pairs()
    
    if not symbols:
        print("❌ Failed to fetch futures pairs, using defaults")
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Ask user if they want to use all 100 pairs or select specific ones
    print(f"\n📊 Found {len(symbols)} pairs")
    use_all = input("Use all pairs? (y/n, default: y): ").strip().lower() != 'n'
    
    if not use_all:
        # Let user select specific pairs
        print(f"\n📋 First 20 pairs: {', '.join(symbols[:20])}")
        symbols_input = input("Enter specific symbols (comma-separated): ").strip()
        if symbols_input:
            selected_symbols = [s.strip() for s in symbols_input.split(',')]
            # Validate that selected symbols exist in our list
            symbols = [s for s in selected_symbols if s in symbols]
            if not symbols:
                print("❌ No valid symbols selected, using defaults")
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Configuration
    duration = input("Enter monitoring duration in minutes (default: 10): ").strip()
    duration_minutes = int(duration) if duration.isdigit() else 10
    
    enable_files = input("Enable file output? (y/n, default: y): ").strip().lower() != 'n'
    
    # Trading mode selection
    print(f"\n🎯 TRADING MODE SELECTION")
    print(f"1. Paper Trading (Simulation - Safe)")
    print(f"2. Live Trading (Real Money - High Risk)")
    print(f"3. Auto Mode (Paper by default, Live with confirmation)")
    
    trading_mode = input("Select trading mode (1/2/3, default: 3): ").strip()
    
    if trading_mode == "2":
        enable_live_trading = True
        paper_trading = False
    elif trading_mode == "1":
        enable_live_trading = False
        paper_trading = True
    else:  # Default to auto mode
        enable_live_trading = False
        paper_trading = True
    
    # Use pre-configured API credentials
    api_key = BINANCE_API_KEY
    api_secret = BINANCE_SECRET_KEY
    
    if enable_live_trading:
        print(f"\n⚠️  WARNING: LIVE TRADING MODE SELECTED!")
        print(f"   - Real money will be used")
        print(f"   - High risk of financial loss")
        print(f"   - Leverage can amplify losses")
        print(f"   - Using pre-configured API credentials")
        confirm = input("Are you sure? Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Switching to paper trading mode for safety")
            enable_live_trading = False
            paper_trading = True
    else:
        print(f"📊 Using pre-configured API credentials for data access")
        print(f"🔧 Trading Mode: Paper Trading (Safe)")
    
    # Enhanced configuration
    max_position_size = float(input("Max position size % of account (default: 10): ").strip() or "10") / 100
    risk_per_trade = float(input("Risk per trade % of account (default: 2): ").strip() or "2") / 100
    max_concurrent_trades = int(input("Max concurrent trades (default: 5): ").strip() or "5")
    max_portfolio_risk = float(input("Max portfolio risk % (default: 10): ").strip() or "10") / 100
    
    # Alert settings
    print(f"\n🔔 TELEGRAM ALERTS")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    enable_alerts = input("Enable Telegram alerts? (y/n, default: y): ").strip().lower() != 'n'
    
    if enable_alerts:
        telegram_bot_token = TELEGRAM_BOT_TOKEN
        telegram_chat_id = input(f"Enter your Telegram Chat ID (default: {DEFAULT_CHAT_ID}): ").strip()
        if not telegram_chat_id:
            telegram_chat_id = DEFAULT_CHAT_ID
            print(f"✅ Using default Chat ID: {DEFAULT_CHAT_ID}")
        print(f"✅ Telegram alerts enabled with pre-configured bot token")
    else:
        telegram_bot_token = None
        telegram_chat_id = None
    
    # Backtesting
    enable_backtesting = input("Enable backtesting? (y/n, default: y): ").strip().lower() != 'n'
    
    print(f"\n🎯 Starting enhanced system with {len(symbols)} pairs...")
    print(f"📊 First 10 pairs: {', '.join(symbols[:10])}")
    print(f"🔧 Trading Mode: {'LIVE TRADING' if enable_live_trading else 'PAPER TRADING'}")
    print(f"💰 Max Position Size: {max_position_size*100:.1f}%")
    print(f"⚠️ Risk Per Trade: {risk_per_trade*100:.1f}%")
    print(f"📊 Max Concurrent Trades: {max_concurrent_trades}")
    print(f"⚠️ Max Portfolio Risk: {max_portfolio_risk*100:.1f}%")
    print(f"🔔 Alerts: {'Enabled' if enable_alerts else 'Disabled'}")
    print(f"📈 Backtesting: {'Enabled' if enable_backtesting else 'Disabled'}")
    
    # Create and start system
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        api_key=api_key,
        api_secret=api_secret,
        enable_file_output=enable_files,
        enable_live_trading=enable_live_trading,
        paper_trading=paper_trading,
        max_position_size=max_position_size,
        risk_per_trade=risk_per_trade,
        enable_backtesting=enable_backtesting,
        enable_alerts=enable_alerts,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        max_concurrent_trades=max_concurrent_trades,
        max_portfolio_risk=max_portfolio_risk
    )
    
    # Test Telegram connection if alerts are enabled
    if enable_alerts and telegram_bot_token and telegram_chat_id:
        print(f"\n🔔 Testing Telegram connection...")
        system.test_telegram_connection()
    
    try:
        system.start_monitoring(duration_minutes=duration_minutes)
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main() 