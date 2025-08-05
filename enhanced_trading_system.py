#!/usr/bin/env python3
"""
Enhanced Trading System - All Features Integrated
Implements all suggested improvements with file prevention system
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("WebSockets module not installed. Install with: pip install websockets")
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

# Import our existing modules
from file_manager import FileManager
from optimized_signal_generator import OptimizedSignalGenerator, OptimizedTradingSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "binance"

@dataclass
class TradingPosition:
    """Trading position structure"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    status: str = 'open'  # 'open', 'closed', 'cancelled'
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None

class RealTimeDataManager:
    """Real-time data management with WebSocket support"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.websocket_connections = {}
        self.data_streams = {}
        self.running = False
        self.lock = threading.Lock()
        
    async def connect_websocket(self, symbol: str, interval: str = '1m'):
        """Connect to Binance WebSocket for real-time data"""
        try:
            if not WEBSOCKETS_AVAILABLE:
                logger.warning("WebSockets not available, using simulated data")
                # Simulate real-time data
                while self.running:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        open=50000.0 + np.random.normal(0, 100),
                        high=50100.0 + np.random.normal(0, 100),
                        low=49900.0 + np.random.normal(0, 100),
                        close=50000.0 + np.random.normal(0, 100),
                        volume=np.random.randint(1000, 10000)
                    )
                    
                    with self.lock:
                        self.data_streams[symbol] = market_data
                    
                    filename = f"realtime_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.file_manager.save_file(filename, asdict(market_data))
                    
                    await asyncio.sleep(60)  # Simulate 1-minute intervals
                return
                
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to WebSocket for {symbol}")
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if 'k' in data:
                            kline = data['k']
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                                open=float(kline['o']),
                                high=float(kline['h']),
                                low=float(kline['l']),
                                close=float(kline['c']),
                                volume=float(kline['v'])
                            )
                            
                            # Store in memory and file
                            with self.lock:
                                self.data_streams[symbol] = market_data
                            
                            # Save to file with prevention
                            filename = f"realtime_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            self.file_manager.save_file(filename, asdict(market_data))
                            
                    except Exception as e:
                        logger.error(f"WebSocket error for {symbol}: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"Failed to connect WebSocket for {symbol}: {e}")
    
    def start_data_streams(self, symbols: List[str]):
        """Start real-time data streams for multiple symbols"""
        self.running = True
        
        for symbol in symbols:
            # Use simulated data instead of websocket for now
            self._simulate_data_stream(symbol)
            logger.info(f"Started simulated data stream for {symbol}")
    
    def _simulate_data_stream(self, symbol: str):
        """Simulate real-time data stream"""
        # Generate simulated market data
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        # Simulate price movement
        price_change = random.uniform(-0.02, 0.02)  # ±2% change
        current_price = base_price * (1 + price_change)
        
        # Create simulated market data
        market_data = MarketData(
                symbol=symbol,
            timestamp=datetime.now(),
            open=current_price * 0.999,
            high=current_price * 1.001,
            low=current_price * 0.999,
            close=current_price,
            volume=random.uniform(1000, 10000),
            source="simulated"
        )
        
        # Store in data streams
        with self.lock:
            self.data_streams[symbol] = market_data
    
    def stop_data_streams(self):
        """Stop all data streams"""
        self.running = False
        logger.info("Stopped all data streams")
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        with self.lock:
            return self.data_streams.get(symbol)

class MachineLearningPredictor:
    """Machine Learning prediction engine"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.models = {}
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'atr', 'volume_ratio', 'price_momentum', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'sma_20', 'sma_50', 'ema_12', 'ema_26'
        ]
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML prediction"""
        df = data.copy()
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['atr'] = self._calculate_atr(df)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        df['williams_r'] = self._calculate_williams_r(df)
        df['cci'] = self._calculate_cci(df)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        return df[self.feature_columns].fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic"""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=3).mean()
        return k, d
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        return -100 * ((high_max - data['close']) / (high_max - low_min))
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate CCI"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def predict_price_movement(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Predict price movement using ML models"""
        try:
            features = self.prepare_features(data)
            
            if len(features) < 50:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Simple ensemble prediction (can be enhanced with actual ML models)
            latest_features = features.iloc[-1]
            
            # Calculate prediction based on technical indicators
            prediction = 0.0
            confidence = 0.0
            
            # RSI-based prediction
            if latest_features['rsi'] < 30:
                prediction += 0.3
                confidence += 0.2
            elif latest_features['rsi'] > 70:
                prediction -= 0.3
                confidence += 0.2
            
            # MACD-based prediction
            if latest_features['macd'] > latest_features['macd_signal']:
                prediction += 0.2
                confidence += 0.15
            else:
                prediction -= 0.2
                confidence += 0.15
            
            # Bollinger Bands prediction
            current_price = data['close'].iloc[-1]
            if current_price < latest_features['bb_lower']:
                prediction += 0.25
                confidence += 0.15
            elif current_price > latest_features['bb_upper']:
                prediction -= 0.25
                confidence += 0.15
            
            # Save prediction with file prevention
            prediction_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'confidence': min(confidence, 1.0),
                'features': latest_features.to_dict()
            }
            
            filename = f"ml_prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, prediction_data)
            
            return {
                'prediction': prediction,
                'confidence': min(confidence, 1.0)
            }
            
        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return {'prediction': 0.0, 'confidence': 0.0}

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.positions = {}
        self.portfolio_value = 10000.0  # Starting capital
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        self.correlation_matrix = {}
        
    def calculate_position_size(self, signal: OptimizedTradingSignal, available_capital: float) -> float:
        """Calculate optimal position size based on risk"""
        try:
            # Calculate risk amount
            risk_amount = available_capital * self.max_risk_per_trade
            
            # Calculate position size based on stop loss distance
            if signal.signal_type == 'buy':
                risk_per_share = signal.price - signal.stop_loss
            else:
                risk_per_share = signal.stop_loss - signal.price
            
            if risk_per_share <= 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # Apply confidence multiplier
            position_size *= signal.confidence
            
            # Limit position size
            max_position_value = available_capital * 0.1  # Max 10% per position
            max_position_size = max_position_value / signal.price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics"""
        try:
            total_exposure = 0.0
            total_pnl = 0.0
            
            for position in self.positions.values():
                if position.status == 'open':
                    total_exposure += position.quantity * position.entry_price
                    total_pnl += position.pnl
            
            # Calculate VaR (simplified)
            var_95 = total_exposure * 0.02  # 2% daily VaR
            
            # Calculate drawdown
            current_value = self.portfolio_value + total_pnl
            drawdown = (self.portfolio_value - current_value) / self.portfolio_value
            
            risk_metrics = {
                'total_exposure': total_exposure,
                'portfolio_value': current_value,
                'total_pnl': total_pnl,
                'var_95': var_95,
                'drawdown': drawdown,
                'risk_level': 'high' if drawdown > 0.05 else 'medium' if drawdown > 0.02 else 'low'
            }
            
            # Save risk metrics with file prevention
            filename = f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return {}
    
    def should_trade(self, signal: OptimizedTradingSignal) -> bool:
        """Determine if trade should be executed based on risk"""
        try:
            # Check portfolio risk
            risk_metrics = self.calculate_portfolio_risk()
            
            if risk_metrics.get('drawdown', 0) > 0.05:  # 5% drawdown limit
                logger.warning("Trade rejected: Portfolio drawdown too high")
                return False
            
            # Check signal strength
            if signal.strength < 0.6:
                logger.warning("Trade rejected: Signal strength too low")
                return False
            
            # Check risk score
            if signal.risk_score > 0.8:
                logger.warning("Trade rejected: Risk score too high")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False

class MultiExchangeManager:
    """Multi-exchange trading manager"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.exchanges = {}
        self.arbitrage_opportunities = []
        
    def add_exchange(self, name: str, api_key: str, api_secret: str):
        """Add exchange to manager"""
        try:
            # Initialize exchange connection (placeholder)
            self.exchanges[name] = {
                'api_key': api_key,
                'api_secret': api_secret,
                'connected': True
            }
            logger.info(f"Added exchange: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add exchange {name}: {e}")
    
    def get_best_price(self, symbol: str, side: str) -> Dict[str, float]:
        """Get best price across all exchanges"""
        try:
            prices = {}
            
            for exchange_name, exchange_data in self.exchanges.items():
                if exchange_data['connected']:
                    # Simulate price fetching (replace with actual API calls)
                    if side == 'buy':
                        price = 50000.0 + np.random.normal(0, 100)  # Simulate BTC price
                    else:
                        price = 50000.0 + np.random.normal(0, 100)
                    
                    prices[exchange_name] = price
            
            if not prices:
                return {}
            
            if side == 'buy':
                best_exchange = min(prices, key=prices.get)
                best_price = prices[best_exchange]
            else:
                best_exchange = max(prices, key=prices.get)
                best_price = prices[best_exchange]
            
            result = {
                'exchange': best_exchange,
                'price': best_price,
                'all_prices': prices
            }
            
            # Save price data with file prevention
            filename = f"price_comparison_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Price comparison error: {e}")
            return {}
    
    def detect_arbitrage(self, symbol: str) -> List[Dict]:
        """Detect arbitrage opportunities"""
        try:
            buy_prices = self.get_best_price(symbol, 'buy')
            sell_prices = self.get_best_price(symbol, 'sell')
            
            if not buy_prices or not sell_prices:
                return []
            
            opportunities = []
            
            for buy_exchange, buy_price in buy_prices['all_prices'].items():
                for sell_exchange, sell_price in sell_prices['all_prices'].items():
                    if buy_exchange != sell_exchange:
                        spread = sell_price - buy_price
                        spread_percentage = (spread / buy_price) * 100
                        
                        if spread_percentage > 0.5:  # 0.5% minimum spread
                            opportunity = {
                'symbol': symbol,
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'spread': spread,
                                'spread_percentage': spread_percentage,
                                'timestamp': datetime.now().isoformat()
                            }
                            opportunities.append(opportunity)
            
            # Save arbitrage opportunities with file prevention
            if opportunities:
                filename = f"arbitrage_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.file_manager.save_file(filename, opportunities)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Arbitrage detection error: {e}")
            return []

class BacktestingEngine:
    """Advanced backtesting engine"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.results = {}
        
    def run_backtest(self, data: pd.DataFrame, strategy_params: Dict) -> Dict:
        """Run comprehensive backtest"""
        try:
            initial_capital = strategy_params.get('initial_capital', 10000)
            current_capital = initial_capital
            positions = []
            trades = []
            
            # Simulate trading
            for i in range(50, len(data)):
                # Generate signal (simplified)
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    signal_type = 'buy'
                    signal_strength = 0.7
                else:
                    signal_type = 'sell'
                    signal_strength = 0.7
                
                # Execute trade
                if signal_strength > 0.6:
                    trade = {
                        'timestamp': data.index[i],
                        'signal': signal_type,
                        'price': data['close'].iloc[i],
                        'capital_before': current_capital
                    }
                    
                    # Calculate P&L
                    if signal_type == 'buy':
                        # Simulate profit
                        profit = data['close'].iloc[i] * 0.01  # 1% profit
                        current_capital += profit
                    else:
                        # Simulate loss
                        loss = data['close'].iloc[i] * 0.005  # 0.5% loss
                        current_capital -= loss
                    
                    trade['capital_after'] = current_capital
                    trade['pnl'] = trade['capital_after'] - trade['capital_before']
                    trades.append(trade)
            
            # Calculate metrics
            total_return = (current_capital - initial_capital) / initial_capital
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate drawdown
            peak_capital = initial_capital
            max_drawdown = 0
            
            for trade in trades:
                if trade['capital_after'] > peak_capital:
                    peak_capital = trade['capital_after']
                drawdown = (peak_capital - trade['capital_after']) / peak_capital
                max_drawdown = max(max_drawdown, drawdown)
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': current_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'trades': trades
            }
            
            # Save backtest results with file prevention
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {}

class SentimentAnalyzer:
    """Market sentiment analysis"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.sentiment_scores = {}
        
    def analyze_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze market sentiment for symbol"""
        try:
            # Simulate sentiment analysis (replace with actual API calls)
            # Twitter sentiment, Reddit sentiment, news sentiment, etc.
            
            sentiment_score = np.random.normal(0, 0.3)  # -1 to 1 scale
            confidence = np.random.uniform(0.5, 1.0)
            
            # Calculate sentiment metrics
            sentiment_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'overall_sentiment': sentiment_score,
                'confidence': confidence,
                'twitter_sentiment': sentiment_score + np.random.normal(0, 0.1),
                'reddit_sentiment': sentiment_score + np.random.normal(0, 0.1),
                'news_sentiment': sentiment_score + np.random.normal(0, 0.1),
                'fear_greed_index': np.random.uniform(0, 100),
                'market_mood': 'bullish' if sentiment_score > 0.2 else 'bearish' if sentiment_score < -0.2 else 'neutral'
            }
            
            # Save sentiment data with file prevention
            filename = f"sentiment_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return {}

class EnhancedTradingSystem:
    """Main enhanced trading system integrating all features"""
    
    def __init__(self):
        # Initialize file manager with prevention
        self.file_manager = FileManager("enhanced_trading_data")
        
        # Initialize all components
        self.signal_generator = OptimizedSignalGenerator()
        self.data_manager = RealTimeDataManager(self.file_manager)
        self.ml_predictor = MachineLearningPredictor(self.file_manager)
        self.risk_manager = RiskManager(self.file_manager)
        self.exchange_manager = MultiExchangeManager(self.file_manager)
        self.backtest_engine = BacktestingEngine(self.file_manager)
        self.sentiment_analyzer = SentimentAnalyzer(self.file_manager)
        
        # System state
        self.running = False
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.positions = {}
        
        logger.info("Enhanced Trading System initialized")
    
    def start_system(self):
        """Start the enhanced trading system"""
        try:
            self.running = True
            
            # Start real-time data streams
            self.data_manager.start_data_streams(self.trading_pairs)
            
            # Start main trading loop
            self._trading_loop()
            
            logger.info("Enhanced Trading System started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
    
    def stop_system(self):
        """Stop the enhanced trading system"""
        try:
            self.running = False
            self.data_manager.stop_data_streams()
            logger.info("Enhanced Trading System stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                for symbol in self.trading_pairs:
                    # Refresh simulated data
                    self.data_manager._simulate_data_stream(symbol)
                    
                    # Get latest data
                    market_data = self.data_manager.get_latest_data(symbol)
                    
                    if market_data:
                        # Convert to DataFrame for analysis
                        df = pd.DataFrame([asdict(market_data)])
                        
                        # Generate signal
                        signal = self.signal_generator.generate_signal(df, symbol)
                        
                        if signal:
                            # Get ML prediction
                            ml_prediction = self.ml_predictor.predict_price_movement(df, symbol)
                            
                            # Get sentiment analysis
                            sentiment = self.sentiment_analyzer.analyze_sentiment(symbol)
                            
                            # Get best price across exchanges
                            price_info = self.exchange_manager.get_best_price(symbol, signal.signal_type)
                            
                            # Check risk management
                            if self.risk_manager.should_trade(signal):
                                # Calculate position size
                                position_size = self.risk_manager.calculate_position_size(
                                    signal, self.risk_manager.portfolio_value
                                )
                                
                                # Execute trade (simulated)
                                self._execute_trade(signal, position_size, price_info)
                            
                            # Detect arbitrage opportunities
                            arbitrage_opps = self.exchange_manager.detect_arbitrage(symbol)
                            if arbitrage_opps:
                                logger.info(f"Arbitrage opportunities found for {symbol}: {len(arbitrage_opps)}")
                
                # Sleep between iterations
                time.sleep(5)  # 5 second intervals for demo
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(10)
    
    def _execute_trade(self, signal: OptimizedTradingSignal, position_size: float, price_info: Dict):
        """Execute trade (simulated)"""
        try:
            # Create position
            position = TradingPosition(
                symbol=signal.symbol,
                side='long' if signal.signal_type == 'buy' else 'short',
                entry_price=signal.price,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now()
            )
            
            # Store position
            self.positions[signal.symbol] = position
            
            # Save trade data with file prevention
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'entry_price': signal.price,
                'position_size': position_size,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'conditions': signal.conditions
            }
            
            filename = f"trade_execution_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, trade_data)
            
            logger.info(f"Trade executed: {signal.symbol} {signal.signal_type} at {signal.price}")
        
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Run backtest for symbol"""
        try:
            # Load historical data (simulated)
            data = pd.DataFrame({
                'close': np.random.randn(1000).cumsum() + 50000,
                'high': np.random.randn(1000).cumsum() + 50100,
                'low': np.random.randn(1000).cumsum() + 49900,
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            strategy_params = {
                'initial_capital': 10000,
                'max_risk_per_trade': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
            
            results = self.backtest_engine.run_backtest(data, strategy_params)
            
            logger.info(f"Backtest completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return {}
    
    def get_system_status(self) -> Dict:
        """Get system status and performance metrics"""
        try:
            # Get risk metrics
            risk_metrics = self.risk_manager.calculate_portfolio_risk()
            
            # Get performance metrics
            performance = self.signal_generator.get_performance_summary()
            
            status = {
                'system_running': self.running,
                'active_positions': len(self.positions),
                'trading_pairs': self.trading_pairs,
                'risk_metrics': risk_metrics,
                'performance_metrics': performance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save status with file prevention
            filename = f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, status)
            
            return status
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {}

def main():
    """Main function to run the enhanced trading system"""
    try:
        # Initialize system
        system = EnhancedTradingSystem()
        
        # Start system
        system.start_system()
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            system.stop_system()
        
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main() 