"""
Enhanced Backtesting System for Real Live Trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import json
import logging
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our custom modules
from divergence_detector import DivergenceDetector
from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    StrategyManager
)
from config import STRATEGY_CONFIG, TRADING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestSystem:
    """Comprehensive backtesting system for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_capital = initial_capital
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Strategy initialization
        self._initialize_strategies()
        self._initialize_detectors()
        
        # Data storage
        self.historical_data = {}
        
        # Trading parameters
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
            'DOGEUSDT', 'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT'
        ]
        
        # Risk management
        self.max_position_size = TRADING_CONFIG['max_position_size']
        self.stop_loss_pct = TRADING_CONFIG.get('stop_loss', 0.02)
        self.take_profit_pct = TRADING_CONFIG.get('take_profit', 0.04)
        self.risk_per_trade = TRADING_CONFIG['risk_per_trade']
        
        print("🚀 BACKTESTING SYSTEM INITIALIZED")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"💸 Commission: {commission*100:.2f}%")
        print(f"📊 Trading Pairs: {len(self.trading_pairs)} pairs")
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        self.strategy_manager = StrategyManager()
        
        # Add strategies
        momentum_strategy = MomentumStrategy(STRATEGY_CONFIG['momentum'])
        mean_reversion_strategy = MeanReversionStrategy(STRATEGY_CONFIG['mean_reversion'])
        pairs_strategy = PairsTradingStrategy(STRATEGY_CONFIG['pairs_trading'])
        divergence_strategy = DivergenceStrategy(STRATEGY_CONFIG['divergence'])
        
        self.strategy_manager.add_strategy(momentum_strategy)
        self.strategy_manager.add_strategy(mean_reversion_strategy)
        self.strategy_manager.add_strategy(pairs_strategy)
        self.strategy_manager.add_strategy(divergence_strategy)
        
        logger.info("All strategies initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize technical detectors"""
        self.divergence_detector = DivergenceDetector(
            rsi_period=STRATEGY_CONFIG['divergence']['rsi_period'],
            macd_fast=STRATEGY_CONFIG['divergence']['macd_fast'],
            macd_slow=STRATEGY_CONFIG['divergence']['macd_slow'],
            macd_signal=STRATEGY_CONFIG['divergence']['macd_signal'],
            min_candles=STRATEGY_CONFIG['divergence']['min_candles'],
            swing_threshold=STRATEGY_CONFIG['divergence']['swing_threshold']
        )
        
        logger.info("All detectors initialized successfully")
    
    def load_historical_data(self, data_dir: str = 'DATA', start_date: str = None, end_date: str = None):
        """Load historical data for backtesting"""
        print(f"📊 Loading historical data from {data_dir}...")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Data directory {data_dir} not found!")
            return False
        
        # Find CSV files
        csv_files = list(data_path.glob('*_binance_historical_data.csv'))
        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            return False
        
        print(f"📁 Found {len(csv_files)} data files")
        
        # Load data for each trading pair
        for symbol in self.trading_pairs:
            matching_files = [f for f in csv_files if symbol in f.name]
            if not matching_files:
                continue
            
            file_path = matching_files[0]
            try:
                # Load CSV data
                df = pd.read_csv(file_path)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                if df.columns[-1].endswith(','):
                    df = df.iloc[:, :-1]
                
                # Remove empty columns and rows
                df = df.dropna(axis=1, how='all').dropna()
                
                # Convert date column
                date_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                
                # Rename columns to standard format
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Filter by date range if specified
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    # Calculate technical indicators
                    df = self._calculate_indicators(df)
                    self.historical_data[symbol] = df
                    print(f"✅ Loaded {len(df)} records for {symbol}")
                
            except Exception as e:
                print(f"❌ Error loading data for {symbol}: {e}")
        
        print(f"📊 Loaded data for {len(self.historical_data)} symbols")
        return len(self.historical_data) > 0
    
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
        
        # MACD
        macd_fast = STRATEGY_CONFIG['divergence']['macd_fast']
        macd_slow = STRATEGY_CONFIG['divergence']['macd_slow']
        macd_signal = STRATEGY_CONFIG['divergence']['macd_signal']
        
        exp1 = df['close'].ewm(span=macd_fast).mean()
        exp2 = df['close'].ewm(span=macd_slow).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
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
    
    def generate_backtest_signals(self, symbol: str, date: datetime) -> Dict:
        """Generate trading signals for backtesting"""
        try:
            if symbol not in self.historical_data:
                return None
            
            df = self.historical_data[symbol]
            current_data = df[df.index <= date]
            if len(current_data) < 200:
                return None
            
            signals = {
                'symbol': symbol,
                'timestamp': date,
                'current_price': current_data['close'].iloc[-1],
                'strategies': {},
                'divergence': None,
                'composite_signal': 0,
                'confidence': 0.0
            }
            
            # Generate strategy signals
            strategy_data = {symbol: current_data}
            all_strategy_signals = self.strategy_manager.get_all_signals(strategy_data)
            
            for strategy_name, strategy_signals in all_strategy_signals.items():
                if symbol in strategy_signals:
                    signal_data = strategy_signals[symbol]
                    if len(signal_data) > 0:
                        latest_signal = signal_data.iloc[-1]
                        
                        signals['strategies'][strategy_name] = {
                            'signal': latest_signal.get('signal', 0),
                            'strength': latest_signal.get('signal_strength', 0.0),
                            'confidence': abs(latest_signal.get('signal', 0)) * 0.8
                        }
            
            # Generate divergence signals
            divergence_analysis = self.divergence_detector.analyze_divergence(current_data)
            if 'signals' in divergence_analysis and divergence_analysis['signals']:
                latest_divergence = divergence_analysis['signals'][0]
                signals['divergence'] = {
                    'type': latest_divergence['type'],
                    'indicator': latest_divergence['indicator'],
                    'strength': latest_divergence['strength'],
                    'signal': 1 if latest_divergence['type'] == 'bullish' else -1
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
            
            # Divergence signals (weight: 0.4)
            if signals['divergence']:
                divergence_weight = 0.4
                divergence_signal = signals['divergence']['signal']
                divergence_strength = signals['divergence']['strength']
                composite_signal += divergence_signal * divergence_strength * divergence_weight
                total_weight += divergence_weight
            
            # Normalize composite signal
            if total_weight > 0:
                signals['composite_signal'] = composite_signal / total_weight
                signals['confidence'] = min(abs(composite_signal), 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return None
    
    def execute_backtest_trade(self, symbol: str, side: str, quantity: float, price: float, timestamp: datetime):
        """Execute a trade in the backtest"""
        # Calculate commission
        commission_cost = abs(quantity * price * self.commission)
        
        if side == 'buy':
            # Check if we have enough capital
            total_cost = quantity * price + commission_cost
            if total_cost > self.current_capital:
                return False
            
            # Update capital
            self.current_capital -= total_cost
            
            # Update position
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_quantity = pos['quantity'] + quantity
                avg_price = ((pos['quantity'] * pos['avg_price']) + (quantity * price)) / total_quantity
                pos['quantity'] = total_quantity
                pos['avg_price'] = avg_price
            else:
                # Create new position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'side': 'long',
                    'entry_time': timestamp
                }
        else:  # sell
            if symbol in self.positions:
                # Close or reduce position
                pos = self.positions[symbol]
                if pos['quantity'] <= quantity:
                    # Close position
                    pnl = (price - pos['avg_price']) * pos['quantity'] - commission_cost
                    self.current_capital += pos['quantity'] * price - commission_cost
                    
                    # Record trade
                    self.trade_history.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': pos['quantity'],
                        'entry_price': pos['avg_price'],
                        'exit_price': price,
                        'pnl': pnl,
                        'entry_time': pos['entry_time'],
                        'exit_time': timestamp,
                        'duration': (timestamp - pos['entry_time']).total_seconds() / 3600,
                        'commission': commission_cost
                    })
                    
                    del self.positions[symbol]
                else:
                    # Reduce position
                    pnl = (price - pos['avg_price']) * quantity - commission_cost
                    self.current_capital += quantity * price - commission_cost
                    pos['quantity'] -= quantity
                    
                    # Record trade
                    self.trade_history.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': quantity,
                        'entry_price': pos['avg_price'],
                        'exit_price': price,
                        'pnl': pnl,
                        'entry_time': pos['entry_time'],
                        'exit_time': timestamp,
                        'duration': (timestamp - pos['entry_time']).total_seconds() / 3600,
                        'commission': commission_cost
                    })
            else:
                return False
        
        return True
    
    def run_backtest(self, start_date: str = None, end_date: str = None):
        """Run comprehensive backtest"""
        print(f"\n🚀 STARTING BACKTEST")
        print("=" * 50)
        print(f"📅 Start Date: {start_date or 'Beginning of data'}")
        print(f"📅 End Date: {end_date or 'End of data'}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print("=" * 50)
        
        # Load historical data
        if not self.load_historical_data(start_date=start_date, end_date=end_date):
            print("❌ Failed to load historical data")
            return
        
        # Get common date range
        all_dates = set()
        for symbol, df in self.historical_data.items():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]
        
        print(f"📊 Backtesting {len(all_dates)} data points across {len(self.historical_data)} symbols")
        
        # Initialize tracking
        self.equity_curve = []
        self.daily_returns = []
        previous_equity = self.initial_capital
        
        # Run backtest
        for i, date in enumerate(all_dates):
            if i % 1000 == 0:
                print(f"📈 Processing {i+1}/{len(all_dates)} ({date.strftime('%Y-%m-%d')})")
            
            # Process each symbol
            for symbol in self.historical_data.keys():
                # Generate signals
                signals = self.generate_backtest_signals(symbol, date)
                if not signals:
                    continue
                
                current_price = signals['current_price']
                composite_signal = signals['composite_signal']
                confidence = signals['confidence']
                
                # Execute trades based on signals
                if abs(composite_signal) > 0.3 and confidence > 0.5:
                    if composite_signal > 0:  # Buy signal
                        if symbol not in self.positions:
                            position_value = self.current_capital * self.risk_per_trade * abs(composite_signal) * confidence
                            max_position_value = self.current_capital * self.max_position_size
                            position_value = min(position_value, max_position_value)
                            
                            if position_value > 0:
                                quantity = position_value / current_price
                                self.execute_backtest_trade(symbol, 'buy', quantity, current_price, date)
                    
                    else:  # Sell signal
                        if symbol in self.positions:
                            pos = self.positions[symbol]
                            self.execute_backtest_trade(symbol, 'sell', pos['quantity'], current_price, date)
            
            # Check stop loss and take profit
            self._check_tp_sl_positions(date)
            
            # Calculate current equity
            current_equity = self._calculate_current_equity(date)
            self.equity_curve.append({
                'date': date,
                'equity': current_equity,
                'capital': self.current_capital,
                'positions_value': current_equity - self.current_capital
            })
            
            # Calculate daily return
            if previous_equity > 0:
                daily_return = (current_equity - previous_equity) / previous_equity
                self.daily_returns.append(daily_return)
            
            previous_equity = current_equity
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Display results
        self._display_backtest_results()
        
        # Generate charts
        self._generate_backtest_charts()
        
        print(f"\n✅ BACKTEST COMPLETED!")
    
    def _check_tp_sl_positions(self, current_date: datetime):
        """Check and execute stop loss and take profit for open positions"""
        for symbol, pos in list(self.positions.items()):
            if symbol not in self.historical_data:
                continue
            
            # Get current price
            df = self.historical_data[symbol]
            current_data = df[df.index <= current_date]
            if len(current_data) == 0:
                continue
            
            current_price = current_data['close'].iloc[-1]
            
            # Calculate P&L percentage
            pnl_pct = (current_price - pos['avg_price']) / pos['avg_price']
            
            # Check stop loss and take profit
            if pnl_pct <= -self.stop_loss_pct:
                # Stop loss hit
                self.execute_backtest_trade(symbol, 'sell', pos['quantity'], current_price, current_date)
            elif pnl_pct >= self.take_profit_pct:
                # Take profit hit
                self.execute_backtest_trade(symbol, 'sell', pos['quantity'], current_price, current_date)
    
    def _calculate_current_equity(self, current_date: datetime) -> float:
        """Calculate current equity including open positions"""
        equity = self.current_capital
        
        for symbol, pos in self.positions.items():
            if symbol in self.historical_data:
                df = self.historical_data[symbol]
                current_data = df[df.index <= current_date]
                if len(current_data) > 0:
                    current_price = current_data['close'].iloc[-1]
                    position_value = pos['quantity'] * current_price
                    equity += position_value
        
        return equity
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return
        
        # Basic metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        winning_pnl = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losing_pnl = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
        
        avg_win = np.mean(winning_pnl) if winning_pnl else 0
        avg_loss = np.mean(losing_pnl) if losing_pnl else 0
        largest_win = max(winning_pnl) if winning_pnl else 0
        largest_loss = min(losing_pnl) if losing_pnl else 0
        
        # Profit factor
        total_wins = sum(winning_pnl) if winning_pnl else 0
        total_losses = abs(sum(losing_pnl)) if losing_pnl else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Trade duration
        avg_duration = np.mean([t['duration'] for t in self.trade_history]) if self.trade_history else 0
        
        # Equity curve metrics
        if self.equity_curve:
            final_equity = self.equity_curve[-1]['equity']
            total_return = (final_equity - self.initial_capital) / self.initial_capital
            
            # Calculate drawdown
            peak = self.initial_capital
            max_drawdown = 0
            for point in self.equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                drawdown = (peak - point['equity']) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio
            if self.daily_returns:
                daily_returns_array = np.array(self.daily_returns)
                sharpe_ratio = np.mean(daily_returns_array) / np.std(daily_returns_array) if np.std(daily_returns_array) > 0 else 0
                sharpe_ratio *= np.sqrt(252)  # Annualize
            else:
                sharpe_ratio = 0
            
            # Annualized return
            if len(self.equity_curve) > 1:
                start_date = self.equity_curve[0]['date']
                end_date = self.equity_curve[-1]['date']
                days = (end_date - start_date).days
                if days > 0:
                    annualized_return = ((final_equity / self.initial_capital) ** (365 / days)) - 1
                else:
                    annualized_return = 0
            else:
                annualized_return = 0
        else:
            total_return = 0
            max_drawdown = 0
            sharpe_ratio = 0
            annualized_return = 0
        
        # Update performance metrics
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration': avg_duration,
            'total_pnl': total_pnl
        }
    
    def _display_backtest_results(self):
        """Display comprehensive backtest results"""
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 50)
        
        # Overall performance
        if self.equity_curve:
            final_equity = self.equity_curve[-1]['equity']
            print(f"💰 Final Equity: ${final_equity:,.2f}")
            print(f"📈 Total Return: {self.performance_metrics['total_return']:.2%}")
            print(f"📊 Annualized Return: {self.performance_metrics['annualized_return']:.2%}")
            print(f"📉 Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
            print(f"📊 Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        
        # Trading statistics
        print(f"\n🔄 TRADING STATISTICS")
        print("-" * 30)
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Winning Trades: {self.performance_metrics['winning_trades']}")
        print(f"Losing Trades: {self.performance_metrics['losing_trades']}")
        print(f"Win Rate: {self.performance_metrics['win_rate']:.2%}")
        print(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        
        # P&L metrics
        print(f"\n💰 P&L METRICS")
        print("-" * 30)
        print(f"Total P&L: ${self.performance_metrics['total_pnl']:,.2f}")
        print(f"Average Win: ${self.performance_metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${self.performance_metrics['avg_loss']:,.2f}")
        print(f"Largest Win: ${self.performance_metrics['largest_win']:,.2f}")
        print(f"Largest Loss: ${self.performance_metrics['largest_loss']:,.2f}")
    
    def _generate_backtest_charts(self):
        """Generate comprehensive backtest charts"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Backtest Results Analysis', fontsize=16, fontweight='bold')
            
            # 1. Equity Curve
            if self.equity_curve:
                dates = [point['date'] for point in self.equity_curve]
                equity = [point['equity'] for point in self.equity_curve]
                
                axes[0, 0].plot(dates, equity, linewidth=2, color='blue')
                axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
                axes[0, 0].set_title('Equity Curve')
                axes[0, 0].set_ylabel('Equity ($)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                
                # Format y-axis as currency
                axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # 2. Drawdown
            if self.equity_curve:
                peak = self.initial_capital
                drawdowns = []
                for point in self.equity_curve:
                    if point['equity'] > peak:
                        peak = point['equity']
                    drawdown = (peak - point['equity']) / peak
                    drawdowns.append(drawdown)
                
                axes[0, 1].fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
                axes[0, 1].plot(dates, drawdowns, color='red', linewidth=1)
                axes[0, 1].set_title('Drawdown')
                axes[0, 1].set_ylabel('Drawdown (%)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # 3. Trade P&L Distribution
            if self.trade_history:
                pnl_values = [trade['pnl'] for trade in self.trade_history]
                axes[1, 0].hist(pnl_values, bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 0].set_title('Trade P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Monthly Returns
            if self.equity_curve:
                # Group by month
                monthly_data = {}
                for point in self.equity_curve:
                    month_key = point['date'].strftime('%Y-%m')
                    if month_key not in monthly_data:
                        monthly_data[month_key] = []
                    monthly_data[month_key].append(point['equity'])
                
                monthly_returns = []
                months = []
                for month, equities in sorted(monthly_data.items()):
                    if len(equities) > 1:
                        monthly_return = (equities[-1] - equities[0]) / equities[0]
                        monthly_returns.append(monthly_return)
                        months.append(month)
                
                if monthly_returns:
                    colors = ['green' if ret >= 0 else 'red' for ret in monthly_returns]
                    axes[1, 1].bar(range(len(months)), monthly_returns, color=colors, alpha=0.7)
                    axes[1, 1].set_title('Monthly Returns')
                    axes[1, 1].set_xlabel('Month')
                    axes[1, 1].set_ylabel('Return (%)')
                    axes[1, 1].set_xticks(range(len(months)))
                    axes[1, 1].set_xticklabels(months, rotation=45)
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_results_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Charts saved to: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
    
    def save_backtest_results(self, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_results_{timestamp}.json'
        
        results = {
            'backtest_info': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'start_date': self.equity_curve[0]['date'].isoformat() if self.equity_curve else None,
                'end_date': self.equity_curve[-1]['date'].isoformat() if self.equity_curve else None,
                'trading_pairs': self.trading_pairs,
                'strategies': list(self.strategy_manager.strategies.keys())
            },
            'performance_metrics': self.performance_metrics,
            'trade_history': [
                {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl': trade['pnl'],
                    'entry_time': trade['entry_time'].isoformat(),
                    'exit_time': trade['exit_time'].isoformat(),
                    'duration': trade['duration'],
                    'commission': trade['commission']
                }
                for trade in self.trade_history
            ],
            'equity_curve': [
                {
                    'date': point['date'].isoformat(),
                    'equity': point['equity'],
                    'capital': point['capital'],
                    'positions_value': point['positions_value']
                }
                for point in self.equity_curve
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Backtest results saved to: {filename}")

def main():
    """Main function to run backtesting"""
    print("🚀 ENHANCED BACKTESTING SYSTEM")
    print("=" * 70)
    
    # Initialize backtest system
    backtest = BacktestSystem(
        initial_capital=100000,  # $100k starting capital
        commission=0.001  # 0.1% commission
    )
    
    # Run backtest
    backtest.run_backtest(
        start_date='2024-01-01',  # Start from January 2024
        end_date='2024-12-31'     # End at December 2024
    )
    
    # Save results
    backtest.save_backtest_results()
    
    print(f"\n🎉 BACKTESTING COMPLETED!")
    print("=" * 70)
    print("📊 Check the generated charts and JSON file for detailed results")
    print("📈 Use these results to optimize your strategies before live trading")

if __name__ == "__main__":
    main()
