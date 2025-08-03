"""
Portfolio Manager for Mini Hedge Fund
Handles position sizing, risk management, and portfolio rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

from config import TRADING_CONFIG, RISK_CONFIG, PERFORMANCE_CONFIG

class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, strategy: str, side: str, 
                 quantity: float, entry_price: float, entry_date: datetime):
        self.symbol = symbol
        self.strategy = strategy
        self.side = side  # 'long' or 'short'
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        
    def update_price(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        
        # Calculate P&L
        if self.side == 'long':
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.quantity
            
        self.pnl_pct = self.pnl / (self.entry_price * self.quantity)
        
        # Update trailing stop
        if self.trailing_stop is not None:
            if self.side == 'long':
                new_stop = current_price * (1 - RISK_CONFIG['trailing_stop'])
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            else:  # short
                new_stop = current_price * (1 + RISK_CONFIG['trailing_stop'])
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
    
    def should_exit(self) -> Tuple[bool, str]:
        """Check if position should be exited based on risk management rules"""
        # Stop loss check
        if self.stop_loss is not None:
            if self.side == 'long' and self.current_price <= self.stop_loss:
                return True, 'stop_loss'
            elif self.side == 'short' and self.current_price >= self.stop_loss:
                return True, 'stop_loss'
        
        # Take profit check
        if self.take_profit is not None:
            if self.side == 'long' and self.current_price >= self.take_profit:
                return True, 'take_profit'
            elif self.side == 'short' and self.current_price <= self.take_profit:
                return True, 'take_profit'
        
        # Trailing stop check
        if self.trailing_stop is not None:
            if self.side == 'long' and self.current_price <= self.trailing_stop:
                return True, 'trailing_stop'
            elif self.side == 'short' and self.current_price >= self.trailing_stop:
                return True, 'trailing_stop'
        
        return False, ''

class PortfolioManager:
    """Manages portfolio positions, risk, and performance"""
    
    def __init__(self, initial_capital: float = None):
        """
        Initialize Portfolio Manager
        
        Args:
            initial_capital: Initial portfolio capital
        """
        self.initial_capital = initial_capital or TRADING_CONFIG['initial_capital']
        self.current_capital = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.closed_positions = []
        self.trade_history = []
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.daily_returns = []
        self.daily_values = []
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        
    def calculate_position_size(self, signal: float, price: float, 
                              strategy: str, symbol: str) -> float:
        """
        Calculate position size based on signal strength and risk management
        
        Args:
            signal: Signal strength (-1 to 1)
            price: Current price
            strategy: Strategy name
            symbol: Symbol to trade
            
        Returns:
            Position size in dollars
        """
        if signal == 0:
            return 0
        
        # Base position size
        base_size = self.current_capital * TRADING_CONFIG['risk_per_trade']
        
        # Adjust for signal strength
        position_size = base_size * abs(signal)
        
        # Apply maximum position size limit
        max_position = self.current_capital * TRADING_CONFIG['max_position_size']
        position_size = min(position_size, max_position)
        
        # Check sector exposure limits
        if not self._check_sector_exposure(symbol, position_size):
            position_size = 0
        
        # Check correlation limits
        if not self._check_correlation_limits(symbol, position_size):
            position_size = 0
        
        return position_size if signal > 0 else -position_size
    
    def _check_sector_exposure(self, symbol: str, position_size: float) -> bool:
        """Check if adding position would exceed sector exposure limits"""
        # This is a simplified check - in practice, you'd map symbols to sectors
        current_exposure = sum([pos.quantity * pos.current_price 
                              for pos in self.positions.values()])
        
        if current_exposure + position_size > self.current_capital * RISK_CONFIG['max_sector_exposure']:
            self.logger.warning(f"Sector exposure limit would be exceeded for {symbol}")
            return False
        
        return True
    
    def _check_correlation_limits(self, symbol: str, position_size: float) -> bool:
        """Check correlation limits with existing positions"""
        # This is a simplified check - in practice, you'd calculate actual correlations
        # For now, we'll limit the number of positions
        if len(self.positions) >= 10:  # Arbitrary limit
            self.logger.warning(f"Too many positions, skipping {symbol}")
            return False
        
        return True
    
    def open_position(self, symbol: str, strategy: str, side: str, 
                     quantity: float, price: float, date: datetime) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name
            side: 'long' or 'short'
            quantity: Position quantity
            price: Entry price
            date: Entry date
            
        Returns:
            True if position opened successfully
        """
        try:
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                self.logger.warning(f"Position already exists for {symbol}")
                return False
            
            # Create position
            position = Position(symbol, strategy, side, quantity, price, date)
            
            # Set stop loss and take profit
            if side == 'long':
                position.stop_loss = price * (1 - RISK_CONFIG['stop_loss'])
                position.take_profit = price * (1 + RISK_CONFIG['take_profit'])
                position.trailing_stop = price * (1 - RISK_CONFIG['trailing_stop'])
            else:  # short
                position.stop_loss = price * (1 + RISK_CONFIG['stop_loss'])
                position.take_profit = price * (1 - RISK_CONFIG['take_profit'])
                position.trailing_stop = price * (1 + RISK_CONFIG['trailing_stop'])
            
            # Add to positions
            self.positions[symbol] = position
            
            # Record trade
            trade = {
                'date': date,
                'symbol': symbol,
                'strategy': strategy,
                'action': 'open',
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': quantity * price
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"Opened {side} position: {symbol} ({quantity} @ {price})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {str(e)}")
            return False
    
    def close_position(self, symbol: str, price: float, date: datetime, 
                      reason: str = 'manual') -> bool:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            price: Exit price
            date: Exit date
            reason: Reason for closing
            
        Returns:
            True if position closed successfully
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Calculate P&L
            if position.side == 'long':
                pnl = (price - position.entry_price) * position.quantity
            else:  # short
                pnl = (position.entry_price - price) * position.quantity
            
            # Update capital
            self.current_capital += pnl
            
            # Record trade
            trade = {
                'date': date,
                'symbol': symbol,
                'strategy': position.strategy,
                'action': 'close',
                'side': position.side,
                'quantity': position.quantity,
                'price': price,
                'value': position.quantity * price,
                'pnl': pnl,
                'reason': reason
            }
            self.trade_history.append(trade)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            self.logger.info(f"Closed position: {symbol} (P&L: ${pnl:.2f}, Reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def update_positions(self, current_prices: Dict[str, float], date: datetime):
        """
        Update all positions with current prices and check exit conditions
        
        Args:
            current_prices: Dictionary of current prices
            date: Current date
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
                
                # Check exit conditions
                should_exit, reason = position.should_exit()
                if should_exit:
                    positions_to_close.append((symbol, reason))
        
        # Close positions that meet exit criteria
        for symbol, reason in positions_to_close:
            self.close_position(symbol, current_prices[symbol], date, reason)
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                if position.side == 'long':
                    portfolio_value += (current_prices[symbol] - position.entry_price) * position.quantity
                else:  # short
                    portfolio_value += (position.entry_price - current_prices[symbol]) * position.quantity
        
        return portfolio_value
    
    def update_performance(self, current_prices: Dict[str, float], date: datetime):
        """Update performance metrics"""
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        # Update peak value and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record daily values
        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'drawdown': current_drawdown
        })
        
        # Check if we've hit maximum drawdown
        if current_drawdown > TRADING_CONFIG['max_drawdown']:
            self.logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
            # In practice, you might want to close all positions or stop trading
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        
        total_pnl = sum([t.get('pnl', 0) for t in self.trade_history])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1,
            'max_drawdown': self.max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'open_positions': len(self.positions)
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        positions_data = []
        
        for symbol, position in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'strategy': position.strategy,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': position.pnl,
                'pnl_pct': position.pnl_pct,
                'entry_date': position.entry_date
            })
        
        return pd.DataFrame(positions_data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trade_history)
    
    def save_portfolio_state(self, filename: str):
        """Save portfolio state to file"""
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value,
            'trade_history': self.trade_history,
            'daily_values': self.daily_values
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, default=str, indent=2)
        
        self.logger.info(f"Portfolio state saved to {filename}")
    
    def load_portfolio_state(self, filename: str):
        """Load portfolio state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.initial_capital = state['initial_capital']
            self.current_capital = state['current_capital']
            self.max_drawdown = state['max_drawdown']
            self.peak_value = state['peak_value']
            self.trade_history = state['trade_history']
            self.daily_values = state['daily_values']
            
            self.logger.info(f"Portfolio state loaded from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {str(e)}") 