"""
Unit tests for trading strategies
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    PairsTradingStrategy, 
    DivergenceStrategy,
    SupportResistanceStrategy,
    FibonacciStrategy,
    StrategyManager
)
from config import config_manager
from error_handler import ErrorCategory, ErrorSeverity


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for MomentumStrategy"""
    
    def setUp(self):
        """Set up test data"""
        self.strategy = MomentumStrategy(config_manager.strategy_config.momentum)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=300, freq='1D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        prices = []
        for i in range(300):
            if i < 100:
                # Trending up
                price = base_price + i * 0.5 + np.random.normal(0, 1)
            elif i < 200:
                # Trending down
                price = base_price + 50 - (i - 100) * 0.3 + np.random.normal(0, 1)
            else:
                # Sideways
                price = base_price + 20 + np.random.normal(0, 1)
            prices.append(max(price, 1))  # Ensure positive prices
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in range(300)]
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, 'Momentum')
        self.assertIsNotNone(self.strategy.config)
    
    def test_generate_signals(self):
        """Test signal generation"""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Check that signals DataFrame has required columns
        required_columns = ['signal', 'signal_strength']
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check signal values are valid
        self.assertTrue(all(signals['signal'].isin([-1, 0, 1])))
        self.assertTrue(all(signals['signal_strength'].between(-1, 1)))
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        # Test buy signal
        position_size = self.strategy.calculate_position_size(1.0, 100.0, 10000.0)
        self.assertGreater(position_size, 0)
        
        # Test sell signal
        position_size = self.strategy.calculate_position_size(-1.0, 100.0, 10000.0)
        self.assertLess(position_size, 0)
        
        # Test no signal
        position_size = self.strategy.calculate_position_size(0.0, 100.0, 10000.0)
        self.assertEqual(position_size, 0)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.strategy._calculate_rsi(self.test_data['close'], 14)
        
        # Check RSI values are within valid range
        self.assertTrue(all(rsi.dropna().between(0, 100)))
        self.assertEqual(len(rsi), len(self.test_data))


class TestMeanReversionStrategy(unittest.TestCase):
    """Test cases for MeanReversionStrategy"""
    
    def setUp(self):
        """Set up test data"""
        self.strategy = MeanReversionStrategy(config_manager.strategy_config.mean_reversion)
        
        # Create sample data with mean reversion patterns
        dates = pd.date_range('2024-01-01', periods=300, freq='1D')
        np.random.seed(42)
        
        # Generate mean-reverting price data
        base_price = 100
        prices = []
        for i in range(300):
            # Add some mean reversion patterns
            if i % 50 < 25:
                # Price above mean
                price = base_price + 10 + np.random.normal(0, 2)
            else:
                # Price below mean
                price = base_price - 10 + np.random.normal(0, 2)
            prices.append(max(price, 1))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in range(300)]
        }, index=dates)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        upper, lower = self.strategy._calculate_bollinger_bands(
            self.test_data['close'], 20, 2
        )
        
        # Check that upper band is above lower band
        self.assertTrue(all(upper.dropna() >= lower.dropna()))
        
        # Check that price is sometimes outside bands
        price_outside_bands = (
            (self.test_data['close'] > upper) | 
            (self.test_data['close'] < lower)
        ).any()
        self.assertTrue(price_outside_bands)
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        atr = self.strategy._calculate_atr(self.test_data, 14)
        
        # Check ATR values are positive
        self.assertTrue(all(atr.dropna() > 0))
        self.assertEqual(len(atr), len(self.test_data))


class TestPairsTradingStrategy(unittest.TestCase):
    """Test cases for PairsTradingStrategy"""
    
    def setUp(self):
        """Set up test data"""
        self.strategy = PairsTradingStrategy(config_manager.strategy_config.pairs_trading)
        
        # Create correlated price data for pairs
        dates = pd.date_range('2024-01-01', periods=300, freq='1D')
        np.random.seed(42)
        
        # Generate correlated prices
        base_price1 = 100
        base_price2 = 50
        
        prices1 = []
        prices2 = []
        
        for i in range(300):
            # Create correlation between prices
            common_factor = np.random.normal(0, 1)
            price1 = base_price1 + common_factor * 2 + np.random.normal(0, 0.5)
            price2 = base_price2 + common_factor * 1 + np.random.normal(0, 0.3)
            
            prices1.append(max(price1, 1))
            prices2.append(max(price2, 1))
        
        self.data1 = pd.DataFrame({
            'open': prices1,
            'high': [p * 1.02 for p in prices1],
            'low': [p * 0.98 for p in prices1],
            'close': prices1,
            'volume': [np.random.randint(1000, 10000) for _ in range(300)]
        }, index=dates)
        
        self.data2 = pd.DataFrame({
            'open': prices2,
            'high': [p * 1.02 for p in prices2],
            'low': [p * 0.98 for p in prices2],
            'close': prices2,
            'volume': [np.random.randint(1000, 10000) for _ in range(300)]
        }, index=dates)
    
    def test_find_pairs(self):
        """Test pair finding functionality"""
        data_dict = {
            'BTCUSDT': self.data1,
            'ETHUSDT': self.data2
        }
        
        pairs = self.strategy.find_pairs(data_dict)
        
        # Should find at least one pair
        self.assertGreater(len(pairs), 0)
        
        # Check pair format
        for pair in pairs:
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
    
    def test_spread_calculation(self):
        """Test spread calculation"""
        spread = self.strategy.calculate_spread(self.data1, self.data2)
        
        # Check spread is calculated
        self.assertEqual(len(spread), len(self.data1))
        self.assertFalse(spread.isna().all())


class TestDivergenceStrategy(unittest.TestCase):
    """Test cases for DivergenceStrategy"""
    
    def setUp(self):
        """Set up test data"""
        self.strategy = DivergenceStrategy(config_manager.strategy_config.divergence)
        
        # Create sample data with divergence patterns
        dates = pd.date_range('2024-01-01', periods=300, freq='1D')
        np.random.seed(42)
        
        # Generate price data with potential divergence
        base_price = 100
        prices = []
        
        for i in range(300):
            if i < 100:
                # Trending up
                price = base_price + i * 0.5 + np.random.normal(0, 1)
            elif i < 200:
                # Trending down
                price = base_price + 50 - (i - 100) * 0.3 + np.random.normal(0, 1)
            else:
                # Sideways with potential divergence
                price = base_price + 20 + np.random.normal(0, 1)
            prices.append(max(price, 1))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in range(300)]
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, 'Divergence')
        self.assertIsNotNone(self.strategy.detector)
    
    def test_generate_signals(self):
        """Test signal generation"""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Check that signals DataFrame has required columns
        required_columns = ['signal', 'divergence_type', 'divergence_indicator']
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check signal values are valid
        self.assertTrue(all(signals['signal'].isin([-1, 0, 1])))


class TestStrategyManager(unittest.TestCase):
    """Test cases for StrategyManager"""
    
    def setUp(self):
        """Set up test data"""
        self.manager = StrategyManager()
        
        # Add strategies
        self.manager.add_strategy(MomentumStrategy())
        self.manager.add_strategy(MeanReversionStrategy())
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='1D')
        np.random.seed(42)
        
        prices = [100 + np.random.normal(0, 1) for _ in range(100)]
        
        self.test_data = {
            'BTCUSDT': pd.DataFrame({
                'open': prices,
                'high': [p * 1.02 for p in prices],
                'low': [p * 0.98 for p in prices],
                'close': prices,
                'volume': [np.random.randint(1000, 10000) for _ in range(100)]
            }, index=dates)
        }
    
    def test_add_strategy(self):
        """Test adding strategies"""
        initial_count = len(self.manager.strategies)
        
        # Add a new strategy
        new_strategy = DivergenceStrategy()
        self.manager.add_strategy(new_strategy)
        
        # Check strategy was added
        self.assertEqual(len(self.manager.strategies), initial_count + 1)
        self.assertIn(new_strategy.name, self.manager.strategies)
    
    def test_get_all_signals(self):
        """Test getting signals from all strategies"""
        signals = self.manager.get_all_signals(self.test_data)
        
        # Check signals were generated
        self.assertIsInstance(signals, dict)
        self.assertGreater(len(signals), 0)
        
        # Check each strategy generated signals
        for strategy_name, strategy_signals in signals.items():
            self.assertIsInstance(strategy_signals, dict)
            self.assertGreater(len(strategy_signals), 0)
    
    def test_get_strategy_summary(self):
        """Test getting strategy summary"""
        summary = self.manager.get_strategy_summary()
        
        # Check summary format
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        
        # Check required columns
        required_columns = ['name', 'positions', 'total_signals']
        for col in required_columns:
            self.assertIn(col, summary.columns)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in strategies"""
    
    def setUp(self):
        """Set up test data"""
        self.strategy = MomentumStrategy()
        
        # Create invalid data
        self.invalid_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'close': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
    
    def test_handle_empty_data(self):
        """Test handling of empty data"""
        # Should not raise exception with empty data
        try:
            signals = self.strategy.generate_signals(self.invalid_data)
            self.assertIsInstance(signals, pd.DataFrame)
        except Exception as e:
            self.fail(f"Strategy should handle empty data gracefully: {e}")
    
    def test_handle_missing_columns(self):
        """Test handling of missing columns"""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102]
        })
        
        # Should not raise exception with missing columns
        try:
            signals = self.strategy.generate_signals(incomplete_data)
            self.assertIsInstance(signals, pd.DataFrame)
        except Exception as e:
            self.fail(f"Strategy should handle missing columns gracefully: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 