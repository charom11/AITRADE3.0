"""
Data Manager for Mini Hedge Fund
Handles data fetching, preprocessing, and storage for multiple asset classes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os

from config import ASSETS, DATA_CONFIG

class DataManager:
    def __init__(self, cache_dir: str = 'data_cache'):
        """
        Initialize Data Manager
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize data storage
        self.price_data = {}
        self.technical_indicators = {}
        
    def get_all_symbols(self) -> List[str]:
        """Get all trading symbols from the asset universe"""
        symbols = []
        for asset_class in ASSETS.values():
            symbols.extend(asset_class.keys())
        return symbols
    
    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_file = os.path.join(self.cache_dir, f"{symbol}.csv")
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    # Check if we need to update the data
                    if df.index[-1] < pd.to_datetime(end_date) - timedelta(days=1):
                        self.logger.info(f"Updating cached data for {symbol}")
                        df = self._fetch_from_yahoo(symbol, start_date, end_date)
                        df.to_csv(cache_file)
                else:
                    self.logger.info(f"Fetching new data for {symbol}")
                    df = self._fetch_from_yahoo(symbol, start_date, end_date)
                    df.to_csv(cache_file)
                
                data[symbol] = df
                self.logger.info(f"Successfully loaded data for {symbol}: {len(df)} rows")
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return data
    
    def _fetch_from_yahoo(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Add symbol column
        df['symbol'] = symbol
        
        return df
    
    def calculate_technical_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators for all symbols
        
        Args:
            data: Dictionary of price data
            
        Returns:
            Dictionary with technical indicators
        """
        indicators = {}
        
        for symbol, df in data.items():
            try:
                indicators[symbol] = self._calculate_indicators_for_symbol(df)
            except Exception as e:
                self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                continue
                
        return indicators
    
    def _calculate_indicators_for_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a single symbol"""
        # Make a copy to avoid modifying original data
        df_indicators = df.copy()
        
        # Moving Averages
        df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
        df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df_indicators['ema_12'] = df_indicators['close'].ewm(span=12).mean()
        df_indicators['ema_26'] = df_indicators['close'].ewm(span=26).mean()
        
        # MACD
        df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
        df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9).mean()
        df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
        
        # RSI
        delta = df_indicators['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
        bb_std = df_indicators['close'].rolling(window=20).std()
        df_indicators['bb_upper'] = df_indicators['bb_middle'] + (bb_std * 2)
        df_indicators['bb_lower'] = df_indicators['bb_middle'] - (bb_std * 2)
        df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        
        # Average True Range (ATR)
        high_low = df_indicators['high'] - df_indicators['low']
        high_close = np.abs(df_indicators['high'] - df_indicators['close'].shift())
        low_close = np.abs(df_indicators['low'] - df_indicators['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df_indicators['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df_indicators['volume_sma'] = df_indicators['volume'].rolling(window=20).mean()
        df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma']
        
        # Price momentum
        df_indicators['returns'] = df_indicators['close'].pct_change()
        df_indicators['returns_5d'] = df_indicators['close'].pct_change(periods=5)
        df_indicators['returns_20d'] = df_indicators['close'].pct_change(periods=20)
        
        # Volatility
        df_indicators['volatility_20d'] = df_indicators['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df_indicators
    
    def get_correlation_matrix(self, data: Dict[str, pd.DataFrame], 
                             lookback_period: int = 252) -> pd.DataFrame:
        """
        Calculate correlation matrix for all symbols
        
        Args:
            data: Dictionary of price data
            lookback_period: Number of days to look back for correlation calculation
            
        Returns:
            Correlation matrix DataFrame
        """
        # Extract returns for all symbols
        returns_data = {}
        for symbol, df in data.items():
            if len(df) >= lookback_period:
                returns_data[symbol] = df['close'].pct_change().tail(lookback_period)
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def get_market_data_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for all symbols
        
        Args:
            data: Dictionary of price data
            
        Returns:
            Summary statistics DataFrame
        """
        summary_data = []
        
        for symbol, df in data.items():
            if len(df) > 0:
                returns = df['close'].pct_change().dropna()
                
                summary = {
                    'symbol': symbol,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'total_return': (df['close'].iloc[-1] / df['close'].iloc[0]) - 1,
                    'annualized_return': returns.mean() * 252,
                    'volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(df['close']),
                    'current_price': df['close'].iloc[-1],
                    'avg_volume': df['volume'].mean()
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def update_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Update data for all symbols or specified symbols
        
        Args:
            symbols: List of symbols to update (None for all)
            
        Returns:
            Updated data dictionary
        """
        if symbols is None:
            symbols = self.get_all_symbols()
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        return self.fetch_data(symbols, start_date, end_date) 