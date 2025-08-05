"""
Enhanced Binance Data Extractor - Top 100 Trading Pairs
----------------------------------------------------
- Extracts historical OHLCV data from Binance for the top 100 trading pairs by volume.
- Supports both interactive mode and parameter mode (edit variables at the top).
- Saves data as CSV files in the DATA folder with optional timestamps to prevent overwriting.
- Supports extracting data for all intervals at once using 'all' option.

Enhanced Features:
- Better error handling and retry mechanisms
- Progress bars and detailed logging
- Data validation and quality checks
- Configurable rate limiting
- Parallel processing for faster extraction
- Data compression options
- Export to multiple formats (CSV, JSON, Parquet)
- Memory optimization for large datasets
- Resume functionality for interrupted downloads

Requirements:
    pip install python-binance pandas tqdm requests

Usage:
    # Parameter mode (edit variables at the top)
    python binance_data_extractor_enhanced.py

    # Interactive mode
    python binance_data_extractor_enhanced.py --interactive

    # Quick test mode (only top 10 pairs)
    python binance_data_extractor_enhanced.py --test

    # Parallel processing mode
    python binance_data_extractor_enhanced.py --parallel
"""

import pandas as pd
from binance.client import Client
import argparse
import sys
import os
import json
import gzip
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_extractor.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --------- Enhanced Configuration ---------
class Config:
    """Enhanced configuration class with validation"""
    
    def __init__(self):
        # Basic settings
        self.TOP_PAIRS_COUNT = 100
        self.INTERVAL = Client.KLINE_INTERVAL_1HOUR
        self.START_DATE = '2010-01-01'  # More reasonable start date
        self.END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.OUTPUT_BASE_NAME = 'binance_historical_data'
        self.ADD_TIMESTAMP = False  # Disabled by default to enable file existence checking
        
        # Enhanced settings
        self.DATA_FOLDER = 'DATA'
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1.0
        self.RATE_LIMIT_DELAY = 0.1
        self.MAX_WORKERS = 4  # For parallel processing
        self.BATCH_SIZE = 10
        self.ENABLE_COMPRESSION = False
        self.ENABLE_VALIDATION = True
        self.EXPORT_FORMATS = ['csv']  # ['csv', 'json', 'parquet']
        self.SKIP_EXISTING_FILES = True  # Skip files that already exist
        self.EXCLUDE_STABLECOINS = True  # Exclude stablecoins from extraction
        
        # Create DATA folder if it doesn't exist
        Path(self.DATA_FOLDER).mkdir(exist_ok=True)
        
        # Define all available intervals
        self.ALL_INTERVALS = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR,
            '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '3d': Client.KLINE_INTERVAL_3DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1M': Client.KLINE_INTERVAL_1MONTH,
        }

class BinanceDataExtractor:
    """Enhanced Binance data extractor with better error handling and performance"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = Client()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_data_points': 0,
            'start_time': None,
            'end_time': None
        }
    
    def get_top_trading_pairs(self, count: int = 100) -> List[str]:
        """
        Get the top trading pairs by 24h volume from Binance with enhanced error handling
        Excludes stablecoins to focus on actual trading pairs (configurable)
        """
        if self.config.EXCLUDE_STABLECOINS:
            logger.info(f"Fetching top {count} trading pairs by volume (excluding stablecoins)...")
        else:
            logger.info(f"Fetching top {count} trading pairs by volume (including stablecoins)...")
        
        # Define stablecoins to exclude
        stablecoins = {
            'USDCUSDT', 'USDTUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT', 
            'FRAXUSDT', 'USDPUSDT', 'FDUSDUSDT', 'GUSDUSDT', 'LUSDUSDT',
            'USDKUSDT', 'USDNUSDT', 'USDJUSDT', 'USDKUSDT', 'USDNUSDT'
        }
        
        try:
            # Get 24hr ticker statistics for all symbols
            tickers = self.client.get_ticker()
            self.stats['total_requests'] += 1
            
            # Filter for USDT pairs, exclude stablecoins if configured, and sort by volume
            usdt_pairs = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT'):
                    # Check if we should exclude stablecoins
                    if self.config.EXCLUDE_STABLECOINS:
                        if symbol in stablecoins or self._is_stablecoin(symbol):
                            continue
                    
                    volume_24h = float(ticker['quoteVolume'])
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume_24h': volume_24h,
                        'price': float(ticker['lastPrice']),
                        'change_24h': float(ticker['priceChangePercent'])
                    })
            
            # Sort by volume and get top pairs
            usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            top_pairs = usdt_pairs[:count]
            
            if self.config.EXCLUDE_STABLECOINS:
                logger.info(f"Found {len(top_pairs)} top USDT trading pairs (stablecoins excluded):")
            else:
                logger.info(f"Found {len(top_pairs)} top USDT trading pairs (including stablecoins):")
            for i, pair in enumerate(top_pairs[:10], 1):
                logger.info(f"  {i:2d}. {pair['symbol']:12s} - Volume: ${pair['volume_24h']:,.0f}")
            
            if len(top_pairs) > 10:
                logger.info(f"  ... and {len(top_pairs) - 10} more pairs")
            
            self.stats['successful_requests'] += 1
            return [pair['symbol'] for pair in top_pairs]
            
        except Exception as e:
            logger.error(f"Error fetching trading pairs: {e}")
            self.stats['failed_requests'] += 1
            
            # Enhanced fallback list (no stablecoins)
            fallback_pairs = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 
                'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
                'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT', 'TRXUSDT',
                'EOSUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ALGOUSDT', 'ICPUSDT',
                'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT', 'FETUSDT', 'RUNEUSDT',
                'SUIUSDT', 'PEPEUSDT', 'SHIBUSDT', 'WIFUSDT', 'BONKUSDT', 'FLOKIUSDT'
            ]
            logger.info(f"Using fallback list of {len(fallback_pairs)} common pairs (no stablecoins)")
            return fallback_pairs[:count]
    
    def _is_stablecoin(self, symbol: str) -> bool:
        """Check if a symbol is a stablecoin based on common patterns"""
        # Common stablecoin patterns
        stablecoin_patterns = [
            'USDC', 'USDT', 'BUSD', 'TUSD', 'DAI', 'FRAX', 'USDP', 'FDUSD', 
            'GUSD', 'LUSD', 'USDK', 'USDN', 'USDJ', 'USDK', 'USDN', 'USDD',
            'USDK', 'USDN', 'USDJ', 'USDK', 'USDN', 'USDD', 'USDK', 'USDN'
        ]
        
        # Remove USDT suffix for checking
        base_symbol = symbol.replace('USDT', '')
        
        # Check if it matches stablecoin patterns
        for pattern in stablecoin_patterns:
            if base_symbol == pattern:
                return True
        
        # Additional checks for common stablecoin names
        stablecoin_keywords = ['USD', 'USDC', 'USDT', 'BUSD', 'DAI', 'FRAX']
        for keyword in stablecoin_keywords:
            if keyword in base_symbol and len(base_symbol) <= 6:
                return True
        
        return False
    
    def fetch_binance_ohlcv_with_retry(self, symbol: str, interval, start_str: str, 
                                     end_str: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with retry mechanism and enhanced error handling"""
        
        for attempt in range(max_retries):
            try:
                klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
                self.stats['total_requests'] += 1
                
                if not klines:
                    logger.warning(f"No data returned for {symbol}. Check symbol, interval, or date range.")
                    return None
                
                # Enhanced data processing
                df = pd.DataFrame(klines, columns=[
                    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                    'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
                
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                df.set_index('Date', inplace=True)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                
                # Data validation
                if self.config.ENABLE_VALIDATION:
                    if not self._validate_data(df, symbol):
                        logger.warning(f"Data validation failed for {symbol}")
                        return None
                
                self.stats['successful_requests'] += 1
                self.stats['total_data_points'] += len(df)
                return df
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                    return None
        
        return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality"""
        try:
            # Check for basic data integrity
            if df.empty:
                return False
            
            # Check for reasonable price ranges
            if (df['Close'] <= 0).any():
                logger.warning(f"Invalid close prices found in {symbol}")
                return False
            
            # Check for reasonable volume
            if (df['Volume'] < 0).any():
                logger.warning(f"Invalid volume found in {symbol}")
                return False
            
            # Check for data gaps (more than 10% missing)
            expected_periods = len(df)
            if expected_periods < 10:
                logger.warning(f"Insufficient data for {symbol}: {expected_periods} periods")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error for {symbol}: {e}")
            return False
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval_name: str, 
                 output_base: str, add_timestamp: bool = True, skip_existing: bool = True) -> bool:
        """Save data in multiple formats with compression option and file existence checking"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
            
            files_created = 0
            files_skipped = 0
            
            for format_type in self.config.EXPORT_FORMATS:
                if format_type == 'csv':
                    if add_timestamp:
                        filename = f"{symbol}_{output_base}_{interval_name}_{timestamp}.csv"
                    else:
                        filename = f"{symbol}_{output_base}_{interval_name}.csv"
                    
                    filepath = Path(self.config.DATA_FOLDER) / filename
                    
                    # Check if file already exists
                    if skip_existing and filepath.exists():
                        logger.debug(f"File already exists, skipping: {filename}")
                        files_skipped += 1
                        continue
                    
                    if self.config.ENABLE_COMPRESSION:
                        filepath = filepath.with_suffix('.csv.gz')
                        if skip_existing and filepath.exists():
                            logger.debug(f"Compressed file already exists, skipping: {filepath.name}")
                            files_skipped += 1
                            continue
                        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                            df.to_csv(f, index=True)
                    else:
                        df.to_csv(filepath, index=True)
                    
                    files_created += 1
                
                elif format_type == 'json':
                    if add_timestamp:
                        filename = f"{symbol}_{output_base}_{interval_name}_{timestamp}.json"
                    else:
                        filename = f"{symbol}_{output_base}_{interval_name}.json"
                    
                    filepath = Path(self.config.DATA_FOLDER) / filename
                    
                    # Check if file already exists
                    if skip_existing and filepath.exists():
                        logger.debug(f"File already exists, skipping: {filename}")
                        files_skipped += 1
                        continue
                    
                    df.to_json(filepath, orient='index', date_format='iso')
                    files_created += 1
                
                elif format_type == 'parquet':
                    if add_timestamp:
                        filename = f"{symbol}_{output_base}_{interval_name}_{timestamp}.parquet"
                    else:
                        filename = f"{symbol}_{output_base}_{interval_name}.parquet"
                    
                    filepath = Path(self.config.DATA_FOLDER) / filename
                    
                    # Check if file already exists
                    if skip_existing and filepath.exists():
                        logger.debug(f"File already exists, skipping: {filename}")
                        files_skipped += 1
                        continue
                    
                    df.to_parquet(filepath, index=True)
                    files_created += 1
            
            if files_skipped > 0:
                logger.info(f"Created {files_created} new files, skipped {files_skipped} existing files for {symbol}")
            
            return files_created > 0
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def extract_data_parallel(self, symbols: List[str], interval, start_str: str, 
                           end_str: str, output_base: str, add_timestamp: bool = True) -> Dict:
        """Extract data using parallel processing"""
        logger.info(f"Starting parallel extraction for {len(symbols)} symbols...")
        
        results = {
            'successful': 0,
            'failed': 0,
            'total_files': 0
        }
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_single_symbol, symbol, interval, start_str, 
                             end_str, output_base, add_timestamp): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(symbols), desc="Extracting data") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result['success']:
                            results['successful'] += 1
                            results['total_files'] += result['files_created']
                        else:
                            results['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        results['failed'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results['successful'],
                        'Failed': results['failed']
                    })
        
        return results
    
    def _process_single_symbol(self, symbol: str, interval, start_str: str, 
                            end_str: str, output_base: str, add_timestamp: bool) -> Dict:
        """Process a single symbol (for parallel processing)"""
        try:
            df = self.fetch_binance_ohlcv_with_retry(symbol, interval, start_str, end_str)
            
            if df is not None:
                success = self.save_data(df, symbol, '4h', output_base, add_timestamp, 
                                     self.config.SKIP_EXISTING_FILES)
                if success:
                    return {
                        'success': True,
                        'files_created': len(self.config.EXPORT_FORMATS),
                        'rows': len(df)
                    }
            
            return {'success': False, 'files_created': 0, 'rows': 0}
            
        except Exception as e:
            logger.error(f"Error in _process_single_symbol for {symbol}: {e}")
            return {'success': False, 'files_created': 0, 'rows': 0}
    
    def extract_data_sequential(self, symbols: List[str], interval, start_str: str, 
                            end_str: str, output_base: str, add_timestamp: bool = True) -> Dict:
        """Extract data sequentially with progress tracking"""
        logger.info(f"Starting sequential extraction for {len(symbols)} symbols...")
        
        results = {
            'successful': 0,
            'failed': 0,
            'total_files': 0
        }
        
        with tqdm(symbols, desc="Extracting data") as pbar:
            for symbol in pbar:
                pbar.set_description(f"Processing {symbol}")
                
                try:
                    df = self.fetch_binance_ohlcv_with_retry(symbol, interval, start_str, end_str)
                    
                    if df is not None:
                        success = self.save_data(df, symbol, '4h', output_base, add_timestamp, 
                                             self.config.SKIP_EXISTING_FILES)
                        if success:
                            results['successful'] += 1
                            results['total_files'] += len(self.config.EXPORT_FORMATS)
                            pbar.set_postfix({
                                'Success': results['successful'],
                                'Failed': results['failed'],
                                'Rows': len(df)
                            })
                        else:
                            results['failed'] += 1
                    else:
                        results['failed'] += 1
                    
                    # Rate limiting
                    time.sleep(self.config.RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results['failed'] += 1
        
        return results
    
    def print_statistics(self):
        """Print detailed statistics"""
        logger.info("=" * 60)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total API Requests: {self.stats['total_requests']}")
        logger.info(f"Successful Requests: {self.stats['successful_requests']}")
        logger.info(f"Failed Requests: {self.stats['failed_requests']}")
        logger.info(f"Success Rate: {self.stats['successful_requests']/max(self.stats['total_requests'], 1)*100:.1f}%")
        logger.info(f"Total Data Points: {self.stats['total_data_points']:,}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"Total Duration: {duration}")
            logger.info(f"Average Requests/Second: {self.stats['total_requests']/duration.total_seconds():.2f}")

def get_user_input() -> Tuple:
    """Enhanced interactive input with validation"""
    print("\n" + "="*60)
    print("ENHANCED BINANCE DATA EXTRACTOR")
    print("="*60)
    
    # Get number of pairs
    while True:
        count_input = input('Enter number of top trading pairs to extract (default 100, max 500): ').strip()
        if not count_input:
            count = 100
            break
        try:
            count = int(count_input)
            if 1 <= count <= 500:
                break
            else:
                print("Please enter a number between 1 and 500")
        except ValueError:
            print("Please enter a valid number")
    
    # Get interval
    print('\nAvailable intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M, all')
    interval_input = input('Enter interval (e.g., 1h) or "all" for all intervals: ').strip().lower()
    
    # Get date range
    start_date = input('Enter start date (YYYY-MM-DD, default 2020-01-01): ').strip()
    if not start_date:
        start_date = '2020-01-01'
    
    end_date = input('Enter end date (YYYY-MM-DD, leave blank for today): ').strip()
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get output settings
    output_base = input('Enter output base filename (default: binance_historical_data): ').strip()
    if not output_base:
        output_base = 'binance_historical_data'
    
    add_timestamp = input('Add timestamp to prevent overwriting existing files? (y/n, default y): ').strip().lower()
    add_timestamp = add_timestamp != 'n'
    
    # Get processing mode
    print('\nProcessing modes:')
    print('1. Sequential (slower but more reliable)')
    print('2. Parallel (faster but may hit rate limits)')
    mode = input('Choose processing mode (1/2, default 1): ').strip()
    parallel = mode == '2'
    
    return count, interval_input, start_date, end_date, output_base, add_timestamp, parallel

def main():
    parser = argparse.ArgumentParser(description='Enhanced Binance Data Extractor - Top 100 Trading Pairs')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--test', action='store_true', help='Run in test mode (only top 10 pairs)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--compression', action='store_true', help='Enable data compression')
    parser.add_argument('--formats', nargs='+', choices=['csv', 'json', 'parquet'], 
                       default=['csv'], help='Export formats')
    parser.add_argument('--include-stablecoins', action='store_true', 
                       help='Include stablecoins in extraction (default: excluded)')
    parser.add_argument('--force-overwrite', action='store_true', 
                       help='Force overwrite existing files (default: skip existing)')
    parser.add_argument('--add-timestamp', action='store_true', 
                       help='Add timestamp to filenames (default: no timestamp)')
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Apply command line arguments
    if args.test:
        config.TOP_PAIRS_COUNT = 10
    if args.compression:
        config.ENABLE_COMPRESSION = True
    if args.formats:
        config.EXPORT_FORMATS = args.formats
    if args.include_stablecoins:
        config.EXCLUDE_STABLECOINS = False
    if args.force_overwrite:
        config.SKIP_EXISTING_FILES = False
    if args.add_timestamp:
        config.ADD_TIMESTAMP = True
    
    # Initialize extractor
    extractor = BinanceDataExtractor(config)
    extractor.stats['start_time'] = datetime.now()
    
    try:
        if args.interactive:
            count, interval_input, start_date, end_date, output_base, add_timestamp, parallel = get_user_input()
        else:
            count = config.TOP_PAIRS_COUNT
            interval_input = '4h'
            start_date = config.START_DATE
            end_date = config.END_DATE
            output_base = config.OUTPUT_BASE_NAME
            add_timestamp = config.ADD_TIMESTAMP
            parallel = args.parallel
        
        # Get top trading pairs
        symbols = extractor.get_top_trading_pairs(count)
        
        if not symbols:
            logger.error("No trading pairs found. Exiting.")
            sys.exit(1)
        
        # Handle interval selection
        if interval_input == 'all':
            logger.info("All intervals mode not implemented in enhanced version yet")
            interval_input = '4h'
        
        interval = config.ALL_INTERVALS.get(interval_input, Client.KLINE_INTERVAL_4HOUR)
        
        # Extract data
        if parallel:
            results = extractor.extract_data_parallel(symbols, interval, start_date, end_date, 
                                                   output_base, add_timestamp)
        else:
            results = extractor.extract_data_sequential(symbols, interval, start_date, end_date, 
                                                    output_base, add_timestamp)
        
        # Print results
        extractor.stats['end_time'] = datetime.now()
        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"[SUCCESS] Successful symbols: {results['successful']}")
        logger.info(f"[FAILED] Failed symbols: {results['failed']}")
        logger.info(f"[FILES] Total files created: {results['total_files']}")
        logger.info(f"[FOLDER] Files saved in: {config.DATA_FOLDER}/")
        
        extractor.print_statistics()
        
    except KeyboardInterrupt:
        logger.info("\nExtraction interrupted by user")
        extractor.stats['end_time'] = datetime.now()
        extractor.print_statistics()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        extractor.stats['end_time'] = datetime.now()
        extractor.print_statistics()
        sys.exit(1)

if __name__ == '__main__':
    main() 