"""
Binance Data Extractor - Top 100 Trading Pairs
---------------------------------------------
- Extracts historical OHLCV data from Binance for the top 100 trading pairs by volume.
- Supports both interactive mode and parameter mode (edit variables at the top).
- Saves data as CSV files in the DATA folder with optional timestamps to prevent overwriting.
- Supports extracting data for all intervals at once using 'all' option.

Requirements:
    pip install python-binance pandas

Usage:
    # Parameter mode (edit variables at the top)
    python binance_data_extractor.py

    # Interactive mode
    python binance_data_extractor.py --interactive

Features:
    - Prevents overwriting existing data by adding timestamps to filenames
    - Configurable timestamp option (ADD_TIMESTAMP parameter)
    - Interactive mode asks user about timestamp preference
"""

import pandas as pd
from binance.client import Client
import argparse
import sys
import os
from datetime import datetime
import time

# --------- Parameter Mode (edit these) ---------
TOP_PAIRS_COUNT = 100        # Number of top trading pairs to extract
INTERVAL = Client.KLINE_INTERVAL_4HOUR  # e.g., Client.KLINE_INTERVAL_1MINUTE, _1HOUR, _1DAY
START_DATE = '2010-01-01'    # Format: 'YYYY-MM-DD'
END_DATE = '2025-08-02'      # End date
OUTPUT_BASE_NAME = 'binance_historical_data'  # Base name for output files
ADD_TIMESTAMP = True  # Add timestamp to prevent overwriting existing files

# Create DATA folder if it doesn't exist
DATA_FOLDER = 'DATA'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"Created {DATA_FOLDER} folder")

# Define all available intervals
ALL_INTERVALS = {
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

def get_top_trading_pairs(count=100):
    """
    Get the top trading pairs by 24h volume from Binance
    """
    client = Client()
    print(f"Fetching top {count} trading pairs by volume...")
    
    try:
        # Get 24hr ticker statistics for all symbols
        tickers = client.get_ticker()
        
        # Filter for USDT pairs and sort by volume
        usdt_pairs = []
        for ticker in tickers:
            if ticker['symbol'].endswith('USDT'):
                volume_24h = float(ticker['quoteVolume'])  # Volume in USDT
                usdt_pairs.append({
                    'symbol': ticker['symbol'],
                    'volume_24h': volume_24h,
                    'price': float(ticker['lastPrice']),
                    'change_24h': float(ticker['priceChangePercent'])
                })
        
        # Sort by volume and get top pairs
        usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
        top_pairs = usdt_pairs[:count]
        
        print(f"Found {len(top_pairs)} top USDT trading pairs:")
        for i, pair in enumerate(top_pairs[:10], 1):
            print(f"  {i:2d}. {pair['symbol']:12s} - Volume: ${pair['volume_24h']:,.0f}")
        
        if len(top_pairs) > 10:
            print(f"  ... and {len(top_pairs) - 10} more pairs")
        
        return [pair['symbol'] for pair in top_pairs]
        
    except Exception as e:
        print(f"Error fetching trading pairs: {e}")
        # Fallback to common pairs if API fails
        fallback_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 
            'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
            'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT', 'TRXUSDT',
            'EOSUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ALGOUSDT', 'ICPUSDT'
        ]
        print(f"Using fallback list of {len(fallback_pairs)} common pairs")
        return fallback_pairs[:count]

# --------- Interactive Mode ---------
def get_user_input():
    count = input('Enter number of top trading pairs to extract (default 100): ').strip()
    count = int(count) if count.isdigit() else 100
    
    print('Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M, all')
    interval_input = input('Enter interval (e.g., 1h) or "all" for all intervals: ').strip().lower()
    
    # Handle "all" option
    if interval_input == 'all':
        interval = 'all'
    else:
        interval = ALL_INTERVALS.get(interval_input, Client.KLINE_INTERVAL_1HOUR)
    
    start_date = input('Enter start date (YYYY-MM-DD): ').strip()
    end_date = input('Enter end date (YYYY-MM-DD, leave blank for up to now): ').strip()
    output_base = input('Enter output base filename: ').strip()
    
    add_timestamp = input('Add timestamp to prevent overwriting existing files? (y/n, default y): ').strip().lower()
    add_timestamp = add_timestamp != 'n'
    
    if not end_date:
        end_date = None
    return count, interval, start_date, end_date, output_base, add_timestamp


def fetch_binance_ohlcv(symbol, interval, start_str, end_str):
    """Fetch OHLCV data for a single symbol"""
    client = Client()
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    
    if not klines:
        print(f"No data returned for {symbol}. Check symbol, interval, or date range.")
        return None
    
    df = pd.DataFrame(klines, columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df


def extract_data_for_pairs(symbols, interval, start_str, end_str, output_base, add_timestamp=True):
    """Extract data for multiple symbols"""
    print(f"\nExtracting data for {len(symbols)} trading pairs...")
    print(f"Interval: {interval}")
    print(f"Date range: {start_str} to {end_str or 'now'}")
    print("-" * 60)
    
    # Generate timestamp for unique filenames if requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    
    successful_extractions = 0
    failed_extractions = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:3d}/{len(symbols)}] Processing {symbol}...", end=" ")
        
        df = fetch_binance_ohlcv(symbol, interval, start_str, end_str)
        
        if df is not None:
            # Save file with timestamp if requested
            if add_timestamp:
                filename = f"{symbol}_{output_base}_{timestamp}.csv"
            else:
                filename = f"{symbol}_{output_base}.csv"
            filepath = os.path.join(DATA_FOLDER, filename)
            df.to_csv(filepath)
            print(f"✓ Saved {len(df)} rows to {filename}")
            successful_extractions += 1
        else:
            print("✗ Failed")
            failed_extractions += 1
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    print("-" * 60)
    print(f"Extraction completed!")
    print(f"✓ Successful: {successful_extractions}")
    print(f"✗ Failed: {failed_extractions}")
    print(f"📁 Files saved in: {DATA_FOLDER}/")


def extract_all_intervals_for_pairs(symbols, start_str, end_str, output_base, add_timestamp=True):
    """Extract data for all intervals for multiple symbols"""
    print(f"\nExtracting data for {len(symbols)} trading pairs across all intervals...")
    print(f"Date range: {start_str} to {end_str or 'now'}")
    print("This may take a while...")
    print("-" * 60)
    
    # Generate timestamp for unique filenames if requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    
    client = Client()
    total_files = 0
    successful_extractions = 0
    failed_extractions = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:3d}/{len(symbols)}] Processing {symbol}...")
        
        symbol_success = 0
        symbol_failed = 0
        
        for interval_name, interval_value in ALL_INTERVALS.items():
            try:
                klines = client.get_historical_klines(symbol, interval_value, start_str, end_str)
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                        'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                    df.set_index('Date', inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    
                    # Save individual file with timestamp if requested
                    if add_timestamp:
                        filename = f"{symbol}_{output_base}_{interval_name}_{timestamp}.csv"
                    else:
                        filename = f"{symbol}_{output_base}_{interval_name}.csv"
                    filepath = os.path.join(DATA_FOLDER, filename)
                    df.to_csv(filepath)
                    symbol_success += 1
                    total_files += 1
                    print(f"  ✓ {interval_name}: {len(df)} rows")
                else:
                    print(f"  ✗ {interval_name}: No data")
                    symbol_failed += 1
                    
            except Exception as e:
                print(f"  ✗ {interval_name}: Error - {e}")
                symbol_failed += 1
                continue
            
            # Add small delay to avoid rate limiting
            time.sleep(0.05)
        
        if symbol_success > 0:
            successful_extractions += 1
        else:
            failed_extractions += 1
        
        print(f"  {symbol}: {symbol_success} successful, {symbol_failed} failed")
        print()
    
    print("-" * 60)
    print(f"Extraction completed!")
    print(f"✓ Successful symbols: {successful_extractions}")
    print(f"✗ Failed symbols: {failed_extractions}")
    print(f"📁 Total files created: {total_files}")
    print(f"📁 Files saved in: {DATA_FOLDER}/")


def main():
    parser = argparse.ArgumentParser(description='Binance Data Extractor - Top 100 Trading Pairs')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    if args.interactive:
        count, interval, start_date, end_date, output_base, add_timestamp = get_user_input()
    else:
        count, interval, start_date, end_date, output_base = (
            TOP_PAIRS_COUNT, INTERVAL, START_DATE, END_DATE, OUTPUT_BASE_NAME)
        add_timestamp = ADD_TIMESTAMP
        if not end_date:
            end_date = None

    # Get top trading pairs
    symbols = get_top_trading_pairs(count)
    
    if not symbols:
        print("No trading pairs found. Exiting.")
        sys.exit(1)

    # Extract data
    if interval == 'all':
        extract_all_intervals_for_pairs(symbols, start_date, end_date, output_base, add_timestamp)
    else:
        extract_data_for_pairs(symbols, interval, start_date, end_date, output_base, add_timestamp)

if __name__ == '__main__':
    main() 