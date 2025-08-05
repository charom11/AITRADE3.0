#!/usr/bin/env python3
"""
Test script to check current symbols from Binance Futures API
"""

import requests
import pandas as pd

def test_binance_symbols():
    """Test fetching symbols from Binance Futures API"""
    
    print("🔍 Testing Binance Futures API symbol fetching...")
    
    try:
        # Get 24hr ticker statistics for all futures pairs
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Successfully fetched {len(data)} symbols from Binance")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter out stablecoins and keep only USDT pairs
        filtered_pairs = []
        for pair in df['symbol']:
            # Keep pairs that end with USDT but are not stablecoin pairs
            if pair.endswith('USDT') and not pair.startswith('USDT'):
                # Exclude stablecoin pairs
                base_asset = pair.replace('USDT', '')
                stablecoins = ['USDT', 'BUSD', 'USDC', 'TUSD', 'DAI', 'FRAX', 'USDP', 'USDD']
                if base_asset not in stablecoins:
                    filtered_pairs.append(pair)
        
        print(f"✅ Filtered to {len(filtered_pairs)} non-stablecoin USDT pairs")
        
        # Filter DataFrame to only include non-stablecoin pairs
        df_filtered = df[df['symbol'].isin(filtered_pairs)].copy()
        
        # Convert volume to numeric
        df_filtered['volume'] = pd.to_numeric(df_filtered['volume'])
        df_filtered['quoteVolume'] = pd.to_numeric(df_filtered['quoteVolume'])
        
        # Sort by 24hr volume (descending)
        df_sorted = df_filtered.sort_values('quoteVolume', ascending=False)
        
        # Get top 20
        top_20 = df_sorted.head(20)
        
        print(f"\n📈 TOP 20 BY VOLUME:")
        print("=" * 80)
        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            volume_usd = float(row['quoteVolume'])
            price_change = float(row['priceChangePercent'])
            print(f"{i:2d}. {row['symbol']:<12} | Vol: ${volume_usd:>12,.0f} | Change: {price_change:>6.2f}%")
        
        # Check if ALPACAUSDT is in the list
        alpaca_found = 'ALPACAUSDT' in top_20['symbol'].values
        alpaca_rank = None
        if alpaca_found:
            alpaca_row = top_20[top_20['symbol'] == 'ALPACAUSDT'].iloc[0]
            alpaca_rank = top_20[top_20['symbol'] == 'ALPACAUSDT'].index[0] + 1
            alpaca_volume = float(alpaca_row['quoteVolume'])
            print(f"\n🔍 ALPACAUSDT found at rank #{alpaca_rank} with volume: ${alpaca_volume:,.0f}")
        else:
            print(f"\n❌ ALPACAUSDT not found in top 20")
        
        # Get just the symbols for main.py
        symbols = top_20['symbol'].tolist()
        
        print(f"\n📋 Top 20 symbols for main.py:")
        print("symbols = [")
        for symbol in symbols:
            print(f"    '{symbol}',")
        print("]")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching futures pairs: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Fallback

if __name__ == "__main__":
    test_binance_symbols() 