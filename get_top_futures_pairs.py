#!/usr/bin/env python3
"""
Get Top 100 Non-Stablecoin Futures Pairs from Binance
"""

import requests
import pandas as pd
from datetime import datetime

def get_top_futures_pairs():
    """Get top 100 non-stablecoin futures pairs from Binance"""
    
    print("🔍 Fetching top futures pairs from Binance...")
    
    try:
        # Get 24hr ticker statistics for all futures pairs
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter out stablecoins and keep only USDT pairs (most liquid)
        # We want pairs like BTCUSDT, ETHUSDT, etc. but not USDTUSDT
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
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"top_100_futures_pairs_{timestamp}.csv"
        top_100.to_csv(filename, index=False)
        
        # Get just the symbols
        symbols = top_100['symbol'].tolist()
        
        print(f"\n💾 Saved to: {filename}")
        print(f"📋 Total pairs: {len(symbols)}")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching futures pairs: {e}")
        return []

def main():
    """Main function"""
    print("🚀 BINANCE FUTURES - TOP 100 NON-STABLECOIN PAIRS")
    print("=" * 80)
    
    symbols = get_top_futures_pairs()
    
    if symbols:
        print(f"\n🎯 Ready to trade {len(symbols)} pairs!")
        print("⚠️  Remember: Futures trading involves high risk!")
        
        # Show first 10 symbols as example
        print(f"\n📋 First 10 symbols: {', '.join(symbols[:10])}")
        
        return symbols
    else:
        print("❌ Failed to fetch futures pairs")
        return []

if __name__ == "__main__":
    main() 