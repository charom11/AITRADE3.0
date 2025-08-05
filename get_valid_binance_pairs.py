#!/usr/bin/env python3
"""
Get Valid Binance Trading Pairs
Fetches actual available trading pairs from Binance and filters out stablecoins
"""

import requests
import json
from datetime import datetime

def get_binance_trading_pairs():
    """Get all available trading pairs from Binance"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('symbols', [])
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching trading pairs: {e}")
        return []

def filter_valid_pairs(symbols, max_pairs=100):
    """Filter valid USDT pairs excluding stablecoins"""
    
    # Stablecoins to exclude
    stablecoins = {
        'USDT', 'USDC', 'BUSD', 'TUSD', 'FRAX', 'USDP', 'USDD', 'GUSD', 
        'LUSD', 'SUSD', 'DAI', 'PAX', 'USDK', 'USDN', 'USDJ', 'USDK',
        'USDN', 'USDJ', 'USDK', 'USDN', 'USDJ', 'USDK', 'USDN', 'USDJ'
    }
    
    valid_pairs = []
    
    for symbol in symbols:
        symbol_info = symbol.get('symbol', '')
        
        # Only USDT pairs
        if not symbol_info.endswith('USDT'):
            continue
            
        # Exclude stablecoins
        base_asset = symbol_info.replace('USDT', '')
        if base_asset in stablecoins:
            continue
            
        # Only include pairs that are currently trading
        if symbol.get('status') == 'TRADING':
            valid_pairs.append(symbol_info)
    
    # Sort by symbol name for consistency
    valid_pairs.sort()
    
    # Return top N pairs
    return valid_pairs[:max_pairs]

def get_24hr_stats():
    """Get 24-hour statistics for all pairs"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        print(f"Error fetching 24hr stats: {e}")
        return []

def sort_by_volume(pairs, stats):
    """Sort pairs by 24-hour volume"""
    # Create a dictionary for quick lookup
    stats_dict = {stat['symbol']: float(stat['volume']) for stat in stats}
    
    # Sort pairs by volume (highest first)
    sorted_pairs = sorted(pairs, key=lambda x: stats_dict.get(x, 0), reverse=True)
    
    return sorted_pairs

def update_config_file(valid_pairs):
    """Update the binance_config.py file with valid pairs"""
    
    config_content = f'''#!/usr/bin/env python3
"""
Binance API Configuration
Store your Binance API credentials here
"""

# Binance API Configuration
BINANCE_CONFIG = {{
    # Your Binance API Key (optional - for public data, leave empty)
    "api_key": "",  # Add your API key here
    
    # Your Binance API Secret (optional - for public data, leave empty)
    "api_secret": "",  # Add your API secret here
    
    # Trading pairs to monitor - Top {len(valid_pairs)} valid pairs from Binance
    "trading_pairs": [
        {', '.join([f'"{pair}"' for pair in valid_pairs])}
    ],
    
    # Signal generation settings
    "signal_settings": {{
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_signal_strength": 0.4,
        "min_volume_ratio": 1.5,
        "risk_reward_ratio": 2.0
    }},
    
    # Risk management settings
    "risk_settings": {{
        "max_position_size": 0.1,  # 10% of portfolio
        "stop_loss_percentage": 0.05,  # 5%
        "take_profit_percentage": 0.10,  # 10%
        "max_daily_loss": 0.02,  # 2%
        "max_portfolio_risk": 0.05  # 5%
    }},
    
    # Data collection settings
    "data_settings": {{
        "kline_interval": "1h",  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        "kline_limit": 100,
        "update_interval": 10,  # seconds
        "historical_days": 30
    }}
}}

def get_binance_config():
    """Get Binance configuration"""
    return BINANCE_CONFIG

def validate_api_credentials():
    """Validate API credentials if provided"""
    config = get_binance_config()
    
    if config["api_key"] and config["api_secret"]:
        print("✅ API credentials found - Full access enabled")
        print("   📊 Can access: Market data, Account info, Trading")
        return True
    else:
        print("⚠️  No API credentials - Public data only")
        print("   📊 Can access: Market data, Order book")
        print("   ❌ Cannot access: Account info, Trading")
        return False

def get_trading_pairs():
    """Get list of trading pairs to monitor"""
    return get_binance_config()["trading_pairs"]

def get_signal_settings():
    """Get signal generation settings"""
    return get_binance_config()["signal_settings"]

def get_risk_settings():
    """Get risk management settings"""
    return get_binance_config()["risk_settings"]

def get_data_settings():
    """Get data collection settings"""
    return get_binance_config()["data_settings"]

if __name__ == "__main__":
    print("🔧 Binance Configuration")
    print("=" * 30)
    
    config = get_binance_config()
    print(f"Trading Pairs: {{len(config['trading_pairs'])}} pairs")
    print(f"API Key: {{'✅ Set' if config['api_key'] else '❌ Not set'}}")
    print(f"API Secret: {{'✅ Set' if config['api_secret'] else '❌ Not set'}}")
    
    validate_api_credentials()
    
    print(f"\\n📊 Signal Settings:")
    for key, value in config['signal_settings'].items():
        print(f"   {{key}}: {{value}}")
    
    print(f"\\n⚠️  Risk Settings:")
    for key, value in config['risk_settings'].items():
        print(f"   {{key}}: {{value}}")
    
    print(f"\\n📈 Data Settings:")
    for key, value in config['data_settings'].items():
        print(f"   {{key}}: {{value}}")
'''
    
    try:
        with open('binance_config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ Updated binance_config.py with {len(valid_pairs)} valid pairs")
        return True
    except Exception as e:
        print(f"❌ Error updating config file: {e}")
        return False

def main():
    """Main function"""
    print("🔍 FETCHING VALID BINANCE TRADING PAIRS")
    print("=" * 50)
    
    # Get all trading pairs
    print("📊 Fetching all trading pairs from Binance...")
    all_symbols = get_binance_trading_pairs()
    print(f"   Found {len(all_symbols)} total symbols")
    
    # Filter valid pairs
    print("🔍 Filtering valid USDT pairs (excluding stablecoins)...")
    valid_pairs = filter_valid_pairs(all_symbols, max_pairs=100)
    print(f"   Found {len(valid_pairs)} valid USDT pairs")
    
    # Get 24-hour stats for volume sorting
    print("📈 Fetching 24-hour volume statistics...")
    stats = get_24hr_stats()
    
    if stats:
        print("📊 Sorting pairs by 24-hour volume...")
        sorted_pairs = sort_by_volume(valid_pairs, stats)
        valid_pairs = sorted_pairs
    
    # Show top 20 pairs
    print(f"\n🏆 TOP 20 TRADING PAIRS BY VOLUME:")
    print("-" * 40)
    for i, pair in enumerate(valid_pairs[:20], 1):
        print(f"{i:2d}. {pair}")
    
    # Update configuration file
    print(f"\n💾 Updating configuration file...")
    if update_config_file(valid_pairs):
        print(f"✅ Successfully updated binance_config.py")
        print(f"📊 Total valid pairs: {len(valid_pairs)}")
        print(f"🎯 Ready for real Binance trading signals!")
    else:
        print(f"❌ Failed to update configuration file")

if __name__ == "__main__":
    main() 