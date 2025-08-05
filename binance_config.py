#!/usr/bin/env python3
"""
Binance API Configuration
Store your Binance API credentials here
"""

# Binance API Configuration
BINANCE_CONFIG = {
    # Your Binance API Key (optional - for public data, leave empty)
    "api_key": "",  # Add your API key here
    
    # Your Binance API Secret (optional - for public data, leave empty)
    "api_secret": "",  # Add your API secret here
    
    # Trading pairs to monitor - Top 100 valid pairs from Binance
    "trading_pairs": [
        "BONKUSDT", "BTTCUSDT", "1000SATSUSDT", "A2ZUSDT", "BOMEUSDT", "1MBABYDOGEUSDT", "1000CATUSDT", "BANANAS31USDT", "1000CHEEMSUSDT", "AMPUSDT", "CKBUSDT", "BROCCOLI714USDT", "BEAMXUSDT", "CFXUSDT", "COSUSDT", "ALTUSDT", "ADAUSDT", "AWEUSDT", "ACHUSDT", "BABYUSDT", "CELRUSDT", "AUDIOUSDT", "CRVUSDT", "ANKRUSDT", "ARBUSDT", "ACTUSDT", "BIOUSDT", "ANIMEUSDT", "CHZUSDT", "ALGOUSDT", "CUSDT", "AIXBTUSDT", "BICOUSDT", "ARPAUSDT", "ASTRUSDT", "BLURUSDT", "ACAUSDT", "CETUSUSDT", "BMTUSDT", "C98USDT", "COTIUSDT", "BAKEUSDT", "COOKIEUSDT", "BIGTIMEUSDT", "ALICEUSDT", "CATIUSDT", "AIUSDT", "1INCHUSDT", "BBUSDT", "CTSIUSDT", "CHRUSDT", "AXLUSDT", "ARDRUSDT", "ARKMUSDT", "CGPTUSDT", "ATAUSDT", "AEVOUSDT", "APEUSDT", "ACXUSDT", "BATUSDT", "ADXUSDT", "AUSDT", "CHESSUSDT", "APTUSDT", "CAKEUSDT", "BELUSDT", "CELOUSDT", "COWUSDT", "CVCUSDT", "AVAXUSDT", "ATMUSDT", "ARKUSDT", "CVXUSDT", "ATOMUSDT", "BERAUSDT", "ACEUSDT", "AVAUSDT", "API3USDT", "ACMUSDT", "CTKUSDT", "AXSUSDT", "ASRUSDT", "CITYUSDT", "BANDUSDT", "AGLDUSDT", "ARUSDT", "ALPINEUSDT", "BARUSDT", "BNTUSDT", "BNBUSDT", "AUCTIONUSDT", "AAVEUSDT", "COMPUSDT", "BANANAUSDT", "BCHUSDT", "ALCXUSDT", "AEURUSDT", "BTCUSDT", "BNSOLUSDT", "BIFIUSDT"
    ],
    
    # Signal generation settings
    "signal_settings": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_signal_strength": 0.4,
        "min_volume_ratio": 1.5,
        "risk_reward_ratio": 2.0
    },
    
    # Risk management settings
    "risk_settings": {
        "max_position_size": 0.1,  # 10% of portfolio
        "stop_loss_percentage": 0.05,  # 5%
        "take_profit_percentage": 0.10,  # 10%
        "max_daily_loss": 0.02,  # 2%
        "max_portfolio_risk": 0.05  # 5%
    },
    
    # Data collection settings
    "data_settings": {
        "kline_interval": "1h",  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        "kline_limit": 100,
        "update_interval": 10,  # seconds
        "historical_days": 30
    }
}

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
    print(f"Trading Pairs: {len(config['trading_pairs'])} pairs")
    print(f"API Key: {'✅ Set' if config['api_key'] else '❌ Not set'}")
    print(f"API Secret: {'✅ Set' if config['api_secret'] else '❌ Not set'}")
    
    validate_api_credentials()
    
    print(f"\n📊 Signal Settings:")
    for key, value in config['signal_settings'].items():
        print(f"   {key}: {value}")
    
    print(f"\n⚠️  Risk Settings:")
    for key, value in config['risk_settings'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 Data Settings:")
    for key, value in config['data_settings'].items():
        print(f"   {key}: {value}")
