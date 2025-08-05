#!/usr/bin/env python3
"""
Run Integrated Futures Trading System with .env credentials
"""

import os
from dotenv import load_dotenv
from integrated_futures_trading_system import IntegratedFuturesTradingSystem, get_top_100_futures_pairs

def load_environment():
    """Load environment variables from .env file"""
    print("🔍 Loading environment variables from .env file...")
    
    # Try to load .env file
    env_loaded = load_dotenv()
    
    if env_loaded:
        print("✅ .env file loaded successfully")
    else:
        print("⚠️  No .env file found, checking existing environment variables")
    
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    
    if api_key and api_secret:
        print("✅ API credentials found in environment")
        print(f"📊 API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"🔐 API Secret: {api_secret[:8]}...{api_secret[-4:]}")
        return api_key, api_secret
    else:
        print("❌ API credentials not found in environment")
        print("💡 Make sure your .env file contains:")
        print("   BINANCE_API_KEY=your_api_key_here")
        print("   BINANCE_SECRET_KEY=your_api_secret_here")
        return None, None

def main():
    """Main function to run the integrated system with .env credentials"""
    print("🚀 INTEGRATED FUTURES TRADING SYSTEM - WITH .ENV CREDENTIALS")
    print("="*80)
    print("⚠️  WARNING: FUTURES TRADING IS HIGH RISK!")
    print("   - Leverage can amplify both profits AND losses")
    print("   - Risk of liquidation if price moves against you")
    print("   - Only trade with money you can afford to lose")
    print("   - This is for educational purposes only!")
    print("="*80)
    
    # Load environment variables
    api_key, api_secret = load_environment()
    
    if not api_key or not api_secret:
        print("\n❌ Cannot proceed without API credentials")
        print("💡 Please check your .env file and try again")
        return
    
    # Get top 100 futures pairs
    print("\n🚀 Loading top 100 non-stablecoin futures pairs...")
    symbols = get_top_100_futures_pairs()
    
    if not symbols:
        print("❌ Failed to fetch futures pairs, using defaults")
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Configuration
    print(f"\n📊 Found {len(symbols)} pairs")
    use_all = input("Use all pairs? (y/n, default: y): ").strip().lower() != 'n'
    
    if not use_all:
        # Let user select specific pairs
        print(f"\n📋 First 20 pairs: {', '.join(symbols[:20])}")
        symbols_input = input("Enter specific symbols (comma-separated): ").strip()
        if symbols_input:
            selected_symbols = [s.strip() for s in symbols_input.split(',')]
            # Validate that selected symbols exist in our list
            symbols = [s for s in selected_symbols if s in symbols]
            if not symbols:
                print("❌ No valid symbols selected, using defaults")
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Get monitoring duration
    duration = input("Enter monitoring duration in minutes (default: 10): ").strip()
    duration_minutes = int(duration) if duration.isdigit() else 10
    
    # File output setting
    enable_files = input("Enable file output? (y/n, default: y): ").strip().lower() != 'n'
    
    print(f"\n🎯 Starting system with {len(symbols)} pairs...")
    print(f"📊 First 10 pairs: {', '.join(symbols[:10])}")
    print(f"⏱️ Duration: {duration_minutes} minutes")
    print(f"💾 File output: {'Enabled' if enable_files else 'Disabled'}")
    
    # Create and start system
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        api_key=api_key,
        api_secret=api_secret,
        enable_file_output=enable_files,
        output_directory="env_integrated_signals"
    )
    
    try:
        print("\n🚀 Starting integrated futures trading system...")
        system.start_monitoring(duration_minutes=duration_minutes)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        system.stop()
        print("✅ System stopped")

if __name__ == "__main__":
    main() 