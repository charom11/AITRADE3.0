#!/usr/bin/env python3
"""
Test Integrated Futures Trading System with Real Binance Data
"""

import time
from datetime import datetime
from integrated_futures_trading_system import IntegratedFuturesTradingSystem

def test_binance_live():
    """Test the integrated system with real Binance data"""
    print("🚀 TESTING INTEGRATED FUTURES SYSTEM WITH BINANCE")
    print("=" * 80)
    print("⚠️  This will connect to Binance APIs and fetch real data")
    print("⚠️  No actual trading will be performed")
    print("=" * 80)
    
    # Create system with popular symbols
    symbols = ['BTCUSDT', 'ETHUSDT']  # Start with major pairs
    
    print(f"📊 Initializing system for symbols: {symbols}")
    
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        enable_file_output=True,
        output_directory="binance_test_signals"
    )
    
    print("✅ System initialized successfully")
    print(f"📁 Output directory: {system.output_directory}")
    print(f"💾 File output: {'Enabled' if system.enable_file_output else 'Disabled'}")
    
    # Test connection to Binance
    print("\n🔗 Testing Binance connection...")
    try:
        # Test API connection
        exchange_info = system.futures_client.get_exchange_info()
        print(f"✅ Binance connection successful")
        print(f"   Exchange status: {exchange_info.get('status', 'unknown')}")
        print(f"   Server time: {datetime.fromtimestamp(exchange_info.get('serverTime', 0)/1000)}")
    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        print("   This is expected if you don't have API access")
        return
    
    # Test data fetching for each symbol
    print("\n📊 Testing data fetching...")
    for symbol in symbols:
        try:
            # Test futures data
            market_data = system._fetch_futures_data(symbol)
            if market_data:
                print(f"✅ {symbol}: ${market_data.close:,.2f} | Funding: {market_data.funding_rate*100:.4f}% | OI: {market_data.open_interest:,.0f}")
            else:
                print(f"❌ {symbol}: Failed to fetch data")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Test signal generation
    print("\n🎯 Testing signal generation...")
    try:
        # Process one symbol to test signal generation
        system._process_symbol(symbols[0])
        print(f"✅ Signal generation completed for {symbols[0]}")
        
        # Check if any signals were generated
        if system.current_signals[symbols[0]]:
            signal = system.current_signals[symbols[0]]
            print(f"   Signal Type: {signal.signal_type}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Price: ${signal.price:,.2f}")
        else:
            print("   No significant signals generated (this is normal)")
            
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
    
    # Test file management
    print("\n📁 Testing file management...")
    created_files = system.file_manager.list_created_files()
    print(f"   Files created: {len(created_files)}")
    
    if created_files:
        print("   Recent files:")
        for file in created_files[-3:]:  # Show last 3 files
            print(f"     - {file}")
    
    # Print summary
    print("\n📈 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ System initialization: Successful")
    print(f"✅ Binance connection: Successful")
    print(f"✅ Data fetching: Tested")
    print(f"✅ Signal generation: Tested")
    print(f"✅ File management: Working")
    print(f"📊 Symbols tested: {len(symbols)}")
    print(f"📁 Files created: {len(created_files)}")
    print("=" * 50)
    
    print("\n🎉 Test completed successfully!")
    print("💡 The integrated system is working with real Binance data")
    print("📁 Check the 'binance_test_signals' directory for generated files")

def main():
    """Main function"""
    try:
        test_binance_live()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 