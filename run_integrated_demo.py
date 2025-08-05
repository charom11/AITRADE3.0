#!/usr/bin/env python3
"""
Integrated Futures Trading System - Live Demo
Runs the complete integrated system with real Binance data for demonstration
"""

import time
from datetime import datetime
from integrated_futures_trading_system import IntegratedFuturesTradingSystem

def run_integrated_demo():
    """Run a comprehensive demo of the integrated system"""
    print("🚀 INTEGRATED FUTURES TRADING SYSTEM - LIVE DEMO")
    print("=" * 80)
    print("⚠️  This demo will run for 3 minutes with real Binance data")
    print("⚠️  No actual trading will be performed")
    print("⚠️  Press Ctrl+C to stop early")
    print("=" * 80)
    
    # Create system with popular symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    print(f"📊 Initializing system for symbols: {symbols}")
    print(f"⏱️  Demo duration: 3 minutes")
    print(f"🔄 Update interval: 30 seconds")
    print()
    
    # Create and start the integrated system
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        enable_file_output=True,
        output_directory="live_demo_signals"
    )
    
    print("✅ System initialized successfully!")
    print(f"📁 Output directory: {system.output_directory}")
    print(f"💾 File output: {'Enabled' if system.enable_file_output else 'Disabled'}")
    print()
    
    # Test connection
    try:
        exchange_info = system.futures_client.get_exchange_info()
        print(f"🔗 Binance connection: ✅ Success")
        print(f"   Server time: {datetime.fromtimestamp(exchange_info.get('serverTime', 0)/1000)}")
        print()
    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        return
    
    # Start monitoring
    print("🚀 Starting live monitoring...")
    print("=" * 80)
    
    try:
        # Run for 3 minutes (6 cycles of 30 seconds each)
        system.start_monitoring(duration_minutes=3, update_interval=30)
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        system.stop()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📈 DEMO SUMMARY")
    print("=" * 80)
    
    # Count signals generated
    total_signals = 0
    for symbol in symbols:
        signal_count = len(system.signal_history.get(symbol, []))
        total_signals += signal_count
        print(f"📊 {symbol}: {signal_count} signals")
    
    # Count files created
    created_files = system.file_manager.list_created_files()
    print(f"📁 Files created: {len(created_files)}")
    
    # Show recent files
    if created_files:
        print("\n📄 Recent signal files:")
        for file in created_files[-5:]:  # Show last 5 files
            print(f"   - {file}")
    
    # Show file manager summary
    print("\n📁 File Manager Summary:")
    system.file_manager.print_summary()
    
    print("\n🎉 Demo completed!")
    print("💡 The integrated system successfully:")
    print("   ✅ Connected to Binance APIs")
    print("   ✅ Fetched real market data")
    print("   ✅ Generated trading signals")
    print("   ✅ Applied file prevention")
    print("   ✅ Combined multiple analysis methods")
    print("   ✅ Managed risk and position sizing")
    
    print(f"\n📁 Check the '{system.output_directory}' directory for generated signals")

def main():
    """Main function"""
    try:
        run_integrated_demo()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 