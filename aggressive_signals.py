#!/usr/bin/env python3
"""
Aggressive Live Trading Signals
Forces signal generation by creating volatile market conditions
"""

import time
import json
import random
from datetime import datetime, timedelta
from enhanced_trading_system import EnhancedTradingSystem, MarketData
from file_manager import FileManager

def create_volatile_market_data(symbol: str) -> MarketData:
    """Create volatile market data to trigger signals"""
    base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    
    # Create more volatile price movements
    price_change = random.uniform(-0.05, 0.05)  # ±5% change for volatility
    current_price = base_price * (1 + price_change)
    
    # Create volatile OHLC data
    high = current_price * (1 + random.uniform(0.01, 0.03))
    low = current_price * (1 - random.uniform(0.01, 0.03))
    open_price = current_price * (1 + random.uniform(-0.02, 0.02))
    
    return MarketData(
        symbol=symbol,
        timestamp=datetime.now(),
        open=open_price,
        high=high,
        low=low,
        close=current_price,
        volume=random.uniform(5000, 20000),  # Higher volume
        source="volatile_simulation"
    )

def show_aggressive_signals(duration_minutes=3):
    """Show aggressive live trading signals"""
    print("🚀 AGGRESSIVE LIVE TRADING SIGNALS")
    print("=" * 50)
    print(f"Duration: {duration_minutes} minutes")
    print("Creating volatile market conditions to trigger signals...\n")
    
    # Initialize system
    system = EnhancedTradingSystem()
    
    # Start data streams
    system.data_manager.start_data_streams(system.trading_pairs)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    signal_count = 0
    
    try:
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            # Create volatile market data for each symbol
            for symbol in system.trading_pairs:
                # Create volatile market data
                volatile_data = create_volatile_market_data(symbol)
                
                # Store in data manager
                with system.data_manager.lock:
                    system.data_manager.data_streams[symbol] = volatile_data
                
                # Convert to DataFrame for analysis
                import pandas as pd
                from dataclasses import asdict
                df = pd.DataFrame([asdict(volatile_data)])
                
                # Generate signal with lower thresholds
                signal = system.signal_generator.generate_signal(df, symbol)
                
                if signal:
                    signal_count += 1
                    
                    # Get additional analysis
                    ml_prediction = system.ml_predictor.predict_price_movement(df, symbol)
                    sentiment = system.sentiment_analyzer.analyze_sentiment(symbol)
                    price_info = system.exchange_manager.get_best_price(symbol, signal.signal_type)
                    
                    # Display signal
                    print(f"🔔 VOLATILE SIGNAL #{signal_count} - {current_time.strftime('%H:%M:%S')}")
                    print(f"   📊 Symbol: {signal.symbol}")
                    print(f"   🎯 Type: {signal.signal_type.upper()}")
                    print(f"   💰 Price: ${signal.price:,.2f}")
                    print(f"   📈 Strength: {signal.strength:.2f}")
                    print(f"   🎯 Confidence: {signal.confidence:.2f}")
                    print(f"   ⚠️  Risk Score: {signal.risk_score:.2f}")
                    print(f"   🛑 Stop Loss: ${signal.stop_loss:,.2f}")
                    print(f"   🎯 Take Profit: ${signal.take_profit:,.2f}")
                    print(f"   🤖 ML Prediction: {ml_prediction.get('prediction', 0):.3f}")
                    print(f"   📰 Sentiment: {sentiment.get('market_mood', 'neutral')}")
                    print(f"   📋 Conditions: {', '.join(signal.conditions[:3])}")
                    
                    # Risk management check
                    if system.risk_manager.should_trade(signal):
                        position_size = system.risk_manager.calculate_position_size(
                            signal, system.risk_manager.portfolio_value
                        )
                        print(f"   ✅ TRADE APPROVED - Position Size: {position_size:.4f}")
                        
                        # Execute trade
                        system._execute_trade(signal, position_size, price_info)
                        print(f"   💼 TRADE EXECUTED!")
                    else:
                        print(f"   ❌ TRADE REJECTED - Risk management")
                    
                    print("-" * 50)
            
            # Show system status
            if signal_count % 2 == 0 and signal_count > 0:
                status = system.get_system_status()
                print(f"📊 SYSTEM STATUS - {current_time.strftime('%H:%M:%S')}")
                print(f"   💼 Active Positions: {status.get('active_positions', 0)}")
                print(f"   💰 Portfolio Value: ${status.get('risk_metrics', {}).get('portfolio_value', 0):,.2f}")
                print(f"   ⚠️  Risk Level: {status.get('risk_metrics', {}).get('risk_level', 'unknown')}")
                print()
            
            time.sleep(3)  # Check every 3 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Aggressive signals stopped by user")
    
    # Final summary
    print(f"\n📈 AGGRESSIVE SIGNALS SUMMARY")
    print(f"   Total Signals Generated: {signal_count}")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Average Signals/Minute: {signal_count/duration_minutes:.1f}")
    
    # Show final status
    final_status = system.get_system_status()
    print(f"\n💰 FINAL PORTFOLIO STATUS")
    print(f"   Active Positions: {final_status.get('active_positions', 0)}")
    print(f"   Portfolio Value: ${final_status.get('risk_metrics', {}).get('portfolio_value', 0):,.2f}")
    print(f"   Total P&L: ${final_status.get('risk_metrics', {}).get('total_pnl', 0):,.2f}")

def main():
    """Main function"""
    print("🎯 AGGRESSIVE LIVE TRADING SIGNALS DEMO")
    print("=" * 60)
    
    try:
        # Show aggressive signals for 3 minutes
        show_aggressive_signals(duration_minutes=3)
        
        print("\n✅ Aggressive signals demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 