#!/usr/bin/env python3
"""
Demo Live Trading Signals
Generates proper dataset and shows realistic live trading signals
"""

import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhanced_trading_system import EnhancedTradingSystem, MarketData
from file_manager import FileManager

def generate_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate realistic historical data for signal generation"""
    base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    
    # Generate realistic price movements
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='1H')
    n_points = len(dates)
    
    # Create price series with realistic volatility
    returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    prices = [base_price]
    
    for i in range(1, n_points):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Create realistic OHLC from price
        high = price * (1 + random.uniform(0, 0.01))
        low = price * (1 - random.uniform(0, 0.01))
        open_price = price * (1 + random.uniform(-0.005, 0.005))
        volume = random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_live_market_data(symbol: str, historical_df: pd.DataFrame) -> MarketData:
    """Create live market data based on historical trends"""
    last_price = historical_df['close'].iloc[-1]
    
    # Create realistic price movement based on recent trend
    recent_trend = historical_df['close'].pct_change().tail(5).mean()
    price_change = recent_trend + random.uniform(-0.01, 0.01)  # Add some noise
    current_price = last_price * (1 + price_change)
    
    # Create OHLC data
    high = current_price * (1 + random.uniform(0, 0.005))
    low = current_price * (1 - random.uniform(0, 0.005))
    open_price = last_price
    
    return MarketData(
        symbol=symbol,
        timestamp=datetime.now(),
        open=open_price,
        high=high,
        low=low,
        close=current_price,
        volume=random.uniform(1000, 10000),
        source="live_simulation"
    )

def show_demo_live_signals(duration_minutes=5):
    """Show demo live trading signals with proper data"""
    print("🎯 DEMO LIVE TRADING SIGNALS")
    print("=" * 50)
    print(f"Duration: {duration_minutes} minutes")
    print("Generating realistic market data and signals...\n")
    
    # Initialize system
    system = EnhancedTradingSystem()
    
    # Generate historical data for each symbol
    historical_data = {}
    for symbol in system.trading_pairs:
        print(f"📊 Generating historical data for {symbol}...")
        historical_data[symbol] = generate_historical_data(symbol, days=30)
        print(f"   ✅ Generated {len(historical_data[symbol])} data points")
    
    print("\n🚀 Starting live signal generation...\n")
    
    # Start data streams
    system.data_manager.start_data_streams(system.trading_pairs)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    signal_count = 0
    last_signal_time = {}
    
    try:
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            for symbol in system.trading_pairs:
                # Create live market data
                live_data = create_live_market_data(symbol, historical_data[symbol])
                
                # Store in data manager
                with system.data_manager.lock:
                    system.data_manager.data_streams[symbol] = live_data
                
                # Update historical data with new point
                new_row = {
                    'timestamp': live_data.timestamp,
                    'open': live_data.open,
                    'high': live_data.high,
                    'low': live_data.low,
                    'close': live_data.close,
                    'volume': live_data.volume
                }
                historical_data[symbol] = pd.concat([historical_data[symbol], pd.DataFrame([new_row])], ignore_index=True)
                
                # Use last 100 points for signal generation
                recent_data = historical_data[symbol].tail(100)
                
                # Generate signal
                signal = system.signal_generator.generate_signal(recent_data, symbol)
                
                if signal:
                    # Check for duplicate signals
                    signal_key = f"{symbol}_{signal.signal_type}_{signal.price:.2f}"
                    if signal_key not in last_signal_time or (current_time - last_signal_time[signal_key]).seconds > 30:
                        signal_count += 1
                        last_signal_time[signal_key] = current_time
                        
                        # Get additional analysis
                        ml_prediction = system.ml_predictor.predict_price_movement(recent_data, symbol)
                        sentiment = system.sentiment_analyzer.analyze_sentiment(symbol)
                        price_info = system.exchange_manager.get_best_price(symbol, signal.signal_type)
                        
                        # Display signal
                        print(f"🔔 LIVE SIGNAL #{signal_count} - {current_time.strftime('%H:%M:%S')}")
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
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo signals stopped by user")
    
    # Final summary
    print(f"\n📈 DEMO SIGNALS SUMMARY")
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
    print("🎯 DEMO LIVE TRADING SIGNALS")
    print("=" * 60)
    
    try:
        # Show demo signals for 5 minutes
        show_demo_live_signals(duration_minutes=5)
        
        print("\n✅ Demo signals demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 