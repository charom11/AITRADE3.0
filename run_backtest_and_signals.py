#!/usr/bin/env python3
"""
Backtest and Live Signals Demo
Runs comprehensive backtest and shows live trading signals
"""

import time
import json
from datetime import datetime, timedelta
from enhanced_trading_system import EnhancedTradingSystem
from file_manager import FileManager

def run_comprehensive_backtest():
    """Run comprehensive backtest on multiple symbols"""
    print("🚀 RUNNING COMPREHENSIVE BACKTEST")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedTradingSystem()
    file_manager = FileManager("backtest_results")
    
    # Test symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n📊 Backtesting {symbol}...")
        
        # Run backtest
        results = system.run_backtest(symbol, "2024-01-01", "2024-12-31")
        
        if results:
            all_results[symbol] = results
            
            # Display results
            print(f"✅ {symbol} Backtest Results:")
            print(f"   Initial Capital: ${results.get('initial_capital', 0):,.2f}")
            print(f"   Final Capital: ${results.get('final_capital', 0):,.2f}")
            print(f"   Total Return: {results.get('total_return', 0)*100:.2f}%")
            print(f"   Total Trades: {results.get('total_trades', 0)}")
            print(f"   Win Rate: {results.get('win_rate', 0)*100:.2f}%")
            print(f"   Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
        else:
            print(f"❌ Failed to run backtest for {symbol}")
    
    # Save comprehensive results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_symbols': len(symbols),
        'results': all_results,
        'overall_performance': {
            'avg_return': sum(r.get('total_return', 0) for r in all_results.values()) / len(all_results),
            'avg_win_rate': sum(r.get('win_rate', 0) for r in all_results.values()) / len(all_results),
            'total_trades': sum(r.get('total_trades', 0) for r in all_results.values())
        }
    }
    
    filename = f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_manager.save_file(filename, summary)
    
    print(f"\n📈 COMPREHENSIVE BACKTEST COMPLETED")
    print(f"   Average Return: {summary['overall_performance']['avg_return']*100:.2f}%")
    print(f"   Average Win Rate: {summary['overall_performance']['avg_win_rate']*100:.2f}%")
    print(f"   Total Trades: {summary['overall_performance']['total_trades']}")
    print(f"   Results saved to: {filename}")
    
    return system

def show_live_signals(system, duration_minutes=5):
    """Show live trading signals for specified duration"""
    print(f"\n🎯 SHOWING LIVE TRADING SIGNALS")
    print("=" * 50)
    print(f"Duration: {duration_minutes} minutes")
    print("Press Ctrl+C to stop early\n")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    signal_count = 0
    
    try:
        while datetime.now() < end_time:
            # Get system status
            status = system.get_system_status()
            
            # Check for new signals
            for symbol in system.trading_pairs:
                # Get latest data
                market_data = system.data_manager.get_latest_data(symbol)
                
                if market_data:
                    # Convert to DataFrame for analysis
                    import pandas as pd
                    from dataclasses import asdict
                    df = pd.DataFrame([asdict(market_data)])
                    
                    # Generate signal
                    signal = system.signal_generator.generate_signal(df, symbol)
                    
                    if signal:
                        signal_count += 1
                        
                        # Get additional analysis
                        ml_prediction = system.ml_predictor.predict_price_movement(df, symbol)
                        sentiment = system.sentiment_analyzer.analyze_sentiment(symbol)
                        
                        # Display signal
                        print(f"🔔 SIGNAL #{signal_count} - {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   Symbol: {signal.symbol}")
                        print(f"   Type: {signal.signal_type.upper()}")
                        print(f"   Price: ${signal.price:,.2f}")
                        print(f"   Strength: {signal.strength:.2f}")
                        print(f"   Confidence: {signal.confidence:.2f}")
                        print(f"   Risk Score: {signal.risk_score:.2f}")
                        print(f"   Stop Loss: ${signal.stop_loss:,.2f}")
                        print(f"   Take Profit: ${signal.take_profit:,.2f}")
                        print(f"   ML Prediction: {ml_prediction.get('prediction', 0):.3f}")
                        print(f"   Sentiment: {sentiment.get('market_mood', 'neutral')}")
                        print(f"   Conditions: {', '.join(signal.conditions[:3])}")
                        
                        # Risk check
                        if system.risk_manager.should_trade(signal):
                            position_size = system.risk_manager.calculate_position_size(
                                signal, system.risk_manager.portfolio_value
                            )
                            print(f"   ✅ TRADE APPROVED - Position Size: {position_size:.4f}")
                        else:
                            print(f"   ❌ TRADE REJECTED - Risk management")
                        
                        print("-" * 40)
            
            # Show system status every 30 seconds
            if signal_count % 6 == 0:
                print(f"📊 System Status - Active Positions: {status.get('active_positions', 0)}")
                print(f"   Portfolio Value: ${status.get('risk_metrics', {}).get('portfolio_value', 0):,.2f}")
                print(f"   Risk Level: {status.get('risk_metrics', {}).get('risk_level', 'unknown')}")
                print()
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Live signals stopped by user")
    
    print(f"\n📈 LIVE SIGNALS SUMMARY")
    print(f"   Total Signals Generated: {signal_count}")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Average Signals/Minute: {signal_count/duration_minutes:.1f}")

def main():
    """Main function"""
    print("🎯 ENHANCED TRADING SYSTEM - BACKTEST & LIVE SIGNALS")
    print("=" * 60)
    
    try:
        # Run comprehensive backtest
        system = run_comprehensive_backtest()
        
        # Show live signals
        show_live_signals(system, duration_minutes=3)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 