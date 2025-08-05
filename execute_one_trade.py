#!/usr/bin/env python3
"""
Execute One Trade - Immediate Trade Execution
Forces execution of one trade with aggressive settings
"""
import sys
import os
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from integrated_futures_trading_system import IntegratedFuturesTradingSystem, get_top_100_futures_pairs

def execute_one_trade():
    print("🎯 EXECUTE ONE TRADE - IMMEDIATE EXECUTION")
    print("="*60)
    print("⚠️  AGGRESSIVE SETTINGS FOR IMMEDIATE TRADE")
    print("📊 Lowered confidence threshold to 0.4")
    print("🎯 Targeting first available signal")
    print("="*60)
    
    # Pre-configured credentials
    TELEGRAM_BOT_TOKEN = "8201084480:AAEsc-cLl8KIelwV7PffT4cGoclZquRpFak"
    BINANCE_API_KEY = "QVa0FqEvAtf1s5vtQE0grudNUkg3sl0IIPcdFx99cW50Cb80gPwHexW9VPGk7h0y"
    BINANCE_SECRET_KEY = "9A8hpWaTRvnaEApeCCwl7in0FvTBPIdFqXO4zidYugJgXXA9FO6TWMU3kn4JKgb0"
    TELEGRAM_CHAT_ID = "1166227057"
    
    print("🔍 Fetching top 10 futures pairs...")
    all_pairs = get_top_100_futures_pairs()
    symbols = all_pairs[:10]  # Use top 10 for faster execution
    print(f"📊 Using top 10 pairs: {', '.join(symbols)}")
    
    # Create system with aggressive settings
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_SECRET_KEY,
        enable_file_output=True,
        enable_live_trading=True,  # LIVE TRADING
        paper_trading=False,
        max_position_size=0.10,    # 10% (increased for one trade)
        risk_per_trade=0.02,       # 2% (increased for one trade)
        enable_backtesting=False,
        enable_alerts=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        max_concurrent_trades=1,   # Only 1 trade
        max_portfolio_risk=0.10
    )
    
    print("🔔 Testing Telegram connection...")
    if system.test_telegram_connection():
        print("✅ Telegram connection successful!")
    else:
        print("❌ Telegram connection failed!")
        return
    
    # Send startup message
    start_message = f"""
🚨 **EXECUTE ONE TRADE - IMMEDIATE MODE**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: Aggressive Trade Execution
💪 **Mode**: Single Trade Target

📊 **Monitoring**: {len(symbols)} symbols
🎯 **Target**: Execute 1 trade immediately
💰 **Position Size**: 10% of account
⚠️ **Risk Per Trade**: 2% of account
📈 **Max Trades**: 1 concurrent

📈 **Top Symbols**: {', '.join(symbols[:5])}
⏰ **Start Time**: {datetime.now().strftime('%H:%M:%S')}

✅ **Aggressive Settings:**
• Lowered confidence threshold to 40%
• Increased position size to 10%
• Single trade target
• Immediate execution mode

⚠️ **WARNING**: Real money will be used!
"""
    system.send_telegram_alert(start_message)
    
    print("🚀 Starting aggressive trade execution...")
    print("🎯 Looking for first signal with confidence >40%...")
    
    trade_executed = False
    attempts = 0
    max_attempts = 20  # Try for 20 cycles (10 minutes)
    
    try:
        while not trade_executed and attempts < max_attempts:
            attempts += 1
            print(f"🔄 Attempt {attempts}/{max_attempts} - Scanning for signals...")
            
            # Process each symbol
            for symbol in symbols:
                if trade_executed:
                    break
                    
                # Fetch current data
                market_data = system._fetch_futures_data(symbol)
                if not market_data:
                    continue
                
                # Update historical data
                system.historical_data[symbol].append(market_data)
                if len(system.historical_data[symbol]) > 100:
                    system.historical_data[symbol] = system.historical_data[symbol][-100:]
                
                # Fetch historical data for analysis
                historical_df = system._fetch_historical_data(symbol, limit=100)
                if historical_df is None:
                    continue
                
                # Generate component signals
                futures_signal = system._generate_futures_signal(symbol, market_data)
                support_resistance_signal = system._generate_support_resistance_signal(symbol, historical_df)
                divergence_signal = system._generate_divergence_signal(symbol, historical_df)
                strategy_signals = system._generate_strategy_signals(symbol, historical_df)
                
                # Combine signals
                integrated_signal = system._combine_signals(
                    symbol, market_data, futures_signal, 
                    support_resistance_signal, divergence_signal, strategy_signals
                )
                
                if integrated_signal and integrated_signal.confidence > 0.4:
                    print(f"🎯 SIGNAL FOUND: {symbol} - Confidence: {integrated_signal.confidence:.3f}")
                    
                    # Analyze market conditions
                    market_condition = system.analyze_market_conditions(symbol, historical_df)
                    
                    # Execute trade immediately
                    if system._validate_signal_with_market_conditions(integrated_signal, market_condition):
                        print(f"🚀 EXECUTING TRADE: {symbol} {integrated_signal.signal_type}")
                        
                        trade_executed = system.execute_trade(integrated_signal)
                        if trade_executed:
                            print(f"✅ TRADE EXECUTED SUCCESSFULLY: {symbol}")
                            
                            # Send execution alert
                            execution_alert = f"""
🚨 **TRADE EXECUTED - IMMEDIATE MODE**

📊 **Symbol**: {symbol}
📈 **Signal**: {integrated_signal.signal_type}
💪 **Confidence**: {integrated_signal.confidence:.3f}
💰 **Entry Price**: ${integrated_signal.price:.4f}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

🎯 **Risk Management:**
🛑 **Stop Loss**: ${integrated_signal.stop_loss:.4f}
🎯 **Take Profit**: ${integrated_signal.take_profit:.4f}
⚡ **Leverage**: {integrated_signal.leverage_suggestion}x

✅ **Status**: Trade executed successfully
🔧 **Mode**: Immediate Execution
🎯 **Target**: 1 trade completed
"""
                            system.send_telegram_alert(execution_alert)
                            
                            # Wait 30 seconds then stop
                            print("⏳ Waiting 30 seconds before stopping...")
                            time.sleep(30)
                            break
                        else:
                            print(f"❌ TRADE FAILED: {symbol}")
                    else:
                        print(f"⚠️ SIGNAL REJECTED: Market validation failed for {symbol}")
            
            if not trade_executed:
                print("⏳ No suitable signals found, waiting 30 seconds...")
                time.sleep(30)
        
        if not trade_executed:
            print("❌ No trades executed within time limit")
            no_trade_alert = f"""
⚠️ **NO TRADE EXECUTED**

📊 **System**: Integrated Futures Trading System
🎯 **Status**: No suitable signals found
⏰ **Duration**: {max_attempts * 30} seconds
📊 **Symbols Scanned**: {len(symbols)}

🔧 **Settings Used:**
• Confidence threshold: 40%
• Position size: 10%
• Risk per trade: 2%

⏰ **End Time**: {datetime.now().strftime('%H:%M:%S')}

No trades met the execution criteria.
"""
            system.send_telegram_alert(no_trade_alert)
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping trade execution...")
    except Exception as e:
        print(f"❌ Error: {e}")
        error_alert = f"""
🚨 **TRADE EXECUTION ERROR**

❌ **Error**: {str(e)}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}
System stopped due to error.
"""
        system.send_telegram_alert(error_alert)
    finally:
        system.stop()
        print("✅ Trade execution stopped!")

if __name__ == "__main__":
    execute_one_trade() 