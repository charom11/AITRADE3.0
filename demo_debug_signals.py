#!/usr/bin/env python3
"""
Debug Signals Demonstration
Shows the debug signals (echo "3", echo "2") in action with the enhanced trading system.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def demo_debug_signals():
    """Demonstrate the debug signals in action"""
    print("🎯 Debug Signals Demonstration")
    print("="*50)
    print("📊 This will show the debug signals (echo '3', echo '2') in action")
    print("⏳ Running for 30 seconds to demonstrate...")
    print()
    
    try:
        # Start the main.py process
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Send demo mode input
        demo_input = "3\n1\n"  # Demo mode, skip ML training
        
        print("🚀 Starting main.py in demo mode...")
        print("📋 Input sent: Demo mode (3), Skip ML training (1)")
        print()
        
        # Run for 30 seconds
        start_time = time.time()
        duration = 30
        
        stdout, stderr = process.communicate(input=demo_input, timeout=duration)
        
        print("✅ Demo completed!")
        print(f"⏱️ Duration: {time.time() - start_time:.1f} seconds")
        print()
        
        # Analyze output for debug signals
        lines = stdout.split('\n')
        debug_signals_found = []
        
        for i, line in enumerate(lines):
            if line.strip() in ['3', '2']:
                debug_signals_found.append(f"Line {i+1}: '{line.strip()}'")
        
        print("🎯 Debug Signal Analysis:")
        print("="*30)
        
        if debug_signals_found:
            print("✅ Debug signals detected:")
            for signal in debug_signals_found:
                print(f"  • {signal}")
            print(f"📊 Total debug signals: {len(debug_signals_found)}")
        else:
            print("❌ No debug signals detected in output")
            print("💡 This might be due to the system not reaching the trading loop in demo mode")
        
        print()
        print("📋 System Output Analysis:")
        print("="*30)
        
        # Check for key system messages
        system_messages = {
            'System Started': 'Unified Comprehensive Trading System' in stdout,
            'Demo Mode': 'Demo Mode' in stdout,
            'Enhanced Models': 'enhanced_btc_eth_models' in stdout or 'Enhanced Models' in stdout,
            'Trading Loop': 'Trading cycle' in stdout,
            'Debug Signals': len(debug_signals_found) > 0
        }
        
        for message, found in system_messages.items():
            status = "✅" if found else "❌"
            print(f"{status} {message}: {found}")
        
        print()
        print("🔍 Raw Output Preview (first 20 lines):")
        print("="*40)
        for i, line in enumerate(lines[:20]):
            if line.strip():
                print(f"{i+1:2d}: {line}")
        
        if len(lines) > 20:
            print(f"... and {len(lines) - 20} more lines")
        
        return {
            'debug_signals_found': len(debug_signals_found),
            'system_started': system_messages['System Started'],
            'demo_mode': system_messages['Demo Mode'],
            'enhanced_models': system_messages['Enhanced Models'],
            'trading_loop': system_messages['Trading Loop'],
            'duration': time.time() - start_time
        }
        
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out (expected)")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return {'status': 'error', 'error': str(e)}

def show_debug_implementation():
    """Show where debug signals are implemented"""
    print("\n🔧 Debug Signal Implementation")
    print("="*40)
    
    print("📍 Location: main.py - trading_loop() method")
    print("📝 Implementation:")
    print()
    print("```python")
    print("def trading_loop(self):")
    print("    \"\"\"Main trading loop\"\"\"")
    print("    try:")
    print("        cycle = 1")
    print("        ")
    print("        while self.running:")
    print("            try:")
    print("                logger.info(f\"Trading cycle #{cycle}\")")
    print("                ")
    print("                # Debug signals as requested")
    print("                print(\"3\")")
    print("                print(\"2\")")
    print("                ")
    print("                # Generate unified signals")
    print("                signals = self.generate_unified_signals()")
    print("                # ... rest of trading logic")
    print("```")
    print()
    print("✅ Debug signals are printed at the beginning of each trading cycle")
    print("🔄 This ensures continuous monitoring and verification")

def main():
    """Main demonstration function"""
    print("🚀 Enhanced Trading System - Debug Signals Demo")
    print("="*60)
    
    # Show implementation
    show_debug_implementation()
    
    # Run demo
    print("\n🎯 Running Debug Signals Demo...")
    results = demo_debug_signals()
    
    # Summary
    print("\n📊 Demo Summary:")
    print("="*20)
    print(f"🎯 Debug signals found: {results.get('debug_signals_found', 0)}")
    print(f"🚀 System started: {results.get('system_started', False)}")
    print(f"📋 Demo mode: {results.get('demo_mode', False)}")
    print(f"🤖 Enhanced models: {results.get('enhanced_models', False)}")
    print(f"🔄 Trading loop: {results.get('trading_loop', False)}")
    print(f"⏱️ Duration: {results.get('duration', 0):.1f} seconds")
    
    print("\n✅ Debug signals demonstration completed!")
    print("📝 Check the output above to see the debug signals in action")

if __name__ == "__main__":
    main()