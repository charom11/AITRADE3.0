#!/usr/bin/env python3
"""
Minimal test to isolate the issue
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def minimal_test():
    """Minimal test to isolate the issue"""
    
    print("=== MINIMAL TEST ===")
    
    try:
        # Import the system class
        print("1. Importing system class...")
        from main import UnifiedComprehensiveTradingSystem
        print("✅ System class imported")
        
        # Create system instance
        print("2. Creating system instance...")
        system = UnifiedComprehensiveTradingSystem()
        print("✅ System instance created")
        
        # Test basic attributes
        print("3. Testing basic attributes...")
        print(f"Symbols: {len(system.symbols)}")
        print(f"Running: {system.running}")
        print(f"Signals: {len(system.signals)}")
        print(f"Positions: {len(system.positions)}")
        print("✅ Basic attributes OK")
        
        # Test ML status
        print("4. Testing ML status...")
        try:
            ml_status = system.get_ml_training_status()
            print(f"ML Status: {ml_status}")
            print("✅ ML status OK")
        except Exception as e:
            print(f"❌ ML status error: {e}")
        
        # Test system status summary
        print("5. Testing system status summary...")
        try:
            status = system.get_system_status_summary()
            print(f"Status: {status}")
            print("✅ System status summary OK")
        except Exception as e:
            print(f"❌ System status summary error: {e}")
            import traceback
            traceback.print_exc()
        
        print("=== TEST COMPLETED ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_test() 