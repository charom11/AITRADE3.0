#!/usr/bin/env python3
"""
Debug script to test the main function step by step
"""

import sys
import os
from io import StringIO

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_main():
    """Debug the main function step by step"""
    
    print("=== DEBUG: Starting main function test ===")
    
    try:
        # Import the main function
        print("1. Importing main function...")
        from main import main, UnifiedComprehensiveTradingSystem
        print("✅ Main function imported successfully")
        
        # Test system initialization
        print("2. Testing system initialization...")
        system = UnifiedComprehensiveTradingSystem()
        print("✅ System initialized successfully")
        
        # Test system status summary
        print("3. Testing system status summary...")
        status = system.get_system_status_summary()
        print("✅ System status summary retrieved")
        print(f"Status: {status}")
        
        # Test with inputs
        print("4. Testing with inputs...")
        input_string = "3\n2\n3\n"  # Demo Mode, Train Top 10, Full Mode
        
        # Redirect stdin
        original_stdin = sys.stdin
        sys.stdin = StringIO(input_string)
        
        try:
            # Call main function
            print("5. Calling main function...")
            main()
            print("✅ Main function completed")
        except Exception as e:
            print(f"❌ Error in main function: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdin = original_stdin
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main() 