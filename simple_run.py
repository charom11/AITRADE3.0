#!/usr/bin/env python3
"""
Simple script to run the main function
"""

import sys
import os
import traceback

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_run():
    """Simple run of the main function"""
    
    print("=== SIMPLE RUN ===")
    
    try:
        # Import and run main function
        print("Importing main function...")
        from main import main
        
        print("Running main function...")
        main()
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    simple_run() 