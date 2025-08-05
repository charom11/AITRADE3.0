#!/usr/bin/env python3
"""
Automated runner for the Unified Comprehensive Trading System
This script runs the main system with automated choices for faster execution.
"""

import subprocess
import sys
import os

def run_automated_system():
    """Run the main system with automated choices"""
    try:
        print("🤖 Starting Automated Trading System")
        print("="*50)
        print("📊 Mode: Paper Trading (Safe)")
        print("🎯 ML Training: Full training for all symbols")
        print("⏳ Starting system...")
        print("="*50)
        
        # Run the main system
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("✅ System completed successfully")
        else:
            print(f"❌ System exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping automated system...")
    except Exception as e:
        print(f"❌ Error running automated system: {e}")

if __name__ == "__main__":
    run_automated_system() 