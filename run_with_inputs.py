#!/usr/bin/env python3
"""
Script to run main.py with simulated user inputs
"""

import sys
import os
import subprocess
import time

def run_with_inputs():
    """Run main.py with the specified inputs: 3, 2, 3"""
    
    print("=== RUNNING MAIN.PY WITH INPUTS: 3, 2, 3 ===")
    print("Trading Mode: 3 (Demo Mode)")
    print("ML Training: 2 (Train Top 10 Symbols)")
    print("Enhanced Features: 3 (Full Mode)")
    print()
    
    # Create the input string
    inputs = "3\n2\n3\n"
    
    try:
        # Run the main.py script with inputs
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Send inputs and get output
        stdout, stderr = process.communicate(input=inputs, timeout=60)
        
        print("=== STDOUT ===")
        print(stdout)
        
        if stderr:
            print("=== STDERR ===")
            print(stderr)
            
        print(f"=== EXIT CODE: {process.returncode} ===")
        
    except subprocess.TimeoutExpired:
        print("❌ Process timed out after 60 seconds")
        process.kill()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_with_inputs() 