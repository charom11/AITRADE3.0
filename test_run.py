#!/usr/bin/env python3
"""
Test script to run the main system with specific inputs
"""

import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_main_with_inputs():
    """Test the main function with specific inputs"""
    
    # Create input string with the specified choices
    input_string = "3\n2\n3\n"  # Demo Mode, Train Top 10, Full Mode
    
    # Redirect stdin to provide inputs
    original_stdin = sys.stdin
    sys.stdin = StringIO(input_string)
    
    try:
        # Import and run main function
        from main import main
        
        # Capture output
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            main()
        
        # Get the output
        result = output.getvalue()
        print("=== SYSTEM OUTPUT ===")
        print(result)
        
    except Exception as e:
        print(f"Error running main: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdin
        sys.stdin = original_stdin

if __name__ == "__main__":
    test_main_with_inputs() 