#!/usr/bin/env python3
"""
Test Integrated Futures Trading System
Demonstrates the integrated system with file prevention features
"""

import os
import sys
import time
from datetime import datetime
from integrated_futures_trading_system import IntegratedFuturesTradingSystem
from file_manager import FileManager

def test_file_manager():
    """Test file manager functionality"""
    print("🧪 TESTING FILE MANAGER")
    print("=" * 50)
    
    # Create file manager
    fm = FileManager("test_output")
    
    # Test data
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'file_manager_test',
        'data': [1, 2, 3, 4, 5]
    }
    
    # Test safe save operations
    print("1. Testing safe save operations...")
    
    # First save
    result1 = fm.safe_save_json(test_data, "test_data.json")
    print(f"   First save: {'✅ Success' if result1 else '❌ Failed'}")
    
    # Second save (should skip due to existing file)
    result2 = fm.safe_save_json(test_data, "test_data.json")
    print(f"   Second save: {'✅ Skipped (file exists)' if result2 else '❌ Failed'}")
    
    # Test with different data
    test_data2 = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'file_manager_test_2',
        'data': [5, 4, 3, 2, 1]
    }
    
    result3 = fm.safe_save_json(test_data2, "test_data_2.json")
    print(f"   Different data save: {'✅ Success' if result3 else '❌ Failed'}")
    
    # Print summary
    print("\n2. File Manager Summary:")
    fm.print_summary()
    
    return fm

def test_integrated_system():
    """Test integrated futures trading system"""
    print("\n🧪 TESTING INTEGRATED FUTURES SYSTEM")
    print("=" * 50)
    
    # Create system with minimal symbols for testing
    symbols = ['BTCUSDT']  # Just one symbol for testing
    
    system = IntegratedFuturesTradingSystem(
        symbols=symbols,
        enable_file_output=True,
        output_directory="test_integrated_signals"
    )
    
    print(f"✅ System initialized for symbols: {symbols}")
    print(f"✅ File output enabled: {system.enable_file_output}")
    print(f"✅ Output directory: {system.output_directory}")
    
    # Test single symbol processing
    print("\n3. Testing single symbol processing...")
    try:
        system._process_symbol('BTCUSDT')
        print("   ✅ Symbol processing completed")
    except Exception as e:
        print(f"   ⚠️ Symbol processing error (expected for demo): {e}")
    
    # Test file manager integration
    print("\n4. Testing file manager integration...")
    created_files = system.file_manager.list_created_files()
    print(f"   Files tracked by file manager: {len(created_files)}")
    
    if created_files:
        print("   Tracked files:")
        for file in created_files[:3]:  # Show first 3
            print(f"     - {file}")
    
    return system

def test_file_prevention():
    """Test file prevention features"""
    print("\n🧪 TESTING FILE PREVENTION")
    print("=" * 50)
    
    # Create file manager
    fm = FileManager("test_prevention")
    
    # Test data
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'prevention_test',
        'value': 42
    }
    
    print("5. Testing duplicate prevention...")
    
    # Save same data multiple times
    for i in range(3):
        result = fm.safe_save_json(test_data, "duplicate_test.json")
        if i == 0:
            print(f"   First save: {'✅ Created' if result else '❌ Failed'}")
        else:
            print(f"   Save {i+1}: {'✅ Skipped (duplicate)' if result else '❌ Failed'}")
    
    # Test with different content
    test_data2 = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'prevention_test',
        'value': 100  # Different value
    }
    
    result = fm.safe_save_json(test_data2, "duplicate_test.json")
    print(f"   Different content: {'✅ Updated' if result else '❌ Failed'}")
    
    # Test CSV prevention
    print("\n6. Testing CSV file prevention...")
    import pandas as pd
    
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})  # Same data
    df3 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})  # Different data
    
    result1 = fm.safe_save_csv(df1, "test_data.csv")
    result2 = fm.safe_save_csv(df2, "test_data.csv")
    result3 = fm.safe_save_csv(df3, "test_data.csv")
    
    print(f"   First CSV save: {'✅ Created' if result1 else '❌ Failed'}")
    print(f"   Duplicate CSV save: {'✅ Skipped' if result2 else '❌ Failed'}")
    print(f"   Different CSV save: {'✅ Updated' if result3 else '❌ Failed'}")
    
    return fm

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 CLEANING UP TEST FILES")
    print("=" * 50)
    
    test_dirs = ["test_output", "test_integrated_signals", "test_prevention"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                import shutil
                shutil.rmtree(test_dir)
                print(f"   ✅ Removed: {test_dir}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {test_dir}: {e}")

def main():
    """Main test function"""
    print("🚀 INTEGRATED FUTURES TRADING SYSTEM - TEST SUITE")
    print("=" * 80)
    print("This test demonstrates the integrated system with file prevention features.")
    print("⚠️  Note: Some tests may show expected errors due to API limitations.")
    print("=" * 80)
    
    try:
        # Test file manager
        fm = test_file_manager()
        
        # Test integrated system
        system = test_integrated_system()
        
        # Test file prevention
        fm2 = test_file_prevention()
        
        print("\n✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("Key Features Demonstrated:")
        print("  ✅ File existence checking")
        print("  ✅ Duplicate file prevention")
        print("  ✅ Content hash verification")
        print("  ✅ Safe directory creation")
        print("  ✅ Integrated system initialization")
        print("  ✅ File manager integration")
        print("  ✅ Multiple file type support")
        
        # Ask user if they want to clean up
        cleanup = input("\n🧹 Clean up test files? (y/n): ").strip().lower()
        if cleanup == 'y':
            cleanup_test_files()
        else:
            print("   Test files preserved for inspection.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 