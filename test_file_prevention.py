#!/usr/bin/env python3
"""
Test File Prevention System
Demonstrates how the file prevention system works
"""

import json
import time
from datetime import datetime
from file_manager import FileManager

def test_file_prevention():
    """Test the file prevention system"""
    print("🧪 TESTING FILE PREVENTION SYSTEM")
    print("=" * 50)
    
    # Initialize file manager
    file_manager = FileManager("test_data")
    
    # Test data
    test_data_1 = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "BTC/USDT",
        "price": 45000.0,
        "signal": "buy",
        "strength": 0.75
    }
    
    test_data_2 = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "ETH/USDT", 
        "price": 3000.0,
        "signal": "sell",
        "strength": 0.65
    }
    
    # Test 1: Create new file
    print("\n📝 Test 1: Creating new file")
    filename_1 = "test_signal_1.json"
    result = file_manager.save_file(filename_1, test_data_1)
    print(f"File created: {result}")
    
    # Test 2: Try to create same file with same content
    print("\n📝 Test 2: Creating same file with same content")
    result = file_manager.save_file(filename_1, test_data_1)
    print(f"File created: {result}")
    
    # Test 3: Create same file with different content
    print("\n📝 Test 3: Creating same file with different content")
    test_data_1_modified = test_data_1.copy()
    test_data_1_modified["price"] = 46000.0
    result = file_manager.save_file(filename_1, test_data_1_modified)
    print(f"File created: {result}")
    
    # Test 4: Create different file
    print("\n📝 Test 4: Creating different file")
    filename_2 = "test_signal_2.json"
    result = file_manager.save_file(filename_2, test_data_2)
    print(f"File created: {result}")
    
    # Test 5: Check file existence
    print("\n📝 Test 5: Checking file existence")
    exists_1 = file_manager.file_exists(filename_1)
    exists_2 = file_manager.file_exists(filename_2)
    print(f"File {filename_1} exists: {exists_1}")
    print(f"File {filename_2} exists: {exists_2}")
    
    # Test 6: Load files
    print("\n📝 Test 6: Loading files")
    loaded_data_1 = file_manager.load_file(filename_1)
    loaded_data_2 = file_manager.load_file(filename_2)
    print(f"Loaded data 1: {loaded_data_1['price']}")
    print(f"Loaded data 2: {loaded_data_2['price']}")
    
    # Test 7: Get file info
    print("\n📝 Test 7: Getting file information")
    info_1 = file_manager.get_file_info(filename_1)
    info_2 = file_manager.get_file_info(filename_2)
    print(f"File 1 info: {info_1['filename']} - Size: {info_1['size']} bytes")
    print(f"File 2 info: {info_2['filename']} - Size: {info_2['size']} bytes")
    
    # Test 8: Get file list
    print("\n📝 Test 8: Getting file list")
    file_list = file_manager.get_file_list()
    print(f"Files in directory: {file_list}")
    
    # Test 9: Force save (overwrite)
    print("\n📝 Test 9: Force saving (overwrite)")
    test_data_1_force = test_data_1.copy()
    test_data_1_force["price"] = 47000.0
    result = file_manager.save_file(filename_1, test_data_1_force, force=True)
    print(f"Force save result: {result}")
    
    # Test 10: Save file hashes
    print("\n📝 Test 10: Saving file hashes")
    file_manager.save_file_hashes()
    print("File hashes saved successfully")
    
    print("\n✅ File prevention system test completed!")
    print("=" * 50)

def test_trading_signals():
    """Test with trading signals"""
    print("\n🚀 TESTING WITH TRADING SIGNALS")
    print("=" * 50)
    
    file_manager = FileManager("trading_test_data")
    
    # Simulate trading signals
    signals = []
    for i in range(5):
        signal = {
            "timestamp": datetime.now().isoformat(),
            "symbol": f"COIN{i}/USDT",
            "signal_type": "buy" if i % 2 == 0 else "sell",
            "strength": 0.5 + (i * 0.1),
            "confidence": 0.6 + (i * 0.05),
            "price": 100.0 + (i * 10),
            "conditions": [f"Condition {j}" for j in range(3)]
        }
        signals.append(signal)
    
    # Save signals with prevention
    for i, signal in enumerate(signals):
        filename = f"signal_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result = file_manager.save_file(filename, signal)
        print(f"Signal {i+1} saved: {result}")
        time.sleep(1)  # Small delay to ensure different timestamps
    
    # Try to save same signals again
    print("\n🔄 Trying to save same signals again...")
    for i, signal in enumerate(signals):
        filename = f"signal_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result = file_manager.save_file(filename, signal)
        print(f"Signal {i+1} duplicate save: {result}")
    
    print("\n✅ Trading signals test completed!")

if __name__ == "__main__":
    test_file_prevention()
    test_trading_signals() 