#!/usr/bin/env python3
"""
Test script for ML training functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MLModelTrainer

def create_sample_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days*24, freq='H')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    
    # Random walk with trend
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    })
    
    return data

def test_ml_training():
    """Test ML model training"""
    print("🤖 Testing ML Model Training...")
    
    # Create ML trainer
    trainer = MLModelTrainer()
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        # Create sample data
        data = create_sample_data(symbol, days=30)
        print(f"   Generated {len(data)} data points")
        
        # Test feature preparation
        features = trainer.prepare_features(data)
        print(f"   Prepared {len(features.columns)} features")
        
        # Test training data preparation
        X, y = trainer.prepare_training_data(data, symbol)
        print(f"   Training data: {len(X)} samples, {len(y)} targets")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # Test model training (only for one model to speed up testing)
        if len(X) >= 50:  # Only train if we have enough data
            print(f"   Training models...")
            
            # Train only gradient boosting for testing
            results = trainer.train_models(symbol, data)
            
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                print(f"   ✅ Training successful!")
                print(f"   🏆 Best model: {best_model[0]}")
                print(f"   📈 F1 Score: {best_model[1]['f1_score']:.3f}")
                print(f"   🎯 Accuracy: {best_model[1]['accuracy']:.3f}")
                
                # Test prediction
                prediction = trainer.predict(symbol, data.tail(10))
                print(f"   🔮 Prediction: {prediction}")
            else:
                print(f"   ❌ Training failed")
        else:
            print(f"   ⚠️ Insufficient data for training")
    
    # Test training summary
    summary = trainer.get_training_summary()
    print(f"\n📊 Training Summary:")
    print(f"   Total symbols trained: {summary.get('total_symbols_trained', 0)}")
    print(f"   Total models trained: {summary.get('total_models_trained', 0)}")
    
    print("\n✅ ML Training Test Complete!")

if __name__ == "__main__":
    test_ml_training() 