#!/usr/bin/env python3
"""
Direct Training Run - Bypass Interactive Input
Configure system directly with Demo Mode (3) + Train Top 10 (2) + Full Enhanced (3)
"""

import sys
import os
from datetime import datetime

# Add current directory to path to import main modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_direct_training():
    """Run the system directly with the requested configuration"""
    print("=" * 80)
    print("DIRECT TRAINING RUN - UNIFIED COMPREHENSIVE TRADING SYSTEM")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("CONFIGURATION:")
    print("- Trading Mode: Demo Mode (3)")
    print("- ML Training: Train Top 10 Symbols (2)")
    print("- Enhanced Features: Full Mode (3)")
    print()
    
    try:
        # Import the main system
        from main import UnifiedComprehensiveTradingSystem
        
        print("Initializing system...")
        system = UnifiedComprehensiveTradingSystem()
        
        # Configure Demo Mode (3)
        print("Setting Demo Mode...")
        system.enable_live_trading = False
        system.paper_trading = False
        system.futures_system.enable_live_trading = False
        system.futures_system.paper_trading = False
        print("✅ Demo Mode: ENABLED (No actual trading)")
        
        # Configure ML Training for Top 10 Symbols (2)
        print("Configuring ML Training for Top 10 Symbols...")
        priority_symbols = ['BTCUSDT', 'ETHUSDT']
        other_symbols = [s for s in system.symbols[:8] if s not in priority_symbols]
        training_symbols = priority_symbols + other_symbols
        system.symbols = training_symbols[:10]
        print(f"✅ Priority symbols: {priority_symbols}")
        print(f"✅ Other symbols: {other_symbols[:8]}")
        print(f"✅ Total training symbols: {len(system.symbols)}")
        
        # Configure Full Enhanced Features (3)
        print("Enabling Full Enhanced Features...")
        system.enhanced_features_enabled = True
        system.sentiment_analysis_enabled = True
        system.news_analysis_enabled = True
        system.social_media_enabled = True
        system.advanced_backtesting_enabled = True
        system.security_auditing_enabled = True
        system.strategy_optimization_enabled = True
        system.advanced_visualization_enabled = True
        print("✅ All Enhanced Features: ENABLED")
        
        # Display system status
        print("\nSystem Status Summary:")
        status_summary = system.get_system_status_summary()
        
        print(f"System Status: {status_summary.get('system_status', 'Unknown')}")
        print(f"Trading Mode: {status_summary.get('trading_mode', 'Unknown')}")
        print(f"Active Symbols: {status_summary.get('active_symbols', 0)}")
        print(f"ML Models: {status_summary.get('ml_models', 0)}")
        print(f"ML Accuracy: {status_summary.get('ml_accuracy', 0):.1f}%")
        
        # Show enhanced features status
        print("\nEnhanced Features Status:")
        available_features = status_summary.get('available_features', {})
        
        features_status = [
            ("Sentiment Analysis", available_features.get('sentiment_analysis', False)),
            ("News Analysis", available_features.get('news_analysis', False)),
            ("Social Media", available_features.get('social_media', False)),
            ("Plotly/Dash", available_features.get('plotly_dash', False)),
            ("TensorFlow", available_features.get('tensorflow', False)),
            ("PyTorch", available_features.get('pytorch', False)),
            ("Optuna", available_features.get('optuna', False)),
            ("SHAP", available_features.get('shap', False)),
            ("MLflow", available_features.get('mlflow', False)),
            ("TA Library", available_features.get('ta_library', False))
        ]
        
        for feature, available in features_status:
            status_icon = "[OK]" if available else "[X]"
            print(f"  {status_icon} {feature}")
        
        print("\nStarting system...")
        print("=" * 80)
        
        # Start the system
        system.start_system()
        
    except KeyboardInterrupt:
        print("\nStopping system...")
        if 'system' in locals():
            system.stop_system()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_direct_training() 