#!/usr/bin/env python3
"""
Performance Tracking and Version Control System
Tracks performance metrics and maintains version control for trading system iterations.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

class PerformanceTracker:
    """Performance tracking and version control system"""
    
    def __init__(self, db_path: str = "performance_tracker.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the performance tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    f1_score REAL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    total_pnl REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    notes TEXT
                )
            ''')
            
            # Create version control table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_control (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    description TEXT,
                    changes_made TEXT,
                    performance_improvement REAL,
                    status TEXT DEFAULT 'active',
                    priority_symbols TEXT,
                    models_trained INTEGER,
                    training_duration REAL
                )
            ''')
            
            # Create iteration tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS iteration_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_number INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    training_results TEXT,
                    evaluation_results TEXT,
                    improvements_made TEXT,
                    next_iteration_planned TEXT,
                    status TEXT DEFAULT 'completed'
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print(f"✅ Performance tracking database initialized: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def log_performance(self, version: str, symbol: str, model_type: str, 
                       metrics: Dict[str, float], notes: str = "") -> bool:
        """Log performance metrics for a specific version and symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_logs 
                (version, timestamp, symbol, model_type, f1_score, accuracy, 
                 precision, recall, win_rate, total_trades, profitable_trades,
                 total_pnl, max_drawdown, sharpe_ratio, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version,
                datetime.now().isoformat(),
                symbol,
                model_type,
                metrics.get('f1_score', 0.0),
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_trades', 0),
                metrics.get('profitable_trades', 0),
                metrics.get('total_pnl', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                notes
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Performance logged for {symbol} ({model_type}) - Version: {version}")
            return True
            
        except Exception as e:
            print(f"❌ Error logging performance: {e}")
            return False
    
    def create_version(self, version: str, description: str, changes_made: str,
                      priority_symbols: List[str] = None, models_trained: int = 0,
                      training_duration: float = 0.0) -> bool:
        """Create a new version entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO version_control 
                (version, timestamp, description, changes_made, priority_symbols,
                 models_trained, training_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                version,
                datetime.now().isoformat(),
                description,
                changes_made,
                json.dumps(priority_symbols) if priority_symbols else "[]",
                models_trained,
                training_duration
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Version {version} created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating version: {e}")
            return False
    
    def log_iteration(self, iteration_number: int, training_results: Dict,
                     evaluation_results: Dict, improvements_made: str,
                     next_iteration_planned: str = "") -> bool:
        """Log a training iteration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO iteration_tracking 
                (iteration_number, timestamp, training_results, evaluation_results,
                 improvements_made, next_iteration_planned)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                iteration_number,
                datetime.now().isoformat(),
                json.dumps(training_results),
                json.dumps(evaluation_results),
                improvements_made,
                next_iteration_planned
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Iteration {iteration_number} logged successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error logging iteration: {e}")
            return False
    
    def get_performance_summary(self, version: str = None, symbol: str = None) -> Dict:
        """Get performance summary for a version or symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM performance_logs WHERE 1=1"
            params = []
            
            if version:
                query += " AND version = ?"
                params.append(version)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return {"message": "No performance data found"}
            
            summary = {
                'total_records': len(df),
                'average_f1_score': df['f1_score'].mean(),
                'average_accuracy': df['accuracy'].mean(),
                'average_win_rate': df['win_rate'].mean(),
                'best_f1_score': df['f1_score'].max(),
                'best_accuracy': df['accuracy'].max(),
                'total_trades': df['total_trades'].sum(),
                'profitable_trades': df['profitable_trades'].sum(),
                'total_pnl': df['total_pnl'].sum(),
                'symbols_tracked': df['symbol'].unique().tolist(),
                'model_types': df['model_type'].unique().tolist()
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting performance summary: {e}")
            return {}
    
    def get_version_history(self) -> List[Dict]:
        """Get version history"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM version_control ORDER BY timestamp DESC", conn)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"❌ Error getting version history: {e}")
            return []
    
    def get_iteration_history(self) -> List[Dict]:
        """Get iteration history"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM iteration_tracking ORDER BY iteration_number DESC", conn)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"❌ Error getting iteration history: {e}")
            return []
    
    def calculate_performance_improvement(self, version1: str, version2: str) -> Dict:
        """Calculate performance improvement between two versions"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get performance data for both versions
            df1 = pd.read_sql_query("SELECT * FROM performance_logs WHERE version = ?", conn, params=[version1])
            df2 = pd.read_sql_query("SELECT * FROM performance_logs WHERE version = ?", conn, params=[version2])
            
            conn.close()
            
            if df1.empty or df2.empty:
                return {"message": "Insufficient data for comparison"}
            
            improvement = {
                'f1_score_improvement': df2['f1_score'].mean() - df1['f1_score'].mean(),
                'accuracy_improvement': df2['accuracy'].mean() - df1['accuracy'].mean(),
                'win_rate_improvement': df2['win_rate'].mean() - df1['win_rate'].mean(),
                'pnl_improvement': df2['total_pnl'].sum() - df1['total_pnl'].sum(),
                'trades_improvement': df2['total_trades'].sum() - df1['total_trades'].sum()
            }
            
            return improvement
            
        except Exception as e:
            print(f"❌ Error calculating performance improvement: {e}")
            return {}
    
    def export_performance_report(self, output_path: str = "performance_report.json") -> bool:
        """Export comprehensive performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'version_history': self.get_version_history(),
                'iteration_history': self.get_iteration_history(),
                'priority_symbols_performance': self.get_performance_summary(symbol='BTCUSDT')
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Performance report exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting performance report: {e}")
            return False

def main():
    """Demo of performance tracking system"""
    print("🚀 Performance Tracking and Version Control System")
    print("="*60)
    
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Create a version
    tracker.create_version(
        version="v2.1.0",
        description="Enhanced BTC/ETH Model Training with Priority Optimization",
        changes_made="Added enhanced model training for BTCUSDT and ETHUSDT, improved ML prediction accuracy",
        priority_symbols=['BTCUSDT', 'ETHUSDT'],
        models_trained=8,
        training_duration=300.0
    )
    
    # Log performance for BTCUSDT
    btc_metrics = {
        'f1_score': 0.76,
        'accuracy': 0.80,
        'precision': 0.77,
        'recall': 0.75,
        'win_rate': 0.68,
        'total_trades': 150,
        'profitable_trades': 102,
        'total_pnl': 1250.50,
        'max_drawdown': -350.25,
        'sharpe_ratio': 1.85
    }
    
    tracker.log_performance(
        version="v2.1.0",
        symbol="BTCUSDT",
        model_type="enhanced_ensemble",
        metrics=btc_metrics,
        notes="Enhanced model showing significant improvement over baseline"
    )
    
    # Log performance for ETHUSDT
    eth_metrics = {
        'f1_score': 0.74,
        'accuracy': 0.78,
        'precision': 0.75,
        'recall': 0.73,
        'win_rate': 0.65,
        'total_trades': 120,
        'profitable_trades': 78,
        'total_pnl': 890.30,
        'max_drawdown': -280.15,
        'sharpe_ratio': 1.62
    }
    
    tracker.log_performance(
        version="v2.1.0",
        symbol="ETHUSDT",
        model_type="enhanced_ensemble",
        metrics=eth_metrics,
        notes="Enhanced model performing well with good risk management"
    )
    
    # Log an iteration
    training_results = {
        'symbols_trained': ['BTCUSDT', 'ETHUSDT'],
        'total_models': 8,
        'average_f1_score': 0.75,
        'training_duration': 300.0
    }
    
    evaluation_results = {
        'btc_performance': btc_metrics,
        'eth_performance': eth_metrics,
        'overall_improvement': 0.15
    }
    
    tracker.log_iteration(
        iteration_number=1,
        training_results=training_results,
        evaluation_results=evaluation_results,
        improvements_made="Enhanced feature engineering, prioritized BTC/ETH training, improved ensemble methods",
        next_iteration_planned="Implement real-time model updates and adaptive learning"
    )
    
    # Get and display summary
    summary = tracker.get_performance_summary()
    print("\n📊 Performance Summary:")
    print(f"Total Records: {summary.get('total_records', 0)}")
    print(f"Average F1 Score: {summary.get('average_f1_score', 0):.3f}")
    print(f"Average Accuracy: {summary.get('average_accuracy', 0):.3f}")
    print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print(f"Symbols Tracked: {summary.get('symbols_tracked', [])}")
    
    # Export report
    tracker.export_performance_report()
    
    print("\n✅ Performance tracking demo completed!")

if __name__ == "__main__":
    main()