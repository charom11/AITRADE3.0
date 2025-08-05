#!/usr/bin/env python3
"""
Dashboard Monitor - Real-time Monitoring & Alerts
Enhanced dashboard with real-time monitoring and smart notifications
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import requests
try:
    import telegram
except ImportError:
    telegram = None
from dataclasses import dataclass, asdict

from file_manager import FileManager
from enhanced_trading_system import EnhancedTradingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert structure"""
    timestamp: datetime
    type: str  # 'info', 'warning', 'error', 'success'
    message: str
    symbol: Optional[str] = None
    data: Optional[Dict] = None
    priority: str = 'medium'  # 'low', 'medium', 'high', 'critical'

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    timestamp: datetime
    total_return: float
    win_rate: float
    total_trades: int
    active_positions: int
    portfolio_value: float
    drawdown: float
    sharpe_ratio: float
    max_drawdown: float

class DashboardMonitor:
    """Real-time dashboard monitoring system"""
    
    def __init__(self, trading_system: EnhancedTradingSystem, file_manager: FileManager):
        self.trading_system = trading_system
        self.file_manager = file_manager
        self.alerts = []
        self.performance_history = []
        self.monitoring_active = False
        self.alert_thresholds = {
            'drawdown_warning': 0.03,  # 3%
            'drawdown_critical': 0.05,  # 5%
            'win_rate_minimum': 0.5,   # 50%
            'position_limit': 10,
            'profit_target': 0.1,      # 10%
            'loss_limit': -0.05        # -5%
        }
        
        # Telegram configuration (optional)
        self.telegram_bot = None
        self.telegram_chat_id = None
        
    def setup_telegram(self, bot_token: str, chat_id: str):
        """Setup Telegram notifications"""
        try:
            if telegram is None:
                logger.warning("Telegram module not installed. Install with: pip install python-telegram-bot")
                return
            self.telegram_bot = telegram.Bot(token=bot_token)
            self.telegram_chat_id = chat_id
            logger.info("Telegram notifications configured")
        except Exception as e:
            logger.error(f"Failed to setup Telegram: {e}")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        logger.info("Dashboard monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Dashboard monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get system status
                status = self.trading_system.get_system_status()
                
                # Analyze performance
                self._analyze_performance(status)
                
                # Check for alerts
                self._check_alerts(status)
                
                # Update dashboard data
                self._update_dashboard(status)
                
                # Send notifications
                self._send_notifications()
                
                # Sleep between checks
                time.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _analyze_performance(self, status: Dict):
        """Analyze system performance"""
        try:
            risk_metrics = status.get('risk_metrics', {})
            performance_metrics = status.get('performance_metrics', {})
            
            # Calculate performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=risk_metrics.get('total_pnl', 0) / risk_metrics.get('portfolio_value', 10000),
                win_rate=performance_metrics.get('win_rate', 0),
                total_trades=performance_metrics.get('total_signals', 0),
                active_positions=status.get('active_positions', 0),
                portfolio_value=risk_metrics.get('portfolio_value', 10000),
                drawdown=risk_metrics.get('drawdown', 0),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                max_drawdown=self._calculate_max_drawdown()
            )
            
            # Store performance history
            self.performance_history.append(asdict(metrics))
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Save performance data with file prevention
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, asdict(metrics))
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.performance_history) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_return = self.performance_history[i-1]['total_return']
                curr_return = self.performance_history[i]['total_return']
                returns.append(curr_return - prev_return)
            
            if not returns:
                return 0.0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return avg_return / std_return
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.performance_history:
                return 0.0
            
            portfolio_values = [p['portfolio_value'] for p in self.performance_history]
            peak = portfolio_values[0]
            max_dd = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {e}")
            return 0.0
    
    def _check_alerts(self, status: Dict):
        """Check for alert conditions"""
        try:
            risk_metrics = status.get('risk_metrics', {})
            performance_metrics = status.get('performance_metrics', {})
            
            # Check drawdown alerts
            drawdown = risk_metrics.get('drawdown', 0)
            if drawdown > self.alert_thresholds['drawdown_critical']:
                self._create_alert(
                    'critical',
                    f"Critical drawdown detected: {drawdown:.2%}",
                    priority='critical'
                )
            elif drawdown > self.alert_thresholds['drawdown_warning']:
                self._create_alert(
                    'warning',
                    f"High drawdown warning: {drawdown:.2%}",
                    priority='high'
                )
            
            # Check win rate alerts
            win_rate = performance_metrics.get('win_rate', 0)
            if win_rate < self.alert_thresholds['win_rate_minimum']:
                self._create_alert(
                    'warning',
                    f"Low win rate: {win_rate:.2%}",
                    priority='medium'
                )
            
            # Check position limit alerts
            active_positions = status.get('active_positions', 0)
            if active_positions > self.alert_thresholds['position_limit']:
                self._create_alert(
                    'warning',
                    f"Too many active positions: {active_positions}",
                    priority='medium'
                )
            
            # Check profit/loss alerts
            total_return = risk_metrics.get('total_pnl', 0) / risk_metrics.get('portfolio_value', 10000)
            if total_return > self.alert_thresholds['profit_target']:
                self._create_alert(
                    'success',
                    f"Profit target reached: {total_return:.2%}",
                    priority='low'
                )
            elif total_return < self.alert_thresholds['loss_limit']:
                self._create_alert(
                    'error',
                    f"Loss limit exceeded: {total_return:.2%}",
                    priority='high'
                )
            
        except Exception as e:
            logger.error(f"Alert check error: {e}")
    
    def _create_alert(self, alert_type: str, message: str, priority: str = 'medium', symbol: str = None):
        """Create new alert"""
        try:
            alert = Alert(
                timestamp=datetime.now(),
                type=alert_type,
                message=message,
                symbol=symbol,
                priority=priority
            )
            
            self.alerts.append(asdict(alert))
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            # Save alert with file prevention
            filename = f"alert_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, asdict(alert))
            
            logger.info(f"Alert created: {message}")
            
        except Exception as e:
            logger.error(f"Alert creation error: {e}")
    
    def _update_dashboard(self, status: Dict):
        """Update dashboard data"""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': status,
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'performance_summary': {
                    'total_return': self._get_latest_metric('total_return'),
                    'win_rate': self._get_latest_metric('win_rate'),
                    'sharpe_ratio': self._get_latest_metric('sharpe_ratio'),
                    'max_drawdown': self._get_latest_metric('max_drawdown'),
                    'active_positions': status.get('active_positions', 0)
                },
                'risk_metrics': status.get('risk_metrics', {}),
                'trading_pairs': status.get('trading_pairs', [])
            }
            
            # Save dashboard data with file prevention
            filename = f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, dashboard_data)
            
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
    
    def _get_latest_metric(self, metric_name: str) -> float:
        """Get latest metric value"""
        try:
            if self.performance_history:
                return self.performance_history[-1].get(metric_name, 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Metric retrieval error: {e}")
            return 0.0
    
    def _send_notifications(self):
        """Send notifications via Telegram"""
        try:
            if not self.telegram_bot or not self.telegram_chat_id:
                return
            
            # Send critical alerts
            critical_alerts = [a for a in self.alerts[-5:] if a['priority'] == 'critical']
            
            for alert in critical_alerts:
                message = f"🚨 CRITICAL ALERT\n{alert['message']}\nTime: {alert['timestamp']}"
                
                try:
                    self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id,
                        text=message
                    )
                    logger.info(f"Telegram notification sent: {alert['message']}")
                except Exception as e:
                    logger.error(f"Telegram send error: {e}")
            
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def get_dashboard_summary(self) -> Dict:
        """Get dashboard summary for display"""
        try:
            latest_status = self.trading_system.get_system_status()
            
            summary = {
                'system_running': latest_status.get('system_running', False),
                'active_positions': latest_status.get('active_positions', 0),
                'total_trades': latest_status.get('performance_metrics', {}).get('total_signals', 0),
                'win_rate': latest_status.get('performance_metrics', {}).get('win_rate', 0),
                'portfolio_value': latest_status.get('risk_metrics', {}).get('portfolio_value', 10000),
                'total_pnl': latest_status.get('risk_metrics', {}).get('total_pnl', 0),
                'drawdown': latest_status.get('risk_metrics', {}).get('drawdown', 0),
                'sharpe_ratio': self._get_latest_metric('sharpe_ratio'),
                'max_drawdown': self._get_latest_metric('max_drawdown'),
                'recent_alerts': self.alerts[-5:],
                'trading_pairs': latest_status.get('trading_pairs', []),
                'last_update': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Dashboard summary error: {e}")
            return {}
    
    def get_performance_chart_data(self, days: int = 30) -> Dict:
        """Get performance data for charts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent performance data
            recent_data = [
                p for p in self.performance_history
                if datetime.fromisoformat(p['timestamp']) > cutoff_date
            ]
            
            chart_data = {
                'timestamps': [p['timestamp'] for p in recent_data],
                'portfolio_values': [p['portfolio_value'] for p in recent_data],
                'total_returns': [p['total_return'] for p in recent_data],
                'drawdowns': [p['drawdown'] for p in recent_data],
                'win_rates': [p['win_rate'] for p in recent_data],
                'sharpe_ratios': [p['sharpe_ratio'] for p in recent_data]
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Chart data error: {e}")
            return {}
    
    def export_performance_report(self, filename: str = None) -> str:
        """Export comprehensive performance report"""
        try:
            if not filename:
                filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'system_summary': self.get_dashboard_summary(),
                'performance_history': self.performance_history,
                'alerts': self.alerts,
                'chart_data': self.get_performance_chart_data(90),  # 90 days
                'analysis': {
                    'total_trades': len(self.performance_history),
                    'avg_return': np.mean([p['total_return'] for p in self.performance_history]) if self.performance_history else 0,
                    'best_day': max([p['total_return'] for p in self.performance_history]) if self.performance_history else 0,
                    'worst_day': min([p['total_return'] for p in self.performance_history]) if self.performance_history else 0,
                    'volatility': np.std([p['total_return'] for p in self.performance_history]) if self.performance_history else 0
                }
            }
            
            # Save report with file prevention
            self.file_manager.save_file(filename, report)
            
            logger.info(f"Performance report exported: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Report export error: {e}")
            return ""

def main():
    """Main function to run dashboard monitor"""
    try:
        # Initialize components
        file_manager = FileManager("dashboard_data")
        trading_system = EnhancedTradingSystem()
        
        # Initialize dashboard
        dashboard = DashboardMonitor(trading_system, file_manager)
        
        # Setup Telegram (optional)
        # dashboard.setup_telegram("YOUR_BOT_TOKEN", "YOUR_CHAT_ID")
        
        # Start monitoring
        dashboard.start_monitoring()
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down dashboard...")
            dashboard.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main() 