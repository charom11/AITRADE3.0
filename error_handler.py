"""
Enhanced Error Handling and Logging System
Provides comprehensive error handling, logging, and monitoring for the trading system
"""

import logging
import traceback
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
import json
import sqlite3
from pathlib import Path
import threading
from dataclasses import dataclass, field
from enum import Enum
import time

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_FETCH = "DATA_FETCH"
    STRATEGY = "STRATEGY"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    CONFIGURATION = "CONFIGURATION"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    VALIDATION = "VALIDATION"
    SYSTEM = "SYSTEM"

@dataclass
class ErrorRecord:
    """Error record for tracking and analysis"""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ErrorHandler:
    """Comprehensive error handling and logging system"""
    
    def __init__(self, 
                 log_file: str = 'trading_system.log',
                 error_db: str = 'error_tracking.db',
                 max_log_size_mb: int = 100,
                 backup_count: int = 5,
                 enable_telegram_alerts: bool = False,
                 telegram_bot_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None):
        """
        Initialize error handler
        
        Args:
            log_file: Path to log file
            error_db: Path to error tracking database
            max_log_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
            enable_telegram_alerts: Enable Telegram error alerts
            telegram_bot_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
        """
        self.log_file = log_file
        self.error_db = error_db
        self.max_log_size_mb = max_log_size_mb
        self.backup_count = backup_count
        self.enable_telegram_alerts = enable_telegram_alerts
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        # Error tracking
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.recent_errors = []
        self.max_recent_errors = 100
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
        
        # Setup database
        self._setup_database()
        
        # Setup Telegram
        if self.enable_telegram_alerts:
            self._setup_telegram()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # File handler with rotation
                logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_log_size_mb * 1024 * 1024,
                    backupCount=self.backup_count
                ),
                # Console handler
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create logger
        self.logger = logging.getLogger('TradingSystem')
        self.logger.setLevel(logging.INFO)
        
        # Suppress third-party library logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('ccxt').setLevel(logging.WARNING)
    
    def _setup_database(self):
        """Setup error tracking database"""
        try:
            conn = sqlite3.connect(self.error_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    exception_type TEXT,
                    exception_message TEXT,
                    traceback TEXT,
                    context TEXT,
                    user_id TEXT,
                    session_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON error_records(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_category 
                ON error_records(category)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_severity 
                ON error_records(severity)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error setting up error database: {e}")
    
    def _setup_telegram(self):
        """Setup Telegram for error alerts"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            self.enable_telegram_alerts = False
            return
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.enable_telegram_alerts = False
            self.logger.warning("Requests library not available, disabling Telegram alerts")
    
    def log_error(self, 
                  error: Exception,
                  category: ErrorCategory,
                  message: str = "",
                  context: Dict[str, Any] = None,
                  severity: ErrorSeverity = ErrorSeverity.ERROR,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None) -> ErrorRecord:
        """
        Log an error with comprehensive tracking
        
        Args:
            error: The exception that occurred
            category: Error category
            message: Additional error message
            context: Additional context information
            severity: Error severity level
            user_id: User ID for tracking
            session_id: Session ID for tracking
            
        Returns:
            ErrorRecord object
        """
        with self._lock:
            # Create error record
            error_record = ErrorRecord(
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=message or str(error),
                exception_type=type(error).__name__,
                exception_message=str(error),
                traceback=traceback.format_exc(),
                context=context or {},
                user_id=user_id,
                session_id=session_id
            )
            
            # Log to file
            log_message = f"[{category.value}] {message}: {str(error)}"
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif severity == ErrorSeverity.ERROR:
                self.logger.error(log_message)
            elif severity == ErrorSeverity.WARNING:
                self.logger.warning(log_message)
            elif severity == ErrorSeverity.INFO:
                self.logger.info(log_message)
            else:
                self.logger.debug(log_message)
            
            # Update error counts
            self.error_counts[category] += 1
            
            # Add to recent errors
            self.recent_errors.append(error_record)
            if len(self.recent_errors) > self.max_recent_errors:
                self.recent_errors.pop(0)
            
            # Save to database
            self._save_error_to_db(error_record)
            
            # Send Telegram alert for critical errors
            if severity == ErrorSeverity.CRITICAL and self.enable_telegram_alerts:
                self._send_telegram_alert(error_record)
            
            return error_record
    
    def _save_error_to_db(self, error_record: ErrorRecord):
        """Save error record to database"""
        try:
            conn = sqlite3.connect(self.error_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO error_records (
                    timestamp, severity, category, message, exception_type,
                    exception_message, traceback, context, user_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_record.timestamp.isoformat(),
                error_record.severity.value,
                error_record.category.value,
                error_record.message,
                error_record.exception_type,
                error_record.exception_message,
                error_record.traceback,
                json.dumps(error_record.context),
                error_record.user_id,
                error_record.session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def _send_telegram_alert(self, error_record: ErrorRecord):
        """Send error alert to Telegram"""
        try:
            message = f"""
🚨 CRITICAL ERROR ALERT

Category: {error_record.category.value}
Message: {error_record.message}
Exception: {error_record.exception_type}
Time: {error_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Context: {json.dumps(error_record.context, indent=2)}
            """.strip()
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = self.requests.post(url, data=data, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to send Telegram alert: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}")
    
    def handle_exception(self, 
                        category: ErrorCategory,
                        message: str = "",
                        context: Dict[str, Any] = None,
                        severity: ErrorSeverity = ErrorSeverity.ERROR,
                        reraise: bool = True):
        """
        Decorator for handling exceptions in functions
        
        Args:
            category: Error category
            message: Error message
            context: Additional context
            severity: Error severity
            reraise: Whether to reraise the exception
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Add function context
                    func_context = context or {}
                    func_context.update({
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    })
                    
                    self.log_error(
                        error=e,
                        category=category,
                        message=message or f"Error in {func.__name__}",
                        context=func_context,
                        severity=severity
                    )
                    
                    if reraise:
                        raise
                    return None
            return wrapper
        return decorator
    
    def retry_on_error(self, 
                      max_retries: int = 3,
                      delay_seconds: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,),
                      category: ErrorCategory = ErrorCategory.SYSTEM):
        """
        Decorator for retrying functions on error
        
        Args:
            max_retries: Maximum number of retries
            delay_seconds: Initial delay between retries
            backoff_factor: Exponential backoff factor
            exceptions: Exceptions to retry on
            category: Error category for logging
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                delay = delay_seconds
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            self.log_error(
                                error=e,
                                category=category,
                                message=f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.1f}s",
                                context={
                                    'function': func.__name__,
                                    'attempt': attempt + 1,
                                    'max_retries': max_retries
                                },
                                severity=ErrorSeverity.WARNING
                            )
                            
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            self.log_error(
                                error=e,
                                category=category,
                                message=f"All {max_retries + 1} attempts failed for {func.__name__}",
                                context={
                                    'function': func.__name__,
                                    'attempts': max_retries + 1
                                },
                                severity=ErrorSeverity.ERROR
                            )
                
                raise last_exception
            return wrapper
        return decorator
    
    def validate_input(self, 
                      validation_func: Callable,
                      error_message: str = "Invalid input",
                      category: ErrorCategory = ErrorCategory.VALIDATION):
        """
        Decorator for input validation
        
        Args:
            validation_func: Function that returns True if input is valid
            error_message: Error message for invalid input
            category: Error category
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not validation_func(*args, **kwargs):
                    error = ValueError(error_message)
                    self.log_error(
                        error=error,
                        category=category,
                        message=error_message,
                        context={
                            'function': func.__name__,
                            'args': str(args),
                            'kwargs': str(kwargs)
                        },
                        severity=ErrorSeverity.ERROR
                    )
                    raise error
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_error_summary(self, 
                         hours: int = 24,
                         category: Optional[ErrorCategory] = None) -> Dict[str, Any]:
        """
        Get error summary for the specified time period
        
        Args:
            hours: Number of hours to look back
            category: Optional category filter
            
        Returns:
            Error summary dictionary
        """
        try:
            conn = sqlite3.connect(self.error_db)
            cursor = conn.cursor()
            
            # Calculate time threshold
            threshold = datetime.now() - timedelta(hours=hours)
            
            # Build query
            query = '''
                SELECT severity, category, COUNT(*) as count
                FROM error_records
                WHERE timestamp >= ?
            '''
            params = [threshold.isoformat()]
            
            if category:
                query += ' AND category = ?'
                params.append(category.value)
            
            query += ' GROUP BY severity, category ORDER BY count DESC'
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Build summary
            summary = {
                'period_hours': hours,
                'total_errors': sum(row[2] for row in results),
                'by_severity': {},
                'by_category': {},
                'recent_errors': []
            }
            
            for severity, cat, count in results:
                if severity not in summary['by_severity']:
                    summary['by_severity'][severity] = 0
                summary['by_severity'][severity] += count
                
                if cat not in summary['by_category']:
                    summary['by_category'][cat] = 0
                summary['by_category'][cat] += count
            
            # Get recent errors
            cursor.execute('''
                SELECT timestamp, severity, category, message
                FROM error_records
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', [threshold.isoformat()])
            
            recent = cursor.fetchall()
            summary['recent_errors'] = [
                {
                    'timestamp': row[0],
                    'severity': row[1],
                    'category': row[2],
                    'message': row[3]
                }
                for row in recent
            ]
            
            conn.close()
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            return {}
    
    def cleanup_old_errors(self, days: int = 30):
        """Clean up old error records from database"""
        try:
            conn = sqlite3.connect(self.error_db)
            cursor = conn.cursor()
            
            threshold = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                DELETE FROM error_records
                WHERE timestamp < ?
            ''', [threshold.isoformat()])
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old error records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old errors: {e}")

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
def log_error(error: Exception, 
              category: ErrorCategory,
              message: str = "",
              context: Dict[str, Any] = None,
              severity: ErrorSeverity = ErrorSeverity.ERROR) -> ErrorRecord:
    """Log an error using the global error handler"""
    return error_handler.log_error(error, category, message, context, severity)

def handle_exception(category: ErrorCategory,
                    message: str = "",
                    context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    reraise: bool = True):
    """Decorator for handling exceptions"""
    return error_handler.handle_exception(category, message, context, severity, reraise)

def retry_on_error(max_retries: int = 3,
                  delay_seconds: float = 1.0,
                  backoff_factor: float = 2.0,
                  exceptions: tuple = (Exception,),
                  category: ErrorCategory = ErrorCategory.SYSTEM):
    """Decorator for retrying functions on error"""
    return error_handler.retry_on_error(max_retries, delay_seconds, backoff_factor, exceptions, category)

def validate_input(validation_func: Callable,
                  error_message: str = "Invalid input",
                  category: ErrorCategory = ErrorCategory.VALIDATION):
    """Decorator for input validation"""
    return error_handler.validate_input(validation_func, error_message, category) 