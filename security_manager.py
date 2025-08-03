"""
Security Manager for AITRADE System
Provides comprehensive security features including API key management, rate limiting, and input validation
"""

import os
import hashlib
import hmac
import time
import json
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_audit_logging: bool = True
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    password_min_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True

class SecurityManager:
    """Comprehensive security manager for the trading system"""
    
    def __init__(self, config: SecurityConfig = None):
        """
        Initialize security manager
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.audit_db = 'security_audit.db'
        self.failed_attempts = {}
        self.rate_limit_cache = {}
        self.session_tokens = {}
        
        # Initialize encryption key
        self._initialize_encryption()
        
        # Setup audit database
        self._setup_audit_database()
        
        logger.info("Security manager initialized")
    
    def _initialize_encryption(self):
        """Initialize encryption key"""
        try:
            # Try to load existing key
            key_file = '.encryption_key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            self.cipher = Fernet(self.encryption_key)
            logger.info("Encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.config.enable_encryption = False
    
    def _setup_audit_database(self):
        """Setup audit logging database"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL,
                    details TEXT,
                    security_level TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS failed_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    action TEXT NOT NULL,
                    attempt_count INTEGER DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    action TEXT NOT NULL,
                    request_count INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to setup audit database: {e}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.config.enable_encryption:
            return data
        
        try:
            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.config.enable_encryption:
            return encrypted_data
        
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        key = base64.b64encode(kdf.derive(password.encode()))
        return key.decode(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            
            key = base64.b64encode(kdf.derive(password.encode()))
            return key.decode() == hashed_password
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> tuple:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def check_rate_limit(self, user_id: str, action: str, max_requests: int = 60, window_minutes: int = 1) -> bool:
        """Check rate limiting for user action"""
        if not self.config.enable_rate_limiting:
            return True
        
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            # Count requests in time window
            cursor.execute('''
                SELECT COUNT(*) FROM rate_limits
                WHERE user_id = ? AND action = ? AND timestamp >= ?
            ''', (user_id, action, window_start.isoformat()))
            
            request_count = cursor.fetchone()[0]
            conn.close()
            
            if request_count >= max_requests:
                self._log_audit_event(
                    user_id=user_id,
                    action=f"RATE_LIMIT_EXCEEDED_{action}",
                    success=False,
                    details=f"Rate limit exceeded: {request_count}/{max_requests} requests"
                )
                return False
            
            # Log this request
            self._log_rate_limit_request(user_id, action)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True
    
    def check_failed_attempts(self, user_id: str, action: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        try:
            current_time = datetime.now()
            lockout_threshold = current_time - timedelta(minutes=self.config.lockout_duration_minutes)
            
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            # Count recent failed attempts
            cursor.execute('''
                SELECT COUNT(*) FROM failed_attempts
                WHERE user_id = ? AND action = ? AND timestamp >= ?
            ''', (user_id, action, lockout_threshold.isoformat()))
            
            failed_count = cursor.fetchone()[0]
            conn.close()
            
            return failed_count < self.config.max_failed_attempts
            
        except Exception as e:
            logger.error(f"Failed attempts check failed: {e}")
            return True
    
    def record_failed_attempt(self, user_id: str, action: str, ip_address: str = None):
        """Record a failed attempt"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO failed_attempts (timestamp, user_id, ip_address, action)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), user_id, ip_address, action))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record failed attempt: {e}")
    
    def _log_rate_limit_request(self, user_id: str, action: str):
        """Log rate limit request"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rate_limits (timestamp, user_id, action)
                VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), user_id, action))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log rate limit request: {e}")
    
    def _log_audit_event(self, user_id: str, action: str, success: bool, 
                        resource: str = None, ip_address: str = None, 
                        user_agent: str = None, details: str = None,
                        security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Log security audit event"""
        if not self.config.enable_audit_logging:
            return
        
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_audit (
                    timestamp, user_id, action, resource, ip_address, 
                    user_agent, success, details, security_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                user_id,
                action,
                resource,
                ip_address,
                user_agent,
                success,
                details,
                security_level.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def validate_api_key(self, api_key: str, api_secret: str) -> bool:
        """Validate API key format and security"""
        try:
            # Check key length
            if len(api_key) < 32 or len(api_secret) < 32:
                return False
            
            # Check for common patterns
            if api_key == api_secret:
                return False
            
            # Check for common weak patterns
            weak_patterns = ['test', 'demo', 'example', '123456', 'password']
            for pattern in weak_patterns:
                if pattern in api_key.lower() or pattern in api_secret.lower():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    def validate_input(self, data: Any, validation_rules: Dict[str, Any]) -> tuple:
        """Validate input data against security rules"""
        errors = []
        
        try:
            for field, rules in validation_rules.items():
                if field not in data:
                    if rules.get('required', False):
                        errors.append(f"Required field '{field}' is missing")
                    continue
                
                value = data[field]
                
                # Type validation
                expected_type = rules.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
                    continue
                
                # Length validation
                if isinstance(value, str):
                    min_length = rules.get('min_length')
                    max_length = rules.get('max_length')
                    
                    if min_length and len(value) < min_length:
                        errors.append(f"Field '{field}' must be at least {min_length} characters")
                    
                    if max_length and len(value) > max_length:
                        errors.append(f"Field '{field}' must be at most {max_length} characters")
                
                # Range validation
                if isinstance(value, (int, float)):
                    min_value = rules.get('min_value')
                    max_value = rules.get('max_value')
                    
                    if min_value is not None and value < min_value:
                        errors.append(f"Field '{field}' must be at least {min_value}")
                    
                    if max_value is not None and value > max_value:
                        errors.append(f"Field '{field}' must be at most {max_value}")
                
                # Pattern validation
                pattern = rules.get('pattern')
                if pattern and isinstance(value, str):
                    import re
                    if not re.match(pattern, value):
                        errors.append(f"Field '{field}' does not match required pattern")
                
                # Custom validation
                custom_validator = rules.get('validator')
                if custom_validator and not custom_validator(value):
                    errors.append(f"Field '{field}' failed custom validation")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def sanitize_input(self, data: str) -> str:
        """Sanitize input data to prevent injection attacks"""
        if not isinstance(data, str):
            return str(data)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}']
        sanitized = data
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            # Calculate time threshold
            threshold = datetime.now() - timedelta(hours=hours)
            
            # Get failed attempts
            cursor.execute('''
                SELECT COUNT(*) FROM failed_attempts
                WHERE timestamp >= ?
            ''', [threshold.isoformat()])
            failed_attempts = cursor.fetchone()[0]
            
            # Get rate limit violations
            cursor.execute('''
                SELECT COUNT(*) FROM security_audit
                WHERE action LIKE 'RATE_LIMIT_EXCEEDED_%' AND timestamp >= ?
            ''', [threshold.isoformat()])
            rate_limit_violations = cursor.fetchone()[0]
            
            # Get security events by level
            cursor.execute('''
                SELECT security_level, COUNT(*) FROM security_audit
                WHERE timestamp >= ?
                GROUP BY security_level
            ''', [threshold.isoformat()])
            events_by_level = dict(cursor.fetchall())
            
            # Get recent security events
            cursor.execute('''
                SELECT timestamp, user_id, action, success, details
                FROM security_audit
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', [threshold.isoformat()])
            
            recent_events = [
                {
                    'timestamp': row[0],
                    'user_id': row[1],
                    'action': row[2],
                    'success': row[3],
                    'details': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                'period_hours': hours,
                'failed_attempts': failed_attempts,
                'rate_limit_violations': rate_limit_violations,
                'events_by_level': events_by_level,
                'recent_events': recent_events,
                'security_status': 'SECURE' if failed_attempts == 0 and rate_limit_violations == 0 else 'ATTENTION'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return {'error': str(e)}

# Security decorators
def require_authentication(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator to require authentication"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Authentication logic would go here
            # For now, just log the attempt
            logger.info(f"Authentication required for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_requests: int = 60, window_minutes: int = 1):
    """Decorator to apply rate limiting"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting logic would go here
            # For now, just log the attempt
            logger.info(f"Rate limit check for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(validation_rules: Dict[str, Any]):
    """Decorator to validate input parameters"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Input validation logic would go here
            # For now, just log the attempt
            logger.info(f"Input validation for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def audit_log(action: str, security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator to log security audit events"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Audit logging logic would go here
            # For now, just log the attempt
            logger.info(f"Audit log for {func.__name__}: {action}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global security manager instance
security_manager = SecurityManager() 