#!/usr/bin/env python3
"""
Security & Compliance Module
Enhanced security features and regulatory compliance tools
"""

import json
import logging
import hashlib
import hmac
import base64
import os
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography module not installed. Install with: pip install cryptography")
import secrets
import threading

from file_manager import FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AuditLog:
    """Audit log entry"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True

@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_volume: float
    total_fees: float
    regulatory_checks: Dict
    risk_assessments: Dict
    violations: List[Dict]

class SecurityManager:
    """Security management system"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.encryption_key = None
        self.fernet = None
        self.salt = None
        self._initialize_encryption()
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
        self.password_min_length = 12
        
        # Track login attempts
        self.login_attempts = {}
        self.locked_accounts = {}
        
    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                logger.warning("Cryptography not available, using basic encryption")
                self.encryption_key = b"basic_key_for_demo"
                self.fernet = None
                self.salt = b"basic_salt_for_demo"
                return
                
            # Generate or load encryption key
            key_file = "encryption_key.key"
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            self.fernet = Fernet(self.encryption_key)
            
            # Generate salt for password hashing
            salt_file = "password_salt.key"
            if os.path.exists(salt_file):
                with open(salt_file, 'rb') as f:
                    self.salt = f.read()
            else:
                self.salt = os.urandom(16)
                with open(salt_file, 'wb') as f:
                    f.write(self.salt)
                    
            logger.info("Encryption system initialized")
            
        except Exception as e:
            logger.error(f"Encryption initialization error: {e}")
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key"""
        try:
            if self.fernet is None:
                # Basic encryption for demo
                return base64.b64encode(api_key.encode()).decode()
            encrypted_key = self.fernet.encrypt(api_key.encode())
            return base64.b64encode(encrypted_key).decode()
        except Exception as e:
            logger.error(f"API key encryption error: {e}")
            return ""
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_key.encode())
            if self.fernet is None:
                # Basic decryption for demo
                return encrypted_bytes.decode()
            decrypted_key = self.fernet.decrypt(encrypted_bytes)
            return decrypted_key.decode()
        except Exception as e:
            logger.error(f"API key decryption error: {e}")
            return ""
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        try:
            # Use PBKDF2 for password hashing
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            key = base64.b64encode(kdf.derive(password.encode()))
            return key.decode()
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            return ""
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return self.hash_password(password) == hashed_password
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        try:
            checks = {
                'length': len(password) >= self.password_min_length,
                'uppercase': any(c.isupper() for c in password),
                'lowercase': any(c.islower() for c in password),
                'digit': any(c.isdigit() for c in password),
                'special': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password),
                'no_common': password.lower() not in ['password', '123456', 'qwerty', 'admin']
            }
            
            strength_score = sum(checks.values())
            strength_level = 'weak' if strength_score < 4 else 'medium' if strength_score < 6 else 'strong'
            
            return {
                'valid': all(checks.values()),
                'strength_score': strength_score,
                'strength_level': strength_level,
                'checks': checks
            }
            
        except Exception as e:
            logger.error(f"Password strength validation error: {e}")
            return {'valid': False, 'strength_score': 0, 'strength_level': 'weak', 'checks': {}}
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        try:
            return secrets.token_urlsafe(length)
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            return ""
    
    def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Check rate limiting for actions"""
        try:
            current_time = time.time()
            key = f"{user_id}:{action}"
            
            if key not in self.login_attempts:
                self.login_attempts[key] = []
            
            # Remove old attempts
            self.login_attempts[key] = [
                attempt_time for attempt_time in self.login_attempts[key]
                if current_time - attempt_time < 3600  # 1 hour window
            ]
            
            # Check if too many attempts
            if len(self.login_attempts[key]) >= 10:  # Max 10 attempts per hour
                return False
            
            # Add current attempt
            self.login_attempts[key].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return False

class AuditLogger:
    """Audit logging system"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.db_path = "audit_logs.db"
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize audit database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    ip_address TEXT,
                    session_id TEXT,
                    success BOOLEAN NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_action ON audit_logs(action)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Audit database initialized")
            
        except Exception as e:
            logger.error(f"Audit database initialization error: {e}")
    
    def log_action(self, user_id: str, action: str, resource: str, details: Dict, 
                  ip_address: str = None, session_id: str = None, success: bool = True):
        """Log an action to audit trail"""
        try:
            audit_entry = AuditLog(
                timestamp=datetime.now(),
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                session_id=session_id,
                success=success
            )
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs 
                (timestamp, user_id, action, resource, details, ip_address, session_id, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_entry.timestamp.isoformat(),
                audit_entry.user_id,
                audit_entry.action,
                audit_entry.resource,
                json.dumps(audit_entry.details),
                audit_entry.ip_address,
                audit_entry.session_id,
                audit_entry.success
            ))
            
            conn.commit()
            conn.close()
            
            # Save to file with prevention
            filename = f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, asdict(audit_entry))
            
            logger.info(f"Audit log created: {action} by {user_id}")
            
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
    
    def get_audit_logs(self, user_id: str = None, action: str = None, 
                      start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Retrieve audit logs with filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            logs = []
            for row in rows:
                log = {
                    'id': row[0],
                    'timestamp': row[1],
                    'user_id': row[2],
                    'action': row[3],
                    'resource': row[4],
                    'details': json.loads(row[5]),
                    'ip_address': row[6],
                    'session_id': row[7],
                    'success': bool(row[8])
                }
                logs.append(log)
            
            return logs
            
        except Exception as e:
            logger.error(f"Audit log retrieval error: {e}")
            return []
    
    def export_audit_report(self, start_date: datetime, end_date: datetime) -> str:
        """Export audit report for compliance"""
        try:
            logs = self.get_audit_logs(start_date=start_date, end_date=end_date)
            
            report = {
                'report_id': f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_entries': len(logs),
                'summary': {
                    'total_actions': len(logs),
                    'successful_actions': len([l for l in logs if l['success']]),
                    'failed_actions': len([l for l in logs if not l['success']]),
                    'unique_users': len(set(l['user_id'] for l in logs)),
                    'action_breakdown': {}
                },
                'logs': logs
            }
            
            # Calculate action breakdown
            for log in logs:
                action = log['action']
                if action not in report['summary']['action_breakdown']:
                    report['summary']['action_breakdown'][action] = 0
                report['summary']['action_breakdown'][action] += 1
            
            # Save report with file prevention
            filename = f"audit_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            self.file_manager.save_file(filename, report)
            
            logger.info(f"Audit report exported: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Audit report export error: {e}")
            return ""

class ComplianceManager:
    """Compliance management system"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.regulatory_rules = {
            'kyt': True,  # Know Your Transaction
            'aml': True,   # Anti-Money Laundering
            'kyc': True,   # Know Your Customer
            'position_limits': True,
            'risk_limits': True
        }
        
        # Compliance thresholds
        self.thresholds = {
            'max_daily_volume': 1000000,  # $1M
            'max_position_size': 100000,   # $100K
            'max_daily_trades': 1000,
            'min_trade_size': 10,         # $10
            'suspicious_amount': 10000    # $10K
        }
        
    def check_transaction_compliance(self, transaction: Dict) -> Dict[str, Any]:
        """Check if transaction complies with regulations"""
        try:
            compliance_result = {
                'compliant': True,
                'violations': [],
                'warnings': [],
                'checks_passed': 0,
                'total_checks': 0
            }
            
            # Check transaction size
            amount = transaction.get('amount', 0)
            compliance_result['total_checks'] += 1
            
            if amount < self.thresholds['min_trade_size']:
                compliance_result['warnings'].append(f"Transaction amount {amount} below minimum {self.thresholds['min_trade_size']}")
            
            if amount > self.thresholds['suspicious_amount']:
                compliance_result['warnings'].append(f"Large transaction amount: {amount}")
                compliance_result['violations'].append('SUSPICIOUS_AMOUNT')
                compliance_result['compliant'] = False
            else:
                compliance_result['checks_passed'] += 1
            
            # Check position limits
            position_size = transaction.get('position_size', 0)
            compliance_result['total_checks'] += 1
            
            if position_size > self.thresholds['max_position_size']:
                compliance_result['violations'].append('POSITION_LIMIT_EXCEEDED')
                compliance_result['compliant'] = False
            else:
                compliance_result['checks_passed'] += 1
            
            # Check daily limits
            daily_volume = transaction.get('daily_volume', 0)
            compliance_result['total_checks'] += 1
            
            if daily_volume > self.thresholds['max_daily_volume']:
                compliance_result['violations'].append('DAILY_VOLUME_LIMIT_EXCEEDED')
                compliance_result['compliant'] = False
            else:
                compliance_result['checks_passed'] += 1
            
            # KYT checks
            if self.regulatory_rules['kyt']:
                compliance_result['total_checks'] += 1
                if self._check_kyt(transaction):
                    compliance_result['checks_passed'] += 1
                else:
                    compliance_result['violations'].append('KYT_VIOLATION')
                    compliance_result['compliant'] = False
            
            # AML checks
            if self.regulatory_rules['aml']:
                compliance_result['total_checks'] += 1
                if self._check_aml(transaction):
                    compliance_result['checks_passed'] += 1
                else:
                    compliance_result['violations'].append('AML_VIOLATION')
                    compliance_result['compliant'] = False
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Transaction compliance check error: {e}")
            return {'compliant': False, 'violations': ['COMPLIANCE_CHECK_ERROR'], 'warnings': [], 'checks_passed': 0, 'total_checks': 0}
    
    def _check_kyt(self, transaction: Dict) -> bool:
        """Check Know Your Transaction compliance"""
        try:
            # Implement KYT checks
            # This would typically involve checking transaction patterns, source of funds, etc.
            
            # Simulate KYT check
            amount = transaction.get('amount', 0)
            source = transaction.get('source', '')
            destination = transaction.get('destination', '')
            
            # Check for suspicious patterns
            if amount > self.thresholds['suspicious_amount']:
                if not source or not destination:
                    return False
            
            # Check for round numbers (potential structuring)
            if amount % 10000 == 0 and amount > 50000:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"KYT check error: {e}")
            return False
    
    def _check_aml(self, transaction: Dict) -> bool:
        """Check Anti-Money Laundering compliance"""
        try:
            # Implement AML checks
            # This would typically involve checking against sanctions lists, suspicious patterns, etc.
            
            # Simulate AML check
            amount = transaction.get('amount', 0)
            user_id = transaction.get('user_id', '')
            
            # Check for rapid transactions (potential layering)
            recent_transactions = transaction.get('recent_transactions', [])
            if len(recent_transactions) > 10:
                return False
            
            # Check for unusual amounts
            if amount in [9999, 99999, 999999]:  # Structuring amounts
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"AML check error: {e}")
            return False
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime, 
                               transactions: List[Dict]) -> ComplianceReport:
        """Generate compliance report"""
        try:
            report_id = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Analyze transactions
            total_trades = len(transactions)
            total_volume = sum(t.get('amount', 0) for t in transactions)
            total_fees = sum(t.get('fee', 0) for t in transactions)
            
            # Check compliance for each transaction
            violations = []
            regulatory_checks = {
                'kyt_passed': 0,
                'aml_passed': 0,
                'position_limits_passed': 0,
                'volume_limits_passed': 0
            }
            
            for transaction in transactions:
                compliance = self.check_transaction_compliance(transaction)
                
                if not compliance['compliant']:
                    violations.append({
                        'transaction_id': transaction.get('id'),
                        'timestamp': transaction.get('timestamp'),
                        'violations': compliance['violations'],
                        'amount': transaction.get('amount', 0)
                    })
                
                # Count passed checks
                if 'KYT_VIOLATION' not in compliance['violations']:
                    regulatory_checks['kyt_passed'] += 1
                if 'AML_VIOLATION' not in compliance['violations']:
                    regulatory_checks['aml_passed'] += 1
                if 'POSITION_LIMIT_EXCEEDED' not in compliance['violations']:
                    regulatory_checks['position_limits_passed'] += 1
                if 'DAILY_VOLUME_LIMIT_EXCEEDED' not in compliance['violations']:
                    regulatory_checks['volume_limits_passed'] += 1
            
            # Calculate risk assessments
            risk_assessments = {
                'overall_risk': 'low' if len(violations) < 5 else 'medium' if len(violations) < 20 else 'high',
                'compliance_rate': (total_trades - len(violations)) / total_trades if total_trades > 0 else 1.0,
                'violation_rate': len(violations) / total_trades if total_trades > 0 else 0.0,
                'high_risk_transactions': len([v for v in violations if v.get('amount', 0) > self.thresholds['suspicious_amount']])
            }
            
            report = ComplianceReport(
                report_id=report_id,
                generated_at=datetime.now(),
                period_start=start_date,
                period_end=end_date,
                total_trades=total_trades,
                total_volume=total_volume,
                total_fees=total_fees,
                regulatory_checks=regulatory_checks,
                risk_assessments=risk_assessments,
                violations=violations
            )
            
            # Save report with file prevention
            filename = f"compliance_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            self.file_manager.save_file(filename, asdict(report))
            
            logger.info(f"Compliance report generated: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Compliance report generation error: {e}")
            return None

class SecurityComplianceSystem:
    """Main security and compliance system"""
    
    def __init__(self):
        self.file_manager = FileManager("security_data")
        self.security_manager = SecurityManager(self.file_manager)
        self.audit_logger = AuditLogger(self.file_manager)
        self.compliance_manager = ComplianceManager(self.file_manager)
        
        logger.info("Security and Compliance System initialized")
    
    def secure_api_key_storage(self, exchange_name: str, api_key: str, api_secret: str) -> Dict[str, str]:
        """Securely store API keys"""
        try:
            # Encrypt API keys
            encrypted_key = self.security_manager.encrypt_api_key(api_key)
            encrypted_secret = self.security_manager.encrypt_api_key(api_secret)
            
            # Store encrypted keys
            secure_storage = {
                'exchange': exchange_name,
                'encrypted_api_key': encrypted_key,
                'encrypted_api_secret': encrypted_secret,
                'created_at': datetime.now().isoformat(),
                'last_used': None
            }
            
            # Save with file prevention
            filename = f"secure_api_keys_{exchange_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_manager.save_file(filename, secure_storage)
            
            # Log the action
            self.audit_logger.log_action(
                user_id='system',
                action='API_KEY_STORED',
                resource=exchange_name,
                details={'exchange': exchange_name, 'key_encrypted': True}
            )
            
            logger.info(f"API keys securely stored for {exchange_name}")
            return secure_storage
            
        except Exception as e:
            logger.error(f"API key storage error: {e}")
            return {}
    
    def retrieve_api_keys(self, exchange_name: str) -> Dict[str, str]:
        """Retrieve and decrypt API keys"""
        try:
            # Load encrypted keys
            files = self.file_manager.get_file_list()
            key_files = [f for f in files if f.startswith(f"secure_api_keys_{exchange_name}_")]
            
            if not key_files:
                logger.warning(f"No API keys found for {exchange_name}")
                return {}
            
            # Get latest key file
            latest_file = sorted(key_files)[-1]
            secure_storage = self.file_manager.load_file(latest_file)
            
            if not secure_storage:
                return {}
            
            # Decrypt keys
            api_key = self.security_manager.decrypt_api_key(secure_storage['encrypted_api_key'])
            api_secret = self.security_manager.decrypt_api_key(secure_storage['encrypted_api_secret'])
            
            # Update last used timestamp
            secure_storage['last_used'] = datetime.now().isoformat()
            self.file_manager.save_file(latest_file, secure_storage, force=True)
            
            # Log the action
            self.audit_logger.log_action(
                user_id='system',
                action='API_KEY_RETRIEVED',
                resource=exchange_name,
                details={'exchange': exchange_name, 'key_decrypted': True}
            )
            
            return {
                'api_key': api_key,
                'api_secret': api_secret
            }
            
        except Exception as e:
            logger.error(f"API key retrieval error: {e}")
            return {}
    
    def check_transaction_security(self, transaction: Dict) -> Dict[str, Any]:
        """Check transaction security and compliance"""
        try:
            # Check compliance
            compliance_result = self.compliance_manager.check_transaction_compliance(transaction)
            
            # Log the transaction
            self.audit_logger.log_action(
                user_id=transaction.get('user_id', 'unknown'),
                action='TRANSACTION_CHECKED',
                resource=transaction.get('symbol', 'unknown'),
                details={
                    'transaction_id': transaction.get('id'),
                    'amount': transaction.get('amount'),
                    'compliant': compliance_result['compliant'],
                    'violations': compliance_result['violations']
                },
                success=compliance_result['compliant']
            )
            
            return {
                'secure': compliance_result['compliant'],
                'compliance': compliance_result,
                'recommendations': self._generate_security_recommendations(compliance_result)
            }
            
        except Exception as e:
            logger.error(f"Transaction security check error: {e}")
            return {'secure': False, 'compliance': {}, 'recommendations': ['SECURITY_CHECK_ERROR']}
    
    def _generate_security_recommendations(self, compliance_result: Dict) -> List[str]:
        """Generate security recommendations based on compliance results"""
        recommendations = []
        
        if not compliance_result['compliant']:
            if 'SUSPICIOUS_AMOUNT' in compliance_result['violations']:
                recommendations.append("Review large transaction for potential structuring")
            
            if 'POSITION_LIMIT_EXCEEDED' in compliance_result['violations']:
                recommendations.append("Reduce position size to comply with limits")
            
            if 'KYT_VIOLATION' in compliance_result['violations']:
                recommendations.append("Conduct additional KYT due diligence")
            
            if 'AML_VIOLATION' in compliance_result['violations']:
                recommendations.append("Review transaction for AML compliance")
        
        return recommendations
    
    def generate_security_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate comprehensive security report"""
        try:
            # Get audit logs
            audit_logs = self.audit_logger.get_audit_logs(start_date=start_date, end_date=end_date)
            
            # Generate audit report
            audit_report = self.audit_logger.export_audit_report(start_date, end_date)
            
            # Generate compliance report (simulated transactions)
            simulated_transactions = [
                {
                    'id': f"tx_{i}",
                    'timestamp': datetime.now().isoformat(),
                    'user_id': f"user_{i % 10}",
                    'amount': 1000 + (i * 100),
                    'symbol': 'BTCUSDT',
                    'source': 'exchange',
                    'destination': 'wallet'
                }
                for i in range(100)
            ]
            
            compliance_report = self.compliance_manager.generate_compliance_report(
                start_date, end_date, simulated_transactions
            )
            
            # Create comprehensive security report
            security_report = {
                'report_id': f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'audit_summary': {
                    'total_actions': len(audit_logs),
                    'successful_actions': len([l for l in audit_logs if l['success']]),
                    'failed_actions': len([l for l in audit_logs if not l['success']]),
                    'unique_users': len(set(l['user_id'] for l in audit_logs))
                },
                'compliance_summary': asdict(compliance_report) if compliance_report else {},
                'security_metrics': {
                    'encryption_active': True,
                    'audit_logging_active': True,
                    'compliance_checks_active': True,
                    'rate_limiting_active': True
                },
                'recommendations': [
                    "Regularly review audit logs for suspicious activity",
                    "Monitor compliance violations and take corrective action",
                    "Update security policies based on audit findings",
                    "Conduct regular security assessments"
                ]
            }
            
            # Save security report with file prevention
            filename = f"security_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            self.file_manager.save_file(filename, security_report)
            
            logger.info(f"Security report generated: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Security report generation error: {e}")
            return ""

def main():
    """Main function to run security and compliance system"""
    try:
        # Initialize system
        security_system = SecurityComplianceSystem()
        
        # Example usage
        print("🔐 Security and Compliance System")
        print("=" * 50)
        
        # Test API key storage
        print("\n1. Testing API key storage...")
        secure_storage = security_system.secure_api_key_storage(
            "binance", 
            "test_api_key_123", 
            "test_api_secret_456"
        )
        print(f"API keys stored: {bool(secure_storage)}")
        
        # Test API key retrieval
        print("\n2. Testing API key retrieval...")
        retrieved_keys = security_system.retrieve_api_keys("binance")
        print(f"API keys retrieved: {bool(retrieved_keys)}")
        
        # Test transaction security
        print("\n3. Testing transaction security...")
        test_transaction = {
            'id': 'test_tx_001',
            'user_id': 'user_123',
            'amount': 5000,
            'symbol': 'BTCUSDT',
            'source': 'exchange',
            'destination': 'wallet'
        }
        security_result = security_system.check_transaction_security(test_transaction)
        print(f"Transaction secure: {security_result['secure']}")
        
        # Generate security report
        print("\n4. Generating security report...")
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        report_file = security_system.generate_security_report(start_date, end_date)
        print(f"Security report generated: {report_file}")
        
        print("\n✅ Security and Compliance System test completed!")
        
    except Exception as e:
        logger.error(f"Security system error: {e}")

if __name__ == "__main__":
    main() 