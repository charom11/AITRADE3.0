"""
Enhanced Configuration file for the Mini Hedge Fund
Defines trading parameters, risk management, and strategy settings with validation
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading configuration with validation"""
    initial_capital: float = 100000
    max_position_size: float = 0.1
    max_drawdown: float = 0.15
    risk_per_trade: float = 0.02
    commission: float = 0.001
    slippage: float = 0.0005
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")

@dataclass
class StrategyConfig:
    """Strategy configuration with validation"""
    momentum: Dict[str, Any] = field(default_factory=lambda: {
        'short_window': 20,
        'long_window': 200,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'risk_per_trade': 0.02
    })
    mean_reversion: Dict[str, Any] = field(default_factory=lambda: {
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'risk_per_trade': 0.015
    })
    pairs_trading: Dict[str, Any] = field(default_factory=lambda: {
        'lookback_period': 252,
        'correlation_threshold': 0.7,
        'z_score_threshold': 2.0,
        'risk_per_trade': 0.015
    })
    divergence: Dict[str, Any] = field(default_factory=lambda: {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'min_candles': 50,
        'swing_threshold': 0.02,
        'confirm_with_support_resistance': True,
        'confirm_with_candlestick': True,
        'risk_per_trade': 0.025
    })
    support_resistance: Dict[str, Any] = field(default_factory=lambda: {
        'min_touches': 2,
        'zone_buffer': 0.003,
        'volume_threshold': 1.5,
        'swing_sensitivity': 0.02,
        'risk_per_trade': 0.02,
        'enable_charts': False,
        'enable_alerts': True
    })
    fibonacci: Dict[str, Any] = field(default_factory=lambda: {
        'buffer_percentage': 0.003,
        'min_swing_strength': 0.6,
        'risk_per_trade': 0.02,
        'enable_charts': False,
        'enable_alerts': True
    })

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_correlation: float = 0.8
    max_sector_exposure: float = 0.3
    stop_loss: float = 0.05
    take_profit: float = 0.10
    trailing_stop: float = 0.03
    max_open_positions: int = 10
    position_sizing_method: str = 'kelly'  # 'fixed', 'kelly', 'volatility'
    
    def __post_init__(self):
        """Validate risk configuration"""
        if not 0 < self.max_correlation <= 1:
            raise ValueError("Max correlation must be between 0 and 1")
        if not 0 < self.max_sector_exposure <= 1:
            raise ValueError("Max sector exposure must be between 0 and 1")

@dataclass
class PerformanceConfig:
    """Performance analysis configuration"""
    risk_free_rate: float = 0.02
    benchmark_symbol: str = 'SPY'
    rolling_window: int = 252
    min_trades_for_analysis: int = 10
    confidence_level: float = 0.95

@dataclass
class DataConfig:
    """Data management configuration"""
    cache_dir: str = 'data_cache'
    max_cache_age_days: int = 7
    data_source: str = 'yfinance'  # 'yfinance', 'alpha_vantage', 'binance'
    update_interval_minutes: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: int = 5

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key_env_var: str = 'BINANCE_API_KEY'
    api_secret_env_var: str = 'BINANCE_SECRET_KEY'
    telegram_token_env_var: str = 'TELEGRAM_BOT_TOKEN'
    telegram_chat_id_env_var: str = 'TELEGRAM_CHAT_ID'
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_ssl_verification: bool = True
    session_timeout_minutes: int = 30

class ConfigManager:
    """Configuration manager with validation and environment support"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to configuration file (JSON or YAML)
        """
        self.config_file = config_file
        self.trading_config = TradingConfig()
        self.strategy_config = StrategyConfig()
        self.risk_config = RiskConfig()
        self.performance_config = PerformanceConfig()
        self.data_config = DataConfig()
        self.security_config = SecurityConfig()
        
        # Load configuration from file if provided
        if config_file:
            self.load_config_file(config_file)
        
        # Override with environment variables
        self.load_environment_overrides()
        
        # Validate all configurations
        self.validate_configurations()
        
        logger.info("Configuration manager initialized successfully")
    
    def load_config_file(self, config_file: str):
        """Load configuration from file"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file {config_file} not found, using defaults")
                return
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml':
                    import yaml
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'trading' in config_data:
                self._update_config(self.trading_config, config_data['trading'])
            if 'strategy' in config_data:
                self._update_config(self.strategy_config, config_data['strategy'])
            if 'risk' in config_data:
                self._update_config(self.risk_config, config_data['risk'])
            if 'performance' in config_data:
                self._update_config(self.performance_config, config_data['performance'])
            if 'data' in config_data:
                self._update_config(self.data_config, config_data['data'])
            if 'security' in config_data:
                self._update_config(self.security_config, config_data['security'])
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise
    
    def load_environment_overrides(self):
        """Override configuration with environment variables"""
        # Trading configuration overrides
        if os.getenv('INITIAL_CAPITAL'):
            self.trading_config.initial_capital = float(os.getenv('INITIAL_CAPITAL'))
        if os.getenv('MAX_POSITION_SIZE'):
            self.trading_config.max_position_size = float(os.getenv('MAX_POSITION_SIZE'))
        if os.getenv('RISK_PER_TRADE'):
            self.trading_config.risk_per_trade = float(os.getenv('RISK_PER_TRADE'))
        
        # Data configuration overrides
        if os.getenv('DATA_SOURCE'):
            self.data_config.data_source = os.getenv('DATA_SOURCE')
        if os.getenv('CACHE_DIR'):
            self.data_config.cache_dir = os.getenv('CACHE_DIR')
        
        # Security configuration overrides
        if os.getenv('ENABLE_RATE_LIMITING'):
            self.security_config.enable_rate_limiting = os.getenv('ENABLE_RATE_LIMITING').lower() == 'true'
        if os.getenv('MAX_REQUESTS_PER_MINUTE'):
            self.security_config.max_requests_per_minute = int(os.getenv('MAX_REQUESTS_PER_MINUTE'))
        
        logger.info("Environment variable overrides applied")
    
    def _update_config(self, config_obj, updates: Dict[str, Any]):
        """Update configuration object with new values"""
        for key, value in updates.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def validate_configurations(self):
        """Validate all configurations"""
        try:
            # Validate trading config
            self.trading_config.__post_init__()
            
            # Validate risk config
            self.risk_config.__post_init__()
            
            # Validate strategy configurations
            for strategy_name, config in self.strategy_config.__dict__.items():
                if isinstance(config, dict) and 'risk_per_trade' in config:
                    if not 0 < config['risk_per_trade'] <= 1:
                        raise ValueError(f"Invalid risk_per_trade in {strategy_name}")
            
            logger.info("All configurations validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials from environment variables"""
        credentials = {}
        
        # Binance credentials
        api_key = os.getenv(self.security_config.api_key_env_var)
        api_secret = os.getenv(self.security_config.api_secret_env_var)
        
        if api_key and api_secret:
            credentials['binance'] = {
                'api_key': api_key,
                'api_secret': api_secret
            }
        
        # Telegram credentials
        telegram_token = os.getenv(self.security_config.telegram_token_env_var)
        telegram_chat_id = os.getenv(self.security_config.telegram_chat_id_env_var)
        
        if telegram_token and telegram_chat_id:
            credentials['telegram'] = {
                'token': telegram_token,
                'chat_id': telegram_chat_id
            }
        
        return credentials
    
    def save_config(self, filename: str = None):
        """Save current configuration to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'config_{timestamp}.json'
        
        config_data = {
            'trading': self.trading_config.__dict__,
            'strategy': self.strategy_config.__dict__,
            'risk': self.risk_config.__dict__,
            'performance': self.performance_config.__dict__,
            'data': self.data_config.__dict__,
            'security': self.security_config.__dict__
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

# Initialize global configuration manager
config_manager = ConfigManager()

# Enhanced Trading System Configuration
TRADING_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '5m',
    'exchange': 'binance',
    'max_position_size': 0.1,  # 10% of portfolio
    'stop_loss_pct': 0.02,     # 2% stop loss
    'take_profit_pct': 0.06,   # 6% take profit
    'risk_per_trade': 0.01,    # 1% risk per trade
    'max_open_trades': 5,
    'min_volume_threshold': 1000000,  # Minimum volume for trade
    'enable_telegram_alerts': True,
    'enable_charts': True,
    'backtest_mode': False,
    'paper_trading': True
}

# Enhanced Strategy Configurations
STRATEGY_CONFIG = {
    'momentum': {
        'sma_short': 20,
        'sma_medium': 50,
        'sma_long': 200,
        'rsi_period': 14,
        'rsi_overbought': 75,  # Increased from 70
        'rsi_oversold': 25,    # Decreased from 30
        'volume_multiplier': 1.5,  # Volume confirmation
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
        'min_trend_strength': 0.6,  # Minimum trend strength
        'max_position': 0.15,  # Increased position size
        'confidence_threshold': 0.7
    },
    
    'mean_reversion': {
        'bb_period': 20,
        'bb_std': 2.2,  # Increased from 2.0
        'rsi_period': 14,
        'rsi_overbought': 80,  # More conservative
        'rsi_oversold': 20,    # More conservative
        'atr_period': 14,
        'min_bounce_pct': 0.02,  # Minimum bounce percentage
        'max_deviation': 3.0,    # Maximum deviation from mean
        'volume_confirmation': True,
        'max_position': 0.12,
        'confidence_threshold': 0.65
    },
    
    'pairs_trading': {
        'correlation_threshold': 0.85,  # Increased correlation requirement
        'z_score_threshold': 2.5,       # More conservative z-score (fixed key name)
        'zscore_threshold': 2.5,        # More conservative z-score
        'lookback_period': 100,
        'min_spread': 0.005,    # Minimum spread for profit
        'max_position': 0.08,
        'confidence_threshold': 0.6
    },
    
    'divergence': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'swing_window': 10,
        'min_divergence_strength': 0.7,
        'volume_confirmation': True,
        'enhanced_confirmation': True,
        'confirm_with_support_resistance': True,  # Added missing config
        'confirm_with_candlestick': True,         # Added missing config
        'max_position': 0.1,
        'confidence_threshold': 0.75,
        'min_candles': 50,  # Minimum candles required for analysis
        'swing_threshold': 0.02  # Minimum swing threshold for detection
    },
    
    'support_resistance': {
        'min_touches': 3,       # Increased from 2
        'zone_buffer': 0.005,  # 0.5% buffer for zone identification
        'volume_threshold': 0.8,
        'swing_sensitivity': 0.02,
        'enable_charts': True,
        'enable_alerts': True,
        'max_position': 0.12,
        'confidence_threshold': 0.7,
        'bounce_threshold': 0.015  # 1.5% bounce confirmation
    },
    
    'fibonacci': {
        'buffer_percentage': 0.003,  # 0.3% buffer for level identification
        'swing_window': 20,
        'min_swing_strength': 0.03,
        'retracement_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
        'extension_levels': [1.272, 1.618, 2.0, 2.618],
        'tolerance': 0.01,
        'volume_confirmation': True,
        'max_position': 0.1,
        'confidence_threshold': 0.65,
        'min_reaction_pct': 0.01  # 1% minimum reaction
    }
}

# Enhanced Risk Management
RISK_CONFIG = {
    'max_daily_loss': 0.05,     # 5% max daily loss
    'max_drawdown': 0.15,       # 15% max drawdown
    'position_sizing_method': 'kelly',  # 'fixed', 'kelly', 'volatility'
    'volatility_lookback': 20,
    'correlation_threshold': 0.7,
    'sector_exposure_limit': 0.3,
    'dynamic_stop_loss': True,
    'trailing_stop': True,
    'trailing_stop_distance': 0.02
}

# Enhanced Performance Metrics
PERFORMANCE_CONFIG = {
    'benchmark': 'BTC/USDT',
    'risk_free_rate': 0.02,
    'min_trades_for_analysis': 10,
    'rolling_window': 30,
    'confidence_interval': 0.95,
    'enable_advanced_metrics': True
}

# Enhanced Data Configuration
DATA_CONFIG = {
    'min_data_points': 1000,
    'data_quality_threshold': 0.95,
    'enable_data_validation': True,
    'cache_duration': 3600,  # 1 hour
    'enable_real_time_updates': True
}

# Asset Universe - Multiple markets for diversification
ASSETS = {
    'equities': {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'IWM': 'Russell 2000 ETF',
        'EFA': 'International Developed Markets ETF',
        'EEM': 'Emerging Markets ETF'
    },
    'commodities': {
        'GLD': 'Gold ETF',
        'SLV': 'Silver ETF',
        'USO': 'Oil ETF',
        'UNG': 'Natural Gas ETF'
    },
    'bonds': {
        'TLT': '20+ Year Treasury Bond ETF',
        'IEF': '7-10 Year Treasury Bond ETF',
        'LQD': 'Investment Grade Corporate Bond ETF'
    },
    'currencies': {
        'UUP': 'US Dollar Bull ETF',
        'FXE': 'Euro ETF',
        'FXY': 'Japanese Yen ETF'
    },
    'crypto': {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum',
        'BNBUSDT': 'Binance Coin',
        'ADAUSDT': 'Cardano',
        'SOLUSDT': 'Solana'
    }
} 