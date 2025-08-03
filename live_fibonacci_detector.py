"""
Live Fibonacci Retracement and Extension Detector
Real-time detection and visualization of Fibonacci levels from live OHLCV data
"""

import pandas as pd
import numpy as np
import ccxt
import time
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import requests
import json
import os
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fibonacci_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevel:
    """Represents a Fibonacci retracement or extension level"""
    level_type: str  # 'retracement' or 'extension'
    percentage: float  # 23.6, 38.2, 50.0, 61.8, 78.6, 127.2, 161.8, 200.0
    price: float
    zone_high: float
    zone_low: float
    strength: float  # 0-1 based on touches and volume
    touches: int
    last_touch: Optional[datetime] = None

@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    point_type: str  # 'high' or 'low'
    price: float
    timestamp: datetime
    volume: float
    strength: float  # Based on surrounding candles

class LiveFibonacciDetector:
    """
    Real-time Fibonacci retracement and extension detector
    """
    
    def __init__(self, 
                 exchange_name: str = 'binance',
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '5m',
                 lookback_periods: int = 200,
                 buffer_percentage: float = 0.003,
                 min_swing_strength: float = 0.6,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None):
        """
        Initialize the Fibonacci detector
        
        Args:
            exchange_name: Exchange name (binance, bybit, etc.)
            symbol: Trading pair symbol
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            lookback_periods: Number of candles to maintain
            buffer_percentage: Price buffer for zones (±0.3% default)
            min_swing_strength: Minimum strength for swing points
            telegram_token: Telegram bot token for alerts
            telegram_chat_id: Telegram chat ID for alerts
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.buffer_percentage = buffer_percentage
        self.min_swing_strength = min_swing_strength
        
        # Telegram settings
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Data storage
        self.data = pd.DataFrame()
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.fibonacci_levels: List[FibonacciLevel] = []
        
        # Current state
        self.current_price = 0.0
        self.is_running = False
        self.last_update = None
        
        # Initialize exchange
        self._setup_exchange()
        
        # Fibonacci ratios
        self.retracement_levels = [23.6, 38.2, 50.0, 61.8, 78.6]
        self.extension_levels = [127.2, 161.8, 200.0]
        
        logger.info(f"Fibonacci detector initialized for {symbol} on {timeframe}")

    def _setup_exchange(self):
        """Setup exchange connection"""
        try:
            if self.exchange_name == 'binance':
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            elif self.exchange_name == 'bybit':
                self.exchange = ccxt.bybit({
                    'enableRateLimit': True
                })
            else:
                self.exchange = getattr(ccxt, self.exchange_name)({
                    'enableRateLimit': True
                })
            
            logger.info(f"Connected to {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            raise

    def fetch_live_data(self) -> pd.DataFrame:
        """Fetch live OHLCV data from exchange"""
        try:
            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=self.lookback_periods
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Update current price
            self.current_price = df['close'].iloc[-1]
            self.last_update = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return pd.DataFrame()

    def detect_swing_points(self, data: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows in the data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        swing_highs = []
        swing_lows = []
        
        if len(data) < 5:
            return swing_highs, swing_lows
        
        # Detect swing highs
        for i in range(2, len(data) - 2):
            high = data['high'].iloc[i]
            high_prev = data['high'].iloc[i-1]
            high_next = data['high'].iloc[i+1]
            high_prev2 = data['high'].iloc[i-2]
            high_next2 = data['high'].iloc[i+2]
            
            # Check if this is a local maximum
            if (high > high_prev and high > high_next and 
                high > high_prev2 and high > high_next2):
                
                # Calculate strength based on surrounding candles
                strength = self._calculate_swing_strength(data, i, 'high')
                
                if strength >= self.min_swing_strength:
                    swing_highs.append(SwingPoint(
                        point_type='high',
                        price=high,
                        timestamp=data.index[i],
                        volume=data['volume'].iloc[i],
                        strength=strength
                    ))
        
        # Detect swing lows
        for i in range(2, len(data) - 2):
            low = data['low'].iloc[i]
            low_prev = data['low'].iloc[i-1]
            low_next = data['low'].iloc[i+1]
            low_prev2 = data['low'].iloc[i-2]
            low_next2 = data['low'].iloc[i+2]
            
            # Check if this is a local minimum
            if (low < low_prev and low < low_next and 
                low < low_prev2 and low < low_next2):
                
                # Calculate strength based on surrounding candles
                strength = self._calculate_swing_strength(data, i, 'low')
                
                if strength >= self.min_swing_strength:
                    swing_lows.append(SwingPoint(
                        point_type='low',
                        price=low,
                        timestamp=data.index[i],
                        volume=data['volume'].iloc[i],
                        strength=strength
                    ))
        
        return swing_highs, swing_lows

    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, point_type: str) -> float:
        """
        Calculate the strength of a swing point
        
        Args:
            data: OHLCV DataFrame
            index: Index of the swing point
            point_type: 'high' or 'low'
            
        Returns:
            Strength value between 0 and 1
        """
        if point_type == 'high':
            current = data['high'].iloc[index]
            prev_avg = (data['high'].iloc[index-2] + data['high'].iloc[index-1]) / 2
            next_avg = (data['high'].iloc[index+1] + data['high'].iloc[index+2]) / 2
            
            # Calculate price difference
            price_diff = min(current - prev_avg, current - next_avg)
            
            # Calculate volume ratio
            current_volume = data['volume'].iloc[index]
            avg_volume = data['volume'].iloc[index-2:index+3].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
        else:  # low
            current = data['low'].iloc[index]
            prev_avg = (data['low'].iloc[index-2] + data['low'].iloc[index-1]) / 2
            next_avg = (data['low'].iloc[index+1] + data['low'].iloc[index+2]) / 2
            
            # Calculate price difference
            price_diff = min(prev_avg - current, next_avg - current)
            
            # Calculate volume ratio
            current_volume = data['volume'].iloc[index]
            avg_volume = data['volume'].iloc[index-2:index+3].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Normalize price difference
        price_strength = min(price_diff / current * 100, 1.0)
        
        # Combine price and volume strength
        strength = (price_strength * 0.7 + min(volume_ratio, 2.0) * 0.3) / 2
        
        return max(0, min(1, strength))

    def calculate_fibonacci_levels(self, swing_high: SwingPoint, swing_low: SwingPoint) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            swing_high: Most recent swing high
            swing_low: Most recent swing low
            
        Returns:
            List of Fibonacci levels
        """
        levels = []
        
        # Ensure swing_high is higher than swing_low
        if swing_high.price <= swing_low.price:
            return levels
        
        high_price = swing_high.price
        low_price = swing_low.price
        price_range = high_price - low_price
        
        # Calculate retracement levels
        for percentage in self.retracement_levels:
            level_price = high_price - (price_range * percentage / 100)
            zone_high = level_price * (1 + self.buffer_percentage)
            zone_low = level_price * (1 - self.buffer_percentage)
            
            levels.append(FibonacciLevel(
                level_type='retracement',
                percentage=percentage,
                price=level_price,
                zone_high=zone_high,
                zone_low=zone_low,
                strength=0.5,  # Will be updated based on touches
                touches=0
            ))
        
        # Calculate extension levels
        for percentage in self.extension_levels:
            level_price = high_price + (price_range * (percentage - 100) / 100)
            zone_high = level_price * (1 + self.buffer_percentage)
            zone_low = level_price * (1 - self.buffer_percentage)
            
            levels.append(FibonacciLevel(
                level_type='extension',
                percentage=percentage,
                price=level_price,
                zone_high=zone_high,
                zone_low=zone_low,
                strength=0.5,  # Will be updated based on touches
                touches=0
            ))
        
        return levels

    def update_fibonacci_levels(self, data: pd.DataFrame):
        """Update Fibonacci levels based on current swing points"""
        # Detect swing points
        swing_highs, swing_lows = self.detect_swing_points(data)
        
        if not swing_highs or not swing_lows:
            return
        
        # Get most recent swing high and low
        latest_high = max(swing_highs, key=lambda x: x.timestamp)
        latest_low = max(swing_lows, key=lambda x: x.timestamp)
        
        # Update stored swing points
        self.swing_highs = swing_highs
        self.swing_lows = swing_lows
        
        # Calculate new Fibonacci levels
        new_levels = self.calculate_fibonacci_levels(latest_high, latest_low)
        
        # Update existing levels with touches
        for new_level in new_levels:
            for existing_level in self.fibonacci_levels:
                if (existing_level.level_type == new_level.level_type and 
                    abs(existing_level.percentage - new_level.percentage) < 0.1):
                    # Check if price touched this level
                    if self._check_level_touch(data, existing_level):
                        existing_level.touches += 1
                        existing_level.last_touch = datetime.now()
                        existing_level.strength = min(1.0, existing_level.touches * 0.2)
        
        # Add new levels
        self.fibonacci_levels.extend(new_levels)
        
        # Remove old levels (keep only last 20)
        if len(self.fibonacci_levels) > 20:
            self.fibonacci_levels = sorted(
                self.fibonacci_levels, 
                key=lambda x: x.last_touch or datetime.min
            )[-20:]

    def _check_level_touch(self, data: pd.DataFrame, level: FibonacciLevel) -> bool:
        """Check if price touched a Fibonacci level"""
        recent_data = data.tail(10)  # Check last 10 candles
        
        for _, candle in recent_data.iterrows():
            if (candle['low'] <= level.zone_high and 
                candle['high'] >= level.zone_low):
                return True
        
        return False

    def check_price_alerts(self) -> List[Dict]:
        """Check if current price triggers any alerts"""
        alerts = []
        
        for level in self.fibonacci_levels:
            # Check if price is approaching level
            if (self.current_price >= level.zone_low * 0.995 and 
                self.current_price <= level.zone_high * 1.005):
                
                alert_type = 'approaching'
                if (self.current_price >= level.zone_low and 
                    self.current_price <= level.zone_high):
                    alert_type = 'touching'
                
                alerts.append({
                    'type': alert_type,
                    'level': level,
                    'current_price': self.current_price,
                    'timestamp': datetime.now()
                })
        
        return alerts

    def send_telegram_alert(self, alert: Dict):
        """Send alert to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        level = alert['level']
        
        message = f"""
🚨 FIBONACCI ALERT - {self.symbol}

{level.level_type.title()} Level: {level.percentage}%
Price: ${level.price:.4f}
Zone: ${level.zone_low:.4f} - ${level.zone_high:.4f}
Current Price: ${alert['current_price']:.4f}
Status: {alert['type'].title()}
Strength: {level.strength:.2f}
Touches: {level.touches}

Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert['type']} {level.percentage}%")
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")

    def export_levels_to_csv(self, filename: str = None):
        """Export Fibonacci levels to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fibonacci_levels_{self.symbol.replace('/', '_')}_{timestamp}.csv"
        
        data = []
        for level in self.fibonacci_levels:
            data.append({
                'level_type': level.level_type,
                'percentage': level.percentage,
                'price': level.price,
                'zone_high': level.zone_high,
                'zone_low': level.zone_low,
                'strength': level.strength,
                'touches': level.touches,
                'last_touch': level.last_touch.isoformat() if level.last_touch else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Fibonacci levels exported to {filename}")

    def create_chart(self, save_path: str = None):
        """Create and save a chart with Fibonacci levels"""
        if self.data.empty:
            logger.warning("No data available for charting")
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot candlestick data
        ax.plot(self.data.index, self.data['close'], label='Price', color='blue', linewidth=1)
        
        # Plot swing points
        for swing in self.swing_highs:
            ax.scatter(swing.timestamp, swing.price, color='red', s=50, marker='^', alpha=0.7)
        
        for swing in self.swing_lows:
            ax.scatter(swing.timestamp, swing.price, color='green', s=50, marker='v', alpha=0.7)
        
        # Plot Fibonacci levels
        colors = ['purple', 'orange', 'red', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
        for i, level in enumerate(self.fibonacci_levels):
            color = colors[i % len(colors)]
            ax.axhline(y=level.price, color=color, linestyle='--', alpha=0.7, 
                      label=f"{level.level_type.title()} {level.percentage}%")
            
            # Add zone shading
            ax.axhspan(level.zone_low, level.zone_high, alpha=0.1, color=color)
        
        # Current price line
        ax.axhline(y=self.current_price, color='black', linestyle='-', linewidth=2, 
                  label=f'Current Price: ${self.current_price:.4f}')
        
        ax.set_title(f'Fibonacci Levels - {self.symbol} ({self.timeframe})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def print_status(self):
        """Print current status"""
        print("\n" + "="*60)
        print(f"📊 FIBONACCI DETECTOR STATUS - {self.symbol}")
        print("="*60)
        print(f"⏰ Last Update: {self.last_update}")
        print(f"💰 Current Price: ${self.current_price:.4f}")
        print(f"📈 Data Points: {len(self.data)}")
        print(f"🎯 Swing Highs: {len(self.swing_highs)}")
        print(f"🎯 Swing Lows: {len(self.swing_lows)}")
        print(f"📐 Fibonacci Levels: {len(self.fibonacci_levels)}")
        
        if self.swing_highs and self.swing_lows:
            latest_high = max(self.swing_highs, key=lambda x: x.timestamp)
            latest_low = max(self.swing_lows, key=lambda x: x.timestamp)
            print(f"📈 Latest Swing High: ${latest_high.price:.4f} ({latest_high.timestamp})")
            print(f"📉 Latest Swing Low: ${latest_low.price:.4f} ({latest_low.timestamp})")
        
        print("\n🔍 ACTIVE FIBONACCI LEVELS:")
        print("-" * 60)
        
        for level in sorted(self.fibonacci_levels, key=lambda x: x.price, reverse=True):
            status = "🟢" if self.current_price >= level.zone_low and self.current_price <= level.zone_high else "⚪"
            print(f"{status} {level.level_type.title()} {level.percentage}%: "
                  f"${level.price:.4f} (Zone: ${level.zone_low:.4f}-${level.zone_high:.4f}) "
                  f"[Strength: {level.strength:.2f}, Touches: {level.touches}]")
        
        print("="*60)

    def start_monitoring(self, update_interval: int = 60):
        """
        Start real-time monitoring
        
        Args:
            update_interval: Update interval in seconds
        """
        self.is_running = True
        logger.info(f"Starting Fibonacci monitoring for {self.symbol}")
        
        def monitor_loop():
            while self.is_running:
                try:
                    # Fetch new data
                    self.data = self.fetch_live_data()
                    
                    if not self.data.empty:
                        # Update Fibonacci levels
                        self.update_fibonacci_levels(self.data)
                        
                        # Check for alerts
                        alerts = self.check_price_alerts()
                        
                        for alert in alerts:
                            logger.info(f"Alert: {alert['type']} {alert['level'].percentage}% level")
                            self.send_telegram_alert(alert)
                        
                        # Print status every 5 minutes
                        if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
                            self.print_status()
                    
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(update_interval)
        
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        logger.info("Fibonacci monitoring stopped")

def main():
    """Main function for interactive setup"""
    print("🚀 LIVE FIBONACCI DETECTOR")
    print("="*60)
    print("Real-time Fibonacci retracement and extension detection")
    print("="*60)
    
    # Get user input
    exchange = input("Enter exchange (binance/bybit): ").strip() or 'binance'
    symbol = input("Enter symbol (e.g., BTC/USDT): ").strip() or 'BTC/USDT'
    timeframe = input("Enter timeframe (1m/5m/15m/1h/4h): ").strip() or '5m'
    
    # Optional Telegram setup
    telegram_token = input("Enter Telegram bot token (optional): ").strip() or None
    telegram_chat_id = input("Enter Telegram chat ID (optional): ").strip() or None
    
    # Create detector
    detector = LiveFibonacciDetector(
        exchange_name=exchange,
        symbol=symbol,
        timeframe=timeframe,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id
    )
    
    print(f"\n✅ Detector created for {symbol} on {timeframe}")
    print("\n📋 Available commands:")
    print("1. Start monitoring")
    print("2. Print current status")
    print("3. Create chart")
    print("4. Export levels to CSV")
    print("5. Stop monitoring")
    print("6. Exit")
    
    while True:
        try:
            command = input("\nEnter command (1-6): ").strip()
            
            if command == '1':
                detector.start_monitoring()
                print("✅ Monitoring started. Press Ctrl+C to stop.")
                
            elif command == '2':
                detector.print_status()
                
            elif command == '3':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"fibonacci_chart_{symbol.replace('/', '_')}_{timestamp}.png"
                detector.create_chart(filename)
                
            elif command == '4':
                detector.export_levels_to_csv()
                
            elif command == '5':
                detector.stop_monitoring()
                print("✅ Monitoring stopped")
                
            elif command == '6':
                detector.stop_monitoring()
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid command")
                
        except KeyboardInterrupt:
            detector.stop_monitoring()
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 