"""
Live Support and Resistance Detector
Real-time detection of support and resistance levels using live market data

Features:
- Live data streaming from Binance
- Swing high/low detection with multiple touch confirmations
- Volume spike analysis
- Price zone detection with buffers
- Real-time alerts and notifications
- Optional chart visualization
- Export capabilities
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
from collections import deque
import requests
import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_resistance_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SupportResistanceZone:
    """Data class for support/resistance zones"""
    level: float
    zone_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 scale
    touches: int
    first_touch: datetime
    last_touch: datetime
    volume_confirmed: bool
    price_range: Tuple[float, float]  # (min, max) with buffer
    is_active: bool = True
    break_count: int = 0
    last_break: Optional[datetime] = None

class LiveSupportResistanceDetector:
    """
    Real-time support and resistance level detector
    """
    
    def __init__(self, 
                 symbol: str = 'BTC/USDT',
                 exchange: str = 'binance',
                 timeframe: str = '5m',
                 window_size: int = 200,
                 min_touches: int = 2,
                 zone_buffer: float = 0.003,  # 0.3%
                 volume_threshold: float = 1.5,
                 swing_sensitivity: float = 0.02,  # 2%
                 enable_charts: bool = False,
                 enable_alerts: bool = True,
                 telegram_bot_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None):
        """
        Initialize the detector
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Candle timeframe (1m, 5m, 15m, etc.)
            window_size: Number of candles to keep in memory
            min_touches: Minimum touches required for zone confirmation
            zone_buffer: Price buffer around zone level (±%)
            volume_threshold: Volume spike multiplier for confirmation
            swing_sensitivity: Minimum swing size for detection
            enable_charts: Enable real-time chart visualization
            enable_alerts: Enable price alerts
            telegram_bot_token: Telegram bot token for alerts
            telegram_chat_id: Telegram chat ID for alerts
        """
        self.symbol = symbol
        self.exchange_name = exchange
        self.timeframe = timeframe
        self.window_size = window_size
        self.min_touches = min_touches
        self.zone_buffer = zone_buffer
        self.volume_threshold = volume_threshold
        self.swing_sensitivity = swing_sensitivity
        self.enable_charts = enable_charts
        self.enable_alerts = enable_alerts
        
        # Initialize exchange connection
        self.exchange = getattr(ccxt, exchange)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Data storage
        self.candles = deque(maxlen=window_size)
        self.support_zones: List[SupportResistanceZone] = []
        self.resistance_zones: List[SupportResistanceZone] = []
        self.current_price = None
        self.last_update = None
        
        # Threading
        self.running = False
        self.data_thread = None
        self.alert_thread = None
        
        # Telegram setup
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        # Chart setup
        if self.enable_charts:
            self.setup_chart()
        
        logger.info(f"Initialized detector for {symbol} on {exchange} ({timeframe})")
    
    def setup_chart(self):
        """Setup real-time chart"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.fig.suptitle(f'Live Support/Resistance - {self.symbol}', fontsize=16)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price')
        self.ax.grid(True, alpha=0.3)
    
    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        """Fetch live OHLCV data from exchange"""
        try:
            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=self.window_size
            )
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def detect_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Detect swing highs and lows in the data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (swing_highs, swing_lows) indices
        """
        highs = []
        lows = []
        
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                # Check if swing is significant enough
                swing_size = (data['high'].iloc[i] - min(data['low'].iloc[i-2:i+3])) / data['close'].iloc[i]
                if swing_size >= self.swing_sensitivity:
                    highs.append(i)
            
            # Check for swing low
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                # Check if swing is significant enough
                swing_size = (max(data['high'].iloc[i-2:i+3]) - data['low'].iloc[i]) / data['close'].iloc[i]
                if swing_size >= self.swing_sensitivity:
                    lows.append(i)
        
        return highs, lows
    
    def check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
        """
        Check if swing point has volume confirmation
        
        Args:
            data: OHLCV DataFrame
            index: Index of swing point
            
        Returns:
            True if volume spike detected
        """
        if index < 5 or index >= len(data) - 5:
            return False
        
        # Calculate average volume around the swing point
        avg_volume = data['volume'].iloc[index-5:index+6].mean()
        swing_volume = data['volume'].iloc[index]
        
        return swing_volume > (avg_volume * self.volume_threshold)
    
    def find_zone_touches(self, level: float, data: pd.DataFrame, swing_indices: List[int]) -> List[int]:
        """
        Find all touches of a price level
        
        Args:
            level: Price level to check
            data: OHLCV DataFrame
            swing_indices: List of swing point indices
            
        Returns:
            List of indices where level was touched
        """
        touches = []
        buffer = level * self.zone_buffer
        
        for idx in swing_indices:
            high = data['high'].iloc[idx]
            low = data['low'].iloc[idx]
            
            # Check if price touched the level (within buffer)
            if low <= (level + buffer) and high >= (level - buffer):
                touches.append(idx)
        
        return touches
    
    def identify_zones(self, data: pd.DataFrame) -> Tuple[List[SupportResistanceZone], List[SupportResistanceZone]]:
        """
        Identify support and resistance zones
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (support_zones, resistance_zones)
        """
        swing_highs, swing_lows = self.detect_swing_points(data)
        
        support_zones = []
        resistance_zones = []
        
        # Find resistance zones from swing highs
        for high_idx in swing_highs:
            level = data['high'].iloc[high_idx]
            touches = self.find_zone_touches(level, data, swing_highs)
            
            if len(touches) >= self.min_touches:
                # Check if this zone already exists
                existing_zone = None
                for zone in resistance_zones:
                    if abs(zone.level - level) <= (level * self.zone_buffer):
                        existing_zone = zone
                        break
                
                if existing_zone:
                    # Update existing zone
                    existing_zone.touches = len(touches)
                    existing_zone.last_touch = data.index[high_idx]
                    existing_zone.strength = min(1.0, len(touches) / 5.0)  # Cap at 1.0
                else:
                    # Create new zone
                    volume_confirmed = self.check_volume_confirmation(data, high_idx)
                    zone = SupportResistanceZone(
                        level=level,
                        zone_type='resistance',
                        strength=min(1.0, len(touches) / 5.0),
                        touches=len(touches),
                        first_touch=data.index[touches[0]],
                        last_touch=data.index[high_idx],
                        volume_confirmed=volume_confirmed,
                        price_range=(level * (1 - self.zone_buffer), level * (1 + self.zone_buffer))
                    )
                    resistance_zones.append(zone)
        
        # Find support zones from swing lows
        for low_idx in swing_lows:
            level = data['low'].iloc[low_idx]
            touches = self.find_zone_touches(level, data, swing_lows)
            
            if len(touches) >= self.min_touches:
                # Check if this zone already exists
                existing_zone = None
                for zone in support_zones:
                    if abs(zone.level - level) <= (level * self.zone_buffer):
                        existing_zone = zone
                        break
                
                if existing_zone:
                    # Update existing zone
                    existing_zone.touches = len(touches)
                    existing_zone.last_touch = data.index[low_idx]
                    existing_zone.strength = min(1.0, len(touches) / 5.0)
                else:
                    # Create new zone
                    volume_confirmed = self.check_volume_confirmation(data, low_idx)
                    zone = SupportResistanceZone(
                        level=level,
                        zone_type='support',
                        strength=min(1.0, len(touches) / 5.0),
                        touches=len(touches),
                        first_touch=data.index[touches[0]],
                        last_touch=data.index[low_idx],
                        volume_confirmed=volume_confirmed,
                        price_range=(level * (1 - self.zone_buffer), level * (1 + self.zone_buffer))
                    )
                    support_zones.append(zone)
        
        return support_zones, resistance_zones
    
    def check_zone_breaks(self, current_price: float) -> List[Dict]:
        """
        Check if current price breaks any zones
        
        Args:
            current_price: Current market price
            
        Returns:
            List of break alerts
        """
        breaks = []
        
        # Check support breaks (price goes below support)
        for zone in self.support_zones:
            if zone.is_active and current_price < zone.price_range[0]:
                zone.break_count += 1
                zone.last_break = datetime.now()
                breaks.append({
                    'type': 'support_break',
                    'zone': zone,
                    'price': current_price,
                    'strength': zone.strength
                })
        
        # Check resistance breaks (price goes above resistance)
        for zone in self.resistance_zones:
            if zone.is_active and current_price > zone.price_range[1]:
                zone.break_count += 1
                zone.last_break = datetime.now()
                breaks.append({
                    'type': 'resistance_break',
                    'zone': zone,
                    'price': current_price,
                    'strength': zone.strength
                })
        
        return breaks
    
    def check_zone_approaches(self, current_price: float) -> List[Dict]:
        """
        Check if price is approaching zones
        
        Args:
            current_price: Current market price
            
        Returns:
            List of approach alerts
        """
        approaches = []
        approach_threshold = 0.01  # 1% from zone
        
        # Check support approaches
        for zone in self.support_zones:
            if zone.is_active:
                distance = (zone.level - current_price) / zone.level
                if 0 < distance <= approach_threshold:
                    approaches.append({
                        'type': 'support_approach',
                        'zone': zone,
                        'price': current_price,
                        'distance': distance
                    })
        
        # Check resistance approaches
        for zone in self.resistance_zones:
            if zone.is_active:
                distance = (current_price - zone.level) / zone.level
                if 0 < distance <= approach_threshold:
                    approaches.append({
                        'type': 'resistance_approach',
                        'zone': zone,
                        'price': current_price,
                        'distance': distance
                    })
        
        return approaches
    
    def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def send_alert(self, alert_type: str, data: Dict):
        """Send alert based on type"""
        if not self.enable_alerts:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if alert_type == 'break':
            zone = data['zone']
            message = f"""
🚨 <b>Zone Break Alert</b> 🚨

Symbol: {self.symbol}
Type: {data['type'].replace('_', ' ').title()}
Price: ${data['price']:.4f}
Zone Level: ${zone.level:.4f}
Strength: {zone.strength:.2f}
Touches: {zone.touches}
Time: {timestamp}
            """
        elif alert_type == 'approach':
            zone = data['zone']
            distance_pct = data['distance'] * 100
            message = f"""
⚠️ <b>Zone Approach Alert</b> ⚠️

Symbol: {self.symbol}
Type: {data['type'].replace('_', ' ').title()}
Price: ${data['price']:.4f}
Zone Level: ${zone.level:.4f}
Distance: {distance_pct:.2f}%
Strength: {zone.strength:.2f}
Time: {timestamp}
            """
        else:
            return
        
        # Send to console
        print(f"\n{message}")
        
        # Send to Telegram
        self.send_telegram_alert(message)
    
    def update_chart(self, data: pd.DataFrame):
        """Update real-time chart"""
        if not self.enable_charts:
            return
        
        try:
            self.ax.clear()
            
            # Plot candlesticks
            self.ax.plot(data.index, data['close'], 'b-', alpha=0.7, linewidth=1)
            
            # Plot support zones
            for zone in self.support_zones:
                if zone.is_active:
                    color = 'g' if zone.volume_confirmed else 'lightgreen'
                    alpha = 0.3 + (zone.strength * 0.4)
                    self.ax.axhline(y=zone.level, color=color, alpha=alpha, linestyle='--', 
                                  label=f'Support ${zone.level:.2f} (S:{zone.strength:.2f})')
            
            # Plot resistance zones
            for zone in self.resistance_zones:
                if zone.is_active:
                    color = 'r' if zone.volume_confirmed else 'lightcoral'
                    alpha = 0.3 + (zone.strength * 0.4)
                    self.ax.axhline(y=zone.level, color=color, alpha=alpha, linestyle='--',
                                  label=f'Resistance ${zone.level:.2f} (S:{zone.strength:.2f})')
            
            # Plot current price
            if self.current_price:
                self.ax.axhline(y=self.current_price, color='orange', alpha=0.8, linewidth=2,
                              label=f'Current Price ${self.current_price:.2f}')
            
            self.ax.set_title(f'Live Support/Resistance - {self.symbol} ({self.timeframe})')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Price')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            logger.error(f"Error updating chart: {e}")
    
    def export_zones(self, filename: str = None):
        """Export zones to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"zones_{self.symbol.replace('/', '_')}_{timestamp}.json"
        
        zones_data = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'export_time': datetime.now().isoformat(),
            'support_zones': [],
            'resistance_zones': []
        }
        
        for zone in self.support_zones:
            zones_data['support_zones'].append({
                'level': zone.level,
                'strength': zone.strength,
                'touches': zone.touches,
                'first_touch': zone.first_touch.isoformat(),
                'last_touch': zone.last_touch.isoformat(),
                'volume_confirmed': zone.volume_confirmed,
                'price_range': zone.price_range,
                'is_active': zone.is_active,
                'break_count': zone.break_count
            })
        
        for zone in self.resistance_zones:
            zones_data['resistance_zones'].append({
                'level': zone.level,
                'strength': zone.strength,
                'touches': zone.touches,
                'first_touch': zone.first_touch.isoformat(),
                'last_touch': zone.last_touch.isoformat(),
                'volume_confirmed': zone.volume_confirmed,
                'price_range': zone.price_range,
                'is_active': zone.is_active,
                'break_count': zone.break_count
            })
        
        try:
            with open(filename, 'w') as f:
                json.dump(zones_data, f, indent=2)
            logger.info(f"Zones exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting zones: {e}")
    
    def print_status(self):
        """Print current status"""
        print(f"\n{'='*60}")
        print(f"📊 LIVE SUPPORT/RESISTANCE DETECTOR")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Exchange: {self.exchange_name}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Current Price: ${self.current_price:.4f}" if self.current_price else "Current Price: N/A")
        print(f"Last Update: {self.last_update}" if self.last_update else "Last Update: N/A")
        print(f"Data Points: {len(self.candles)}")
        
        print(f"\n🛡️ SUPPORT ZONES ({len(self.support_zones)}):")
        for i, zone in enumerate(self.support_zones, 1):
            status = "✅" if zone.is_active else "❌"
            volume_icon = "📈" if zone.volume_confirmed else "📊"
            print(f"  {i}. {status} ${zone.level:.4f} | S:{zone.strength:.2f} | T:{zone.touches} | {volume_icon}")
        
        print(f"\n🚀 RESISTANCE ZONES ({len(self.resistance_zones)}):")
        for i, zone in enumerate(self.resistance_zones, 1):
            status = "✅" if zone.is_active else "❌"
            volume_icon = "📈" if zone.volume_confirmed else "📊"
            print(f"  {i}. {status} ${zone.level:.4f} | S:{zone.strength:.2f} | T:{zone.touches} | {volume_icon}")
        
        print(f"{'='*60}")
    
    def data_loop(self):
        """Main data processing loop"""
        while self.running:
            try:
                # Fetch live data
                data = self.fetch_live_data()
                if data is None:
                    time.sleep(5)
                    continue
                
                # Update current price
                self.current_price = data['close'].iloc[-1]
                self.last_update = datetime.now()
                
                # Store data
                self.candles.extend(data.to_dict('records'))
                
                # Identify zones
                support_zones, resistance_zones = self.identify_zones(data)
                
                # Update zone lists
                self.support_zones = support_zones
                self.resistance_zones = resistance_zones
                
                # Check for breaks and approaches
                breaks = self.check_zone_breaks(self.current_price)
                approaches = self.check_zone_approaches(self.current_price)
                
                # Send alerts
                for break_alert in breaks:
                    self.send_alert('break', break_alert)
                
                for approach_alert in approaches:
                    self.send_alert('approach', approach_alert)
                
                # Update chart
                if self.enable_charts:
                    self.update_chart(data)
                
                # Sleep based on timeframe
                if self.timeframe == '1m':
                    time.sleep(30)  # Update every 30 seconds
                elif self.timeframe == '5m':
                    time.sleep(60)  # Update every minute
                else:
                    time.sleep(120)  # Update every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in data loop: {e}")
                time.sleep(10)
    
    def start(self):
        """Start the detector"""
        if self.running:
            logger.warning("Detector is already running")
            return
        
        self.running = True
        self.data_thread = threading.Thread(target=self.data_loop, daemon=True)
        self.data_thread.start()
        
        logger.info(f"Started live support/resistance detector for {self.symbol}")
        
        # Print initial status
        self.print_status()
        
        # Main loop for status updates
        try:
            while self.running:
                time.sleep(30)  # Print status every 30 seconds
                self.print_status()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the detector"""
        self.running = False
        if self.enable_charts:
            plt.ioff()
            plt.close()
        logger.info("Detector stopped")

def main():
    """Main function to run the detector"""
    print("🚀 LIVE SUPPORT/RESISTANCE DETECTOR")
    print("="*50)
    
    # Configuration
    symbol = input("Enter symbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Enter timeframe (1m/5m/15m, default: 5m): ").strip() or "5m"
    enable_charts = input("Enable charts? (y/n, default: n): ").strip().lower() == 'y'
    enable_alerts = input("Enable alerts? (y/n, default: y): ").strip().lower() != 'n'
    
    # Telegram setup (optional)
    telegram_bot_token = None
    telegram_chat_id = None
    if enable_alerts:
        use_telegram = input("Use Telegram alerts? (y/n, default: n): ").strip().lower() == 'y'
        if use_telegram:
            telegram_bot_token = input("Enter Telegram bot token: ").strip()
            telegram_chat_id = input("Enter Telegram chat ID: ").strip()
    
    # Create detector
    detector = LiveSupportResistanceDetector(
        symbol=symbol,
        timeframe=timeframe,
        enable_charts=enable_charts,
        enable_alerts=enable_alerts,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id
    )
    
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping detector...")
        detector.stop()
        
        # Export zones
        export = input("\nExport zones to file? (y/n): ").strip().lower() == 'y'
        if export:
            detector.export_zones()

if __name__ == "__main__":
    main() 