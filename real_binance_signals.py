#!/usr/bin/env python3
"""
Real Binance Live Trading Signals
Connects to actual Binance APIs and shows real live trading signals
"""

import time
import json
import requests
import hmac
import hashlib
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class RealMarketData:
    """Real market data from Binance"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "binance_api"

class BinanceAPIClient:
    """Real Binance API client"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.binance.com"
        
        # Try to load from config if not provided
        if api_key is None or api_secret is None:
            try:
                from binance_config import get_binance_config
                config = get_binance_config()
                self.api_key = api_key or config.get("api_key", "")
                self.api_secret = api_secret or config.get("api_secret", "")
            except ImportError:
                self.api_key = api_key or ""
                self.api_secret = api_secret or ""
        else:
            self.api_key = api_key
            self.api_secret = api_secret
        
    def _generate_signature(self, params: str) -> str:
        """Generate HMAC signature for authenticated requests"""
        if not self.api_secret:
            return ""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make HTTP request to Binance API"""
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
            
        if signed and self.api_key:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            
            # Create query string
            query_string = urllib.parse.urlencode(params)
            signature = self._generate_signature(query_string)
            query_string += f"&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            response = requests.get(f"{url}?{query_string}", headers=headers)
        else:
            if params:
                query_string = urllib.parse.urlencode(params)
                response = requests.get(f"{url}?{query_string}")
            else:
                response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24-hour ticker statistics"""
        return self._make_request("/api/v3/ticker/24hr", {"symbol": symbol})
    
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List:
        """Get kline/candlestick data"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._make_request("/api/v3/klines", params)
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book"""
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return self._make_request("/api/v3/depth", params)
    
    def get_account_info(self) -> Dict:
        """Get account information (requires API key)"""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials required"}
        return self._make_request("/api/v3/account", signed=True)
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return self._make_request("/api/v3/exchangeInfo")

class RealBinanceSignalGenerator:
    """Generate trading signals from real Binance data"""
    
    def __init__(self):
        self.binance = BinanceAPIClient()
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: List[float]) -> Dict:
        """Calculate MACD"""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
            
        ema12 = sum(prices[-12:]) / 12
        ema26 = sum(prices[-26:]) / 26
        macd = ema12 - ema26
        
        # Simple signal line (9-period EMA of MACD)
        signal = macd  # Simplified for demo
        
        histogram = macd - signal
        
        return {
            "macd": macd,
            "signal": signal,
            "histogram": histogram
        }
    
    def generate_signal(self, market_data: RealMarketData, historical_data: List[RealMarketData]) -> Optional[Dict]:
        """Generate trading signal from real market data"""
        if len(historical_data) < 30:
            return None
            
        # Extract prices
        prices = [data.close for data in historical_data]
        current_price = market_data.close
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        macd_data = self.calculate_macd(prices)
        
        # Calculate moving averages
        sma20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else current_price
        sma50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else current_price
        
        # Signal conditions
        signal_type = None
        strength = 0.0
        conditions = []
        
        # RSI conditions
        if rsi < 30:
            signal_type = "BUY"
            strength += 0.3
            conditions.append("RSI oversold")
        elif rsi > 70:
            signal_type = "SELL"
            strength += 0.3
            conditions.append("RSI overbought")
        
        # MACD conditions
        if macd_data["histogram"] > 0 and macd_data["macd"] > 0:
            if signal_type == "BUY":
                strength += 0.2
                conditions.append("MACD bullish")
        elif macd_data["histogram"] < 0 and macd_data["macd"] < 0:
            if signal_type == "SELL":
                strength += 0.2
                conditions.append("MACD bearish")
        
        # Moving average conditions
        if current_price > sma20 > sma50:
            if signal_type == "BUY":
                strength += 0.2
                conditions.append("Price above MAs")
        elif current_price < sma20 < sma50:
            if signal_type == "SELL":
                strength += 0.2
                conditions.append("Price below MAs")
        
        # Volume confirmation
        avg_volume = sum([data.volume for data in historical_data[-10:]]) / 10
        if market_data.volume > avg_volume * 1.5:
            strength += 0.1
            conditions.append("High volume")
        
        # Only generate signal if strength is sufficient
        if signal_type and strength >= 0.4:
            return {
                "symbol": market_data.symbol,
                "signal_type": signal_type,
                "price": current_price,
                "strength": min(strength, 1.0),
                "confidence": strength,
                "risk_score": 1.0 - strength,
                "stop_loss": current_price * (0.95 if signal_type == "BUY" else 1.05),
                "take_profit": current_price * (1.05 if signal_type == "BUY" else 0.95),
                "conditions": conditions,
                "rsi": rsi,
                "macd": macd_data,
                "sma20": sma20,
                "sma50": sma50,
                "volume_ratio": market_data.volume / avg_volume if avg_volume > 0 else 1.0
            }
        
        return None

def show_real_binance_signals(duration_minutes=10, symbols=None):
    """Show real live trading signals from Binance"""
    if symbols is None:
        # Import configuration
        try:
            from binance_config import get_trading_pairs
            symbols = get_trading_pairs()
        except ImportError:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    print("🚀 REAL BINANCE LIVE TRADING SIGNALS")
    print("=" * 60)
    print(f"Duration: {duration_minutes} minutes")
    print(f"Symbols: {', '.join(symbols)}")
    print("Connecting to real Binance APIs...\n")
    
    # Initialize components
    binance_client = BinanceAPIClient()
    signal_generator = RealBinanceSignalGenerator()
    
    # Test connection
    try:
        exchange_info = binance_client.get_exchange_info()
        print("✅ Successfully connected to Binance API")
        print(f"   Exchange status: {exchange_info.get('status', 'unknown')}")
        print(f"   Server time: {datetime.fromtimestamp(exchange_info.get('serverTime', 0)/1000)}")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to Binance API: {e}")
        return
    
    # Initialize historical data storage
    historical_data = {symbol: [] for symbol in symbols}
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    signal_count = 0
    
    try:
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            for symbol in symbols:
                try:
                    # Get real-time data
                    ticker = binance_client.get_24hr_ticker(symbol)
                    klines = binance_client.get_klines(symbol, interval="1h", limit=50)
                    
                    # Parse current market data
                    current_data = RealMarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(klines[-1][0]/1000),
                        open=float(klines[-1][1]),
                        high=float(klines[-1][2]),
                        low=float(klines[-1][3]),
                        close=float(klines[-1][4]),
                        volume=float(klines[-1][5]),
                        source="binance_api"
                    )
                    
                    # Update historical data
                    historical_data[symbol].append(current_data)
                    
                    # Keep only last 100 data points
                    if len(historical_data[symbol]) > 100:
                        historical_data[symbol] = historical_data[symbol][-100:]
                    
                    # Generate signal if we have enough data
                    if len(historical_data[symbol]) >= 30:
                        signal = signal_generator.generate_signal(current_data, historical_data[symbol])
                        
                        if signal:
                            signal_count += 1
                            
                            # Get order book for additional analysis
                            try:
                                order_book = binance_client.get_order_book(symbol, limit=5)
                                bid_price = float(order_book['bids'][0][0])
                                ask_price = float(order_book['asks'][0][0])
                                spread = ((ask_price - bid_price) / bid_price) * 100
                            except:
                                bid_price = ask_price = current_data.close
                                spread = 0.0
                            
                            # Display comprehensive signal
                            print(f"🔔 REAL SIGNAL #{signal_count} - {current_time.strftime('%H:%M:%S')}")
                            print(f"   📊 Symbol: {signal['symbol']}")
                            print(f"   🎯 Type: {signal['signal_type']}")
                            print(f"   💰 Price: ${signal['price']:,.2f}")
                            print(f"   📈 Strength: {signal['strength']:.2f}")
                            print(f"   🎯 Confidence: {signal['confidence']:.2f}")
                            print(f"   ⚠️  Risk Score: {signal['risk_score']:.2f}")
                            print(f"   🛑 Stop Loss: ${signal['stop_loss']:,.2f}")
                            print(f"   🎯 Take Profit: ${signal['take_profit']:,.2f}")
                            print(f"   📊 RSI: {signal['rsi']:.1f}")
                            print(f"   📈 MACD: {signal['macd']['macd']:.4f}")
                            print(f"   📊 SMA20: ${signal['sma20']:,.2f}")
                            print(f"   📊 SMA50: ${signal['sma50']:,.2f}")
                            print(f"   📊 Volume Ratio: {signal['volume_ratio']:.2f}")
                            print(f"   💰 Bid: ${bid_price:,.2f} | Ask: ${ask_price:,.2f}")
                            print(f"   📊 Spread: {spread:.3f}%")
                            print(f"   📋 Conditions: {', '.join(signal['conditions'])}")
                            
                            # Calculate potential profit
                            if signal['signal_type'] == 'BUY':
                                potential_profit = ((signal['take_profit'] - signal['price']) / signal['price']) * 100
                                potential_loss = ((signal['price'] - signal['stop_loss']) / signal['price']) * 100
                            else:
                                potential_profit = ((signal['price'] - signal['take_profit']) / signal['price']) * 100
                                potential_loss = ((signal['stop_loss'] - signal['price']) / signal['price']) * 100
                            
                            print(f"   💰 Potential Profit: {potential_profit:.2f}%")
                            print(f"   ⚠️  Potential Loss: {potential_loss:.2f}%")
                            print(f"   📊 Risk/Reward: {potential_profit/potential_loss:.2f}:1")
                            
                            print("-" * 60)
                    
                    # Show market status every 30 seconds
                    if signal_count % 3 == 0 and signal_count > 0:
                        print(f"📊 MARKET STATUS - {current_time.strftime('%H:%M:%S')}")
                        print(f"   📈 {symbol} 24h Change: {float(ticker['priceChangePercent']):+.2f}%")
                        print(f"   💰 24h High: ${float(ticker['highPrice']):,.2f}")
                        print(f"   📉 24h Low: ${float(ticker['lowPrice']):,.2f}")
                        print(f"   📊 24h Volume: {float(ticker['volume']):,.0f}")
                        print()
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    continue
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Real signals stopped by user")
    
    # Final summary
    print(f"\n📈 REAL BINANCE SIGNALS SUMMARY")
    print(f"   Total Signals Generated: {signal_count}")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Average Signals/Minute: {signal_count/duration_minutes:.1f}")
    print(f"   Data Source: Real Binance APIs")
    print(f"   Symbols Monitored: {', '.join(symbols)}")

def main():
    """Main function"""
    print("🎯 REAL BINANCE LIVE TRADING SIGNALS")
    print("=" * 60)
    print("⚠️  DISCLAIMER: This is for educational purposes only!")
    print("   Do not use for actual trading without proper risk management.")
    print("   Always test with small amounts first.\n")
    
    try:
        # Show real signals for 5 minutes
        show_real_binance_signals(duration_minutes=5)
        
        print("\n✅ Real Binance signals demonstration completed!")
        print("\n💡 To use with your own API keys:")
        print("   1. Create API keys in your Binance account")
        print("   2. Update the BinanceAPIClient initialization")
        print("   3. Enable trading permissions (optional)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 