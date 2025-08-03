"""
Futures Leverage Fetcher
Comprehensive system to fetch all available leverage options from Binance Futures
"""

import requests
import pandas as pd
import ccxt
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

class FuturesLeverageFetcher:
    """Comprehensive futures leverage fetching system"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize the leverage fetcher
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet or live
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize exchange connection
        self.exchange = self._setup_exchange()
        
        # Cache for leverage data
        self.leverage_cache = {}
        self.last_fetch_time = None
        
    def _setup_exchange(self):
        """Setup exchange connection"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            }
            
            if self.api_key and self.api_secret:
                exchange_config.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret
                })
            
            if self.testnet:
                exchange_config['sandbox'] = True
                print("🔧 Using Binance Testnet for leverage fetching")
            else:
                print("🔧 Using Binance Live for leverage fetching")
            
            return ccxt.binance(exchange_config)
            
        except Exception as e:
            print(f"❌ Error setting up exchange: {e}")
            return None
    
    def fetch_all_leverage_info(self) -> Dict:
        """Fetch comprehensive leverage information for all futures pairs"""
        print("🔍 FETCHING ALL FUTURES LEVERAGE INFORMATION")
        print("=" * 60)
        
        try:
            # Use REST API to get all exchange info
            print("📡 Fetching exchange info from Binance...")
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'symbols' not in data:
                print("❌ No symbols data found in response")
                return {}
            
            # Filter for USDT perpetual futures
            futures_pairs = []
            for symbol_data in data['symbols']:
                if (symbol_data.get('status') == 'TRADING' and 
                    symbol_data['symbol'].endswith('USDT') and 
                    symbol_data.get('contractType') == 'PERPETUAL'):
                    futures_pairs.append(symbol_data['symbol'])
            
            print(f"✅ Found {len(futures_pairs)} USDT perpetual futures pairs")
            
            # Fetch leverage information
            leverage_data = {}
            
            print("📊 Fetching leverage information...")
            
            for symbol in futures_pairs[:100]:  # Limit to first 100 for performance
                try:
                    # Get leverage info from the symbol data
                    leverage_info = self._get_leverage_from_symbol_data(symbol, data['symbols'])
                    
                    if leverage_info:
                        leverage_data[symbol] = leverage_info
                    else:
                        leverage_data[symbol] = {
                            'max_leverage': 0,
                            'leverage_tiers': [],
                            'status': 'No leverage info'
                        }
                        
                except Exception as e:
                    print(f"⚠️ Error fetching leverage for {symbol}: {e}")
                    leverage_data[symbol] = {
                        'max_leverage': 0,
                        'leverage_tiers': [],
                        'status': f'Error: {str(e)}'
                    }
            
            # Cache the data
            self.leverage_cache = leverage_data
            self.last_fetch_time = datetime.now()
            
            return leverage_data
            
        except Exception as e:
            print(f"❌ Error fetching leverage information: {e}")
            return {}
    
    def _get_leverage_from_symbol_data(self, symbol: str, symbols_data: List) -> Dict:
        """Get leverage information from symbol data"""
        try:
            for symbol_data in symbols_data:
                if symbol_data['symbol'] == symbol:
                    # Get leverage info from filters
                    for filter_data in symbol_data.get('filters', []):
                        if filter_data.get('filterType') == 'LEVERAGE':
                            max_leverage = int(filter_data.get('maxLeverage', 0))
                            
                            # Get leverage brackets if available
                            leverage_tiers = []
                            if 'leverageBrackets' in filter_data:
                                for bracket in filter_data['leverageBrackets']:
                                    leverage_tiers.append({
                                        'notional_floor': bracket.get('notionalFloor', 0),
                                        'notional_cap': bracket.get('notionalCap', float('inf')),
                                        'leverage': bracket.get('initialLeverage', 0)
                                    })
                            
                            return {
                                'max_leverage': max_leverage,
                                'leverage_tiers': leverage_tiers,
                                'status': 'Available'
                            }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error getting leverage info for {symbol}: {e}")
            return None
    
    def get_leverage_summary(self, leverage_data: Dict = None) -> Dict:
        """Get summary statistics of leverage data"""
        if leverage_data is None:
            leverage_data = self.leverage_cache
        
        if not leverage_data:
            return {}
        
        # Calculate statistics
        max_leverages = [data['max_leverage'] for data in leverage_data.values() 
                        if isinstance(data['max_leverage'], (int, float)) and data['max_leverage'] > 0]
        
        if not max_leverages:
            return {
                'total_pairs': len(leverage_data),
                'available_pairs': 0,
                'max_leverage': 0,
                'min_leverage': 0,
                'avg_leverage': 0,
                'leverage_distribution': {}
            }
        
        summary = {
            'total_pairs': len(leverage_data),
            'available_pairs': len([d for d in leverage_data.values() if d['status'] == 'Available']),
            'max_leverage': max(max_leverages),
            'min_leverage': min(max_leverages),
            'avg_leverage': sum(max_leverages) / len(max_leverages),
            'leverage_distribution': {}
        }
        
        # Count leverage distribution
        for leverage in max_leverages:
            if leverage not in summary['leverage_distribution']:
                summary['leverage_distribution'][leverage] = 0
            summary['leverage_distribution'][leverage] += 1
        
        return summary
    
    def get_pairs_by_leverage(self, min_leverage: int = 0, max_leverage: int = None) -> List[str]:
        """Get pairs filtered by leverage range"""
        if not self.leverage_cache:
            self.fetch_all_leverage_info()
        
        filtered_pairs = []
        
        for symbol, data in self.leverage_cache.items():
            if data['status'] == 'Available':
                leverage = data['max_leverage']
                if isinstance(leverage, (int, float)) and leverage >= min_leverage and (max_leverage is None or leverage <= max_leverage):
                    filtered_pairs.append(symbol)
        
        return filtered_pairs
    
    def get_optimal_trading_pairs(self, target_leverage: int = 125, min_volume: float = 1000000) -> List[Dict]:
        """Get optimal trading pairs based on leverage and volume"""
        print(f"🎯 FINDING OPTIMAL TRADING PAIRS (Target: {target_leverage}x leverage)")
        print("=" * 60)
        
        # Get pairs with target leverage
        target_pairs = self.get_pairs_by_leverage(target_leverage, target_leverage)
        
        print(f"📊 Found {len(target_pairs)} pairs with {target_leverage}x leverage")
        
        # Get volume data for these pairs
        optimal_pairs = []
        
        for symbol in target_pairs[:50]:  # Limit to first 50 for performance
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                volume_24h = ticker.get('quoteVolume', 0)
                
                if volume_24h >= min_volume:
                    optimal_pairs.append({
                        'symbol': symbol,
                        'leverage': target_leverage,
                        'volume_24h': volume_24h,
                        'price': ticker.get('last', 0),
                        'change_24h': ticker.get('percentage', 0)
                    })
                    
            except Exception as e:
                print(f"⚠️ Error fetching data for {symbol}: {e}")
        
        # Sort by volume
        optimal_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        return optimal_pairs
    
    def display_leverage_report(self, leverage_data: Dict = None):
        """Display comprehensive leverage report"""
        if leverage_data is None:
            leverage_data = self.leverage_cache
        
        if not leverage_data:
            print("❌ No leverage data available")
            return
        
        summary = self.get_leverage_summary(leverage_data)
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE LEVERAGE REPORT")
        print("="*60)
        
        print(f"📈 Total Pairs: {summary['total_pairs']}")
        print(f"✅ Available Pairs: {summary['available_pairs']}")
        print(f"🏆 Max Leverage: {summary['max_leverage']}x")
        print(f"📉 Min Leverage: {summary['min_leverage']}x")
        print(f"📊 Average Leverage: {summary['avg_leverage']:.1f}x")
        
        print(f"\n📊 LEVERAGE DISTRIBUTION:")
        print("-" * 40)
        for leverage, count in sorted(summary['leverage_distribution'].items()):
            percentage = (count / summary['available_pairs']) * 100
            print(f"   {leverage}x: {count} pairs ({percentage:.1f}%)")
        
        # Show top pairs by leverage
        print(f"\n🏆 TOP PAIRS BY LEVERAGE:")
        print("-" * 40)
        
        sorted_pairs = sorted(leverage_data.items(), 
                            key=lambda x: x[1]['max_leverage'] if isinstance(x[1]['max_leverage'], (int, float)) else 0, 
                            reverse=True)
        
        for i, (symbol, data) in enumerate(sorted_pairs[:20]):
            if data['status'] == 'Available':
                print(f"   {i+1:2d}. {symbol:<15} {data['max_leverage']}x")
        
        # Show our current trading pairs
        current_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 
                        'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT']
        
        print(f"\n🎯 OUR CURRENT TRADING PAIRS:")
        print("-" * 40)
        
        for pair in current_pairs:
            if pair in leverage_data:
                data = leverage_data[pair]
                status = "✅" if data['status'] == 'Available' else "❌"
                leverage = data['max_leverage'] if data['max_leverage'] > 0 else "N/A"
                print(f"   {status} {pair:<15} {leverage}x")
            else:
                print(f"   ❌ {pair:<15} Not found")
    
    def save_leverage_data(self, filename: str = None):
        """Save leverage data to file"""
        if not self.leverage_cache:
            self.fetch_all_leverage_info()
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"futures_leverage_data_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.leverage_cache, f, indent=2)
            
            print(f"💾 Leverage data saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving leverage data: {e}")
            return None
    
    def get_leverage_for_pair(self, symbol: str) -> Dict:
        """Get detailed leverage information for a specific pair"""
        if not self.leverage_cache:
            self.fetch_all_leverage_info()
        
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
        else:
            return {'status': 'Not found', 'max_leverage': 0, 'leverage_tiers': []}

def main():
    """Main function to demonstrate leverage fetching"""
    print("🚀 FUTURES LEVERAGE FETCHER")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = FuturesLeverageFetcher(testnet=True)  # Use testnet for safety
    
    if not fetcher.exchange:
        print("❌ Failed to initialize exchange connection")
        return
    
    # Fetch all leverage data
    leverage_data = fetcher.fetch_all_leverage_info()
    
    if leverage_data:
        # Display comprehensive report
        fetcher.display_leverage_report(leverage_data)
        
        # Get optimal trading pairs
        optimal_pairs = fetcher.get_optimal_trading_pairs(target_leverage=125, min_volume=1000000)
        
        print(f"\n🎯 OPTIMAL TRADING PAIRS (125x leverage, >$1M volume):")
        print("-" * 60)
        
        for i, pair in enumerate(optimal_pairs[:10]):
            volume_m = pair['volume_24h'] / 1000000
            print(f"   {i+1:2d}. {pair['symbol']:<15} ${pair['price']:<10.4f} ${volume_m:.1f}M {pair['change_24h']:+.2f}%")
        
        # Save data
        fetcher.save_leverage_data()
        
        print(f"\n✅ Leverage fetching completed!")
        print(f"💡 Found {len(leverage_data)} futures pairs with leverage information")
        
    else:
        print("❌ Failed to fetch leverage data")

if __name__ == "__main__":
    main() 