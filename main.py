"""
🚀 COMPREHENSIVE AI TRADING BOT - MAIN EXECUTION FILE
Connects all trading strategies and systems for easy execution

Available Systems:
- Mini Hedge Fund (Traditional Strategies)
- Live Crypto Trading (Real-time crypto)
- Enhanced Futures Trading (ML-enhanced futures)
- Continuous Trading System (24/7 automated)
- Advanced ML Systems (Multiple pairs, optimization)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ComprehensiveTradingBot:
    """Main trading bot that connects all available systems"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the comprehensive trading bot
        
        Args:
            initial_capital: Initial capital for trading
        """
        self.initial_capital = initial_capital
        self.is_running = False
        
        # Initialize all trading systems
        self._initialize_systems()
        
        # Performance tracking
        self.performance_history = []
        self.current_mode = None
        
        logger.info(f"Comprehensive Trading Bot initialized with ${initial_capital:,.2f} capital")
    
    def _initialize_systems(self):
        """Initialize all available trading systems"""
        # Initialize all systems as None - they will be loaded on demand
        self.mini_hedge_fund = None
        self.live_crypto_trader = None
        self.enhanced_futures = None
        self.continuous_trading = None
        self.ml_optimization = None
        self.leverage_fetcher = None
        
        logger.info("Systems will be loaded on demand")
    
    def _setup_mini_hedge_strategies(self):
        """Setup strategies for mini hedge fund"""
        if self.mini_hedge_fund:
            try:
                # Add momentum strategy
                momentum_strategy = self.mini_hedge_fund['momentum_strategy']()
                self.mini_hedge_fund['strategy_manager'].add_strategy(momentum_strategy)
        
        # Add mean reversion strategy
                mean_reversion_strategy = self.mini_hedge_fund['mean_reversion_strategy']()
                self.mini_hedge_fund['strategy_manager'].add_strategy(mean_reversion_strategy)
        
                        # Add pairs trading strategy
                pairs_strategy = self.mini_hedge_fund['pairs_strategy']()
                self.mini_hedge_fund['strategy_manager'].add_strategy(pairs_strategy)
                
                # Add divergence strategy
                divergence_strategy = self.mini_hedge_fund['divergence_strategy']()
                self.mini_hedge_fund['strategy_manager'].add_strategy(divergence_strategy)
                
                logger.info("Mini Hedge Fund strategies configured")
            except Exception as e:
                logger.error(f"Error setting up strategies: {e}")
    
    def _load_mini_hedge_fund(self):
        """Load Mini Hedge Fund system on demand"""
        if self.mini_hedge_fund is None:
            try:
                from data_manager import DataManager
                from strategies import MomentumStrategy, MeanReversionStrategy, PairsTradingStrategy, DivergenceStrategy, StrategyManager
                from portfolio_manager import PortfolioManager
                from performance_analyzer import PerformanceAnalyzer
                from config import DATA_CONFIG, TRADING_CONFIG, PERFORMANCE_CONFIG
                
                self.mini_hedge_fund = {
                    'data_manager': DataManager(),
                    'strategy_manager': StrategyManager(),
                    'portfolio_manager': PortfolioManager(self.initial_capital),
                    'performance_analyzer': PerformanceAnalyzer(),
                    'config': {
                        'data': DATA_CONFIG,
                        'trading': TRADING_CONFIG,
                        'performance': PERFORMANCE_CONFIG
                    },
                    'momentum_strategy': MomentumStrategy,
                    'mean_reversion_strategy': MeanReversionStrategy,
                    'pairs_strategy': PairsTradingStrategy,
                    'divergence_strategy': DivergenceStrategy
                }
                
                # Setup strategies for mini hedge fund
                self._setup_mini_hedge_strategies()
                
                logger.info("Mini Hedge Fund system loaded")
                
            except ImportError as e:
                logger.warning(f"Mini Hedge Fund components not available: {e}")
                return False
        
        return self.mini_hedge_fund is not None

    def run_mini_hedge_fund(self, start_date: str = '2022-01-01', end_date: str = '2023-12-31') -> Dict:
        """Run Mini Hedge Fund backtesting"""
        if not self._load_mini_hedge_fund():
            logger.error("Mini Hedge Fund system not available")
            return {}
        
        logger.info("📊 Running Mini Hedge Fund backtest...")
        
        # Get all symbols
        symbols = self.mini_hedge_fund['data_manager'].get_all_symbols()
        logger.info(f"Trading {len(symbols)} symbols: {symbols}")
        
        # Fetch historical data
        data = self.mini_hedge_fund['data_manager'].fetch_data(symbols, start_date, end_date)
        
        if not data:
            logger.error("No data fetched. Exiting backtest.")
            return {}
        
        # Calculate technical indicators
        indicators = self.mini_hedge_fund['data_manager'].calculate_technical_indicators(data)
        
        # Get trading signals from all strategies
        all_signals = self.mini_hedge_fund['strategy_manager'].get_all_signals(indicators)
        
        # Run simulation
        results = self._run_mini_hedge_simulation(data, all_signals, start_date, end_date)
        
        # Generate reports (only if files don't exist)
        self._generate_mini_hedge_reports(results)
        
        logger.info("✅ Mini Hedge Fund backtest completed")
        return results
    
    def _run_mini_hedge_simulation(self, data: Dict[str, pd.DataFrame], 
                       all_signals: Dict[str, Dict[str, pd.DataFrame]], 
                       start_date: str, end_date: str) -> Dict:
        """Run mini hedge fund simulation"""
        # Get common date range
        all_dates = set()
        for symbol, df in data.items():
            all_dates.update(df.index)
        
        trading_dates = sorted(list(all_dates))
        trading_dates = [d for d in trading_dates 
                        if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]
        
        logger.info(f"Running simulation for {len(trading_dates)} trading days")
        
        # Track performance
        daily_values = []
        trades_executed = 0
        
        for date in trading_dates:
            # Get current prices
            current_prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'close']
            
            # Update existing positions
            self.mini_hedge_fund['portfolio_manager'].update_positions(current_prices, date)
            
            # Check for new signals
            for strategy_name, strategy_signals in all_signals.items():
                for symbol, signal_df in strategy_signals.items():
                    if symbol in current_prices and date in signal_df.index:
                        signal = signal_df.loc[date, 'signal']
                        signal_strength = signal_df.loc[date, 'signal_strength']
                        
                        if signal != 0 and symbol not in self.mini_hedge_fund['portfolio_manager'].positions:
                            # Calculate position size
                            position_size = self.mini_hedge_fund['portfolio_manager'].calculate_position_size(
                                signal_strength, current_prices[symbol], strategy_name, symbol
                            )
                            
                            if position_size != 0:
                                # Open position
                                quantity = abs(position_size) / current_prices[symbol]
                                side = 'long' if signal > 0 else 'short'
                                
                                success = self.mini_hedge_fund['portfolio_manager'].open_position(
                                    symbol, strategy_name, side, quantity, 
                                    current_prices[symbol], date
                                )
                                
                                if success:
                                    trades_executed += 1
            
            # Update performance
            self.mini_hedge_fund['portfolio_manager'].update_performance(current_prices, date)
            
            # Record daily value
            portfolio_value = self.mini_hedge_fund['portfolio_manager'].calculate_portfolio_value(current_prices)
            daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.mini_hedge_fund['portfolio_manager'].current_capital,
                'positions': len(self.mini_hedge_fund['portfolio_manager'].positions)
            })
        
        logger.info(f"Simulation completed. {trades_executed} trades executed.")
        
        # Generate performance report
        report = self.mini_hedge_fund['performance_analyzer'].generate_performance_report(
            daily_values, 
            self.mini_hedge_fund['portfolio_manager'].trade_history,
            self.mini_hedge_fund['config']['performance']['benchmark']
        )
        
        return {
            'daily_values': daily_values,
            'trade_history': self.mini_hedge_fund['portfolio_manager'].trade_history,
            'performance_report': report,
            'portfolio_summary': self.mini_hedge_fund['portfolio_manager'].get_portfolio_summary(),
            'strategy_summary': self.mini_hedge_fund['strategy_manager'].get_strategy_summary()
        }
    
    def _generate_mini_hedge_reports(self, results: Dict):
        """Generate reports for mini hedge fund (only if files don't exist)"""
        if not results:
            return
        
        output_dir = 'reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if files already exist before creating
        report_filename = os.path.join(output_dir, f'mini_hedge_performance_{timestamp}.xlsx')
        if not os.path.exists(report_filename):
            self.mini_hedge_fund['performance_analyzer'].export_report(
                results['performance_report'], report_filename
            )
            logger.info(f"📊 Performance report saved: {report_filename}")
        
        portfolio_filename = os.path.join(output_dir, f'mini_hedge_portfolio_{timestamp}.json')
        if not os.path.exists(portfolio_filename):
            self.mini_hedge_fund['portfolio_manager'].save_portfolio_state(portfolio_filename)
            logger.info(f"💼 Portfolio state saved: {portfolio_filename}")
        
        plot_filename = os.path.join(output_dir, f'mini_hedge_performance_{timestamp}.png')
        if not os.path.exists(plot_filename):
            self.mini_hedge_fund['performance_analyzer'].plot_performance(
                results['performance_report'], plot_filename
            )
            logger.info(f"📈 Performance plot saved: {plot_filename}")
    
    def _load_live_crypto_trader(self):
        """Load Live Crypto Trading system on demand"""
        if self.live_crypto_trader is None:
            try:
                from live_crypto_trader import LiveCryptoTrader
                
                # Load API keys from environment
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_SECRET_KEY')
                
                # Check if we have API keys for real trading
                if api_key and api_secret:
                    print("🔑 API keys found! Setting up REAL LIVE TRADING...")
                    print("⚠️  WARNING: This will use REAL MONEY!")
                    confirm = input("Are you sure you want to proceed with real trading? (yes/no): ")
                    
                    if confirm.lower() == 'yes':
                        paper_trading = False
                        print("🚀 REAL LIVE TRADING ENABLED!")
                    else:
                        paper_trading = True
                        print("📝 Paper trading mode enabled")
                else:
                    paper_trading = True
                    print("📝 No API keys found - using paper trading mode")
                
                self.live_crypto_trader = LiveCryptoTrader(
                    api_key=api_key,
                    secret=api_secret,
                    initial_capital=self.initial_capital,
                    paper_trading=paper_trading
                )
                logger.info(f"Live Crypto Trading system loaded (Paper Trading: {paper_trading})")
                
            except ImportError as e:
                logger.warning(f"Live Crypto Trading components not available: {e}")
                return False
        
        return self.live_crypto_trader is not None

    def _load_leverage_fetcher(self):
        """Load Futures Leverage Fetcher system on demand"""
        if self.leverage_fetcher is None:
            try:
                from futures_leverage_fetcher import FuturesLeverageFetcher
                
                # Load API keys from environment
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_SECRET_KEY')
                
                self.leverage_fetcher = FuturesLeverageFetcher(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=False  # Use live for leverage fetching
                )
                logger.info("Futures Leverage Fetcher system loaded")
                
            except ImportError as e:
                logger.warning(f"Futures Leverage Fetcher components not available: {e}")
                return False
        
        return self.leverage_fetcher is not None

    def run_live_crypto_trading(self, update_interval: int = 60):
        """Run Live Crypto Trading System"""
        if not self._load_live_crypto_trader():
            logger.error("Live Crypto Trading system not available")
            return
        
        logger.info("Starting Live Crypto Trading...")
        self.current_mode = 'live_crypto'
        
        try:
            self.live_crypto_trader.run_live_trading(update_interval)
        except KeyboardInterrupt:
            logger.info("Live Crypto Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in Live Crypto Trading: {e}")
    
    def run_real_live_trading(self, update_interval: int = 30):
        """Run REAL Live Trading with API (Real Money)"""
        if not self._load_live_crypto_trader():
            logger.error("Live Crypto Trading system not available")
            return
        
        # Force real trading mode
        if hasattr(self.live_crypto_trader, 'paper_trading'):
            self.live_crypto_trader.paper_trading = False
        
        logger.info("🚀 Starting REAL LIVE TRADING with API...")
        logger.info("⚠️  WARNING: This will use REAL MONEY!")
        self.current_mode = 'real_live_trading'
        
        try:
            print("\n" + "="*60)
            print("🚀 REAL LIVE TRADING SYSTEM")
            print("="*60)
            print("📊 Trading with REAL MONEY")
            print("⏱️  Update Interval: {update_interval} seconds")
            print("🛑 Press Ctrl+C to stop")
            print("="*60)
            
            self.live_crypto_trader.run_live_trading(update_interval)
            
        except KeyboardInterrupt:
            logger.info("🛑 Real Live Trading stopped by user")
            print("\n🛑 Real Live Trading stopped!")
            print("📊 Final portfolio status saved")
        except Exception as e:
            logger.error(f"❌ Error in Real Live Trading: {e}")
            print(f"\n❌ Error: {e}")
    
    def run_futures_leverage_fetch(self):
        """Run Futures Leverage Fetching System"""
        if not self._load_leverage_fetcher():
            logger.error("Futures Leverage Fetcher system not available")
            return
        
        logger.info("🔍 Starting Futures Leverage Fetching...")
        self.current_mode = 'leverage_fetch'
        
        try:
            print("\n" + "="*60)
            print("🔍 FUTURES LEVERAGE FETCHING SYSTEM")
            print("="*60)
            print("📊 Fetching all available leverage options")
            print("🎯 Finding optimal trading pairs")
            print("💾 Saving leverage data to files")
            print("="*60)
            
            # Fetch all leverage information
            leverage_data = self.leverage_fetcher.fetch_all_leverage_info()
            
            if leverage_data:
                # Display comprehensive report
                self.leverage_fetcher.display_leverage_report(leverage_data)
                
                # Get optimal trading pairs
                optimal_pairs = self.leverage_fetcher.get_optimal_trading_pairs(
                    target_leverage=125, 
                    min_volume=1000000
                )
                
                print(f"\n🎯 OPTIMAL TRADING PAIRS (125x leverage, >$1M volume):")
                print("-" * 60)
                
                for i, pair in enumerate(optimal_pairs[:15]):
                    volume_m = pair['volume_24h'] / 1000000
                    print(f"   {i+1:2d}. {pair['symbol']:<15} ${pair['price']:<10.4f} ${volume_m:.1f}M {pair['change_24h']:+.2f}%")
                
                # Save leverage data
                filename = self.leverage_fetcher.save_leverage_data()
                
                # Get summary
                summary = self.leverage_fetcher.get_leverage_summary(leverage_data)
                
                print(f"\n📊 LEVERAGE FETCHING SUMMARY:")
                print("-" * 40)
                print(f"✅ Total Pairs: {summary.get('total_pairs', 0)}")
                print(f"✅ Available Pairs: {summary.get('available_pairs', 0)}")
                print(f"🏆 Max Leverage: {summary.get('max_leverage', 0)}x")
                print(f"📊 Average Leverage: {summary.get('avg_leverage', 0):.1f}x")
                print(f"🎯 Optimal Pairs Found: {len(optimal_pairs)}")
                
                if filename:
                    print(f"💾 Data saved to: {filename}")
                
                logger.info("✅ Futures Leverage Fetching completed successfully")
                
            else:
                logger.error("❌ Failed to fetch leverage data")
                print("❌ Failed to fetch leverage data")
                
        except KeyboardInterrupt:
            logger.info("🛑 Futures Leverage Fetching stopped by user")
            print("\n🛑 Leverage fetching stopped by user!")
        except Exception as e:
            logger.error(f"❌ Error in Futures Leverage Fetching: {e}")
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_enhanced_futures_trading(self, duration_hours: int = 24):
        """Run Enhanced Futures Trading System"""
        if not self.enhanced_futures:
            logger.error("❌ Enhanced Futures Trading system not available")
            return
        
        logger.info("🚀 Starting Enhanced Futures Trading...")
        self.current_mode = 'enhanced_futures'
        
        try:
            self.enhanced_futures.run_trading_session(duration_hours)
        except KeyboardInterrupt:
            logger.info("⏹️ Enhanced Futures Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in Enhanced Futures Trading: {e}")
    
    def run_continuous_trading(self):
        """Run Continuous Trading System (24/7)"""
        if not self.continuous_trading:
            logger.error("❌ Continuous Trading System not available")
            return
        
        logger.info("🚀 Starting Continuous Trading System (24/7)...")
        self.current_mode = 'continuous_trading'
        
        try:
            self.continuous_trading.start_continuous_trading()
        except KeyboardInterrupt:
            logger.info("⏹️ Continuous Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in Continuous Trading: {e}")
    
    def run_ml_optimization(self, duration_hours: int = 24):
        """Run ML Optimization System"""
        if not self.ml_optimization:
            logger.error("❌ ML Optimization System not available")
            return
        
        logger.info("🤖 Starting ML Optimization System...")
        self.current_mode = 'ml_optimization'
        
        try:
            self.ml_optimization.run_optimization_cycle(duration_hours)
        except KeyboardInterrupt:
            logger.info("⏹️ ML Optimization stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in ML Optimization: {e}")
    
    def run_demo_mode(self):
        """Run a quick demo of all available systems"""
        logger.info("Starting Demo Mode...")
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TRADING BOT DEMO")
        print("="*60)
        
        # Show available systems
        available_systems = []
        if self._load_mini_hedge_fund():
            available_systems.append("Mini Hedge Fund (Traditional Strategies)")
        if self._load_live_crypto_trader():
            available_systems.append("Live Crypto Trading (Real-time)")
        if self._load_leverage_fetcher():
            available_systems.append("Futures Leverage Fetcher (Comprehensive)")
        
        print("\nAvailable Trading Systems:")
        for system in available_systems:
            print(f"   - {system}")
        
        # Run quick backtest if mini hedge fund is available
        if self.mini_hedge_fund:
            print("\nRunning quick Mini Hedge Fund backtest...")
            results = self.run_mini_hedge_fund(
                start_date='2023-01-01',
                end_date='2023-06-30'
            )
            
            if results and 'performance_report' in results:
                self._print_performance_summary(results['performance_report'])
        
        print("\nDemo completed!")
        print("Use specific commands to run individual systems:")
        print("   python main.py --mini-hedge")
        print("   python main.py --live-crypto")
        print("   python main.py --real-trading    (REAL MONEY)")
        print("   python main.py --futures")
        print("   python main.py --leverage        (FETCH ALL LEVERAGE)")
        print("   python main.py --continuous")
        print("   python main.py --ml-optimization")
    
    def _print_performance_summary(self, report: Dict):
        """Print performance summary"""
        if 'summary_metrics' not in report:
            return
        
        metrics = report['summary_metrics']
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"VaR (95%): {metrics['var_95']:.2%}")
        print("="*60)
    
    def get_system_status(self):
        """Get status of all trading systems"""
        status = {
            'mini_hedge_fund': self.mini_hedge_fund is not None,
            'live_crypto_trader': self.live_crypto_trader is not None,
            'enhanced_futures': self.enhanced_futures is not None,
            'continuous_trading': self.continuous_trading is not None,
            'ml_optimization': self.ml_optimization is not None,
            'leverage_fetcher': self.leverage_fetcher is not None,
            'current_mode': self.current_mode,
            'is_running': self.is_running
        }
        
        return status

def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(description='Comprehensive AI Trading Bot')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--mini-hedge', action='store_true', help='Run Mini Hedge Fund')
    parser.add_argument('--live-crypto', action='store_true', help='Run Live Crypto Trading')
    parser.add_argument('--real-trading', action='store_true', help='Run REAL Live Trading with API')
    parser.add_argument('--futures', action='store_true', help='Run Enhanced Futures Trading')
    parser.add_argument('--leverage', action='store_true', help='Fetch all available futures leverage')
    parser.add_argument('--continuous', action='store_true', help='Run Continuous Trading')
    parser.add_argument('--ml-optimization', action='store_true', help='Run ML Optimization')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    print("COMPREHENSIVE AI TRADING BOT")
    print("="*60)
    
    # Initialize the trading bot
    bot = ComprehensiveTradingBot(initial_capital=args.capital)
    
    # Show system status if requested
    if args.status:
        status = bot.get_system_status()
        print("\nSYSTEM STATUS:")
        for system, available in status.items():
            if system not in ['current_mode', 'is_running']:
                status_icon = "OK" if available else "NO"
                print(f"   {status_icon} {system.replace('_', ' ').title()}")
        print(f"   Current Mode: {status['current_mode'] or 'None'}")
        print(f"   Running: {status['is_running']}")
        return
    
    # Run requested mode
    if args.demo:
        bot.run_demo_mode()
    elif args.mini_hedge:
        bot.run_mini_hedge_fund()
    elif args.live_crypto:
        bot.run_live_crypto_trading()
    elif args.real_trading:
        bot.run_real_live_trading()
    elif args.futures:
        bot.run_enhanced_futures_trading()
    elif args.leverage:
        bot.run_futures_leverage_fetch()
    elif args.continuous:
        bot.run_continuous_trading()
    elif args.ml_optimization:
        bot.run_ml_optimization()
    else:
        # Default: run demo mode
        bot.run_demo_mode()
    
    print("\nTrading Bot execution completed!")

if __name__ == "__main__":
    main() 