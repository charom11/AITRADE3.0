"""
Performance Analyzer for Mini Hedge Fund
Calculates key performance metrics and generates reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf

from config import PERFORMANCE_CONFIG

class PerformanceAnalyzer:
    """Analyzes portfolio performance and generates reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_returns(self, portfolio_values: List[Dict]) -> pd.Series:
        """
        Calculate daily returns from portfolio values
        
        Args:
            portfolio_values: List of dictionaries with 'date' and 'portfolio_value'
            
        Returns:
            Series of daily returns
        """
        df = pd.DataFrame(portfolio_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Calculate returns
        returns = df['portfolio_value'].pct_change().dropna()
        
        return returns
    
    def calculate_benchmark_returns(self, benchmark_symbol: str, 
                                  start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Calculate benchmark returns for comparison
        
        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'SPY')
            start_date: Start date
            end_date: End date
            
        Returns:
            Series of benchmark returns
        """
        try:
            benchmark = yf.download(benchmark_symbol, start=start_date, end=end_date)
            benchmark_returns = benchmark['Close'].pct_change().dropna()
            return benchmark_returns
        except Exception as e:
            self.logger.error(f"Error fetching benchmark data: {str(e)}")
            return pd.Series()
    
    def calculate_metrics(self, returns: pd.Series, 
                        risk_free_rate: float = None) -> Dict:
        """
        Calculate key performance metrics
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary of performance metrics
        """
        if risk_free_rate is None:
            risk_free_rate = PERFORMANCE_CONFIG['risk_free_rate']
        
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_trades': len(returns),
            'positive_trades': len(winning_trades),
            'negative_trades': len(losing_trades)
        }
        
        return metrics
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window=window).mean() * 252
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        risk_free_rate = PERFORMANCE_CONFIG['risk_free_rate']
        excess_returns = returns - risk_free_rate / 252
        rolling_metrics['rolling_sharpe'] = (excess_returns.rolling(window=window).mean() / 
                                           returns.rolling(window=window).std()) * np.sqrt(252)
        
        # Rolling drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window).max()
        rolling_metrics['rolling_drawdown'] = (cumulative_returns - rolling_max) / rolling_max
        
        return rolling_metrics
    
    def compare_with_benchmark(self, portfolio_returns: pd.Series, 
                             benchmark_returns: pd.Series) -> Dict:
        """
        Compare portfolio performance with benchmark
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with comparison metrics
        """
        # Align returns
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1)
        aligned_returns.columns = ['portfolio', 'benchmark']
        aligned_returns = aligned_returns.dropna()
        
        if len(aligned_returns) == 0:
            return {}
        
        portfolio_returns_aligned = aligned_returns['portfolio']
        benchmark_returns_aligned = aligned_returns['benchmark']
        
        # Calculate metrics for both
        portfolio_metrics = self.calculate_metrics(portfolio_returns_aligned)
        benchmark_metrics = self.calculate_metrics(benchmark_returns_aligned)
        
        # Calculate excess returns
        excess_returns = portfolio_returns_aligned - benchmark_returns_aligned
        
        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Beta and Alpha
        covariance = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)[0, 1]
        benchmark_variance = benchmark_returns_aligned.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        risk_free_rate = PERFORMANCE_CONFIG['risk_free_rate']
        alpha = (portfolio_returns_aligned.mean() - 
                risk_free_rate / 252 - 
                beta * (benchmark_returns_aligned.mean() - risk_free_rate / 252)) * 252
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        comparison = {
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'excess_return': excess_returns.sum(),
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'correlation': portfolio_returns_aligned.corr(benchmark_returns_aligned)
        }
        
        return comparison
    
    def generate_performance_report(self, portfolio_values: List[Dict], 
                                  trade_history: List[Dict],
                                  benchmark_symbol: str = None) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            portfolio_values: List of portfolio values
            trade_history: List of trades
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Dictionary with complete performance report
        """
        # Calculate returns
        returns = self.calculate_returns(portfolio_values)
        
        if len(returns) == 0:
            return {'error': 'No returns data available'}
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns)
        rolling_metrics = self.calculate_rolling_metrics(returns)
        
        # Trade analysis
        trade_analysis = self.analyze_trades(trade_history)
        
        # Benchmark comparison
        benchmark_comparison = {}
        if benchmark_symbol:
            start_date = pd.to_datetime(portfolio_values[0]['date'])
            end_date = pd.to_datetime(portfolio_values[-1]['date'])
            benchmark_returns = self.calculate_benchmark_returns(benchmark_symbol, start_date, end_date)
            
            if len(benchmark_returns) > 0:
                benchmark_comparison = self.compare_with_benchmark(returns, benchmark_returns)
        
        # Create report
        report = {
            'summary_metrics': metrics,
            'rolling_metrics': rolling_metrics,
            'trade_analysis': trade_analysis,
            'benchmark_comparison': benchmark_comparison,
            'returns': returns,
            'portfolio_values': portfolio_values
        }
        
        return report
    
    def analyze_trades(self, trade_history: List[Dict]) -> Dict:
        """
        Analyze trading performance
        
        Args:
            trade_history: List of trade dictionaries
            
        Returns:
            Dictionary with trade analysis
        """
        if not trade_history:
            return {}
        
        df = pd.DataFrame(trade_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Separate open and close trades
        open_trades = df[df['action'] == 'open']
        close_trades = df[df['action'] == 'close']
        
        # Calculate trade statistics
        if len(close_trades) > 0:
            winning_trades = close_trades[close_trades['pnl'] > 0]
            losing_trades = close_trades[close_trades['pnl'] < 0]
            
            trade_stats = {
                'total_trades': len(close_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(close_trades),
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': close_trades['pnl'].max(),
                'largest_loss': close_trades['pnl'].min(),
                'total_pnl': close_trades['pnl'].sum(),
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) 
                               if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
            }
        else:
            trade_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'total_pnl': 0,
                'profit_factor': 0
            }
        
        # Strategy performance
        strategy_performance = {}
        if len(close_trades) > 0:
            for strategy in close_trades['strategy'].unique():
                strategy_trades = close_trades[close_trades['strategy'] == strategy]
                strategy_pnl = strategy_trades['pnl'].sum()
                strategy_win_rate = len(strategy_trades[strategy_trades['pnl'] > 0]) / len(strategy_trades)
                
                strategy_performance[strategy] = {
                    'total_trades': len(strategy_trades),
                    'total_pnl': strategy_pnl,
                    'win_rate': strategy_win_rate,
                    'avg_pnl': strategy_trades['pnl'].mean()
                }
        
        return {
            'trade_statistics': trade_stats,
            'strategy_performance': strategy_performance
        }
    
    def plot_performance(self, report: Dict, save_path: str = None):
        """
        Create performance visualization plots
        
        Args:
            report: Performance report dictionary
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16)
        
        # 1. Cumulative returns
        returns = report['returns']
        cumulative_returns = (1 + returns).cumprod()
        benchmark_returns = None
        
        if 'benchmark_comparison' in report and report['benchmark_comparison']:
            benchmark_returns = report['benchmark_comparison']['benchmark_metrics']
            # You would need to calculate cumulative benchmark returns here
        
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio', linewidth=2)
        if benchmark_returns:
            axes[0, 0].plot(cumulative_returns.index, benchmark_returns, label='Benchmark', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Rolling Sharpe ratio
        rolling_metrics = report['rolling_metrics']
        axes[0, 1].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], linewidth=2)
        axes[0, 1].set_title('Rolling Sharpe Ratio (252-day)')
        axes[0, 1].grid(True)
        
        # 3. Drawdown
        axes[1, 0].fill_between(rolling_metrics.index, rolling_metrics['rolling_drawdown'], 0, 
                               alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].grid(True)
        
        # 4. Monthly returns heatmap
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_matrix = monthly_returns.groupby([monthly_returns.index.year, 
                                                        monthly_returns.index.month]).first().unstack()
        
        sns.heatmap(monthly_returns_matrix, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_report(self, report: Dict, filename: str):
        """
        Export performance report to CSV/Excel
        
        Args:
            report: Performance report
            filename: Output filename
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary metrics
                summary_df = pd.DataFrame([report['summary_metrics']]).T
                summary_df.columns = ['Value']
                summary_df.to_excel(writer, sheet_name='Summary_Metrics')
                
                # Rolling metrics
                report['rolling_metrics'].to_excel(writer, sheet_name='Rolling_Metrics')
                
                # Trade analysis
                if 'trade_analysis' in report and report['trade_analysis']:
                    trade_stats_df = pd.DataFrame([report['trade_analysis']['trade_statistics']]).T
                    trade_stats_df.columns = ['Value']
                    trade_stats_df.to_excel(writer, sheet_name='Trade_Statistics')
                    
                    if report['trade_analysis']['strategy_performance']:
                        strategy_df = pd.DataFrame(report['trade_analysis']['strategy_performance']).T
                        strategy_df.to_excel(writer, sheet_name='Strategy_Performance')
                
                # Returns
                report['returns'].to_frame('Returns').to_excel(writer, sheet_name='Daily_Returns')
            
            self.logger.info(f"Performance report exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {str(e)}") 