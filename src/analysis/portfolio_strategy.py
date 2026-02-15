"""
Bubble-Adjusted Portfolio Strategy - OPTIMIZED
Pre-downloads all data once, then rebalances quickly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.bubble_indicators import BubbleRiskAnalyzer
from src.data.fetch import download_stock_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class BubbleAdjustedPortfolio:
    """
    Portfolio that rebalances monthly based on bubble scores
    """
    
    def __init__(self, tickers: list, start_date: str, end_date: str, initial_capital: float = 100000):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.portfolio_history = []
        
    def backtest(self, rebalance_frequency: int = 30) -> pd.DataFrame:
        """Run backtest - OPTIMIZED VERSION"""
        
        print(f"\n{'='*70}")
        print("BUBBLE-ADJUSTED PORTFOLIO BACKTEST (OPTIMIZED)")
        print(f"{'='*70}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Stocks: {len(self.tickers)}")
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"{'='*70}\n")
        
        # PRE-DOWNLOAD ALL DATA ONCE
        print("Step 1: Downloading all stock data...")
        stock_data = {}
        for ticker in self.tickers:
            try:
                data = download_stock_data(ticker, self.start_date, self.end_date)
                stock_data[ticker] = data
                print(f"  ✓ {ticker}: {len(data)} days")
            except Exception as e:
                print(f"  ✗ {ticker}: Failed")
        
        print("\nStep 2: Downloading benchmark...")
        spy_data = download_stock_data('SPY', self.start_date, self.end_date)
        print(f"  ✓ SPY: {len(spy_data)} days")
        
        # Get rebalance dates (monthly)
        trading_dates = spy_data.index
        rebalance_dates = []
        last_rebalance = None
        
        for date in trading_dates:
            if last_rebalance is None or (date - last_rebalance).days >= rebalance_frequency:
                rebalance_dates.append(date)
                last_rebalance = date
        
        print(f"\nStep 3: Calculating bubble scores at {len(rebalance_dates)} rebalance points...")
        
        # PRE-CALCULATE BUBBLE SCORES AT EACH REBALANCE DATE
        bubble_scores_history = {}
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"\n📊 {i+1}/{len(rebalance_dates)} - {rebal_date.strftime('%Y-%m-%d')}")
            
            scores = {}
            for ticker in self.tickers:
                try:
                    # Use data up to rebalance date
                    analyzer = BubbleRiskAnalyzer(
                        ticker, 
                        lookback_days=365,
                        end_date=rebal_date.strftime('%Y-%m-%d')
                    )
                    result = analyzer.get_composite_score()
                    scores[ticker] = result['composite_score']
                    print(f"  {ticker}: {result['composite_score']:.1f}")
                except Exception as e:
                    scores[ticker] = 50.0
                    print(f"  {ticker}: 50.0 (default)")
            
            bubble_scores_history[rebal_date] = scores
        
        print("\nStep 4: Running backtest simulation...")
        
        # Initialize
        portfolio_value = self.initial_capital
        benchmark_value = self.initial_capital
        holdings = {}
        current_weights = None
        current_rebalance_date = None
        
        for date in trading_dates:
            # Check if rebalance needed
            if date in bubble_scores_history:
                current_rebalance_date = date
                bubble_scores = bubble_scores_history[date]
                
                # Calculate inverse weights
                inverse_scores = {t: max(100 - s, 10) for t, s in bubble_scores.items()}
                total_inverse = sum(inverse_scores.values())
                weights = {t: inv / total_inverse for t, inv in inverse_scores.items()}
                
                # Allocate capital
                holdings = {}
                for ticker, weight in weights.items():
                    if ticker in stock_data and date in stock_data[ticker].index:
                        price = stock_data[ticker].loc[date, 'Close']
                        shares = (portfolio_value * weight) / price
                        holdings[ticker] = shares
            
            # Calculate daily portfolio value
            portfolio_value = 0
            for ticker, shares in holdings.items():
                if ticker in stock_data and date in stock_data[ticker].index:
                    price = stock_data[ticker].loc[date, 'Close']
                    portfolio_value += shares * price
            
            # Benchmark
            spy_return = spy_data.loc[date, 'Close'] / spy_data.iloc[0]['Close']
            benchmark_value = self.initial_capital * spy_return
            
            # Record
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'benchmark_value': benchmark_value,
                'portfolio_return': (portfolio_value / self.initial_capital - 1) * 100,
                'benchmark_return': (benchmark_value / self.initial_capital - 1) * 100
            })
        
        print("\n✅ Backtest complete!")
        return pd.DataFrame(self.portfolio_history)
    
    def print_results(self, results: pd.DataFrame):
        """Print performance summary"""
        final_portfolio = results.iloc[-1]['portfolio_value']
        final_benchmark = results.iloc[-1]['benchmark_value']
        
        portfolio_return = (final_portfolio / self.initial_capital - 1) * 100
        benchmark_return = (final_benchmark / self.initial_capital - 1) * 100
        outperformance = portfolio_return - benchmark_return
        
        # Sharpe ratio
        daily_returns = results['portfolio_value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + results['portfolio_value'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        print(f"\n{'='*70}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"FINAL RESULTS:")
        print(f"  Bubble-Adjusted Portfolio: ${final_portfolio:,.0f} ({portfolio_return:+.2f}%)")
        print(f"  S&P 500 Benchmark:         ${final_benchmark:,.0f} ({benchmark_return:+.2f}%)")
        print(f"  Outperformance:            {outperformance:+.2f}%")
        
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio:              {sharpe:.2f}")
        print(f"  Max Drawdown:              {max_drawdown:.2f}%")
        
        print(f"\n{'='*70}\n")
    
    def plot_results(self, results: pd.DataFrame, save_path: str = None):
        """Plot portfolio vs benchmark"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1 = axes[0]
        ax1.plot(results['date'], results['portfolio_value'], 
                label='Bubble-Adjusted Portfolio', linewidth=2, color='blue')
        ax1.plot(results['date'], results['benchmark_value'], 
                label='S&P 500', linewidth=2, color='gray', linestyle='--')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Bubble-Adjusted Portfolio vs S&P 500', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        ax2 = axes[1]
        ax2.plot(results['date'], results['portfolio_return'], 
                label='Bubble-Adjusted Portfolio', linewidth=2, color='blue')
        ax2.plot(results['date'], results['benchmark_return'], 
                label='S&P 500', linewidth=2, color='gray', linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    tickers = ['NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META', 'AMZN']
    
    portfolio = BubbleAdjustedPortfolio(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=100000
    )
    
    results = portfolio.backtest(rebalance_frequency=30)
    portfolio.print_results(results)
    portfolio.plot_results(results)
    
    print("\n✅ Portfolio strategy complete!")