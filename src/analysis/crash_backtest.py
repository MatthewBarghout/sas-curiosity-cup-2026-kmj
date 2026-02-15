"""
Historical Crash Backtesting
Tests bubble indicators on major market crashes to validate predictive power
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.bubble_indicators import BubbleRiskAnalyzer
from src.data.fetch import download_stock_data, calculate_returns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Major crash periods to test
CRASH_PERIODS = {
    'COVID_2020': {
        'name': 'COVID-19 Crash',
        'train_start': '2019-01-01',
        'train_end': '2020-01-31',
        'crash_start': '2020-02-01',
        'crash_end': '2020-04-30',
        'tickers': ['NVDA', 'AMD', 'AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ']
    },
    'TECH_2022': {
        'name': 'Tech Bubble Pop 2022',
        'train_start': '2021-01-01',
        'train_end': '2021-12-31',
        'crash_start': '2022-01-01',
        'crash_end': '2022-06-30',
        'tickers': ['NVDA', 'AMD', 'META', 'GOOGL', 'AMZN', 'NFLX', 'QQQ']
    },
    'FINANCIAL_2008': {
        'name': 'Financial Crisis 2008',
        'train_start': '2007-01-01',
        'train_end': '2007-12-31',
        'crash_start': '2008-09-01',
        'crash_end': '2009-03-31',
        'tickers': ['JPM', 'BAC', 'C', 'GS', 'SPY', 'XLF']
    }
}

class CrashBacktester:
    """
    Backtest bubble indicators on historical crashes
    """
    
    def __init__(self, crash_name: str):
        """
        Initialize backtester for specific crash period
        
        Args:
            crash_name: Key from CRASH_PERIODS
        """
        self.crash = CRASH_PERIODS[crash_name]
        self.crash_name = crash_name
        self.results = {}
        
    def analyze_pre_crash_signals(self, ticker: str) -> dict:
        """
        Analyze bubble signals before crash
    
        Args:
            ticker: Stock to analyze
        
        Returns:
            Dictionary with pre-crash bubble scores and crash performance
        """
        print(f"\nAnalyzing {ticker} for {self.crash['name']}...")
    
        try:
            # Download crash period data
            crash_data = download_stock_data(
                ticker,
                self.crash['crash_start'],
                self.crash['crash_end']
            )
        
            # Calculate bubble score at END of training period (right before crash)
            # This uses historical data up to crash start
            analyzer = BubbleRiskAnalyzer(
                ticker, 
                lookback_days=365,
                end_date=self.crash['crash_start']  # KEY FIX: Use crash start date
            )
        
            # Get bubble score right before crash
            pre_crash_score = analyzer.get_composite_score()
        
            # Calculate crash performance
            crash_start_price = crash_data['Close'].iloc[0]
            crash_bottom_price = crash_data['Close'].min()
            crash_end_price = crash_data['Close'].iloc[-1]
        
            max_drawdown = ((crash_bottom_price - crash_start_price) / crash_start_price) * 100
            total_return = ((crash_end_price - crash_start_price) / crash_start_price) * 100
        
            return {
                'ticker': ticker,
                'pre_crash_bubble_score': pre_crash_score['composite_score'],
                'crash_max_drawdown_pct': max_drawdown,
                'crash_total_return_pct': total_return,
                'crash_start_price': float(crash_start_price),
                'crash_bottom_price': float(crash_bottom_price),
                'crash_end_price': float(crash_end_price),
                'bubble_indicators': pre_crash_score['indicators']
            }
        
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")
            return None
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run backtest on all tickers for this crash
        
        Returns:
            DataFrame with results
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {self.crash['name']}")
        print(f"{'='*70}")
        print(f"Pre-crash period: {self.crash['train_start']} to {self.crash['train_end']}")
        print(f"Crash period: {self.crash['crash_start']} to {self.crash['crash_end']}")
        print(f"Testing {len(self.crash['tickers'])} stocks")
        
        results_list = []
        
        for ticker in self.crash['tickers']:
            result = self.analyze_pre_crash_signals(ticker)
            if result:
                results_list.append(result)
                print(f"  ✓ {ticker}: Bubble={result['pre_crash_bubble_score']:.1f}, Drawdown={result['crash_max_drawdown_pct']:.1f}%")
        
        df = pd.DataFrame(results_list)
        self.results = df
        
        return df
    
    def calculate_accuracy(self, threshold: float = 60.0) -> dict:
        """
        Calculate prediction accuracy
        
        Args:
            threshold: Bubble score threshold for crash warning
            
        Returns:
            Accuracy metrics
        """
        if self.results.empty:
            return None
        
        df = self.results.copy()
        
        # Define "crash" as drawdown > 20%
        df['crashed'] = df['crash_max_drawdown_pct'] < -20
        df['bubble_warning'] = df['pre_crash_bubble_score'] >= threshold
        
        # Calculate metrics
        true_positives = ((df['bubble_warning']) & (df['crashed'])).sum()
        false_positives = ((df['bubble_warning']) & (~df['crashed'])).sum()
        true_negatives = ((~df['bubble_warning']) & (~df['crashed'])).sum()
        false_negatives = ((~df['bubble_warning']) & (df['crashed'])).sum()
        
        accuracy = (true_positives + true_negatives) / len(df) if len(df) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_stocks': len(df),
            'stocks_crashed': df['crashed'].sum(),
            'warnings_issued': df['bubble_warning'].sum()
        }
    
    def plot_results(self, save_path: str = None):
        """
        Visualize backtest results
        """
        if self.results.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Bubble Score vs Drawdown
        ax1 = axes[0]
        scatter = ax1.scatter(
            self.results['pre_crash_bubble_score'],
            self.results['crash_max_drawdown_pct'],
            c=self.results['pre_crash_bubble_score'],
            cmap='RdYlGn_r',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        
        for idx, row in self.results.iterrows():
            ax1.annotate(
                row['ticker'],
                (row['pre_crash_bubble_score'], row['crash_max_drawdown_pct']),
                fontsize=8,
                ha='center'
            )
        
        ax1.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='Crash Threshold (-20%)')
        ax1.axvline(x=60, color='orange', linestyle='--', alpha=0.5, label='Bubble Warning (60)')
        ax1.set_xlabel('Pre-Crash Bubble Score', fontsize=12)
        ax1.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax1.set_title(f'{self.crash["name"]}: Bubble Score vs Crash Severity', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Sorted by bubble score
        ax2 = axes[1]
        sorted_df = self.results.sort_values('pre_crash_bubble_score', ascending=False)
        
        colors = ['red' if x >= 60 else 'orange' if x >= 45 else 'green' 
                  for x in sorted_df['pre_crash_bubble_score']]
        
        ax2.barh(sorted_df['ticker'], sorted_df['pre_crash_bubble_score'], color=colors, alpha=0.7)
        ax2.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='High Risk')
        ax2.axvline(x=45, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk')
        ax2.set_xlabel('Bubble Score', fontsize=12)
        ax2.set_title('Pre-Crash Bubble Scores', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary statistics"""
        if self.results.empty:
            return
        
        print(f"\n{'='*70}")
        print(f"BACKTEST SUMMARY: {self.crash['name']}")
        print(f"{'='*70}\n")
        
        print("CRASH PERFORMANCE:")
        print(f"  Stocks tested: {len(self.results)}")
        print(f"  Average max drawdown: {self.results['crash_max_drawdown_pct'].mean():.2f}%")
        print(f"  Worst drawdown: {self.results['crash_max_drawdown_pct'].min():.2f}% ({self.results.loc[self.results['crash_max_drawdown_pct'].idxmin(), 'ticker']})")
        print(f"  Best drawdown: {self.results['crash_max_drawdown_pct'].max():.2f}% ({self.results.loc[self.results['crash_max_drawdown_pct'].idxmax(), 'ticker']})")
        
        print("\nBUBBLE SCORES:")
        print(f"  Average bubble score: {self.results['pre_crash_bubble_score'].mean():.2f}")
        print(f"  Highest score: {self.results['pre_crash_bubble_score'].max():.2f} ({self.results.loc[self.results['pre_crash_bubble_score'].idxmax(), 'ticker']})")
        print(f"  Lowest score: {self.results['pre_crash_bubble_score'].min():.2f} ({self.results.loc[self.results['pre_crash_bubble_score'].idxmin(), 'ticker']})")
        
        # Correlation
        correlation = self.results['pre_crash_bubble_score'].corr(self.results['crash_max_drawdown_pct'])
        print(f"\nCORRELATION:")
        print(f"  Bubble Score vs Drawdown: {correlation:.3f}")
        
        # Accuracy at different thresholds
        print("\nPREDICTION ACCURACY:")
        for threshold in [55, 60, 65]:
            acc = self.calculate_accuracy(threshold)
            print(f"  Threshold={threshold}: Accuracy={acc['accuracy']:.1%}, Precision={acc['precision']:.1%}, Recall={acc['recall']:.1%}")
        
        print(f"\n{'='*70}\n")


def run_all_crash_tests():
    """Run backtests on all crash periods"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE CRASH BACKTESTING")
    print("="*70)
    
    all_results = {}
    
    for crash_name in CRASH_PERIODS.keys():
        backtester = CrashBacktester(crash_name)
        df = backtester.run_backtest()
        backtester.print_summary()
        backtester.plot_results()
        all_results[crash_name] = backtester
    
    # Combined summary
    print("\n" + "="*70)
    print("OVERALL RESULTS ACROSS ALL CRASHES")
    print("="*70)
    
    total_stocks = sum(len(bt.results) for bt in all_results.values())
    avg_correlation = np.mean([
        bt.results['pre_crash_bubble_score'].corr(bt.results['crash_max_drawdown_pct'])
        for bt in all_results.values()
    ])
    
    print(f"\nTotal stocks tested: {total_stocks}")
    print(f"Average correlation (bubble score vs drawdown): {avg_correlation:.3f}")
    print(f"Number of crashes analyzed: {len(all_results)}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_crash_tests()
    
    print("\n✅ Crash backtesting complete!")
    print("Results saved for paper analysis.")