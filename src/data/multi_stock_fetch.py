"""
Multi-Stock Data Pipeline

Extends the single-stock fetch.py to handle batch processing of multiple stocks
with error handling, caching, and parallel parameter calculation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import pickle
from pathlib import Path

# Import existing functions to maintain compatibility
from .fetch import calculate_returns, calculate_mu_sigma


# Constants
TRADING_DAYS_PER_YEAR = 252
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class MultiStockDataPipeline:
    """
    Pipeline for fetching and analyzing multiple stocks efficiently.

    Features:
    - Batch downloading with error handling
    - Optional caching to reduce API calls
    - Parallel parameter calculation
    - Correlation matrix computation
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 use_cache: bool = True, cache_expiry_hours: int = 24):
        """
        Initialize the multi-stock pipeline.

        Args:
            tickers: List of stock symbols (e.g., ['NVDA', 'AAPL', 'GOOGL'])
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            use_cache: Whether to cache downloaded data
            cache_expiry_hours: Hours before cache expires
        """
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.use_cache = use_cache
        self.cache_expiry_hours = cache_expiry_hours

        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.returns_data: Dict[str, pd.Series] = {}
        self.parameters: Dict[str, dict] = {}
        self.failed_tickers: List[str] = []

        # Ensure cache directory exists
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        return CACHE_DIR / f"{ticker}_{self.start_date}_{self.end_date}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False

        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        return age_hours < self.cache_expiry_hours

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        cache_path = self._get_cache_path(ticker)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not cache {ticker}: {e}")

    def download_single_stock(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download data for a single stock with error handling and caching.

        Args:
            ticker: Stock symbol

        Returns:
            DataFrame with stock data or None if failed
        """
        # Try cache first
        if self.use_cache:
            cached_data = self._load_from_cache(ticker)
            if cached_data is not None:
                print(f"[CACHE] Loaded {ticker} from cache")
                return cached_data

        # Download from yfinance
        try:
            print(f"[DOWNLOAD] Fetching {ticker}...")
            data = yf.download(ticker, start=self.start_date, end=self.end_date,
                              progress=False, auto_adjust=True)

            if data.empty:
                print(f"[ERROR] No data returned for {ticker}")
                return None

            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            print(f"[OK] {ticker}: {len(data)} days downloaded")

            # Cache the data
            if self.use_cache:
                self._save_to_cache(ticker, data)

            return data

        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")
            return None

    def download_all(self) -> Dict[str, pd.DataFrame]:
        """
        Download data for all tickers.

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        print(f"\n{'='*50}")
        print(f"Downloading {len(self.tickers)} stocks")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"{'='*50}\n")

        self.raw_data = {}
        self.failed_tickers = []

        for ticker in self.tickers:
            data = self.download_single_stock(ticker)
            if data is not None:
                self.raw_data[ticker] = data
            else:
                self.failed_tickers.append(ticker)

        print(f"\n{'='*50}")
        print(f"Download complete: {len(self.raw_data)}/{len(self.tickers)} successful")
        if self.failed_tickers:
            print(f"Failed: {', '.join(self.failed_tickers)}")
        print(f"{'='*50}\n")

        return self.raw_data

    def calculate_all_returns(self) -> Dict[str, pd.Series]:
        """
        Calculate returns for all downloaded stocks.

        Returns:
            Dictionary mapping ticker -> returns Series
        """
        if not self.raw_data:
            print("No data available. Run download_all() first.")
            return {}

        self.returns_data = {}

        for ticker, data in self.raw_data.items():
            try:
                returns = calculate_returns(data)
                self.returns_data[ticker] = returns
            except Exception as e:
                print(f"[ERROR] Could not calculate returns for {ticker}: {e}")

        return self.returns_data

    def calculate_all_parameters(self) -> Dict[str, dict]:
        """
        Calculate Monte Carlo parameters for all stocks.

        Returns:
            Dictionary mapping ticker -> parameters dict
        """
        if not self.raw_data:
            print("No data available. Run download_all() first.")
            return {}

        if not self.returns_data:
            self.calculate_all_returns()

        self.parameters = {}

        for ticker in self.raw_data.keys():
            if ticker not in self.returns_data:
                continue

            try:
                data = self.raw_data[ticker]
                returns = self.returns_data[ticker]

                # Get current price (last closing price)
                S0 = float(data['Close'].iloc[-1])

                # Calculate mu and sigma
                mu, sigma = calculate_mu_sigma(returns)

                self.parameters[ticker] = {
                    'ticker': ticker,
                    'S0': S0,
                    'mu': mu,
                    'sigma': sigma,
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'n_days': len(data),
                    'daily_mean_return': float(returns.mean()),
                    'daily_volatility': float(returns.std())
                }

            except Exception as e:
                print(f"[ERROR] Could not calculate parameters for {ticker}: {e}")

        return self.parameters

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns across all stocks.

        Returns:
            DataFrame with correlation coefficients
        """
        if not self.returns_data:
            self.calculate_all_returns()

        # Combine all returns into a single DataFrame
        returns_df = pd.DataFrame(self.returns_data)

        # Calculate correlation matrix
        correlation = returns_df.corr()

        return correlation

    def get_combined_returns_df(self) -> pd.DataFrame:
        """
        Get all returns combined into a single DataFrame.

        Returns:
            DataFrame with each stock's returns as a column
        """
        if not self.returns_data:
            self.calculate_all_returns()

        return pd.DataFrame(self.returns_data)

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all stocks.

        Returns:
            DataFrame with summary stats for each stock
        """
        if not self.parameters:
            self.calculate_all_parameters()

        summary_data = []

        for ticker, params in self.parameters.items():
            summary_data.append({
                'Ticker': ticker,
                'Current Price': params['S0'],
                'Annual Return (mu)': params['mu'],
                'Annual Volatility (sigma)': params['sigma'],
                'Daily Mean Return': params['daily_mean_return'],
                'Daily Volatility': params['daily_volatility'],
                'Trading Days': params['n_days']
            })

        df = pd.DataFrame(summary_data)
        df.set_index('Ticker', inplace=True)

        return df

    def run_full_pipeline(self) -> Tuple[Dict[str, dict], pd.DataFrame]:
        """
        Run the complete data pipeline.

        Returns:
            Tuple of (parameters dict, correlation matrix)
        """
        self.download_all()
        self.calculate_all_returns()
        self.calculate_all_parameters()
        correlation = self.get_correlation_matrix()

        return self.parameters, correlation


def fetch_multiple_stocks(tickers: List[str], start_date: str, end_date: str,
                         use_cache: bool = True) -> Dict[str, dict]:
    """
    Convenience function to fetch and analyze multiple stocks.

    Args:
        tickers: List of stock symbols
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        use_cache: Whether to use caching

    Returns:
        Dictionary mapping ticker -> parameters dict
    """
    pipeline = MultiStockDataPipeline(tickers, start_date, end_date, use_cache)
    parameters, _ = pipeline.run_full_pipeline()
    return parameters


def get_default_stock_universe() -> List[str]:
    """
    Get a default list of diversified stocks for analysis.

    Returns:
        List of stock tickers across different sectors
    """
    return [
        # Technology
        'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',
        # Finance
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'UNH', 'PFE',
        # Consumer
        'AMZN', 'WMT', 'HD',
        # Energy
        'XOM', 'CVX',
        # Industrial
        'CAT', 'BA',
        # ETFs for benchmarking
        'SPY', 'QQQ'
    ]


# Testing
if __name__ == "__main__":
    print("Testing Multi-Stock Data Pipeline...\n")

    # Define test parameters
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Test with a small set of stocks
    test_tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'SPY']

    # Create pipeline
    pipeline = MultiStockDataPipeline(
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )

    # Run full pipeline
    params, correlation = pipeline.run_full_pipeline()

    # Print results
    print("\n" + "="*60)
    print("STOCK PARAMETERS")
    print("="*60)

    for ticker, p in params.items():
        print(f"\n{ticker}:")
        print(f"  Current Price: ${p['S0']:.2f}")
        print(f"  Annual Return (mu): {p['mu']*100:.2f}%")
        print(f"  Annual Volatility (sigma): {p['sigma']*100:.2f}%")

    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)
    print(correlation.round(3))

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(pipeline.get_summary_statistics().round(4))

    print("\n\nTest complete!")
