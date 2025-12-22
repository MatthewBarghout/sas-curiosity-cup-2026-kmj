
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data.
    
    Args:
        ticker: Stock symbol (e.g., 'NVDA')
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        
    Returns:
        DataFrame with stock prices
    """
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    print(f"Downloaded {len(data)} days of data")
    return data


def calculate_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate daily returns from closing prices.
    
    Args:
        prices: DataFrame with 'Close' column
        
    Returns:
        Series of daily returns
    """
    returns = prices['Close'].pct_change().dropna()
    return returns


def calculate_mu_sigma(returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate annualized drift (mu) and volatility (sigma) from returns.
    
    Args:
        returns: Series of daily returns
        
    Returns:
        (mu, sigma): Annual drift and volatility
    """
    # Daily statistics
    daily_mean = returns.mean()
    daily_std = returns.std()
    
    # Annualize (252 trading days per year)
    mu = float(daily_mean * 252)
    sigma = float(daily_std * np.sqrt(252))
    
    return mu, sigma


def get_stock_parameters(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Get all parameters needed for Monte Carlo simulation.
    
    Args:
        ticker: Stock symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        
    Returns:
        Dictionary with S0, mu, sigma, and other stats
    """
    # Download data
    data = download_stock_data(ticker, start_date, end_date)
    
    # Calculate returns
    returns = calculate_returns(data)
    
    # Get current price (last closing price)
    S0 = float(data['Close'].iloc[-1])
    
    # Calculate mu and sigma
    mu, sigma = calculate_mu_sigma(returns)
    
    return {
        'ticker': ticker,
        'S0': S0,
        'mu': mu,
        'sigma': sigma,
        'start_date': start_date,
        'end_date': end_date,
        'n_days': len(data),
        'daily_mean_return': returns.mean(),
        'daily_volatility': returns.std()
    }


# Test it
if __name__ == "__main__":
    print("Testing data fetcher...\n")
    
    # Get last 2 years of NVDA data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Get parameters
    params = get_stock_parameters('NVDA', start_date, end_date)
    
    # Print results
    print(f"\n{params['ticker']} Analysis:")
    print(f"Current Price (S0): ${params['S0']:.2f}")
    print(f"Annual Return (mu): {params['mu']*100:.2f}%")
    print(f"Annual Volatility (sigma): {params['sigma']*100:.2f}%")
    print(f"Data from: {params['start_date']} to {params['end_date']}")
    print(f"Trading days: {params['n_days']}")
    
    print("\nTest complete")