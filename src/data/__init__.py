"""
Data module for stock data fetching and processing.
"""

from .fetch import (
    download_stock_data,
    calculate_returns,
    calculate_mu_sigma,
    get_stock_parameters
)

from .multi_stock_fetch import (
    MultiStockDataPipeline,
    fetch_multiple_stocks,
    get_default_stock_universe
)

__all__ = [
    # Single stock functions
    'download_stock_data',
    'calculate_returns',
    'calculate_mu_sigma',
    'get_stock_parameters',
    # Multi-stock classes and functions
    'MultiStockDataPipeline',
    'fetch_multiple_stocks',
    'get_default_stock_universe'
]
