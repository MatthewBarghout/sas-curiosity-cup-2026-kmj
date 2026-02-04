"""
Data module

Provides data fetching and sentiment analysis utilities.
"""

from .fetch import (
    download_stock_data,
    calculate_returns,
    calculate_mu_sigma,
    download_vix,
    get_vix_level,
    get_vix_regime,
    compare_vix_to_garch,
    download_multiple_stocks,
    calculate_returns_multiple
)

from .sentiment import (
    GoogleTrendsSentiment,
    download_google_trends,
    get_sentiment_score,
    download_ai_nvda_sentiment,
    get_bubble_sentiment_indicator
)

from .multi_stock_fetch import (
    MultiStockDataPipeline,
    fetch_multiple_stocks,
    get_default_stock_universe
)