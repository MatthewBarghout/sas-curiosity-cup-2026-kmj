"""
Google Trends Sentiment Analysis

Downloads and processes Google Trends data for market sentiment indicators.
Provides sentiment scores for keywords like "AI bubble", "NVDA stock", etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("Warning: pytrends not installed. Run 'pip install pytrends' for Google Trends functionality.")


class GoogleTrendsSentiment:
    """
    Download and analyze Google Trends data for market sentiment.

    Converts search interest into sentiment scores (0-100) where:
    - Higher scores = more search interest (potentially fear/concern for bearish terms)
    - Lower scores = less search interest
    """

    # Default keywords for market sentiment analysis
    DEFAULT_KEYWORDS = [
        'AI bubble',
        'NVDA stock',
        'stock market crash',
        'market correction',
        'buy the dip'
    ]

    # Bearish keywords (high search = negative sentiment)
    BEARISH_KEYWORDS = [
        'stock market crash',
        'market correction',
        'recession',
        'AI bubble',
        'tech bubble',
        'sell stocks'
    ]

    # Bullish keywords (high search = positive sentiment)
    BULLISH_KEYWORDS = [
        'buy the dip',
        'best stocks to buy',
        'stock market rally',
        'bull market'
    ]

    def __init__(self, hl: str = 'en-US', tz: int = 360):
        """
        Initialize Google Trends client.

        Args:
            hl: Host language for trends
            tz: Timezone offset (360 = US Central)
        """
        if not PYTRENDS_AVAILABLE:
            raise ImportError(
                "pytrends is required for Google Trends functionality. "
                "Install it with: pip install pytrends"
            )

        self.pytrends = TrendReq(hl=hl, tz=tz)
        self.cache = {}

    def download_trends(self,
                       keywords: List[str],
                       start_date: str,
                       end_date: str,
                       geo: str = 'US') -> pd.DataFrame:
        """
        Download Google Trends data for specified keywords.

        Args:
            keywords: List of search terms (max 5 per request)
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            geo: Geographic region code (e.g., 'US', '')

        Returns:
            DataFrame with trends data (0-100 scale) indexed by date
        """
        # Google Trends uses a specific timeframe format
        timeframe = f'{start_date} {end_date}'

        # Can only query 5 keywords at once
        if len(keywords) > 5:
            print(f"Warning: Only first 5 keywords will be used (got {len(keywords)})")
            keywords = keywords[:5]

        try:
            self.pytrends.build_payload(
                kw_list=keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )

            df = self.pytrends.interest_over_time()

            if 'isPartial' in df.columns:
                df = df.drop('isPartial', axis=1)

            return df

        except Exception as e:
            print(f"Error fetching trends: {e}")
            return pd.DataFrame()

    def get_sentiment_score(self,
                           keywords: List[str] = None,
                           start_date: str = None,
                           end_date: str = None,
                           geo: str = 'US') -> Dict:
        """
        Get sentiment score (0-100) based on Google Trends data.

        The sentiment score is calculated by:
        1. Normalizing each keyword's search interest (0-100)
        2. Inverting bearish keywords (high search = low sentiment)
        3. Averaging across all keywords

        Args:
            keywords: Search terms to analyze (defaults to DEFAULT_KEYWORDS)
            start_date: Start date (defaults to 90 days ago)
            end_date: End date (defaults to today)
            geo: Geographic region

        Returns:
            Dictionary with sentiment scores and raw data
        """
        if keywords is None:
            keywords = self.DEFAULT_KEYWORDS[:5]  # Max 5

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        # Download trends
        trends_df = self.download_trends(keywords, start_date, end_date, geo)

        if trends_df.empty:
            return {
                'sentiment_score': 50.0,  # Neutral default
                'confidence': 'low',
                'error': 'No data available',
                'keywords': keywords
            }

        # Calculate sentiment scores for each keyword
        keyword_scores = {}

        for keyword in trends_df.columns:
            series = trends_df[keyword]
            current_value = series.iloc[-1] if len(series) > 0 else 50
            avg_value = series.mean()
            max_value = series.max()

            # Normalize to 0-100 (already in this range from Google)
            normalized = current_value

            # For bearish keywords, invert the score (high search = low sentiment)
            if any(bear in keyword.lower() for bear in ['crash', 'bubble', 'correction', 'recession', 'sell']):
                sentiment_contribution = 100 - normalized
            else:
                sentiment_contribution = normalized

            keyword_scores[keyword] = {
                'current': float(current_value),
                'average': float(avg_value),
                'max': float(max_value),
                'sentiment_contribution': float(sentiment_contribution)
            }

        # Calculate overall sentiment score
        contributions = [v['sentiment_contribution'] for v in keyword_scores.values()]
        overall_sentiment = np.mean(contributions)

        # Confidence based on data availability and variance
        confidence = 'high' if len(trends_df) > 30 else 'medium' if len(trends_df) > 7 else 'low'

        return {
            'sentiment_score': float(overall_sentiment),
            'confidence': confidence,
            'interpretation': self._interpret_sentiment(overall_sentiment),
            'keyword_scores': keyword_scores,
            'data_points': len(trends_df),
            'date_range': f"{start_date} to {end_date}",
            'raw_data': trends_df
        }

    def _interpret_sentiment(self, score: float) -> str:
        """Convert sentiment score to human-readable interpretation."""
        if score >= 70:
            return "Very Bullish"
        elif score >= 55:
            return "Bullish"
        elif score >= 45:
            return "Neutral"
        elif score >= 30:
            return "Bearish"
        else:
            return "Very Bearish"

    def get_fear_greed_proxy(self,
                            start_date: str = None,
                            end_date: str = None) -> Dict:
        """
        Create a Fear & Greed proxy index using Google Trends.

        Uses a combination of fear-related and greed-related search terms
        to create a sentiment index similar to CNN's Fear & Greed Index.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with fear/greed scores
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        # Fear keywords
        fear_keywords = ['stock market crash', 'recession', 'market correction']

        # Greed keywords
        greed_keywords = ['best stocks to buy', 'bull market']

        try:
            fear_data = self.download_trends(fear_keywords, start_date, end_date)
            greed_data = self.download_trends(greed_keywords, start_date, end_date)

            if fear_data.empty and greed_data.empty:
                return {'fear_greed_index': 50, 'status': 'Neutral', 'error': 'No data'}

            # Calculate fear score (average of fear keywords)
            fear_score = fear_data.mean(axis=1).iloc[-1] if not fear_data.empty else 50

            # Calculate greed score
            greed_score = greed_data.mean(axis=1).iloc[-1] if not greed_data.empty else 50

            # Fear & Greed Index: 0 = Extreme Fear, 100 = Extreme Greed
            # Higher greed and lower fear = higher index
            if fear_score + greed_score > 0:
                fear_greed_index = (greed_score / (fear_score + greed_score)) * 100
            else:
                fear_greed_index = 50

            # Determine status
            if fear_greed_index >= 75:
                status = "Extreme Greed"
            elif fear_greed_index >= 55:
                status = "Greed"
            elif fear_greed_index >= 45:
                status = "Neutral"
            elif fear_greed_index >= 25:
                status = "Fear"
            else:
                status = "Extreme Fear"

            return {
                'fear_greed_index': float(fear_greed_index),
                'status': status,
                'fear_score': float(fear_score),
                'greed_score': float(greed_score),
                'date': end_date
            }

        except Exception as e:
            return {'fear_greed_index': 50, 'status': 'Neutral', 'error': str(e)}


def download_google_trends(keywords: List[str],
                          start_date: str,
                          end_date: str,
                          geo: str = 'US') -> pd.DataFrame:
    """
    Convenience function to download Google Trends data.

    Args:
        keywords: List of search terms (max 5)
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        geo: Geographic region

    Returns:
        DataFrame with trends data
    """
    sentiment = GoogleTrendsSentiment()
    return sentiment.download_trends(keywords, start_date, end_date, geo)


def get_sentiment_score(keywords: List[str] = None,
                       start_date: str = None,
                       end_date: str = None) -> float:
    """
    Get a simple sentiment score (0-100) for given keywords.

    Args:
        keywords: Search terms to analyze (defaults to AI/NVDA related terms)
        start_date: Start date
        end_date: End date

    Returns:
        Sentiment score from 0 (very bearish) to 100 (very bullish)
    """
    if keywords is None:
        keywords = ['AI bubble', 'NVDA stock']

    sentiment = GoogleTrendsSentiment()
    result = sentiment.get_sentiment_score(keywords, start_date, end_date)

    return result['sentiment_score']


def download_ai_nvda_sentiment(start_date: str = None,
                               end_date: str = None) -> Dict:
    """
    Download sentiment data specifically for AI bubble and NVDA stock searches.

    This is the main deliverable function for Person 3's Feature 2.

    Args:
        start_date: Start date (defaults to 90 days ago)
        end_date: End date (defaults to today)

    Returns:
        Dictionary with sentiment scores (0-100) for AI and NVDA keywords
    """
    keywords = ['AI bubble', 'NVDA stock']

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    sentiment = GoogleTrendsSentiment()
    result = sentiment.get_sentiment_score(keywords, start_date, end_date)

    return result


# Testing
if __name__ == "__main__":
    print("Testing Google Trends Sentiment Analysis...\n")

    if not PYTRENDS_AVAILABLE:
        print("Please install pytrends: pip install pytrends")
        exit(1)

    # Test the main deliverable function
    print("=" * 60)
    print("AI Bubble & NVDA Stock Sentiment Analysis")
    print("=" * 60)

    try:
        result = download_ai_nvda_sentiment()

        print(f"\nOverall Sentiment Score: {result['sentiment_score']:.1f}/100")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Date Range: {result['date_range']}")

        print("\nKeyword Breakdown:")
        for keyword, scores in result.get('keyword_scores', {}).items():
            print(f"  {keyword}:")
            print(f"    Current Interest: {scores['current']:.0f}")
            print(f"    Average Interest: {scores['average']:.1f}")
            print(f"    Sentiment Contribution: {scores['sentiment_contribution']:.1f}")

        print("\n" + "=" * 60)
        print("Fear & Greed Proxy Index")
        print("=" * 60)

        sentiment = GoogleTrendsSentiment()
        fg_result = sentiment.get_fear_greed_proxy()

        print(f"\nFear & Greed Index: {fg_result['fear_greed_index']:.1f}/100")
        print(f"Status: {fg_result['status']}")
        print(f"Fear Score: {fg_result['fear_score']:.1f}")
        print(f"Greed Score: {fg_result['greed_score']:.1f}")

    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nNote: Google Trends may rate-limit requests. Try again in a few minutes.")

    print("\nTest complete!")
