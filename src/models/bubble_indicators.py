"""
Bubble Risk Indicator System

Calculates 8 bubble indicators and combines them into a composite score (0-100)
where higher scores indicate higher bubble risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import yfinance as yf


class BubbleRiskAnalyzer:
    """
    Analyzes bubble risk for a stock using 8 different indicators.
    """
    
    def __init__(self, ticker: str, lookback_days: int = 365, end_date: str = None):
        """
        Initialize analyzer for a stock.
    
        Args:
            ticker: Stock symbol (e.g., 'NVDA')
            lookback_days: Days of historical data to analyze
            end_date: End date for analysis (YYYY-MM-DD). If None, uses today.
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.end_date = end_date
    
        # Download data
        self._download_data()
        
    def _download_data(self):
        """Download price and volume data."""
        if self.end_date:
            end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
    
        start_date = end_date - timedelta(days=self.lookback_days + 100)
    
        print(f"Downloading {self.ticker} data...")
        self.data = yf.download(
            self.ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
    
        if self.data.empty:
            raise ValueError(f"No data found for {self.ticker}")
    
        # Get stock info
        stock = yf.Ticker(self.ticker)
        self.info = stock.info
    
        print(f"✓ Downloaded {len(self.data)} days of data")
    
    def _calculate_pe_score(self) -> float:
        """Indicator 1: P/E Ratio Score"""
        try:
            pe_ratio = self.info.get('trailingPE', None)
            
            if pe_ratio is None or pe_ratio <= 0:
                return 50.0
            
            if pe_ratio < 15:
                score = (pe_ratio / 15) * 30
            elif pe_ratio < 25:
                score = 30 + ((pe_ratio - 15) / 10) * 20
            elif pe_ratio < 40:
                score = 50 + ((pe_ratio - 25) / 15) * 20
            else:
                score = 70 + min((pe_ratio - 40) / 60 * 30, 30)
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: P/E score error: {e}")
            return 50.0
    
    def _calculate_rsi_score(self, period: int = 14) -> float:
        """Indicator 2: RSI"""
        try:
            closes = self.data['Close'].values.flatten()
            deltas = np.diff(closes)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = pd.Series(gains).rolling(window=period).mean().values
            avg_loss = pd.Series(losses).rolling(window=period).mean().values
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi[-1]
            
            if current_rsi < 30:
                score = (current_rsi / 30) * 20
            elif current_rsi < 50:
                score = 20 + ((current_rsi - 30) / 20) * 20
            elif current_rsi < 70:
                score = 40 + ((current_rsi - 50) / 20) * 30
            else:
                score = 70 + ((current_rsi - 70) / 30) * 30
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: RSI error: {e}")
            return 50.0
    
    def _calculate_ma_deviation_score(self) -> float:
        """Indicator 3: Moving Average Deviation"""
        try:
            closes = self.data['Close'].values.flatten()
            
            if len(closes) < 200:
                return 50.0
            
            ma_200 = np.mean(closes[-200:])
            current_price = closes[-1]
            deviation = (current_price - ma_200) / ma_200
            
            if deviation < 0:
                score = 30 * (1 + deviation)
            elif deviation < 0.20:
                score = 30 + (deviation / 0.20) * 20
            elif deviation < 0.50:
                score = 50 + ((deviation - 0.20) / 0.30) * 25
            else:
                score = 75 + min((deviation - 0.50) / 0.50 * 25, 25)
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: MA deviation error: {e}")
            return 50.0
    
    def _calculate_volume_spike_score(self) -> float:
        """Indicator 4: Volume Spikes"""
        try:
            volumes = self.data['Volume'].values.flatten()
            
            if len(volumes) < 90:
                return 50.0
            
            avg_volume = np.mean(volumes[-90:])
            recent_volume = np.mean(volumes[-10:])
            volume_ratio = recent_volume / avg_volume
            
            if volume_ratio < 0.8:
                score = (volume_ratio / 0.8) * 30
            elif volume_ratio < 1.5:
                score = 30 + ((volume_ratio - 0.8) / 0.7) * 20
            elif volume_ratio < 2.5:
                score = 50 + ((volume_ratio - 1.5) / 1.0) * 25
            else:
                score = 75 + min((volume_ratio - 2.5) / 2.5 * 25, 25)
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: Volume error: {e}")
            return 50.0
    
    def _calculate_vix_comparison_score(self) -> float:
        """Indicator 5: VIX Comparison"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            from data.fetch import download_vix
            
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            vix_data = download_vix(start, end)
            
            # Fix: Handle Series properly
            if isinstance(vix_data['Close'].iloc[-1], pd.Series):
                current_vix = float(vix_data['Close'].iloc[-1].iloc[0]) / 100
            else:
                current_vix = float(vix_data['Close'].iloc[-1]) / 100
            
            returns = self.data['Close'].pct_change().dropna()
            stock_vol = returns.std() * np.sqrt(252)
            stock_vol = float(stock_vol.iloc[0]) if isinstance(stock_vol, pd.Series) else float(stock_vol)
            
            ratio = current_vix / stock_vol
            
            if ratio > 1.2:
                score = 30 * (1 - min((ratio - 1.2) / 0.8, 1))
            elif ratio > 0.8:
                score = 30 + ((1.2 - ratio) / 0.4) * 20
            else:
                score = 50 + ((0.8 - ratio) / 0.8) * 50
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: VIX comparison error: {e}")
            return 50.0
    
    def _calculate_google_trends_score(self) -> float:
        """Indicator 6: Google Trends"""
        try:
            # Fix: Correct import path
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            parent_dir = str(Path(__file__).parent.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Try importing sentiment module
            try:
                from data.sentiment import get_sentiment_score
            except ImportError:
                # Alternative: try direct import
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "sentiment",
                    Path(__file__).parent.parent / "data" / "sentiment.py"
                )
                sentiment = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sentiment)
                get_sentiment_score = sentiment.get_sentiment_score
            
            keywords = [f'{self.ticker} stock', 'AI bubble']
            result = get_sentiment_score(keywords)
            sentiment_score = result if isinstance(result, float) else 50.0
            
            return float(np.clip(sentiment_score, 0, 100))
            
        except Exception as e:
            print(f"Warning: Google Trends error: {e}")
            return 50.0
    
    def _calculate_sector_correlation_score(self) -> float:
        """Indicator 7: Sector Correlation"""
        try:
            # Define sector peers
            sector_peers = {
                'NVDA': ['AMD', 'INTC', 'TSM'],
                'AMD': ['NVDA', 'INTC', 'TSM'],
                'INTC': ['NVDA', 'AMD', 'TSM'],
                'AAPL': ['MSFT', 'GOOGL', 'META'],
                'MSFT': ['AAPL', 'GOOGL', 'META'],
                'GOOGL': ['MSFT', 'META', 'AAPL'],
                'META': ['GOOGL', 'AAPL', 'MSFT'],
                'TSLA': ['F', 'GM', 'RIVN'],
                'AMZN': ['WMT', 'TGT', 'COST']
            }
            
            peers = sector_peers.get(self.ticker, [])
            print(f"  Checking correlation with peers: {peers}")
            
            if not peers:
                print(f"  No peers defined for {self.ticker}")
                return 50.0
            
        
            # Calculate date range based on historical end_date if provided
            if self.end_date:
                end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
                end = self.end_date
                start = (end_date_obj - timedelta(days=180)).strftime('%Y-%m-%d')
            else:
                end = datetime.now().strftime('%Y-%m-%d')
                start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            stock_returns = self.data['Close'].pct_change().dropna()
            
            correlations = []
            for peer in peers:
                try:
                    print(f"  Downloading {peer}...")
                    peer_data = yf.download(peer, start=start, end=end, progress=False, auto_adjust=True)
                    
                    if peer_data.empty:
                        print(f"  {peer}: No data")
                        continue
                    
                   # Handle potential DataFrame/Series issue
                    if isinstance(peer_data['Close'], pd.DataFrame):
                        peer_close = peer_data['Close'].iloc[:, 0]  # Get first column
                    else:
                        peer_close = peer_data['Close']

                    peer_returns = peer_close.pct_change().dropna()
                    
                    # Find common dates
                    common_dates = stock_returns.index.intersection(peer_returns.index)
                    
                    print(f"  {peer}: {len(common_dates)} common dates")
                    
                    if len(common_dates) > 50:
                        # Calculate correlation using numpy for cleaner handling
                        stock_vals = stock_returns.loc[common_dates].values.flatten()
                        peer_vals = peer_returns.loc[common_dates].values.flatten()
                        corr = np.corrcoef(stock_vals, peer_vals)[0, 1]
                        correlations.append(float(corr))
                        print(f"  {peer}: correlation = {corr:.3f}")
                except Exception as e:
                    print(f"  {peer}: Error - {e}")
                    continue
            
            if not correlations:
                print(f"  No valid correlations calculated")
                return 50.0
            
            avg_correlation = np.mean(correlations)
            print(f"  Average correlation: {avg_correlation:.3f}")
            
            # Score based on correlation
            if avg_correlation > 0.7:
                score = 20 + ((1.0 - avg_correlation) / 0.3) * 20
            elif avg_correlation > 0.4:
                score = 40 + ((0.7 - avg_correlation) / 0.3) * 20
            elif avg_correlation > 0:
                score = 60 + ((0.4 - avg_correlation) / 0.4) * 20
            else:
                score = 80 + min(abs(avg_correlation) * 20, 20)
            
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"Warning: Sector correlation error: {e}")
            return 50.0
    
    def _calculate_price_acceleration_score(self) -> float:
        """Indicator 8: Price Acceleration"""
        try:
            closes = self.data['Close'].values.flatten()
        
            if len(closes) < 90:
                return 50.0
        
            # Calculate returns over different periods
            return_30d = (closes[-1] - closes[-30]) / closes[-30]
            return_60d = (closes[-1] - closes[-60]) / closes[-60]
            return_90d = (closes[-1] - closes[-90]) / closes[-90]
        
            # Annualize
            annual_30d = (1 + return_30d) ** (252/30) - 1
            annual_60d = (1 + return_60d) ** (252/60) - 1
            annual_90d = (1 + return_90d) ** (252/90) - 1
        
            # Average acceleration
            avg_acceleration = (annual_30d + annual_60d + annual_90d) / 3
        
            # ADJUSTED THRESHOLDS - less aggressive
            # Score based on acceleration
            # <0%: declining, no risk (0-20)
            # 0-50%: normal growth (20-50)
            # 50-100%: elevated growth (50-75)
            # >100%: parabolic, bubble risk (75-100)
        
            if avg_acceleration < 0:
                score = 20 * (1 + avg_acceleration)  # Negative reduces score
            elif avg_acceleration < 0.50:  # Changed from 0.20
                score = 20 + (avg_acceleration / 0.50) * 30
            elif avg_acceleration < 1.00:  # Changed from 0.50
                score = 50 + ((avg_acceleration - 0.50) / 0.50) * 25
            else:
                score = 75 + min((avg_acceleration - 1.00) / 1.00 * 25, 25)
        
            return float(np.clip(score, 0, 100))
        
        except Exception as e:
            print(f"Warning: Acceleration error: {e}")
            return 50.0
    
    def calculate_all_indicators(self) -> Dict[str, float]:
        """Calculate all 8 indicators"""
        print(f"\nCalculating bubble indicators for {self.ticker}...")
        
        indicators = {
            'valuation': self._calculate_pe_score(),
            'momentum': self._calculate_rsi_score(),
            'trend': self._calculate_ma_deviation_score(),
            'volume': self._calculate_volume_spike_score(),
            'market_fear': self._calculate_vix_comparison_score(),
            'sentiment': self._calculate_google_trends_score(),
            'correlation': self._calculate_sector_correlation_score(),
            'acceleration': self._calculate_price_acceleration_score()
        }
        
        return indicators
    
    def get_composite_score(self) -> Dict:
        """Calculate composite score"""
        indicators = self.calculate_all_indicators()
        
        weights = {
            'valuation': 0.15,
            'momentum': 0.15,
            'trend': 0.12,
            'volume': 0.10,
            'market_fear': 0.15,
            'sentiment': 0.13,
            'correlation': 0.10,
            'acceleration': 0.10
        }
        
        composite = sum(indicators[key] * weights[key] for key in indicators.keys())
        
        if composite >= 75:
            interpretation = "EXTREME BUBBLE RISK"
            color = "🔴"
        elif composite >= 60:
            interpretation = "HIGH BUBBLE RISK"
            color = "🟠"
        elif composite >= 45:
            interpretation = "MODERATE RISK"
            color = "🟡"
        elif composite >= 30:
            interpretation = "LOW RISK"
            color = "🟢"
        else:
            interpretation = "VERY LOW RISK"
            color = "🟢"
        
        return {
            'ticker': self.ticker,
            'composite_score': round(composite, 2),
            'interpretation': interpretation,
            'color': color,
            'indicators': {k: round(v, 2) for k, v in indicators.items()}
        }
    
    def print_report(self):
        """Print formatted report"""
        result = self.get_composite_score()
        
        print("\n" + "="*70)
        print(f"BUBBLE RISK ANALYSIS: {result['ticker']}")
        print("="*70)
        
        print(f"\n{result['color']} COMPOSITE SCORE: {result['composite_score']}/100")
        print(f"   {result['interpretation']}")
        
        print("\n" + "-"*70)
        print("INDICATOR BREAKDOWN:")
        print("-"*70)
        
        for name, score in result['indicators'].items():
            bar_length = int(score / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{name.capitalize():20s} [{bar}] {score:5.1f}/100")
        
        print("="*70 + "\n")
        
        return result


def get_bubble_score(ticker: str) -> float:
    """Quick function to get bubble score"""
    analyzer = BubbleRiskAnalyzer(ticker)
    result = analyzer.get_composite_score()
    return result['composite_score']


if __name__ == "__main__":
    print("="*70)
    print("TESTING BUBBLE RISK ANALYZER")
    print("="*70)
    
    print("\nAnalyzing NVDA...")
    nvda_analyzer = BubbleRiskAnalyzer('NVDA')
    nvda_result = nvda_analyzer.print_report()
    
    print("\n✅ Bubble Risk Analyzer test complete!")
