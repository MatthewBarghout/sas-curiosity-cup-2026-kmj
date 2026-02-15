"""
Live API Update System
Runs hourly to update all stock data, GARCH forecasts, and bubble scores
Saves to JSON for API consumption
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetch import download_multiple_stocks, calculate_returns_multiple
from src.models.garch import MultiStockGARCH
from src.models.bubble_indicators import BubbleRiskAnalyzer
from datetime import datetime, timedelta
import pandas as pd
import time
import json

# Stock universe - 41 stocks
TICKERS = [
    # Technology
    'NVDA', 'AMD', 'INTC', 'QCOM', 'MU', 'TSM', 'ASML', 'MSFT', 'GOOGL', 'META',
    # Finance
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'NEE', 'ENPH',
    # Healthcare
    'JNJ', 'PFE', 'ABBV', 'MRK', 'GILD', 'AMGN',
    # Consumer
    'AMZN', 'WMT', 'TGT', 'PG', 'KO', 'COST',
    # Industrial
    'BA', 'CAT', 'GE', 'UNP', 'HON'
]

SECTORS = {
    'Technology': ['NVDA', 'AMD', 'INTC', 'QCOM', 'MU', 'TSM', 'ASML', 'MSFT', 'GOOGL', 'META'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'NEE', 'ENPH'],
    'Healthcare': ['JNJ', 'PFE', 'ABBV', 'MRK', 'GILD', 'AMGN'],
    'Consumer': ['AMZN', 'WMT', 'TGT', 'PG', 'KO', 'COST'],
    'Industrial': ['BA', 'CAT', 'GE', 'UNP', 'HON']
}

UPDATE_INTERVAL = 3600  # 1 hour
DATA_DIR = Path(__file__).parent.parent / "data" / "live"

def update_cycle():
    """Run one complete update cycle"""
    timestamp = datetime.now()
    print(f"\n{'='*70}")
    print(f"UPDATE CYCLE - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # 1. Download latest data
    print("1. Downloading stock data...")
    end = timestamp.strftime('%Y-%m-%d')
    start = (timestamp - timedelta(days=730)).strftime('%Y-%m-%d')
    
    data_dict = download_multiple_stocks(TICKERS, start, end)
    returns_dict = calculate_returns_multiple(data_dict)
    successful = sum(1 for d in data_dict.values() if d is not None)
    print(f"   ✓ Downloaded {successful}/{len(TICKERS)} stocks ({time.time()-start_time:.1f}s)")
    
    # 2. Fit GARCH models
    print("\n2. Fitting GARCH models...")
    garch_start = time.time()
    garch = MultiStockGARCH(returns_dict)
    garch.fit_all(verbose=False)
    forecasts_dict = garch.forecast_all(180)
    current_vols = garch.get_current_volatilities()
    
    # Convert to avg
    forecasts = {t: float(arr.mean()) if arr is not None else None 
                 for t, arr in forecasts_dict.items()}
    
    print(f"   ✓ Fitted {len([m for m in garch.models.values() if m])}/{len(TICKERS)} models ({time.time()-garch_start:.1f}s)")
    
    # 3. Calculate bubble scores
    print("\n3. Calculating bubble scores...")
    bubble_start = time.time()
    bubble_scores = {}
    bubble_details = {}
    
    for ticker in TICKERS:
        if data_dict.get(ticker) is not None:
            try:
                analyzer = BubbleRiskAnalyzer(ticker, lookback_days=365)
                result = analyzer.get_composite_score()
                bubble_scores[ticker] = result['composite_score']
                bubble_details[ticker] = result['indicators']
            except Exception as e:
                print(f"   Warning: {ticker} bubble score failed - {e}")
                bubble_scores[ticker] = None
                bubble_details[ticker] = None
    
    successful_scores = sum(1 for s in bubble_scores.values() if s is not None)
    print(f"   ✓ Calculated {successful_scores}/{len(TICKERS)} scores ({time.time()-bubble_start:.1f}s)")
    
    # 4. Calculate sector averages
    print("\n4. Calculating sector statistics...")
    sector_stats = {}
    
    for sector, stocks in SECTORS.items():
        sector_bubble = []
        sector_vol = []
        
        for ticker in stocks:
            if bubble_scores.get(ticker):
                sector_bubble.append(bubble_scores[ticker])
            if current_vols.get(ticker):
                sector_vol.append(current_vols[ticker] * 100)
        
        sector_stats[sector] = {
            'avg_bubble_score': float(sum(sector_bubble) / len(sector_bubble)) if sector_bubble else None,
            'avg_volatility': float(sum(sector_vol) / len(sector_vol)) if sector_vol else None,
            'stock_count': len(stocks)
        }
    
    # 5. Build results
    print("\n5. Building JSON output...")
    results = {
        'metadata': {
            'timestamp': timestamp.isoformat(),
            'update_duration_seconds': time.time() - start_time,
            'total_stocks': len(TICKERS),
            'successful_downloads': successful,
            'successful_garch': len([m for m in garch.models.values() if m]),
            'successful_bubble': successful_scores
        },
        'stocks': {},
        'sectors': sector_stats,
        'rankings': {
            'highest_risk': [],
            'lowest_risk': []
        }
    }
    
    # Add individual stocks
    for ticker in TICKERS:
        current_price = None
        if data_dict.get(ticker) is not None:
            try:
                current_price = float(data_dict[ticker]['Close'].iloc[-1])
            except:
                pass
        
        results['stocks'][ticker] = {
            'current_price': current_price,
            'garch_vol_pct': float(current_vols.get(ticker, 0) * 100) if current_vols.get(ticker) else None,
            'forecast_vol_pct': float(forecasts.get(ticker, 0) * 100) if forecasts.get(ticker) else None,
            'bubble_score': float(bubble_scores.get(ticker, 0)) if bubble_scores.get(ticker) else None,
            'bubble_indicators': bubble_details.get(ticker)
        }
    
    # Rankings
    valid_scores = [(t, s) for t, s in bubble_scores.items() if s is not None]
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
    
    results['rankings']['highest_risk'] = [
        {'ticker': t, 'score': float(s)} for t, s in sorted_scores[:10]
    ]
    results['rankings']['lowest_risk'] = [
        {'ticker': t, 'score': float(s)} for t, s in sorted_scores[-10:]
    ]
    
    # 6. Save results
    print("\n6. Saving results...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save timestamped file
    filename = f"update_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = DATA_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as latest
    latest_path = DATA_DIR / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Saved to {filepath}")
    print(f"   ✓ Updated latest.json")
    
    print(f"\n{'='*70}")
    print(f"✅ UPDATE COMPLETE - Total time: {time.time()-start_time:.1f}s")
    print(f"{'='*70}\n")
    
    return results

def run_continuously():
    """Run updates every hour"""
    print("="*70)
    print("LIVE API UPDATE SYSTEM")
    print("="*70)
    print(f"Update interval: {UPDATE_INTERVAL/3600:.1f} hours")
    print(f"Total stocks: {len(TICKERS)}")
    print(f"Output directory: {DATA_DIR}")
    print(f"Press Ctrl+C to stop\n")
    
    while True:
        try:
            update_cycle()
            print(f"Next update in {UPDATE_INTERVAL/60:.0f} minutes...")
            time.sleep(UPDATE_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n✋ Stopping update system...")
            break
        except Exception as e:
            print(f"\n❌ Error in update cycle: {e}")
            print("Retrying in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Live API update system for bubble risk analysis')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=3600, help='Update interval in seconds (default: 3600)')
    args = parser.parse_args()
    
    if args.interval:
        UPDATE_INTERVAL = args.interval
    
    if args.once:
        update_cycle()
    else:
        run_continuously()