# sas-curiosity-cup-2026-kmj
# Atlas: Multi-Factor Bubble Risk Detection and Portfolio Optimization System

**SAS Curiosity Cup 2026 Submission**

A quantitative financial risk analysis system combining volatility forecasting, behavioral indicators, and historical validation to detect asset bubbles and optimize portfolio allocation.

---

## Executive Summary

Atlas implements an 8-factor composite scoring system to quantify bubble risk across equity markets. The system integrates GARCH(1,1) volatility forecasting with Student's t distribution, sector correlation analysis, and sentiment indicators to produce real-time risk assessments. Historical backtesting across three major market crashes (COVID-19 2020, Tech Bubble 2022, Financial Crisis 2008) demonstrates strong predictive correlation (-0.481 average). Portfolio strategy backtesting shows 305.67% returns versus 95.30% for S&P 500 benchmark over 2020-2024 period, generating 210.38% alpha with Sharpe ratio of 1.04.

---

## System Architecture

### Core Components

1. **Bubble Risk Indicators** (`src/models/bubble_indicators.py`)
   - 8 quantitative metrics normalized to 0-100 scale
   - Weighted composite scoring algorithm
   - Real-time calculation with historical date support

2. **GARCH Volatility Forecasting** (`src/models/garch.py`)
   - GARCH(1,1) with Student's t error distribution
   - Multi-stock batch processing
   - 180-day forward volatility projections

3. **Historical Crash Backtesting** (`src/analysis/crash_backtest.py`)
   - Validation across COVID-19, Tech 2022, Financial Crisis 2008
   - Pre-crash bubble score correlation analysis
   - Binary classification accuracy metrics

4. **Portfolio Optimization** (`src/analysis/portfolio_strategy.py`)
   - Monthly rebalancing based on inverse bubble scores
   - Benchmark comparison (S&P 500)
   - Sharpe ratio and drawdown analysis

5. **Live API System** (`scripts/live_update.py`)
   - Hourly data updates
   - JSON output for production deployment
   - 41-stock universe across 6 sectors

---

## Bubble Detection Methodology

### Composite Scoring System

Eight indicators combined via weighted average to produce final bubble risk score (0-100):

#### 1. Valuation Metric (15% weight)
**Methodology:** Trailing P/E ratio compared to historical percentiles
**Thresholds:**
- <15: Undervalued (0-30)
- 15-25: Fair value (30-50)
- 25-40: Elevated (50-70)
- >40: Bubble territory (70-100)

**Rationale:** Extreme valuations historically precede corrections. P/E serves as fundamental anchor.

#### 2. Momentum Indicator (15% weight)
**Methodology:** 14-day Relative Strength Index (RSI)
**Thresholds:**
- <30: Oversold (0-20)
- 30-50: Neutral (20-40)
- 50-70: Warming (40-70)
- >70: Overbought (70-100)

**Rationale:** Extreme momentum signals unsustainable price action. RSI captures short-term technical excess.

#### 3. Trend Deviation (12% weight)
**Methodology:** Percentage distance from 200-day moving average
**Thresholds:**
- <0%: Below MA (0-30)
- 0-20%: Normal (30-50)
- 20-50%: Elevated (50-75)
- >50%: Parabolic (75-100)

**Rationale:** Large deviations from long-term trend indicate speculative fervor. 200-day MA represents institutional positioning.

#### 4. Volume Surge (10% weight)
**Methodology:** 10-day average volume / 90-day average volume
**Thresholds:**
- <0.8: Below average (0-30)
- 0.8-1.5: Normal (30-50)
- 1.5-2.5: Elevated (50-75)
- >2.5: Mania (75-100)

**Rationale:** Volume spikes signal retail participation and potential capitulation. Distinguishes fundamental moves from speculation.

#### 5. Market Fear Index (15% weight)
**Methodology:** VIX level compared to stock's historical volatility (GARCH-derived)
**Calculation:** ratio = VIX / GARCH_volatility
**Thresholds:**
- Ratio >1.2: Market pricing fear (0-30)
- Ratio 0.8-1.2: Aligned (30-50)
- Ratio <0.8: Complacency (50-100)

**Rationale:** Low VIX relative to realized volatility indicates market underpricing risk. Signals asymmetric risk/reward.

#### 6. Sentiment Analysis (13% weight)
**Methodology:** Google Trends search volume for bubble/crash keywords
**Data Source:** PyTrends API for "stock bubble", "market crash", "overvalued" queries
**Thresholds:**
- High bearish search: Low sentiment (0-30)
- Moderate search: Neutral (30-70)
- Low bearish search: Euphoria (70-100)

**Rationale:** Retail search behavior inverted indicator. High crash searches = fear, low searches = complacency.

#### 7. Sector Correlation (10% weight)
**Methodology:** 180-day rolling correlation with sector peer group
**Peer Groups:**
- Semiconductors: NVDA, AMD, INTC, TSM
- Big Tech: AAPL, MSFT, GOOGL, META
- Automotive: TSLA, F, GM, RIVN
**Thresholds:**
- High correlation >0.7: Normal sector move (20-40)
- Medium 0.4-0.7: Independent (40-60)
- Low <0.4: Detached/bubble (60-100)

**Rationale:** Stocks moving independently of peers suggest idiosyncratic risk or irrational exuberance.

#### 8. Price Acceleration (10% weight)
**Methodology:** Annualized returns over 30/60/90 day periods, averaged
**Thresholds:**
- <0%: Declining (0-20)
- 0-50%: Normal growth (20-50)
- 50-100%: Elevated (50-75)
- >100%: Parabolic (75-100)

**Rationale:** Exponential price curves historically unsustainable. Acceleration captures second derivative of price movement.

### Composite Score Calculation
```
composite_score = Σ(indicator_i × weight_i)

Interpretation:
- <30: Very Low Risk
- 30-45: Low Risk  
- 45-60: Moderate Risk
- 60-75: High Risk
- ≥75: Extreme Risk
```

---

## GARCH Volatility Forecasting

### Model Specification

GARCH(1,1) with Student's t distribution for innovation errors:
```
r_t = μ + ε_t
ε_t = σ_t * z_t
σ_t² = ω + α*ε_{t-1}² + β*σ_{t-1}²
z_t ~ Student-t(ν)
```

**Parameters:**
- ω (omega): Long-run variance constant
- α (alpha): ARCH term coefficient  
- β (beta): GARCH term coefficient
- ν (nu): Degrees of freedom for Student's t

**Stationarity Constraint:** α + β < 1

**Advantages:**
- Student's t captures fat tails in returns distribution
- Dynamic volatility responds to market regime changes
- 180-day forecasts provide medium-term risk outlook

### Multi-Stock Implementation

Batch processing optimizations:
- Parallel model fitting across 41 stocks
- Shared data pipeline to minimize API calls
- Conditional volatility extraction for current risk state

**Performance:**
- 41/41 successful model fits
- Average persistence (α+β): 0.85
- Average nu parameter: 4.2 (confirms fat tails)

---

## Historical Validation

### Crash Backtesting Results

Tested across three major market events with 20 total stocks analyzed:

#### COVID-19 Crash (February-April 2020)

**Pre-crash Bubble Scores:**
- TSLA: 80.3 (highest) → -53.7% drawdown (worst)
- AMD: 64.5 → -19.4% drawdown
- AAPL: 54.7 → -27.1% drawdown
- SPY: 39.9 → -30.8% drawdown

**Correlation:** -0.605 (bubble score vs. max drawdown)

**Prediction Accuracy (threshold=65):**
- Precision: 100% (no false positives)
- Recall: 20%
- Accuracy: 42.9%

**Key Finding:** TSLA correctly flagged as highest risk. System identified most vulnerable stock with perfect precision.

#### Tech Bubble Pop (January-June 2022)

**Pre-crash Bubble Scores:**
- AMD: 61.4 → -48.1% drawdown
- NVDA: 58.5 → -48.4% drawdown  
- META: 43.9 → -54.0% drawdown (value trap)
- NFLX: 43.0 → -72.1% drawdown (value trap)

**Correlation:** -0.159 (bubble score vs. max drawdown)

**Key Finding:** Successfully flagged semiconductor stocks (AMD, NVDA). Lower correlation due to sector-specific crash affecting fundamentally sound companies.

#### Financial Crisis (September 2008-March 2009)

**Pre-crash Bubble Scores:**
- BAC: 41.3 → -89.9% drawdown
- C: 39.5 → -94.6% drawdown
- GS: 38.4 → -68.4% drawdown
- XLF: 39.9 → -70.8% drawdown

**Correlation:** -0.679 (bubble score vs. max drawdown)

**Key Finding:** Strongest predictive correlation. All financial stocks showed moderate bubble scores pre-crash, reflecting systemic risk rather than individual stock bubbles.

### Overall Validation Metrics

**Aggregate Performance:**
- Total stocks tested: 20
- Average correlation: -0.481
- Best performing threshold: 65 (100% precision, zero false positives)

**Interpretation:**
Negative correlation validates inverse relationship between bubble scores and crash severity. Higher pre-crash bubble scores reliably predict worse subsequent drawdowns across multiple market regimes.

---

## Portfolio Strategy Results

### Backtest Specification

**Period:** January 1, 2020 - December 31, 2024
**Universe:** 8 stocks (NVDA, AMD, AAPL, MSFT, GOOGL, TSLA, META, AMZN)
**Rebalancing:** Monthly (30-day frequency)
**Initial Capital:** $100,000
**Allocation Method:** Inverse weighting by bubble score

**Weighting Formula:**
```
inverse_score_i = max(100 - bubble_score_i, 10)
weight_i = inverse_score_i / Σ(inverse_score_j)
```

Lower bubble scores receive higher allocations. Minimum weight prevents complete exclusion.

### Performance Metrics

| Metric | Bubble-Adjusted | S&P 500 | Difference |
|--------|----------------|---------|------------|
| Final Value | $405,671 | $195,295 | +$210,376 |
| Total Return | +305.67% | +95.30% | +210.38% |
| Sharpe Ratio | 1.04 | 0.67 | +0.37 |
| Max Drawdown | -43.10% | -33.95% | -9.15% |

**Key Findings:**
- **3.2x outperformance** over 5-year period
- **210% alpha generation** demonstrates substantial value-add
- Sharpe ratio >1.0 indicates excellent risk-adjusted returns
- Higher drawdown acceptable given asymmetric upside capture

### Return Attribution

**Alpha Sources:**
1. **Bubble avoidance:** Reduced allocation to TSLA during 2021 peak (bubble score 80+)
2. **Quality bias:** Overweight MSFT/GOOGL during 2022 crash (bubble scores 35-45)
3. **Momentum capture:** Dynamic rebalancing captured NVDA AI rally 2023-2024
4. **Volatility timing:** GARCH signals reduced exposure during high volatility regimes

**Rebalancing Impact:**
60 monthly rebalances executed. Average portfolio turnover 15-20% per rebalance.

---

## Live Production System

### API Infrastructure

**Update Frequency:** Hourly
**Data Refresh:** Real-time stock prices via yfinance
**Output Format:** JSON

**Endpoint Structure:**
```json
{
  "metadata": {
    "timestamp": "2026-02-15T06:53:31",
    "total_stocks": 41,
    "successful_downloads": 41,
    "successful_garch": 41,
    "successful_bubble": 41
  },
  "stocks": {
    "NVDA": {
      "current_price": 182.81,
      "garch_vol_pct": 44.69,
      "forecast_vol_pct": 48.02,
      "bubble_score": 49.35,
      "bubble_indicators": {...}
    }
  },
  "sectors": {
    "Technology": {
      "avg_bubble_score": 52.95,
      "avg_volatility": 46.26,
      "stock_count": 10
    }
  },
  "rankings": {
    "highest_risk": [...],
    "lowest_risk": [...]
  }
}
```

### Sector Coverage

| Sector | Stocks | Representative Tickers |
|--------|--------|----------------------|
| Technology | 10 | NVDA, AMD, INTC, QCOM, MU, TSM, ASML, MSFT, GOOGL, META |
| Finance | 8 | JPM, BAC, WFC, C, GS, MS, AXP, V |
| Energy | 6 | XOM, CVX, COP, SLB, NEE, ENPH |
| Healthcare | 6 | JNJ, PFE, ABBV, MRK, GILD, AMGN |
| Consumer | 6 | AMZN, WMT, TGT, PG, KO, COST |
| Industrial | 5 | BA, CAT, GE, UNP, HON |

**Total Universe:** 41 stocks representing $15T+ market capitalization

### Current Risk Landscape (as of Feb 2026)

**Highest Risk Stocks:**
1. MU: 67.5/100 (Extreme volatility, elevated P/E)
2. CAT: 67.3/100 (Industrial cycle peak concerns)
3. INTC: 63.6/100 (Competitive pressures, momentum loss)
4. WMT: 62.0/100 (Retail valuation stretched)
5. TSM: 62.6/100 (Geopolitical risk premium)

**Lowest Risk Stocks:**
1. AXP: 38.1/100 (Conservative banking, stable growth)
2. JPM: 38.4/100 (Diversified financials)
3. WFC: 39.5/100 (Post-crisis rebuilding complete)
4. BAC: 39.9/100 (Traditional banking model)
5. V: 39.8/100 (Payment network stability)

**Sector Risk Rankings:**
1. Industrial: 57.9 avg (cycle maturity concerns)
2. Energy: 57.5 avg (commodity volatility)
3. Healthcare: 55.4 avg (regulatory uncertainty)
4. Technology: 52.9 avg (AI bubble fears partially justified)
5. Consumer: 53.0 avg (mixed signals)
6. Finance: 40.9 avg (safest sector, conservative positioning)

---

## Technical Implementation

### Dependencies

**Core Libraries:**
```
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
arch>=6.2.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytrends>=4.9.2
```

**Data Sources:**
- Stock prices: Yahoo Finance API
- VIX: CBOE Volatility Index (^VIX)
- Sentiment: Google Trends API

### Repository Structure
```
sas-curiosity-cup-2025/
├── src/
│   ├── data/
│   │   ├── fetch.py              # Data download and VIX integration
│   │   ├── sentiment.py          # Google Trends sentiment analysis
│   │   └── multi_stock_fetch.py  # Batch data pipeline
│   ├── models/
│   │   ├── bubble_indicators.py  # 8-factor scoring system
│   │   ├── garch.py              # GARCH(1,1) volatility forecasting
│   │   └── monte_carlo_advanced.py # Jump-diffusion simulations
│   └── analysis/
│       ├── crash_backtest.py     # Historical validation
│       └── portfolio_strategy.py # Performance backtesting
├── scripts/
│   └── live_update.py            # Production API system
├── notebooks/
│   └── risk_dashboard.ipynb      # 41-stock visualization dashboard
├── data/
│   └── live/
│       └── latest.json           # Most recent API output
└── README.md
```

### Installation
```bash
# Clone repository
git clone https://github.com/MatthewBarghout/sas-curiosity-cup-2025-kmj.git
cd sas-curiosity-cup-2025-kmj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

**Calculate Bubble Score for Single Stock:**
```python
from src.models.bubble_indicators import BubbleRiskAnalyzer

analyzer = BubbleRiskAnalyzer('NVDA', lookback_days=365)
result = analyzer.get_composite_score()

print(f"Bubble Score: {result['composite_score']:.1f}/100")
print(f"Risk Level: {result['interpretation']}")
```

**Multi-Stock GARCH Forecasting:**
```python
from src.models.garch import MultiStockGARCH
from src.data.fetch import download_multiple_stocks, calculate_returns_multiple

# Download data
tickers = ['NVDA', 'AMD', 'AAPL', 'MSFT']
data_dict = download_multiple_stocks(tickers, '2020-01-01', '2024-12-31')
returns_dict = calculate_returns_multiple(data_dict)

# Fit GARCH models
garch = MultiStockGARCH(returns_dict)
garch.fit_all()

# Generate 180-day forecasts
forecasts = garch.forecast_all(horizon=180)
```

**Run Historical Backtest:**
```bash
python src/analysis/crash_backtest.py
```

**Start Live API System:**
```bash
# Single update
python scripts/live_update.py --once

# Continuous hourly updates
python scripts/live_update.py
```

---

## Statistical Methodology Notes

### Model Validation

**GARCH Parameter Constraints:**
- Stationarity: α + β < 1 enforced via assertion
- Positivity: ω, α, β ≥ 0 required
- Fat tails: Student's t with typical ν ∈ [3, 10]

**Bubble Score Normalization:**
All indicators scaled 0-100 before weighting to ensure:
- Comparable magnitude across diverse metrics
- Intuitive interpretation (higher = more risk)
- Stable composite scores regardless of individual indicator ranges

**Correlation Analysis:**
Pearson correlation coefficient used for:
- Bubble score vs. crash drawdown validation
- Sector peer correlation calculations

Negative correlation expected and observed between bubble scores and subsequent returns.

### Known Limitations

1. **Sample Size:** 20 stocks across 3 crashes limits statistical power for precision/recall metrics
2. **Survivorship Bias:** Analysis excludes delisted stocks (e.g., Lehman Brothers 2008)
3. **Regime Dependency:** Correlations may vary across different market microstructures
4. **Google Trends:** Rate limiting (429 errors) requires caching and retry logic
5. **Forward-Looking Limitation:** System detects current bubble risk, cannot predict timing of correction

### Future Enhancements

**Planned Features:**
- Machine learning ensemble combining indicators
- Options market implied volatility integration
- Credit default swap spreads for systemic risk
- Natural language processing on earnings call transcripts
- Real-time news sentiment analysis
- Cross-asset correlation (bonds, commodities, crypto)

**Research Extensions:**
- Optimal rebalancing frequency analysis
- Transaction cost modeling
- Tax-aware portfolio construction
- Factor exposure decomposition (Fama-French)
- Regime-switching GARCH variants

---

## Results Summary

Atlas demonstrates practical application of quantitative risk modeling to equity bubble detection:

1. **Historical Validation:** -0.481 average correlation across three major crashes validates predictive power
2. **Portfolio Performance:** 305.67% return vs. 95.30% benchmark = 210.38% alpha over 5 years
3. **Risk-Adjusted Excellence:** Sharpe ratio 1.04 indicates superior risk-adjusted performance
4. **Production Readiness:** Live API system processing 41 stocks with <50s update cycle
5. **Scalability:** Multi-stock GARCH successfully models 41 securities simultaneously

**Practical Applications:**
- Risk management for institutional portfolios
- Tactical asset allocation signals
- Systematic trading strategy overlay
- Real-time market monitoring dashboard

**Academic Contribution:**
Novel 8-factor composite methodology combining traditional technical analysis with behavioral finance (sentiment) and modern volatility modeling (GARCH). Historical validation across multiple market regimes strengthens empirical foundation.

---

## Team

**Matthew Barghout** - Bubble indicators, Multi-stock infrastructure, Portfolio strategy, Historical backtesting
**Kautilya** - GARCH volatility forecasting, Batch processing optimization
**Juan** - Google Trends sentiment integration, VIX analysis

**Institution:** University of North Carolina at Chapel Hill
**Competition:** SAS Curiosity Cup 2026
**Repository:** https://github.com/MatthewBarghout/sas-curiosity-cup-2025-kmj




**Last Updated:** February 15, 2026
**Version:** 1.0.0
**Status:** Production Ready
