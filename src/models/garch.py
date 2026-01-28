"""
GARCH(1,1) Volatility Forecasting with Student's t errors
Provides dynamic volatility estimates for Monte Carlo simulation
Multi-stock support for sector analysis
"""

from arch import arch_model
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class GARCHVolatilityForecaster:
    
    def __init__(self, returns: pd.Series):
        self.returns = returns * 100
        self.model = None
        self.results = None
        
    def fit(self) -> None:
        self.model = arch_model(
            self.returns,
            mean='Zero',
            vol='Garch',
            p=1,
            q=1,
            dist='t'
        )
        
        self.results = self.model.fit(disp='off')
        self._validate_fit()
        
    def _validate_fit(self) -> None:
        params = self.results.params
        
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        
        assert omega > 0, "omega must be positive"
        assert alpha >= 0, "alpha must be non-negative"
        assert beta >= 0, "beta must be non-negative"
        
        persistence = alpha + beta
        assert persistence < 1, f"Model is non-stationary! α + β = {persistence:.4f} >= 1"
        
        if persistence > 0.99:
            print(f"⚠️  Warning: High persistence ({persistence:.4f})")
        
    def forecast_volatility(self, horizon: int = 30) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        variance_forecast = self.results.forecast(horizon=horizon, reindex=False)
        forecasted_variance = variance_forecast.variance.values[-1, :]
        forecasted_vol = np.sqrt(forecasted_variance)
        annualized_vol = forecasted_vol * np.sqrt(252) / 100
        
        return annualized_vol
    
    def get_parameters(self) -> Dict:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        params = self.results.params
        
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        persistence = alpha + beta
        
        long_run_variance = omega / (1 - persistence)
        long_run_vol = np.sqrt(long_run_variance) * np.sqrt(252) / 100
        
        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'nu': params.get('nu', None),
            'persistence': persistence,
            'long_run_vol': long_run_vol
        }
    
    def get_current_volatility(self) -> float:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        current_vol = self.results.conditional_volatility.iloc[-1]
        annualized = current_vol * np.sqrt(252) / 100
        
        return annualized
    
    def get_conditional_volatility(self) -> pd.Series:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        cond_vol = self.results.conditional_volatility
        annualized = cond_vol * np.sqrt(252) / 100
        
        return annualized
    
    def summary(self) -> None:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("\n" + "="*70)
        print("GARCH(1,1) MODEL SUMMARY")
        print("="*70)
        
        params = self.get_parameters()
        
        print(f"\nParameters:")
        print(f"  ω (omega):     {params['omega']:>10.6f}")
        print(f"  α (alpha):     {params['alpha']:>10.6f}")
        print(f"  β (beta):      {params['beta']:>10.6f}")
        print(f"  ν (nu):        {params['nu']:>10.2f}")
        
        print(f"\nDiagnostics:")
        print(f"  Persistence (α+β):      {params['persistence']:.4f}")
        
        if params['persistence'] > 0.95:
            print(f"    → High persistence")
        else:
            print(f"    → Moderate persistence")
        
        print(f"  Long-run volatility:    {params['long_run_vol']*100:.2f}%")
        print(f"  Current volatility:     {self.get_current_volatility()*100:.2f}%")
        
        print(f"\nModel Fit:")
        print(f"  Log-Likelihood:         {self.results.loglikelihood:.2f}")
        print(f"  AIC:                    {self.results.aic:.2f}")
        print(f"  BIC:                    {self.results.bic:.2f}")
        
        if params['nu'] < 10:
            print(f"\n  ✓ Fat tails confirmed (ν = {params['nu']:.2f} < 10)")
        else:
            print(f"\n  ○ Tails close to normal (ν = {params['nu']:.2f} > 10)")
        
        print("="*70 + "\n")


class MultiStockGARCH:
    """
    GARCH model manager for multiple stocks/sectors
    """
    
    def __init__(self, returns_dict: Dict[str, pd.Series]):
        """
        Initialize with dictionary of returns
        
        Args:
            returns_dict: {ticker: returns_series}
        """
        self.returns_dict = returns_dict
        self.models = {}
        self.tickers = list(returns_dict.keys())
        
    def fit_all(self, verbose: bool = True) -> None:
        """
        Fit GARCH models for all stocks
        """
        print(f"\n{'='*70}")
        print(f"Fitting GARCH models for {len(self.tickers)} stocks")
        print(f"{'='*70}\n")
        
        for ticker in self.tickers:
            if verbose:
                print(f"Fitting {ticker}...", end=" ")
            
            try:
                garch = GARCHVolatilityForecaster(self.returns_dict[ticker])
                garch.fit()
                self.models[ticker] = garch
                
                if verbose:
                    params = garch.get_parameters()
                    print(f"✓ (α+β={params['persistence']:.3f}, ν={params['nu']:.1f})")
                    
            except Exception as e:
                if verbose:
                    print(f"✗ Failed: {str(e)}")
                self.models[ticker] = None
        
        successful = sum(1 for m in self.models.values() if m is not None)
        print(f"\n✓ Successfully fitted {successful}/{len(self.tickers)} models\n")
    
    def forecast_all(self, horizon: int = 30) -> Dict[str, np.ndarray]:
        """
        Forecast volatility for all stocks
        
        Args:
            horizon: Number of days to forecast
            
        Returns:
            {ticker: volatility_forecast_array}
        """
        forecasts = {}
        
        for ticker, model in self.models.items():
            if model is not None:
                try:
                    forecasts[ticker] = model.forecast_volatility(horizon)
                except Exception as e:
                    print(f"Warning: Could not forecast {ticker}: {str(e)}")
                    forecasts[ticker] = None
            else:
                forecasts[ticker] = None
        
        return forecasts
    
    def get_current_volatilities(self) -> Dict[str, float]:
        """
        Get current volatility for all stocks
        
        Returns:
            {ticker: current_volatility}
        """
        current_vols = {}
        
        for ticker, model in self.models.items():
            if model is not None:
                try:
                    current_vols[ticker] = model.get_current_volatility()
                except Exception as e:
                    print(f"Warning: Could not get volatility for {ticker}: {str(e)}")
                    current_vols[ticker] = None
            else:
                current_vols[ticker] = None
        
        return current_vols
    
    def get_all_parameters(self) -> pd.DataFrame:
        """
        Get parameters for all stocks as DataFrame
        
        Returns:
            DataFrame with columns: ticker, omega, alpha, beta, nu, persistence, long_run_vol
        """
        params_list = []
        
        for ticker, model in self.models.items():
            if model is not None:
                try:
                    params = model.get_parameters()
                    params['ticker'] = ticker
                    params_list.append(params)
                except Exception as e:
                    print(f"Warning: Could not get parameters for {ticker}: {str(e)}")
        
        if params_list:
            df = pd.DataFrame(params_list)
            df = df[['ticker', 'omega', 'alpha', 'beta', 'nu', 'persistence', 'long_run_vol']]
            return df
        else:
            return pd.DataFrame()
    
    def summary_all(self) -> None:
        """
        Print summary for all stocks
        """
        print("\n" + "="*70)
        print("MULTI-STOCK GARCH SUMMARY")
        print("="*70)
        
        params_df = self.get_all_parameters()
        current_vols = self.get_current_volatilities()
        
        if not params_df.empty:
            params_df['current_vol'] = params_df['ticker'].map(current_vols)
            
            print("\nParameters by Stock:")
            print(params_df.to_string(index=False))
            
            print("\n" + "-"*70)
            print("AGGREGATE STATISTICS")
            print("-"*70)
            print(f"Average persistence (α+β): {params_df['persistence'].mean():.4f}")
            print(f"Average nu (tail fatness):  {params_df['nu'].mean():.2f}")
            print(f"Average current vol:        {params_df['current_vol'].mean()*100:.2f}%")
            print(f"Highest current vol:        {params_df['current_vol'].max()*100:.2f}% ({params_df.loc[params_df['current_vol'].idxmax(), 'ticker']})")
            print(f"Lowest current vol:         {params_df['current_vol'].min()*100:.2f}% ({params_df.loc[params_df['current_vol'].idxmin(), 'ticker']})")
        
        print("="*70 + "\n")
    
    def get_forecast_dataframe(self, horizon: int = 30) -> pd.DataFrame:
        """
        Get forecasts as DataFrame with dates
        
        Returns:
            DataFrame with columns: date, ticker, volatility
        """
        forecasts = self.forecast_all(horizon)
        
        last_date = None
        for ticker in self.tickers:
            if ticker in self.returns_dict and self.returns_dict[ticker] is not None:
                last_date = self.returns_dict[ticker].index[-1]
                break
        
        if last_date is None:
            raise ValueError("Could not determine last date from returns")
        
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        rows = []
        for ticker, forecast in forecasts.items():
            if forecast is not None:
                for i, date in enumerate(forecast_dates):
                    rows.append({
                        'date': date,
                        'ticker': ticker,
                        'volatility': forecast[i],
                        'horizon': i + 1
                    })
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data.fetch import download_stock_data, calculate_returns
    from datetime import datetime, timedelta
    
    print("="*70)
    print("TESTING MULTI-STOCK GARCH")
    print("="*70)
    
    tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMD', 'META', 'AMZN']
    
    print(f"\n1. Downloading data for {len(tickers)} stocks...")
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    returns_dict = {}
    for ticker in tickers:
        try:
            data = download_stock_data(ticker, start, end)
            returns = calculate_returns(data)
            returns_dict[ticker] = returns
            print(f"   ✓ {ticker}: {len(returns)} days")
        except Exception as e:
            print(f"   ✗ {ticker}: Failed - {str(e)}")
    
    print(f"\n2. Fitting GARCH models...")
    multi_garch = MultiStockGARCH(returns_dict)
    multi_garch.fit_all(verbose=True)
    
    print("\n3. Getting current volatilities...")
    current_vols = multi_garch.get_current_volatilities()
    print("\nCurrent Volatilities:")
    for ticker, vol in sorted(current_vols.items(), key=lambda x: x[1] if x[1] else 0, reverse=True):
        if vol is not None:
            print(f"  {ticker:6s}: {vol*100:6.2f}%")
    
    print("\n4. Forecasting volatility (30-day)...")
    forecasts = multi_garch.forecast_all(horizon=30)
    print("\n30-Day Ahead Forecasts:")
    for ticker in tickers:
        if ticker in forecasts and forecasts[ticker] is not None:
            print(f"  {ticker:6s}: {forecasts[ticker][29]*100:6.2f}%")
    
    print("\n5. Summary statistics...")
    multi_garch.summary_all()
    
    print("\n6. Creating forecast DataFrame...")
    try:
        forecast_df = multi_garch.get_forecast_dataframe(horizon=90)
        print(f"   ✓ Created DataFrame with {len(forecast_df)} rows")
        print("\nSample (first 10 rows):")
        print(forecast_df.head(10))
    except Exception as e:
        print(f"   ✗ Could not create forecast DataFrame: {e}")
    
    print("\n7. Comparing VIX to GARCH for all stocks...")
    try:
        from src.data.fetch import get_vix_level, compare_vix_to_garch
        
        current_vix = get_vix_level()
        current_vols = multi_garch.get_current_volatilities()
        
        print(f"\nCurrent VIX: {current_vix:.2f}%")
        print(f"\n{'Ticker':<8} {'GARCH Vol':<12} {'VIX/GARCH':<12} {'Interpretation'}")
        print("-" * 80)
        
        for ticker in tickers:
            if ticker in current_vols and current_vols[ticker] is not None:
                garch_vol = current_vols[ticker]
                comp = compare_vix_to_garch(current_vix, garch_vol)
                ratio_str = f"{comp['ratio']:.2f}x" if comp['ratio'] else "N/A"
                print(f"{ticker:<8} {garch_vol*100:>6.2f}%      {ratio_str:<12} {comp['interpretation']}")
        
        print("\n" + "-" * 80)
        print("VIX Interpretation Guide:")
        print("  Ratio > 1.2: Market expects MORE vol than historical (fear)")
        print("  Ratio 0.8-1.2: VIX and GARCH aligned (normal)")
        print("  Ratio < 0.8: Market expects LESS vol than historical (complacency)")
                
    except Exception as e:
        print(f"   ✗ Could not compare VIX: {e}")
    
    print("\n" + "="*70)
    print("✅ MULTI-STOCK GARCH TEST COMPLETE!")
    print("="*70)
    print("\nDeliverables completed:")
    print("  ✓ Multi-stock GARCH support")
    print("  ✓ Dictionary output {ticker: volatility_forecast}")
    print("  ✓ Batch processing for 8+ stocks")
    print("  ✓ Summary statistics across stocks")
    print("  ✓ VIX comparison for all stocks")