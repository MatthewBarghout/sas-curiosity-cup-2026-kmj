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
            try:
                garch = GARCHVolatilityForecaster(self.returns_dict[ticker])
                garch.fit()
                self.models[ticker] = garch
                
                if verbose:
                    params = garch.get_parameters()
                    print(f"Fitting {ticker}... ✓ (α+β={params['persistence']:.3f}, ν={params['nu']:.1f})")
                    
            except AssertionError as e:
                # Non-stationary model
                if verbose:
                    print(f"Fitting {ticker}... ⚠️  Non-stationary (skipped)")
                self.models[ticker] = None
                
            except Exception as e:
                if verbose:
                    print(f"Fitting {ticker}... ✗ Failed: {str(e)}")
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
