"""
Historical Backtesting Framework

Infrastructure for testing trading strategies and portfolio allocations
using historical data, with integration to Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Callable
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.monte_carlo import MonteCarloSimulator


# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate (can be updated)


class PerformanceMetrics:
    """
    Calculate various performance metrics for portfolio evaluation.
    """

    @staticmethod
    def total_return(prices: np.ndarray) -> float:
        """Calculate total return from price series."""
        return (prices[-1] - prices[0]) / prices[0]

    @staticmethod
    def annualized_return(prices: np.ndarray, trading_days: int = None) -> float:
        """Calculate annualized return."""
        if trading_days is None:
            trading_days = len(prices)
        total_ret = PerformanceMetrics.total_return(prices)
        years = trading_days / TRADING_DAYS_PER_YEAR
        return (1 + total_ret) ** (1 / years) - 1

    @staticmethod
    def volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
        return vol

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of daily returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
        """
        Calculate Sortino ratio (penalizes only downside volatility).

        Args:
            returns: Array of daily returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        annualized_excess = np.mean(excess_returns) * TRADING_DAYS_PER_YEAR

        return annualized_excess / downside_std

    @staticmethod
    def max_drawdown(prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Maximum drawdown as a positive decimal (e.g., 0.20 for 20% drawdown)
        """
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)

    @staticmethod
    def calmar_ratio(prices: np.ndarray) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        ann_ret = PerformanceMetrics.annualized_return(prices)
        max_dd = PerformanceMetrics.max_drawdown(prices)
        if max_dd == 0:
            return 0.0
        return ann_ret / max_dd

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate percentage of positive return days."""
        return np.sum(returns > 0) / len(returns)

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / losses


class Backtester:
    """
    Historical backtesting engine for single and multi-stock strategies.

    Supports:
    - Buy and hold strategies
    - Custom allocation strategies
    - Rolling window analysis
    - Walk-forward testing
    """

    def __init__(self, prices_data: Dict[str, pd.DataFrame],
                 initial_capital: float = 100000.0):
        """
        Initialize the backtester.

        Args:
            prices_data: Dictionary mapping ticker -> DataFrame with 'Close' column
            initial_capital: Starting capital in dollars
        """
        self.prices_data = prices_data
        self.initial_capital = initial_capital
        self.tickers = list(prices_data.keys())

        # Align all price data to common dates
        self._align_data()

    def _align_data(self) -> None:
        """Align all price series to common dates."""
        # Find common date range
        all_indices = [df.index for df in self.prices_data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        self.common_dates = common_index.sort_values()

        # Create aligned close prices DataFrame
        self.close_prices = pd.DataFrame(index=self.common_dates)
        for ticker, df in self.prices_data.items():
            self.close_prices[ticker] = df.loc[self.common_dates, 'Close']

        # Calculate returns
        self.returns = self.close_prices.pct_change().dropna()

    def backtest_buy_and_hold(self, weights: Dict[str, float] = None) -> Dict:
        """
        Backtest a buy and hold strategy.

        Args:
            weights: Dictionary of ticker -> weight (must sum to 1.0)
                    If None, equal weight allocation is used.

        Returns:
            Dictionary with portfolio values and performance metrics
        """
        if weights is None:
            # Equal weight allocation
            weights = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

        # Validate weights
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            print(f"Warning: Weights sum to {weight_sum}, normalizing...")
            weights = {k: v / weight_sum for k, v in weights.items()}

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=self.returns.index)
        for ticker, weight in weights.items():
            if ticker in self.returns.columns:
                portfolio_returns += weight * self.returns[ticker]

        # Calculate portfolio value over time
        portfolio_values = self.initial_capital * (1 + portfolio_returns).cumprod()

        # Prepend initial capital
        portfolio_values = pd.concat([
            pd.Series([self.initial_capital], index=[self.common_dates[0]]),
            portfolio_values
        ])

        # Calculate metrics
        prices_array = portfolio_values.values
        returns_array = portfolio_returns.values

        metrics = self._calculate_all_metrics(prices_array, returns_array)

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'weights': weights,
            'metrics': metrics,
            'start_date': str(self.common_dates[0].date()),
            'end_date': str(self.common_dates[-1].date()),
            'trading_days': len(self.returns)
        }

    def _calculate_all_metrics(self, prices: np.ndarray, returns: np.ndarray) -> Dict:
        """Calculate all performance metrics."""
        return {
            'total_return': PerformanceMetrics.total_return(prices),
            'annualized_return': PerformanceMetrics.annualized_return(prices, len(returns)),
            'volatility': PerformanceMetrics.volatility(returns),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(prices),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(prices),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'final_value': prices[-1],
            'initial_value': prices[0]
        }

    def backtest_rebalanced(self, weights: Dict[str, float],
                           rebalance_frequency: int = 21) -> Dict:
        """
        Backtest a strategy with periodic rebalancing.

        Args:
            weights: Target weights for each ticker
            rebalance_frequency: Days between rebalancing (default 21 = monthly)

        Returns:
            Dictionary with results and metrics
        """
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        portfolio_returns_list = []

        current_weights = weights.copy()
        days_since_rebalance = 0

        for i, date in enumerate(self.returns.index):
            # Calculate today's return based on current weights
            daily_return = sum(
                current_weights.get(ticker, 0) * self.returns.loc[date, ticker]
                for ticker in self.tickers
                if ticker in self.returns.columns
            )

            portfolio_returns_list.append(daily_return)
            portfolio_value *= (1 + daily_return)
            portfolio_values.append(portfolio_value)

            days_since_rebalance += 1

            # Rebalance if needed
            if days_since_rebalance >= rebalance_frequency:
                current_weights = weights.copy()
                days_since_rebalance = 0
            else:
                # Update weights based on price movements
                total = 0
                for ticker in self.tickers:
                    if ticker in current_weights:
                        current_weights[ticker] *= (1 + self.returns.loc[date, ticker])
                        total += current_weights[ticker]
                # Normalize
                if total > 0:
                    current_weights = {k: v / total for k, v in current_weights.items()}

        portfolio_values = pd.Series(portfolio_values,
                                     index=[self.common_dates[0]] + list(self.returns.index))
        portfolio_returns = pd.Series(portfolio_returns_list, index=self.returns.index)

        metrics = self._calculate_all_metrics(
            portfolio_values.values,
            portfolio_returns.values
        )

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'weights': weights,
            'rebalance_frequency': rebalance_frequency,
            'metrics': metrics,
            'start_date': str(self.common_dates[0].date()),
            'end_date': str(self.common_dates[-1].date()),
            'trading_days': len(self.returns)
        }

    def rolling_backtest(self, weights: Dict[str, float],
                        window_size: int = 63) -> pd.DataFrame:
        """
        Perform rolling window backtest analysis.

        Args:
            weights: Portfolio weights
            window_size: Rolling window size in days (default 63 = quarterly)

        Returns:
            DataFrame with rolling metrics
        """
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=self.returns.index)
        for ticker, weight in weights.items():
            if ticker in self.returns.columns:
                portfolio_returns += weight * self.returns[ticker]

        rolling_metrics = []

        for i in range(window_size, len(portfolio_returns)):
            window_returns = portfolio_returns.iloc[i - window_size:i].values
            window_prices = self.initial_capital * (1 + portfolio_returns.iloc[:i]).cumprod().values

            rolling_metrics.append({
                'date': portfolio_returns.index[i],
                'rolling_sharpe': PerformanceMetrics.sharpe_ratio(window_returns),
                'rolling_volatility': PerformanceMetrics.volatility(window_returns),
                'rolling_return': PerformanceMetrics.annualized_return(
                    window_prices[-window_size:], window_size
                ),
                'rolling_max_drawdown': PerformanceMetrics.max_drawdown(
                    window_prices[-window_size:]
                )
            })

        return pd.DataFrame(rolling_metrics).set_index('date')

    def compare_strategies(self, strategies: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple allocation strategies.

        Args:
            strategies: Dictionary of strategy_name -> weights dict

        Returns:
            DataFrame comparing all strategies
        """
        results = []

        for name, weights in strategies.items():
            bt_result = self.backtest_buy_and_hold(weights)
            metrics = bt_result['metrics']
            metrics['strategy'] = name
            results.append(metrics)

        comparison = pd.DataFrame(results).set_index('strategy')

        return comparison


class WalkForwardAnalyzer:
    """
    Walk-forward analysis combining backtesting with Monte Carlo forecasting.

    Splits historical data into in-sample (training) and out-of-sample (testing)
    periods, then uses Monte Carlo to forecast future performance.
    """

    def __init__(self, prices_data: Dict[str, pd.DataFrame],
                 parameters: Dict[str, dict],
                 in_sample_ratio: float = 0.7):
        """
        Initialize walk-forward analyzer.

        Args:
            prices_data: Historical price data
            parameters: Monte Carlo parameters from multi_stock_fetch
            in_sample_ratio: Fraction of data to use for in-sample period
        """
        self.prices_data = prices_data
        self.parameters = parameters
        self.in_sample_ratio = in_sample_ratio

        # Split data
        self._split_data()

    def _split_data(self) -> None:
        """Split data into in-sample and out-of-sample periods."""
        # Get common dates
        all_indices = [df.index for df in self.prices_data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        common_index = common_index.sort_values()

        split_idx = int(len(common_index) * self.in_sample_ratio)

        self.in_sample_dates = common_index[:split_idx]
        self.out_of_sample_dates = common_index[split_idx:]

        # Split prices
        self.in_sample_prices = {}
        self.out_of_sample_prices = {}

        for ticker, df in self.prices_data.items():
            self.in_sample_prices[ticker] = df.loc[
                df.index.isin(self.in_sample_dates)
            ]
            self.out_of_sample_prices[ticker] = df.loc[
                df.index.isin(self.out_of_sample_dates)
            ]

    def run_analysis(self, weights: Dict[str, float],
                    n_mc_simulations: int = 1000,
                    forecast_days: int = 126) -> Dict:
        """
        Run complete walk-forward analysis.

        Args:
            weights: Portfolio weights
            n_mc_simulations: Number of Monte Carlo simulations
            forecast_days: Days to forecast ahead

        Returns:
            Dictionary with in-sample, out-of-sample, and forecast results
        """
        # In-sample backtest
        in_sample_bt = Backtester(self.in_sample_prices)
        in_sample_results = in_sample_bt.backtest_buy_and_hold(weights)

        # Out-of-sample backtest
        out_of_sample_bt = Backtester(self.out_of_sample_prices)
        out_of_sample_results = out_of_sample_bt.backtest_buy_and_hold(weights)

        # Monte Carlo forecast for portfolio
        portfolio_params = self._calculate_portfolio_params(weights)
        mc_forecast = self._run_portfolio_mc(
            portfolio_params,
            n_mc_simulations,
            forecast_days
        )

        return {
            'in_sample': in_sample_results,
            'out_of_sample': out_of_sample_results,
            'monte_carlo_forecast': mc_forecast,
            'in_sample_dates': (str(self.in_sample_dates[0].date()),
                               str(self.in_sample_dates[-1].date())),
            'out_of_sample_dates': (str(self.out_of_sample_dates[0].date()),
                                   str(self.out_of_sample_dates[-1].date()))
        }

    def _calculate_portfolio_params(self, weights: Dict[str, float]) -> Dict:
        """Calculate weighted portfolio parameters."""
        portfolio_mu = 0.0
        portfolio_sigma_sq = 0.0

        for ticker, weight in weights.items():
            if ticker in self.parameters:
                params = self.parameters[ticker]
                portfolio_mu += weight * params['mu']
                portfolio_sigma_sq += (weight * params['sigma']) ** 2

        # Get weighted average starting price (normalized to 100)
        portfolio_S0 = 100.0  # Normalized starting value

        return {
            'S0': portfolio_S0,
            'mu': portfolio_mu,
            'sigma': np.sqrt(portfolio_sigma_sq)
        }

    def _run_portfolio_mc(self, params: Dict, n_simulations: int,
                         forecast_days: int) -> Dict:
        """Run Monte Carlo simulation for portfolio."""
        simulator = MonteCarloSimulator(
            S0=params['S0'],
            mu=params['mu'],
            sigma=params['sigma'],
            days=forecast_days
        )

        paths = simulator.run_simulation(n_simulations)
        final_prices = paths[:, -1]
        stats = simulator.get_statistics(final_prices)

        return {
            'paths': paths,
            'final_prices': final_prices,
            'statistics': stats,
            'forecast_days': forecast_days,
            'n_simulations': n_simulations
        }


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics dictionary as a readable report.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "PERFORMANCE METRICS",
        "=" * 50,
        f"Total Return: {metrics['total_return']*100:.2f}%",
        f"Annualized Return: {metrics['annualized_return']*100:.2f}%",
        f"Volatility (Annual): {metrics['volatility']*100:.2f}%",
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}",
        f"Sortino Ratio: {metrics['sortino_ratio']:.3f}",
        f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%",
        f"Calmar Ratio: {metrics['calmar_ratio']:.3f}",
        f"Win Rate: {metrics['win_rate']*100:.1f}%",
        f"Profit Factor: {metrics['profit_factor']:.2f}",
        f"Final Value: ${metrics['final_value']:,.2f}",
        "=" * 50
    ]
    return "\n".join(lines)


class COVIDBacktester:
    """
    Specialized backtester for analyzing COVID crash period (Feb-Apr 2020).

    Compares Monte Carlo predicted drawdowns with actual market drawdowns
    during the COVID-19 market crash.
    """

    # COVID crash period dates
    COVID_START = '2020-02-19'  # Market peak before crash
    COVID_BOTTOM = '2020-03-23'  # Market bottom
    COVID_END = '2020-04-30'     # Recovery period end

    def __init__(self, tickers: List[str] = None):
        """
        Initialize COVID backtester.

        Args:
            tickers: List of stock tickers to analyze. Defaults to ['NVDA', 'SPY', 'AAPL']
        """
        self.tickers = tickers or ['NVDA', 'SPY', 'AAPL']
        self.prices_data = {}
        self.returns_data = {}
        self.crash_results = {}

    def fetch_covid_period_data(self,
                                 start_date: str = '2020-01-01',
                                 end_date: str = '2020-06-30') -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data covering the COVID crash period.

        Args:
            start_date: Start date for data fetch (before crash)
            end_date: End date for data fetch (after recovery begins)

        Returns:
            Dictionary of ticker -> DataFrame with price data
        """
        import yfinance as yf

        print(f"Fetching COVID period data for {self.tickers}...")

        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    self.prices_data[ticker] = data
                    self.returns_data[ticker] = data['Close'].pct_change().dropna()
                    print(f"  {ticker}: {len(data)} days of data")
                else:
                    print(f"  {ticker}: No data available")
            except Exception as e:
                print(f"  {ticker}: Error fetching - {e}")

        return self.prices_data

    def analyze_crash_drawdown(self, ticker: str) -> Dict:
        """
        Analyze the actual drawdown during COVID crash for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with drawdown analysis
        """
        if ticker not in self.prices_data:
            raise ValueError(f"No data for {ticker}. Call fetch_covid_period_data first.")

        df = self.prices_data[ticker]
        prices = df['Close']

        # Flatten multi-level column index if present
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        # Find peak before crash
        pre_crash = prices[prices.index <= self.COVID_START]
        if len(pre_crash) == 0:
            peak_price = float(prices.iloc[0])
            peak_date = prices.index[0]
        else:
            peak_price = float(pre_crash.max())
            peak_date = pre_crash.idxmax()
            if isinstance(peak_date, pd.Series):
                peak_date = peak_date.iloc[0]

        # Find bottom during crash
        crash_period = prices[(prices.index >= self.COVID_START) &
                              (prices.index <= self.COVID_BOTTOM)]
        if len(crash_period) > 0:
            bottom_price = float(crash_period.min())
            bottom_date = crash_period.idxmin()
            if isinstance(bottom_date, pd.Series):
                bottom_date = bottom_date.iloc[0]
        else:
            bottom_price = float(prices.min())
            bottom_date = prices.idxmin()
            if isinstance(bottom_date, pd.Series):
                bottom_date = bottom_date.iloc[0]

        # Calculate actual drawdown
        actual_drawdown = (peak_price - bottom_price) / peak_price

        # Recovery analysis
        recovery_period = prices[prices.index >= self.COVID_BOTTOM]
        recovery_to_peak_date = None
        days_to_recover = None

        if len(recovery_period) > 0:
            recovered = recovery_period[recovery_period >= peak_price]
            if len(recovered) > 0:
                recovery_to_peak_date = recovered.index[0]
                # Convert to datetime if needed
                if hasattr(bottom_date, 'to_pydatetime'):
                    bottom_dt = bottom_date.to_pydatetime()
                else:
                    bottom_dt = pd.to_datetime(bottom_date)
                if hasattr(recovery_to_peak_date, 'to_pydatetime'):
                    recovery_dt = recovery_to_peak_date.to_pydatetime()
                else:
                    recovery_dt = pd.to_datetime(recovery_to_peak_date)
                days_to_recover = (recovery_dt - bottom_dt).days

        return {
            'ticker': ticker,
            'peak_price': float(peak_price),
            'peak_date': str(peak_date.date()) if hasattr(peak_date, 'date') else str(peak_date),
            'bottom_price': float(bottom_price),
            'bottom_date': str(bottom_date.date()) if hasattr(bottom_date, 'date') else str(bottom_date),
            'actual_drawdown': float(actual_drawdown),
            'actual_drawdown_pct': f"{actual_drawdown * 100:.2f}%",
            'recovery_date': str(recovery_to_peak_date.date()) if recovery_to_peak_date and hasattr(recovery_to_peak_date, 'date') else None,
            'days_to_recover': days_to_recover
        }

    def run_monte_carlo_prediction(self, ticker: str,
                                    pre_crash_lookback_days: int = 252,
                                    n_simulations: int = 10000,
                                    forecast_days: int = 30) -> Dict:
        """
        Run Monte Carlo simulation using pre-crash data to see what it would have predicted.

        Args:
            ticker: Stock ticker
            pre_crash_lookback_days: Days of data before crash to use for parameters
            n_simulations: Number of MC simulations
            forecast_days: Days to forecast (crash period was ~23 trading days)

        Returns:
            Dictionary with MC predictions and VaR/CVaR analysis
        """
        if ticker not in self.prices_data:
            raise ValueError(f"No data for {ticker}. Call fetch_covid_period_data first.")

        df = self.prices_data[ticker]

        # Get pre-crash data for parameter estimation
        pre_crash_data = df[df.index < self.COVID_START]

        if len(pre_crash_data) < 20:
            # If not enough pre-crash data in our range, fetch more
            import yfinance as yf
            extended_start = (pd.to_datetime(self.COVID_START) -
                            pd.Timedelta(days=pre_crash_lookback_days * 2)).strftime('%Y-%m-%d')
            extended_data = yf.download(ticker, start=extended_start,
                                       end=self.COVID_START, progress=False)
            pre_crash_data = extended_data.tail(pre_crash_lookback_days)

        # Calculate parameters from pre-crash data
        close_prices = pre_crash_data['Close']
        # Flatten multi-level column if present
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        pre_crash_returns = close_prices.pct_change().dropna()
        daily_mu = float(pre_crash_returns.mean())
        daily_sigma = float(pre_crash_returns.std())

        # Starting price (day before crash)
        S0 = float(close_prices.iloc[-1])

        # Run Monte Carlo simulation
        dt = 1 / 252  # Daily time step
        np.random.seed(42)  # For reproducibility

        # Simulate paths
        paths = np.zeros((n_simulations, forecast_days + 1))
        paths[:, 0] = S0

        for t in range(1, forecast_days + 1):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (daily_mu - 0.5 * daily_sigma**2) + daily_sigma * z
            )

        # Calculate drawdowns from each simulation
        peak_prices = np.maximum.accumulate(paths, axis=1)
        drawdowns = (peak_prices - paths) / peak_prices
        max_drawdowns = np.max(drawdowns, axis=1)

        # Calculate VaR and CVaR for drawdowns
        var_95 = np.percentile(max_drawdowns, 95)
        var_99 = np.percentile(max_drawdowns, 99)
        cvar_95 = np.mean(max_drawdowns[max_drawdowns >= var_95])
        cvar_99 = np.mean(max_drawdowns[max_drawdowns >= var_99])

        # Final prices
        final_prices = paths[:, -1]

        return {
            'ticker': ticker,
            'S0': S0,
            'daily_mu': float(daily_mu),
            'daily_sigma': float(daily_sigma),
            'annual_mu': float(daily_mu * 252),
            'annual_sigma': float(daily_sigma * np.sqrt(252)),
            'n_simulations': n_simulations,
            'forecast_days': forecast_days,
            'predicted_drawdown_mean': float(np.mean(max_drawdowns)),
            'predicted_drawdown_median': float(np.median(max_drawdowns)),
            'predicted_drawdown_std': float(np.std(max_drawdowns)),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'predicted_final_price_mean': float(np.mean(final_prices)),
            'predicted_final_price_5th': float(np.percentile(final_prices, 5)),
            'predicted_final_price_95th': float(np.percentile(final_prices, 95)),
            'paths': paths,
            'max_drawdowns': max_drawdowns
        }

    def run_covid_backtest(self, n_simulations: int = 10000) -> Dict:
        """
        Run full COVID backtest comparing predicted vs actual drawdowns.

        Args:
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with complete backtest results
        """
        if not self.prices_data:
            self.fetch_covid_period_data()

        results = {}

        for ticker in self.tickers:
            if ticker not in self.prices_data:
                continue

            print(f"\nAnalyzing {ticker}...")

            # Get actual drawdown
            actual = self.analyze_crash_drawdown(ticker)

            # Get MC predictions (using 30 trading days to cover crash period)
            predicted = self.run_monte_carlo_prediction(
                ticker,
                n_simulations=n_simulations,
                forecast_days=30
            )

            # Compare predicted vs actual
            actual_dd = actual['actual_drawdown']
            predicted_mean = predicted['predicted_drawdown_mean']
            var_95 = predicted['var_95']
            var_99 = predicted['var_99']

            # Was actual drawdown within predictions?
            was_captured_by_var95 = actual_dd <= var_95
            was_captured_by_var99 = actual_dd <= var_99

            # Percentile of actual drawdown in simulated distribution
            actual_percentile = np.mean(predicted['max_drawdowns'] <= actual_dd) * 100

            results[ticker] = {
                'actual': actual,
                'predicted': {
                    'drawdown_mean': predicted['predicted_drawdown_mean'],
                    'drawdown_median': predicted['predicted_drawdown_median'],
                    'var_95': predicted['var_95'],
                    'var_99': predicted['var_99'],
                    'cvar_95': predicted['cvar_95'],
                    'cvar_99': predicted['cvar_99']
                },
                'comparison': {
                    'actual_drawdown': actual_dd,
                    'predicted_mean_drawdown': predicted_mean,
                    'prediction_error': actual_dd - predicted_mean,
                    'prediction_error_pct': f"{(actual_dd - predicted_mean) * 100:.2f}%",
                    'actual_percentile': f"{actual_percentile:.1f}%",
                    'captured_by_var95': was_captured_by_var95,
                    'captured_by_var99': was_captured_by_var99,
                    'var95_vs_actual': f"VaR95 {var_95*100:.1f}% vs Actual {actual_dd*100:.1f}%"
                },
                'model_params': {
                    'S0': predicted['S0'],
                    'annual_mu': predicted['annual_mu'],
                    'annual_sigma': predicted['annual_sigma']
                }
            }

        self.crash_results = results
        return results

    def format_backtest_report(self) -> str:
        """Generate formatted report of COVID backtest results."""
        if not self.crash_results:
            return "No results available. Run run_covid_backtest() first."

        lines = [
            "=" * 70,
            "COVID-19 CRASH BACKTEST RESULTS (Feb-Apr 2020)",
            "=" * 70,
            ""
        ]

        for ticker, data in self.crash_results.items():
            actual = data['actual']
            predicted = data['predicted']
            comparison = data['comparison']

            lines.extend([
                f"--- {ticker} ---",
                f"Peak Date: {actual['peak_date']} | Peak Price: ${actual['peak_price']:.2f}",
                f"Bottom Date: {actual['bottom_date']} | Bottom Price: ${actual['bottom_price']:.2f}",
                f"",
                f"ACTUAL Drawdown: {actual['actual_drawdown_pct']}",
                f"PREDICTED Mean Drawdown: {predicted['drawdown_mean']*100:.2f}%",
                f"PREDICTED VaR 95%: {predicted['var_95']*100:.2f}%",
                f"PREDICTED VaR 99%: {predicted['var_99']*100:.2f}%",
                f"",
                f"Prediction Error: {comparison['prediction_error_pct']}",
                f"Actual was at {comparison['actual_percentile']} percentile of predictions",
                f"Captured by VaR95: {'✓' if comparison['captured_by_var95'] else '✗'}",
                f"Captured by VaR99: {'✓' if comparison['captured_by_var99'] else '✗'}",
                f"",
                f"Days to Recover: {actual['days_to_recover'] or 'N/A'}",
                ""
            ])

        lines.append("=" * 70)
        return "\n".join(lines)


def run_covid_backtest(tickers: List[str] = None,
                       n_simulations: int = 10000) -> Dict:
    """
    Convenience function to run COVID backtest.

    Args:
        tickers: List of stock tickers (defaults to ['NVDA', 'SPY', 'AAPL'])
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Dictionary with backtest results
    """
    backtester = COVIDBacktester(tickers)
    backtester.fetch_covid_period_data()
    results = backtester.run_covid_backtest(n_simulations)
    print(backtester.format_backtest_report())
    return results


# Testing
if __name__ == "__main__":
    print("Testing Backtester Framework...\n")

    # Import data pipeline
    from data.multi_stock_fetch import MultiStockDataPipeline

    # Set up test data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    test_tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'SPY']

    # Fetch data
    print("Fetching data...")
    pipeline = MultiStockDataPipeline(
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    params, correlation = pipeline.run_full_pipeline()

    # Create backtester
    print("\nInitializing backtester...")
    backtester = Backtester(pipeline.raw_data, initial_capital=100000)

    # Test equal-weight buy and hold
    print("\n--- Equal Weight Buy & Hold ---")
    results = backtester.backtest_buy_and_hold()
    print(format_metrics_report(results['metrics']))

    # Test custom allocation
    print("\n--- Tech-Heavy Allocation ---")
    tech_weights = {
        'NVDA': 0.30,
        'AAPL': 0.25,
        'GOOGL': 0.20,
        'MSFT': 0.15,
        'SPY': 0.10
    }
    tech_results = backtester.backtest_buy_and_hold(tech_weights)
    print(format_metrics_report(tech_results['metrics']))

    # Compare strategies
    print("\n--- Strategy Comparison ---")
    strategies = {
        'Equal Weight': None,  # Will use default equal weight
        'Tech Heavy': tech_weights,
        'SPY Only': {'SPY': 1.0},
        'NVDA Only': {'NVDA': 1.0}
    }

    # Fix None for equal weight
    strategies['Equal Weight'] = {t: 1/len(test_tickers) for t in test_tickers}

    comparison = backtester.compare_strategies(strategies)
    print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown']].round(3))

    # Test rebalancing
    print("\n--- Monthly Rebalanced Portfolio ---")
    rebal_results = backtester.backtest_rebalanced(tech_weights, rebalance_frequency=21)
    print(format_metrics_report(rebal_results['metrics']))

    print("\n\nBacktest framework test complete!")
