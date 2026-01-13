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
