
import numpy as np
from typing import List, Tuple, Optional


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for stock prices.
    
    
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, days: int):
        """
        Initialize the simulator.
        
        Args:
            S0: Starting stock price
            mu: Expected annual return (drift)
            sigma: Annual volatility
            days: Number of days to simulate
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.dt = 1 / 252  
        
    def simulate_one_path(self) -> List[float]:
        """
        Simulate one possible price path.
        
        Returns:
            List of prices, one for each day
        """
        prices = [self.S0]
        
        for _ in range(self.days):
            daily_return = np.random.normal(
                self.mu * self.dt,
                self.sigma * np.sqrt(self.dt)
            )
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
            
        return prices
    
    def run_simulation(self, n_simulations: int) -> np.ndarray:
        """
        Run multiple simulations.
        
        Args:
            n_simulations: Number of price paths to simulate
            
        Returns:
            Array of shape (n_simulations, days+1) with all price paths
        """
        all_paths = []
        
        for _ in range(n_simulations):
            path = self.simulate_one_path()
            all_paths.append(path)
            
        return np.array(all_paths)
    
    def calculate_var(self, final_prices: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            final_prices: Array of final prices from simulations
            confidence: Confidence level (default 0.95 for 95% VaR)
            
        Returns:
            VaR value (the price at the (1-confidence) percentile)
        """
        percentile = (1 - confidence) * 100
        var = np.percentile(final_prices, percentile)
        return var
    
    def get_statistics(self, final_prices: np.ndarray) -> dict:
        """
        Calculate statistics from simulation results.
        
        Args:
            final_prices: Array of final prices
            
        Returns:
            Dictionary with mean, median, std, VaR, etc.
        """
        return {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'var_95': self.calculate_var(final_prices, 0.95),
            'var_99': self.calculate_var(final_prices, 0.99),
            'prob_loss': np.sum(final_prices < self.S0) / len(final_prices)
        }


# Testing
if __name__ == "__main__":
    print("Testing Monte Carlo Simulator...")
    
    # Create simulator
    sim = MonteCarloSimulator(
        S0=100.0,
        mu=0.15,
        sigma=0.30,
        days=252
    )
    
    # Run 1000 simulations
    print("Running 1000 simulations...")
    paths = sim.run_simulation(n_simulations=1000)
    
    # Get final prices
    final_prices = paths[:, -1]
    
    # Calculate statistics
    stats = sim.get_statistics(final_prices)
    
    print("\nResults:")
    print(f"Mean final price: ${stats['mean']:.2f}")
    print(f"Median final price: ${stats['median']:.2f}")
    print(f"95% VaR: ${stats['var_95']:.2f}")
    print(f"Probability of loss: {stats['prob_loss']*100:.1f}%")
    
    print("\nTest complete!")