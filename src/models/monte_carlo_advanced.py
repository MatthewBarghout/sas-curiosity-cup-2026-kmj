"""
Advanced Monte Carlo: Jump Diffusion with Garch Volatitlity
1. Incorporates jumps (sudden crashes)
2. Uses GARCH forecasted volatility (not constant)
3. Log returns (better for large moves)
"""

import numpy as np
from typing import List, Optional

class JumpDiffusionMC:
    """Jumps Diffusion Geometric Brownian Motion"""
    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        jump_intensity: float = 0.05,
        jump_mean: float = -0.02,
        jump_std: float = 0.05
    ):
        """Initalize Jump Diffusion simulator"""
        self.S0= S0
        self.mu = mu
        self.sigma = sigma
        self.days = days
        self.dt= 1/252

        self.lambda_jump = jump_intensity
        self.mu_jump= jump_mean
        self.mu_jump = jump_mean
        self.sigma_jump= jump_std

    def simulate_one_path(self)-> List[float]:
        """Simulate one possible price path with jumps"""
        prices=[self.S0]
        for day in range(self.days):
            diffusion_return = np.random.normal(self.mu * self.dt, self.sigma * np.sqrt(self.dt))
            
            jump_occurs = np.random.random() < self.lambda_jump

            if jump_occurs:
                jump_return = np.random.normal(self.mu_jump, self.sigma_jump)
            else:
                jump_return = 0
            
            total_return = diffusion_return + jump_return
            
            new_price = prices[-1] * np.exp(total_return)
            prices.append(new_price)
        return prices
    
    def run_simulation(self, n_simulations: int=10000) -> np.ndarray:
        """Runs multiple simulations"""
        all_paths=[]

        for i in range(n_simulations):
            if i % 1000==0:
                print(f"Simulation {i}/{n_simulations}")
            
            path = self.simulate_one_path()
            all_paths.append(path)
        return np.array(all_paths)


    def calculate_statistics(self, final_prices: np.ndarray)-> dict:
        return{
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'var_95': np.percentile(final_prices, 5),
            'var_99': np.percentile(final_prices, 1),
            'prob_loss': np.sum(final_prices < self.S0) / len(final_prices),
            'prob_large_loss': np.sum(final_prices < self.S0 * 0.8) / len(final_prices)
        }
    

# Test the Jump Diffusion Monte Carlo
if __name__ == "__main__":
    print("Testing Jump Diffusion Monte Carlo...\n")
    
    # Create simulator with test parameters
    jd_mc = JumpDiffusionMC(
        S0=180.0,
        mu=0.15,
        sigma=0.50,
        days=180
    )
    
    print("Running 1,000 simulations (quick test)...")
    paths = jd_mc.run_simulation(n_simulations=1000)
    
    # Get final prices
    final_prices = paths[:, -1]
    
    # Calculate statistics
    stats = jd_mc.calculate_statistics(final_prices)
    
    print("\nResults (6-month forecast):")
    print(f"Starting price: ${jd_mc.S0:.2f}")
    print(f"Expected price: ${stats['mean']:.2f}")
    print(f"Median price: ${stats['median']:.2f}")
    print(f"95% VaR: ${stats['var_95']:.2f}")
    print(f"99% VaR: ${stats['var_99']:.2f}")
    print(f"Probability of loss: {stats['prob_loss']*100:.1f}%")
    print(f"Probability of 20%+ loss: {stats['prob_large_loss']*100:.1f}%")
    print("\nJump Diffusion test complete!")

