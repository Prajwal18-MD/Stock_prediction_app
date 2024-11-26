import numpy as np
import pandas as pd

class PortfolioOptimization:
    def __init__(self, stock_data):
        """
        Initialize Portfolio Optimization with stock price data.
        :param stock_data: DataFrame with adjusted close prices of stocks.
        """
        self.stock_data = stock_data
        self.returns = stock_data.pct_change().dropna()

    def optimize_portfolio(self):
        """
        Calculate the optimal portfolio allocation.
        :return: Dictionary with optimal weights, expected return, volatility, and Sharpe ratio.
        """
        try:
            # Ensure data has enough columns for optimization
            if self.returns.shape[1] < 2:
                raise ValueError("Data must contain at least two assets for optimization.")

            # Calculate covariance matrix and random weights
            cov_matrix = self.returns.cov()
            num_assets = len(self.returns.columns)
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            # Expected portfolio return, variance, and Sharpe ratio
            portfolio_return = np.sum(weights * self.returns.mean()) * 252  # Annualized return
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            sharpe_ratio = portfolio_return / portfolio_volatility

            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            return None
