import numpy as np
import pandas as pd

class PortfolioOptimization:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.returns = stock_data.pct_change().dropna()

    def optimize_portfolio(self):
        """
        Calculate the optimal portfolio allocation.
        :return: Dictionary with optimal weights for each stock.
        """
        cov_matrix = self.returns.cov()
        num_assets = len(self.returns.columns)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Expected portfolio return and variance
        portfolio_return = np.sum(weights * self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_volatility

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
