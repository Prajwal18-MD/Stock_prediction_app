import numpy as np

class RiskManagement:
    def __init__(self, returns, confidence_level=0.05):
        self.returns = returns
        self.confidence_level = confidence_level

    def stop_loss(self, current_price, stop_loss_percentage):
        """
        Calculate stop-loss price based on a given percentage.
        """
        stop_loss_price = current_price * (1 - stop_loss_percentage / 100)
        return stop_loss_price

    def value_at_risk(self):
        """
        Calculate the Value-at-Risk (VaR) at a specified confidence level.
        """
        var = np.percentile(self.returns, self.confidence_level * 100)
        return var

# Define standalone functions for compatibility with app.py
def stop_loss_take_profit(data, stop_loss_percentage=5, take_profit_percentage=10):
    """
    Simulate stop-loss and take-profit strategy.
    :param data: DataFrame with stock price data (must include 'Close').
    :param stop_loss_percentage: Percentage for stop-loss (e.g., 5%).
    :param take_profit_percentage: Percentage for take-profit (e.g., 10%).
    :return: Dictionary with stop-loss and take-profit levels.
    """
    current_price = data['Close'].iloc[-1]
    stop_loss_price = current_price * (1 - stop_loss_percentage / 100)
    take_profit_price = current_price * (1 + take_profit_percentage / 100)

    return {
        "current_price": current_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
    }

def calculate_var(data, confidence_level=0.05):
    """
    Calculate Value-at-Risk (VaR) for stock price data.
    :param data: DataFrame with stock price data (must include 'Close').
    :param confidence_level: Confidence level for VaR (e.g., 0.05 for 5%).
    :return: Value-at-Risk (VaR) value.
    """
    returns = data['Close'].pct_change().dropna()
    var = np.percentile(returns, confidence_level * 100)
    return var
