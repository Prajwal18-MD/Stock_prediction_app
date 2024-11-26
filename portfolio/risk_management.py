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
