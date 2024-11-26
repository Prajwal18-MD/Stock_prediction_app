import math

class PutCallParity:
    def __init__(self, stock_price, strike_price, interest_rate, time_to_expiration):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.time_to_expiration = time_to_expiration

    def calculate_parity(self, call_price, put_price):
        """
        Calculate the put-call parity.
        """
        parity = self.stock_price - self.strike_price * math.exp(-self.interest_rate * self.time_to_expiration)
        return parity, parity - call_price, parity + put_price
