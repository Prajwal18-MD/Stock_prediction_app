class StraddleStrategy:
    def __init__(self, call_price, put_price):
        self.call_price = call_price
        self.put_price = put_price

    def calculate_straddle(self):
        """
        Calculate the cost of the straddle strategy.
        """
        return self.call_price + self.put_price
